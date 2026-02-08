mod support;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

use hub::audio::AudioBus;
use hub::device::run_device;
use hub::grpc::proto::audio_input::Audio;
use hub::grpc::proto::naila_ai_server::{NailaAi, NailaAiServer};
use hub::grpc::proto::{AudioInput, AudioOutput};
use hub::grpc::{run_grpc_client, GrpcConfig};
use hub::metrics::HubMetrics;
use hub::vad::VadConfig;
use support::{speech_frame, MockTransport};
use webrtc_vad::VadMode;

// ── Mock AI server ───────────────────────────────────────────────────────────

/// Echoes every AudioInput back as an AudioOutput with the same PCM data.
struct EchoAiService;

#[tonic::async_trait]
impl NailaAi for EchoAiService {
    type StreamConversationStream =
        tokio_stream::wrappers::ReceiverStream<Result<AudioOutput, Status>>;

    async fn stream_conversation(
        &self,
        request: Request<Streaming<AudioInput>>,
    ) -> Result<Response<Self::StreamConversationStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<AudioOutput, Status>>(128);

        tokio::spawn(async move {
            while let Ok(Some(input)) = inbound.message().await {
                let pcm = match input.audio {
                    Some(Audio::AudioPcm(data)) => data,
                    _ => continue,
                };

                let output = AudioOutput {
                    device_id: input.device_id,
                    conversation_id: input.conversation_id,
                    audio_pcm: pcm,
                    sample_rate: input.sample_rate,
                    is_final: false,
                    ..Default::default()
                };

                if tx.send(Ok(output)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

// ── Test ──────────────────────────────────────────────────────────────────────

/// Full pipeline loopback: MockTransport → run_device → AudioBus →
/// run_grpc_client → echo server → run_grpc_client → tts_sub →
/// run_device → MockTransport.
#[tokio::test]
async fn audio_loopback_through_grpc() {
    // 1. Start mock gRPC server on a random port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap();
    let server_addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(NailaAiServer::new(EchoAiService))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    // 2. Create AudioBus.
    let (audio_tx, audio_rx) = mpsc::channel(256);
    let bus = Arc::new(AudioBus {
        audio_tx,
        tts_sub: DashMap::new(),
    });

    // 3. Start gRPC client pointed at the mock server.
    let cancel = CancellationToken::new();
    let grpc_config = GrpcConfig {
        server_addr: format!("http://{server_addr}"),
        initial_backoff: Duration::from_millis(50),
        max_backoff: Duration::from_millis(200),
    };
    let metrics = Arc::new(HubMetrics::new());
    let grpc_bus = Arc::clone(&bus);
    let grpc_cancel = cancel.clone();
    let grpc_metrics = Arc::clone(&metrics);
    let grpc_handle = tokio::spawn(async move {
        run_grpc_client(grpc_config, grpc_bus, audio_rx, grpc_cancel, grpc_metrics).await;
    });

    // 4. Start device task with MockTransport.
    let (transport, mut handles) = MockTransport::new("test-device");
    let device_tasks: Arc<DashMap<Arc<str>, CancellationToken>> = Arc::new(DashMap::new());
    let device_cancel = CancellationToken::new();
    let device_metrics = Arc::clone(&metrics);
    let device_handle = tokio::spawn(run_device(
        transport,
        Arc::clone(&bus),
        Arc::from("conv-loopback"),
        VadConfig {
            onset_threshold: 3,
            hangover_threshold: 5,
            mode: VadMode::VeryAggressive,
        },
        device_cancel.clone(),
        Arc::clone(&device_tasks),
        device_metrics,
    ));

    // Give the gRPC client time to connect to the mock server.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 5. Push speech frames — enough to trigger VAD onset and get data flowing.
    for _ in 0..10 {
        handles.frame_tx.send(speech_frame()).await.unwrap();
    }

    // 6. Wait for echoed TTS to arrive back at the mock transport.
    let tts = tokio::time::timeout(Duration::from_secs(5), handles.tts_rx.recv())
        .await
        .expect("timed out waiting for loopback echo")
        .expect("TTS channel closed unexpectedly");

    assert!(!tts.data.is_empty(), "echoed audio should not be empty");
    assert_eq!(tts.sample_rate, 48_000);

    // Metrics should reflect the full pipeline.
    assert!(metrics.vad_onsets.load(Ordering::Relaxed) >= 1);
    assert!(metrics.frames_forwarded.load(Ordering::Relaxed) >= 1);
    assert!(metrics.tts_frames_routed.load(Ordering::Relaxed) >= 1);

    // Cleanup.
    cancel.cancel();
    device_cancel.cancel();
    let _ = tokio::join!(grpc_handle, device_handle);
    server_handle.abort();
}
