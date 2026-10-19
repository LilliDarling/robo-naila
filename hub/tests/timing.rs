#[allow(dead_code)]
mod support;
use bytes::Bytes;
use dashmap::DashMap;
use hub::{
    audio::{AudioBus, TtsFrame},
    http::{AppState, router},
    metrics::HubMetrics,
};
use opentelemetry::{Value, global};
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SdkTracerProvider, SpanData, SpanExporter},
};
use std::{
    sync::{Arc, Mutex, atomic::AtomicU64},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
    time::timeout,
};
use webrtc::{
    api::{
        APIBuilder,
        media_engine::{MIME_TYPE_OPUS, MediaEngine},
    },
    peer_connection::{
        configuration::RTCConfiguration, peer_connection_state::RTCPeerConnectionState,
        sdp::session_description::RTCSessionDescription,
    },
    rtp::{header::Header, packet::Packet},
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{
        TrackLocal, TrackLocalWriter, track_local_static_rtp::TrackLocalStaticRTP,
    },
};

#[derive(Clone, Debug, Default)]
struct Capture(Arc<Mutex<Vec<SpanData>>>);
impl SpanExporter for Capture {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.0.lock().unwrap().extend(batch);
        Ok(())
    }
}
fn number(span: &SpanData, key: &str) -> f64 {
    match &span.attributes.iter().find(|kv| kv.key.as_str() == key).unwrap().value {
        Value::F64(v) => *v,
        Value::I64(v) => *v as f64,
        other => panic!("{other:?}"),
    }
}
fn bus() -> (Arc<AudioBus>, mpsc::Receiver<hub::audio::TaggedFrame>) {
    let (audio_tx, rx) = mpsc::channel(256);
    (Arc::new(AudioBus { audio_tx, tts_sub: DashMap::new(), device_configs: DashMap::new() }), rx)
}

/// Real /connect, ICE/DTLS/RTP and Opus in both directions. No post-decoder injection.
#[tokio::test]
async fn real_webrtc_and_task_owned_measurements() {
    timeout(Duration::from_secs(20), async {
        let capture = Capture::default();
        let provider = SdkTracerProvider::builder().with_simple_exporter(capture.clone()).build();
        global::set_tracer_provider(provider.clone());
        let (bus, mut rx) = bus();
        let tasks = Arc::new(DashMap::new());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = AppState { audio_bus: bus.clone(), connection_counter: Arc::new(AtomicU64::new(0)), device_tasks: tasks.clone(), metrics: Arc::new(HubMetrics::new()) };
        let server = tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap(); });
        let mut media = MediaEngine::default(); media.register_default_codecs().unwrap();
        let pc = Arc::new(APIBuilder::new().with_media_engine(media).build().new_peer_connection(RTCConfiguration::default()).await.unwrap());
        let (ready_tx, mut ready_rx) = mpsc::channel(8);
        pc.on_peer_connection_state_change(Box::new(move |s| { let tx = ready_tx.clone(); Box::pin(async move { if s == RTCPeerConnectionState::Connected { tx.send(()).await.unwrap(); } }) }));
        let (returned_tx, mut returned_rx) = mpsc::channel(8);
        pc.on_track(Box::new(move |track, _, _| { let tx = returned_tx.clone(); Box::pin(async move {
            tokio::spawn(async move { while let Ok((packet, _)) = track.read_rtp().await { if tx.send(packet).await.is_err() { break; } } });
        }) }));
        let outbound = Arc::new(TrackLocalStaticRTP::new(RTCRtpCodecCapability { mime_type: MIME_TYPE_OPUS.into(), ..Default::default() }, "mic".into(), "fixture".into()));
        let sender = pc.add_track(outbound.clone() as Arc<dyn TrackLocal + Send + Sync>).await.unwrap();
        let rtcp = tokio::spawn(async move { let mut buf = [0; 1500]; while sender.read(&mut buf).await.is_ok() {} });
        let offer = pc.create_offer(None).await.unwrap();
        let mut gathered = pc.gathering_complete_promise().await;
        pc.set_local_description(offer).await.unwrap(); gathered.recv().await;
        let sdp = pc.local_description().await.unwrap().sdp.replace('\\', "\\\\").replace('"', "\\\"").replace('\r', "\\r").replace('\n', "\\n");
        let body = format!(r#"{{"device_id":"timing-peer","sdp":"{sdp}"}}"#);
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        socket.write_all(format!("POST /connect HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\ntraceparent: 00-0123456789abcdef0123456789abcdef-0123456789abcdef-01\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        let mut response = String::new(); socket.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let answer = response.split("\r\n\r\n").nth(1).unwrap().strip_prefix("{\"sdp\":\"").unwrap().strip_suffix("\"}").unwrap().replace("\\r", "\r").replace("\\n", "\n");
        pc.set_remote_description(RTCSessionDescription::answer(answer).unwrap()).await.unwrap();
        ready_rx.recv().await.unwrap();
        let mut encoder = opus::Encoder::new(48000, opus::Channels::Mono, opus::Application::Voip).unwrap();
        let mut encoded = [0; 4000];
        let samples: Vec<i16> = support::speech_frame().data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let n = encoder.encode(&samples, &mut encoded).unwrap();
        let payload = Bytes::copy_from_slice(&encoded[..n]);
        // Network waiting belongs to span lifetime, never decoder compute.
        for seq in 0..6u16 {
            tokio::time::sleep(Duration::from_millis(30)).await;
            outbound.write_rtp(&Packet { header: Header { version: 2, payload_type: 111, sequence_number: seq, timestamp: seq as u32 * 960, ssrc: 42, ..Default::default() }, payload: payload.clone() }).await.unwrap();
        }
        // A bus frame proves the actual decoder and VAD have consumed ingress.
        rx.recv().await.unwrap();
        for opus in [true, false] {
            bus.tts_sub.get("timing-peer").unwrap().send(TtsFrame { data: if opus { payload.clone() } else { support::speech_frame().data }, sample_rate: 48000, is_final: false, is_opus: opus }).await.unwrap();
            let packet = returned_rx.recv().await.unwrap();
            if opus { assert_eq!(packet.payload, payload); }
            let mut decoder = opus::Decoder::new(48000, opus::Channels::Mono).unwrap();
            assert_eq!(decoder.decode(&packet.payload, &mut [0; 960], false).unwrap(), 960);
        }
        pc.close().await.unwrap();
        // Explicit cancellation makes read_rtp terminate even if the remote close
        // is not observed promptly by the library.
        tasks.get("timing-peer").unwrap().cancel();
        loop {
            if capture.0.lock().unwrap().iter().any(|s| s.name == "hub.webrtc.receive") && tasks.is_empty() { break; }
            tokio::task::yield_now().await;
        }
        let spans = capture.0.lock().unwrap().clone();
        let receive = spans.iter().find(|s| s.name == "hub.webrtc.receive").unwrap();
        assert_eq!(number(receive, "opus_decode_ms.count"), 6.0);
        assert_eq!(number(receive, "receive_dispatch_ms.count"), 6.0);
        assert!(number(receive, "opus_decode_ms.sum") < 100.0);
        let session = spans.iter().find(|s| s.name == "hub.session").unwrap();
        assert_eq!(number(session, "vad_ms.count"), 6.0); // includes onset-suppressed frames
        assert_eq!(number(session, "outbound_send_ms.count"), 2.0);
        assert_eq!(number(session, "outbound.opus_forward_count"), 1.0);
        assert_eq!(number(session, "outbound.pcm_encode_count"), 1.0);
        assert!(spans.iter().any(|s| s.name == "hub.connect"));
        assert!(spans.iter().all(|s| s.span_context.trace_id() == receive.span_context.trace_id()));
        server.abort(); rtcp.abort(); provider.shutdown().unwrap();
    }).await.expect("WebRTC test deadline");
}
