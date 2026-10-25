use bytes::Bytes;
use opentelemetry::{
    KeyValue,
    trace::{FutureExt, TraceContextExt},
};
use opus::{Application, Channels, Decoder as OpusDecoder, Encoder as OpusEncoder};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use webrtc::rtp::header::Header;
use webrtc::track::track_local::TrackLocalWriter;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_remote::TrackRemote;
use webrtc::{peer_connection::RTCPeerConnection, rtp::packet::Packet};

use crate::audio::{AudioFrame, AudioTransport, TransportError, TtsFrame};

/// Opus operates at 48kHz. All audio entering/leaving WebRTC uses this rate.
const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Opus frame duration in milliseconds. 20ms is the standard for voice.
const OPUS_FRAME_MS: u32 = 20;

/// Samples per Opus frame: 48000 * 20 / 1000 = 960.
const SAMPLES_PER_FRAME: usize = (OPUS_SAMPLE_RATE * OPUS_FRAME_MS / 1000) as usize;

/// Maximum encoded Opus packet size in bytes. Opus typically produces much
/// smaller packets (50-150 bytes for voice), but the spec allows up to ~4KB.
const MAX_OPUS_PACKET: usize = 4000;

/// WebRTC audio transport for Raspberry Pi connections.
///
/// Bridges WebRTC's callback-driven track API into AudioTransport's
/// pull-based async interface via an internal mpsc channel.
///
/// Opus codec is fully encapsulated — downstream consumers only see PCM.
/// The signaling layer constructs this after SDP/ICE negotiation completes.
pub struct WebRtcTransport {
    device_id: Arc<str>,
    audio_rx: Receiver<AudioFrame>,

    // `Arc` = Atomic Reference Counted pointer. Like `Rc`, it enables shared
    // ownership — multiple handles to the same heap allocation. Unlike `Rc`,
    // Arc uses atomic operations for the reference count, making it safe to
    // share across threads (`Send + Sync`). The tradeoff is slightly higher
    // overhead than Rc, but required for async code where tasks may run on
    // different threads.
    //
    // WebRTC types use Arc because the same track/connection is accessed from
    // multiple places: the signaling layer, the transport, and internal webrtc
    // tasks that fire callbacks.
    outbound_track: Arc<TrackLocalStaticRTP>,
    opus_encoder: OpusEncoder,
    peer_connection: Arc<RTCPeerConnection>,

    // Reusable buffers allocated once in `new()`, reused across all `send()`
    // calls to avoid per-frame heap allocations.
    opus_buf: Vec<u8>,
    samples_buf: Vec<i16>,

    rtp_sequence: u16,
    rtp_timestamp: u32,
}

impl WebRtcTransport {
    /// Create a transport from an established WebRTC session.
    ///
    /// Called by the signaling layer after the SDP exchange. The signaling
    /// layer is responsible for:
    ///   - Creating the RTCPeerConnection
    ///   - Registering on_track to spawn a reader task that decodes
    ///     inbound Opus → PCM and feeds `audio_rx`
    ///   - Adding the outbound audio track to the peer connection
    ///   - Completing ICE negotiation
    pub fn new(
        device_id: Arc<str>,
        audio_rx: Receiver<AudioFrame>,
        outbound_track: Arc<TrackLocalStaticRTP>,
        peer_connection: Arc<RTCPeerConnection>,
    ) -> Result<Self, TransportError> {
        let opus_encoder = OpusEncoder::new(OPUS_SAMPLE_RATE, Channels::Mono, Application::Voip)
            .map_err(|e| TransportError::Codec(format!("opus encoder init: {e}")))?;

        Ok(Self {
            device_id,
            audio_rx,
            outbound_track,
            opus_encoder,
            peer_connection,
            opus_buf: vec![0u8; MAX_OPUS_PACKET],
            samples_buf: Vec::new(),
            rtp_sequence: 0,
            rtp_timestamp: 0,
        })
    }
}

impl AudioTransport for WebRtcTransport {
    fn id(&self) -> &str {
        &self.device_id
    }

    async fn recv(&mut self) -> Option<AudioFrame> {
        self.audio_rx.recv().await
    }

    async fn send(&mut self, frame: TtsFrame) -> Result<(), TransportError> {
        if frame.data.is_empty() {
            return Ok(());
        }

        if frame.is_opus {
            // Pre-encoded Opus: forward directly as a single RTP packet.
            let packet = Packet {
                header: Header {
                    version: 2,
                    payload_type: 111,
                    sequence_number: self.rtp_sequence,
                    timestamp: self.rtp_timestamp,
                    ..Default::default()
                },
                payload: Bytes::copy_from_slice(&frame.data),
            };
            self.outbound_track
                .write_rtp(&packet)
                .await
                .map_err(|_| TransportError::Disconnected)?;
            self.rtp_sequence = self.rtp_sequence.wrapping_add(1);
            self.rtp_timestamp = self.rtp_timestamp.wrapping_add(SAMPLES_PER_FRAME as u32);
            return Ok(());
        }

        // PCM path: encode locally with Opus.
        if frame.sample_rate != OPUS_SAMPLE_RATE {
            warn!(
                expected = OPUS_SAMPLE_RATE,
                actual = frame.sample_rate,
                "TtsFrame sample rate mismatch — audio will be garbled"
            );
        }

        if frame.data.len() % 2 != 0 {
            warn!(
                len = frame.data.len(),
                "TtsFrame has odd byte length — last byte will be dropped"
            );
        }

        let sample_count = frame.data.len() / 2;
        self.samples_buf.clear();
        self.samples_buf.reserve(sample_count);
        for c in frame.data.chunks_exact(2) {
            self.samples_buf.push(i16::from_le_bytes([c[0], c[1]]));
        }

        for chunk in self.samples_buf.chunks(SAMPLES_PER_FRAME) {
            if chunk.len() < SAMPLES_PER_FRAME {
                break;
            }

            let encoded_len = self
                .opus_encoder
                .encode(chunk, &mut self.opus_buf)
                .map_err(|e| TransportError::Codec(format!("opus encode: {e}")))?;

            let packet = Packet {
                header: Header {
                    version: 2,
                    payload_type: 111,
                    sequence_number: self.rtp_sequence,
                    timestamp: self.rtp_timestamp,
                    ..Default::default()
                },
                payload: Bytes::copy_from_slice(&self.opus_buf[..encoded_len]),
            };

            self.outbound_track
                .write_rtp(&packet)
                .await
                .map_err(|_| TransportError::Disconnected)?;

            self.rtp_sequence = self.rtp_sequence.wrapping_add(1);
            self.rtp_timestamp = self.rtp_timestamp.wrapping_add(SAMPLES_PER_FRAME as u32);
        }
        Ok(())
    }
}

// `Drop` is Rust's destructor trait — called automatically when a value goes
// out of scope. This is the foundation of RAII (Resource Acquisition Is
// Initialization): resources are tied to object lifetimes, so cleanup happens
// deterministically without manual free() calls or garbage collection.
//
// Here we need to close the WebRTC peer connection, but `close()` is async
// and Drop can't be async. The workaround: clone the Arc (cheap refcount bump)
// and spawn a detached task to do the cleanup. The `let _ =` discards the
// Result since we can't handle errors during drop anyway.
impl Drop for WebRtcTransport {
    fn drop(&mut self) {
        let pc = self.peer_connection.clone();
        tokio::spawn(
            async move {
                let _ = pc.close().await;
            }
            .with_current_context(),
        );
    }
}

/// Spawns a task that reads RTP from an inbound audio track, decodes
/// Opus → PCM, and feeds frames into the provided channel.
///
/// Called by the signaling layer inside the on_track callback.
/// Returns when the track ends or the channel is dropped.
pub fn spawn_track_reader(
    track: Arc<TrackRemote>,
    audio_tx: Sender<AudioFrame>,
    parent: opentelemetry::Context,
    cancel: CancellationToken,
) {
    let mut decoder =
        OpusDecoder::new(OPUS_SAMPLE_RATE, Channels::Mono).expect("opus decoder init");
    tokio::spawn(
        receive_audio(
            move || {
                let track = track.clone();
                async move { track.read_rtp().await.ok().map(|(p, _)| p) }
            },
            move |payload, pcm| decoder.decode(payload, pcm, false),
            audio_tx,
            parent.clone(),
            cancel,
        )
        .with_context(parent),
    );
}

// webrtc 0.17 TrackRemote::read_rtp delivers a packet after transport buffering;
// it supplies no decoder/dispatch timing (src/track/track_remote/mod.rs:272).
// Keeping packet delivery outside the measured calls also permits deterministic
// boundary tests without introducing a production delay option.
async fn receive_audio<R, F, D>(
    mut read: R,
    mut decode: D,
    audio_tx: Sender<AudioFrame>,
    parent: opentelemetry::Context,
    cancel: CancellationToken,
) where
    R: FnMut() -> F,
    F: Future<Output = Option<Packet>>,
    D: FnMut(&[u8], &mut [i16]) -> Result<usize, opus::Error>,
{
    let _cancel_on_exit = cancel.clone().drop_guard();
    let cx = crate::timing::child("hub.webrtc.receive", &parent);
    let mut decode_cost = crate::timing::Costs::default();
    let mut dispatch_cost = crate::timing::Costs::default();
    let mut decode_errors = 0i64;
    let mut dispatch_errors = 0i64;
    let mut pcm_buf = vec![0i16; SAMPLES_PER_FRAME];

    loop {
        let result = tokio::select! {
            result = read() => result,
            _ = cancel.cancelled() => break,
        };
        match result {
            Some(packet) => {
                let before = Instant::now();
                let result = decode(&packet.payload, &mut pcm_buf);
                decode_cost.record(before.elapsed());
                let decoded = match result {
                    Ok(n) => n,
                    Err(e) => {
                        decode_errors += 1;
                        debug!("opus decode error (skipping frame): {e}");
                        continue;
                    }
                };

                let dispatch_start = Instant::now();
                let byte_data: Vec<u8> = pcm_buf[..decoded]
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect();
                let frame = AudioFrame {
                    data: Bytes::from(byte_data),
                    sample_rate: OPUS_SAMPLE_RATE,
                    timestamp: packet.header.timestamp as u64,
                };
                let failed = tokio::select! {
                    result = audio_tx.send(frame) => result.is_err(),
                    _ = cancel.cancelled() => true,
                };
                dispatch_cost.record(dispatch_start.elapsed());
                if failed {
                    dispatch_errors += 1;
                    break;
                }
            }
            None => break,
        }
    }
    decode_cost.export(&cx, "opus_decode_ms");
    dispatch_cost.export(&cx, "receive_dispatch_ms");
    cx.span().set_attributes([
        KeyValue::new("decode_errors", decode_errors),
        KeyValue::new("dispatch_errors", dispatch_errors),
    ]);
    cx.span().end();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio::AudioBus, device::run_device, metrics::HubMetrics, vad::VadConfig};
    use dashmap::DashMap;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
    use crate::test_telemetry::number;
    use std::time::Duration;

    struct Transport {
        id: &'static str,
        rx: Receiver<AudioFrame>,
    }
    impl AudioTransport for Transport {
        fn id(&self) -> &str {
            self.id
        }
        async fn recv(&mut self) -> Option<AudioFrame> {
            self.rx.recv().await
        }
        async fn send(&mut self, _: TtsFrame) -> Result<(), TransportError> {
            tokio::time::sleep(Duration::from_millis(25)).await;
            Ok(())
        }
    }
    fn frame() -> AudioFrame {
        AudioFrame { data: Bytes::from(vec![0; 1920]), sample_rate: 48000, timestamp: 0 }
    }

    #[tokio::test]
    async fn delayed_reader_eof_cancels_only_its_device_with_retained_sender() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let capture = InMemorySpanExporter::default();
            let provider = SdkTracerProvider::builder().with_simple_exporter(capture.clone()).build();
            // Only this unit test installs an exporting provider.
            opentelemetry::global::set_tracer_provider(provider.clone());
            let (audio_tx, audio_rx) = tokio::sync::mpsc::channel(1);
            let retained_sender = audio_tx.clone();
            audio_tx.send(frame()).await.unwrap(); // force dispatch backpressure
            let cancel = CancellationToken::new();
            let other_cancel = CancellationToken::new();
            let (bus_tx, _bus_rx) = tokio::sync::mpsc::channel(256);
            let bus = Arc::new(AudioBus {
                audio_tx: bus_tx,
                tts_sub: DashMap::new(),
                device_configs: DashMap::new(),
            });
            let tasks = Arc::new(DashMap::new());
            tasks.insert(Arc::from("first"), cancel.clone());
            tasks.insert(Arc::from("other"), other_cancel.clone());
            let (other_tx, other_rx) = tokio::sync::mpsc::channel(1);
            let other = tokio::spawn(run_device(
                Transport { id: "other", rx: other_rx },
                bus.clone(),
                Arc::from("2"),
                VadConfig::default(),
                other_cancel.clone(),
                tasks.clone(),
                Arc::new(HubMetrics::new()),
                opentelemetry::Context::new(),
            ));
            let decoded = Arc::new(tokio::sync::Notify::new());
            let notify = decoded.clone();
            let mut calls = 0;
            let mut decodes = 0;
            let reader = tokio::spawn(receive_audio(
                move || {
                    calls += 1;
                    let index = calls;
                    async move {
                        tokio::time::sleep(Duration::from_millis(60)).await;
                        (index <= 2).then(Packet::default)
                    }
                },
                move |_, _| {
                    std::thread::sleep(Duration::from_millis(10));
                    notify.notify_one();
                    decodes += 1;
                    if decodes == 2 {
                        return OpusDecoder::new(48000, Channels::Mono).unwrap().decode(
                            &[0xff],
                            &mut [],
                            false,
                        );
                    }
                    Ok(960)
                },
                audio_tx,
                opentelemetry::Context::new(),
                cancel.clone(),
            ));
            decoded.notified().await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            let device = tokio::spawn(run_device(
                Transport { id: "first", rx: audio_rx },
                bus.clone(),
                Arc::from("1"),
                VadConfig::default(),
                cancel.clone(),
                tasks.clone(),
                Arc::new(HubMetrics::new()),
                opentelemetry::Context::new(),
            ));
            loop {
                if bus.tts_sub.contains_key("first") {
                    break;
                }
                tokio::task::yield_now().await;
            }
            bus.tts_sub
                .get("first")
                .unwrap()
                .send(TtsFrame {
                    data: Bytes::from_static(&[1]),
                    sample_rate: 48000,
                    is_final: false,
                    is_opus: true,
                })
                .await
                .unwrap();
            reader.await.unwrap();
            device.await.unwrap();
            assert!(cancel.is_cancelled());
            assert!(!other_cancel.is_cancelled());
            assert!(!other.is_finished());
            assert!(!bus.tts_sub.contains_key("first"));
            assert!(bus.tts_sub.contains_key("other"));
            let spans = capture.get_finished_spans().unwrap();
            let receive = spans.iter().find(|s| s.name == "hub.webrtc.receive").unwrap();
            assert_eq!(number(receive, "opus_decode_ms.count"), 2.0);
            assert_eq!(number(receive, "decode_errors"), 1.0);
            assert_eq!(number(receive, "receive_dispatch_ms.count"), 1.0);
            assert!(number(receive, "opus_decode_ms.sum") >= 20.0);
            assert!(number(receive, "opus_decode_ms.sum") < 100.0); // excludes 180ms packet waits
            assert!(number(receive, "receive_dispatch_ms.sum") >= 30.0);
            let first = spans
                .iter()
                .find(|s| {
                    s.name == "hub.session"
                        && s.attributes
                            .iter()
                            .any(|kv| kv.key.as_str() == "device_id" && kv.value.as_str() == "first")
                })
                .unwrap();
            assert_eq!(number(first, "vad_ms.count"), 2.0); // both silent/suppressed
            assert_eq!(number(first, "outbound_send_ms.count"), 1.0);
            assert!(number(first, "outbound_send_ms.sum") >= 25.0);
            other_cancel.cancel();
            other.await.unwrap();
            drop((retained_sender, other_tx));
            provider.shutdown().unwrap();
        })
        .await
        .unwrap();
    }
}
