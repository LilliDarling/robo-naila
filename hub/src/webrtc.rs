use bytes::Bytes;
use opus::{Application, Channels, Decoder as OpusDecoder, Encoder as OpusEncoder};
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::{debug, warn};
use webrtc::rtp::header::Header;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::TrackLocalWriter;
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

    // Reusable buffer for Opus encoding output. Allocated once in `new()`,
    // reused across all `send()` calls to avoid per-frame heap allocations.
    opus_buf: Vec<u8>,

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
        // Validate sample rate — mismatches cause garbled audio with no other symptom.
        if frame.sample_rate != OPUS_SAMPLE_RATE {
            warn!(
                expected = OPUS_SAMPLE_RATE,
                actual = frame.sample_rate,
                "TtsFrame sample rate mismatch — audio will be garbled"
            );
        }

        // PCM is 16-bit (2 bytes per sample). Odd-length data means truncated sample.
        if frame.data.len() % 2 != 0 {
            warn!(
                len = frame.data.len(),
                "TtsFrame has odd byte length — last byte will be dropped"
            );
        }

        let samples: Vec<i16> = frame
            .data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Encode PCM → Opus in frame-sized chunks.
        for chunk in samples.chunks(SAMPLES_PER_FRAME) {
            if chunk.len() < SAMPLES_PER_FRAME {
                break; // Drop incomplete frames.
            }

            let encoded_len = self
                .opus_encoder
                .encode(chunk, &mut self.opus_buf)
                .map_err(|e| TransportError::Codec(format!("opus encode: {e}")))?;

            let packet = Packet {
                header: Header {
                    version: 2,
                    payload_type: 111, // standard Opus PT
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

            // `wrapping_add` performs addition that wraps around on overflow instead
            // of panicking (debug) or being undefined behavior (release). RTP sequence
            // numbers and timestamps are designed to wrap — a u16 sequence goes
            // 65534 → 65535 → 0 → 1, and receivers handle this gracefully.
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
        tokio::spawn(async move {
            let _ = pc.close().await;
        });
    }
}

/// Spawns a task that reads RTP from an inbound audio track, decodes
/// Opus → PCM, and feeds frames into the provided channel.
///
/// Called by the signaling layer inside the on_track callback.
/// Returns when the track ends or the channel is dropped.
pub fn spawn_track_reader(track: Arc<TrackRemote>, audio_tx: Sender<AudioFrame>) {
    // `tokio::spawn` schedules an async task to run concurrently on the Tokio
    // runtime. Unlike a regular function call, spawn returns immediately — the
    // task runs in the background.
    //
    // `async move { ... }` creates an async block that takes ownership of
    // captured variables (`track`, `audio_tx`). The `move` keyword transfers
    // ownership into the closure; without it, the closure would try to borrow,
    // which fails because the spawned task may outlive the current scope.
    tokio::spawn(async move {
        // `.expect()` panics with a message if the Result is Err. Unlike `unwrap()`,
        // it documents *why* we believe this can't fail. Here, decoder creation
        // only fails with invalid parameters (wrong sample rate, channel count),
        // which are compile-time constants — so failure indicates a programmer
        // error, not a runtime condition we should handle gracefully.
        let mut decoder =
            OpusDecoder::new(OPUS_SAMPLE_RATE, Channels::Mono).expect("opus decoder init");
        let mut pcm_buf = vec![0i16; SAMPLES_PER_FRAME];

        loop {
            match track.read_rtp().await {
                Ok((packet, _)) => {
                    let decoded = match decoder.decode(&packet.payload, &mut pcm_buf, false) {
                        Ok(n) => n,
                        Err(e) => {
                            debug!("opus decode error (skipping frame): {e}");
                            continue;
                        }
                    };

                    // Convert i16 PCM samples to bytes (little-endian).
                    // `flat_map` yields individual bytes from each sample's 2-byte representation.
                    // `collect()` allocates a Vec, then `Bytes::from()` takes ownership without
                    // copying — this is more efficient than copy_from_slice with a reused buffer.
                    let byte_data: Vec<u8> = pcm_buf[..decoded]
                        .iter()
                        .flat_map(|s| s.to_le_bytes())
                        .collect();

                    let frame = AudioFrame {
                        data: Bytes::from(byte_data),
                        sample_rate: OPUS_SAMPLE_RATE,
                        timestamp: packet.header.timestamp as u64,
                    };

                    if audio_tx.send(frame).await.is_err() {
                        break; // transport dropped
                    }
                }
                Err(_) => break, // track ended
            }
        }
    });
}
