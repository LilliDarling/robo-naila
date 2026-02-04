use bytes::Bytes;
use opus::{Application, Channels, Decoder as OpusDecoder, Encoder as OpusEncoder};
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, Sender};
use webrtc::rtp::header::Header;
use webrtc::rtp_transceiver::rtp_codec::RTPCodecType;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::TrackLocalWriter;
use webrtc::track::track_remote::TrackRemote;
use webrtc::{peer_connection::RTCPeerConnection, rtp::packet::Packet};

use crate::transport::audio::{AudioFrame, AudioTransport, TransportError, TtsFrame};

/// Opus operates at 48kHz. All audio entering/leaving WebRTC uses this rate.
const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Opus frame duration in milliseconds. 20ms is the standard for voice.
const OPUS_FRAME_MS: u32 = 20;

/// Samples per Opus frame: 48000 * 20 / 1000 = 960.
const OPUS_FRAME_SIZE: usize = (OPUS_SAMPLE_RATE * OPUS_FRAME_MS / 1000) as usize;

/// Max size of a decoded PCM frame buffer (mono, 16-bit = 2 bytes per sample).
const PCM_BUF_SIZE: usize = OPUS_FRAME_SIZE;

/// WebRTC audio transport for Raspberry Pi connections.
///
/// Bridges WebRTC's callback-driven track API into AudioTransport's
/// pull-based async interface via an internal mpsc channel.
///
/// Opus codec is fully encapsulated — downstream consumers only see PCM.
/// The signaling layer constructs this after SDP/ICE negotiation completes.
pub struct WebRtcTransport {
    device_id: String,
    audio_rx: Receiver<AudioFrame>,
    outbound_track: Arc<TrackLocalStaticRTP>,
    opus_encoder: OpusEncoder,
    peer_connection: Arc<RTCPeerConnection>,
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
        device_id: String,
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
        let samples: Vec<i16> = frame
            .data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Encode PCM → Opus in frame-sized chunks.
        for chunk in samples.chunks(PCM_BUF_SIZE) {
            if chunk.len() < OPUS_FRAME_SIZE {
                break; // Drop incomplete frames.
            }

            let mut opus_buf = vec![0u8; 4000]; // max Opus packet size
            let encoded_len = self
                .opus_encoder
                .encode(chunk, &mut opus_buf)
                .map_err(|e| TransportError::Codec(format!("opus encode: {e}")))?;

            let packet = Packet {
                header: Header {
                    version: 2,
                    payload_type: 111, // standard Opus PT
                    sequence_number: self.rtp_sequence,
                    timestamp: self.rtp_timestamp,
                    ..Default::default()
                },
                payload: Bytes::copy_from_slice(&opus_buf[..encoded_len]),
            };

            self.outbound_track
                .write_rtp(&packet)
                .await
                .map_err(|_| TransportError::Disconnected)?;

            self.rtp_sequence = self.rtp_sequence.wrapping_add(1);
            self.rtp_timestamp = self.rtp_timestamp.wrapping_add(OPUS_FRAME_SIZE as u32);
        }
        Ok(())
    }
}

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
    tokio::spawn(async move {
        let mut decoder =
            OpusDecoder::new(OPUS_SAMPLE_RATE, Channels::Mono).expect("opus decoder init");
        let mut pcm_buf = vec![0i16; PCM_BUF_SIZE];

        loop {
            match track.read_rtp().await {
                Ok((packet, _)) => {
                    let decoded = match decoder.decode(&packet.payload, &mut pcm_buf, false) {
                        Ok(n) => n,
                        Err(_) => continue, // skip corrupted frames
                    };

                    // Convert i16 PCM samples to bytes (little-endian).
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
