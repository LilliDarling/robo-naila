use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::mpsc;

use hub::audio::{AudioFrame, AudioTransport, TransportError, TtsFrame};

/// 48kHz × 20ms = 960 samples per frame.
const SAMPLES_PER_FRAME: usize = 960;

/// A test-only `AudioTransport` backed by channels.
///
/// - Push `AudioFrame`s into `frame_tx` to simulate mic input.
/// - Read `TtsFrame`s from `tts_rx` to verify outbound playback.
/// - Drop `frame_tx` to simulate a transport disconnect.
pub struct MockTransport {
    device_id: Arc<str>,
    frame_rx: mpsc::Receiver<AudioFrame>,
    tts_tx: mpsc::Sender<TtsFrame>,
}

/// Handles returned to the test code for driving a `MockTransport`.
pub struct MockTransportHandles {
    /// Send frames here to simulate device mic input.
    pub frame_tx: mpsc::Sender<AudioFrame>,
    /// Receive TTS frames that `run_device` sends back to the device.
    pub tts_rx: mpsc::Receiver<TtsFrame>,
}

impl MockTransport {
    pub fn new(device_id: &str) -> (Self, MockTransportHandles) {
        let (frame_tx, frame_rx) = mpsc::channel(64);
        let (tts_tx, tts_rx) = mpsc::channel(64);

        let transport = Self {
            device_id: Arc::from(device_id),
            frame_rx,
            tts_tx,
        };

        let handles = MockTransportHandles { frame_tx, tts_rx };

        (transport, handles)
    }
}

impl AudioTransport for MockTransport {
    fn id(&self) -> &str {
        &self.device_id
    }

    async fn recv(&mut self) -> Option<AudioFrame> {
        self.frame_rx.recv().await
    }

    async fn send(&mut self, frame: TtsFrame) -> Result<(), TransportError> {
        self.tts_tx
            .send(frame)
            .await
            .map_err(|_| TransportError::Disconnected)
    }
}

/// A 20ms frame of synthetic speech-like noise.
///
/// Alternating ±16000 square wave at ~1kHz — loud and periodic enough that
/// webrtc-vad reliably classifies it as voice activity.
pub fn speech_frame() -> AudioFrame {
    let bytes_per_frame = SAMPLES_PER_FRAME * 2;
    let mut buf = Vec::with_capacity(bytes_per_frame);
    // ~1kHz square wave: flip polarity every 24 samples (48000/2/1000 ≈ 24).
    let half_period = 24;
    let amplitude: i16 = 16_000;
    for i in 0..SAMPLES_PER_FRAME {
        let sample = if (i / half_period) % 2 == 0 {
            amplitude
        } else {
            -amplitude
        };
        buf.extend_from_slice(&sample.to_le_bytes());
    }
    AudioFrame {
        data: Bytes::from(buf),
        sample_rate: 48_000,
        timestamp: 0,
    }
}
