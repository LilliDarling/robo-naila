use crate::audio::{AudioFrame, SpeechEvent};
use webrtc_vad::{SampleRate, Vad, VadMode};

/// Configuration for the VAD filter.
pub struct VadConfig {
    /// Consecutive speech frames required before confirming speech onset.
    /// Higher values reduce false triggers but add latency.
    /// Default: 3 frames = 60ms at 20ms/frame.
    pub onset_threshold: u32,
    /// Consecutive silence frames after speech before emitting End.
    /// Prevents premature cutoff during brief pauses mid-sentence.
    /// Default: 25 frames = 500ms at 20ms/frame.
    pub hangover_threshold: u32,
    /// VAD aggressiveness. Higher = fewer false positives but more missed speech.
    pub mode: VadMode,
}


impl Default for VadConfig {
    fn default() -> Self {
        Self {
            onset_threshold: 3,
            // #38 operator amendment: bridge speech pauses with 500ms hangover.
            hangover_threshold: 25,
            mode: VadMode::VeryAggressive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    /// No speech detected. Frames are suppressed (not forwarded).
    Silence,
    /// Speech possibly starting. Buffering frames until onset_threshold is met.
    MaybeSpeech,
    /// Confirmed speech. Frames are forwarded with Continue events.
    Speaking,
    /// Speech possibly ending. Still forwarding during hangover countdown.
    MaybeSilence,
}

/// Wraps `webrtc-vad` with a state machine that tracks speech boundaries
/// and produces `SpeechEvent` tags for each frame.
///
/// Not Send (libfvad C pointer) — must live on a single task.
/// 48kHz * 20ms = 960 samples per frame.
const SAMPLES_PER_FRAME: usize = 960;

// Safety: The inner `Vad` holds a `*mut Fvad` (C FFI pointer) which makes it
// `!Send` by Rust's conservative auto-trait rules. However, libfvad's `Fvad`
// struct is purely instance-local data (VadInstT + rate index) with no
// thread-local storage, no global mutable state, and no OS thread-affine
// resources — it is safe to move between threads. The remaining VadFilter
// fields are all plain owned types that are Send.
unsafe impl Send for VadFilter {}

pub struct VadFilter {
    vad: Vad,
    state: VadState,
    onset_count: u32,
    hangover_count: u32,
    onset_threshold: u32,
    hangover_threshold: u32,
    /// Frames buffered during MaybeSpeech, flushed on confirmed onset.
    onset_buf: Vec<AudioFrame>,
    /// Reusable sample buffer — avoids allocation per frame.
    sample_buf: Vec<i16>,
}

/// Result of processing a single frame through the VAD filter.
pub enum VadResult {
    /// Frame(s) should be forwarded. Contains the event tag and all frames
    /// to send (may be multiple if flushing onset buffer).
    Emit(SpeechEvent, Vec<AudioFrame>),
    /// Frame suppressed (silence, or unconfirmed onset that was discarded).
    Suppress,
}

impl VadFilter {
    pub fn new(config: VadConfig) -> Self {
        let vad = Vad::new_with_rate_and_mode(SampleRate::Rate48kHz, config.mode);

        Self {
            vad,
            state: VadState::Silence,
            onset_count: 0,
            hangover_count: 0,
            onset_threshold: config.onset_threshold,
            hangover_threshold: config.hangover_threshold,
            onset_buf: Vec::with_capacity(config.onset_threshold as usize),
            sample_buf: Vec::with_capacity(SAMPLES_PER_FRAME),
        }
    }

    /// Feed a 20ms PCM frame (960 samples at 48kHz). Returns whether and how
    /// to forward the frame(s) to the audio bus.
    pub fn process(&mut self, frame: AudioFrame) -> VadResult {
        // Reuse the sample buffer — no allocation after the first frame.
        self.sample_buf.clear();
        self.sample_buf.extend(
            frame
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]])),
        );

        let is_speech = self.vad.is_voice_segment(&self.sample_buf).unwrap_or(false);

        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.state = VadState::MaybeSpeech;
                    self.onset_count = 1;
                    self.onset_buf.clear();
                    self.onset_buf.push(frame);
                    VadResult::Suppress
                } else {
                    VadResult::Suppress
                }
            }

            VadState::MaybeSpeech => {
                if is_speech {
                    self.onset_count += 1;
                    self.onset_buf.push(frame);

                    if self.onset_count >= self.onset_threshold {
                        // Confirmed speech — flush buffered onset frames.
                        self.state = VadState::Speaking;
                        let frames = std::mem::take(&mut self.onset_buf);
                        VadResult::Emit(SpeechEvent::Start, frames)
                    } else {
                        VadResult::Suppress
                    }
                } else {
                    // False alarm — discard buffered frames.
                    self.state = VadState::Silence;
                    self.onset_count = 0;
                    self.onset_buf.clear();
                    VadResult::Suppress
                }
            }

            VadState::Speaking => {
                if is_speech {
                    VadResult::Emit(SpeechEvent::Continue, vec![frame])
                } else {
                    self.state = VadState::MaybeSilence;
                    self.hangover_count = 1;
                    // Still forward during hangover.
                    VadResult::Emit(SpeechEvent::Continue, vec![frame])
                }
            }

            VadState::MaybeSilence => {
                if is_speech {
                    // Speech resumed — back to Speaking.
                    self.state = VadState::Speaking;
                    self.hangover_count = 0;
                    VadResult::Emit(SpeechEvent::Continue, vec![frame])
                } else {
                    self.hangover_count += 1;
                    if self.hangover_count >= self.hangover_threshold {
                        // Hangover expired — utterance is over.
                        self.state = VadState::Silence;
                        self.hangover_count = 0;
                        VadResult::Emit(SpeechEvent::End, vec![frame])
                    } else {
                        // Still in hangover — keep forwarding.
                        VadResult::Emit(SpeechEvent::Continue, vec![frame])
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use opus::{Channels, Decoder};
    use std::{
        io::{Cursor, Read},
        path::Path,
        process::Command,
    };

    #[test]
    fn unchanged_question_through_opus_is_one_utterance() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let python = std::env::var_os("NAILA_TEST_PEER_PYTHON")
            .unwrap_or_else(|| root.join("devices/audio-client/.venv/bin/python").into());
        let available = Command::new(&python)
            .args(["-c", "from aiortc.codecs.opus import OpusEncoder; OpusEncoder()"])
            .output();
        if !available.is_ok_and(|output| output.status.success()) {
            eprintln!("SKIP unchanged_question_through_opus_is_one_utterance: audio-client Python or aiortc Opus encoder unavailable ({python:?})");
            return;
        }
        // aiortc 1.14.0 OpusEncoder.encode returns payloads, not a framed stream:
        // https://github.com/aiortc/aiortc/blob/1.14.0/src/aiortc/codecs/opus.py
        // Length-prefix only this test's subprocess output; use the real peer codec.
        let encoded = Command::new(&python)
            .current_dir(root)
            .env("PYTHONPATH", root)
            .args([
                "-c",
                r#"
import sys
from fractions import Fraction
from aiortc.codecs.opus import OpusEncoder
from av import AudioFrame
from benchmarks.peer import Fixture
fixture = Fixture.load('benchmarks/fixtures/question.wav')
pcm = bytes(25 * 1920) + fixture.pcm + bytes(50 * 1920)
encoder = OpusEncoder()
for offset in range(0, len(pcm), 1920):
    frame = AudioFrame(format='s16', layout='mono', samples=960)
    frame.planes[0].update(pcm[offset:offset + 1920])
    frame.sample_rate, frame.time_base, frame.pts = 48000, Fraction(1, 48000), offset // 2
    packets, _ = encoder.encode(frame)
    for packet in packets:
        sys.stdout.buffer.write(len(packet).to_bytes(2, 'little') + packet)
    "#,
            ])
            .output()
            .expect("install the benchmark environment documented in benchmarks/README.md");
        assert!(
            encoded.status.success(),
            "{}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        let mut input = Cursor::new(encoded.stdout);
        let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
        let mut filters = [
            VadFilter::new(VadConfig::default()),
            VadFilter::new(VadConfig {
                hangover_threshold: 15,
                ..VadConfig::default()
            }),
        ];
        let mut events = [(0, 0); 2];
        let mut pcm = [0i16; 960];
        while input.position() < input.get_ref().len() as u64 {
            let mut length = [0u8; 2];
            input.read_exact(&mut length).unwrap();
            let mut packet = vec![0; u16::from_le_bytes(length) as usize];
            input.read_exact(&mut packet).unwrap();
            assert_eq!(decoder.decode(&packet, &mut pcm, false).unwrap(), 960);
            let data: Bytes = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
            for (filter, (starts, ends)) in filters.iter_mut().zip(&mut events) {
                let frame = AudioFrame {
                    data: data.clone(),
                    sample_rate: 48_000,
                    timestamp: 0,
                };
                match filter.process(frame) {
                    VadResult::Emit(SpeechEvent::Start, _) => *starts += 1,
                    VadResult::Emit(SpeechEvent::End, _) => *ends += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(
            events[1],
            (2, 2),
            "15-frame control must reproduce the split"
        );
        assert_eq!(
            events[0],
            (1, 1),
            "default must bridge the internal speech pause"
        );
    }
}
