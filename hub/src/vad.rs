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
    /// Default: 15 frames = 300ms at 20ms/frame.
    pub hangover_threshold: u32,
    /// VAD aggressiveness. Higher = fewer false positives but more missed speech.
    pub mode: VadMode,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            onset_threshold: 3,
            hangover_threshold: 15,
            mode: VadMode::Quality,
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
