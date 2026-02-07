use std::sync::Arc;
use tokio::sync::mpsc::Sender;
// `Bytes` is a reference-counted, cheaply cloneable byte buffer from the `bytes` crate.
// Unlike Vec<u8>, cloning Bytes doesn't copy the underlying data—it just increments
// a reference count. This is ideal for network code where the same buffer might be
// shared across multiple tasks.
use bytes::Bytes;
use dashmap::DashMap;

/// A frame of audio captured from a robot's microphone.
pub struct AudioFrame {
    /// Raw audio sample data as bytes.
    ///
    /// This is typically PCM (Pulse Code Modulation) audio data—a sequence of
    /// amplitude values representing the sound waveform. The exact format depends
    /// on the encoding (e.g., 16-bit signed integers, little-endian).
    ///
    /// For example, at 16-bit depth: each sample is 2 bytes, so a 16kHz mono
    /// stream produces 32,000 bytes per second.
    pub data: Bytes,

    /// Samples per second (Hz). Common values: 8000, 16000, 44100, 48000.
    pub sample_rate: u32,

    /// Timestamp for synchronization, likely microseconds since some epoch.
    /// Used to order frames and detect gaps in audio streams.
    pub timestamp: u64,
}

/// Speech event tags for VAD-gated audio frames.
///
/// The hub's VAD filter assigns these to each frame it forwards. The AI server
/// uses them to detect utterance boundaries without running its own VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechEvent {
    Start,
    Continue,
    End,
    Interrupt,
}

/// An audio frame tagged with routing metadata, ready for the bus.
///
/// Created by `run_device` after VAD processing. The gRPC send loop reads
/// these off the bus and maps them into `AudioInput` proto messages.
pub struct TaggedFrame {
    pub device_id: Arc<str>,
    pub conversation_id: Arc<str>,
    pub frame: AudioFrame,
    pub event: SpeechEvent,
}

/// TTS = Text-to-Speech
///
/// This frame contains synthesized audio to be played back on the robot's speaker.
/// The AI server converts LLM text responses into audio using models like OuteTTS,
/// then sends these frames back to the robot via MQTT.
pub struct TtsFrame {
    /// Synthesized audio data (same PCM format as AudioFrame).
    pub data: Bytes,
    pub sample_rate: u32,
    /// Hub-consumed signal: last frame of this response. The hub uses this
    /// to distinguish "pause in TTS generation" from "response complete"
    /// for barge-in vs new-utterance detection.
    pub is_final: bool,
}

/// Errors that can occur during audio transport operations.
#[derive(Debug)]
pub enum TransportError {
    /// The connection to the robot was lost (MQTT disconnect, network failure, etc.)
    Disconnected,
    /// Audio encoding/decoding failed (invalid format, corruption, etc.)
    Codec(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Disconnected => write!(f, "transport disconnected"),
            TransportError::Codec(msg) => write!(f, "codec error: {msg}"),
        }
    }
}

impl std::error::Error for TransportError {}

// Native `async fn` in traits has been stable since Rust 1.75. We use it here
// with static dispatch (generics), which gives zero-overhead monomorphized code
// per transport implementation — no heap allocation per call.
//
// Note: native async traits are NOT dyn-compatible. You cannot write
// `Box<dyn AudioTransport>` — the compiler can't build a vtable because each
// impl's futures are different sizes. If you ever need trait objects (e.g. mixing
// transport types in one collection), you'd need the `async_trait` crate or
// `dynosaur` to box the futures. We don't need that here — each robot connection
// is spawned as its own task with a known concrete transport type.
pub trait AudioTransport: Send + 'static {
    // ─────────────────────────────────────────────────────────────────────────────
    // TRAIT BOUNDS EXPLAINED: `Send + 'static`
    // ─────────────────────────────────────────────────────────────────────────────
    //
    // `Send` means: "This type can be safely transferred to another thread."
    //   - Required because async runtimes (like Tokio) may move tasks between
    //     threads. Without Send, you couldn't spawn this in tokio::spawn().
    //   - Most types are Send. Exceptions: Rc<T>, raw pointers, some FFI types.
    //
    // `'static` means: "This type contains no borrowed references (or only 'static ones)."
    //   - This does NOT mean "lives forever"—it means "owns all its data."
    //   - Required for spawning tasks because the runtime can't guarantee when
    //     the task will complete, so it can't hold borrowed data.
    //   - Example: `String` is 'static (owns its data), `&str` is not (borrows).
    //
    // WHY NOT `Send + Sync`?
    //   - `Sync` means: "This type can be safely shared between threads via &T."
    //   - We use `&mut self` in recv/send, so we have exclusive access—no sharing.
    //   - `Send` suffices because we're *moving* or *mutably borrowing*, not sharing.
    //   - If methods took `&self` and we needed concurrent access, we'd need Sync.
    //
    // TL;DR:
    //   - Send  = safe to move between threads (transfer ownership)
    //   - Sync  = safe to share between threads (shared references)
    //   - 'static = owns its data (no borrowed references)
    // ─────────────────────────────────────────────────────────────────────────────

    /// Returns the unique identifier for this transport (e.g., robot's MQTT client ID).
    fn id(&self) -> &str;

    /// Receives the next audio frame from the robot.
    /// Returns `None` when the stream ends (robot disconnected).
    async fn recv(&mut self) -> Option<AudioFrame>;

    /// Sends a TTS audio frame to the robot for playback.
    async fn send(&mut self, frame: TtsFrame) -> Result<(), TransportError>;
}

/// Routes audio between device tasks and the processing layer.
///
/// Shared across all device tasks via `Arc<AudioBus>`. Each device task
/// sends captured audio into `audio_tx` and receives TTS responses from
/// its own channel in `tts_sub`.
///
/// `DashMap` is overkill for 3 devices — `RwLock<HashMap>` would work
/// identically — but gives a cleaner API without explicit lock management.
pub struct AudioBus {
    // We use `Arc<str>` instead of `String` for device IDs throughout the bus.
    // `Arc<str>` is a reference-counted string slice — cloning just bumps an
    // atomic counter instead of copying the string data. This is ideal when
    // the same ID is passed through multiple channels and stored in maps.
    //
    // Why `Arc<str>` instead of `Arc<String>`?
    //   - `Arc<str>` stores the string data inline (one allocation)
    //   - `Arc<String>` has two indirections: Arc → String → str (two allocations)
    //   - `Arc<str>` is created via `Arc::from(s)` where s: String or &str
    pub audio_tx: Sender<TaggedFrame>,
    pub tts_sub: DashMap<Arc<str>, Sender<TtsFrame>>,
}
