use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

/// Shared counters and gauges for the hub. Created once in `main()`,
/// passed as `Arc<HubMetrics>` to subsystems that bump the values.
/// Exposed via `GET /health`.
pub struct HubMetrics {
    /// Total audio frames forwarded from devices to the gRPC bus.
    pub frames_forwarded: AtomicU64,
    /// Number of VAD speech-onset events (Start).
    pub vad_onsets: AtomicU64,
    /// Number of VAD speech-end events (End).
    pub vad_ends: AtomicU64,
    /// TTS frames routed from gRPC back to devices.
    pub tts_frames_routed: AtomicU64,
    /// gRPC reconnection attempts.
    pub grpc_reconnects: AtomicU64,
    /// Whether the gRPC stream is currently established.
    pub grpc_connected: AtomicBool,
    /// Process start time (for uptime calculation).
    pub start_time: Instant,
}

impl HubMetrics {
    pub fn new() -> Self {
        Self {
            frames_forwarded: AtomicU64::new(0),
            vad_onsets: AtomicU64::new(0),
            vad_ends: AtomicU64::new(0),
            tts_frames_routed: AtomicU64::new(0),
            grpc_reconnects: AtomicU64::new(0),
            grpc_connected: AtomicBool::new(false),
            start_time: Instant::now(),
        }
    }

    /// Snapshot the current state into a serializable response.
    pub fn snapshot(&self, active_devices: usize) -> HealthResponse {
        HealthResponse {
            status: "ok",
            uptime_secs: self.start_time.elapsed().as_secs(),
            active_devices,
            grpc_connected: self.grpc_connected.load(Ordering::Relaxed),
            frames_forwarded: self.frames_forwarded.load(Ordering::Relaxed),
            vad_onsets: self.vad_onsets.load(Ordering::Relaxed),
            vad_ends: self.vad_ends.load(Ordering::Relaxed),
            tts_frames_routed: self.tts_frames_routed.load(Ordering::Relaxed),
            grpc_reconnects: self.grpc_reconnects.load(Ordering::Relaxed),
        }
    }
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub uptime_secs: u64,
    pub active_devices: usize,
    pub grpc_connected: bool,
    pub frames_forwarded: u64,
    pub vad_onsets: u64,
    pub vad_ends: u64,
    pub tts_frames_routed: u64,
    pub grpc_reconnects: u64,
}
