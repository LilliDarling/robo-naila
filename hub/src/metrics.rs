use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

use crate::grpc::proto;

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
    /// Cached AI server status from the last successful GetStatus call.
    ai_status: Mutex<Option<AiServerStatus>>,
}

/// Cached snapshot of the AI server's GetStatus response, converted to
/// serializable types so we don't hold proto types across the Mutex boundary.
#[derive(Clone, Serialize)]
pub struct AiServerStatus {
    pub health: String,
    pub server_version: String,
    pub uptime_seconds: u64,
    pub max_concurrent_streams: u32,
    pub components: Vec<AiComponentHealth>,
    pub models: Vec<AiModelInfo>,
    pub cpu_utilization: f32,
    pub memory_utilization: f32,
}

#[derive(Clone, Serialize)]
pub struct AiComponentHealth {
    pub name: String,
    pub health: String,
    pub message: String,
}

#[derive(Clone, Serialize)]
pub struct AiModelInfo {
    pub name: String,
    pub model_id: String,
    pub loaded: bool,
    pub device: String,
}

fn health_str(value: i32) -> String {
    match proto::ServerHealth::try_from(value) {
        Ok(proto::ServerHealth::Healthy) => "healthy",
        Ok(proto::ServerHealth::Degraded) => "degraded",
        Ok(proto::ServerHealth::Unhealthy) => "unhealthy",
        _ => "unknown",
    }
    .to_owned()
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
            ai_status: Mutex::new(None),
        }
    }

    /// Cache the AI server's status response.
    pub fn update_ai_status(&self, status: proto::StatusResponse) {
        let components = status
            .components
            .iter()
            .map(|c| AiComponentHealth {
                name: c.name.clone(),
                health: health_str(c.health),
                message: c.message.clone(),
            })
            .collect();

        let models = [
            ("stt", &status.stt_model),
            ("llm", &status.llm_model),
            ("tts", &status.tts_model),
            ("vision", &status.vision_model),
        ]
        .into_iter()
        .filter_map(|(name, maybe_model)| {
            let m = maybe_model.as_ref()?;
            if m.model_id.is_empty() && !m.loaded {
                return None;
            }
            Some(AiModelInfo {
                name: name.to_owned(),
                model_id: m.model_id.clone(),
                loaded: m.loaded,
                device: m.device.clone(),
            })
        })
        .collect();

        let (cpu, mem) = status
            .metrics
            .as_ref()
            .map(|m| (m.cpu_utilization, m.memory_utilization))
            .unwrap_or((0.0, 0.0));

        *self.ai_status.lock().unwrap() = Some(AiServerStatus {
            health: health_str(status.health),
            server_version: status.server_version,
            uptime_seconds: status.uptime_seconds,
            max_concurrent_streams: status.max_concurrent_streams,
            components,
            models,
            cpu_utilization: cpu,
            memory_utilization: mem,
        });
    }

    /// Clear cached AI status (e.g., on disconnect).
    pub fn clear_ai_status(&self) {
        *self.ai_status.lock().unwrap() = None;
    }

    /// Snapshot the current state into a serializable response.
    pub fn snapshot(&self, active_devices: usize) -> HealthResponse {
        let ai_server = self.ai_status.lock().unwrap().clone();

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
            ai_server,
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
    /// AI server status from the last successful GetStatus call.
    /// `None` if the server hasn't been reached or doesn't support GetStatus.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_server: Option<AiServerStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::proto;

    /// Fully populated healthy StatusResponse for test reuse.
    fn healthy_status() -> proto::StatusResponse {
        proto::StatusResponse {
            health: proto::ServerHealth::Healthy as i32,
            server_version: "test-1.0.0".to_owned(),
            uptime_seconds: 42,
            max_concurrent_streams: 4,
            components: vec![
                proto::ComponentHealth {
                    name: "stt".to_owned(),
                    health: proto::ServerHealth::Healthy as i32,
                    message: "ready".to_owned(),
                },
                proto::ComponentHealth {
                    name: "llm".to_owned(),
                    health: proto::ServerHealth::Healthy as i32,
                    message: "ready".to_owned(),
                },
            ],
            stt_model: Some(proto::ModelInfo {
                model_id: "whisper-small.en".to_owned(),
                version: "1.0".to_owned(),
                loaded: true,
                device: "cuda:0".to_owned(),
            }),
            llm_model: Some(proto::ModelInfo {
                model_id: "llama-3-8b".to_owned(),
                version: "1.0".to_owned(),
                loaded: true,
                device: "cuda:0".to_owned(),
            }),
            tts_model: None,
            vision_model: None,
            metrics: Some(proto::ServerMetrics {
                cpu_utilization: 0.45,
                memory_utilization: 0.72,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn update_caches_health_version_uptime() {
        let m = HubMetrics::new();
        m.update_ai_status(healthy_status());

        let ai = m.snapshot(0).ai_server.unwrap();
        assert_eq!(ai.health, "healthy");
        assert_eq!(ai.server_version, "test-1.0.0");
        assert_eq!(ai.uptime_seconds, 42);
        assert_eq!(ai.max_concurrent_streams, 4);
    }

    #[test]
    fn update_maps_components() {
        let m = HubMetrics::new();
        m.update_ai_status(healthy_status());

        let ai = m.snapshot(0).ai_server.unwrap();
        assert_eq!(ai.components.len(), 2);
        assert_eq!(ai.components[0].name, "stt");
        assert_eq!(ai.components[0].health, "healthy");
        assert_eq!(ai.components[0].message, "ready");
        assert_eq!(ai.components[1].name, "llm");
    }

    #[test]
    fn update_maps_loaded_models_and_filters_absent() {
        let m = HubMetrics::new();
        m.update_ai_status(healthy_status());

        let ai = m.snapshot(0).ai_server.unwrap();
        // stt and llm have models; tts and vision are None → filtered out
        assert_eq!(ai.models.len(), 2);
        assert_eq!(ai.models[0].name, "stt");
        assert_eq!(ai.models[0].model_id, "whisper-small.en");
        assert!(ai.models[0].loaded);
        assert_eq!(ai.models[0].device, "cuda:0");
        assert_eq!(ai.models[1].name, "llm");
        assert_eq!(ai.models[1].model_id, "llama-3-8b");
    }

    #[test]
    fn update_filters_empty_unloaded_models() {
        let m = HubMetrics::new();
        let mut status = healthy_status();
        // Empty model_id + not loaded → should be filtered out
        status.tts_model = Some(proto::ModelInfo {
            model_id: String::new(),
            loaded: false,
            ..Default::default()
        });
        m.update_ai_status(status);

        let ai = m.snapshot(0).ai_server.unwrap();
        assert_eq!(ai.models.len(), 2); // stt + llm only
    }

    #[test]
    fn update_extracts_metrics() {
        let m = HubMetrics::new();
        m.update_ai_status(healthy_status());

        let ai = m.snapshot(0).ai_server.unwrap();
        assert!((ai.cpu_utilization - 0.45).abs() < 0.001);
        assert!((ai.memory_utilization - 0.72).abs() < 0.001);
    }

    #[test]
    fn update_defaults_metrics_when_absent() {
        let m = HubMetrics::new();
        let mut status = healthy_status();
        status.metrics = None;
        m.update_ai_status(status);

        let ai = m.snapshot(0).ai_server.unwrap();
        assert_eq!(ai.cpu_utilization, 0.0);
        assert_eq!(ai.memory_utilization, 0.0);
    }

    #[test]
    fn update_maps_degraded_health() {
        let m = HubMetrics::new();
        let mut status = healthy_status();
        status.health = proto::ServerHealth::Degraded as i32;
        m.update_ai_status(status);

        assert_eq!(m.snapshot(0).ai_server.unwrap().health, "degraded");
    }

    #[test]
    fn update_maps_unknown_for_invalid_health_value() {
        let m = HubMetrics::new();
        let mut status = healthy_status();
        status.health = 99; // not a valid ServerHealth variant
        m.update_ai_status(status);

        assert_eq!(m.snapshot(0).ai_server.unwrap().health, "unknown");
    }

    #[test]
    fn clear_removes_cached_status() {
        let m = HubMetrics::new();
        m.update_ai_status(healthy_status());
        assert!(m.snapshot(0).ai_server.is_some());

        m.clear_ai_status();
        assert!(m.snapshot(0).ai_server.is_none());
    }

    #[test]
    fn snapshot_without_status_has_none() {
        let m = HubMetrics::new();
        let snap = m.snapshot(0);
        assert!(snap.ai_server.is_none());
        assert_eq!(snap.status, "ok");
    }

    #[test]
    fn snapshot_reflects_counters_and_devices() {
        let m = HubMetrics::new();
        m.frames_forwarded.store(100, Ordering::Relaxed);
        m.vad_onsets.store(5, Ordering::Relaxed);
        m.vad_ends.store(4, Ordering::Relaxed);
        m.tts_frames_routed.store(50, Ordering::Relaxed);
        m.grpc_reconnects.store(2, Ordering::Relaxed);
        m.grpc_connected.store(true, Ordering::Relaxed);

        let snap = m.snapshot(3);
        assert_eq!(snap.active_devices, 3);
        assert_eq!(snap.frames_forwarded, 100);
        assert_eq!(snap.vad_onsets, 5);
        assert_eq!(snap.vad_ends, 4);
        assert_eq!(snap.tts_frames_routed, 50);
        assert_eq!(snap.grpc_reconnects, 2);
        assert!(snap.grpc_connected);
    }
}
