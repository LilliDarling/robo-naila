use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::{get, post}, Json, Router};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::api::APIBuilder;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTPCodecType};
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::TrackLocal;

use crate::audio::{AudioBus, AudioFrame};
use crate::metrics::HubMetrics;
use crate::webrtc::{spawn_track_reader, WebRtcTransport};

// ─────────────────────────────────────────────────────────────────────────────
// Request / Response types for the signaling endpoint
// ─────────────────────────────────────────────────────────────────────────────

/// Pi sends this: its SDP offer and a device identifier.
#[derive(Deserialize)]
pub struct ConnectRequest {
    /// Device identifier (e.g., "pi-kitchen", "pi-office").
    pub device_id: String,
    /// SDP offer from the Pi's WebRTC client.
    pub sdp: String,
}

/// Server responds with its SDP answer.
#[derive(Serialize)]
pub struct ConnectResponse {
    /// SDP answer with ICE candidates gathered.
    pub sdp: String,
}

/// Error response body.
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared application state injected into axum handlers
// ─────────────────────────────────────────────────────────────────────────────

/// Holds everything the signaling handler needs to create peer connections
/// and wire them into the rest of the system.
#[derive(Clone)]
pub struct AppState {
    pub audio_bus: Arc<AudioBus>,
    /// Monotonic counter for conversation IDs. Each POST /connect bumps this;
    /// the value becomes the conversation_id for that WebRTC session.
    pub connection_counter: Arc<AtomicU64>,
    /// Active device sessions. Used for reconnect dedup — if a device
    /// reconnects, we cancel the old session before starting a new one.
    pub device_tasks: Arc<DashMap<Arc<str>, CancellationToken>>,
    pub metrics: Arc<HubMetrics>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────────────────────────────────────

/// Creates the axum router with the signaling endpoint.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/connect", post(handle_connect))
        .route("/health", get(handle_health))
        .with_state(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler
// ─────────────────────────────────────────────────────────────────────────────

/// GET /health
async fn handle_health(State(state): State<AppState>) -> impl IntoResponse {
    let snapshot = state.metrics.snapshot(state.device_tasks.len());
    Json(snapshot)
}

/// POST /connect
///
/// Accepts an SDP offer from a Pi, negotiates a WebRTC session, and returns
/// the SDP answer. On success, a `WebRtcTransport` is created and a device
/// task is spawned via `run_device`.
///
/// Flow:
///   1. Build a MediaEngine with Opus codec
///   2. Create RTCPeerConnection (no ICE servers — local network)
///   3. Add outbound audio track (server → Pi for TTS playback)
///   4. Register on_track callback (Pi → server for mic capture)
///   5. Set remote description (Pi's offer)
///   6. Create answer, set as local description
///   7. Wait for ICE gathering to complete
///   8. Return SDP answer with candidates baked in
async fn handle_connect(
    // Axum "extractors" pull data from the incoming request using pattern matching.
    // `State(state)` extracts the shared AppState we registered with `.with_state()`.
    // `Json(req)` deserializes the request body as JSON into ConnectRequest.
    //
    // Extractors are the core of Axum's design — they're composable, type-safe,
    // and the order matters (body-consuming extractors like Json must come last).
    State(state): State<AppState>,
    Json(req): Json<ConnectRequest>,
    // `impl IntoResponse` lets us return different response types from the same handler.
    // Axum will call `.into_response()` on whatever we return. This is more flexible
    // than a concrete type because we can return Ok (200 + JSON) or Err (500 + JSON)
    // without wrapping in Result — both tuple types implement IntoResponse.
) -> impl IntoResponse {
    info!(device_id = %req.device_id, "incoming WebRTC connection");

    match negotiate_session(&state, req).await {
        Ok(answer_sdp) => {
            (StatusCode::OK, Json(ConnectResponse { sdp: answer_sdp })).into_response()
        }
        Err(e) => {
            error!("signaling failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    }
}

async fn negotiate_session(
    state: &AppState,
    req: ConnectRequest,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Convert device_id to Arc<str> once, then use Arc::clone() everywhere.
    // This avoids String clones in callbacks and the run_device loop.
    let device_id: Arc<str> = req.device_id.into();

    // Reconnect dedup: if this device already has an active session, cancel it.
    if let Some((_, old_cancel)) = state.device_tasks.remove(device_id.as_ref()) {
        warn!(device_id = %device_id, "replacing existing device session");
        old_cancel.cancel();
    }

    // Per-device cancel token — fired on peer disconnect or reconnect dedup.
    // Inserted into device_tasks later, after all fallible setup succeeds.
    let device_cancel = CancellationToken::new();

    // Bump the global counter to get a unique conversation ID for this session.
    let conversation_id: Arc<str> = state
        .connection_counter
        .fetch_add(1, Ordering::Relaxed)
        .to_string()
        .into();

    // ── 1. Media engine with Opus ──────────────────────────────────────
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;

    // ── 2. Interceptors (RTCP feedback, NACK, etc.) ────────────────────
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;

    // ── 3. Build the API ───────────────────────────────────────────────
    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();

    // ── 4. Create peer connection (no ICE servers for local network) ───
    let config = RTCConfiguration {
        ice_servers: vec![],
        ..Default::default()
    };
    let peer_connection = Arc::new(api.new_peer_connection(config).await?);

    // ── 5. Add outbound audio track (server → Pi speaker) ──────────────
    let outbound_track = Arc::new(TrackLocalStaticRTP::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_owned(),
            ..Default::default()
        },
        "audio".to_owned(),      // track id
        "tts-stream".to_owned(), // stream id
    ));

    let rtp_sender = peer_connection
        .add_track(Arc::clone(&outbound_track) as Arc<dyn TrackLocal + Send + Sync>)
        .await?;

    // Drain RTCP packets from the sender (required for interceptors like NACK).
    tokio::spawn(async move {
        let mut buf = vec![0u8; 1500];
        while let Ok((_, _)) = rtp_sender.read(&mut buf).await {}
    });

    // ── 6. on_track: wire inbound audio into the transport channel ──────
    //
    // The channel bridges WebRTC's callback world into our pull-based
    // AudioTransport. spawn_track_reader decodes Opus → PCM and pushes
    // AudioFrames into audio_tx. The transport reads from audio_rx.
    let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(64);

    let device_id_for_track = Arc::clone(&device_id);
    // WebRTC callbacks require a specific signature: Box<dyn FnMut(...) -> Pin<Box<dyn Future>>>
    // This unusual pattern exists because:
    //   1. The callback might be called multiple times (FnMut, not FnOnce)
    //   2. It needs to be stored in the PeerConnection (Box<dyn ...>)
    //   3. It returns an async block, but trait objects can't have async fn
    //      so we manually box the future with Box::pin(async move { ... })
    //
    // The `move` keywords transfer ownership into each closure layer:
    //   - Outer `move`: captures device_id_for_track, audio_tx into the FnMut
    //   - Inner `move` (in Box::pin): moves clones into the async block
    peer_connection.on_track(Box::new(move |track, _receiver, _transceiver| {
        let audio_tx = audio_tx.clone();
        let device_id = Arc::clone(&device_id_for_track);

        Box::pin(async move {
            // Only handle audio tracks.
            if track.kind() == RTPCodecType::Audio {
                info!(device_id = %device_id, "audio track negotiated");
                spawn_track_reader(track, audio_tx);
            }
        })
    }));

    // ── 7. Connection state monitoring + cancel on peer failure ─────────
    let device_id_for_state = Arc::clone(&device_id);
    let device_cancel_for_state = device_cancel.clone();
    peer_connection.on_peer_connection_state_change(Box::new(
        move |state: RTCPeerConnectionState| {
            let device_id = Arc::clone(&device_id_for_state);
            let cancel = device_cancel_for_state.clone();
            Box::pin(async move {
                match state {
                    RTCPeerConnectionState::Connected => {
                        info!(device_id = %device_id, "peer connected");
                    }
                    RTCPeerConnectionState::Disconnected => {
                        warn!(device_id = %device_id, "peer disconnected");
                    }
                    RTCPeerConnectionState::Failed | RTCPeerConnectionState::Closed => {
                        error!(device_id = %device_id, "peer connection failed/closed");
                        cancel.cancel();
                    }
                    _ => {}
                }
            })
        },
    ));

    // ── 8. SDP exchange ─────────────────────────────────────────────────
    let offer = RTCSessionDescription::offer(req.sdp.clone())?;
    peer_connection.set_remote_description(offer).await?;

    let answer = peer_connection.create_answer(None).await?;

    // Set local description to start ICE gathering.
    let mut gathering_complete = peer_connection.gathering_complete_promise().await;
    peer_connection.set_local_description(answer).await?;

    // Wait for ICE gathering to finish. On a local network this is fast
    // (typically <100ms). The answer we return will contain all candidates
    // baked into the SDP, so the Pi doesn't need trickle ICE.
    let _ = gathering_complete.recv().await;

    // ── 9. Extract the final SDP answer ─────────────────────────────────
    let local_desc = peer_connection
        .local_description()
        .await
        .ok_or("local description missing after ICE gathering")?;

    // ── 10. Build the transport and spawn the device task ────────────────
    let transport = WebRtcTransport::new(
        Arc::clone(&device_id),
        audio_rx,
        outbound_track,
        peer_connection,
    )?;

    // Register in device_tasks only after all fallible setup has succeeded.
    // This avoids leaking a stale entry if negotiation fails partway through.
    state
        .device_tasks
        .insert(Arc::clone(&device_id), device_cancel.clone());

    let audio_bus = Arc::clone(&state.audio_bus);
    let conversation_id_for_device = Arc::clone(&conversation_id);
    let device_tasks_ref = Arc::clone(&state.device_tasks);
    let metrics = Arc::clone(&state.metrics);
    tokio::spawn(async move {
        crate::device::run_device(
            transport,
            audio_bus,
            conversation_id_for_device,
            crate::vad::VadConfig::default(),
            device_cancel,
            device_tasks_ref,
            metrics,
        )
        .await;
    });

    info!(device_id = %device_id, "session negotiated, device task spawned");

    Ok(local_desc.sdp)
}
