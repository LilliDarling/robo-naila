pub mod audio;
pub mod grpc;
pub mod http;
pub mod webrtc;

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use audio::AudioBus;
use grpc::{run_grpc_client, GrpcConfig};
use http::{router, AppState};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cancel = CancellationToken::new();

    // ── AudioBus ────────────────────────────────────────────────────────
    // The bus channel carries (device_id, AudioFrame) tuples from all
    // device tasks to the gRPC client. 256 frames of buffer gives ~5s
    // of audio at 20ms/frame, enough to absorb short gRPC stalls.
    let (audio_tx, audio_rx) = mpsc::channel(256);
    let audio_bus = Arc::new(AudioBus {
        audio_tx,
        tts_sub: DashMap::new(),
    });

    // ── gRPC client ─────────────────────────────────────────────────────
    let grpc_cancel = cancel.clone();
    let grpc_bus = Arc::clone(&audio_bus);
    let grpc_handle = tokio::spawn(async move {
        run_grpc_client(GrpcConfig::default(), grpc_bus, audio_rx, grpc_cancel).await;
    });

    // ── HTTP signaling server ───────────────────────────────────────────
    let state = AppState {
        audio_bus: Arc::clone(&audio_bus),
    };
    let app = router(state);
    let bind_addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .expect("failed to bind HTTP listener");
    info!(addr = %bind_addr, "signaling server listening");

    let http_cancel = cancel.clone();
    let http_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(http_cancel.cancelled_owned())
            .await
            .expect("HTTP server error");
    });

    // ── Shutdown ────────────────────────────────────────────────────────
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl-c");
    info!("ctrl-c received, shutting down");
    cancel.cancel();

    let _ = tokio::join!(grpc_handle, http_handle);
    info!("shutdown complete");
}
