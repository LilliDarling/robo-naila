use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use std::sync::atomic::AtomicU64;

use hub::audio::AudioBus;
use hub::grpc::{run_grpc_client, GrpcConfig};
use hub::http::{router, AppState};

// `#[tokio::main]` transforms `async fn main()` into a regular `fn main()` that
// creates a Tokio runtime and blocks on the async body. Without this, you'd need:
//   fn main() { tokio::runtime::Runtime::new().unwrap().block_on(async { ... }) }
//
// The macro uses the multi-threaded runtime by default. For single-threaded:
//   #[tokio::main(flavor = "current_thread")]
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cancel = CancellationToken::new();

    // ── AudioBus ────────────────────────────────────────────────────────
    // The bus channel carries TaggedFrames from all device tasks to the
    // gRPC client. 256 frames of buffer gives ~5s of audio at 20ms/frame,
    // enough to absorb short gRPC stalls.
    let (audio_tx, audio_rx) = mpsc::channel(256);
    let audio_bus = Arc::new(AudioBus {
        audio_tx,
        tts_sub: DashMap::new(),
    });

    // ── gRPC client ─────────────────────────────────────────────────────
    // `tokio::spawn` schedules a task to run concurrently. Unlike calling an
    // async function directly (which runs inline when awaited), spawn creates
    // an independent task that runs in the background on the runtime's thread pool.
    //
    // The returned `JoinHandle` lets us await the task's completion later.
    // If we drop the handle without awaiting, the task keeps running (detached).
    let grpc_cancel = cancel.clone();
    let grpc_bus = Arc::clone(&audio_bus);
    let grpc_handle = tokio::spawn(async move {
        run_grpc_client(GrpcConfig::default(), grpc_bus, audio_rx, grpc_cancel).await;
    });

    // ── HTTP signaling server ───────────────────────────────────────────
    let state = AppState {
        audio_bus: Arc::clone(&audio_bus),
        connection_counter: Arc::new(AtomicU64::new(0)),
        device_tasks: Arc::new(DashMap::new()),
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
    // `tokio::signal::ctrl_c()` returns a future that completes when the process
    // receives SIGINT (Ctrl+C on Unix, Ctrl+C/Ctrl+Break on Windows). This is
    // the standard way to implement graceful shutdown in async Rust.
    //
    // Pattern: main awaits the signal, then cancels all child tasks, then awaits
    // their completion. This ensures clean shutdown (connections closed, buffers
    // flushed) rather than abrupt termination.
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl-c");
    info!("ctrl-c received, shutting down");
    cancel.cancel();

    // `tokio::join!` waits for ALL futures to complete (unlike `select!` which
    // returns on the FIRST). The `let _ =` discards the Results since we're
    // shutting down anyway — errors at this point aren't actionable.
    let _ = tokio::join!(grpc_handle, http_handle);
    info!("shutdown complete");
}
