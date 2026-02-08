mod support;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use hub::audio::{AudioBus, SpeechEvent, TtsFrame};
use hub::device::run_device;
use hub::metrics::HubMetrics;
use hub::vad::VadConfig;
use support::{speech_frame, MockTransport};
use webrtc_vad::VadMode;

/// Small thresholds so tests complete quickly.
fn test_vad_config() -> VadConfig {
    VadConfig {
        onset_threshold: 3,
        hangover_threshold: 5,
        mode: VadMode::Quality,
    }
}

/// Build a fresh AudioBus and return both the bus and the receiver for tagged frames.
fn test_bus() -> (Arc<AudioBus>, mpsc::Receiver<hub::audio::TaggedFrame>) {
    let (audio_tx, audio_rx) = mpsc::channel(256);
    let bus = Arc::new(AudioBus {
        audio_tx,
        tts_sub: DashMap::new(),
    });
    (bus, audio_rx)
}

/// Push enough speech frames through a mock transport to pass VAD onset,
/// then verify tagged frames arrive on the audio bus with the right metadata.
#[tokio::test]
async fn frames_reach_bus_after_vad_onset() {
    let (transport, handles) = MockTransport::new("test-device");
    let (bus, mut audio_rx) = test_bus();
    let device_tasks: Arc<DashMap<Arc<str>, CancellationToken>> = Arc::new(DashMap::new());
    let cancel = CancellationToken::new();

    let metrics = Arc::new(HubMetrics::new());
    let task = tokio::spawn(run_device(
        transport,
        Arc::clone(&bus),
        Arc::from("conv-1"),
        test_vad_config(),
        cancel.clone(),
        Arc::clone(&device_tasks),
        Arc::clone(&metrics),
    ));

    // Send 3 speech frames to trigger onset, then a 4th to get a Continue.
    for _ in 0..4 {
        handles.frame_tx.send(speech_frame()).await.unwrap();
    }

    // The onset flush should produce tagged frames on the bus.
    // onset_threshold=3 means the first Emit has 3 frames, plus 1 Continue.
    let mut received = Vec::new();
    for _ in 0..4 {
        let frame = tokio::time::timeout(std::time::Duration::from_secs(2), audio_rx.recv())
            .await
            .expect("timed out waiting for tagged frame")
            .expect("bus channel closed");
        received.push(frame);
    }

    assert_eq!(received[0].device_id.as_ref(), "test-device");
    assert_eq!(received[0].conversation_id.as_ref(), "conv-1");
    assert_eq!(received[0].event, SpeechEvent::Start);

    // Remaining frames should be Continue.
    for frame in &received[1..] {
        assert_eq!(frame.event, SpeechEvent::Continue);
    }

    // Metrics should reflect the activity.
    assert!(metrics.vad_onsets.load(Ordering::Relaxed) >= 1);
    assert!(metrics.frames_forwarded.load(Ordering::Relaxed) >= 4);

    // Clean up.
    cancel.cancel();
    let _ = task.await;
}

/// Send a TTS frame into the bus's tts_sub channel and verify the mock
/// transport receives it on the outbound side.
#[tokio::test]
async fn tts_routed_back_to_transport() {
    let (transport, mut handles) = MockTransport::new("test-device");
    let (bus, _audio_rx) = test_bus();
    let device_tasks: Arc<DashMap<Arc<str>, CancellationToken>> = Arc::new(DashMap::new());
    let cancel = CancellationToken::new();

    let metrics = Arc::new(HubMetrics::new());
    let task = tokio::spawn(run_device(
        transport,
        Arc::clone(&bus),
        Arc::from("conv-1"),
        test_vad_config(),
        cancel.clone(),
        Arc::clone(&device_tasks),
        metrics,
    ));

    // Give the device task a moment to register its tts_sub entry.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Send a TTS frame through the bus.
    let tts = TtsFrame {
        data: Bytes::from(vec![42u8; 100]),
        sample_rate: 48_000,
        is_final: false,
    };

    if let Some(tx) = bus.tts_sub.get("test-device") {
        tx.send(tts).await.unwrap();
    } else {
        panic!("device should have registered in tts_sub");
    }

    // The mock transport should receive it.
    let received = tokio::time::timeout(std::time::Duration::from_secs(2), handles.tts_rx.recv())
        .await
        .expect("timed out waiting for TTS frame")
        .expect("tts channel closed");

    assert_eq!(received.data.len(), 100);
    assert_eq!(received.sample_rate, 48_000);
    assert!(!received.is_final);

    cancel.cancel();
    let _ = task.await;
}

/// Dropping the mock transport's frame sender simulates a device disconnect.
/// run_device should exit and clean up its tts_sub and device_tasks entries.
#[tokio::test]
async fn transport_close_triggers_cleanup() {
    let (transport, handles) = MockTransport::new("test-device");
    let (bus, _audio_rx) = test_bus();
    let device_tasks: Arc<DashMap<Arc<str>, CancellationToken>> = Arc::new(DashMap::new());
    let cancel = CancellationToken::new();

    let metrics = Arc::new(HubMetrics::new());
    let task = tokio::spawn(run_device(
        transport,
        Arc::clone(&bus),
        Arc::from("conv-1"),
        test_vad_config(),
        cancel.clone(),
        Arc::clone(&device_tasks),
        metrics,
    ));

    // Wait for device task to register.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(bus.tts_sub.contains_key("test-device"), "should be registered");

    // Simulate disconnect by dropping the frame sender.
    drop(handles.frame_tx);

    // run_device should exit.
    tokio::time::timeout(std::time::Duration::from_secs(2), task)
        .await
        .expect("run_device should exit after transport closes")
        .expect("task panicked");

    // Cleanup: tts_sub and device_tasks entries should be removed.
    assert!(
        !bus.tts_sub.contains_key("test-device"),
        "tts_sub should be cleaned up after disconnect"
    );
    assert!(
        !device_tasks.contains_key("test-device"),
        "device_tasks should be cleaned up after disconnect"
    );
}

/// Cancelling the device token (simulating reconnect dedup) causes the
/// running device task to exit.
#[tokio::test]
async fn cancel_token_stops_device() {
    let (transport, _handles) = MockTransport::new("test-device");
    let (bus, _audio_rx) = test_bus();
    let device_tasks: Arc<DashMap<Arc<str>, CancellationToken>> = Arc::new(DashMap::new());
    let cancel = CancellationToken::new();

    let metrics = Arc::new(HubMetrics::new());
    let task = tokio::spawn(run_device(
        transport,
        Arc::clone(&bus),
        Arc::from("conv-1"),
        test_vad_config(),
        cancel.clone(),
        Arc::clone(&device_tasks),
        metrics,
    ));

    // Wait for task to start.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Cancel the token (as the signaling handler would on reconnect).
    cancel.cancel();

    // Task should exit promptly.
    tokio::time::timeout(std::time::Duration::from_secs(2), task)
        .await
        .expect("run_device should exit after cancellation")
        .expect("task panicked");
}
