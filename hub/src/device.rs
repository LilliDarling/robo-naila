use std::sync::atomic::Ordering;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, info_span, Instrument};

use crate::audio::{AudioBus, AudioTransport, SpeechEvent, TaggedFrame, TtsFrame};
use crate::metrics::HubMetrics;
use crate::vad::{VadConfig, VadFilter, VadResult};

/// Transport-level relay states. The hub tracks what's flowing in which
/// direction — it doesn't interpret content or make conversation-level decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelayState {
    /// No speech detected. Inbound audio is fed to VAD but suppressed.
    Idle,
    /// Confirmed speech. Inbound frames are forwarded to the audio bus.
    Streaming,
}

/// Per-device relay loop. Reads audio from the transport, applies VAD gating,
/// forwards tagged frames to the AudioBus, and routes TTS responses back.
///
/// Exits when the transport closes, the cancel token fires (peer disconnect
/// or reconnect dedup), or the audio bus closes.
pub async fn run_device<T: AudioTransport>(
    mut transport: T,
    bus: Arc<AudioBus>,
    conversation_id: Arc<str>,
    vad_config: VadConfig,
    device_cancel: CancellationToken,
    device_tasks: Arc<DashMap<Arc<str>, CancellationToken>>,
    metrics: Arc<HubMetrics>,
) {
    let device_id: Arc<str> = transport.id().into();

    let span = info_span!("device", %device_id, %conversation_id);

    async {
        // Subscribe to TTS responses for this device.
        let (tts_tx, mut tts_rx) = mpsc::channel::<TtsFrame>(32);
        bus.tts_sub.insert(Arc::clone(&device_id), tts_tx);

        let mut vad = VadFilter::new(vad_config);
        let mut state = RelayState::Idle;

        info!("device task started");

        'outer: loop {
            tokio::select! {
                // Inbound: mic audio from device
                frame = transport.recv() => {
                    let audio_frame = match frame {
                        Some(f) => f,
                        None => {
                            info!("transport stream ended");
                            break 'outer;
                        }
                    };

                    let vad_result = vad.process(audio_frame);

                    match state {
                        RelayState::Idle => {
                            if let VadResult::Emit(SpeechEvent::Start, frames) = vad_result {
                                state = RelayState::Streaming;
                                metrics.vad_onsets.fetch_add(1, Ordering::Relaxed);
                                let count = frames.len() as u64;
                                if !send_frames(&bus, &device_id, &conversation_id, frames, SpeechEvent::Start).await {
                                    break 'outer;
                                }
                                metrics.frames_forwarded.fetch_add(count, Ordering::Relaxed);
                            }
                        }

                        RelayState::Streaming => {
                            match vad_result {
                                VadResult::Emit(event, frames) => {
                                    if event == SpeechEvent::End {
                                        state = RelayState::Idle;
                                        metrics.vad_ends.fetch_add(1, Ordering::Relaxed);
                                    }
                                    let count = frames.len() as u64;
                                    if !send_frames(&bus, &device_id, &conversation_id, frames, event).await {
                                        break 'outer;
                                    }
                                    metrics.frames_forwarded.fetch_add(count, Ordering::Relaxed);
                                }
                                VadResult::Suppress => {}
                            }
                        }
                    }
                }

                // Outbound: TTS response → device speaker
                tts = tts_rx.recv() => {
                    match tts {
                        Some(tts_frame) => {
                            if let Err(e) = transport.send(tts_frame).await {
                                error!("send TTS failed: {e:?}");
                                break 'outer;
                            }
                        }
                        None => {
                            info!("TTS channel closed");
                            break 'outer;
                        }
                    }
                }

                // Peer disconnect or reconnect dedup — exit promptly.
                _ = device_cancel.cancelled() => {
                    info!("device cancelled");
                    break 'outer;
                }
            }
        }

        // Notify AI server that this device session is over.
        let _ = bus
            .audio_tx
            .send(TaggedFrame {
                device_id: Arc::clone(&device_id),
                conversation_id: Arc::clone(&conversation_id),
                frame: crate::audio::AudioFrame {
                    data: bytes::Bytes::new(),
                    sample_rate: 0,
                    timestamp: 0,
                },
                event: SpeechEvent::End,
            })
            .await;

        // Mark our token as cancelled so cleanup can identify our entries.
        // Idempotent — no-op if already cancelled by reconnect dedup or peer failure.
        // Without this, normal exits (transport closed) leave uncancelled entries.
        device_cancel.cancel();

        // Cleanup: only remove entries that belong to THIS session.
        // If a reconnect replaced us, device_tasks will have a new non-cancelled
        // token — leave tts_sub alone so the new session keeps receiving TTS.
        if device_tasks
            .get(device_id.as_ref())
            .map_or(true, |entry| entry.value().is_cancelled())
        {
            bus.tts_sub.remove(device_id.as_ref());
        }
        device_tasks.remove_if(device_id.as_ref(), |_, token| token.is_cancelled());
        info!("device session cleaned up");
    }
    .instrument(span)
    .await
}

/// Send tagged frames to the bus. Returns false if the bus is closed.
async fn send_frames(
    bus: &AudioBus,
    device_id: &Arc<str>,
    conversation_id: &Arc<str>,
    frames: Vec<crate::audio::AudioFrame>,
    first_event: SpeechEvent,
) -> bool {
    for (i, f) in frames.into_iter().enumerate() {
        let event = if i == 0 {
            first_event
        } else {
            SpeechEvent::Continue
        };
        let tagged = TaggedFrame {
            device_id: Arc::clone(device_id),
            conversation_id: Arc::clone(conversation_id),
            frame: f,
            event,
        };
        if bus.audio_tx.send(tagged).await.is_err() {
            error!("audio bus closed");
            return false;
        }
    }
    true
}
