use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::audio::{AudioBus, AudioTransport, SpeechEvent, TaggedFrame, TtsFrame};
use crate::vad::{VadConfig, VadFilter, VadResult};


/// Transport-level relay states. The hub tracks what's flowing in which
/// direction — it doesn't interpret content or make conversation-level decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelayState {
    /// No speech detected. Inbound audio is fed to VAD but suppressed.
    Idle,
    /// Confirmed speech. Inbound frames are forwarded to the audio bus.
    Streaming,
    /// TTS response is being played to the device speaker.
    /// Inbound audio is monitored for barge-in.
    Playing,
}

/// Per-device relay loop. Reads audio from the transport, applies VAD gating,
/// forwards tagged frames to the AudioBus, routes TTS responses back, and
/// handles barge-in detection.
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
) {
    let device_id: Arc<str> = transport.id().into();

    // Subscribe to TTS responses for this device.
    let (tts_tx, mut tts_rx) = mpsc::channel::<TtsFrame>(32);
    bus.tts_sub.insert(Arc::clone(&device_id), tts_tx);

    let mut vad = VadFilter::new(vad_config);
    let mut state = RelayState::Idle;

    info!(device_id = %device_id, "device task started");

    'outer: loop {
        tokio::select! {
            // Inbound: mic audio from device
            frame = transport.recv() => {
                let audio_frame = match frame {
                    Some(f) => f,
                    None => {
                        info!(device_id = %device_id, "transport stream ended");
                        break 'outer;
                    }
                };

                let vad_result = vad.process(audio_frame);

                match state {
                    RelayState::Idle => {
                        if let VadResult::Emit(SpeechEvent::Start, frames) = vad_result {
                            state = RelayState::Streaming;
                            if !send_frames(&bus, &device_id, &conversation_id, frames, SpeechEvent::Start).await {
                                break 'outer;
                            }
                        }
                    }

                    RelayState::Streaming => {
                        match vad_result {
                            VadResult::Emit(event, frames) => {
                                if event == SpeechEvent::End {
                                    state = RelayState::Idle;
                                }
                                if !send_frames(&bus, &device_id, &conversation_id, frames, event).await {
                                    break 'outer;
                                }
                            }
                            VadResult::Suppress => {}
                        }
                    }

                    RelayState::Playing => {
                        // Monitor for barge-in: user speaks during TTS playback.
                        if let VadResult::Emit(SpeechEvent::Start, frames) = vad_result {
                            // Send interrupt signal to AI server.
                            let interrupt = TaggedFrame {
                                device_id: Arc::clone(&device_id),
                                conversation_id: Arc::clone(&conversation_id),
                                frame: crate::audio::AudioFrame {
                                    data: bytes::Bytes::new(),
                                    sample_rate: 0,
                                    timestamp: 0,
                                },
                                event: SpeechEvent::Interrupt,
                            };
                            if bus.audio_tx.send(interrupt).await.is_err() {
                                error!(device_id = %device_id, "audio bus closed");
                                break 'outer;
                            }

                            // Drain buffered TTS frames.
                            while tts_rx.try_recv().is_ok() {}

                            // Forward the onset frames as the new utterance.
                            if !send_frames(&bus, &device_id, &conversation_id, frames, SpeechEvent::Start).await {
                                break 'outer;
                            }

                            state = RelayState::Streaming;
                        }
                    }
                }
            }

            // Outbound: TTS response → device speaker
            tts = tts_rx.recv() => {
                match tts {
                    Some(tts_frame) => {
                        let is_final = tts_frame.is_final;

                        // Transition to Playing if we weren't already.
                        if state != RelayState::Playing {
                            state = RelayState::Playing;
                        }

                        if let Err(e) = transport.send(tts_frame).await {
                            error!(device_id = %device_id, "send TTS failed: {e:?}");
                            break 'outer;
                        }

                        if is_final {
                            state = RelayState::Idle;
                        }
                    }
                    None => {
                        info!(device_id = %device_id, "TTS channel closed");
                        break 'outer;
                    }
                }
            }

            // Peer disconnect or reconnect dedup — exit promptly.
            _ = device_cancel.cancelled() => {
                info!(device_id = %device_id, "device cancelled");
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
    info!(device_id = %device_id, "device session cleaned up");
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
        let event = if i == 0 { first_event } else { SpeechEvent::Continue };
        let tagged = TaggedFrame {
            device_id: Arc::clone(device_id),
            conversation_id: Arc::clone(conversation_id),
            frame: f,
            event,
        };
        if bus.audio_tx.send(tagged).await.is_err() {
            error!(device_id = %device_id, "audio bus closed");
            return false;
        }
    }
    true
}
