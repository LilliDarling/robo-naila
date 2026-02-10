# Pi Audio Device Client (Python)

## Context

The hub expects a WebRTC client that sends Opus audio and receives TTS Opus audio back. No device code exists yet. This plan creates a Python device client at `devices/pi-audio/` that captures mic audio, runs AEC, sends to hub via WebRTC, and plays TTS responses through the speaker. The `devices/` directory is structured to hold multiple device types in the future.

## Directory Structure

```
devices/
└── pi-audio/
    ├── pyproject.toml
    ├── pi_audio/
    │   ├── __init__.py
    │   ├── main.py            # Entry point, reconnect loop, signal handling
    │   ├── config.py           # DeviceConfig dataclass from env/args
    │   ├── signaling.py        # POST /connect SDP exchange
    │   ├── aec.py              # SpeexDSP echo cancellation wrapper
    │   ├── audio_io.py         # sounddevice full-duplex capture + playback + AEC
    │   └── webrtc_client.py    # aiortc peer connection, MicTrack, TTS receive
    └── tests/
        ├── __init__.py
        ├── test_aec.py
        ├── test_signaling.py
        └── test_audio_pipeline.py
```

## Dependencies

| Package | Role |
|---------|------|
| `aiortc` | WebRTC: SDP, RTP, Opus encode/decode |
| `aiohttp` | HTTP client for `POST /connect` signaling |
| `sounddevice` | PortAudio wrapper: mic capture + speaker playback |
| `numpy` | Audio buffer manipulation (int16 arrays) |
| `speexdsp` | Acoustic echo cancellation |

Pi system deps: `libportaudio2 libspeexdsp-dev libopus0 libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev`

## Audio Pipeline

```
INBOUND (mic → hub):
  Mic → [sounddevice callback] → AEC.process(mic, reference) → mic_queue → MicTrack.recv() → aiortc Opus → RTP → Hub

OUTBOUND (hub → speaker):
  Hub → RTP → aiortc Opus decode → speaker_queue → [sounddevice callback] → Speaker
                                                                            ↓
                                                            played samples become AEC reference
```

The `sounddevice.Stream` full-duplex callback handles both directions in one call, giving time-aligned mic capture and speaker output — exactly what AEC needs.

## Module Details

### `config.py`
- `DeviceConfig` dataclass with: `hub_url`, `device_id`, `input_device`, `output_device`, `sample_rate` (48000), `frame_duration_ms` (20), `channels` (1), `reconnect_delay`, `max_reconnect_delay`, `log_level`
- Loaded from env vars (`NAILA_HUB_URL`, `NAILA_DEVICE_ID`, etc.) with argparse override

### `signaling.py`
- Single async function: `exchange_sdp(hub_url, device_id, offer_sdp) -> answer_sdp`
- POSTs `{"device_id": "...", "sdp": "..."}` to `{hub_url}/connect`
- Returns the SDP answer string
- Ref: `hub/src/http.rs` lines 31-43 for request/response format

### `aec.py`
- `EchoCanceller` wrapping `speexdsp.EchoState`
- `process(mic_frame, reference_frame) -> cleaned_frame` (all int16 numpy arrays, 960 samples)
- Filter tail: 100ms (4800 samples) — covers desk robot acoustic path
- Reference signal = what the speaker just played, or silence when no TTS active

### `audio_io.py`
- `AudioPipeline` class managing `sounddevice.Stream` in full-duplex mode
- PortAudio callback (runs on audio thread):
  1. Dequeue speaker samples (or silence) → write to `outdata`
  2. Read `indata` from mic
  3. `aec.process(mic, speaker_samples)` → push cleaned to mic_queue
- Thread-safe `queue.Queue` for mic_queue and speaker_queue (PortAudio thread ↔ asyncio thread)
- `read_mic_frame()`: async bridge via `loop.run_in_executor`
- `queue_playback(samples)`: non-blocking put

### `webrtc_client.py`
- `MicTrack(MediaStreamTrack)`: subclass that reads from `AudioPipeline.read_mic_frame()`, converts to `av.AudioFrame` (s16, mono, 48kHz), returns from `recv()`
- `WebRTCClient`:
  - Creates `RTCPeerConnection` with no ICE servers (local network)
  - Adds `MicTrack` as outbound track
  - Handles inbound `"track"` event → decodes TTS frames → `pipeline.queue_playback()`
  - Creates SDP offer, waits for ICE gathering, calls `signaling.exchange_sdp()`, applies answer
  - Hub expects Opus at 48kHz mono, 20ms frames — aiortc handles this automatically via SDP negotiation

### `main.py`
- `run(config)`: reconnect loop with exponential backoff (3s initial, 2x, 30s cap)
- Starts `AudioPipeline`, creates `WebRTCClient`, connects, waits for disconnect, reconnects
- `SIGINT`/`SIGTERM` → close WebRTC, stop audio, exit
- Two threads total: asyncio event loop + PortAudio audio thread

## Hub Compatibility

Must match these constants from `hub/src/webrtc.rs`:
- `OPUS_SAMPLE_RATE = 48000`
- `OPUS_FRAME_MS = 20`
- `SAMPLES_PER_FRAME = 960`
- RTP payload type 111 (Opus) — negotiated automatically via SDP
- No STUN/TURN — `ice_servers: []`
- Hub creates outbound track (track_id: "audio", stream_id: "tts-stream")
- Hub waits for ICE gathering to complete before returning SDP answer

## Implementation Order

1. `config.py` — parse config
2. `signaling.py` — HTTP SDP exchange
3. `aec.py` — echo cancellation wrapper
4. `audio_io.py` — full-duplex audio with AEC
5. `webrtc_client.py` — aiortc connection + tracks
6. `main.py` — reconnect loop + shutdown

Each step is independently testable.

## Verification

1. Start hub locally
2. Run `python -m pi_audio --hub-url http://localhost:8080 --device-id pi-test`
3. Check hub `/health` endpoint shows `active_devices: 1`
4. Speak into mic → verify hub logs show `frames_forwarded` incrementing
5. If AI server is running: verify full loop (speak → STT → response → TTS → speaker)
6. Kill hub → verify device logs reconnection attempts → restart hub → verify reconnects
