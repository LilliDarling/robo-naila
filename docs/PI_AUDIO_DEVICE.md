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
    │   ├── http.py             # POST /connect SDP exchange
    │   ├── aec.py              # SpeexDSP echo cancellation wrapper
    │   ├── audio.py            # sounddevice full-duplex capture + playback + AEC
    │   ├── webrtc.py           # aiortc peer connection, MicTrack, TTS receive
    │   └── metrics.py          # Counters, gauges, periodic logging
    └── tests/
        ├── __init__.py
        ├── test_aec.py
        ├── test_http.py
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

### `http.py`
- Single async function: `exchange_sdp(hub_url, device_id, offer_sdp) -> answer_sdp`
- POSTs `{"device_id": "...", "sdp": "..."}` to `{hub_url}/connect`
- Returns the SDP answer string
- Ref: `hub/src/http.rs` lines 31-43 for request/response format

### `aec.py`
- `EchoCanceller` wrapping `speexdsp.EchoState`
- `process(mic_frame, reference_frame) -> cleaned_frame` (all int16 numpy arrays, 960 samples)
- Filter tail: 100ms (4800 samples) — covers desk robot acoustic path
- Reference signal = what the speaker just played, or silence when no TTS active

### `audio.py`
- `AudioPipeline` class managing `sounddevice.Stream` in full-duplex mode
- PortAudio callback (runs on audio thread):
  1. Dequeue speaker samples (or silence) → write to `outdata`
  2. Read `indata` from mic
  3. `aec.process(mic, speaker_samples)` → push cleaned to mic_queue
- Thread-safe `queue.Queue` for mic_queue and speaker_queue (PortAudio thread ↔ asyncio thread)
- `read_mic_frame()`: async bridge via `loop.run_in_executor`
- `queue_playback(samples)`: non-blocking put

### `webrtc.py`
- `MicTrack(MediaStreamTrack)`: subclass that reads from `AudioPipeline.read_mic_frame()`, converts to `av.AudioFrame` (s16, mono, 48kHz), returns from `recv()`
- `WebRTCClient`:
  - Creates `RTCPeerConnection` with no ICE servers (local network)
  - Adds `MicTrack` as outbound track
  - Handles inbound `"track"` event → decodes TTS frames → `pipeline.queue_playback()`
  - Creates SDP offer, waits for ICE gathering, calls `http.exchange_sdp()`, applies answer
  - Hub expects Opus at 48kHz mono, 20ms frames — aiortc handles this automatically via SDP negotiation

### `metrics.py`
- `DeviceMetrics` class with atomic counters and gauges, thread-safe (accessed from both audio and asyncio threads)
- Counters (monotonically increasing):
  - `mic_frames_captured` — frames read from mic
  - `mic_frames_sent` — frames delivered to MicTrack.recv()
  - `mic_frames_dropped` — mic_queue overflow (queue full, frame discarded)
  - `tts_frames_received` — frames decoded from inbound RTP
  - `tts_frames_played` — frames written to speaker
  - `tts_frames_dropped` — speaker_queue overflow
  - `aec_frames_processed` — frames through echo canceller
  - `connections` — total WebRTC connections established
  - `connection_failures` — failed connection attempts
- Gauges (point-in-time):
  - `mic_queue_depth` — current mic_queue size
  - `speaker_queue_depth` — current speaker_queue size
  - `connected` — bool, WebRTC connection is active
  - `audio_callback_duration_ms` — last PortAudio callback execution time (detects overruns)
  - `uptime_seconds` — time since process start
- Health endpoint:
  - `GET :8081/health` — aiohttp server started in `main.py` alongside the event loop
  - Returns 200 when connected to hub, 503 when disconnected
  - Response body is the full metrics snapshot as JSON:
    ```json
    {
      "connected": true,
      "uptime_seconds": 3421,
      "mic_frames_sent": 124000,
      "mic_frames_dropped": 0,
      "tts_frames_received": 82000,
      "tts_frames_played": 82000,
      "mic_queue_depth": 1,
      "speaker_queue_depth": 3,
      "audio_callback_duration_ms": 0.4,
      "connections": 3,
      "connection_failures": 2
    }
    ```
  - Same JSON body regardless of status code — 503 just signals "not ready" to external health checks
  - Port 8081 hardcoded default, no config needed
- Periodic log dump: every 10s, log all counters/gauges at INFO level as a single structured line
  - Format: `metrics | mic_sent=12400 mic_dropped=0 tts_received=8200 tts_played=8200 mic_q=1 spk_q=3 cb_ms=0.4 connected=true`
  - Reset delta counters each interval so the log shows rates, keep absolute counters on the object for total lifetime stats
- Integration points:
  - `audio.py` callback: increment mic/tts/aec counters, measure callback duration
  - `webrtc.py`: increment connection/frame counters, update connected gauge
  - `main.py`: start health server and periodic log task on startup, stop on shutdown

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
2. `http.py` — HTTP SDP exchange
3. `aec.py` — echo cancellation wrapper
4. `audio.py` — full-duplex audio with AEC
5. `webrtc.py` — aiortc connection + tracks
6. `metrics.py` — counters, gauges, periodic logging
7. `main.py` — reconnect loop + shutdown, wire metrics

Each step is independently testable.

## Verification

1. Start hub locally
2. Run `python -m pi_audio --hub-url http://localhost:8080 --device-id pi-test`
3. Check hub `/health` endpoint shows `active_devices: 1`
4. Speak into mic → verify hub logs show `frames_forwarded` incrementing
5. Check device logs for periodic metrics lines with non-zero `mic_sent` and zero `mic_dropped`
6. If AI server is running: verify full loop (speak → STT → response → TTS → speaker), check `tts_received` / `tts_played` counts match
7. Kill hub → verify device logs reconnection attempts and `connection_failures` incrementing → restart hub → verify reconnects
