# Hub

The hub is a Rust service that sits between the robot devices (Raspberry Pis) and the AI server. It accepts WebRTC connections from devices, applies voice activity detection to filter speech from silence, forwards speech audio to the AI server over gRPC, and routes synthesized TTS audio back to the originating device for playback. It handles multiple concurrent devices, reconnect deduplication, and graceful shutdown.

## Getting Started

Requires a working Rust toolchain (edition 2024). The proto definitions are compiled at build time via `tonic-build`, so `protoc` must be on your `PATH`.

```bash
# from the repo root
cd hub
cargo run
```

The hub starts two subsystems:

- **HTTP signaling server** on `0.0.0.0:8080`
- **gRPC client** connecting to `http://127.0.0.1:50051` (the AI server)

Both addresses are currently hardcoded defaults. The gRPC client reconnects automatically with exponential backoff (1s initial, 30s max) if the AI server is unavailable.

## HTTP Endpoints

### `POST /connect`

WebRTC signaling endpoint. A device sends its SDP offer and receives an SDP answer with ICE candidates baked in.

**Request:**

```json
{
  "device_id": "pi-kitchen",
  "sdp": "<SDP offer>"
}
```

**Response (200):**

```json
{
  "sdp": "<SDP answer>"
}
```

On success the hub creates a peer connection, spawns a device task, and begins relaying audio. If a device reconnects with the same `device_id`, the previous session is cancelled before the new one starts.

### `GET /health`

Returns a JSON snapshot of hub state and counters.

**Response:**

```json
{
  "status": "ok",
  "uptime_secs": 3600,
  "active_devices": 2,
  "grpc_connected": true,
  "frames_forwarded": 54000,
  "vad_onsets": 12,
  "vad_ends": 11,
  "tts_frames_routed": 8400,
  "grpc_reconnects": 0
}
```

## Metrics

All metrics are atomic counters/gauges exposed through `GET /health`.

| Field | Type | Description |
|---|---|---|
| `uptime_secs` | gauge | Process uptime in seconds |
| `active_devices` | gauge | Number of currently connected devices |
| `grpc_connected` | gauge | Whether the gRPC stream to the AI server is active |
| `frames_forwarded` | counter | Audio frames forwarded from devices to the gRPC bus |
| `vad_onsets` | counter | VAD speech-onset events (silence to speech transitions) |
| `vad_ends` | counter | VAD speech-end events (speech to silence transitions) |
| `tts_frames_routed` | counter | TTS audio frames routed from the AI server back to devices |
| `grpc_reconnects` | counter | gRPC reconnection attempts after failures |

## gRPC

The hub acts as a **client** to the AI server's `NailaAI` gRPC service, defined in `proto/nailaV1.proto`.

### `StreamConversation` (bidirectional streaming)

```
rpc StreamConversation(stream AudioInput) returns (stream AudioOutput);
```

A single long-lived bidirectional stream carries audio for all devices. The hub multiplexes frames from all connected devices onto one outbound stream and demultiplexes responses back to the correct device by `device_id`.

**`AudioInput`** (hub to AI server):

| Field | Description |
|---|---|
| `device_id` | Originating device identifier |
| `conversation_id` | Session identifier (monotonic counter) |
| `audio_pcm` / `audio_opus` | Audio payload (oneof) |
| `codec` | `PCM_S16LE` or `OPUS` |
| `sample_rate` | Sample rate in Hz (16000 or 48000) |
| `chunk_duration_ms` | Frame duration (20ms) |
| `timestamp_ms` | Capture timestamp |
| `sequence_num` | Monotonic frame counter |
| `event` | `START`, `CONTINUE`, `END`, or `INTERRUPT` |

**`AudioOutput`** (AI server to hub):

| Field | Description |
|---|---|
| `device_id` | Target device identifier |
| `audio_pcm` | Synthesized TTS audio (PCM s16le) |
| `sample_rate` | Output sample rate |
| `sequence_num` | Frame ordering |
| `is_final` | True on the last chunk of a response |
| `final_stt` | STT transcription of the user's utterance |
| `error_code` | `NONE`, `STT_FAILED`, `LLM_FAILED`, `TTS_FAILED`, or `INTERNAL` |
| `error_message` | Human-readable error description |

## WebRTC Server

The hub uses the `webrtc` crate (pure Rust) with Opus as the sole audio codec. There are no STUN/TURN servers configured — the hub assumes it runs on the same local network as the devices.

### Audio pipeline

```
Device mic → [Opus/RTP] → Hub inbound track → Opus decode → PCM 48kHz
  → VAD filter → gRPC AudioInput → AI server
  → gRPC AudioOutput → PCM → Opus encode → [RTP] → Hub outbound track
  → Device speaker
```

### Configuration

| Parameter | Value |
|---|---|
| Codec | Opus (mono) |
| Sample rate | 48 kHz |
| Frame duration | 20 ms |
| Samples per frame | 960 |
| ICE servers | none (local network) |
| Outbound track ID | `audio` |
| Outbound stream ID | `tts-stream` |

### VAD

Voice activity detection uses `webrtc-vad` (libfvad) with a four-state machine: `Silence` → `MaybeSpeech` → `Speaking` → `MaybeSilence`. Defaults:

- **Onset threshold:** 3 consecutive speech frames (60ms) before confirming speech
- **Hangover threshold:** 15 consecutive silence frames (300ms) before ending an utterance
- **Mode:** Quality (lowest false-positive rate)

Frames are suppressed during silence. On confirmed onset, buffered frames are flushed with a `Start` event. During speech, frames are forwarded with `Continue`. After hangover expires, a final `End` event is emitted.
