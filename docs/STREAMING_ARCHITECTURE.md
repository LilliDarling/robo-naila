# Streaming Architecture

Real-time voice streaming for NAILA, enabling low-latency conversations between devices and the AI server.

## System Overview

```
┌──────────────┐        WebRTC (Opus)        ┌──────────────┐       gRPC (protobuf)      ┌──────────────┐
│    Device    │ ◄══════════════════════════► │     Hub      │ ◄══════════════════════════► │  AI Server   │
│  (Pi Audio)  │                              │    (Rust)    │                              │   (Python)   │
│              │    HTTP signaling (SDP)       │              │                              │              │
│  • Mic       │ ─────────────────────────── ►│  • WebRTC    │    StreamConversation RPC    │  • STT       │
│  • Speaker   │                              │  • VAD       │    (bidirectional stream)    │  • LLM       │
│  • AEC       │                              │  • Opus      │                              │  • TTS       │
│              │                              │  • Metrics   │                              │  • Vision    │
└──────────────┘                              └──────────────┘                              └──────────────┘
```

### Components

| Component | Language | Role |
|-----------|----------|------|
| **Pi Audio** (`devices/pi-audio/`) | Python | Captures mic audio, plays TTS, runs echo cancellation |
| **Hub** (`hub/`) | Rust | WebRTC server, VAD filtering, gRPC relay to AI server |
| **AI Server** (`ai-server/`) | Python | STT → LLM → TTS pipeline, MQTT for commands/alerts |

### Protocols

| Connection | Protocol | Purpose |
|------------|----------|---------|
| Device ↔ Hub | WebRTC (Opus, UDP) | Real-time bidirectional audio |
| Device → Hub | HTTP POST `/connect` | WebRTC signaling (SDP exchange) |
| Hub ↔ AI Server | gRPC (HTTP/2) | Filtered audio streaming, TTS responses |
| AI Server ↔ Devices | MQTT | Commands, alerts, text chat, device status |

## Voice Conversation Flow

```
User speaks
    │
    ▼
Pi Audio captures mic (48kHz, Opus, 20ms frames)
    │
    ▼
AEC removes speaker echo (SpeexDSP)
    │
    ▼
WebRTC stream → Hub receives Opus/RTP
    │
    ▼
Hub decodes Opus → PCM 48kHz
    │
    ▼
VAD filters silence (webrtc-vad, libfvad)
    │ only speech passes
    ▼
Hub sends gRPC AudioInput → AI Server
    │
    ▼
AI Server: STT → LLM → TTS
    │
    ▼
AI Server streams gRPC AudioOutput (PCM)
    │
    ▼
Hub encodes PCM → Opus → RTP
    │
    ▼
WebRTC stream → Pi Audio speaker
```

## gRPC Protocol

Defined in `proto/nailaV1.proto` (full spec) and `proto/naila.proto` (minimal working version).

### `StreamConversation` (bidirectional streaming)

```protobuf
service NailaAI {
  rpc StreamConversation(stream AudioInput) returns (stream AudioOutput);
  rpc GetStatus(StatusRequest) returns (StatusResponse);
}
```

A single long-lived stream carries audio for all devices. The hub multiplexes frames from all connected devices onto one outbound stream and demultiplexes responses back to the correct device by `device_id`.

### AudioInput (Hub → AI Server)

| Field | Description |
|-------|-------------|
| `device_id` | Originating device identifier |
| `conversation_id` | Session identifier |
| `audio_pcm` / `audio_opus` | Audio payload (oneof) |
| `codec` | `PCM_S16LE` or `OPUS` |
| `sample_rate` | Sample rate in Hz (16000 or 48000) |
| `chunk_duration_ms` | Frame duration (20ms) |
| `timestamp_ms` | Capture timestamp |
| `sequence_num` | Monotonic frame counter |
| `event` | `START`, `CONTINUE`, `END`, or `INTERRUPT` |

### AudioOutput (AI Server → Hub)

| Field | Description |
|-------|-------------|
| `device_id` | Target device for playback |
| `audio_pcm` | Synthesized TTS audio (PCM s16le) |
| `sample_rate` | Output sample rate |
| `sequence_num` | Frame ordering |
| `is_final` | True on last chunk of response |
| `final_stt` | STT transcription of user utterance |
| `error_code` | `NONE`, `STT_FAILED`, `LLM_FAILED`, `TTS_FAILED`, `INTERNAL` |

### Speech Events (VAD State Machine)

```
Silence → MaybeSpeech → Speaking → MaybeSilence → Silence
```

- **Onset:** 3 consecutive speech frames (60ms) before confirming speech → `START`
- **During speech:** frames forwarded with `CONTINUE`
- **Hangover:** 15 consecutive silence frames (300ms) before ending → `END`
- **Barge-in:** user speaks during TTS playback → `INTERRUPT`

## Audio Configuration

| Parameter | Value |
|-----------|-------|
| Codec | Opus (mono) |
| Sample rate | 48 kHz |
| Frame duration | 20 ms |
| Samples per frame | 960 |
| ICE servers | none (local network) |
| Echo cancellation | SpeexDSP (device-side) |
| VAD mode | Quality (lowest false-positive rate) |

## Hub Endpoints

### `POST /connect` — WebRTC signaling

```json
// Request
{ "device_id": "pi-kitchen", "sdp": "<SDP offer>" }

// Response (200)
{ "sdp": "<SDP answer>" }
```

Reconnecting with the same `device_id` cancels the previous session.

### `GET /health` — Hub metrics

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

## Latency Budget

| Stage | Target |
|-------|--------|
| Audio capture + AEC | ~20ms |
| Network (WebRTC, local WiFi) | ~20-50ms |
| Opus decode | ~1ms |
| VAD | ~5ms |
| STT (streaming Whisper) | ~150-200ms |
| LLM inference | ~100-300ms |
| TTS (first chunk) | ~50-100ms |
| Opus encode + network back | ~20-50ms |
| **Total to first audio** | **~400-700ms** |

## Future Work

These items are specced in the proto file and streaming docs but not yet implemented:

- **Video pipeline** — WebRTC video tracks, motion detection, YOLO frame analysis
- **Multi-device routing** — room-based audio routing, device registry
- **MQTT signaling** — replace HTTP signaling with MQTT-based SDP exchange
- **Cross-room context** — user tracking across rooms, context-aware responses
- **Media input flexibility** — MQTT snapshots, web/app uploads, historical media retrieval
- **`GetStatus` RPC** — server health and capability discovery (defined in `nailaV1.proto`)

See `ai-server/docs/REALTIME_STREAMING_IMPLEMENTATION.md` for the full specification of planned features.
