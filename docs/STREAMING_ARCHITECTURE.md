# NAILA Streaming Architecture

## Overview

This document extends the core NAILA architecture (see `ARCHITECTURE.md`) with real-time streaming capabilities for low-latency voice conversations, continuous vision monitoring, and cross-room context awareness.

**This document adds:**
- WebRTC for real-time audio/video streaming
- gRPC for efficient Command Center ↔ AI server communication
- Streaming pipeline optimizations

**Existing infrastructure (unchanged):**
- MQTT as central message broker
- Command topics for device control
- HTTP Server for historical data
- Web Services for monitoring

See also: `MQTT_PROTOCOL.md` for topic specifications.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEVICE LAYER                                   │
│                                                                             │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐  │
│   │  Room A   │  │  Room B   │  │  Room C   │  │  Mobile   │  │  Browser │  │
│   │   Pi      │  │  ESP32    │  │   Pi      │  │   Phone   │  │   Web    │  │
│   │  Mic+Cam  │  │  Mic+Cam  │  │  Mic+Cam  │  │   App     │  │  Client  │  │
│   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────┬─────┘  │
│         │              │              │              │              │       │
│     WebRTC          MQTT          WebRTC         WebRTC         WebRTC      │
│         │              │              │              │              │       │
│         └──────────────┴──────────────┼──────────────┴──────────────┘       │
└───────────────────────────────────────┼─────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│                           COMMAND CENTER (Rust)                             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  WebRTC Server  │  │  Media Router   │  │  Filtering                  │  │
│  │                 │──│                 │──│  • VAD (voice detection)    │  │
│  │  • Audio tracks │  │  • Per-device   │  │  • Motion detection         │  │
│  │  • Video tracks │  │  • Per-room     │  │  • Silence suppression      │  │
│  └─────────────────┘  └─────────────────┘  └──────────────┬──────────────┘  │
│                                                           │                 │
│                                              Only relevant data passes      │
│                                                           │                 │
│  ┌────────────────────────────────────────────────────────▼──────────────┐  │
│  │                         gRPC Client                                   │  │
│  │                    (bidirectional streaming)                          │  │
│  └────────────────────────────────────────────────────────┬──────────────┘  │
└───────────────────────────────────────────────────────────┼─────────────────┘
                                                            │
                                               gRPC (protobuf, HTTP/2)
                                                            │
┌───────────────────────────────────────────────────────────▼─────────────────┐
│                            AI SERVER LAYER (Python)                         │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │  gRPC Service   │◄── Single endpoint for Command Center                  │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      AI Processing Pipeline                         │    │
│  │                                                                     │    │
│  │  ┌───────┐    ┌─────────┐    ┌───────┐    ┌───────┐    ┌───────┐    │    │
│  │  │  STT  │───▶│ Context │───▶│  LLM  │───▶│  TTS │──▶│ Output│   │    │
│  │  └───────┘    └─────────┘    └───────┘    └───────┘    └───────┘    │    │
│  │                    ▲                                                │    │
│  │  ┌───────┐         │                                                │    │
│  │  │Vision │─────────┘                                                │    │
│  │  └───────┘                                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                              MQTT                                   │    │
│  │  • Device commands          • Text chat           • Alerts          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Knowledge & Memory                           │    │
│  │         Context Store (PostgreSQL)    Vector DB (Chroma)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Communication Protocols

| Connection | Protocol | Purpose |
|------------|----------|---------|
| Device ↔ Command Center | WebRTC | Real-time audio/video (Pi, phones, browsers) |
| Device ↔ Command Center | MQTT | Audio streaming (ESP32, simple devices) |
| Device ↔ Command Center | MQTT | WebRTC signaling (SDP, ICE) |
| Command Center ↔ AI Server | gRPC | Filtered audio/video, responses |
| AI Server ↔ Devices | MQTT | Commands, alerts, text chat |

## Supported Device Types

| Device | Audio | Video | Protocol | Notes |
|--------|-------|-------|----------|-------|
| Raspberry Pi | Yes | Yes | WebRTC | Full capability |
| ESP32 | Yes | Yes | MQTT | Audio chunks, JPEG frames |
| Mobile App | Yes | Yes | WebRTC | iOS/Android |
| Web Browser | Yes | Optional | WebRTC | Desktop/mobile browsers |
| Smart Speaker | Yes | No | WebRTC/MQTT | Audio only |

## Data Flows

### Voice Conversation

```
User speaks (Room A)
    │
    ▼
Device captures audio (WebRTC)
    │
    ▼
Command Center receives audio stream
    │
    ▼
Command Center VAD detects speech ──────► Silence discarded
    │
    ▼ (speech detected)
Command Center streams to AI Server (gRPC)
    │
    ▼
AI Server: STT → Context → LLM → TTS
    │
    ▼
AI Server streams TTS audio (gRPC)
    │
    ▼
Command Center routes to Room A speaker
    │
    ▼
User hears response
```

### Context Awareness

```
User says "I need to eat" (Room A)
    │
    ▼
AI Server stores context: {user: A, intent: eat, location: Room A, time: T}
    │
    ▼
User moves to Room B, starts doing something else
    │
    ▼
Camera detects user in Room B
    │
    ▼
AI Server checks context: "User A mentioned eating 10 minutes ago"
    │
    ▼
AI Server triggers reminder via Room B speaker
```

### Text Chat

```
User sends text (web/app)
    │
    ▼
MQTT → AI Server
    │
    ▼
AI Server processes (no Command Center involvement)
    │
    ▼
AI Server responds via MQTT
```

## Component Details

### Command Center (Rust)

The Command Center handles all device connections, media processing, and coordination between devices and the AI server. This document focuses on the streaming capabilities; additional Command Center responsibilities will be added:

- Status tracking & reporting
- Sequencing & coordination
- Device command management
- Translation & dispatching

**Current Responsibilities (Streaming):**
- Manage WebRTC connections (Pi, phones, browsers)
- Manage MQTT audio streams (ESP32, simple devices)
- Perform VAD to filter silence/noise
- Detect motion in video streams
- Route audio/video to correct rooms
- Stream relevant data to AI server via gRPC
- Receive TTS responses and route to correct devices

**Technology:**
- Language: Rust
- WebRTC: webrtc-rs
- gRPC: tonic + prost
- MQTT: rumqttc
- VAD: webrtc-vad

### AI Server (Python)

The AI server focuses exclusively on AI processing.

**Responsibilities:**
- Receive filtered audio/video via gRPC
- Speech-to-text transcription
- Context and memory management
- LLM response generation
- Text-to-speech synthesis
- Vision analysis
- Send commands via MQTT

**Technology:**
- Language: Python
- gRPC: grpcio
- STT: faster-whisper
- LLM: llama-cpp-python
- TTS: OuteTTS/Piper
- Vision: YOLOv8
- Orchestration: LangGraph

### Device Clients

Lightweight clients running on each device, adapted to device capabilities.

**Responsibilities:**
- Capture audio from microphone
- Capture video from camera (if available)
- Stream to Command Center (WebRTC or MQTT depending on device)
- Play TTS audio from Command Center
- Handle signaling via MQTT

**Implementations:**

| Platform | Language | Audio/Video | Transport |
|----------|----------|-------------|-----------|
| Raspberry Pi | Python | aiortc, sounddevice | WebRTC |
| ESP32 | C++ | I2S, camera driver | MQTT |
| Mobile App | Swift/Kotlin | Native WebRTC | WebRTC |
| Web Browser | JavaScript | Browser WebRTC API | WebRTC |

## gRPC Service Definition

```protobuf
syntax = "proto3";
package naila;

service NailaAI {
  // Bidirectional audio streaming for voice conversations
  rpc StreamConversation(stream AudioInput) returns (stream AudioOutput);

  // Video frame analysis
  rpc AnalyzeFrame(FrameInput) returns (FrameAnalysis);

  // Context updates from Command Center
  rpc UpdateContext(ContextEvent) returns (Ack);
}

message AudioInput {
  string device_id = 1;
  string room_id = 2;
  bytes audio_pcm = 3;
  uint32 sample_rate = 4;
  uint64 timestamp_ms = 5;
  SpeechEvent event = 6;
}

enum SpeechEvent {
  CONTINUE = 0;
  START = 1;
  END = 2;
}

message AudioOutput {
  string device_id = 1;
  string room_id = 2;
  bytes audio_pcm = 3;
  uint32 sample_rate = 4;
  bool is_final = 5;
}

message FrameInput {
  string device_id = 1;
  string room_id = 2;
  bytes frame_jpeg = 3;
  uint64 timestamp_ms = 4;
}

message FrameAnalysis {
  repeated string users_detected = 1;
  string scene_description = 2;
}

message ContextEvent {
  string room_id = 1;
  string user_id = 2;
  string event_type = 3;
  string details = 4;
  uint64 timestamp_ms = 5;
}

message Ack {
  bool success = 1;
}
```

## Directory Structure

```
robo-naila/
├── command-center/                # Rust Command Center
│   ├── Cargo.toml
│   ├── build.rs
│   ├── proto/
│   │   └── naila.proto
│   └── src/
│       ├── main.rs
│       ├── config.rs
│       ├── grpc/                  # gRPC client to AI server
│       ├── webrtc/                # WebRTC connections
│       ├── mqtt/                  # MQTT connections (ESP32, signaling)
│       ├── media/                 # VAD, motion, routing
│       └── devices/               # Device registry, room mapping
│
├── ai-server/
│   ├── grpc/                      # gRPC service
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── service.py
│   ├── proto/
│   │   └── naila.proto
│   ├── services/                  # AI services (STT, LLM, TTS, Vision)
│   ├── agents/                    # LangGraph agents
│   └── ...
│
├── clients/                       # Device client implementations
│   ├── pi/                        # Raspberry Pi client (Python)
│   │   ├── main.py
│   │   ├── config.py
│   │   └── requirements.txt
│   ├── web/                       # Browser client (JavaScript)
│   │   └── ...
│   └── mobile/                    # Mobile app (future)
│       └── ...
│
├── firmware/                      # ESP32 firmware (C++)
│   └── ...
│
├── proto/                         # Shared protocol definitions
│   └── naila.proto
│
└── docs/
    ├── ARCHITECTURE.md
    └── STREAMING_ARCHITECTURE.md
```

## Implementation Phases

### Phase 1: gRPC Infrastructure
- Define shared .proto file
- Implement gRPC server in AI server (Python)
- Implement gRPC client in Command Center (Rust)
- Test bidirectional streaming

### Phase 2: Command Center Core
- Set up Rust project structure
- Implement WebRTC server
- Implement VAD filtering
- Connect to AI server via gRPC

### Phase 3: Voice Pipeline
- Stream audio from device → Command Center → AI server
- Process STT → LLM → TTS in AI server
- Stream TTS back: AI server → Command Center → device
- Measure latency (target: <600ms)

### Phase 4: Video/Context
- Add video track handling in Command Center
- Implement motion detection
- Add frame analysis in AI server
- Implement cross-room context tracking

### Phase 5: Multi-Device
- Device registry with room assignments
- Room-based audio/video routing
- Multi-user tracking
- Stress test with 5+ devices

## Performance Targets

| Metric | Target |
|--------|--------|
| Voice response latency | <600ms to first audio |
| Concurrent devices | 5+ without degradation |
| Audio quality | 48kHz, Opus codec |
| Video analysis | 1-5 FPS per camera |
| Context recall | <100ms |

## Migration Path

1. Define shared .proto file in `proto/`
2. Build Rust Command Center with gRPC client
3. Add gRPC service to AI server
4. Test Command Center ↔ AI server streaming
5. Implement Pi client connecting to Command Center
6. Update ESP32 firmware to route through Command Center
7. Remove legacy `gateway/` Python code
