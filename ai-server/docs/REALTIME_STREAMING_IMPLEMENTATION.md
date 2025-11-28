# Real-Time Streaming Implementation Guide

## NAILA AI Server - WebRTC + MQTT Architecture

**Version:** 1.0
**Date:** November 2025
**Status:** Implementation Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Protocol Selection Rationale](#3-protocol-selection-rationale)
4. [WebRTC Media Layer](#4-webrtc-media-layer)
5. [MQTT Control Layer](#5-mqtt-control-layer)
6. [Voice Conversation Pipeline](#6-voice-conversation-pipeline)
7. [Vision Pipeline](#7-vision-pipeline)
8. [Media Input Methods](#8-media-input-methods)
9. [Signaling Server](#9-signaling-server)
10. [Implementation Phases](#10-implementation-phases)
11. [File Structure](#11-file-structure)
12. [Configuration](#12-configuration)
13. [Testing Strategy](#13-testing-strategy)
14. [Performance Considerations](#14-performance-considerations)
15. [Security Considerations](#15-security-considerations)
16. [Migration Guide](#16-migration-guide)

**Note:** This doc is for alignment purposes only and not meant as a final version.

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies a **WebRTC + MQTT hybrid architecture** for the NAILA AI Server, enabling:

1. **Seamless Voice Conversation** - Sub-100ms latency, interruptible voice with built-in echo cancellation
2. **Continuous Security Monitoring** - Real-time video analysis with efficient codec compression
3. **Conversational Visual Understanding** - On-demand image context for natural language queries
4. **Multi-Device Support** - Multiple audio/video sources across WiFi network
5. **Flexible Media Input** - Support for real-time streams, snapshots, uploads, and historical retrieval

### 1.2 Key Goals

| Goal | Current State | Target State |
|------|---------------|--------------|
| Voice response latency | 2-5 seconds | <300ms |
| Conversation interruption | Not supported | Full support (WebRTC handles this) |
| Echo cancellation | None | Built-in (WebRTC AEC) |
| Audio codec | Raw PCM/base64 | Opus (10x compression) |
| Video codec | JPEG snapshots | VP8/H264 (efficient streaming) |
| NAT traversal | Not supported | Automatic (ICE/STUN) |
| Network adaptation | None | Automatic bitrate adjustment |
| Jitter handling | None | Built-in jitter buffer |

### 1.3 Architecture Decision

**Hybrid Protocol Approach:**

| Layer | Protocol | Purpose |
|-------|----------|---------|
| **Media** | WebRTC | Audio/video streaming (voice, cameras) |
| **Control** | MQTT | Commands, events, alerts, device coordination |
| **Signaling** | MQTT | WebRTC session establishment |

**Why this combination:**
- **WebRTC** handles all the hard media problems (codecs, jitter, echo, adaptation)
- **MQTT** provides reliable pub/sub for control plane (already in use)
- **No new infrastructure** - MQTT doubles as signaling transport

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           NAILA AI Server                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                        WebRTC Layer                                 │    │
│   │                                                                     │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │    │
│   │   │ Audio Track │    │ Video Track │    │ Data Channel│             │    │
│   │   │   (Opus)    │    │(VP8/H264)   │    │  (optional) │             │    │
│   │   └──────┬──────┘    └──────┬──────┘    └─────────────┘             │    │
│   │          │                  │                                       │    │
│   │          │    ┌─────────────┴─────────────┐                         │    │
│   │          │    │                           │                         │    │
│   │          ▼    ▼                           ▼                         │    │
│   │   ┌─────────────────┐              ┌─────────────────┐              │    │
│   │   │  Voice Pipeline │              │ Vision Pipeline │              │    │
│   │   │                 │              │                 │              │    │
│   │   │ • VAD (built-in)│              │ • Frame decode  │              │    │
│   │   │ • STT           │              │ • Motion detect │              │    │
│   │   │ • LLM           │              │ • YOLO          │              │    │
│   │   │ • TTS           │              │ • Alerts        │              │    │
│   │   └────────┬────────┘              └────────┬────────┘              │    │
│   │            │                                │                       │    │
│   │            │ Audio response                 │ Alerts                │    │
│   │            ▼                                ▼                       │    │
│   │   ┌─────────────┐                   ┌─────────────┐                 │    │
│   │   │ Audio Track │                   │    MQTT     │                 │    │
│   │   │   (Out)     │                   │  Publish    │                 │    │
│   │   └─────────────┘                   └─────────────┘                 │    │
│   │                                                                     │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                        MQTT Layer                                   │    │
│   │                                                                     │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │    │
│   │   │  Signaling  │    │   Control   │    │   Alerts    │             │    │
│   │   │             │    │             │    │             │             │    │
│   │   │ • SDP offer │    │ • Commands  │    │ • Motion    │             │    │
│   │   │ • SDP answer│    │ • Status    │    │ • Person    │             │    │
│   │   │ • ICE cands │    │ • Config    │    │ • Events    │             │    │
│   │   └─────────────┘    └─────────────┘    └─────────────┘             │    │
│   │                                                                     │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Connection Flow

```
┌──────────┐                    ┌──────────┐                    ┌──────────┐
│  Device  │                    │   MQTT   │                    │    AI    │
│          │                    │  Broker  │                    │  Server  │
└────┬─────┘                    └────┬─────┘                    └────┬─────┘
     │                               │                               │
     │  1. MQTT Connect              │                               │
     │──────────────────────────────>│                               │
     │                               │                               │
     │  2. Subscribe to signaling    │                               │
     │──────────────────────────────>│                               │
     │                               │                               │
     │  3. Publish: "device online"  │                               │
     │──────────────────────────────>│──────────────────────────────>│
     │                               │                               │
     │                               │  4. Server creates PeerConn   │
     │                               │                               │
     │                               │  5. SDP Offer                 │
     │<──────────────────────────────│<──────────────────────────────│
     │                               │                               │
     │  6. SDP Answer                │                               │
     │──────────────────────────────>│──────────────────────────────>│
     │                               │                               │
     │  7. ICE Candidates (both ways)│                               │
     │<─────────────────────────────>│<─────────────────────────────>│
     │                               │                               │
     │  8. WebRTC Connected (P2P or via TURN)                        │
     │<─────────────────────────────────────────────────────────────>│
     │                               │                               │
     │  9. Audio/Video streaming     │                               │
     │<═════════════════════════════════════════════════════════════>│
     │                               │                               │
```

### 2.3 Data Flow Summary

| Data Type | Transport | Direction | Format |
|-----------|-----------|-----------|--------|
| Voice audio | WebRTC | Bidirectional | Opus |
| Video frames | WebRTC | Device → Server | VP8/H264 |
| TTS audio | WebRTC | Server → Device | Opus |
| Signaling | MQTT | Bidirectional | JSON |
| Commands | MQTT | Bidirectional | JSON |
| Alerts | MQTT | Server → Clients | JSON |
| Device status | MQTT | Device → Server | JSON |

---

## 3. Protocol Selection Rationale

### 3.1 Why WebRTC over WebSocket

| Requirement | WebSocket | WebRTC |
|-------------|-----------|--------|
| Audio codec (Opus) | Manual implementation | Built-in |
| Echo cancellation | Manual implementation | Built-in (AEC) |
| Noise suppression | Manual implementation | Built-in |
| Jitter buffer | Manual implementation | Built-in |
| Adaptive bitrate | Manual implementation | Built-in |
| Packet loss handling | TCP retransmit (adds latency) | FEC + concealment |
| NAT traversal | Not supported | ICE/STUN/TURN |
| Latency | ~100-300ms | ~50-150ms |

**Bottom line:** WebSocket requires rebuilding 10+ years of real-time media engineering. WebRTC provides it out of the box.

### 3.2 Why MQTT for Control (Not WebRTC Data Channels)

- **Already deployed** - MQTT broker exists in current architecture
- **Pub/Sub model** - Better fit for alerts, multi-subscriber events
- **Retained messages** - Device status persists across reconnects
- **QoS levels** - Guaranteed delivery for critical commands
- **Simpler debugging** - Standard MQTT tools work

### 3.3 Why MQTT for Signaling (Not HTTP/WebSocket)

- **No additional server** - Reuse existing MQTT broker
- **Bidirectional** - Both parties can initiate
- **Reliable** - QoS 1 ensures delivery
- **Simple** - Just pub/sub to device-specific topics

---

## 4. WebRTC Media Layer

### 4.1 Server-Side WebRTC (aiortc)

**Library:** `aiortc` - Pure Python WebRTC implementation

**Capabilities:**
- Full WebRTC stack (ICE, DTLS, SRTP)
- Audio/video track handling
- Opus and VP8 codec support
- Works with asyncio

**PeerConnection Manager:**

Responsibilities:
- Create and manage RTCPeerConnection per device
- Handle ICE candidate exchange
- Add/remove media tracks
- Monitor connection state
- Reconnect on failure

**Audio Track Handling:**
- Receive Opus-encoded audio from device
- Decode to PCM for STT processing
- Encode TTS output to Opus
- Send back via audio track

**Video Track Handling:**
- Receive VP8/H264 frames from device cameras
- Decode to raw frames for vision processing
- No video sent back to device (audio-only response)

### 4.2 Device-Side WebRTC

**For Linux/Raspberry Pi:** `libwebrtc` or `aiortc`

**For ESP32/Embedded:** `esp-webrtc` or audio-only via `opus` library

**For Mobile/Browser:** Native WebRTC APIs

**Device Responsibilities:**
- Capture audio from microphone
- Capture video from camera (if equipped)
- Establish WebRTC connection via signaling
- Play received audio through speaker
- Handle reconnection

### 4.3 Media Configuration

**Audio:**
- Codec: Opus
- Sample rate: 48kHz (Opus native)
- Channels: Mono
- Bitrate: 24-32 kbps (configurable)
- Frame size: 20ms

**Video:**
- Codec: VP8 (or H264 if hardware encoding available)
- Resolution: 640x480 (configurable)
- Framerate: 5-15 FPS (adaptive)
- Bitrate: 500-1500 kbps (adaptive)

### 4.4 ICE Configuration

**For local WiFi network:**
```
ICE Servers: []  # No STUN/TURN needed on local network
ICE Transport Policy: "all"
```

**For internet deployment (future):**
```
ICE Servers:
  - urls: "stun:stun.l.google.com:19302"
  - urls: "turn:your-turn-server.com"
    username: "user"
    credential: "pass"
```

---

## 5. MQTT Control Layer

### 5.1 Topic Structure

```
naila/
├── devices/
│   └── {device_id}/
│       ├── status          # Device status (retained)
│       ├── command         # Commands to device
│       ├── signaling/
│       │   ├── offer       # SDP offers
│       │   ├── answer      # SDP answers
│       │   └── ice         # ICE candidates
│       ├── media/
│       │   ├── snapshot    # On-demand image capture (device → server)
│       │   ├── recording   # Audio/video clip upload (device → server)
│       │   └── request     # Server requests media (server → device)
│       └── events          # Device events
├── server/
│   ├── status              # Server status (retained)
│   └── broadcast           # Broadcast to all devices
├── uploads/
│   └── {session_id}/
│       ├── image           # Web/app image uploads
│       ├── audio           # Web/app audio uploads
│       └── video           # Web/app video uploads
└── alerts/
    ├── {zone}/
    │   └── {alert_type}    # Zone-specific alerts
    └── all                 # All alerts
```

### 5.2 Message Types

**Device Status (retained):**
```json
{
  "device_id": "living_room_hub",
  "online": true,
  "capabilities": ["audio", "video", "speaker"],
  "firmware_version": "1.2.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Signaling - SDP Offer/Answer:**
```json
{
  "type": "offer",
  "sdp": "v=0\r\no=- 123456 ...",
  "device_id": "living_room_hub",
  "session_id": "abc123"
}
```

**Signaling - ICE Candidate:**
```json
{
  "candidate": "candidate:1 1 UDP 2130706431 ...",
  "sdpMid": "audio",
  "sdpMLineIndex": 0,
  "session_id": "abc123"
}
```

**Command:**
```json
{
  "command": "start_video",
  "params": {
    "resolution": "640x480",
    "fps": 10
  },
  "request_id": "req_123"
}
```

**Alert:**
```json
{
  "alert_type": "person_detected",
  "zone": "front_door",
  "confidence": 0.92,
  "source_device": "front_camera",
  "timestamp": "2024-01-15T10:30:00Z",
  "snapshot_available": true
}
```

**Media Request (server → device):**
```json
{
  "request_type": "snapshot",
  "request_id": "req_456",
  "params": {
    "resolution": "1280x720",
    "format": "jpeg",
    "quality": 85
  }
}
```

**Media Upload (device → server):**
```json
{
  "request_id": "req_456",
  "media_type": "image",
  "format": "jpeg",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": "<base64 encoded>"
}
```

**Web/App Upload:**
```json
{
  "session_id": "web_session_789",
  "media_type": "image",
  "format": "jpeg",
  "filename": "photo.jpg",
  "context": "What's in this picture?",
  "data": "<base64 encoded>"
}
```

### 5.3 QoS Levels

| Message Type | QoS | Rationale |
|--------------|-----|-----------|
| Signaling | 1 | Must be delivered for connection |
| Commands | 1 | Must be delivered |
| Status | 1 | Retained, must be accurate |
| Alerts | 1 | Must be delivered |
| Telemetry | 0 | High frequency, loss acceptable |

---

## 6. Voice Conversation Pipeline

### 6.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VOICE CONVERSATION FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    WebRTC     ┌──────────────────────────────────────┐    │
│  │  Device  │    (Opus)     │            AI Server                 │    │
│  │          │═══════════════│                                      │    │
│  │  • Mic   │──────────────>│  ┌─────────┐                         │    │
│  │          │               │  │ Decode  │                         │    │
│  │          │               │  │  Opus   │                         │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ PCM                          │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │   VAD   │ (WebRTC or Silero)      │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ Speech segments              │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │   STT   │ (Whisper)               │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ Transcription                │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │   LLM   │ + Personality           │    │
│  │          │               │  └────┬────┘   + Visual context      │    │
│  │          │               │       │ Response text                │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │   TTS   │ (with emotion)          │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ PCM                          │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │<──────────────│  │ Encode  │                         │    │
│  │ • Speaker│    (Opus)     │  │  Opus   │                         │    │
│  └──────────┘               │  └─────────┘                         │    │
│                             └──────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Voice Activity Detection

**Option 1: WebRTC Built-in VAD**
- Enabled by default in WebRTC audio tracks
- Good enough for most cases
- Zero additional CPU

**Option 2: Silero VAD (Higher Accuracy)**
- Run on decoded PCM
- Better for noisy environments
- ~10ms per 30ms audio frame

### 6.3 Interruption Handling

WebRTC makes this natural:

1. **Simultaneous streams** - Both directions active at once
2. **Echo cancellation** - Device won't pick up its own speaker
3. **Server detects speech** - While TTS is playing
4. **Server stops TTS** - Immediately, mid-sentence
5. **Process new input** - No audio cleanup needed

### 6.4 Latency Breakdown

| Stage | Target | Notes |
|-------|--------|-------|
| Audio capture | ~20ms | WebRTC frame size |
| Network (WebRTC) | ~20-50ms | Local WiFi, UDP |
| Opus decode | ~1ms | Hardware accelerated |
| VAD | ~5ms | Per-frame |
| STT (streaming) | ~150-200ms | Whisper streaming |
| LLM | ~100-300ms | Depends on model |
| TTS (streaming) | ~50-100ms | First chunk |
| Opus encode | ~1ms | Hardware accelerated |
| Network back | ~20-50ms | Local WiFi, UDP |
| **Total** | **~400-700ms** | First audio response |

---

## 7. Vision Pipeline

### 7.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VISION MONITORING FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    WebRTC     ┌──────────────────────────────────────┐    │
│  │  Camera  │    (VP8)      │            AI Server                 │    │
│  │          │═══════════════│                                      │    │
│  │          │──────────────>│  ┌─────────┐                         │    │
│  │          │               │  │ Decode  │                         │    │
│  │          │               │  │  VP8    │                         │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ Raw frames                   │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │ Motion  │                         │    │
│  │          │               │  │ Detect  │                         │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ If motion                    │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐                         │    │
│  │          │               │  │  YOLO   │                         │    │
│  │          │               │  │ Detect  │                         │    │
│  │          │               │  └────┬────┘                         │    │
│  │          │               │       │ Detections                   │    │
│  │          │               │       ▼                              │    │
│  │          │               │  ┌─────────┐     ┌─────────┐         │    │
│  │          │               │  │  Zone   │────>│  Alert  │──> MQTT │    │
│  │          │               │  │ Filter  │     │  Dedup  │         │    │
│  │          │               │  └─────────┘     └─────────┘         │    │
│  │          │               │                                      │    │
│  └──────────┘               └──────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Efficient Video Processing

**Frame Rate Management:**
- WebRTC adaptive: 5-15 FPS based on bandwidth
- Process every Nth frame for detection (configurable)
- Motion detection on all frames (cheap)
- YOLO only when motion detected (expensive)

**Resource Optimization:**
- Decode only keyframes when idle
- Skip B-frames during high load
- Resize before YOLO (640x480 → 320x240)
- Batch detections when possible

### 7.3 Multi-Camera Support

Each camera is a separate WebRTC peer connection:

- Independent streams and processing
- Per-camera zone configuration
- Cross-camera alert deduplication
- Prioritization (front door > backyard)

---

## 8. Media Input Methods

The system supports multiple ways to receive media, ensuring flexibility across device types and use cases.

### 8.1 Real-Time vs Asynchronous Input

The fundamental distinction in this architecture is between **real-time streaming** and **asynchronous uploads**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME vs ASYNCHRONOUS COMPARISON                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REAL-TIME (WebRTC)                    ASYNCHRONOUS (MQTT)                  │
│  ─────────────────                     ────────────────────                 │
│                                                                             │
│  ┌─────────┐    continuous     ┌─────────┐                                  │
│  │ Device  │ ═══════════════►  │ Server  │     Voice conversation           │
│  │   Mic   │    audio stream   │   STT   │     <500ms latency               │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
│  ┌─────────┐    continuous     ┌─────────┐                                  │
│  │ Camera  │ ═══════════════►  │ Server  │     Security monitoring          │
│  │         │    video stream   │  YOLO   │     Real-time alerts             │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────┐    single image   ┌─────────┐                                  │
│  │ Device  │ ───────────────►  │ Server  │     On-demand snapshot           │
│  │ Camera  │    via MQTT       │ Vision  │     ~200-500ms                   │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
│  ┌─────────┐    audio clip     ┌─────────┐                                  │
│  │ Device  │ ───────────────►  │ Server  │     Voice message                │
│  │   Mic   │    via MQTT       │   STT   │     Async processing             │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
│  ┌─────────┐    file upload    ┌─────────┐                                  │
│  │ Browser │ ───────────────►  │ Server  │     User uploads image/audio     │
│  │  /App   │    via MQTT       │ Vision  │     Like ChatGPT attachments     │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
│  ┌─────────┐    retrieve       ┌─────────┐                                  │
│  │Database │ ───────────────►  │ Server  │     Historical media             │
│  │         │    internal       │ Vision  │     "Show me last night's alert" │
│  └─────────┘                   └─────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Aspect | Real-Time (WebRTC) | Asynchronous (MQTT/Internal) |
|--------|-------------------|------------------------------|
| **Transport** | UDP via WebRTC | MQTT messages or internal |
| **Latency** | <100ms | 200ms - seconds |
| **Connection** | Persistent bidirectional | Request/response |
| **Use case** | Live conversation, monitoring | Uploads, snapshots, history |
| **Data flow** | Continuous stream | Single item per request |
| **Interruption** | Supported (can talk over) | N/A |
| **Echo cancellation** | Built-in | N/A |
| **Codec** | Opus (audio), VP8 (video) | Any format, decoded on receive |

### 8.2 Input Method Summary

| Method | Transport | Use Case | Latency |
|--------|-----------|----------|---------|
| **WebRTC Stream** | WebRTC (UDP) | Continuous monitoring, live conversation | Real-time |
| **MQTT Snapshot** | MQTT | On-demand capture from non-streaming devices | ~200-500ms |
| **MQTT Recording** | MQTT | Audio/video clips from devices | Async |
| **MQTT Upload** | MQTT | Web/app uploads (via web service) | Async |
| **Database Retrieval** | Internal | Historical media lookup | Async |

### 8.3 Media Input Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────┐    ┌───────────────────────────────┐ │
│  │         EXTERNAL CLIENTS          │    │         AI SERVER             │ │
│  │                                   │    │    (This Document)            │ │
│  │  ┌─────────┐  ┌─────────────────┐ │    │                               │ │
│  │  │ Devices │  │   Web Service   │ │    │  ┌─────────────────────────┐  │ │
│  │  │         │  │   (separate)    │ │    │  │   Media Input Handler   │  │ │
│  │  │ • ESP32 │  │                 │ │    │  │                         │  │ │
│  │  │ • RPi   │  │  • Browser UI   │ │    │  │  • WebRTC streams       │  │ │
│  │  │ • Cams  │  │  • Mobile app   │ │    │  │  • MQTT snapshots       │  │ │
│  │  │         │  │  • HTTP API     │ │    │  │  • MQTT uploads         │  │ │
│  │  └────┬────┘  └────────┬────────┘ │    │  └───────────┬─────────────┘  │ │
│  │       │                │          │    │              │                │ │
│  └───────┼────────────────┼──────────┘    │              ▼                │ │
│          │                │               │  ┌─────────────────────────┐  │ │
│          │                │               │  │  Unified Media Context  │  │ │
│          │    ┌───────────┘               │  │                         │  │ │
│          │    │                           │  │  • Image → Vision       │  │ │
│          │    │  Bridges to               │  │  • Audio → STT          │  │ │
│          │    │  MQTT/WebRTC              │  │  • Video → Frames       │  │ │
│          │    │                           │  └───────────┬─────────────┘  │ │
│          │    │                           │              │                │ │
│          ▼    ▼                           │              ▼                │ │
│  ┌────────────────────┐                   │         LLM Context           │ │
│  │   MQTT + WebRTC    │◄──────────────────┤                               │ │
│  │                    │──────────────────►│                               │ │
│  └────────────────────┘                   └───────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Separation of Concerns:**
- **AI Server** - Receives media via MQTT/WebRTC only, processes AI workloads
- **Web Service** - Handles HTTP, WebSocket, browser UI, bridges to MQTT/WebRTC
- **Devices** - Connect directly via MQTT/WebRTC

### 8.4 Method 1: WebRTC Stream (Real-Time)

**Best for:** Continuous monitoring, live conversations, security cameras

```
Camera streaming via WebRTC
         │
         ▼
┌─────────────────┐
│ Grab Frame from │  Server extracts frame from live stream
│ Video Track     │  No request to device needed
└────────┬────────┘
         │
         ▼
    Process immediately
```

**Advantages:**
- Lowest latency
- No additional request/response
- Frame always available
- Higher resolution possible

**When to use:**
- Device is already streaming
- Real-time visual Q&A
- Security monitoring

### 8.5 Method 2: MQTT Snapshot (On-Demand)

**Best for:** Devices not continuously streaming, power-saving mode, legacy devices

```
Server                              Device
   │                                   │
   │  1. Request snapshot              │
   │  naila/devices/{id}/media/request │
   │──────────────────────────────────>│
   │                                   │
   │                                   │  2. Capture image
   │                                   │
   │  3. Return snapshot               │
   │  naila/devices/{id}/media/snapshot│
   │<──────────────────────────────────│
   │                                   │
   ▼
Process image
```

**Request:**
```json
{
  "request_type": "snapshot",
  "request_id": "req_456",
  "params": {
    "resolution": "1280x720",
    "format": "jpeg",
    "quality": 85
  }
}
```

**Response:**
```json
{
  "request_id": "req_456",
  "media_type": "image",
  "format": "jpeg",
  "width": 1280,
  "height": 720,
  "timestamp": "2024-01-15T10:30:00Z",
  "data": "<base64 encoded JPEG>"
}
```

**When to use:**
- Device doesn't support WebRTC
- Device is in low-power mode
- One-time capture needed

### 8.6 Method 3: MQTT Recording (Clips)

**Best for:** Audio messages, video clips, event recordings

```json
{
  "request_id": "req_789",
  "media_type": "audio",
  "format": "opus",
  "duration_ms": 5000,
  "sample_rate": 48000,
  "timestamp": "2024-01-15T10:30:00Z",
  "data": "<base64 encoded audio>"
}
```

**Supported formats:**
- Audio: Opus, WAV, MP3
- Video: MP4 (H264), WebM (VP8)
- Image: JPEG, PNG, WebP

### 8.7 Method 4: Web/Mobile Upload (via Web Service)

**Best for:** Browser uploads, mobile apps

Web browsers and mobile apps connect to a **separate web service** (not the AI server). The web service handles HTTP uploads and bridges them to MQTT for the AI server.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         WEB UPLOAD FLOW                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Browser/App    │     │   Web Service   │     │   AI Server     │       │
│  │                 │     │   (separate)    │     │  (this doc)     │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                │
│           │  1. HTTP POST         │                       │                │
│           │     /api/upload       │                       │                │
│           │──────────────────────>│                       │                │
│           │                       │                       │                │
│           │                       │  2. MQTT Publish      │                │
│           │                       │     naila/uploads/... │                │
│           │                       │──────────────────────>│                │
│           │                       │                       │                │
│           │                       │                       │  3. Process    │
│           │                       │                       │     media      │
│           │                       │                       │                │
│           │                       │  4. MQTT Response     │                │
│           │                       │<──────────────────────│                │
│           │                       │                       │                │
│           │  5. HTTP Response     │                       │                │
│           │<──────────────────────│                       │                │
│           │                       │                       │                │
└────────────────────────────────────────────────────────────────────────────┘
```

The web service publishes to MQTT, and the AI server receives it just like any other MQTT upload.

### 8.8 Method 5: Database Retrieval (Historical)

**Best for:** Accessing previously stored media, reviewing past alerts, conversation context

```
User: "Show me what happened at the front door last night"
         │
         ▼
┌─────────────────┐
│ Query Analysis  │  Detects historical intent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Database Query  │  Search by time, location, event type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Media  │  Load stored image/video/audio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Analysis │  Process if needed
└────────┬────────┘
         │
         ▼
    LLM Response with historical context
```

**Storage scenarios:**
- Alert snapshots (motion detected, person detected)
- Conversation recordings (if enabled)
- User-uploaded media history
- Periodic camera snapshots

**Query examples:**
- "What did the front door camera see at 3am?"
- "Show me the last alert from the backyard"
- "What was I asking about yesterday?"

### 8.9 MQTT Upload Format (from any source)

All uploads arrive at the AI server via MQTT in the same format, regardless of origin (device, web service bridge, or direct MQTT client):

**Topic:** `naila/uploads/{session_id}/image`

```json
{
  "session_id": "web_session_123",
  "media_type": "image",
  "format": "jpeg",
  "filename": "photo.jpg",
  "context": "What's in this picture?",
  "conversation_id": "conv_456",
  "data": "<base64 encoded>"
}
```

### 8.10 Media Source Selection Logic

When visual context is needed, the system chooses the best source:

```
┌─────────────────────────────────────────────────────────────┐
│                  MEDIA SOURCE SELECTION                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Check for attached media in current message             │
│     └─► If found: Use uploaded media                        │
│                                                             │
│  2. Check if query is about historical data                 │
│     └─► If historical: Query database for stored media      │
│                                                             │
│  3. Check for active WebRTC video stream                    │
│     └─► If streaming: Grab frame from stream                │
│                                                             │
│  4. Check device capabilities                               │
│     └─► If device has camera: Request MQTT snapshot         │
│                                                             │
│  5. Check other devices in same zone                        │
│     └─► If available: Request from best camera              │
│                                                             │
│  6. No visual context available                             │
│     └─► Respond without visual, or ask user to provide      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.11 Visual Context Format (Unified)

Regardless of input method, visual context is normalized:

```json
{
  "source": {
    "type": "webrtc_stream | mqtt_snapshot | http_upload | mqtt_upload",
    "device_id": "living_room_camera",
    "session_id": "web_session_123"
  },
  "media": {
    "type": "image",
    "format": "jpeg",
    "width": 1280,
    "height": 720,
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "analysis": {
    "description": "Office desk with laptop and accessories",
    "objects": [
      {"label": "laptop", "confidence": 0.95, "bbox": [100, 100, 400, 300]},
      {"label": "coffee mug", "confidence": 0.88, "bbox": [450, 200, 100, 120]}
    ],
    "scene_type": "indoor_office",
    "lighting": "natural",
    "text_detected": ["NAILA", "Meeting at 3pm"]
  },
  "user_context": "What's on my desk?"
}
```

### 8.12 Visual Query Detection

Keywords that trigger visual context capture:

- **Direct:** "see", "look", "show", "camera", "picture", "photo", "image"
- **Spatial:** "here", "there", "this", "that"
- **Descriptive:** "describe", "what's", "what is", "how many", "is there"
- **Location:** "desk", "room", "door", "window", "outside", "screen"
- **Analysis:** "read", "OCR", "text", "recognize", "identify"

### 8.13 Audio Input Methods

Same flexibility applies to audio:

| Method | Use Case |
|--------|----------|
| WebRTC stream | Live conversation |
| MQTT recording | Voice message from device |
| HTTP upload | Audio file from web/app |
| MQTT upload | Audio clip via WebSocket |

All audio is normalized to PCM and processed through STT.

---

## 9. Signaling Server

### 9.1 Signaling via MQTT

No separate signaling server needed - MQTT handles it:

**Server Subscribes To:**
- `naila/devices/+/status` - Device online/offline
- `naila/devices/+/signaling/answer` - SDP answers
- `naila/devices/+/signaling/ice` - ICE candidates from devices

**Server Publishes To:**
- `naila/devices/{device_id}/signaling/offer` - SDP offers
- `naila/devices/{device_id}/signaling/ice` - ICE candidates to devices

### 9.2 Session Establishment

**When device comes online:**

1. Device publishes to `naila/devices/{id}/status` with `online: true`
2. Server receives status, checks device capabilities
3. If device has audio/video, server creates RTCPeerConnection
4. Server creates SDP offer, publishes to `naila/devices/{id}/signaling/offer`
5. Device receives offer, creates answer
6. Device publishes answer to `naila/devices/{id}/signaling/answer`
7. Both exchange ICE candidates via `signaling/ice` topics
8. WebRTC connection established

### 9.3 Reconnection Handling

**On WebRTC disconnect:**
1. Server detects connection state change
2. Server closes old PeerConnection
3. Server creates new offer
4. Signaling repeats

**On MQTT disconnect:**
1. Device reconnects to MQTT (automatic with client library)
2. Device publishes new status
3. Server initiates new WebRTC session

---

## 10. Implementation Phases

### Phase 1: WebRTC Infrastructure
- `aiortc` integration
- MQTT signaling
- Single device audio connection
- Basic audio passthrough test

**Deliverable:** Voice audio streaming working

### Phase 2: Voice Pipeline
- STT integration with WebRTC audio
- LLM response generation
- TTS with Opus encoding back to device
- Interruption handling

**Deliverable:** Full voice conversation working

### Phase 3: Video Pipeline
- Video track handling
- Motion detection on frames
- YOLO integration
- Alert generation and MQTT publishing

**Deliverable:** Security monitoring working

### Phase 4: Media Input Flexibility
- MQTT snapshot request/response
- HTTP upload endpoint
- MQTT upload handler
- Media source selection logic
- Unified media context format

**Deliverable:** All input methods working (stream, snapshot, upload)

### Phase 5: Multi-Device Support
- Multiple peer connections
- Device registry
- Audio source selection
- Camera prioritization

**Deliverable:** Multi-room support working

### Phase 6: Polish & Production
- Reconnection robustness
- Error handling
- Performance optimization
- Monitoring and metrics

**Deliverable:** Production-ready system

### Phase Summary

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| 1 | WebRTC Infrastructure | Audio streaming |
| 2 | Voice Pipeline | Full conversation |
| 3 | Video Pipeline | Security monitoring |
| 4 | Media Input Flexibility | All input methods (stream, snapshot, MQTT upload) |
| 5 | Multi-Device | Multi-room support |
| 6 | Polish | Production ready |

---

## 11. File Structure

### 11.1 New Files

| File | Purpose |
|------|---------|
| `webrtc/__init__.py` | Module exports |
| `webrtc/peer_manager.py` | PeerConnection lifecycle management |
| `webrtc/audio_track.py` | Audio track handling (receive/send) |
| `webrtc/video_track.py` | Video track handling |
| `webrtc/signaling.py` | MQTT-based signaling |
| `media/__init__.py` | Media handling exports |
| `media/input_handler.py` | Unified media input handler |
| `media/upload_handler.py` | MQTT upload processing |
| `media/snapshot_handler.py` | MQTT snapshot request/response |
| `media/source_selector.py` | Media source selection logic |
| `media/context_builder.py` | Build unified media context for LLM |
| `pipelines/__init__.py` | Pipeline exports |
| `pipelines/voice.py` | Voice conversation pipeline |
| `pipelines/vision.py` | Video processing pipeline |
| `config/webrtc.py` | WebRTC configuration |
| `config/media.py` | Media handling configuration |

### 11.2 Modified Files

| File | Changes |
|------|---------|
| `server/naila_server.py` | Initialize WebRTC |
| `server/lifecycle.py` | Add WebRTC startup stage |
| `services/stt.py` | Accept PCM from WebRTC or uploaded audio |
| `services/tts.py` | Output PCM for WebRTC encoder |
| `services/vision.py` | Accept frames from any source |
| `mqtt/handlers/device_handlers.py` | Add signaling and media handlers |
| `config/__init__.py` | Export new configs |
| `requirements.txt` | Add aiortc, Pillow, pydub |

---

## 12. Configuration

### 12.1 Configuration Classes

**WebRTCConfig:**
- ICE servers (empty for local network)
- Audio codec preferences
- Video codec preferences
- Bitrate limits

**AudioConfig:**
- Opus bitrate (24-32 kbps)
- VAD mode (WebRTC or Silero)
- Echo cancellation settings

**VideoConfig:**
- Resolution (640x480)
- Framerate (5-15 FPS)
- Processing frame skip
- Motion threshold

**MediaConfig:**
- Max upload size
- Supported formats
- Snapshot timeout

### 12.2 Environment Variables

```bash
# WebRTC
WEBRTC_ICE_SERVERS=[]  # Empty for local network
WEBRTC_AUDIO_BITRATE=32000
WEBRTC_VIDEO_BITRATE=1000000

# Audio
AUDIO_VAD_MODE=webrtc  # or "silero"
AUDIO_ECHO_CANCELLATION=true
AUDIO_NOISE_SUPPRESSION=true

# Video
VIDEO_RESOLUTION=640x480
VIDEO_MAX_FPS=15
VIDEO_MOTION_THRESHOLD=25
VIDEO_PROCESS_EVERY_N_FRAMES=3

# Media Input
MEDIA_MAX_UPLOAD_SIZE=52428800  # 50MB
MEDIA_SUPPORTED_IMAGE_FORMATS=jpeg,png,webp,gif
MEDIA_SUPPORTED_AUDIO_FORMATS=opus,wav,mp3,m4a
MEDIA_SUPPORTED_VIDEO_FORMATS=mp4,webm
MEDIA_SNAPSHOT_TIMEOUT_MS=5000

```

---

## 13. Testing Strategy

### 13.1 Unit Tests

**WebRTC:**
- PeerConnection creation/teardown
- SDP offer/answer exchange
- ICE candidate handling
- Track addition/removal

**Voice Pipeline:**
- Audio decode → STT → LLM → TTS → encode
- Interruption mid-response
- VAD accuracy

**Vision Pipeline:**
- Frame decode → motion → YOLO → alert
- Zone filtering
- Alert deduplication

**Media Input:**
- MQTT upload handling
- MQTT snapshot request/response
- Media source selection
- Format conversion and normalization

### 13.2 Integration Tests

- Full signaling flow via MQTT
- Voice conversation round trip
- Video streaming with alerts
- Multi-device scenarios
- MQTT upload → vision → LLM → response
- MQTT snapshot flow end-to-end

### 13.3 Network Tests

- Packet loss simulation
- Bandwidth constraint simulation
- Reconnection scenarios
- NAT traversal (if configured)

---

## 14. Performance Considerations

### 14.1 Latency Budget (Voice)

| Stage | Target | Optimization |
|-------|--------|--------------|
| Network (WebRTC) | 20-50ms | UDP, local WiFi |
| Audio decode | <5ms | Opus hardware decode |
| VAD + STT | 150-200ms | Streaming Whisper |
| LLM | 100-300ms | Fast model, streaming |
| TTS | 50-100ms | Streaming synthesis |
| Audio encode | <5ms | Opus hardware encode |
| Network back | 20-50ms | UDP, local WiFi |
| **Total first audio** | **~400-700ms** | |

### 14.2 CPU Optimization

**WebRTC (handled by library):**
- Hardware codec acceleration when available
- Efficient RTP packetization
- Built-in congestion control

**Vision:**
- Process every Nth frame
- Motion detection gates YOLO
- Batch frames when possible
- GPU for YOLO inference

### 14.3 Memory Management

**Audio:**
- Ring buffers with fixed size
- Clear after STT processing
- No raw audio persistence

**Video:**
- Frame pool with max size
- Discard old frames under load
- No frame persistence (except snapshots)

---

## 15. Security Considerations

### 15.1 WebRTC Security

**Built-in:**
- DTLS for signaling encryption
- SRTP for media encryption
- Certificate fingerprint verification

**Configuration:**
- Use secure MQTT (TLS) for signaling
- Validate device IDs in signaling
- ICE candidate filtering (optional)

### 15.2 MQTT Security

- TLS encryption
- Username/password or certificate auth
- ACLs per device (limit topic access)
- Rate limiting

### 15.3 General

- No media persistence by default
- Secure alert delivery
- Input validation on all messages
- Audit logging for security events
- Uploaded files scanned before processing

---

## 16. Migration Guide

### 16.1 From Current MQTT-Only Architecture

**Phase 1: Add WebRTC Server-Side**
1. Install aiortc
2. Add WebRTC module
3. Add signaling handlers
4. Test with single device

**Phase 2: Update Device Firmware**
1. Add WebRTC library to device
2. Implement signaling client
3. Connect audio to WebRTC track
4. Test voice conversation

**Phase 3: Migrate Features**
1. Move voice from MQTT to WebRTC
2. Add video streaming
3. Keep commands/alerts on MQTT
4. Deprecate old audio handlers

### 16.2 Rollback Plan

1. Device firmware supports both paths
2. Server flag to disable WebRTC
3. Falls back to MQTT audio (degraded)
4. No data loss

---

## Appendix A: Dependencies

### Required

```
# WebRTC
aiortc>=1.6.0           # WebRTC for Python
aioice>=0.9.0           # ICE implementation
av>=11.0.0              # FFmpeg bindings (codec support)

# Media processing
numpy>=1.24.0           # Audio/video processing
opencv-python-headless>=4.8.0  # Video processing
Pillow>=10.0.0          # Image handling
pydub>=0.25.1           # Audio format conversion
```

### Optional

```
silero-vad>=4.0.0       # Better VAD than WebRTC built-in
torch>=2.0.0            # GPU acceleration for YOLO/Whisper
```

---

## Appendix B: aiortc Quick Reference

### Creating a PeerConnection

```python
from aiortc import RTCPeerConnection, RTCSessionDescription

pc = RTCPeerConnection()

# Add handlers
pc.on("track", handle_track)
pc.on("connectionstatechange", handle_state_change)

# Create offer
offer = await pc.createOffer()
await pc.setLocalDescription(offer)

# Send offer.sdp via MQTT signaling
# Receive answer via MQTT signaling

answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
await pc.setRemoteDescription(answer)
```

### Handling Audio Track

```python
async def handle_track(track):
    if track.kind == "audio":
        while True:
            frame = await track.recv()
            # frame.to_ndarray() gives PCM samples
            pcm_data = frame.to_ndarray()
            await process_audio(pcm_data)
```

### Sending Audio Back

```python
from aiortc.contrib.media import MediaPlayer, MediaRecorder

# Create audio track from TTS output
class TTSAudioTrack(AudioStreamTrack):
    async def recv(self):
        # Return next audio frame from TTS
        pcm_data = await tts_queue.get()
        return AudioFrame.from_ndarray(pcm_data, format="s16", layout="mono")

# Add to peer connection
pc.addTrack(TTSAudioTrack())
```

---

## Appendix C: Device Implementation Notes

### Raspberry Pi (Python)

Use `aiortc` same as server:
```python
# Same library, just reversed roles
pc = RTCPeerConnection()
pc.addTrack(MediaPlayer("/dev/audio").audio)
# Handle incoming audio track for speaker output
```

### ESP32 (C/C++)

Options:
1. **Audio only:** Use Opus library + custom UDP transport
2. **Full WebRTC:** Use `libwebrtc` or `pjsip`
3. **Simplified:** Stream raw audio, let server handle codec

### Browser (JavaScript)

Native WebRTC:
```javascript
const pc = new RTCPeerConnection();
const stream = await navigator.mediaDevices.getUserMedia({audio: true});
stream.getTracks().forEach(track => pc.addTrack(track, stream));
```

---

*Document End*
