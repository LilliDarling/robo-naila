# Real-Time Streaming Implementation Guide

## NAILA AI Server - WebRTC + MQTT Architecture

**Version:** 2.0
**Date:** December 2025
**Status:** Implementation Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Protocol Selection Rationale](#3-protocol-selection-rationale)
4. [Split Architecture Design](#4-split-architecture-design)
5. [Pi Gateway Layer](#5-pi-gateway-layer)
6. [AI Server Integration](#6-ai-server-integration)
7. [Voice Conversation Pipeline](#7-voice-conversation-pipeline)
8. [MQTT Control Layer](#8-mqtt-control-layer)
9. [Signaling Flow](#9-signaling-flow)
10. [Vision Pipeline](#10-vision-pipeline)
11. [Media Input Methods](#11-media-input-methods)
12. [Implementation Phases](#12-implementation-phases)
13. [File Structure](#13-file-structure)
14. [Configuration](#14-configuration)
15. [Testing Strategy](#15-testing-strategy)
16. [Performance Considerations](#16-performance-considerations)
17. [Security Considerations](#17-security-considerations)
18. [Hardware Requirements](#18-hardware-requirements)
19. [Migration Guide](#19-migration-guide)

**Note:** This doc is for alignment purposes only and not meant as a final version.

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies a **Split Gateway Architecture** for the NAILA AI Server, enabling:

1. **Seamless Voice Conversation** - Sub-500ms latency, interruptible voice with built-in echo cancellation
2. **Resource Isolation** - Real-time audio handling separated from CPU-intensive inference
3. **Multi-Device Support** - Multiple audio/video sources across WiFi network
4. **Flexible Media Input** - Support for real-time streams, snapshots, uploads, and historical retrieval
5. **Scalable Design** - Gateway can be scaled independently of AI inference

### 1.2 Key Goals

| Goal | Current State | Target State |
|------|---------------|--------------|
| Voice response latency | 2-5 seconds | <500ms (first audio) |
| Conversation interruption | Not supported | Full support (WebRTC handles this) |
| Echo cancellation | None | Built-in (WebRTC AEC) |
| Audio codec | Raw PCM/base64 | Opus (10x compression) |
| CPU contention | N/A (no streaming) | Isolated (gateway separate from inference) |
| NAT traversal | Not supported | Automatic (ICE/STUN) |
| Network adaptation | None | Automatic bitrate adjustment |

### 1.3 Architecture Decision

**Split Gateway Architecture:**

| Component | Location | Purpose |
|-----------|----------|---------|
| **Pi Gateway** | Raspberry Pi | WebRTC, codecs, VAD, real-time audio |
| **AI Server** | Main Server | STT, LLM, TTS, Vision inference |
| **MQTT Broker** | Main Server | Signaling, control, alerts |

**Why this approach:**
- **CPU Isolation** - AI inference (Whisper, Llama, Piper) can saturate CPU without affecting real-time audio
- **WebRTC on Pi** - Dedicated hardware for timing-critical operations
- **Simple Integration** - TCP socket between gateway and AI server adds ~2ms latency
- **Existing Infrastructure** - MQTT remains for signaling and control plane

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL NETWORK                                      │
│                                                                                 │
│  ┌──────────────┐       ┌───────────────────┐       ┌───────────────────┐       │
│  │   Devices    │       │   Raspberry Pi    │       │    Main Server    │       │
│  │              │       │    (Gateway)      │       │    (AI Server)    │       │
│  │  • ESP32     │       │                   │       │                   │       │
│  │  • Phones    │WebRTC │  • aiortc         │  TCP  │  • Whisper (STT)  │       │
│  │  • Browsers  │◀════▶│  • Opus codec     │◀─────▶│  • Llama (LLM)   │       │
│  │  • Cameras   │ Audio │  • VAD            │ Audio │  • Piper (TTS)    │       │
│  │              │       │  • Buffering      │  PCM  │  • YOLOv8 (Vision)│       │
│  └──────────────┘       │  • MQTT signaling │       │  • Orchestrator   │       │
│                         └─────────┬─────────┘       └─────────┬─────────┘       │
│                                   │                           │                 │
│                                   │          MQTT             │                 │
│                                   └───────────┬───────────────┘                 │
│                                               │                                 │
│                                   ┌───────────▼───────────┐                     │
│                                   │     MQTT Broker       │                     │
│                                   │                       │                     │
│                                   │  • Signaling          │                     │
│                                   │  • Device status      │                     │
│                                   │  • Commands/Alerts    │                     │
│                                   └───────────────────────┘                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSIBILITY SPLIT                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PI GATEWAY (Real-Time Critical)       AI SERVER (Inference)                   │
│   ─────────────────────────────         ─────────────────────                   │
│                                                                                 │
│   ┌─────────────────────────┐           ┌─────────────────────────┐             │
│   │ • WebRTC connections    │           │ • Speech-to-Text        │             │
│   │ • Opus encode/decode    │           │ • Language model        │             │
│   │ • Voice activity detect │    TCP    │ • Text-to-Speech        │             │
│   │ • Audio buffering       │◀────────▶│ • Vision analysis       │             │
│   │ • Jitter handling       │   ~2ms    │ • Conversation memory   │             │
│   │ • Echo cancellation     │           │ • Response generation   │             │
│   │ • Device management     │           │ • MQTT publish          │             │
│   └─────────────────────────┘           └─────────────────────────┘             │
│                                                                                 │
│   CPU: 20-40% (headroom)                CPU: Up to 100% (OK)                    │
│   Latency: Timing-critical              Latency: Best-effort                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Connection Flow

```
┌──────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
│  Device  │          │   MQTT   │          │    Pi    │          │    AI    │
│          │          │  Broker  │          │  Gateway │          │  Server  │
└────┬─────┘          └────┬─────┘          └────┬─────┘          └────┬─────┘
     │                     │                     │                     │
     │  1. MQTT Connect    │                     │                     │
     │────────────────────>│                     │                     │
     │                     │                     │                     │
     │  2. Publish status  │                     │                     │
     │────────────────────>│────────────────────>│                     │
     │                     │                     │                     │
     │                     │  3. Pi creates PeerConnection             │
     │                     │                     │                     │
     │                     │  4. SDP Offer       │                     │
     │<────────────────────│<────────────────────│                     │
     │                     │                     │                     │
     │  5. SDP Answer      │                     │                     │
     │────────────────────>│────────────────────>│                     │
     │                     │                     │                     │
     │  6. ICE Candidates  │                     │                     │
     │<───────────────────>│<───────────────────>│                     │
     │                     │                     │                     │
     │  7. WebRTC Connected                      │                     │
     │<═════════════════════════════════════════>│                     │
     │                     │                     │                     │
     │  8. Audio streaming │                     │  9. TCP: PCM audio  │
     │<═════════════════════════════════════════>│<═══════════════════>│
     │                     │                     │                     │
```

### 2.4 Data Flow Summary

| Data Type | Transport | Path | Format |
|-----------|-----------|------|--------|
| Voice (in) | WebRTC | Device → Pi | Opus |
| Voice (out) | WebRTC | Pi → Device | Opus |
| Audio (processing) | TCP | Pi ↔ AI Server | Raw PCM |
| Signaling | MQTT | Device ↔ Pi | JSON |
| Commands | MQTT | All components | JSON |
| Alerts | MQTT | AI Server → Clients | JSON |
| Device status | MQTT | Device → Pi/Server | JSON |

---

## 3. Protocol Selection Rationale

### 3.1 Why WebRTC for Device Communication

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

### 3.2 Why Split Gateway from AI Server

| Concern | Single Server | Split Architecture |
|---------|---------------|-------------------|
| CPU contention | WebRTC stutters during inference | Isolated, no impact |
| Latency consistency | Variable (0-500ms jitter) | Consistent (~25ms) |
| Scalability | Limited by slowest component | Scale independently |
| Debugging | Complex interactions | Clear boundaries |
| Hardware flexibility | All-or-nothing | Mix CPU/GPU as needed |

**Key insight:** AI inference (Whisper, Llama, Piper) can saturate 100% CPU for 200-500ms. Without isolation, WebRTC audio drops or stutters during these spikes.

### 3.3 Why TCP for Gateway-Server Communication

| Option | Latency | Complexity | Reliability |
|--------|---------|------------|-------------|
| Unix socket | ~0.1ms | Low | High |
| TCP (localhost) | ~0.5ms | Low | High |
| TCP (network) | ~2ms | Low | High |
| MQTT | ~5-10ms | Medium | High |
| gRPC | ~3-5ms | High | High |

**Decision:** TCP socket for simplicity. The ~2ms network latency is negligible compared to the 200-400ms inference time.

### 3.4 Why MQTT for Signaling and Control

- **Already deployed** - MQTT broker exists in current architecture
- **Pub/Sub model** - Better fit for alerts, multi-subscriber events
- **Retained messages** - Device status persists across reconnects
- **QoS levels** - Guaranteed delivery for critical commands
- **Simpler debugging** - Standard MQTT tools work

---

## 4. Split Architecture Design

### 4.1 Design Principles

1. **Separation of Concerns** - Real-time audio handling isolated from inference
2. **Fail Independently** - Gateway failure doesn't crash AI server and vice versa
3. **Simple Protocol** - TCP with minimal framing between components
4. **Stateless Gateway** - Conversation state lives in AI server only

### 4.2 Communication Protocol

**Gateway → AI Server (Audio Request):**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUDIO REQUEST FRAME                        │
├─────────────────────────────────────────────────────────────────┤
│  Header (text, newline-terminated):                             │
│    {device_id}|{audio_length}|{sample_rate}|{request_id}\n      │
│                                                                 │
│  Body (binary):                                                 │
│    Raw PCM audio bytes (int16, mono)                            │
└─────────────────────────────────────────────────────────────────┘

Example:
  "living_room|96000|48000|req_abc123\n" + <96000 bytes of PCM>
```

**AI Server → Gateway (Audio Response):**

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUDIO RESPONSE FRAME                       │
├─────────────────────────────────────────────────────────────────┤
│  Header (text, newline-terminated):                             │
│    {device_id}|{audio_length}|{sample_rate}|{request_id}\n      │
│                                                                 │
│  Body (binary):                                                 │
│    Raw PCM audio bytes (int16, mono)                            │
└─────────────────────────────────────────────────────────────────┘

Example:
  "living_room|144000|48000|req_abc123\n" + <144000 bytes of PCM>
```

### 4.3 Connection Management

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONNECTION LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Gateway Startup:                                               │
│    1. Connect to MQTT broker                                    │
│    2. Connect TCP to AI server (with retry)                     │
│    3. Subscribe to device status topics                         │
│    4. Publish gateway online status                             │
│                                                                 │
│  Device Connection:                                             │
│    1. Device publishes status (via MQTT)                        │
│    2. Gateway receives status, creates PeerConnection           │
│    3. Gateway sends SDP offer (via MQTT)                        │
│    4. Device responds with SDP answer                           │
│    5. ICE candidates exchanged                                  │
│    6. WebRTC connected, audio flows                             │
│                                                                 │
│  Reconnection:                                                  │
│    - Gateway reconnects to AI server automatically              │
│    - Pending audio requests are dropped (acceptable)            │
│    - Device WebRTC sessions maintained during AI server outage  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Error Handling

| Scenario | Gateway Behavior | AI Server Behavior |
|----------|------------------|-------------------|
| AI server disconnect | Buffer audio, retry connection | N/A |
| AI server slow response | Timeout after 10s, notify device | Log, continue processing |
| Device disconnect | Clean up PeerConnection | Notified via MQTT |
| Invalid audio | Log and skip | Return empty response |
| Gateway crash | Devices lose audio | Continue MQTT operations |

---

## 5. Pi Gateway Layer

### 5.1 Overview

The Pi Gateway runs on a Raspberry Pi and handles all real-time audio operations using `aiortc`.

**Library:** `aiortc` - Pure Python WebRTC implementation

**Responsibilities:**
- Manage WebRTC connections to all devices
- Opus codec encode/decode
- Voice Activity Detection (VAD)
- Buffer and forward audio to AI server
- Receive TTS audio and stream to devices
- MQTT signaling coordination

### 5.2 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      PI GATEWAY COMPONENTS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  MQTT Handler   │    │  TCP Client     │                     │
│  │                 │    │                 │                     │
│  │  • Signaling    │    │  • AI server    │                     │
│  │  • Device status│    │    connection   │                     │
│  │  • Commands     │    │  • Send PCM     │                     │
│  └────────┬────────┘    │  • Recv TTS     │                     │
│           │             └────────┬────────┘                     │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌──────────────────────────────────────────┐                   │
│  │           Session Manager                │                   │
│  │                                          │                   │
│  │  • Device sessions (PeerConnection each) │                   │
│  │  • Audio buffers per device              │                   │
│  │  • Request/response correlation          │                   │
│  └──────────────────────────────────────────┘                   │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │           Audio Pipeline                 │                   │
│  │                                          │                   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐   │                   │
│  │  │  Opus   │  │   VAD   │  │ Buffer  │   │                   │
│  │  │ Decode  │─▶│ Detect  │─▶│ Speech │   │                   │
│  │  └─────────┘  └─────────┘  └─────────┘   │                   │
│  │                                          │                   │
│  │  ┌─────────┐  ┌─────────┐                │                   │
│  │  │  Opus   │  │  Queue  │                │                   │
│  │  │ Encode  │◀─│   TTS   │               │                    │
│  │  └─────────┘  └─────────┘                │                   │
│  │                                          │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Device Session

Each connected device gets a session:

```python
@dataclass
class DeviceSession:
    device_id: str
    pc: RTCPeerConnection
    audio_buffer: np.ndarray      # Accumulates speech
    buffer_pos: int = 0
    is_speaking: bool = False
    silence_frames: int = 0
    pending_requests: dict = field(default_factory=dict)
```

### 5.4 Voice Activity Detection

**Option 1: Energy-based (Simple)**
```python
def is_speech(pcm: np.ndarray, threshold: int = 500) -> bool:
    return np.abs(pcm).mean() > threshold
```

**Option 2: Silero VAD (Accurate)**
```python
from silero_vad import load_silero_vad

vad_model = load_silero_vad()

def is_speech(pcm: np.ndarray) -> bool:
    # Silero expects float32, normalized
    audio = pcm.astype(np.float32) / 32768.0
    confidence = vad_model(audio, sample_rate=48000)
    return confidence > 0.5
```

**Utterance Detection:**
- Speech starts: Begin buffering
- Speech continues: Append to buffer
- Silence detected: Count frames
- 300ms silence: End of utterance, send to AI server

### 5.5 WebRTC Configuration

**Audio Settings:**
```python
audio_config = {
    "codec": "opus",
    "sample_rate": 48000,
    "channels": 1,
    "bitrate": 32000,      # 32 kbps
    "frame_size": 960,     # 20ms at 48kHz
}
```

**ICE Configuration (Local Network):**
```python
ice_config = RTCConfiguration(
    iceServers=[]  # No STUN/TURN needed on local network
)
```

### 5.6 Audio Track Implementation

**Receiving Audio (from device):**
```python
@pc.on("track")
async def on_track(track: MediaStreamTrack):
    if track.kind == "audio":
        while True:
            frame = await track.recv()
            pcm = frame.to_ndarray().flatten().astype(np.int16)
            await process_audio(session, pcm)
```

**Sending Audio (TTS to device):**
```python
class TTSOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio_queue: asyncio.Queue):
        super().__init__()
        self.queue = audio_queue
        self.sample_rate = 48000
        self.frame_size = 960  # 20ms

    async def recv(self) -> AudioFrame:
        # Get audio from queue or return silence
        try:
            pcm = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pcm = np.zeros(self.frame_size, dtype=np.int16)

        frame = AudioFrame.from_ndarray(
            pcm.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        return frame
```

### 5.7 Dependencies

```
# Pi Gateway requirements
aiortc>=1.6.0,<2.0.0
aioice>=0.9.0
av>=11.0.0,<12.0.0
numpy>=1.24.0
asyncio-mqtt>=0.16.0
silero-vad>=4.0.0        # Optional, for better VAD
```

---

## 6. AI Server Integration

### 6.1 Audio Socket Server

The AI server exposes a TCP socket for receiving audio from the Pi gateway.

```python
class AudioInferenceServer:
    """
    TCP server that receives PCM audio, runs inference, returns TTS.
    """

    def __init__(self, stt, llm, tts, orchestrator):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.orchestrator = orchestrator

    async def start(self, host: str = "0.0.0.0", port: int = 9999):
        server = await asyncio.start_server(
            self._handle_client, host, port
        )
        await server.serve_forever()

    async def _handle_client(self, reader, writer):
        while True:
            # Read header: device_id|audio_len|sample_rate|request_id\n
            header = await reader.readline()
            if not header:
                break

            device_id, audio_len, sample_rate, request_id = (
                header.decode().strip().split("|")
            )

            # Read PCM audio
            pcm_bytes = await reader.readexactly(int(audio_len))

            # Process and respond
            response_audio = await self._process(device_id, pcm_bytes)

            # Send response
            resp_header = f"{device_id}|{len(response_audio)}|48000|{request_id}\n"
            writer.write(resp_header.encode() + response_audio)
            await writer.drain()
```

### 6.2 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI SERVER PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PCM Audio (from Gateway)                                       │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │  Format Audio   │  Convert to WAV for Whisper                │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │      STT        │  Whisper: PCM → Text                       │
│  │   (Whisper)     │  ~150-300ms (CPU)                          │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  Orchestrator   │  Context, memory, routing                  │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │      LLM        │  Llama: Text → Response                    │
│  │    (Llama)      │  ~100-400ms (CPU)                          │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │      TTS        │  Piper: Text → PCM                         │
│  │    (Piper)      │  ~50-150ms (CPU)                           │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  PCM Audio (to Gateway)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Integration with Existing Services

Minimal changes to existing ai-server:

**New file:** `server/audio_socket.py`
- TCP server for gateway communication
- Request/response handling

**Modified:** `server/lifecycle.py`
- Start audio socket server on boot

**No changes to:**
- `services/stt.py` - Already accepts audio bytes
- `services/llm.py` - Already generates responses
- `services/tts.py` - Already produces PCM
- `agents/orchestrator.py` - Already coordinates pipeline

### 6.4 Processing Flow

```python
async def _process(self, device_id: str, pcm_bytes: bytes) -> bytes:
    """STT → LLM → TTS pipeline."""

    # 1. Convert PCM to WAV (Whisper expects WAV)
    wav_data = pcm_to_wav(pcm_bytes, sample_rate=48000)

    # 2. Speech to text
    result = await self.stt.transcribe(wav_data)
    if not result.text.strip():
        return b""  # Silence/noise

    # 3. Generate response via orchestrator
    response = await self.orchestrator.process_text(
        device_id=device_id,
        text=result.text
    )

    # 4. Text to speech
    audio = await self.tts.synthesize(
        text=response.text,
        output_format="raw"  # Raw PCM for gateway
    )

    return audio.data
```

---

## 7. Voice Conversation Pipeline

### 7.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       VOICE CONVERSATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐  WebRTC   ┌──────────┐   TCP    ┌──────────────────────────┐  │
│  │  Device  │  (Opus)   │    Pi    │  (PCM)   │       AI Server          │  │
│  │          │◀════════▶│  Gateway │◀═══════▶│                          │  │
│  └──────────┘           └──────────┘          └──────────────────────────┘  │
│                                                                             │
│  DETAILED FLOW:                                                             │
│                                                                             │
│  Device        Pi Gateway              AI Server                            │
│    │               │                       │                                │
│    │──── Opus ───▶│                       │                                │
│    │               │── Decode ──▶ PCM      │                               │
│    │               │── VAD ────▶ Speech?   │                               │
│    │               │── Buffer ─▶ Utterance │                               │
│    │               │                       │                                │
│    │               │════ PCM (TCP) ═══════▶│                               │
│    │               │                       │── STT ──▶ Text                │
│    │               │                       │── LLM ──▶ Response            │
│    │               │                       │── TTS ──▶ PCM                 │
│    │               │◀════ PCM (TCP) ═══════│                               │
│    │               │                       │                                │
│    │               │── Queue ──▶ Buffer    │                               │
│    │               │── Encode ─▶ Opus      │                               │
│    │◀─── Opus ─────│                       │                               │
│    │               │                       │                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Interruption Handling

Interruptions are handled at the Pi gateway:

1. **User speaks while TTS playing**
   - Gateway detects speech via VAD
   - Gateway clears TTS output queue
   - Gateway sends interrupt signal to device (optional)
   - New utterance buffered and sent to AI server

2. **Implementation:**
```python
async def handle_audio(session, pcm):
    is_speech = vad.is_speech(pcm)

    if is_speech and session.tts_playing:
        # Clear TTS queue - stop playback
        session.tts_queue.clear()
        session.tts_playing = False

    # Continue with normal VAD/buffering...
```

### 7.3 Latency Breakdown

| Stage | Location | Target | Notes |
|-------|----------|--------|-------|
| Audio capture | Device | ~20ms | WebRTC frame size |
| Network (WebRTC) | Device→Pi | ~20-50ms | Local WiFi, UDP |
| Opus decode | Pi | ~1ms | Software decode |
| VAD + buffer | Pi | ~100ms | Wait for utterance end |
| Network (TCP) | Pi→Server | ~2ms | Wired ethernet |
| STT (Whisper) | Server | ~150-300ms | CPU inference |
| LLM (Llama) | Server | ~100-400ms | CPU inference |
| TTS (Piper) | Server | ~50-150ms | CPU inference |
| Network (TCP) | Server→Pi | ~2ms | Wired ethernet |
| Opus encode | Pi | ~1ms | Software encode |
| Network (WebRTC) | Pi→Device | ~20-50ms | Local WiFi, UDP |
| **Total** | | **~450-950ms** | First audio response |

### 7.4 Latency Optimization

**Priority optimizations:**

1. **Streaming TTS** - Send first chunk before full synthesis complete
2. **Smaller LLM** - Use quantized model (Q4) for faster inference
3. **Whisper tiny/base** - Trade accuracy for speed
4. **Wired Pi connection** - Eliminate one WiFi hop

---

## 8. MQTT Control Layer

### 8.1 Topic Structure

```
naila/
├── gateway/
│   ├── status                # Gateway online status (retained)
│   └── metrics               # Gateway performance metrics
├── devices/
│   └── {device_id}/
│       ├── status            # Device status (retained)
│       ├── command           # Commands to device
│       ├── signaling/
│       │   ├── offer         # SDP offers (from gateway)
│       │   ├── answer        # SDP answers (from device)
│       │   └── ice           # ICE candidates
│       ├── media/
│       │   ├── snapshot      # On-demand image capture
│       │   └── request       # Media requests
│       └── events            # Device events
├── server/
│   ├── status                # AI server status (retained)
│   └── broadcast             # Broadcast to all
└── alerts/
    ├── {zone}/
    │   └── {alert_type}      # Zone-specific alerts
    └── all                   # All alerts
```

### 8.2 Message Types

**Device Status (retained):**
```json
{
  "device_id": "living_room_hub",
  "online": true,
  "capabilities": ["audio", "speaker"],
  "firmware_version": "1.2.0",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Signaling - SDP Offer (Gateway → Device):**
```json
{
  "type": "offer",
  "sdp": "v=0\r\no=- 123456 ...",
  "session_id": "abc123"
}
```

**Signaling - SDP Answer (Device → Gateway):**
```json
{
  "type": "answer",
  "sdp": "v=0\r\no=- 789012 ...",
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

**Alert:**
```json
{
  "alert_type": "person_detected",
  "zone": "front_door",
  "confidence": 0.92,
  "source_device": "front_camera",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### 8.3 QoS Levels

| Message Type | QoS | Rationale |
|--------------|-----|-----------|
| Signaling | 1 | Must be delivered for connection |
| Commands | 1 | Must be delivered |
| Status | 1 | Retained, must be accurate |
| Alerts | 1 | Must be delivered |
| Telemetry | 0 | High frequency, loss acceptable |

---

## 9. Signaling Flow

### 9.1 Device Connection Sequence

```
┌──────────┐          ┌──────────┐          ┌──────────┐
│  Device  │          │   MQTT   │          │    Pi    │
│          │          │  Broker  │          │  Gateway │
└────┬─────┘          └────┬─────┘          └────┬─────┘
     │                     │                     │
     │  1. Connect MQTT    │                     │
     │────────────────────>│                     │
     │                     │                     │
     │  2. Publish status  │                     │
     │  (online: true)     │                     │
     │────────────────────>│────────────────────>│
     │                     │                     │
     │                     │  3. Gateway receives status
     │                     │     Creates PeerConnection
     │                     │     Generates SDP offer
     │                     │                     │
     │  4. SDP Offer       │                     │
     │  signaling/offer    │                     │
     │<────────────────────│<────────────────────│
     │                     │                     │
     │  5. Device creates answer                 │
     │                     │                     │
     │  6. SDP Answer      │                     │
     │  signaling/answer   │                     │
     │────────────────────>│────────────────────>│
     │                     │                     │
     │  7. ICE Candidates (both directions)      │
     │<───────────────────>│<───────────────────>│
     │                     │                     │
     │  8. WebRTC Connection Established         │
     │<═══════════════════════════════════════>  │
     │                     │                     │
```

### 9.2 Gateway MQTT Subscriptions

```python
# Gateway subscribes to:
subscriptions = [
    "naila/devices/+/status",           # Device online/offline
    "naila/devices/+/signaling/answer", # SDP answers
    "naila/devices/+/signaling/ice",    # ICE candidates from devices
]

# Gateway publishes to:
# naila/devices/{device_id}/signaling/offer  - SDP offers
# naila/devices/{device_id}/signaling/ice    - ICE candidates to devices
# naila/gateway/status                        - Gateway status
```

### 9.3 Reconnection Handling

**Device Reconnect:**
1. Device reconnects to MQTT
2. Device publishes new status
3. Gateway receives status, initiates new WebRTC session
4. Old PeerConnection cleaned up

**Gateway Reconnect:**
1. Gateway reconnects to MQTT
2. Gateway publishes online status
3. Gateway re-subscribes to device topics
4. Devices detect gateway via status, may re-publish their status

---

## 10. Vision Pipeline

### 10.1 Current Approach (Phase 1-2)

Vision remains on MQTT snapshots during initial audio-focused phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION (MQTT SNAPSHOTS)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Device                    AI Server                            │
│    │                           │                                │
│    │── MQTT: image ──────────▶│                                │
│    │   (base64 JPEG)          │── Decode ──▶ Image             │
│    │                          │── YOLO ───▶ Detections         │
│    │                          │── Alert ──▶ MQTT               │
│    │                          │                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Future Approach (Phase 3+)

Video streaming via Pi gateway (after audio is stable):

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION (WebRTC STREAMING)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera      Pi Gateway           AI Server                     │
│    │             │                     │                        │
│    │── WebRTC ─▶│                     │                        │
│    │   (VP8)     │── Decode frames     │                        │
│    │             │── Motion detect     │                        │
│    │             │                     │                        │
│    │             │── TCP: frame ─────▶│  (only if motion)      │
│    │             │   (JPEG)            │── YOLO ──▶ Detect     │
│    │             │                     │── Alert ─▶ MQTT       │
│    │             │                     │                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Vision Processing Notes

- Motion detection runs on Pi (cheap, reduces server load)
- Only motion frames sent to AI server
- YOLO runs on AI server (CPU intensive)
- Alerts published via MQTT to all subscribers

---

## 11. Media Input Methods

The system supports multiple ways to receive media, ensuring flexibility across device types and use cases.

### 11.1 Real-Time vs Asynchronous Input

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

### 11.2 Input Method Summary

| Method | Transport | Use Case | Latency |
|--------|-----------|----------|---------|
| **WebRTC Stream** | WebRTC (UDP) | Continuous monitoring, live conversation | Real-time |
| **MQTT Snapshot** | MQTT | On-demand capture from non-streaming devices | ~200-500ms |
| **MQTT Recording** | MQTT | Audio/video clips from devices | Async |
| **MQTT Upload** | MQTT | Web/app uploads (via web service) | Async |
| **Database Retrieval** | Internal | Historical media lookup | Async |

### 11.3 Media Input Architecture

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

### 11.4 Method 1: WebRTC Stream (Real-Time)

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

### 11.5 Method 2: MQTT Snapshot (On-Demand)

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

### 11.6 Method 3: MQTT Recording (Clips)

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

### 11.7 Method 4: Web/Mobile Upload (via Web Service)

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

### 11.8 Method 5: Database Retrieval (Historical)

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

### 11.9 MQTT Upload Format (from any source)

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

### 11.10 Media Source Selection Logic

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

### 11.11 Visual Context Format (Unified)

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

### 11.12 Visual Query Detection

Keywords that trigger visual context capture:

- **Direct:** "see", "look", "show", "camera", "picture", "photo", "image"
- **Spatial:** "here", "there", "this", "that"
- **Descriptive:** "describe", "what's", "what is", "how many", "is there"
- **Location:** "desk", "room", "door", "window", "outside", "screen"
- **Analysis:** "read", "OCR", "text", "recognize", "identify"

### 11.13 Audio Input Methods

Same flexibility applies to audio:

| Method | Use Case |
|--------|----------|
| WebRTC stream | Live conversation |
| MQTT recording | Voice message from device |
| HTTP upload | Audio file from web/app |
| MQTT upload | Audio clip via WebSocket |

All audio is normalized to PCM and processed through STT.

---

## 12. Implementation Phases

### Phase 1: Pi Gateway Setup
- Set up Raspberry Pi with aiortc
- Basic MQTT signaling implementation
- Single device WebRTC audio connection
- Audio passthrough test (device → Pi → device)

**Deliverable:** WebRTC audio streaming on Pi

### Phase 2: AI Server Integration
- TCP socket server on AI server
- Gateway-to-server audio forwarding
- Basic STT → LLM → TTS pipeline
- End-to-end voice test

**Deliverable:** Full voice conversation working

### Phase 3: Production Hardening
- VAD tuning and optimization
- Interruption handling
- Reconnection robustness
- Multi-device support on gateway
- Error handling and logging

**Deliverable:** Reliable multi-device voice

### Phase 4: Vision Integration (Future)
- Video track support on Pi gateway
- Motion detection on Pi
- Frame forwarding to AI server
- YOLO integration
- Alert generation

**Deliverable:** Video monitoring via WebRTC

### Phase Summary

| Phase | Focus | Location | Key Deliverable |
|-------|-------|----------|-----------------|
| 1 | Gateway Setup | Pi | WebRTC audio working |
| 2 | Server Integration | Both | Voice conversation |
| 3 | Hardening | Both | Multi-device reliable |
| 4 | Vision | Both | Video monitoring |

---

## 13. File Structure

### 13.1 Pi Gateway (New Repository)

```
pi-gateway/
├── main.py                 # Entry point
├── gateway/
│   ├── __init__.py
│   ├── audio_gateway.py    # Main gateway class
│   ├── session.py          # Device session management
│   ├── vad.py              # Voice activity detection
│   └── tracks.py           # Audio track implementations
├── mqtt/
│   ├── __init__.py
│   ├── client.py           # MQTT connection
│   └── signaling.py        # WebRTC signaling handlers
├── tcp/
│   ├── __init__.py
│   └── ai_client.py        # TCP client to AI server
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration
├── requirements.txt
└── README.md
```

### 13.2 AI Server Additions

| File | Purpose |
|------|---------|
| `server/audio_socket.py` | TCP server for gateway communication |
| `utils/pcm_utils.py` | PCM ↔ WAV conversion utilities |

### 13.3 AI Server Modifications

| File | Changes |
|------|---------|
| `server/lifecycle.py` | Start audio socket server |
| `config/__init__.py` | Add audio socket config |

---

## 14. Configuration

### 14.1 Configuration Classes

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

### 14.2 Environment Variables

**Pi Gateway:**
```bash
# MQTT
MQTT_HOST=main-server.local
MQTT_PORT=1883
MQTT_USERNAME=gateway
MQTT_PASSWORD=secret

# AI Server Connection
AI_SERVER_HOST=main-server.local
AI_SERVER_PORT=9999

# WebRTC
WEBRTC_ICE_SERVERS=[]  # Empty for local network

# Audio
AUDIO_VAD_MODE=energy  # or "silero"
AUDIO_VAD_THRESHOLD=500
AUDIO_SILENCE_FRAMES=15  # 300ms at 20ms/frame
AUDIO_SAMPLE_RATE=48000

# Logging
LOG_LEVEL=INFO
```

**AI Server:**
```bash
# Audio Socket
AUDIO_SOCKET_HOST=0.0.0.0
AUDIO_SOCKET_PORT=9999

# Existing configs unchanged
# ...
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

**Pi Gateway:**
- MQTT signaling handlers
- WebRTC PeerConnection lifecycle
- VAD accuracy
- Audio buffering logic
- TCP client connection

**AI Server:**
- Audio socket server
- PCM ↔ WAV conversion
- Existing service tests unchanged

### 15.2 Integration Tests

- Full signaling flow via MQTT
- Gateway ↔ AI server communication
- Voice conversation round trip
- Interruption handling
- Multi-device scenarios
- Reconnection after failures

### 15.3 End-to-End Tests

- Device → Gateway → Server → Gateway → Device
- Latency measurement
- Audio quality assessment
- Stress testing (multiple simultaneous conversations)

---

## 16. Performance Considerations

### 16.1 Latency Budget (Voice)

| Stage | Location | Target | Notes |
|-------|----------|--------|-------|
| Audio capture | Device | ~20ms | WebRTC frame |
| Network (WebRTC) | Device→Pi | ~30ms | WiFi |
| Opus decode + VAD | Pi | ~5ms | Software |
| Utterance buffer | Pi | ~100ms | Wait for silence |
| Network (TCP) | Pi→Server | ~2ms | Ethernet |
| STT (Whisper) | Server | ~200ms | CPU |
| LLM (Llama) | Server | ~300ms | CPU |
| TTS (Piper) | Server | ~100ms | CPU |
| Network (TCP) | Server→Pi | ~2ms | Ethernet |
| Opus encode | Pi | ~2ms | Software |
| Network (WebRTC) | Pi→Device | ~30ms | WiFi |
| **Total** | | **~800ms** | First audio |

### 16.2 CPU Distribution

**Pi Gateway (~30% CPU):**
- WebRTC: ~10% per connection
- Opus codec: ~5% per stream
- VAD: ~5%
- Headroom for spikes

**AI Server (up to 100% during inference):**
- Whisper: 100-200% (1-2 cores)
- Llama: 200-400% (2-4 cores)
- Piper: 50-100% (0.5-1 core)
- Inference is sequential, not concurrent

### 16.3 Memory Management

**Pi Gateway:**
- Audio buffers: ~2MB per device (30s max)
- PeerConnection overhead: ~10MB per device
- Total for 5 devices: ~60MB

**AI Server:**
- Model memory (existing)
- Audio socket buffers: ~1MB per request

---

## 17. Security Considerations

### 17.1 WebRTC Security

**Built-in:**
- DTLS for signaling encryption
- SRTP for media encryption
- Certificate fingerprint verification

**Configuration:**
- Use secure MQTT (TLS) for signaling
- Validate device IDs in signaling
- ICE candidate filtering (optional)

### 17.2 Network Security

**Pi ↔ Server:**
- Use wired ethernet (not WiFi) between Pi and server
- Consider TLS for TCP socket if on untrusted network
- Firewall: Only allow Pi IP to audio socket port

**MQTT:**
- TLS encryption
- Username/password or certificate auth
- ACLs per device (limit topic access)
- Rate limiting

### 17.3 General

- No media persistence by default
- Input validation on all messages
- Audit logging for security events

---

## 18. Hardware Requirements

### 18.1 Pi Gateway

**Minimum: Raspberry Pi 4 (2GB)**
- Handles 2-3 simultaneous devices
- Energy-based VAD only
- Basic functionality

**Recommended: Raspberry Pi 4 (4GB) or Pi 5 (4GB)**
- Handles 5+ simultaneous devices
- Silero VAD for better accuracy
- Headroom for video (future)
- Cost: ~$55-80

**Network:**
- Wired Ethernet to AI server (required)
- WiFi for device connections (or wired)

### 18.2 AI Server

**Minimum:**
- 4-core CPU (Whisper, Llama, Piper are CPU-bound)
- 16GB RAM (for models)
- SSD storage

**Recommended:**
- 8+ core CPU
- 32GB RAM
- GPU optional (improves inference speed)

### 18.3 Network Topology

```
┌─────────────────────────────────────────────────────────┐
│                    Recommended Setup                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Devices ───WiFi───▶ Router ◀───Ethernet───┐           │
│                        │                    │           │
│                        │              ┌─────▼─────┐     │
│                   Ethernet            │    Pi     │     │
│                        │              │  Gateway  │     │
│                        ▼              └─────┬─────┘     │
│                  ┌──────────┐               │           │
│                  │   MQTT   │          Ethernet         │
│                  │  Broker  │               │           │
│                  └────┬─────┘               │           │
│                       │              ┌──────▼──────┐    │
│                       └──────────────│  AI Server  │    │
│                                      └─────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 19. Migration Guide

### 19.1 From Current MQTT-Only Architecture

**Step 1: Set up Pi Gateway**
1. Install Raspberry Pi OS
2. Install Python 3.11+
3. Clone pi-gateway repository
4. Install dependencies
5. Configure MQTT and AI server addresses
6. Test MQTT connectivity

**Step 2: Add Audio Socket to AI Server**
1. Add `server/audio_socket.py`
2. Modify `server/lifecycle.py` to start socket server
3. Test TCP connectivity from Pi

**Step 3: Update Device Firmware**
1. Add WebRTC support to device
2. Implement MQTT signaling client
3. Test WebRTC connection to Pi gateway

**Step 4: End-to-End Testing**
1. Test full voice conversation
2. Tune VAD settings
3. Measure latency

### 19.2 Rollback Plan

1. Pi gateway can be bypassed
2. Devices fall back to MQTT audio (existing)
3. AI server continues to accept MQTT audio
4. No data loss or functionality regression

---

## Appendix A: Dependencies

### Pi Gateway

```
# requirements.txt for pi-gateway
aiortc>=1.6.0,<2.0.0    # WebRTC for Python
aioice>=0.9.0           # ICE implementation
av>=11.0.0,<12.0.0      # FFmpeg bindings (codec support)
numpy>=1.24.0           # Audio processing
asyncio-mqtt>=0.16.0    # Async MQTT client

# Optional for better VAD
silero-vad>=4.0.0
torch>=2.0.0            # Required by Silero
```

### AI Server Additions

```
# Add to existing requirements.txt
soundfile>=0.12.0       # If not already present (WAV handling)
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
