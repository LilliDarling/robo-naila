# Real-Time Conversational Audio Streaming Implementation Plan

## Overview

Transform the gateway into a full-duplex conversational audio system with:
- **Full streaming pipeline**: Audio → STT → LLM (token-by-token) → TTS (chunked) → Audio playback
- **Wake word activation** on devices before conversation starts
- **Barge-in/interruption** support for natural conversation flow
- **WebSocket** bidirectional streaming between gateway and AI server

## Architecture

```
Device (Pi #3)  ←──WebRTC──→  Gateway (Pi #2)  ←──WebSocket──→  AI Server
     │                              │                              │
 Wake Word                    State Machine                  Streaming Pipeline
 Detection                    VAD + Buffering                STT → LLM → TTS
```

### Conversation State Flow
```
IDLE → [wake word] → LISTENING → [silence] → PROCESSING → [first TTS] → SPEAKING → [complete/timeout] → IDLE
                          ↑                                     │
                          └────────── [barge-in] ───────────────┘
```

**States:**
- `IDLE` - Waiting for wake word, audio not streaming to server
- `LISTENING` - Active listening, streaming audio chunks to server
- `PROCESSING` - STT complete, LLM generating response
- `SPEAKING` - TTS audio playing back to user

---

## WebSocket Protocol

### Gateway → AI Server Messages

```python
# Audio chunk (streaming input)
{
    "type": "audio_chunk",
    "session_id": "sess_abc123",
    "device_id": "pi-mic-01",
    "timestamp": 1705600000.123,
    "payload": {
        "audio_b64": "<base64 PCM int16>",
        "sample_rate": 16000,
        "sequence": 42,
        "is_speech": true
    }
}

# Session control
{
    "type": "session_control",
    "session_id": "sess_abc123",
    "payload": {
        "action": "start" | "end" | "interrupt" | "cancel",
        "reason": "wake_word" | "vad_timeout" | "user_interrupt" | "error"
    }
}
```

### AI Server → Gateway Messages

```python
# STT partial result
{
    "type": "stt_partial",
    "session_id": "sess_abc123",
    "payload": {
        "text": "Hello, how are",
        "is_final": false,
        "confidence": 0.85
    }
}

# STT final result
{
    "type": "stt_final",
    "session_id": "sess_abc123",
    "payload": {
        "text": "Hello, how are you today?",
        "confidence": 0.92,
        "language": "en"
    }
}

# LLM token (streaming)
{
    "type": "llm_token",
    "session_id": "sess_abc123",
    "payload": {
        "token": "Hello",
        "is_first": true,
        "is_last": false
    }
}

# TTS audio chunk (streaming)
{
    "type": "tts_chunk",
    "session_id": "sess_abc123",
    "payload": {
        "audio_b64": "<base64 PCM int16>",
        "sample_rate": 22050,
        "sequence": 0,
        "is_last": false
    }
}

# State change notification
{
    "type": "state_change",
    "session_id": "sess_abc123",
    "payload": {
        "new_state": "speaking",
        "previous_state": "processing"
    }
}

# Error
{
    "type": "error",
    "session_id": "sess_abc123",
    "payload": {
        "code": "stt_failed",
        "message": "Transcription failed",
        "recoverable": true
    }
}
```

---

## Implementation Phases

### Phase 1: WebSocket Infrastructure

**AI Server - New Files:**
| File | Purpose |
|------|---------|
| `ai-server/streaming/__init__.py` | Module init |
| `ai-server/streaming/websocket_server.py` | FastAPI WebSocket endpoint |
| `ai-server/streaming/session.py` | Streaming session management |

**AI Server - Modify:**
| File | Changes |
|------|---------|
| `ai-server/main.py` | Add uvicorn/FastAPI alongside MQTT |
| `ai-server/server/naila_server.py` | Initialize streaming services |

**Gateway - New Files:**
| File | Purpose |
|------|---------|
| `gateway/pi-gateway/streaming/__init__.py` | Module init |
| `gateway/pi-gateway/streaming/ws_client.py` | WebSocket client to AI server |

**Gateway - Modify:**
| File | Changes |
|------|---------|
| `gateway/pi-gateway/main.py` | Replace TCP with WebSocket client |
| `gateway/pi-gateway/config/settings.py` | Add WebSocket URL config |

---

### Phase 2: Streaming Pipeline (AI Server)

**Modify `ai-server/services/llm.py`:**

Add streaming token generation:
```python
async def generate_chat_streaming(
    self,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Generate chat completion with streaming tokens."""
    prompt = self._format_chat_prompt(messages)

    for output in self.model(
        prompt,
        max_tokens=max_tokens or 256,
        stream=True,  # Enable streaming
        stop=["<|eot_id|>"]
    ):
        token = output["choices"][0]["text"]
        yield token
```

**Modify `ai-server/services/tts.py`:**

Add sentence-level streaming:
```python
async def synthesize_streaming(
    self,
    token_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[bytes, None]:
    """Synthesize TTS audio incrementally as LLM tokens arrive."""
    buffer = ""
    delimiters = ".!?"

    async for token in token_stream:
        buffer += token

        # Check for sentence boundary
        for delim in delimiters:
            if delim in buffer:
                sentence, buffer = buffer.split(delim, 1)
                sentence = sentence + delim

                if sentence.strip():
                    audio = await self.synthesize(sentence.strip(), output_format="raw")
                    yield audio.audio_bytes
                break

    # Handle remaining text
    if buffer.strip():
        audio = await self.synthesize(buffer.strip(), output_format="raw")
        yield audio.audio_bytes
```

**New `ai-server/streaming/orchestrator.py`:**

Pipeline coordinator:
```python
class StreamingOrchestrator:
    async def process_session(self, websocket, session):
        # 1. Accumulate audio until speech ends
        audio_data = await self._collect_audio(session)

        # 2. Transcribe (STT)
        transcription = await self.stt.transcribe_audio(audio_data)
        await self._send_stt_result(websocket, transcription)

        # 3. Stream LLM response
        messages = self._build_messages(session, transcription.text)
        full_text = ""

        async for token in self.llm.generate_chat_streaming(messages):
            if session.interrupted:
                break
            full_text += token
            await self._send_llm_token(websocket, token)

        # 4. Stream TTS audio
        async for audio_chunk in self.tts.synthesize_streaming(full_text):
            if session.interrupted:
                break
            await self._send_tts_chunk(websocket, audio_chunk)
```

**STT Approach:**
- Faster-Whisper doesn't support true streaming
- Accumulate audio chunks until VAD detects speech end
- Transcribe accumulated audio in batches
- Target: <500ms for typical utterances

---

### Phase 3: State Machine (Gateway)

**New `gateway/pi-gateway/streaming/state_machine.py`:**

```python
from enum import Enum, auto
from dataclasses import dataclass
import asyncio

class ConversationState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

@dataclass
class ConversationSession:
    session_id: str
    device_id: str
    state: ConversationState = ConversationState.IDLE
    state_entered_at: float = 0.0
    tts_playing: bool = False
    interrupt_requested: bool = False

class ConversationStateMachine:
    SILENCE_TIMEOUT = 10.0  # Return to IDLE after 10s silence
    CONTINUE_TIMEOUT = 3.0  # Wait 3s for user to continue after TTS

    async def handle_event(self, event: str, **kwargs) -> ConversationState:
        """Handle state transition based on event."""
        transitions = {
            ConversationState.IDLE: {
                "wake_word_detected": ConversationState.LISTENING,
            },
            ConversationState.LISTENING: {
                "vad_speech_end": ConversationState.PROCESSING,
                "timeout": ConversationState.IDLE,
            },
            ConversationState.PROCESSING: {
                "first_tts_chunk": ConversationState.SPEAKING,
                "error": ConversationState.IDLE,
            },
            ConversationState.SPEAKING: {
                "user_interrupt": ConversationState.LISTENING,
                "tts_complete": ConversationState.LISTENING,
                "silence_timeout": ConversationState.IDLE,
            },
        }

        current = self.session.state
        new_state = transitions.get(current, {}).get(event, current)

        if new_state != current:
            self.session.state = new_state
            self.session.state_entered_at = time.time()

        return new_state
```

**Modify `gateway/pi-gateway/gateway/session.py`:**

Add state and streaming fields:
```python
@dataclass
class DeviceSession:
    device_id: str
    pc: RTCPeerConnection

    # New streaming fields
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: ConversationState = ConversationState.IDLE
    audio_sequence: int = 0

    # Existing fields...
```

**Modify `gateway/pi-gateway/gateway/audio_gateway.py`:**

Integrate streaming:
```python
async def _handle_audio_track(self, session: DeviceSession, track):
    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray().flatten().astype(np.int16)

        # Check state before processing
        if session.state == ConversationState.IDLE:
            continue  # Ignore audio when idle (waiting for wake word)

        # Check for barge-in
        is_speech = self.vad.is_speech(pcm)
        if is_speech and session.tts_playing:
            await self._handle_interruption(session)
            continue

        # Stream to AI server
        if session.state == ConversationState.LISTENING:
            await self.ws_client.send_audio_chunk(
                session_id=session.session_id,
                device_id=session.device_id,
                audio_pcm=pcm.tobytes(),
                sample_rate=16000,
                sequence=session.audio_sequence,
                is_speech=is_speech
            )
            session.audio_sequence += 1
```

---

### Phase 4: Barge-In Support

**New `gateway/pi-gateway/streaming/interruption.py`:**

```python
class InterruptionDetector:
    MIN_SPEECH_FRAMES = 3  # ~60ms to trigger

    def __init__(self, vad):
        self.vad = vad
        self.consecutive_speech = 0
        self.tts_playing = False

    def check(self, audio_frame: np.ndarray) -> bool:
        """Returns True if user is interrupting."""
        if not self.tts_playing:
            return False

        if self.vad.is_speech(audio_frame):
            self.consecutive_speech += 1
            return self.consecutive_speech >= self.MIN_SPEECH_FRAMES
        else:
            self.consecutive_speech = 0
            return False
```

**Interruption handling in gateway:**
```python
async def _handle_interruption(self, session: DeviceSession):
    logger.info(f"[{session.device_id}] Barge-in detected")

    # Clear TTS queue
    session.clear_tts_queue()
    session.tts_playing = False

    # Notify AI server
    await self.ws_client.send_session_control(
        session.session_id,
        action="interrupt",
        reason="user_interrupt"
    )

    # Transition to listening
    session.state = ConversationState.LISTENING
```

---

### Phase 5: Wake Word Integration

**New `gateway/pi-device/wake_word.py`:**

```python
import pvporcupine

class WakeWordDetector:
    def __init__(self, access_key: str, keywords: list = None):
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=keywords or ["jarvis"],
            sensitivities=[0.7]
        )
        self.on_wake_word = None

    def process_frame(self, pcm_frame: bytes) -> bool:
        """Returns True if wake word detected."""
        samples = struct.unpack_from("h" * self.porcupine.frame_length, pcm_frame)
        keyword_index = self.porcupine.process(samples)

        if keyword_index >= 0:
            if self.on_wake_word:
                self.on_wake_word(keyword_index)
            return True
        return False
```

**Modify `gateway/pi-device/main.py`:**

```python
class DeviceClient:
    def __init__(self):
        self.wake_word = WakeWordDetector(
            access_key=settings.porcupine_access_key,
            keywords=["naila"]
        )
        self.wake_word.on_wake_word = self._on_wake_word
        self.conversation_active = False

    async def _on_wake_word(self, keyword_index: int):
        logger.info("Wake word detected!")
        self.conversation_active = True

        # Notify gateway
        await self.mqtt.publish(
            f"{self.prefix}/devices/{self.device_id}/wake_word",
            json.dumps({"device_id": self.device_id, "timestamp": time.time()})
        )
```

---

## File Summary

### Files to Create

| File | Purpose |
|------|---------|
| `ai-server/streaming/__init__.py` | Module init |
| `ai-server/streaming/websocket_server.py` | WebSocket handler |
| `ai-server/streaming/orchestrator.py` | STT→LLM→TTS pipeline |
| `ai-server/streaming/session.py` | Streaming session state |
| `gateway/pi-gateway/streaming/__init__.py` | Module init |
| `gateway/pi-gateway/streaming/ws_client.py` | WebSocket client |
| `gateway/pi-gateway/streaming/state_machine.py` | Conversation state |
| `gateway/pi-gateway/streaming/interruption.py` | Barge-in detection |
| `gateway/pi-device/wake_word.py` | Porcupine wake word |

### Files to Modify

| File | Changes |
|------|---------|
| `ai-server/main.py` | Add FastAPI app with WebSocket endpoint |
| `ai-server/server/naila_server.py` | Initialize streaming services |
| `ai-server/services/llm.py` | Add `generate_chat_streaming()` |
| `ai-server/services/tts.py` | Add `synthesize_streaming()` |
| `gateway/pi-gateway/main.py` | Replace TCP with WebSocket |
| `gateway/pi-gateway/gateway/audio_gateway.py` | Streaming dispatch, state integration |
| `gateway/pi-gateway/gateway/session.py` | Add state, session_id |
| `gateway/pi-gateway/gateway/tracks.py` | TTS completion signaling |
| `gateway/pi-device/main.py` | Wake word integration |
| `gateway/pi-device/config.py` | Wake word settings |

### Files to Deprecate

| File | Reason |
|------|--------|
| `gateway/pi-gateway/tcp/ai_client.py` | Replaced by WebSocket client |

---

## Dependencies

**AI Server (`ai-server/pyproject.toml`):**
```toml
fastapi = ">=0.109.0"
uvicorn = ">=0.27.0"
websockets = ">=12.0"
```

**Gateway (`gateway/pi-gateway/requirements.txt`):**
```
websockets>=12.0
```

**Device (`gateway/pi-device/requirements.txt`):**
```
pvporcupine>=3.0.0
```

---

## Latency Targets

| Stage | Target |
|-------|--------|
| Wake word detection | <100ms |
| Audio buffering | 50-100ms |
| STT processing | 200-500ms |
| LLM first token | 100-300ms |
| TTS first chunk | 100-200ms |
| **End-to-end** | **500-1200ms** |

---

## Testing Checklist

- [ ] WebSocket connection established and maintained
- [ ] Audio chunks stream from gateway to AI server
- [ ] STT partial/final results received
- [ ] LLM tokens stream incrementally
- [ ] TTS audio chunks play with low latency
- [ ] State transitions work correctly
- [ ] Barge-in stops TTS and resumes listening
- [ ] Wake word activates conversation
- [ ] Conversation ends after timeout
- [ ] Reconnection after disconnect
- [ ] Full conversation loop feels natural
