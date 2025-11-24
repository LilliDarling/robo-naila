# Real-Time Streaming Implementation Guide

## NAILA AI Server - Voice & Vision Streaming Architecture

**Version:** 1.0
**Date:** November 2024
**Status:** Implementation Specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Target Architecture](#3-target-architecture)
4. [WebSocket Infrastructure](#4-websocket-infrastructure)
5. [Voice Streaming Pipeline](#5-voice-streaming-pipeline)
6. [Vision Streaming Pipeline](#6-vision-streaming-pipeline)
7. [Protocol Specifications](#7-protocol-specifications)
8. [Implementation Phases](#8-implementation-phases)
9. [File-by-File Changes](#9-file-by-file-changes)
10. [Configuration](#10-configuration)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Considerations](#12-performance-considerations)
13. [Security Considerations](#13-security-considerations)
14. [Migration Guide](#14-migration-guide)

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies the implementation of real-time streaming capabilities for the NAILA AI Server, enabling:

1. **Seamless Voice Conversation** - Low-latency, interruptible voice interaction similar to modern voice assistants (Alexa, Google Home, ChatGPT Voice)
2. **Continuous Security Monitoring** - Real-time video analysis from security cameras with motion detection and alerting

### 1.2 Key Goals

| Goal | Current State | Target State |
|------|---------------|--------------|
| Voice response latency | 2-5 seconds | 200-500ms |
| Conversation interruption | Not supported | Full support |
| Partial transcription | Not available | Real-time display |
| Camera monitoring | On-demand only | Continuous streaming |
| Security alerts | None | Zone-based, intelligent |

### 1.3 Architecture Decision

**Hybrid Protocol Approach:**
- **WebSocket** for continuous bidirectional streams (audio, video)
- **MQTT** retained for commands, events, and multi-device coordination

This approach leverages the strengths of each protocol while maintaining backward compatibility.

---

## 2. Current Architecture Analysis

### 2.1 Existing Communication Flow

```
┌──────────────┐     MQTT (batch)      ┌──────────────┐
│    Device    │ ───────────────────── │  AI Server   │
│              │                        │              │
│  • Captures  │   Complete audio/      │  • STT       │
│    audio     │   image sent as        │  • Vision    │
│  • Captures  │   single message       │  • LLM       │
│    video     │                        │  • TTS       │
└──────────────┘                        └──────────────┘
```

### 2.2 Current Limitations

| Component | Limitation | Impact |
|-----------|------------|--------|
| Audio Input | Complete utterance required before processing | High latency, no interruption |
| Audio Output | Complete TTS file generated before playback | Delayed response start |
| Video Input | Single frames on-demand | No continuous monitoring |
| Data Encoding | Base64 over JSON | 33% size overhead |
| Buffering | Task queue (max 1000) drops data | Lost frames under load |

### 2.3 Relevant Existing Files

| File | Purpose | Streaming Relevance |
|------|---------|---------------------|
| `server/naila_server.py` | Server initialization | Add WebSocket server |
| `server/lifecycle.py` | Startup/shutdown stages | Add streaming stages |
| `services/stt.py` | Speech-to-text | Add streaming transcription |
| `services/tts.py` | Text-to-speech | Add streaming synthesis |
| `services/vision.py` | Object detection | Add continuous monitoring |
| `mqtt/handlers/device_handlers.py` | Device message handling | Coordinate with streams |
| `memory/conversation.py` | Conversation state | Add interruption state |
| `config/stt.py` | STT configuration | VAD already configured |

---

## 3. Target Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           NAILA AI Server                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────┐              ┌─────────────────────┐          │
│   │   WebSocket Server  │              │    MQTT Broker      │          │
│   │   Port: 8765        │              │    Port: 1883       │          │
│   │                     │              │                     │          │
│   │   • Audio streams   │              │   • Commands        │          │
│   │   • Video streams   │              │   • Events/Alerts   │          │
│   │   • TTS output      │              │   • Device status   │          │
│   │   • Control msgs    │              │   • System health   │          │
│   └──────────┬──────────┘              └──────────┬──────────┘          │
│              │                                    │                      │
│              └────────────┬───────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│              ┌─────────────────────────┐                                 │
│              │    Stream Manager       │                                 │
│              │                         │                                 │
│              │  • Device sessions      │                                 │
│              │  • Stream routing       │                                 │
│              │  • State coordination   │                                 │
│              └────────────┬────────────┘                                 │
│                           │                                              │
│         ┌─────────────────┼─────────────────┐                           │
│         │                 │                 │                           │
│         ▼                 ▼                 ▼                           │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐                       │
│   │  Voice    │    │  Vision   │    │  Response │                       │
│   │  Pipeline │    │  Pipeline │    │  Pipeline │                       │
│   │           │    │           │    │           │                       │
│   │ • VAD     │    │ • Motion  │    │ • LLM     │                       │
│   │ • STT     │    │ • YOLO    │    │ • TTS     │                       │
│   │ • Buffer  │    │ • Alerts  │    │ • Stream  │                       │
│   └───────────┘    └───────────┘    └───────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow - Voice Conversation

```
┌─────────┐   WebSocket    ┌─────────┐   Internal   ┌─────────┐
│ Device  │ ─────────────► │  Voice  │ ───────────► │   LLM   │
│   Mic   │  Audio chunks  │ Pipeline│  Transcript  │ Service │
└─────────┘   (20-100ms)   └─────────┘              └────┬────┘
                                                         │
                                                         │ Response
                                                         ▼
┌─────────┐   WebSocket    ┌─────────┐   Internal   ┌─────────┐
│ Device  │ ◄───────────── │Response │ ◄─────────── │   TTS   │
│ Speaker │  Audio chunks  │ Pipeline│   Text       │ Service │
└─────────┘   (streaming)  └─────────┘              └─────────┘
```

### 3.3 Data Flow - Security Monitoring

```
┌─────────┐   WebSocket    ┌─────────┐
│ Camera  │ ─────────────► │ Vision  │
│         │  Video frames  │ Pipeline│
└─────────┘   (1-30 FPS)   └────┬────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
             ┌───────────┐          ┌───────────┐
             │  Motion   │          │  Object   │
             │ Detection │          │ Detection │
             └─────┬─────┘          └─────┬─────┘
                   │                      │
                   └──────────┬───────────┘
                              │
                              ▼
                       ┌───────────┐      MQTT        ┌─────────┐
                       │  Alert    │ ───────────────► │ Clients │
                       │  System   │   Events         │         │
                       └───────────┘                  └─────────┘
```

---

## 4. WebSocket Infrastructure

### 4.1 New File: `streaming/__init__.py`

```python
"""
Real-time streaming module for NAILA AI Server.

This module provides WebSocket-based streaming capabilities for:
- Audio streaming (voice conversation)
- Video streaming (security monitoring)
- Response streaming (TTS output)
"""

from .websocket_server import WebSocketServer
from .stream_manager import StreamManager
from .audio_stream import AudioStreamPipeline
from .video_stream import VideoStreamPipeline
from .protocols import StreamMessage, AudioChunk, VideoFrame

__all__ = [
    "WebSocketServer",
    "StreamManager",
    "AudioStreamPipeline",
    "VideoStreamPipeline",
    "StreamMessage",
    "AudioChunk",
    "VideoFrame",
]
```

### 4.2 New File: `streaming/websocket_server.py`

```python
"""
WebSocket server for real-time streaming connections.

Handles:
- Device authentication and session management
- Routing streams to appropriate pipelines
- Connection lifecycle and reconnection
- Binary and JSON message handling
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Callable, Any
from weakref import WeakValueDictionary

import websockets
from websockets.server import WebSocketServerProtocol

from utils import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    STREAMING = "streaming"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class DeviceSession:
    """Represents an active device streaming session"""
    device_id: str
    websocket: WebSocketServerProtocol
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Stream states
    audio_streaming: bool = False
    video_streaming: bool = False

    # Metrics
    audio_chunks_received: int = 0
    video_frames_received: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


class WebSocketServer:
    """
    WebSocket server for real-time device streaming.

    Features:
    - Device authentication via token or device_id
    - Automatic reconnection handling
    - Stream routing to audio/video pipelines
    - Graceful shutdown with drain
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        stream_manager: Optional["StreamManager"] = None,
        auth_callback: Optional[Callable[[str, str], bool]] = None,
    ):
        self.host = host
        self.port = port
        self.stream_manager = stream_manager
        self.auth_callback = auth_callback or self._default_auth

        # Active sessions by device_id
        self._sessions: Dict[str, DeviceSession] = {}

        # Server state
        self._server: Optional[websockets.WebSocketServer] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Message handlers by type
        self._handlers: Dict[str, Callable] = {
            "auth": self._handle_auth,
            "audio_chunk": self._handle_audio_chunk,
            "video_frame": self._handle_video_frame,
            "control": self._handle_control,
            "ping": self._handle_ping,
        }

    async def start(self):
        """Start the WebSocket server"""
        if self._running:
            logger.warning("websocket_server_already_running")
            return

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB max message size
            compression=None,  # Disable compression for low latency
        )

        self._running = True
        logger.info(
            "websocket_server_started",
            host=self.host,
            port=self.port
        )

    async def stop(self):
        """Stop the WebSocket server gracefully"""
        if not self._running:
            return

        logger.info("websocket_server_stopping")
        self._running = False
        self._shutdown_event.set()

        # Close all active sessions
        close_tasks = []
        for session in list(self._sessions.values()):
            close_tasks.append(self._close_session(session, "server_shutdown"))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Close the server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("websocket_server_stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection"""
        device_id = None
        session = None

        try:
            # Create temporary session
            session = DeviceSession(
                device_id="pending",
                websocket=websocket,
                state=ConnectionState.AUTHENTICATING
            )

            logger.info(
                "websocket_connection_received",
                remote=websocket.remote_address,
                path=path
            )

            # Wait for authentication (5 second timeout)
            try:
                auth_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5.0
                )
                auth_data = json.loads(auth_message)

                if auth_data.get("type") != "auth":
                    await self._send_error(websocket, "first_message_must_be_auth")
                    return

                device_id = await self._handle_auth(session, auth_data)
                if not device_id:
                    return  # Auth failed, connection closed

            except asyncio.TimeoutError:
                await self._send_error(websocket, "auth_timeout")
                return

            # Update session with authenticated device_id
            session.device_id = device_id
            session.state = ConnectionState.AUTHENTICATED
            self._sessions[device_id] = session

            logger.info(
                "websocket_device_authenticated",
                device_id=device_id
            )

            # Notify stream manager of new connection
            if self.stream_manager:
                await self.stream_manager.on_device_connected(device_id, session)

            # Main message loop
            await self._message_loop(session)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(
                "websocket_connection_closed",
                device_id=device_id,
                code=e.code,
                reason=e.reason
            )
        except Exception as e:
            logger.error(
                "websocket_connection_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            # Cleanup session
            if device_id and device_id in self._sessions:
                del self._sessions[device_id]

                if self.stream_manager:
                    await self.stream_manager.on_device_disconnected(device_id)

                logger.info(
                    "websocket_session_cleaned_up",
                    device_id=device_id
                )

    async def _message_loop(self, session: DeviceSession):
        """Process messages from a connected device"""
        session.state = ConnectionState.STREAMING

        async for message in session.websocket:
            if not self._running:
                break

            session.update_activity()

            try:
                # Handle binary messages (audio/video data)
                if isinstance(message, bytes):
                    await self._handle_binary_message(session, message)
                    session.bytes_received += len(message)
                else:
                    # JSON messages
                    data = json.loads(message)
                    session.bytes_received += len(message)

                    msg_type = data.get("type")
                    handler = self._handlers.get(msg_type)

                    if handler:
                        await handler(session, data)
                    else:
                        logger.warning(
                            "websocket_unknown_message_type",
                            device_id=session.device_id,
                            type=msg_type
                        )

            except json.JSONDecodeError as e:
                logger.warning(
                    "websocket_invalid_json",
                    device_id=session.device_id,
                    error=str(e)
                )
            except Exception as e:
                logger.error(
                    "websocket_message_handling_error",
                    device_id=session.device_id,
                    error=str(e),
                    error_type=type(e).__name__
                )

    async def _handle_binary_message(self, session: DeviceSession, data: bytes):
        """
        Handle binary message (raw audio/video data).

        Binary format:
        - First byte: message type (0x01 = audio, 0x02 = video)
        - Next 4 bytes: sequence number (big endian)
        - Next 8 bytes: timestamp ms (big endian)
        - Remaining: payload data
        """
        if len(data) < 13:
            logger.warning("websocket_binary_message_too_short", device_id=session.device_id)
            return

        msg_type = data[0]
        sequence = int.from_bytes(data[1:5], "big")
        timestamp = int.from_bytes(data[5:13], "big")
        payload = data[13:]

        if msg_type == 0x01:  # Audio
            session.audio_chunks_received += 1
            if self.stream_manager:
                await self.stream_manager.process_audio_chunk(
                    session.device_id,
                    payload,
                    sequence,
                    timestamp
                )
        elif msg_type == 0x02:  # Video
            session.video_frames_received += 1
            if self.stream_manager:
                await self.stream_manager.process_video_frame(
                    session.device_id,
                    payload,
                    sequence,
                    timestamp
                )
        else:
            logger.warning(
                "websocket_unknown_binary_type",
                device_id=session.device_id,
                type=msg_type
            )

    async def _handle_auth(
        self,
        session: DeviceSession,
        data: dict
    ) -> Optional[str]:
        """Handle authentication message"""
        device_id = data.get("device_id")
        token = data.get("token", "")

        if not device_id:
            await self._send_error(session.websocket, "device_id_required")
            return None

        # Check if device already connected
        if device_id in self._sessions:
            old_session = self._sessions[device_id]
            logger.info(
                "websocket_replacing_session",
                device_id=device_id
            )
            await self._close_session(old_session, "replaced_by_new_connection")

        # Authenticate
        if not self.auth_callback(device_id, token):
            await self._send_error(session.websocket, "auth_failed")
            return None

        # Send auth success
        await self._send_message(session.websocket, {
            "type": "auth_success",
            "device_id": device_id,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "capabilities": {
                "audio_streaming": True,
                "video_streaming": True,
                "binary_protocol": True,
            }
        })

        return device_id

    async def _handle_audio_chunk(self, session: DeviceSession, data: dict):
        """Handle JSON-formatted audio chunk"""
        session.audio_chunks_received += 1

        if self.stream_manager:
            import base64
            audio_data = base64.b64decode(data.get("data", ""))
            await self.stream_manager.process_audio_chunk(
                session.device_id,
                audio_data,
                data.get("sequence", 0),
                data.get("timestamp", 0)
            )

    async def _handle_video_frame(self, session: DeviceSession, data: dict):
        """Handle JSON-formatted video frame"""
        session.video_frames_received += 1

        if self.stream_manager:
            import base64
            frame_data = base64.b64decode(data.get("data", ""))
            await self.stream_manager.process_video_frame(
                session.device_id,
                frame_data,
                data.get("sequence", 0),
                data.get("timestamp", 0)
            )

    async def _handle_control(self, session: DeviceSession, data: dict):
        """Handle control messages"""
        action = data.get("action")

        if action == "start_audio":
            session.audio_streaming = True
            if self.stream_manager:
                await self.stream_manager.start_audio_stream(session.device_id)

        elif action == "stop_audio":
            session.audio_streaming = False
            if self.stream_manager:
                await self.stream_manager.stop_audio_stream(session.device_id)

        elif action == "start_video":
            session.video_streaming = True
            if self.stream_manager:
                await self.stream_manager.start_video_stream(session.device_id)

        elif action == "stop_video":
            session.video_streaming = False
            if self.stream_manager:
                await self.stream_manager.stop_video_stream(session.device_id)

        logger.info(
            "websocket_control_action",
            device_id=session.device_id,
            action=action
        )

    async def _handle_ping(self, session: DeviceSession, data: dict):
        """Handle ping message"""
        await self._send_message(session.websocket, {
            "type": "pong",
            "timestamp": data.get("timestamp"),
            "server_time": datetime.now(timezone.utc).isoformat()
        })

    async def _close_session(self, session: DeviceSession, reason: str):
        """Close a device session"""
        session.state = ConnectionState.CLOSING

        try:
            await self._send_message(session.websocket, {
                "type": "close",
                "reason": reason
            })
            await session.websocket.close(1000, reason)
        except Exception:
            pass  # Ignore errors during close

        session.state = ConnectionState.CLOSED

    async def _send_message(self, websocket: WebSocketServerProtocol, data: dict):
        """Send a JSON message"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(
                "websocket_send_error",
                error=str(e),
                error_type=type(e).__name__
            )

    async def _send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send an error message"""
        await self._send_message(websocket, {
            "type": "error",
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def send_to_device(self, device_id: str, data: dict) -> bool:
        """Send a message to a specific device"""
        session = self._sessions.get(device_id)
        if not session:
            return False

        await self._send_message(session.websocket, data)
        session.bytes_sent += len(json.dumps(data))
        return True

    async def send_binary_to_device(self, device_id: str, data: bytes) -> bool:
        """Send binary data to a specific device"""
        session = self._sessions.get(device_id)
        if not session:
            return False

        try:
            await session.websocket.send(data)
            session.bytes_sent += len(data)
            return True
        except Exception as e:
            logger.error(
                "websocket_binary_send_error",
                device_id=device_id,
                error=str(e)
            )
            return False

    def _default_auth(self, device_id: str, token: str) -> bool:
        """Default authentication (accepts all)"""
        # In production, implement proper token validation
        return bool(device_id)

    def get_session(self, device_id: str) -> Optional[DeviceSession]:
        """Get session for a device"""
        return self._sessions.get(device_id)

    def get_connected_devices(self) -> list[str]:
        """Get list of connected device IDs"""
        return list(self._sessions.keys())

    def get_stats(self) -> dict:
        """Get server statistics"""
        total_audio = sum(s.audio_chunks_received for s in self._sessions.values())
        total_video = sum(s.video_frames_received for s in self._sessions.values())
        total_bytes_in = sum(s.bytes_received for s in self._sessions.values())
        total_bytes_out = sum(s.bytes_sent for s in self._sessions.values())

        return {
            "connected_devices": len(self._sessions),
            "total_audio_chunks": total_audio,
            "total_video_frames": total_video,
            "total_bytes_received": total_bytes_in,
            "total_bytes_sent": total_bytes_out,
        }
```

### 4.3 New File: `streaming/stream_manager.py`

```python
"""
Stream Manager - Coordinates all streaming pipelines.

Responsibilities:
- Device session lifecycle
- Routing data to appropriate pipelines
- Cross-pipeline coordination (e.g., interruption)
- State synchronization with MQTT handlers
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from enum import Enum

from utils import get_logger

logger = get_logger(__name__)


class StreamType(Enum):
    """Types of streams"""
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class DeviceStreamState:
    """Tracks streaming state for a device"""
    device_id: str

    # Audio state
    audio_active: bool = False
    audio_pipeline: Optional["AudioStreamPipeline"] = None

    # Video state
    video_active: bool = False
    video_pipeline: Optional["VideoStreamPipeline"] = None

    # Conversation state
    is_speaking: bool = False  # User is speaking
    is_responding: bool = False  # System is responding (TTS playing)

    # Timestamps
    last_audio_chunk: Optional[datetime] = None
    last_video_frame: Optional[datetime] = None

    # Metrics
    audio_chunks_processed: int = 0
    video_frames_processed: int = 0


class StreamManager:
    """
    Central coordinator for all streaming pipelines.

    Manages:
    - Device streaming sessions
    - Audio pipeline routing
    - Video pipeline routing
    - Cross-pipeline coordination
    - Integration with MQTT for events
    """

    def __init__(
        self,
        stt_service: Optional[Any] = None,
        tts_service: Optional[Any] = None,
        vision_service: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        mqtt_service: Optional[Any] = None,
    ):
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.vision_service = vision_service
        self.llm_service = llm_service
        self.mqtt_service = mqtt_service

        # Device states
        self._device_states: Dict[str, DeviceStreamState] = {}

        # WebSocket server reference (set after initialization)
        self._websocket_server: Optional[Any] = None

        # Event callbacks
        self._on_transcription_callbacks: list[Callable] = []
        self._on_alert_callbacks: list[Callable] = []

        # Pipeline factories
        self._audio_pipeline_factory: Optional[Callable] = None
        self._video_pipeline_factory: Optional[Callable] = None

    def set_websocket_server(self, server: Any):
        """Set WebSocket server reference"""
        self._websocket_server = server

    def set_audio_pipeline_factory(self, factory: Callable):
        """Set factory for creating audio pipelines"""
        self._audio_pipeline_factory = factory

    def set_video_pipeline_factory(self, factory: Callable):
        """Set factory for creating video pipelines"""
        self._video_pipeline_factory = factory

    # ==================== Device Lifecycle ====================

    async def on_device_connected(self, device_id: str, session: Any):
        """Handle new device connection"""
        state = DeviceStreamState(device_id=device_id)
        self._device_states[device_id] = state

        logger.info(
            "stream_manager_device_connected",
            device_id=device_id
        )

        # Notify via MQTT
        if self.mqtt_service:
            self.mqtt_service.publish_system_message(
                "devices",
                "connected",
                {
                    "device_id": device_id,
                    "connection_type": "websocket",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

    async def on_device_disconnected(self, device_id: str):
        """Handle device disconnection"""
        state = self._device_states.pop(device_id, None)

        if state:
            # Stop any active pipelines
            if state.audio_pipeline:
                await state.audio_pipeline.stop()
            if state.video_pipeline:
                await state.video_pipeline.stop()

        logger.info(
            "stream_manager_device_disconnected",
            device_id=device_id
        )

        # Notify via MQTT
        if self.mqtt_service:
            self.mqtt_service.publish_system_message(
                "devices",
                "disconnected",
                {
                    "device_id": device_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

    # ==================== Audio Streaming ====================

    async def start_audio_stream(self, device_id: str):
        """Start audio streaming for a device"""
        state = self._device_states.get(device_id)
        if not state:
            logger.warning(
                "stream_manager_unknown_device",
                device_id=device_id,
                action="start_audio"
            )
            return

        if state.audio_active:
            logger.debug(
                "stream_manager_audio_already_active",
                device_id=device_id
            )
            return

        # Create audio pipeline
        if self._audio_pipeline_factory:
            state.audio_pipeline = self._audio_pipeline_factory(
                device_id=device_id,
                stt_service=self.stt_service,
                on_transcription=lambda t: self._handle_transcription(device_id, t),
                on_speech_start=lambda: self._handle_speech_start(device_id),
                on_speech_end=lambda: self._handle_speech_end(device_id),
            )
            await state.audio_pipeline.start()

        state.audio_active = True

        logger.info(
            "stream_manager_audio_started",
            device_id=device_id
        )

    async def stop_audio_stream(self, device_id: str):
        """Stop audio streaming for a device"""
        state = self._device_states.get(device_id)
        if not state or not state.audio_active:
            return

        if state.audio_pipeline:
            await state.audio_pipeline.stop()
            state.audio_pipeline = None

        state.audio_active = False

        logger.info(
            "stream_manager_audio_stopped",
            device_id=device_id
        )

    async def process_audio_chunk(
        self,
        device_id: str,
        audio_data: bytes,
        sequence: int,
        timestamp: int
    ):
        """Process incoming audio chunk"""
        state = self._device_states.get(device_id)
        if not state:
            return

        state.last_audio_chunk = datetime.now(timezone.utc)
        state.audio_chunks_processed += 1

        # Check for interruption (user speaking while system responding)
        if state.is_responding:
            await self._handle_interruption(device_id)

        # Route to audio pipeline
        if state.audio_pipeline:
            await state.audio_pipeline.process_chunk(audio_data, sequence, timestamp)

    # ==================== Video Streaming ====================

    async def start_video_stream(self, device_id: str):
        """Start video streaming for a device"""
        state = self._device_states.get(device_id)
        if not state:
            logger.warning(
                "stream_manager_unknown_device",
                device_id=device_id,
                action="start_video"
            )
            return

        if state.video_active:
            return

        # Create video pipeline
        if self._video_pipeline_factory:
            state.video_pipeline = self._video_pipeline_factory(
                device_id=device_id,
                vision_service=self.vision_service,
                on_alert=lambda a: self._handle_vision_alert(device_id, a),
            )
            await state.video_pipeline.start()

        state.video_active = True

        logger.info(
            "stream_manager_video_started",
            device_id=device_id
        )

    async def stop_video_stream(self, device_id: str):
        """Stop video streaming for a device"""
        state = self._device_states.get(device_id)
        if not state or not state.video_active:
            return

        if state.video_pipeline:
            await state.video_pipeline.stop()
            state.video_pipeline = None

        state.video_active = False

        logger.info(
            "stream_manager_video_stopped",
            device_id=device_id
        )

    async def process_video_frame(
        self,
        device_id: str,
        frame_data: bytes,
        sequence: int,
        timestamp: int
    ):
        """Process incoming video frame"""
        state = self._device_states.get(device_id)
        if not state:
            return

        state.last_video_frame = datetime.now(timezone.utc)
        state.video_frames_processed += 1

        # Route to video pipeline
        if state.video_pipeline:
            await state.video_pipeline.process_frame(frame_data, sequence, timestamp)

    # ==================== Response Streaming ====================

    async def send_tts_response(
        self,
        device_id: str,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None
    ):
        """
        Generate and stream TTS response to device.

        Uses streaming TTS to begin playback before full generation.
        """
        state = self._device_states.get(device_id)
        if not state:
            logger.warning(
                "stream_manager_unknown_device",
                device_id=device_id,
                action="send_tts"
            )
            return

        state.is_responding = True

        try:
            if self.tts_service and hasattr(self.tts_service, 'synthesize_stream'):
                # Streaming TTS
                sequence = 0
                async for audio_chunk in self.tts_service.synthesize_stream(
                    text,
                    voice=voice,
                    emotion=emotion
                ):
                    # Check for interruption
                    if not state.is_responding:
                        logger.info(
                            "stream_manager_tts_interrupted",
                            device_id=device_id
                        )
                        break

                    # Send chunk to device
                    await self._send_tts_chunk(device_id, audio_chunk, sequence)
                    sequence += 1

                # Send end marker
                await self._send_tts_end(device_id)

            elif self.tts_service:
                # Fallback: non-streaming TTS
                audio_data = await self.tts_service.synthesize(
                    text,
                    voice=voice,
                    emotion=emotion
                )
                await self._send_tts_chunk(device_id, audio_data.audio_bytes, 0, is_final=True)

        except Exception as e:
            logger.error(
                "stream_manager_tts_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            state.is_responding = False

    async def _send_tts_chunk(
        self,
        device_id: str,
        audio_data: bytes,
        sequence: int,
        is_final: bool = False
    ):
        """Send TTS audio chunk to device"""
        if self._websocket_server:
            # Binary format: type(1) + sequence(4) + is_final(1) + data
            header = bytes([0x03])  # 0x03 = TTS audio
            seq_bytes = sequence.to_bytes(4, "big")
            final_byte = bytes([1 if is_final else 0])

            message = header + seq_bytes + final_byte + audio_data
            await self._websocket_server.send_binary_to_device(device_id, message)

    async def _send_tts_end(self, device_id: str):
        """Send TTS end marker"""
        if self._websocket_server:
            await self._websocket_server.send_to_device(device_id, {
                "type": "tts_complete",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    # ==================== Event Handlers ====================

    async def _handle_transcription(self, device_id: str, transcription: dict):
        """Handle transcription result from audio pipeline"""
        logger.info(
            "stream_manager_transcription",
            device_id=device_id,
            text=transcription.get("text", "")[:50],
            is_final=transcription.get("is_final", False)
        )

        # Send partial transcription to device
        if self._websocket_server:
            await self._websocket_server.send_to_device(device_id, {
                "type": "transcription",
                **transcription
            })

        # If final transcription, trigger response generation
        if transcription.get("is_final"):
            await self._generate_response(device_id, transcription.get("text", ""))

        # Notify callbacks
        for callback in self._on_transcription_callbacks:
            try:
                await callback(device_id, transcription)
            except Exception as e:
                logger.error(
                    "stream_manager_callback_error",
                    callback="transcription",
                    error=str(e)
                )

    def _handle_speech_start(self, device_id: str):
        """Handle speech start detection"""
        state = self._device_states.get(device_id)
        if state:
            state.is_speaking = True

        logger.debug(
            "stream_manager_speech_start",
            device_id=device_id
        )

    def _handle_speech_end(self, device_id: str):
        """Handle speech end detection"""
        state = self._device_states.get(device_id)
        if state:
            state.is_speaking = False

        logger.debug(
            "stream_manager_speech_end",
            device_id=device_id
        )

    async def _handle_interruption(self, device_id: str):
        """Handle user interruption during TTS playback"""
        state = self._device_states.get(device_id)
        if not state:
            return

        # Stop TTS playback
        state.is_responding = False

        # Notify device to stop playback
        if self._websocket_server:
            await self._websocket_server.send_to_device(device_id, {
                "type": "interrupt",
                "reason": "user_speaking",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        logger.info(
            "stream_manager_interruption",
            device_id=device_id
        )

    async def _handle_vision_alert(self, device_id: str, alert: dict):
        """Handle vision alert from video pipeline"""
        logger.info(
            "stream_manager_vision_alert",
            device_id=device_id,
            alert_type=alert.get("type"),
            confidence=alert.get("confidence")
        )

        # Publish alert via MQTT
        if self.mqtt_service:
            self.mqtt_service.publish_system_message(
                "alerts",
                "vision",
                {
                    "device_id": device_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **alert
                },
                qos=1
            )

        # Notify callbacks
        for callback in self._on_alert_callbacks:
            try:
                await callback(device_id, alert)
            except Exception as e:
                logger.error(
                    "stream_manager_callback_error",
                    callback="alert",
                    error=str(e)
                )

    async def _generate_response(self, device_id: str, text: str):
        """Generate and send response to user utterance"""
        if not text.strip():
            return

        try:
            # Get response from LLM
            if self.llm_service:
                response = await self.llm_service.generate_response(
                    text,
                    device_id=device_id
                )

                # Stream TTS response
                await self.send_tts_response(
                    device_id,
                    response.text,
                    emotion=response.emotion
                )
        except Exception as e:
            logger.error(
                "stream_manager_response_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__
            )

    # ==================== Callbacks ====================

    def on_transcription(self, callback: Callable):
        """Register transcription callback"""
        self._on_transcription_callbacks.append(callback)

    def on_alert(self, callback: Callable):
        """Register alert callback"""
        self._on_alert_callbacks.append(callback)

    # ==================== Stats ====================

    def get_stats(self) -> dict:
        """Get stream manager statistics"""
        active_audio = sum(1 for s in self._device_states.values() if s.audio_active)
        active_video = sum(1 for s in self._device_states.values() if s.video_active)
        total_audio = sum(s.audio_chunks_processed for s in self._device_states.values())
        total_video = sum(s.video_frames_processed for s in self._device_states.values())

        return {
            "connected_devices": len(self._device_states),
            "active_audio_streams": active_audio,
            "active_video_streams": active_video,
            "total_audio_chunks_processed": total_audio,
            "total_video_frames_processed": total_video,
        }
```

---

## 5. Voice Streaming Pipeline

### 5.1 New File: `streaming/audio_stream.py`

```python
"""
Audio Streaming Pipeline for real-time voice processing.

Features:
- Audio chunk buffering with ring buffer
- Voice Activity Detection (VAD)
- Streaming Speech-to-Text
- Speech boundary detection
- Noise handling
"""

import asyncio
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable, Any, AsyncIterator
import numpy as np

from utils import get_logger

logger = get_logger(__name__)


class SpeechState(Enum):
    """Voice activity states"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


@dataclass
class AudioChunk:
    """Represents an audio chunk"""
    data: bytes
    sequence: int
    timestamp: int  # milliseconds
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds"""
        samples = len(self.data) // (self.channels * self.sample_width)
        return (samples / self.sample_rate) * 1000

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.frombuffer(self.data, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    # Energy-based VAD
    energy_threshold: float = 0.01  # RMS energy threshold

    # Timing
    speech_pad_ms: int = 300  # Padding before/after speech
    min_speech_ms: int = 250  # Minimum speech duration
    min_silence_ms: int = 500  # Silence duration to end speech
    max_speech_ms: int = 30000  # Maximum continuous speech

    # Smoothing
    smoothing_window: int = 5  # Number of chunks for smoothing


class VoiceActivityDetector:
    """
    Simple energy-based Voice Activity Detection.

    For production, consider using:
    - Silero VAD (torch-based, high accuracy)
    - WebRTC VAD (fast, C-based)
    - Whisper's internal VAD
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()

        # State tracking
        self._state = SpeechState.SILENCE
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # Energy history for smoothing
        self._energy_history: deque = deque(maxlen=self.config.smoothing_window)

    def process_chunk(self, chunk: AudioChunk) -> SpeechState:
        """
        Process audio chunk and return speech state.

        Returns the current speech state after processing.
        """
        current_time = time.time() * 1000  # Current time in ms

        # Calculate RMS energy
        audio_np = chunk.to_numpy()
        energy = np.sqrt(np.mean(audio_np ** 2))

        # Smooth energy
        self._energy_history.append(energy)
        smoothed_energy = np.mean(list(self._energy_history))

        # Detect speech based on energy
        is_speech = smoothed_energy > self.config.energy_threshold

        # State machine
        if self._state == SpeechState.SILENCE:
            if is_speech:
                self._state = SpeechState.SPEECH_START
                self._speech_start_time = current_time
                self._silence_start_time = None

        elif self._state == SpeechState.SPEECH_START:
            # Transition to speaking after minimum duration
            speech_duration = current_time - self._speech_start_time
            if speech_duration >= self.config.min_speech_ms:
                self._state = SpeechState.SPEAKING
            elif not is_speech:
                # False start, go back to silence
                self._state = SpeechState.SILENCE
                self._speech_start_time = None

        elif self._state == SpeechState.SPEAKING:
            if not is_speech:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                else:
                    silence_duration = current_time - self._silence_start_time
                    if silence_duration >= self.config.min_silence_ms:
                        self._state = SpeechState.SPEECH_END
            else:
                self._silence_start_time = None

                # Check for max speech duration
                speech_duration = current_time - self._speech_start_time
                if speech_duration >= self.config.max_speech_ms:
                    self._state = SpeechState.SPEECH_END

        elif self._state == SpeechState.SPEECH_END:
            # Reset for next utterance
            self._state = SpeechState.SILENCE
            self._speech_start_time = None
            self._silence_start_time = None

        return self._state

    def reset(self):
        """Reset VAD state"""
        self._state = SpeechState.SILENCE
        self._speech_start_time = None
        self._silence_start_time = None
        self._energy_history.clear()


class AudioBuffer:
    """
    Ring buffer for audio chunks with speech boundary tracking.

    Maintains a rolling buffer of audio for:
    - Pre-speech context (audio before speech detected)
    - Full utterance accumulation
    - Post-speech padding
    """

    def __init__(
        self,
        max_duration_ms: int = 60000,  # 60 seconds max
        pre_speech_ms: int = 500,  # Keep 500ms before speech
        sample_rate: int = 16000,
    ):
        self.max_duration_ms = max_duration_ms
        self.pre_speech_ms = pre_speech_ms
        self.sample_rate = sample_rate

        # Buffers
        self._pre_buffer: deque[AudioChunk] = deque()
        self._speech_buffer: list[AudioChunk] = []

        # State
        self._is_accumulating = False
        self._total_duration_ms = 0

    def add_chunk(self, chunk: AudioChunk):
        """Add chunk to buffer"""
        if self._is_accumulating:
            # Accumulating speech
            self._speech_buffer.append(chunk)
            self._total_duration_ms += chunk.duration_ms

            # Check max duration
            if self._total_duration_ms >= self.max_duration_ms:
                logger.warning("audio_buffer_max_duration_reached")
        else:
            # Pre-speech buffer (rolling)
            self._pre_buffer.append(chunk)

            # Trim pre-buffer to max duration
            while self._get_pre_buffer_duration() > self.pre_speech_ms:
                self._pre_buffer.popleft()

    def start_accumulating(self):
        """Start accumulating speech"""
        self._is_accumulating = True

        # Move pre-buffer to speech buffer
        self._speech_buffer = list(self._pre_buffer)
        self._total_duration_ms = sum(c.duration_ms for c in self._speech_buffer)
        self._pre_buffer.clear()

    def get_utterance(self) -> bytes:
        """Get accumulated utterance as single audio bytes"""
        if not self._speech_buffer:
            return b""

        return b"".join(chunk.data for chunk in self._speech_buffer)

    def get_utterance_chunks(self) -> list[AudioChunk]:
        """Get accumulated utterance as list of chunks"""
        return self._speech_buffer.copy()

    def clear(self):
        """Clear speech buffer and reset"""
        self._speech_buffer.clear()
        self._total_duration_ms = 0
        self._is_accumulating = False

    def _get_pre_buffer_duration(self) -> float:
        """Get total duration of pre-buffer in ms"""
        return sum(c.duration_ms for c in self._pre_buffer)

    @property
    def duration_ms(self) -> float:
        """Get current accumulated duration"""
        return self._total_duration_ms

    @property
    def is_accumulating(self) -> bool:
        """Check if currently accumulating speech"""
        return self._is_accumulating


class AudioStreamPipeline:
    """
    Complete audio streaming pipeline.

    Processes incoming audio chunks through:
    1. Buffering
    2. Voice Activity Detection
    3. Speech-to-Text (streaming or batch)
    4. Event callbacks

    Usage:
        pipeline = AudioStreamPipeline(
            device_id="robot-001",
            stt_service=stt_service,
            on_transcription=handle_transcription
        )
        await pipeline.start()

        # Process chunks as they arrive
        await pipeline.process_chunk(audio_data, sequence, timestamp)

        await pipeline.stop()
    """

    def __init__(
        self,
        device_id: str,
        stt_service: Optional[Any] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,

        # Callbacks
        on_transcription: Optional[Callable[[dict], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_partial_transcription: Optional[Callable[[str], None]] = None,

        # Configuration
        vad_config: Optional[VADConfig] = None,
    ):
        self.device_id = device_id
        self.stt_service = stt_service
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms

        # Callbacks
        self._on_transcription = on_transcription
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_partial_transcription = on_partial_transcription

        # Components
        self._vad = VoiceActivityDetector(vad_config)
        self._buffer = AudioBuffer(sample_rate=sample_rate)

        # State
        self._running = False
        self._processing_lock = asyncio.Lock()

        # Metrics
        self._chunks_processed = 0
        self._utterances_processed = 0
        self._total_audio_ms = 0

    async def start(self):
        """Start the pipeline"""
        self._running = True
        logger.info(
            "audio_pipeline_started",
            device_id=self.device_id
        )

    async def stop(self):
        """Stop the pipeline"""
        self._running = False

        # Process any remaining audio
        if self._buffer.is_accumulating:
            await self._process_utterance()

        logger.info(
            "audio_pipeline_stopped",
            device_id=self.device_id,
            chunks_processed=self._chunks_processed,
            utterances_processed=self._utterances_processed
        )

    async def process_chunk(
        self,
        audio_data: bytes,
        sequence: int,
        timestamp: int
    ):
        """Process incoming audio chunk"""
        if not self._running:
            return

        async with self._processing_lock:
            # Create chunk object
            chunk = AudioChunk(
                data=audio_data,
                sequence=sequence,
                timestamp=timestamp,
                sample_rate=self.sample_rate
            )

            self._chunks_processed += 1
            self._total_audio_ms += chunk.duration_ms

            # Run VAD
            speech_state = self._vad.process_chunk(chunk)

            # Add to buffer
            self._buffer.add_chunk(chunk)

            # Handle state transitions
            if speech_state == SpeechState.SPEECH_START:
                self._buffer.start_accumulating()
                if self._on_speech_start:
                    self._on_speech_start()

            elif speech_state == SpeechState.SPEECH_END:
                if self._on_speech_end:
                    self._on_speech_end()
                await self._process_utterance()

    async def _process_utterance(self):
        """Process completed utterance through STT"""
        audio_bytes = self._buffer.get_utterance()
        duration_ms = self._buffer.duration_ms

        # Clear buffer
        self._buffer.clear()
        self._vad.reset()

        if not audio_bytes or duration_ms < 100:  # Skip very short audio
            return

        self._utterances_processed += 1

        logger.info(
            "audio_pipeline_processing_utterance",
            device_id=self.device_id,
            duration_ms=round(duration_ms)
        )

        # Transcribe
        if self.stt_service:
            try:
                result = await self.stt_service.transcribe_audio(
                    audio_data=audio_bytes,
                    format="raw",
                    sample_rate=self.sample_rate
                )

                transcription = {
                    "text": result.text,
                    "confidence": result.confidence,
                    "language": result.language,
                    "duration_ms": duration_ms,
                    "is_final": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                if self._on_transcription:
                    await self._on_transcription(transcription)

            except Exception as e:
                logger.error(
                    "audio_pipeline_transcription_error",
                    device_id=self.device_id,
                    error=str(e),
                    error_type=type(e).__name__
                )

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            "device_id": self.device_id,
            "running": self._running,
            "chunks_processed": self._chunks_processed,
            "utterances_processed": self._utterances_processed,
            "total_audio_ms": round(self._total_audio_ms),
            "buffer_accumulating": self._buffer.is_accumulating,
            "buffer_duration_ms": round(self._buffer.duration_ms),
        }
```

### 5.2 STT Service Modifications

**File: `services/stt.py`**

Add streaming transcription capability:

```python
# Add to existing STTService class

async def transcribe_stream(
    self,
    audio_stream: AsyncIterator[bytes],
    sample_rate: int = 16000,
    language: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Streaming transcription with partial results.

    Yields partial transcription results as audio is processed.

    Args:
        audio_stream: Async iterator of audio chunks
        sample_rate: Audio sample rate
        language: Optional language hint

    Yields:
        dict with keys:
            - text: Transcribed text so far
            - is_final: Whether this is the final result
            - confidence: Confidence score (final only)
    """
    # Buffer for accumulating audio
    audio_buffer = []
    total_duration = 0

    # Process interval (transcribe every N ms of audio)
    process_interval_ms = 500
    last_process_duration = 0

    async for chunk in audio_stream:
        audio_buffer.append(chunk)
        chunk_duration = len(chunk) / (sample_rate * 2) * 1000  # 16-bit audio
        total_duration += chunk_duration

        # Emit partial result periodically
        if total_duration - last_process_duration >= process_interval_ms:
            combined_audio = b"".join(audio_buffer)

            # Run partial transcription
            try:
                result = await self._transcribe_partial(
                    combined_audio,
                    sample_rate,
                    language
                )

                yield {
                    "text": result,
                    "is_final": False,
                    "duration_ms": total_duration
                }

                last_process_duration = total_duration

            except Exception as e:
                logger.warning(
                    "stt_partial_transcription_error",
                    error=str(e)
                )

    # Final transcription
    if audio_buffer:
        combined_audio = b"".join(audio_buffer)
        result = await self.transcribe_audio(
            audio_data=combined_audio,
            format="raw",
            sample_rate=sample_rate,
            language=language
        )

        yield {
            "text": result.text,
            "is_final": True,
            "confidence": result.confidence,
            "language": result.language,
            "duration_ms": total_duration
        }

async def _transcribe_partial(
    self,
    audio_data: bytes,
    sample_rate: int,
    language: Optional[str] = None
) -> str:
    """
    Quick partial transcription for streaming.

    Uses faster settings for lower latency.
    """
    # Use faster-whisper with beam_size=1 for speed
    # This is a simplified version - actual implementation
    # depends on your Whisper setup

    result = await self.transcribe_audio(
        audio_data=audio_data,
        format="raw",
        sample_rate=sample_rate,
        language=language
    )

    return result.text
```

### 5.3 TTS Service Modifications

**File: `services/tts.py`**

Add streaming synthesis capability:

```python
# Add to existing TTSService class

async def synthesize_stream(
    self,
    text: str,
    voice: Optional[str] = None,
    emotion: Optional[str] = None,
    chunk_size_ms: int = 200,
) -> AsyncIterator[bytes]:
    """
    Streaming TTS synthesis.

    Generates and yields audio chunks as they're produced,
    allowing playback to begin before full synthesis completes.

    Args:
        text: Text to synthesize
        voice: Voice ID
        emotion: Emotion preset
        chunk_size_ms: Target chunk duration in milliseconds

    Yields:
        Audio data chunks (WAV format, 16-bit PCM)
    """
    # Split text into sentences for chunked processing
    sentences = self._split_into_sentences(text)

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        try:
            # Synthesize sentence
            audio_data = await self.synthesize(
                sentence,
                voice=voice,
                emotion=emotion
            )

            # Yield in chunks
            audio_bytes = audio_data.audio_bytes
            chunk_size = int(
                audio_data.sample_rate *
                (chunk_size_ms / 1000) *
                2  # 16-bit = 2 bytes per sample
            )

            for j in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[j:j + chunk_size]
                yield chunk

                # Small delay to simulate streaming
                # (remove if using truly streaming TTS backend)
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(
                "tts_stream_sentence_error",
                sentence_index=i,
                error=str(e)
            )
            continue

    logger.info(
        "tts_stream_complete",
        sentences=len(sentences),
        text_length=len(text)
    )

def _split_into_sentences(self, text: str) -> list[str]:
    """Split text into sentences for chunked synthesis"""
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Further split very long sentences
    result = []
    max_length = 200

    for sentence in sentences:
        if len(sentence) <= max_length:
            result.append(sentence)
        else:
            # Split on commas or other natural breaks
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            current = ""
            for part in parts:
                if len(current) + len(part) <= max_length:
                    current += (" " if current else "") + part
                else:
                    if current:
                        result.append(current)
                    current = part
            if current:
                result.append(current)

    return result
```

---

## 6. Vision Streaming Pipeline

### 6.1 New File: `streaming/video_stream.py`

```python
"""
Video Streaming Pipeline for continuous security monitoring.

Features:
- Frame buffering with rate limiting
- Motion/change detection
- Object detection (YOLO)
- Zone-based alerting
- Event deduplication
"""

import asyncio
import hashlib
import io
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable, Any, Dict, List, Tuple
import numpy as np

from utils import get_logger

logger = get_logger(__name__)


# Optional imports for image processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available, image processing limited")


class AlertType(Enum):
    """Types of security alerts"""
    MOTION_DETECTED = "motion_detected"
    PERSON_DETECTED = "person_detected"
    VEHICLE_DETECTED = "vehicle_detected"
    ANIMAL_DETECTED = "animal_detected"
    OBJECT_DETECTED = "object_detected"
    ZONE_INTRUSION = "zone_intrusion"
    FACE_DETECTED = "face_detected"


@dataclass
class VideoFrame:
    """Represents a video frame"""
    data: bytes
    sequence: int
    timestamp: int  # milliseconds
    width: int = 0
    height: int = 0
    format: str = "jpeg"  # jpeg, png, raw

    def to_pil(self) -> Optional["Image.Image"]:
        """Convert to PIL Image"""
        if not HAS_PIL:
            return None
        try:
            return Image.open(io.BytesIO(self.data))
        except Exception:
            return None

    def to_numpy(self) -> Optional[np.ndarray]:
        """Convert to numpy array"""
        img = self.to_pil()
        if img:
            return np.array(img)
        return None

    def content_hash(self) -> str:
        """Get content hash for deduplication"""
        return hashlib.md5(self.data).hexdigest()[:16]


@dataclass
class Zone:
    """Defines a monitoring zone within the frame"""
    name: str
    points: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    alert_types: List[AlertType] = field(default_factory=list)
    cooldown_seconds: float = 30.0  # Minimum time between alerts

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside zone (normalized coords)"""
        # Ray casting algorithm for point-in-polygon
        n = len(self.points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside


@dataclass
class Alert:
    """Security alert"""
    type: AlertType
    timestamp: datetime
    device_id: str
    confidence: float
    zone: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    frame_hash: Optional[str] = None


@dataclass
class MotionConfig:
    """Motion detection configuration"""
    enabled: bool = True
    threshold: float = 0.05  # Percentage of pixels changed
    min_area: float = 0.001  # Minimum detection area (as fraction of frame)
    blur_size: int = 21  # Gaussian blur kernel size
    dilate_iterations: int = 2


@dataclass
class VideoStreamConfig:
    """Video stream pipeline configuration"""
    # Frame processing
    target_fps: float = 5.0  # Target processing FPS
    max_fps: float = 30.0  # Maximum accepted FPS
    skip_similar_frames: bool = True
    similarity_threshold: float = 0.95  # Skip frames more similar than this

    # Motion detection
    motion: MotionConfig = field(default_factory=MotionConfig)

    # Object detection
    detect_objects: bool = True
    object_confidence_threshold: float = 0.5
    detect_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "motorcycle", "bicycle",
        "dog", "cat", "bird"
    ])

    # Alerting
    alert_cooldown_seconds: float = 60.0  # Global alert cooldown
    deduplicate_alerts: bool = True


class MotionDetector:
    """
    Motion detection using frame differencing.

    Compares consecutive frames to detect motion.
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        self._previous_frame: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> Tuple[bool, float, List[Tuple[int, int, int, int]]]:
        """
        Detect motion in frame.

        Returns:
            - motion_detected: bool
            - motion_score: float (0-1)
            - motion_regions: list of (x, y, w, h) bounding boxes
        """
        try:
            import cv2
        except ImportError:
            return False, 0.0, []

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Apply blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.config.blur_size, self.config.blur_size), 0)

        # First frame, nothing to compare
        if self._previous_frame is None:
            self._previous_frame = gray
            return False, 0.0, []

        # Compute absolute difference
        frame_delta = cv2.absdiff(self._previous_frame, gray)

        # Threshold
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill holes
        thresh = cv2.dilate(thresh, None, iterations=self.config.dilate_iterations)

        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate motion score
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_score = motion_pixels / total_pixels

        # Find motion regions
        motion_regions = []
        min_area = self.config.min_area * total_pixels

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))

        # Update previous frame
        self._previous_frame = gray

        motion_detected = motion_score > self.config.threshold and len(motion_regions) > 0

        return motion_detected, motion_score, motion_regions

    def reset(self):
        """Reset motion detector state"""
        self._previous_frame = None


class AlertDeduplicator:
    """
    Deduplicates alerts to prevent spam.

    Tracks recent alerts and filters duplicates based on:
    - Alert type
    - Zone
    - Time window
    """

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self._recent_alerts: Dict[str, float] = {}  # key -> timestamp

    def should_alert(self, alert: Alert) -> bool:
        """Check if alert should be emitted"""
        key = self._get_alert_key(alert)
        current_time = time.time()

        # Check cooldown
        last_alert_time = self._recent_alerts.get(key)
        if last_alert_time and (current_time - last_alert_time) < self.cooldown_seconds:
            return False

        # Update last alert time
        self._recent_alerts[key] = current_time

        # Cleanup old entries
        self._cleanup()

        return True

    def _get_alert_key(self, alert: Alert) -> str:
        """Generate key for alert deduplication"""
        return f"{alert.device_id}:{alert.type.value}:{alert.zone or 'global'}"

    def _cleanup(self):
        """Remove old entries"""
        current_time = time.time()
        cutoff = current_time - (self.cooldown_seconds * 2)

        self._recent_alerts = {
            k: v for k, v in self._recent_alerts.items()
            if v > cutoff
        }


class VideoStreamPipeline:
    """
    Complete video streaming pipeline for security monitoring.

    Processes incoming video frames through:
    1. Rate limiting / frame skipping
    2. Motion detection
    3. Object detection (YOLO)
    4. Zone checking
    5. Alert generation

    Usage:
        pipeline = VideoStreamPipeline(
            device_id="camera-front",
            vision_service=vision_service,
            on_alert=handle_alert
        )
        await pipeline.start()

        # Process frames as they arrive
        await pipeline.process_frame(frame_data, sequence, timestamp)

        await pipeline.stop()
    """

    def __init__(
        self,
        device_id: str,
        vision_service: Optional[Any] = None,
        config: Optional[VideoStreamConfig] = None,

        # Callbacks
        on_alert: Optional[Callable[[Alert], None]] = None,
        on_motion: Optional[Callable[[float, List], None]] = None,
        on_objects: Optional[Callable[[List[dict]], None]] = None,

        # Zones
        zones: Optional[List[Zone]] = None,
    ):
        self.device_id = device_id
        self.vision_service = vision_service
        self.config = config or VideoStreamConfig()

        # Callbacks
        self._on_alert = on_alert
        self._on_motion = on_motion
        self._on_objects = on_objects

        # Zones
        self._zones = zones or []

        # Components
        self._motion_detector = MotionDetector(self.config.motion)
        self._deduplicator = AlertDeduplicator(self.config.alert_cooldown_seconds)

        # Frame tracking
        self._last_frame_hash: Optional[str] = None
        self._last_process_time: float = 0
        self._frame_interval = 1.0 / self.config.target_fps

        # State
        self._running = False
        self._processing_lock = asyncio.Lock()

        # Metrics
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_skipped = 0
        self._alerts_generated = 0
        self._motion_events = 0

    async def start(self):
        """Start the pipeline"""
        self._running = True
        logger.info(
            "video_pipeline_started",
            device_id=self.device_id,
            target_fps=self.config.target_fps
        )

    async def stop(self):
        """Stop the pipeline"""
        self._running = False
        logger.info(
            "video_pipeline_stopped",
            device_id=self.device_id,
            frames_received=self._frames_received,
            frames_processed=self._frames_processed,
            frames_skipped=self._frames_skipped,
            alerts_generated=self._alerts_generated
        )

    async def process_frame(
        self,
        frame_data: bytes,
        sequence: int,
        timestamp: int
    ):
        """Process incoming video frame"""
        if not self._running:
            return

        self._frames_received += 1

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_process_time < self._frame_interval:
            self._frames_skipped += 1
            return

        async with self._processing_lock:
            # Create frame object
            frame = VideoFrame(
                data=frame_data,
                sequence=sequence,
                timestamp=timestamp
            )

            # Skip similar frames
            if self.config.skip_similar_frames:
                frame_hash = frame.content_hash()
                if frame_hash == self._last_frame_hash:
                    self._frames_skipped += 1
                    return
                self._last_frame_hash = frame_hash

            self._last_process_time = current_time
            self._frames_processed += 1

            # Convert to numpy for processing
            frame_np = frame.to_numpy()
            if frame_np is None:
                logger.warning(
                    "video_pipeline_frame_decode_failed",
                    device_id=self.device_id,
                    sequence=sequence
                )
                return

            # Update frame dimensions
            frame.height, frame.width = frame_np.shape[:2]

            # Run motion detection
            motion_detected = False
            motion_regions = []

            if self.config.motion.enabled:
                motion_detected, motion_score, motion_regions = \
                    self._motion_detector.detect(frame_np)

                if motion_detected:
                    self._motion_events += 1

                    if self._on_motion:
                        await self._on_motion(motion_score, motion_regions)

                    # Generate motion alert
                    await self._generate_alert(
                        AlertType.MOTION_DETECTED,
                        confidence=motion_score,
                        frame=frame,
                        details={"regions": motion_regions}
                    )

            # Run object detection (only if motion detected or always)
            if self.config.detect_objects and (motion_detected or not self.config.motion.enabled):
                await self._run_object_detection(frame, frame_np)

    async def _run_object_detection(self, frame: VideoFrame, frame_np: np.ndarray):
        """Run YOLO object detection"""
        if not self.vision_service:
            return

        try:
            # Get detections from vision service
            detections = await self.vision_service.detect_objects(frame.data)

            if self._on_objects:
                await self._on_objects(detections)

            # Process detections
            for detection in detections:
                obj_class = detection.get("class", "").lower()
                confidence = detection.get("confidence", 0)
                bbox = detection.get("bbox", {})

                # Skip low confidence
                if confidence < self.config.object_confidence_threshold:
                    continue

                # Skip non-target classes
                if obj_class not in self.config.detect_classes:
                    continue

                # Determine alert type
                alert_type = self._get_alert_type_for_class(obj_class)

                # Check zones
                center_x = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2 / frame.width
                center_y = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2 / frame.height

                triggered_zone = None
                for zone in self._zones:
                    if zone.contains_point(center_x, center_y):
                        if alert_type in zone.alert_types or not zone.alert_types:
                            triggered_zone = zone.name
                            break

                # Generate alert
                await self._generate_alert(
                    alert_type,
                    confidence=confidence,
                    frame=frame,
                    zone=triggered_zone,
                    details={
                        "class": obj_class,
                        "bbox": bbox,
                    }
                )

        except Exception as e:
            logger.error(
                "video_pipeline_detection_error",
                device_id=self.device_id,
                error=str(e),
                error_type=type(e).__name__
            )

    def _get_alert_type_for_class(self, obj_class: str) -> AlertType:
        """Map object class to alert type"""
        mapping = {
            "person": AlertType.PERSON_DETECTED,
            "car": AlertType.VEHICLE_DETECTED,
            "truck": AlertType.VEHICLE_DETECTED,
            "motorcycle": AlertType.VEHICLE_DETECTED,
            "bicycle": AlertType.VEHICLE_DETECTED,
            "dog": AlertType.ANIMAL_DETECTED,
            "cat": AlertType.ANIMAL_DETECTED,
            "bird": AlertType.ANIMAL_DETECTED,
        }
        return mapping.get(obj_class, AlertType.OBJECT_DETECTED)

    async def _generate_alert(
        self,
        alert_type: AlertType,
        confidence: float,
        frame: VideoFrame,
        zone: Optional[str] = None,
        details: Optional[dict] = None
    ):
        """Generate and emit alert"""
        alert = Alert(
            type=alert_type,
            timestamp=datetime.now(timezone.utc),
            device_id=self.device_id,
            confidence=confidence,
            zone=zone,
            details=details or {},
            frame_hash=frame.content_hash()
        )

        # Deduplication
        if self.config.deduplicate_alerts and not self._deduplicator.should_alert(alert):
            return

        self._alerts_generated += 1

        logger.info(
            "video_pipeline_alert",
            device_id=self.device_id,
            alert_type=alert_type.value,
            confidence=round(confidence, 2),
            zone=zone
        )

        # Emit alert
        if self._on_alert:
            alert_dict = {
                "type": alert_type.value,
                "timestamp": alert.timestamp.isoformat(),
                "device_id": alert.device_id,
                "confidence": alert.confidence,
                "zone": alert.zone,
                "details": alert.details,
            }
            await self._on_alert(alert_dict)

    # ==================== Zone Management ====================

    def add_zone(self, zone: Zone):
        """Add monitoring zone"""
        self._zones.append(zone)
        logger.info(
            "video_pipeline_zone_added",
            device_id=self.device_id,
            zone_name=zone.name
        )

    def remove_zone(self, zone_name: str):
        """Remove monitoring zone"""
        self._zones = [z for z in self._zones if z.name != zone_name]

    def get_zones(self) -> List[Zone]:
        """Get all zones"""
        return self._zones.copy()

    # ==================== Stats ====================

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            "device_id": self.device_id,
            "running": self._running,
            "frames_received": self._frames_received,
            "frames_processed": self._frames_processed,
            "frames_skipped": self._frames_skipped,
            "processing_rate": self._frames_processed / max(1, self._frames_received),
            "alerts_generated": self._alerts_generated,
            "motion_events": self._motion_events,
            "zones_configured": len(self._zones),
        }
```

### 6.2 Vision Service Modifications

**File: `services/vision.py`**

Add continuous monitoring support:

```python
# Add to existing VisionService class

async def start_continuous_monitoring(
    self,
    device_id: str,
    callback: Callable[[dict], None],
    config: Optional[dict] = None
) -> str:
    """
    Start continuous monitoring for a device.

    Returns monitoring session ID.
    """
    session_id = f"{device_id}_{int(time.time())}"

    # Store monitoring session
    self._monitoring_sessions[session_id] = {
        "device_id": device_id,
        "callback": callback,
        "config": config or {},
        "started_at": datetime.now(timezone.utc),
        "frames_processed": 0,
    }

    logger.info(
        "vision_monitoring_started",
        device_id=device_id,
        session_id=session_id
    )

    return session_id

async def stop_continuous_monitoring(self, session_id: str):
    """Stop continuous monitoring session"""
    session = self._monitoring_sessions.pop(session_id, None)

    if session:
        logger.info(
            "vision_monitoring_stopped",
            session_id=session_id,
            frames_processed=session["frames_processed"]
        )

async def process_monitoring_frame(
    self,
    session_id: str,
    frame_data: bytes
) -> Optional[dict]:
    """
    Process frame for monitoring session.

    Returns detection results if any objects found.
    """
    session = self._monitoring_sessions.get(session_id)
    if not session:
        return None

    session["frames_processed"] += 1

    # Run detection
    detections = await self.detect_objects(frame_data)

    if detections and session["callback"]:
        result = {
            "session_id": session_id,
            "device_id": session["device_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detections": detections,
        }
        await session["callback"](result)
        return result

    return None
```

---

## 7. Protocol Specifications

### 7.1 WebSocket Message Types

#### Client → Server Messages

| Type | Description | Binary | Fields |
|------|-------------|--------|--------|
| `auth` | Authentication | No | `device_id`, `token` |
| `audio_chunk` | Audio data (JSON) | No | `data` (base64), `sequence`, `timestamp` |
| `video_frame` | Video data (JSON) | No | `data` (base64), `sequence`, `timestamp` |
| `control` | Control command | No | `action`, `params` |
| `ping` | Keepalive | No | `timestamp` |
| Binary 0x01 | Audio chunk | Yes | See binary format |
| Binary 0x02 | Video frame | Yes | See binary format |

#### Server → Client Messages

| Type | Description | Binary | Fields |
|------|-------------|--------|--------|
| `auth_success` | Auth confirmed | No | `device_id`, `capabilities` |
| `error` | Error message | No | `error`, `timestamp` |
| `transcription` | STT result | No | `text`, `is_final`, `confidence` |
| `tts_complete` | TTS finished | No | `timestamp` |
| `interrupt` | Stop playback | No | `reason` |
| `pong` | Keepalive response | No | `timestamp`, `server_time` |
| Binary 0x03 | TTS audio chunk | Yes | See binary format |

### 7.2 Binary Message Format

```
┌─────────┬──────────┬───────────┬─────────────┐
│  Type   │ Sequence │ Timestamp │   Payload   │
│ 1 byte  │ 4 bytes  │  8 bytes  │  Variable   │
└─────────┴──────────┴───────────┴─────────────┘
```

**Type Values:**
- `0x01` - Audio chunk (client → server)
- `0x02` - Video frame (client → server)
- `0x03` - TTS audio (server → client)

**Sequence:** Big-endian uint32, incrementing per stream

**Timestamp:** Big-endian uint64, milliseconds since epoch

### 7.3 Audio Format Specification

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| Channels | 1 (mono) |
| Sample Width | 16-bit (2 bytes) |
| Encoding | PCM (raw) or Opus |
| Chunk Duration | 20-100ms recommended |

### 7.4 Video Format Specification

| Parameter | Value |
|-----------|-------|
| Format | JPEG (preferred) or PNG |
| Resolution | Configurable, recommend 640x480 or 1280x720 |
| Quality | JPEG quality 70-85 |
| Frame Rate | 1-30 FPS configurable |

---

## 8. Implementation Phases

### Phase 1: WebSocket Infrastructure (Week 1)

**Goal:** Establish WebSocket server alongside existing MQTT

**Tasks:**
1. Create `streaming/` module structure
2. Implement `WebSocketServer` class
3. Implement `StreamManager` class
4. Integrate into `NailaAIServer`
5. Add WebSocket startup to lifecycle
6. Basic authentication flow
7. Unit tests for WebSocket server

**Deliverables:**
- Working WebSocket server accepting connections
- Device authentication
- Message routing infrastructure

### Phase 2: Voice Streaming (Week 2)

**Goal:** Real-time voice conversation

**Tasks:**
1. Implement `AudioStreamPipeline`
2. Implement `VoiceActivityDetector`
3. Implement `AudioBuffer`
4. Modify `STTService` for streaming
5. Modify `TTSService` for streaming
6. Implement interruption handling
7. Integration tests

**Deliverables:**
- Low-latency voice processing
- Partial transcription results
- Streaming TTS playback
- Conversation interruption

### Phase 3: Vision Streaming (Week 3)

**Goal:** Continuous security camera monitoring

**Tasks:**
1. Implement `VideoStreamPipeline`
2. Implement `MotionDetector`
3. Implement zone-based alerting
4. Implement `AlertDeduplicator`
5. Modify `VisionService` for continuous monitoring
6. MQTT alert publishing
7. Integration tests

**Deliverables:**
- Continuous frame processing
- Motion detection
- Object detection alerts
- Zone intrusion alerts

### Phase 4: Integration & Polish (Week 4)

**Goal:** Production-ready system

**Tasks:**
1. End-to-end testing
2. Performance optimization
3. Memory leak testing
4. Load testing
5. Documentation
6. Monitoring/metrics
7. Error handling improvements

**Deliverables:**
- Production-ready streaming system
- Performance benchmarks
- Operations documentation

---

## 9. File-by-File Changes

### 9.1 New Files

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `streaming/__init__.py` | Module exports | ~30 |
| `streaming/websocket_server.py` | WebSocket server | ~400 |
| `streaming/stream_manager.py` | Stream coordination | ~350 |
| `streaming/audio_stream.py` | Audio pipeline | ~450 |
| `streaming/video_stream.py` | Video pipeline | ~500 |
| `streaming/protocols.py` | Message definitions | ~100 |
| `config/streaming.py` | Streaming configuration | ~80 |
| `tests/unit/test_websocket_server.py` | WebSocket tests | ~200 |
| `tests/unit/test_audio_pipeline.py` | Audio tests | ~250 |
| `tests/unit/test_video_pipeline.py` | Video tests | ~250 |
| `tests/integration/test_streaming.py` | Integration tests | ~300 |

### 9.2 Modified Files

| File | Changes | Impact |
|------|---------|--------|
| `server/naila_server.py` | Add WebSocket server init | Low |
| `server/lifecycle.py` | Add streaming startup stage | Low |
| `services/stt.py` | Add streaming transcription | Medium |
| `services/tts.py` | Add streaming synthesis | Medium |
| `services/vision.py` | Add continuous monitoring | Medium |
| `config/__init__.py` | Export streaming config | Low |
| `requirements.txt` | Add websockets package | Low |

### 9.3 Server Initialization Changes

**File: `server/naila_server.py`**

```python
# Add imports
from streaming import WebSocketServer, StreamManager

class NailaAIServer:
    def __init__(self):
        # ... existing code ...

        # NEW: Initialize streaming components
        self.stream_manager = StreamManager(
            stt_service=self.stt_service,
            tts_service=self.tts_service,
            vision_service=self.vision_service,
            llm_service=self.llm_service,
            mqtt_service=self.mqtt_service,
        )

        self.websocket_server = WebSocketServer(
            host="0.0.0.0",
            port=8765,
            stream_manager=self.stream_manager,
        )

        # Link back reference
        self.stream_manager.set_websocket_server(self.websocket_server)

        # Update lifecycle manager
        self.lifecycle = ServerLifecycleManager(
            mqtt_service=self.mqtt_service,
            protocol_handlers=self.protocol_handlers,
            llm_service=self.llm_service,
            stt_service=self.stt_service,
            tts_service=self.tts_service,
            vision_service=self.vision_service,
            websocket_server=self.websocket_server,  # NEW
            stream_manager=self.stream_manager,  # NEW
        )
```

**File: `server/lifecycle.py`**

```python
class StartupStage(Enum):
    LOAD_AI_MODELS = "Loading AI models"
    REGISTER_HANDLERS = "Registering protocol handlers"
    START_MQTT = "Starting MQTT service"
    START_WEBSOCKET = "Starting WebSocket server"  # NEW
    START_HEALTH_MONITORING = "Starting health monitoring"
    PUBLISH_STATUS = "Publishing initial system status"


class ServerLifecycleManager:
    def __init__(
        self,
        mqtt_service,
        protocol_handlers,
        llm_service=None,
        stt_service=None,
        tts_service=None,
        vision_service=None,
        websocket_server=None,  # NEW
        stream_manager=None,  # NEW
    ):
        # ... existing code ...
        self.websocket_server = websocket_server
        self.stream_manager = stream_manager

    async def start_server(self):
        # ... existing stages ...

        # NEW: Stage: Start WebSocket server
        if self.websocket_server:
            logger.info("startup_stage", stage=StartupStage.START_WEBSOCKET.value)
            await self.websocket_server.start()
            logger.info("websocket_server_ready", port=self.websocket_server.port)

        # ... continue with other stages ...

    async def stop_server(self):
        # ... existing cleanup ...

        # NEW: Stop WebSocket server
        if self.websocket_server:
            logger.info("shutdown_stage", stage="Stopping WebSocket server")
            await self.websocket_server.stop()
```

---

## 10. Configuration

### 10.1 New File: `config/streaming.py`

```python
"""
Streaming configuration for real-time audio/video processing.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WebSocketConfig:
    """WebSocket server configuration"""
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: int = 30
    ping_timeout: int = 10
    max_message_size: int = 10 * 1024 * 1024  # 10MB

    @classmethod
    def from_env(cls) -> "WebSocketConfig":
        return cls(
            host=os.getenv("WS_HOST", "0.0.0.0"),
            port=int(os.getenv("WS_PORT", "8765")),
            ping_interval=int(os.getenv("WS_PING_INTERVAL", "30")),
            ping_timeout=int(os.getenv("WS_PING_TIMEOUT", "10")),
            max_message_size=int(os.getenv("WS_MAX_MESSAGE_SIZE", str(10 * 1024 * 1024))),
        )


@dataclass
class AudioStreamConfig:
    """Audio streaming configuration"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit
    chunk_duration_ms: int = 100

    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.01
    speech_pad_ms: int = 300
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    max_speech_ms: int = 30000

    # Buffer settings
    pre_speech_buffer_ms: int = 500
    max_utterance_ms: int = 60000

    @classmethod
    def from_env(cls) -> "AudioStreamConfig":
        return cls(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            chunk_duration_ms=int(os.getenv("AUDIO_CHUNK_MS", "100")),
            vad_enabled=os.getenv("VAD_ENABLED", "true").lower() == "true",
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.01")),
            min_silence_ms=int(os.getenv("VAD_MIN_SILENCE_MS", "500")),
            max_speech_ms=int(os.getenv("VAD_MAX_SPEECH_MS", "30000")),
        )


@dataclass
class VideoStreamConfig:
    """Video streaming configuration"""
    target_fps: float = 5.0
    max_fps: float = 30.0
    skip_similar_frames: bool = True
    similarity_threshold: float = 0.95

    # Motion detection
    motion_enabled: bool = True
    motion_threshold: float = 0.05
    motion_min_area: float = 0.001

    # Object detection
    detect_objects: bool = True
    object_confidence_threshold: float = 0.5
    detect_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "dog", "cat"
    ])

    # Alerting
    alert_cooldown_seconds: float = 60.0
    deduplicate_alerts: bool = True

    @classmethod
    def from_env(cls) -> "VideoStreamConfig":
        classes = os.getenv("VISION_DETECT_CLASSES", "person,car,truck,dog,cat")
        return cls(
            target_fps=float(os.getenv("VIDEO_TARGET_FPS", "5.0")),
            max_fps=float(os.getenv("VIDEO_MAX_FPS", "30.0")),
            motion_enabled=os.getenv("MOTION_ENABLED", "true").lower() == "true",
            motion_threshold=float(os.getenv("MOTION_THRESHOLD", "0.05")),
            detect_objects=os.getenv("DETECT_OBJECTS", "true").lower() == "true",
            object_confidence_threshold=float(os.getenv("OBJECT_CONFIDENCE", "0.5")),
            detect_classes=classes.split(","),
            alert_cooldown_seconds=float(os.getenv("ALERT_COOLDOWN_SECONDS", "60.0")),
        )


@dataclass
class StreamingConfig:
    """Combined streaming configuration"""
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    audio: AudioStreamConfig = field(default_factory=AudioStreamConfig)
    video: VideoStreamConfig = field(default_factory=VideoStreamConfig)

    @classmethod
    def from_env(cls) -> "StreamingConfig":
        return cls(
            websocket=WebSocketConfig.from_env(),
            audio=AudioStreamConfig.from_env(),
            video=VideoStreamConfig.from_env(),
        )
```

### 10.2 Environment Variables

```bash
# WebSocket Server
WS_HOST=0.0.0.0
WS_PORT=8765
WS_PING_INTERVAL=30
WS_PING_TIMEOUT=10
WS_MAX_MESSAGE_SIZE=10485760

# Audio Streaming
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_MS=100

# Voice Activity Detection
VAD_ENABLED=true
VAD_THRESHOLD=0.01
VAD_MIN_SILENCE_MS=500
VAD_MAX_SPEECH_MS=30000

# Video Streaming
VIDEO_TARGET_FPS=5.0
VIDEO_MAX_FPS=30.0

# Motion Detection
MOTION_ENABLED=true
MOTION_THRESHOLD=0.05

# Object Detection
DETECT_OBJECTS=true
OBJECT_CONFIDENCE=0.5
VISION_DETECT_CLASSES=person,car,truck,dog,cat

# Alerting
ALERT_COOLDOWN_SECONDS=60.0
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

**WebSocket Server Tests:**
```python
# tests/unit/test_websocket_server.py

import pytest
import asyncio
from streaming.websocket_server import WebSocketServer, DeviceSession

class TestWebSocketServer:

    @pytest.fixture
    async def server(self):
        server = WebSocketServer(port=0)  # Random port
        await server.start()
        yield server
        await server.stop()

    async def test_server_starts_and_stops(self, server):
        assert server._running
        await server.stop()
        assert not server._running

    async def test_device_authentication(self, server):
        # Test auth flow
        pass

    async def test_binary_message_parsing(self):
        # Test binary protocol parsing
        pass

    async def test_session_cleanup_on_disconnect(self):
        pass
```

**Audio Pipeline Tests:**
```python
# tests/unit/test_audio_pipeline.py

import pytest
import numpy as np
from streaming.audio_stream import (
    AudioStreamPipeline,
    VoiceActivityDetector,
    AudioBuffer,
    AudioChunk,
    SpeechState
)

class TestVoiceActivityDetector:

    def test_silence_detection(self):
        vad = VoiceActivityDetector()

        # Generate silence
        silence = np.zeros(1600, dtype=np.int16).tobytes()
        chunk = AudioChunk(data=silence, sequence=0, timestamp=0)

        state = vad.process_chunk(chunk)
        assert state == SpeechState.SILENCE

    def test_speech_detection(self):
        vad = VoiceActivityDetector()

        # Generate loud signal
        speech = (np.sin(np.linspace(0, 100, 1600)) * 10000).astype(np.int16).tobytes()
        chunk = AudioChunk(data=speech, sequence=0, timestamp=0)

        state = vad.process_chunk(chunk)
        assert state in [SpeechState.SPEECH_START, SpeechState.SPEAKING]

class TestAudioBuffer:

    def test_pre_speech_buffer(self):
        buffer = AudioBuffer(pre_speech_ms=500)

        # Add chunks
        for i in range(10):
            chunk = AudioChunk(
                data=b"\x00" * 3200,  # 100ms at 16kHz
                sequence=i,
                timestamp=i * 100
            )
            buffer.add_chunk(chunk)

        # Pre-buffer should be limited
        assert not buffer.is_accumulating

    def test_accumulation(self):
        buffer = AudioBuffer()

        buffer.start_accumulating()

        chunk = AudioChunk(data=b"\x00" * 3200, sequence=0, timestamp=0)
        buffer.add_chunk(chunk)

        assert buffer.is_accumulating
        assert buffer.duration_ms > 0
```

**Video Pipeline Tests:**
```python
# tests/unit/test_video_pipeline.py

import pytest
from streaming.video_stream import (
    VideoStreamPipeline,
    MotionDetector,
    AlertDeduplicator,
    Zone,
    AlertType,
    Alert
)

class TestMotionDetector:

    def test_no_motion_identical_frames(self):
        detector = MotionDetector()

        # Two identical frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detector.detect(frame)
        motion, score, regions = detector.detect(frame)

        assert not motion
        assert score < 0.01

class TestAlertDeduplicator:

    def test_first_alert_passes(self):
        dedup = AlertDeduplicator(cooldown_seconds=60)

        alert = Alert(
            type=AlertType.PERSON_DETECTED,
            timestamp=datetime.now(timezone.utc),
            device_id="cam-1",
            confidence=0.9
        )

        assert dedup.should_alert(alert)

    def test_duplicate_blocked(self):
        dedup = AlertDeduplicator(cooldown_seconds=60)

        alert = Alert(
            type=AlertType.PERSON_DETECTED,
            timestamp=datetime.now(timezone.utc),
            device_id="cam-1",
            confidence=0.9
        )

        assert dedup.should_alert(alert)
        assert not dedup.should_alert(alert)  # Duplicate blocked

class TestZone:

    def test_point_in_zone(self):
        zone = Zone(
            name="entrance",
            points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        )

        assert zone.contains_point(0.5, 0.5)  # Inside
        assert not zone.contains_point(0.05, 0.05)  # Outside
```

### 11.2 Integration Tests

```python
# tests/integration/test_streaming.py

import pytest
import asyncio
import websockets
import json

class TestStreamingIntegration:

    @pytest.fixture
    async def server(self):
        # Start full server with streaming
        from server.naila_server import NailaAIServer
        server = NailaAIServer()
        await server.start()
        yield server
        await server.stop()

    async def test_websocket_connection_and_auth(self, server):
        uri = f"ws://localhost:{server.websocket_server.port}"

        async with websockets.connect(uri) as ws:
            # Send auth
            await ws.send(json.dumps({
                "type": "auth",
                "device_id": "test-device"
            }))

            # Receive auth success
            response = await ws.recv()
            data = json.loads(response)

            assert data["type"] == "auth_success"
            assert data["device_id"] == "test-device"

    async def test_audio_streaming_round_trip(self, server):
        uri = f"ws://localhost:{server.websocket_server.port}"

        async with websockets.connect(uri) as ws:
            # Auth
            await ws.send(json.dumps({
                "type": "auth",
                "device_id": "test-device"
            }))
            await ws.recv()

            # Start audio stream
            await ws.send(json.dumps({
                "type": "control",
                "action": "start_audio"
            }))

            # Send audio chunks
            for i in range(10):
                chunk = b"\x00" * 3200  # 100ms silence
                header = bytes([0x01])  # Audio type
                seq = i.to_bytes(4, "big")
                ts = (i * 100).to_bytes(8, "big")

                await ws.send(header + seq + ts + chunk)

            # Allow processing time
            await asyncio.sleep(0.5)
```

### 11.3 Load Tests

```python
# tests/performance/test_streaming_load.py

import pytest
import asyncio
import websockets
import time

class TestStreamingLoad:

    async def test_multiple_concurrent_streams(self, server):
        """Test server handles multiple devices streaming simultaneously"""
        num_devices = 10
        chunks_per_device = 100

        async def device_stream(device_id: str):
            uri = f"ws://localhost:{server.websocket_server.port}"

            async with websockets.connect(uri) as ws:
                # Auth
                await ws.send(json.dumps({
                    "type": "auth",
                    "device_id": device_id
                }))
                await ws.recv()

                # Stream chunks
                for i in range(chunks_per_device):
                    chunk = b"\x00" * 3200
                    await ws.send(chunk)
                    await asyncio.sleep(0.1)  # 100ms chunks

        start = time.time()

        # Run all devices concurrently
        await asyncio.gather(*[
            device_stream(f"device-{i}")
            for i in range(num_devices)
        ])

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < chunks_per_device * 0.15  # Allow 50% overhead
```

---

## 12. Performance Considerations

### 12.1 Latency Budget

**Voice Conversation Target: <500ms end-to-end**

| Stage | Target | Optimization |
|-------|--------|--------------|
| Audio capture + transmit | <50ms | Small chunks (20-50ms) |
| VAD + buffering | <10ms | Simple energy-based VAD |
| STT processing | <200ms | Streaming Whisper, GPU |
| LLM generation | <150ms | Fast model, streaming |
| TTS synthesis | <50ms | Streaming TTS |
| Audio transmit + playback | <40ms | Pre-buffering |

### 12.2 Memory Management

**Audio Pipeline:**
- Ring buffer limits: 60 seconds max
- Chunk pooling to reduce allocations
- Clear buffers after processing

**Video Pipeline:**
- Frame pooling
- Skip similar frames
- Limit detection queue depth
- Release numpy arrays promptly

### 12.3 CPU Optimization

**Audio:**
- Use numpy for audio math
- Batch STT when possible
- Run VAD on dedicated thread

**Video:**
- Adaptive FPS based on activity
- Skip frames when processing lags
- Use GPU for YOLO inference
- Resize frames before detection

### 12.4 Network Optimization

- Binary protocol reduces overhead by ~25% vs JSON+base64
- Opus codec for audio (optional, ~10x compression)
- JPEG quality tuning for video
- Disable WebSocket compression (already compressed data)

---

## 13. Security Considerations

### 13.1 Authentication

- Device token validation on WebSocket connect
- Token rotation support
- Connection rate limiting
- Failed auth lockout

### 13.2 Data Security

- TLS for WebSocket (wss://)
- No sensitive data in logs
- Audio/video not persisted by default
- Secure alert delivery

### 13.3 Denial of Service Prevention

- Max connections per IP
- Max message size limits
- Rate limiting per device
- Queue depth limits with backpressure

### 13.4 Input Validation

- Validate message types
- Validate sequence numbers
- Validate timestamps (reject stale)
- Sanitize zone definitions

---

## 14. Migration Guide

### 14.1 Backward Compatibility

The streaming system is **additive** - existing MQTT-based communication continues to work unchanged.

**Devices can:**
1. Use MQTT only (existing behavior)
2. Use WebSocket only (new streaming)
3. Use both (recommended for transition)

### 14.2 Migration Steps

**Phase 1: Server Update**
1. Deploy server with streaming support
2. WebSocket server starts alongside MQTT
3. Existing devices continue using MQTT

**Phase 2: Device Update**
1. Update device firmware to support WebSocket
2. Device connects via WebSocket for streaming
3. Falls back to MQTT if WebSocket unavailable

**Phase 3: Full Migration**
1. All devices using WebSocket for streaming
2. MQTT used for events/commands only
3. Monitor and optimize

### 14.3 Rollback Plan

If issues arise:
1. Disable WebSocket server (config flag)
2. Devices fall back to MQTT
3. No data loss, degraded experience only

---

## Appendix A: Dependencies

### Required Packages

```
# requirements.txt additions

# WebSocket
websockets>=12.0

# Audio processing
numpy>=1.24.0

# Video processing (optional, for motion detection)
opencv-python-headless>=4.8.0
Pillow>=10.0.0

# Streaming STT (optional upgrade)
faster-whisper>=1.0.0
```

### Optional Packages

```
# For Opus audio codec
opuslib>=3.0.0

# For advanced VAD
silero-vad>=4.0.0
webrtcvad>=2.0.10

# For GPU acceleration
torch>=2.0.0
```

---

## Appendix B: Monitoring & Observability

### Metrics to Track

**WebSocket:**
- `ws_connections_active` - Current connections
- `ws_connections_total` - Total connections
- `ws_messages_received` - Messages received
- `ws_messages_sent` - Messages sent
- `ws_bytes_received` - Bytes received
- `ws_bytes_sent` - Bytes sent
- `ws_errors` - Error count

**Audio Pipeline:**
- `audio_chunks_processed` - Chunks processed
- `audio_utterances_processed` - Utterances processed
- `audio_vad_detections` - Speech detections
- `audio_transcription_latency_ms` - STT latency

**Video Pipeline:**
- `video_frames_received` - Frames received
- `video_frames_processed` - Frames processed
- `video_frames_skipped` - Frames skipped
- `video_motion_detections` - Motion events
- `video_object_detections` - Object detections
- `video_alerts_generated` - Alerts generated

### Health Checks

```python
# Add to health_monitor.py

async def check_streaming_health(self) -> dict:
    """Check streaming system health"""
    return {
        "websocket": {
            "running": self.websocket_server._running if self.websocket_server else False,
            "connections": len(self.websocket_server._sessions) if self.websocket_server else 0,
        },
        "streams": {
            "active_audio": self.stream_manager.get_stats()["active_audio_streams"],
            "active_video": self.stream_manager.get_stats()["active_video_streams"],
        }
    }
```

---

## Appendix C: Client Implementation Guide

### JavaScript/TypeScript Client Example

```typescript
class NailaStreamingClient {
    private ws: WebSocket | null = null;
    private deviceId: string;
    private audioContext: AudioContext | null = null;
    private mediaRecorder: MediaRecorder | null = null;

    constructor(deviceId: string) {
        this.deviceId = deviceId;
    }

    async connect(serverUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(serverUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                // Send auth
                this.ws!.send(JSON.stringify({
                    type: 'auth',
                    device_id: this.deviceId
                }));
            };

            this.ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);

                    if (data.type === 'auth_success') {
                        resolve();
                    }
                } else {
                    this.handleBinaryMessage(event.data);
                }
            };

            this.ws.onerror = reject;
        });
    }

    async startAudioStream(): Promise<void> {
        // Get microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        // Create audio processor
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(stream);
        const processor = this.audioContext.createScriptProcessor(1600, 1, 1);

        let sequence = 0;

        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const pcmData = new Int16Array(inputData.length);

            // Convert float32 to int16
            for (let i = 0; i < inputData.length; i++) {
                pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
            }

            // Send binary chunk
            this.sendAudioChunk(pcmData.buffer, sequence++);
        };

        source.connect(processor);
        processor.connect(this.audioContext.destination);

        // Notify server
        this.ws!.send(JSON.stringify({
            type: 'control',
            action: 'start_audio'
        }));
    }

    private sendAudioChunk(data: ArrayBuffer, sequence: number): void {
        const header = new Uint8Array(13);
        header[0] = 0x01;  // Audio type

        const view = new DataView(header.buffer);
        view.setUint32(1, sequence, false);  // Big endian
        view.setBigUint64(5, BigInt(Date.now()), false);

        const message = new Uint8Array(header.length + data.byteLength);
        message.set(header);
        message.set(new Uint8Array(data), header.length);

        this.ws!.send(message);
    }

    private handleMessage(data: any): void {
        switch (data.type) {
            case 'transcription':
                console.log('Transcription:', data.text, data.is_final);
                break;
            case 'interrupt':
                console.log('Interrupted:', data.reason);
                // Stop audio playback
                break;
        }
    }

    private handleBinaryMessage(data: ArrayBuffer): void {
        const header = new Uint8Array(data.slice(0, 13));
        const type = header[0];

        if (type === 0x03) {
            // TTS audio chunk
            const audioData = data.slice(13);
            this.playAudioChunk(audioData);
        }
    }

    private playAudioChunk(data: ArrayBuffer): void {
        // Play audio using Web Audio API
        // Implementation depends on audio format
    }
}
```

---

*Document End*
