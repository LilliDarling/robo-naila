 ```
      ___           ___                       ___       ___     
     /\__\         /\  \          ___        /\__\     /\  \    
    /::|  |       /::\  \        /\  \      /:/  /    /::\  \   
   /:|:|  |      /:/\:\  \       \:\  \    /:/  /    /:/\:\  \  
  /:/|:|  |__   /::\~\:\  \      /::\__\  /:/  /    /::\~\:\  \ 
 /:/ |:| /\__\ /:/\:\ \:\__\  __/:/\/__/ /:/__/    /:/\:\ \:\__\
 \/__|:|/:/  / \/__\:\/:/  / /\/:/  /    \:\  \    \/__\:\/:/  /
     |:/:/  /       \::/  /  \::/__/      \:\  \        \::/  / 
     |::/  /        /:/  /    \:\__\       \:\  \       /:/  /  
     /:/  /        /:/  /      \/__/        \:\__\     /:/  /   
     \/__/         \/__/                     \/__/     \/__/    
                                                                                                           
```

# NAILA

Local AI assistant with voice conversations, vision, and personality. Runs entirely on a local network with no cloud dependencies.

## Table of Contents

1. [About](#1-about)
2. [Features & Capabilities](#2-features--capabilities)
3. [System Architecture](#3-system-architecture)
4. [Hardware Overview](#4-hardware-overview)
5. [Software Components](#5-software-components)
6. [Future Enhancements](#6-future-enhancements)
7. [Copyright & Licensing](#7-copyright--licensing)

---

## 1. About

NAILA is an AI companion powered by local models for privacy-focused, low-latency interaction. The system combines speech recognition, language understanding, voice synthesis, and computer vision into a real-time conversational assistant.

## 2. Features & Capabilities

### Voice Conversation
- **Speech-to-Text:** Transcribes speech using **faster-whisper** (Whisper small.en model)
- **LLM Processing:** Generates responses using **Llama 3 8B Instruct** via `llama-cpp-python`
- **Text-to-Speech:** Synthesizes voice with **Piper** (Lessac voice, ONNX) including SSML support and 24 emotion presets
- **Real-Time Streaming:** Sub-700ms voice response latency via WebRTC + gRPC pipeline

### Vision
- **Object Detection:** Identifies objects using **YOLOv8 Nano** via `ultralytics` (80 COCO classes)
- **Scene Analysis:** Generates natural language scene descriptions
- **Visual Queries:** Answers questions about what the camera sees

### AI Orchestration
- **LangGraph Pipeline:** `process_input → process_vision → retrieve_context → generate_response → execute_actions`
- **Conversation Memory:** Per-device conversation history and context tracking
- **MQTT Integration:** Pub/sub messaging for device coordination and commands

### Device Support
- **Raspberry Pi:** Full-duplex WebRTC audio with echo cancellation

### Maintainability
- **Hardware Auto-Detection:** Automatic CPU/CUDA/MPS selection and optimization

## 3. System Architecture

[![NAILA Software Architecture](public/architecture.png)](https://link.excalidraw.com/readonly/hPAa6SyIcYey19s1j2yO?darkMode=true)

```
Pi Audio ─────── WebRTC ──► Hub ──── gRPC ─► AI Server
AI Server ────── MQTT ────────────────────► All Devices
```

**Device Layer:** Raspberry Pi devices stream audio over WebRTC through the Hub.

**Hub (Rust):** Accepts WebRTC connections from Pi devices, runs voice activity detection to filter silence, and relays speech audio to the AI server over gRPC. Routes TTS audio back to devices.

**AI Server (Python):** Runs all AI models (STT, LLM, TTS, Vision) orchestrated by LangGraph. Communicates with devices via MQTT for commands, alerts, and text chat.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## 4. Hardware Overview

* **Raspberry Pi:** Primary device platform with PortAudio mic/speaker and SpeexDSP echo cancellation
* **Camera:** USB or Pi camera module for vision processing
* **Audio Input:** Microphone via PortAudio
* **Audio Output:** Speaker with amplifier

## 5. Software Components

### `ai-server/` — AI Orchestration & Processing

Python service running all AI models with LangGraph orchestration.

| Service | Model | Library |
|---------|-------|---------|
| STT | Whisper small.en | faster-whisper |
| LLM | Llama 3 8B Instruct | llama-cpp-python |
| TTS | Piper (Lessac, ONNX) | piper-tts |
| Vision | YOLOv8 Nano | ultralytics |

**Key modules:** `services/` (AI models), `graphs/` (LangGraph orchestration), `agents/` (orchestrator, input processor, response generator), `mqtt/` (client, handlers, routing), `config/` (per-service configuration), `memory/` (conversation history)

See [ai-server/README.md](ai-server/README.md)

### `hub/` — Streaming Relay

Rust service bridging devices to the AI server via WebRTC + gRPC.

- WebRTC server with Opus codec (48kHz, 20ms frames)
- Voice activity detection (webrtc-vad)
- gRPC bidirectional streaming to AI server
- HTTP signaling on `:8080`, health on `/health`

See [hub/README.md](hub/README.md)

### `devices/pi-audio/` — Raspberry Pi Audio Client

Python WebRTC client for Raspberry Pi.

- Full-duplex audio via PortAudio
- SpeexDSP echo cancellation
- aiortc WebRTC connection to hub
- Automatic reconnect with exponential backoff

See [devices/pi-audio/README.md](devices/pi-audio/README.md)

### `proto/` — gRPC Protocol Definitions

Protocol buffer definitions for Hub ↔ AI Server communication.

- `nailaV1.proto` — Full specification with session config, status, and error handling
- `naila.proto` — Minimal working version for initial pipeline

### `docs/` — Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture overview
- [STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md) — WebRTC/gRPC streaming protocol
- [MQTT_PROTOCOL.md](docs/MQTT_PROTOCOL.md) — MQTT topic hierarchy and message formats
- [FLOW.md](docs/FLOW.md) — Multi-agent interaction flow
- [SETUP.md](docs/SETUP.md) — Installation and setup guide

## 6. Future Enhancements

* **Video Pipeline:** WebRTC video streaming with motion detection and YOLO frame analysis
* **Multi-Device Routing:** Room-based audio/video routing with device registry
* **Advanced Embodiment:** More complex motor control for nuanced expressions and body language
* **Personalization:** Learning user preferences, long-term memory with vector database
* **External Integrations:** Weather, calendar, smart home APIs
* **Mobile App:** Companion app for remote control and monitoring
* **Multi-Room Context:** Cross-room awareness and context-aware responses

## 7. Copyright & Licensing

©2025 Valkyrie Remedy LLC. All rights reserved.

This code is provided for informational and educational purposes only. You are welcome to view the code.

**Unless explicitly granted by the copyright holder, no one is permitted to:**
* Use, copy, modify, distribute, or create derivative works from this code for any purpose, commercial or non-commercial.
* Offer contributions to this repository. All development is managed internally.

If you wish to discuss potential collaborations or licensing for specific use cases, please contact Lillith@valkyrieremedy.com.