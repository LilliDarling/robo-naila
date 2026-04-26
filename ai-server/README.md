# NAILA AI Server

Python service that runs all AI models for the NAILA assistant — speech-to-text,
LLM, text-to-speech, and vision — orchestrated by LangGraph. Talks to devices
via MQTT (control plane) and gRPC + WebRTC through the [Hub](../hub/README.md)
(real-time audio).

For overall system context see the top-level [README](../README.md) and
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Models

| Service | Model | Library |
|---------|-------|---------|
| STT | Whisper small.en | [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) |
| LLM | Llama 3.1 8B Instruct (Q4_K_M GGUF) | [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) |
| TTS | Kokoro 82M (ONNX) | [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) |
| Vision | YOLOv8 Nano | [`ultralytics`](https://github.com/ultralytics/ultralytics) |

All models run locally. CPU works; CUDA accelerates LLM and STT significantly
(see [GPU Acceleration](#gpu-acceleration)).

## Project Structure

```
ai-server/
├── main.py                  Entry point — wires the server and starts the loop.
├── server/                  Server lifecycle, health monitoring, top-level orchestration.
│   ├── naila_server.py      NailaAIServer — composes services, MQTT, gRPC, lifecycle.
│   ├── lifecycle.py         Ordered startup/shutdown of services.
│   └── health_monitor.py    Periodic health checks.
├── services/                Stateful AI model wrappers.
│   ├── stt.py               faster-whisper transcription.
│   ├── llm.py               llama-cpp-python chat completion.
│   ├── tts.py               Kokoro speech synthesis with phrase-cache + warmup.
│   └── vision.py            YOLOv8 detection + scene description.
├── agents/                  Stateless LangGraph nodes.
│   ├── orchestrator.py      NAILAOrchestrator — runs the graph, applies callbacks.
│   ├── input_processor.py   Text/audio input normalization, intent detection.
│   └── response_generator.py LLM-driven (or pattern fallback) response generation.
├── graphs/                  LangGraph definitions.
│   ├── orchestration.py     process_input → process_vision → retrieve_context
│   │                         → generate_response → execute_actions
│   └── states.py            Graph state schema.
├── managers/
│   └── ai_model.py          Centralized model loading with shared hardware detection.
├── mqtt/                    MQTT client, handlers, topic routing.
├── rpc/                     gRPC server (proto in /proto, generated in rpc/generated/).
│   ├── server.py            grpc.aio server lifecycle.
│   └── service.py           NailaAIServicer — bidirectional StreamConversation.
├── memory/
│   └── conversation.py      Per-device conversation history with TTL + max-history caps.
├── config/                  Per-service env-driven config (one module per service).
├── utils/                   Logging, hardware detection, text normalization, caches.
├── prompts/                 LLM system prompt(s).
├── models/                  Model files live here (created by download_models.sh).
└── tests/                   pytest suite (unit / integration / performance).
```

## Setup

Requires Python 3.12 and `uv`.

```bash
cd ai-server
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

System dependencies (one-time):

```bash
# macOS
brew install ffmpeg

# Ubuntu / WSL
sudo apt install ffmpeg portaudio19-dev libopus0
```

(WSL users: see [`docs/WSL_SETUP.md`](../docs/WSL_SETUP.md) for the full
walkthrough including the PortAudio rebuild.)

### Download models

```bash
bash download_models.sh
```

Pulls Whisper small.en, Llama 3.1 8B Q4_K_M, Kokoro v1.0 + voices, and
YOLOv8n into `models/` (~5GB total).

### GPU acceleration

`llama-cpp-python` ships CPU-only wheels by default. Without rebuilding,
LLM inference runs at ~2 tok/s on CPU vs 20-40+ tok/s on GPU.

```bash
# Linux / WSL with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --no-cache-dir

# macOS (Apple Silicon, Metal)
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --no-cache-dir
```

Verify: `python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`
should print `True`.

faster-whisper uses CTranslate2 — GPU support requires NVIDIA cuDNN. If cuDNN
isn't installed, force STT to CPU with `STT_DEVICE=cpu STT_COMPUTE_TYPE=int8`
to avoid runtime errors.

## Running

```bash
uv run python main.py
```

The server initializes models in parallel (~30 seconds typical), starts the
gRPC server on port `50051`, and connects to the MQTT broker (default
`localhost:1883`). Wait for `grpc_server_started` in the logs before pointing
the hub at it.

Common environment overrides for development:

```bash
STT_VAD_FILTER=false STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 uv run python main.py
```

| Variable | Why |
|---|---|
| `STT_VAD_FILTER=false` | Hub already does VAD; double-filtering rejects short utterances |
| `STT_DEVICE=cpu` | Avoids cuDNN crashes when cuDNN isn't installed |
| `STT_COMPUTE_TYPE=int8` | Best STT performance on CPU |

Full config surface lives in `config/{stt,llm,tts,vision,mqtt,grpc}.py` —
each module documents its own env vars.

## Communication

### gRPC (real-time audio loop)

`NailaAI.StreamConversation` is a bidirectional stream. The hub sends
`AudioInput` messages tagged with VAD events (Start/Continue/End) and the
server replies with `AudioOutput` messages carrying Opus-encoded TTS audio.
See [`/proto/naila.proto`](../proto/naila.proto) for the full schema and
[`docs/STREAMING_ARCHITECTURE.md`](../docs/STREAMING_ARCHITECTURE.md) for the
end-to-end flow.

### MQTT (device control plane)

Per-device topics for text input, command messages, status, and TTS audio
delivery to non-WebRTC devices. See
[`docs/MQTT_PROTOCOL.md`](../docs/MQTT_PROTOCOL.md) for the topic hierarchy.

## Testing

```bash
uv run pytest tests/unit                  # fast, runs by default
uv run pytest tests/integration           # requires MQTT broker running
uv run pytest tests/performance           # benchmark-only, skip in CI
```

Conftest mocks hardware detection by default so tests are CPU/GPU agnostic.

## Helper scripts

- `scripts/test_grpc_client.py` — talk to the running gRPC server with
  recorded audio for end-to-end debugging without the hub.
- `scripts/test_mqtt_client.py` — publish/subscribe sanity check against the
  MQTT broker.
- `scripts/verify_tts.py` — smoke-test TTS synthesis after model download.
- `scripts/generate_grpc.sh` — regenerate proto stubs into `rpc/generated/`
  after editing `/proto/naila.proto`.
