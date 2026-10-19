#!/usr/bin/env bash
# Run the AI server on this machine; the remote hub connects to TCP 50051.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ai-server"
# Machine-local settings (e.g. NAILA_BENCHMARK) live in the gitignored .env; they win over the defaults below.
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export GRPC_HOST="${GRPC_HOST:-0.0.0.0}"
export GRPC_PORT="${GRPC_PORT:-50051}"
export MQTT_BROKER_HOST="${MQTT_BROKER_HOST:-127.0.0.1}"
export STT_DEVICE="${STT_DEVICE:-cpu}"
export STT_COMPUTE_TYPE="${STT_COMPUTE_TYPE:-int8}"
export STT_VAD_FILTER="${STT_VAD_FILTER:-false}"
export STT_MODEL_PATH="${STT_MODEL_PATH:-models/stt/small.en}"
# Preserve the installed native build (Metal on Apple Silicon).
exec uv run --no-sync python main.py
