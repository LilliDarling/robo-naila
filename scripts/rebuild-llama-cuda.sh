#!/usr/bin/env bash
# Rebuild llama-cpp-python from source with CUDA support enabled.
#
# Why this exists: the PyPI wheel for llama-cpp-python is CPU-only. Every
# `uv sync` (or any `uv pip install` that touches the venv) replaces the
# CUDA-built copy with that CPU wheel and silently regresses LLM throughput
# from ~50 tok/s back to ~3 tok/s. This script does the rebuild in one
# command so recovery is trivial.
#
# Usage (from repo root or anywhere):
#   ./scripts/rebuild-llama-cuda.sh
#
# Prerequisites:
#   - CUDA Toolkit installed in WSL (`nvcc --version` should work)
#   - The ai-server venv exists at ai-server/.venv
#   - You can spare ~10 minutes for the build
#
# After this finishes, restart the AI server so it loads the new build:
#   tmux send-keys -t naila:ai-server C-c
#   sleep 5
#   tmux send-keys -t naila:ai-server "..." Enter

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_SERVER_DIR="$(cd "$SCRIPT_DIR/../ai-server" && pwd)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install the CUDA Toolkit first:" >&2
    echo "  https://developer.nvidia.com/cuda-toolkit (WSL-Ubuntu version)" >&2
    echo "Or via apt: see docs/WSL_SETUP.md GPU section." >&2
    exit 1
fi

if [[ ! -d "$AI_SERVER_DIR/.venv" ]]; then
    echo "ERROR: ai-server venv not found at $AI_SERVER_DIR/.venv" >&2
    echo "Run \`uv sync\` in ai-server/ first." >&2
    exit 1
fi

echo "Rebuilding llama-cpp-python with CUDA support..."
echo "(This takes ~10 minutes — compiles llama.cpp + CUDA kernels from source.)"
echo

cd "$AI_SERVER_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python \
    --no-binary llama-cpp-python \
    --force-reinstall \
    --no-cache-dir

echo
echo "Verifying GPU support..."
.venv/bin/python -c "
import llama_cpp
ok = llama_cpp.llama_supports_gpu_offload()
if ok:
    print('  llama_supports_gpu_offload(): True  — rebuild succeeded')
else:
    print('  llama_supports_gpu_offload(): False — rebuild did NOT pick up CUDA')
    raise SystemExit(1)
"

echo
echo "Done. Restart the AI server to pick up the new build."
echo "  tmux send-keys -t naila:ai-server C-c"
echo "  sleep 5"
echo "  tmux send-keys -t naila:ai-server \"source .venv/bin/activate && \\"
echo "    STT_VAD_FILTER=false STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 \\"
echo "    uv run --no-sync python main.py\" Enter"
