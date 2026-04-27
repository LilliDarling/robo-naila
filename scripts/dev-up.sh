#!/usr/bin/env bash
# Bring up the full NAILA dev stack (mosquitto + ai-server + hub + audio-client)
# in a tmux session. Each service runs in its own window so logs are visible.
#
# Usage:
#   ./scripts/dev-up.sh
#
# Environment overrides:
#   NAILA_DEVICE_ID          override the audio-client device id (default: dev-$HOSTNAME)
#   NAILA_INPUT_DEVICE       override the input device (default: RDPSource)
#   NAILA_OUTPUT_DEVICE      override the output device (default: RDPSink)
#   NAILA_SKIP_AUDIO_CHECK   set to 1 to skip the pactl pre-flight check
#   NAILA_SKIP_AUDIO_CLIENT  set to 1 to launch only mqtt + ai-server + hub
#                            (useful when you want to run the audio client
#                            on Windows or a Pi instead of inside WSL)

set -euo pipefail

SESSION="naila"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DEVICE_ID="${NAILA_DEVICE_ID:-dev-${HOSTNAME:-laptop}}"
INPUT_DEVICE="${NAILA_INPUT_DEVICE:-RDPSource}"
OUTPUT_DEVICE="${NAILA_OUTPUT_DEVICE:-RDPSink}"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not installed — install it first:" >&2
    echo "  sudo apt install tmux" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running."
    echo "  Attach:    tmux attach -t $SESSION"
    echo "  Tear down: ./scripts/dev-down.sh"
    exit 1
fi

# Verify PulseAudio is reachable BEFORE spinning up everything. WSLg's
# pulseaudio is the most common reason the audio client fails to start.
if [[ -z "${NAILA_SKIP_AUDIO_CLIENT:-}" && -z "${NAILA_SKIP_AUDIO_CHECK:-}" ]]; then
    if command -v pactl >/dev/null 2>&1 && ! timeout 3 pactl info >/dev/null 2>&1; then
        echo "WARNING: pactl info failed — PulseAudio is not responding." >&2
        echo "" >&2
        echo "On WSLg this usually means audio is dead. Try, in order:" >&2
        echo "  1. From Windows PowerShell as admin: Restart-Service -Force Audiosrv" >&2
        echo "  2. wsl --shutdown (loses every WSL terminal)" >&2
        echo "" >&2
        echo "Or set NAILA_SKIP_AUDIO_CLIENT=1 to bring up everything except" >&2
        echo "the audio client (run that on Windows or a Pi separately)." >&2
        echo "" >&2
        printf "Continue anyway? [y/N] " >&2
        read -r answer
        [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# MQTT broker (idempotent — start if not running)
# ─────────────────────────────────────────────────────────────────────────────

if command -v systemctl >/dev/null 2>&1; then
    if ! systemctl is-active --quiet mosquitto 2>/dev/null; then
        echo "Starting mosquitto..."
        sudo systemctl start mosquitto
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# tmux session: one window per service. Reconnect logic in hub and
# audio-client means we don't have to wait for predecessors to be ready —
# they'll back off and connect when their dependency comes up.
# ─────────────────────────────────────────────────────────────────────────────

tmux new-session -d -s "$SESSION" -n ai-server -c "$ROOT/ai-server"
# `uv run --no-sync` skips uv's implicit dep-resync. Without it, every start
# overwrites manually-built native packages (e.g. llama-cpp-python compiled
# with CUDA) by reinstalling the PyPI CPU wheel from uv.lock. Run `uv sync`
# explicitly when you want to update deps.
tmux send-keys -t "$SESSION:ai-server" \
    "source .venv/bin/activate && STT_VAD_FILTER=false STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 uv run --no-sync python main.py" \
    C-m

tmux new-window -t "$SESSION" -n hub -c "$ROOT/hub"
tmux send-keys -t "$SESSION:hub" "cargo run --release" C-m

if [[ -z "${NAILA_SKIP_AUDIO_CLIENT:-}" ]]; then
    tmux new-window -t "$SESSION" -n audio-client -c "$ROOT/devices/audio-client"
    tmux send-keys -t "$SESSION:audio-client" \
        "source .venv/bin/activate && uv run python -m audio_client \
            --hub-url http://localhost:8080 \
            --device-id $DEVICE_ID \
            --input-device $INPUT_DEVICE \
            --output-device $OUTPUT_DEVICE" \
        C-m
fi

cat <<EOF

NAILA stack starting in tmux session '$SESSION'.

  Attach:        tmux attach -t $SESSION
  Switch window: Ctrl-b then [number]   (or Ctrl-b n / Ctrl-b p)
  Detach:        Ctrl-b d
  Tear down:     ./scripts/dev-down.sh

Windows:
  0: ai-server     (model load takes ~30s; wait for grpc_server_started)
  1: hub           (waits for AI server with backoff)
EOF

if [[ -z "${NAILA_SKIP_AUDIO_CLIENT:-}" ]]; then
    cat <<EOF
  2: audio-client  (waits for hub with backoff)

Audio client is using:
  device-id      $DEVICE_ID
  input-device   $INPUT_DEVICE
  output-device  $OUTPUT_DEVICE
EOF
else
    cat <<EOF

audio-client SKIPPED (NAILA_SKIP_AUDIO_CLIENT=1).
Run it separately on Windows / a Pi pointing at:
  --hub-url http://<this-machine-ip>:8080
EOF
fi

echo
