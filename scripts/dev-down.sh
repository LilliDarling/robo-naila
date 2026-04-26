#!/usr/bin/env bash
# Tear down the NAILA dev tmux session started by dev-up.sh.
# Sends Ctrl-C to each window first for graceful shutdown, then kills.

set -euo pipefail

SESSION="naila"
GRACE_SECONDS="${NAILA_SHUTDOWN_GRACE:-3}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No '$SESSION' tmux session running."
    exit 0
fi

echo "Sending Ctrl-C to all windows..."
for window in $(tmux list-windows -t "$SESSION" -F "#{window_name}"); do
    tmux send-keys -t "$SESSION:$window" C-c
done

echo "Waiting ${GRACE_SECONDS}s for graceful shutdown..."
sleep "$GRACE_SECONDS"

tmux kill-session -t "$SESSION"
echo "NAILA stack stopped."
