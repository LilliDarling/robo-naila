#!/usr/bin/env bash
# Install launchd daemons so mosquitto and the AI server run at boot, for any login.
# If ai-server/.env sets NAILA_OTLP_RELAY_TARGET=host:port, also install a root loopback relay to
# that trace collector (see deploy/mac/otlp-relay.pl); point OTEL_EXPORTER_OTLP_TRACES_ENDPOINT at
# http://127.0.0.1:4318/v1/traces. Re-run after changing the target.
# Usage: sudo RUN_USER=your-user bash deploy/mac/install.sh        (uninstall: --uninstall)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

RUN_USER="${RUN_USER:-${SUDO_USER:?set RUN_USER to the service account}}"
RUN_GROUP="${RUN_GROUP:-$(id -gn "$RUN_USER")}"
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOME_DIR="${RUN_HOME:-/Users/$RUN_USER}"
LOGS="${RUN_LOGS:-$HOME_DIR/Library/Logs/robo-naila}"
DAEMONS=/Library/LaunchDaemons
LABELS=(com.robonaila.mosquitto com.robonaila.ai-server com.robonaila.otlp-relay)
# Root-owned copy, so only root can change what the root relay runs.
RELAY_DIR="/Library/Application Support/robonaila"

for value in "$RUN_USER" "$RUN_GROUP" "$ROOT" "$HOME_DIR" "$LOGS" "$HOMEBREW_PREFIX"; do
  case "$value" in *'&'*|*'<'*|*'>'*) echo "plist values must not contain XML metacharacters"; exit 1 ;; esac
done

for label in "${LABELS[@]}"; do
  launchctl bootout "system/$label" 2>/dev/null || true
done
# bootout returns before the job is gone; the AI server takes seconds to shut down, and
# bootstrapping before then fails with "5: Input/output error".
for label in "${LABELS[@]}"; do
  for _ in {1..60}; do launchctl print "system/$label" >/dev/null 2>&1 || break; sleep 0.5; done
done
if [[ "${1:-}" == "--uninstall" ]]; then
  for label in "${LABELS[@]}"; do rm -f "$DAEMONS/$label.plist"; done
  rm -rf "$RELAY_DIR"
  echo "uninstalled"; exit 0
fi

sudo -u "$RUN_USER" mkdir -p "$LOGS"

# Parse, never source: the .env is user-writable and this script runs as root.
RELAY_TARGET=""
if [[ -f "$ROOT/ai-server/.env" ]]; then
  RELAY_TARGET="$(sed -n 's/^NAILA_OTLP_RELAY_TARGET=//p' "$ROOT/ai-server/.env" 2>/dev/null | tail -1 | tr -d '"[:space:]')"
fi
if [[ -n "$RELAY_TARGET" && ! "$RELAY_TARGET" =~ ^[A-Za-z0-9.-]+:[0-9]+$ ]]; then
  echo "invalid NAILA_OTLP_RELAY_TARGET '$RELAY_TARGET' (want host:port)"; exit 1
fi

common() { # $1=label $2=log name
  cat <<EOF
  <key>Label</key><string>$1</string>
  <key>UserName</key><string>$RUN_USER</string>
  <key>GroupName</key><string>$RUN_GROUP</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$LOGS/$2.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$2.log</string>
EOF
}

cat > "$DAEMONS/com.robonaila.mosquitto.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
$(common com.robonaila.mosquitto mosquitto)
  <key>ProgramArguments</key><array><string>$HOMEBREW_PREFIX/sbin/mosquitto</string></array>
</dict>
</plist>
EOF

cat > "$DAEMONS/com.robonaila.ai-server.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
$(common com.robonaila.ai-server ai-server)
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/deploy/mac/orchestrator-run.sh</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProcessType</key><string>Interactive</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>$HOME_DIR</string>
    <key>PATH</key><string>$HOMEBREW_PREFIX/bin:$HOMEBREW_PREFIX/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF

rm -f "$DAEMONS/com.robonaila.otlp-relay.plist"
if [[ -n "$RELAY_TARGET" ]]; then
  mkdir -p "$RELAY_DIR"
  install -o root -g wheel -m 755 "$ROOT/deploy/mac/otlp-relay.pl" "$RELAY_DIR/otlp-relay.pl"
  # Root, no UserName: launchd daemons running as root are exempt from Local Network privacy.
  cat > "$DAEMONS/com.robonaila.otlp-relay.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.robonaila.otlp-relay</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$LOGS/otlp-relay.log</string>
  <key>StandardErrorPath</key><string>$LOGS/otlp-relay.log</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/perl</string><string>$RELAY_DIR/otlp-relay.pl</string>
    <string>127.0.0.1:4318</string><string>$RELAY_TARGET</string>
  </array>
</dict>
</plist>
EOF
fi

for label in "${LABELS[@]}"; do
  [[ -f "$DAEMONS/$label.plist" ]] || continue
  plutil -lint "$DAEMONS/$label.plist" >/dev/null
  chown root:wheel "$DAEMONS/$label.plist"; chmod 644 "$DAEMONS/$label.plist"
  launchctl bootstrap system "$DAEMONS/$label.plist"
done
echo "installed; logs in $LOGS"
