#!/usr/bin/env bash
# Install launchd daemons so mosquitto and the AI server run at boot, for any login.
# Usage: sudo bash scripts/install-daemons.sh        (uninstall: --uninstall)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

RUN_USER=tp
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="/Users/$RUN_USER"
LOGS="$HOME_DIR/Library/Logs/robo-naila"
DAEMONS=/Library/LaunchDaemons
LABELS=(com.robonaila.mosquitto com.robonaila.ai-server)

for label in "${LABELS[@]}"; do
  launchctl bootout "system/$label" 2>/dev/null || true
done
if [[ "${1:-}" == "--uninstall" ]]; then
  for label in "${LABELS[@]}"; do rm -f "$DAEMONS/$label.plist"; done
  echo "uninstalled"; exit 0
fi

sudo -u "$RUN_USER" mkdir -p "$LOGS"

common() { # $1=label $2=log name
  cat <<EOF
  <key>Label</key><string>$1</string>
  <key>UserName</key><string>$RUN_USER</string>
  <key>GroupName</key><string>staff</string>
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
  <key>ProgramArguments</key><array><string>/opt/homebrew/sbin/mosquitto</string></array>
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
  <array><string>/bin/bash</string><string>$ROOT/scripts/orchestrator-run.sh</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProcessType</key><string>Interactive</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>$HOME_DIR</string>
    <key>PATH</key><string>/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF

for label in "${LABELS[@]}"; do
  plutil -lint "$DAEMONS/$label.plist" >/dev/null
  chown root:wheel "$DAEMONS/$label.plist"; chmod 644 "$DAEMONS/$label.plist"
  launchctl bootstrap system "$DAEMONS/$label.plist"
done
echo "installed; logs in $LOGS"
