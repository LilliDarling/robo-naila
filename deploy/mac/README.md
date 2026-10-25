# Mac AI server deployment

Run from the repository root after installing the AI dependencies, models, `uv`
and Homebrew mosquitto. Follow [AI server setup](../../ai-server/README.md#setup)
for Python 3.12, ffmpeg and model downloads. Start a broker with `mosquitto -p 1883`
(or install boot services below); check the listener with
`lsof -nP -iTCP:1883 -sTCP:LISTEN`. The Pi only needs gRPC, so MQTT can stay local.
For a foreground server:

```sh
./deploy/mac/orchestrator-run.sh
```

The launcher reads `ai-server/.env`; its gRPC, MQTT and STT defaults can all be
overridden there or in the environment. Defaults bind gRPC on `0.0.0.0:50051`,
use CPU/int8 Whisper with hub-side VAD, and preserve the installed native LLM build.
The Whisper model must be CTranslate2 format (`models/stt/small.en` by default);
set `STT_MODEL_PATH` to override. A whisper.cpp GGML model is incompatible.

Find the Mac LAN address with `networksetup -listallhardwareports` and
`ipconfig getifaddr en0` (substitute the correct interface). Keep the Mac awake,
allow incoming Python connections if prompted, and configure the Pi's
`NAILA_AI_GRPC_URL=http://MAC_LAN_IP:50051`. Wait for `grpc_server_started`, then
check `http://PI:8080/health` for `grpc_connected: true` and healthy `ai_server`.
Use [Pi deployment](../pi/README.md) for hub installation and troubleshooting.

Install boot services for an existing, unprivileged local account:

```sh
sudo RUN_USER="$USER" bash deploy/mac/install.sh
sudo RUN_USER="$USER" bash deploy/mac/install.sh --uninstall
```

`RUN_USER` defaults to `SUDO_USER` when omitted. Optional installer environment:

| Variable | Default |
| --- | --- |
| `RUN_GROUP` | Account's primary group (`id -gn`) |
| `RUN_HOME` | `/Users/$RUN_USER` |
| `RUN_LOGS` | `$RUN_HOME/Library/Logs/robo-naila` |
| `HOMEBREW_PREFIX` | `/opt/homebrew` (set `/usr/local` for Intel Homebrew) |

Pass overrides after `sudo`, e.g. `sudo RUN_USER=alice HOMEBREW_PREFIX=/usr/local
bash deploy/mac/install.sh`. The checkout's absolute path is discovered from the
installer location and recorded in launchd plists. Keep it accessible to the
service account; rerun the installer after moving it. Paths must not contain XML
metacharacters (`&`, `<`, `>`). Installation replaces the three `com.robonaila.*`
system jobs; uninstall removes those jobs and the root-owned relay copy.

macOS Local Network privacy can block outgoing LAN connections from user
LaunchDaemons (`errno 65`, "No route to host"). Incoming gRPC remains unaffected.
The root loopback relay bypasses that restriction while the AI server remains
unprivileged. The AI server uses OTLP/HTTP on 4318; hub traces use OTLP/gRPC
on 4317 directly from the Pi, with both collector listeners enabled.
For a remote AI trace collector, set `NAILA_OTLP_RELAY_TARGET=COLLECTOR_HOST:4318`
and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces` in
`ai-server/.env`, then rerun installation. Without a relay target only MQTT and
the AI server are installed. The optional root relay listens only on loopback;
its destination is supplied by configuration, never a fixed LAN address.
