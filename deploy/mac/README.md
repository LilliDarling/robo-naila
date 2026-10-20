# Mac AI server deployment

Run from the repository root after installing the AI dependencies, models, `uv`
and Homebrew mosquitto. For a foreground server:

```sh
./deploy/mac/orchestrator-run.sh
```

The launcher reads `ai-server/.env`; its gRPC, MQTT and STT defaults can all be
overridden there or in the environment. See [AI server setup](../../ai-server/README.md#mac-orchestrator-with-a-remote-hub).

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

For a remote trace collector, set `NAILA_OTLP_RELAY_TARGET=COLLECTOR_HOST:4318`
and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces` in
`ai-server/.env`, then rerun installation. Without a relay target only MQTT and
the AI server are installed. The optional root relay listens only on loopback;
its destination is supplied by configuration, never a fixed LAN address.
