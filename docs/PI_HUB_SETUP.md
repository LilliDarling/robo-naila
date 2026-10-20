# Pi hub → Mac AI server: first deployment

Run only the Rust hub on the Pi; orchestration and models stay on the Mac.
This establishes the deployment pair for #38. Benchmarks, OTel, YOLO measurement
and physical audio-device setup are separate work. Repeat the pair checks
below after rebuilding or changing either host's address.

Provisioned pair (2026-10-15): Pi 4 Model B Rev 1.5, Debian 13 (trixie),
`aarch64`, SSH `hub@hub.local` (Ethernet preferred; Wi-Fi also configured).
The Pi routes to the Mac via Ethernet. Mac: MacBook Air M4, 16 GB, macOS 26.6.2,
on the same LAN. Find its current address with the commands below.
The operator already flashed this Pi: resume with SSH, not another card write.

For Ethernet and Wi-Fi on the same subnet, host configuration lives in
[`deploy/pi/`](../deploy/pi/), mirroring the Pi's `/etc` paths. With both DHCP
connections active, run `sudo bash deploy/pi/install.sh` from the repo root.
It installs scoped ARP/source routing; rollback is
`sudo /root/naila-dual-uplink-backup/rollback.sh`.

Verified on this pair at 2026-10-15 17:31 America/Boise: native release build
passed in 23m 16s using Rust 1.99.1, protoc 3.21.12 and system Opus 1.5.2.
Both Pi-local and Mac-to-Pi `/health` returned `grpc_connected: true`,
`ai_server.health: healthy`, version `0.1.0` and all four models loaded.
Hub logs confirmed the remote address and `bidirectional stream established`.
No audio peer was connected; this verifies gRPC connectivity and model readiness.

## 1. Image and boot the Pi

Use [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html).
Select the actual Pi model, **Raspberry Pi OS Lite (64-bit)** and the intended
microSD card (writing erases that card; Imager handles formatting). If Imager
does not offer the 64-bit image for that model, stop and record the model.
Customise hostname `hub`, username `hub`, a chosen password, timezone
`America/Boise`, and SSH using the Mac's public key (`~/.ssh/id_ed25519.pub`),
or password login if no key exists. Configure Wi-Fi SSID/password/country if
not using Ethernet. Write, verify, eject, insert in the Pi and power it on.
Use the same LAN as the Mac; avoid guest/client isolation. Ethernet is preferred.

The provisioned pair uses hostname/user `hub`/`hub`; keep the existing image
and account when resuming this guide. On the Mac, `ssh hub@hub.local`.
If mDNS fails, find the Pi's DHCP lease
in the router and use its IP instead. Record the actual model, OS and address:

```sh
# Pi
tr -d '\0' </proc/device-tree/model
cat /etc/os-release
uname -m                       # expect aarch64
hostname -I
sudo timedatectl set-timezone America/Boise  # if the flashed image still uses UTC
sudo apt update
sudo apt install -y build-essential pkg-config cmake libopus-dev protobuf-compiler curl ca-certificates rsync
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustc --version
protoc --version
```

Build **on the Pi**: this avoids a Mac-to-Linux cross toolchain and matching
ARM Opus/C libraries. It takes longer but makes this first install simpler.
Use current stable Rust (edition 2024; dependencies may need newer than 1.85).
[Rust installation reference](https://rust-lang.org/tools/install/).
No Python/model/audio capture packages are needed on the Pi.

## 2. Prepare and run the Mac AI server

In the Mac checkout, follow [AI server setup](../ai-server/README.md#setup)
for Python 3.12, uv, ffmpeg, dependencies and model downloads if not already
installed. From `ai-server/`, the setup commands are `uv venv --python 3.12`,
`uv sync`, and `bash download_models.sh`; install ffmpeg with `brew install ffmpeg`.
The normal server also starts MQTT before gRPC. If there is no local broker,
install one with `brew install mosquitto`, then run `mosquitto -p 1883` in a
separate terminal. With no config file it listens locally. Confirm the actual
listener with `lsof -nP -iTCP:1883 -sTCP:LISTEN` before starting the AI server.
`brew services start mosquitto` reported success on this Mac without a working
listener (missing `mosquitto.conf`), so service status alone is insufficient.
Leave that broker local; the Pi hub only needs gRPC.

```sh
# Mac: find the LAN interface/address (substitute en1 etc. if appropriate)
networksetup -listallhardwareports
ipconfig getifaddr en0
cd /Users/j/code/robo-naila/ai-server
GRPC_HOST=0.0.0.0 GRPC_PORT=50051 MQTT_BROKER_HOST=localhost \
  STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 STT_VAD_FILTER=false \
  NAILA_MEMORY_DB_PATH=:memory: uv run python main.py
```

Wait for `grpc_server_started`; model initialization alone is insufficient.
Existing defaults already bind gRPC on `0.0.0.0`; the explicit setting above
also overrides a shell configured for loopback. CPU STT avoids CUDA requirements
on the Mac. The temporary memory setting keeps deployment checks out of stored
conversation history. Existing model paths/defaults apply; do not copy placeholder
model paths or MQTT credentials from `.env.example` into the shell.
Allow incoming connections for the server's Python executable if macOS prompts.
The Pi must reach **Mac LAN IP TCP 50051**; `127.0.0.1` on the Pi means the Pi.
Keep the Mac awake and use its current DHCP address (or reserve a lease).

## 3. Copy this checkout, build and start the Pi hub

Copy the current checkout, including any uncommitted fixes. A remote clone
only includes pushed commits. Transfer the workspace manifests, hub and protos:

```sh
# Mac, repository root; replace hostname with Pi IP if necessary
ssh hub@hub.local 'mkdir -p ~/robo-naila/hub ~/robo-naila/proto'
rsync -av Cargo.toml Cargo.lock hub@hub.local:robo-naila/
rsync -av --exclude target hub/ hub@hub.local:robo-naila/hub/
rsync -av proto/ hub@hub.local:robo-naila/proto/
rsync -av deploy/pi/ hub@hub.local:robo-naila/deploy/pi/
```

```sh
# Pi, in the SSH session; substitute the Mac's actual LAN IP
cd ~/robo-naila
. "$HOME/.cargo/env"
cargo build --locked --release -p hub -j 1
sudo bash deploy/pi/install.sh --hub-only
sudoedit /etc/naila/hub.env  # set NAILA_AI_GRPC_URL; RUST_LOG=info
sudo systemctl start naila-hub
journalctl -u naila-hub -f
```

`-j 1` limits build memory demand. The service starts at boot, restarts on failure,
and logs to the journal. Installation preserves an existing `/etc/naila/hub.env`;
the repo ships a placeholder template. Tracing stays off by default. The hub's
gRPC reconnect (1–30 seconds backoff) handles DHCP delay, so the unit orders after
`network.target` without waiting for `network-online.target`. After env edits,
run `sudo systemctl restart naila-hub`. To remove only the service (keeping env),
run `sudo bash deploy/pi/rollback.sh --hub-only`.

## 4. Check the actual pair

From a second Pi SSH session:

```sh
curl --fail --max-time 5 http://127.0.0.1:8080/health
```

Require `grpc_connected: true`, and inspect `ai_server` for the Mac's status,
version and loaded models. Its device labels currently reflect shared hardware
detection: `mps` in status does not prove every service runs on the GPU (the Mac
command above explicitly selects CPU STT). Hub logs should show the chosen
address, a received AI status and `bidirectional stream established`.
From the Mac also run:

```sh
curl --fail --max-time 5 http://hub.local:8080/health
```

`status: ok` alone only proves HTTP is alive. If gRPC remains disconnected,
check the Mac startup log and `lsof -nP -iTCP:50051 -sTCP:LISTEN`; then check
the URL, Mac firewall and LAN isolation. A connection refusal usually means
wrong address/bind or no listener; a timeout suggests routing/firewall problems.
HTTP health works before gRPC connects, so repeat after the reconnect delay.

For later WebRTC clients, allow Mac/client → Pi TCP 8080 and LAN UDP ICE/RTP
traffic (dynamic ports). The hub has no STUN/TURN; use a mutually reachable LAN.
Health and stream establishment prove signaling/status and an open gRPC stream,
not STT/LLM/TTS or an audio round trip. Save both health responses, hub/server
logs and actual hardware/OS facts as deployment evidence. The real audio path
and measured performance remain gates for the later #38 implementation.
