# Pi hub → Mac AI server: first deployment

Run only the Rust hub on the Pi; orchestration and models stay on the Mac.
This establishes the deployment pair for #38. Benchmarks, OTel, YOLO measurement
and physical audio-device setup are separate work. These steps have not yet
been run on the operator's Pi; successful local tests do not prove deployment.

## 1. Image and boot the Pi

Use [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html).
Select the actual Pi model, **Raspberry Pi OS Lite (64-bit)** and the intended
microSD card (writing erases that card; Imager handles formatting). If Imager
does not offer the 64-bit image for that model, stop and record the model.
Customise hostname `naila-hub`, username `naila`, a chosen password, timezone
`America/Boise`, and SSH using the Mac's public key (`~/.ssh/id_ed25519.pub`),
or password login if no key exists. Configure Wi-Fi SSID/password/country if
not using Ethernet. Write, verify, eject, insert in the Pi and power it on.
Use the same LAN as the Mac; avoid guest/client isolation. Ethernet is preferred.

On the Mac, `ssh naila@naila-hub.local`. If mDNS fails, find the Pi's DHCP lease
in the router and use its IP instead. Record the actual model, OS and address:

```sh
# Pi
tr -d '\0' </proc/device-tree/model
cat /etc/os-release
uname -m                       # expect aarch64
hostname -I
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
install/start one with `brew install mosquitto` and `brew services start mosquitto`.
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

The branch changes are uncommitted, so cloning the remote alone will miss them.
Copy only the workspace manifests, hub and active protos from the Mac:

```sh
# Mac, repository root; replace hostname with Pi IP if necessary
ssh naila@naila-hub.local 'mkdir -p ~/robo-naila/hub ~/robo-naila/proto'
rsync -av Cargo.toml Cargo.lock naila@naila-hub.local:robo-naila/
rsync -av --exclude target hub/ naila@naila-hub.local:robo-naila/hub/
rsync -av proto/ naila@naila-hub.local:robo-naila/proto/
```

```sh
# Pi, in the SSH session; substitute the Mac's actual LAN IP
cd ~/robo-naila
. "$HOME/.cargo/env"
cargo build --locked --release -p hub -j 1
export NAILA_AI_GRPC_URL=http://192.168.1.10:50051
RUST_LOG=info ./target/release/hub
```

`-j 1` limits build memory demand. Leave this foreground process running;
Ctrl-C shuts it down. Re-export the URL after logging in again. The hub has
automatic gRPC reconnect (1–30 seconds backoff); no supervisor is required today.
An unset URL retains `http://127.0.0.1:50051`. Supply a nonempty HTTP URL.

## 4. Check the actual pair

From a second Pi SSH session:

```sh
curl --fail --max-time 5 http://127.0.0.1:8080/health
```

Require `grpc_connected: true`, and inspect `ai_server` for the Mac's status,
version and loaded models. Hub logs should show the chosen address, a received
AI status and `bidirectional stream established`. From the Mac also run:

```sh
curl --fail --max-time 5 http://naila-hub.local:8080/health
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
