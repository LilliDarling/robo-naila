# Pi hub deployment

Run the Rust hub on the Pi; models and orchestration run on the Mac.
Use [Mac deployment](../mac/README.md) to start the AI server first, then repeat
the checks below after rebuilding or changing either host's address.

The provisioned Pi is a Pi 4B, Debian 13, aarch64, accessed as `hub@hub.local`.
Keep its existing image/account when resuming. Ethernet is preferred; if mDNS
fails, find the Pi's DHCP lease in the router. Use a mutually reachable LAN
without guest/client isolation. Reserve the Mac's DHCP lease if possible.

## Prepare the Pi

For a new card only, use [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html)
with the actual Pi model and a 64-bit Lite image. Writing erases the selected
card. Configure hostname/user, timezone, SSH key and Wi-Fi before writing.
SSH into the Pi, then install native build dependencies and current stable Rust
(edition 2024; no Python/model/audio capture packages are needed):

```sh
tr -d '\0' </proc/device-tree/model
cat /etc/os-release
uname -m
hostname -I
sudo apt update
sudo apt install -y build-essential pkg-config cmake libopus-dev protobuf-compiler curl ca-certificates rsync
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustc --version
protoc --version
```

Build on the Pi to use its native ARM Opus/C libraries. For Ethernet and Wi-Fi
on the same subnet, this directory mirrors the Pi's `/etc` paths. With both DHCP
connections active, `sudo bash deploy/pi/install.sh` installs scoped ARP/source
routing and the hub service. Network rollback is
`sudo /root/naila-dual-uplink-backup/rollback.sh`. Use `--hub-only` below to leave
existing networking alone.

## Copy, build and start

From the Mac repository root, copy this checkout including uncommitted fixes;
a remote clone only includes pushed commits. Replace hub@hub.local as needed:

```sh
ssh hub@hub.local 'mkdir -p ~/robo-naila/hub ~/robo-naila/proto'
rsync -av Cargo.toml Cargo.lock hub@hub.local:robo-naila/
rsync -av --exclude target hub/ hub@hub.local:robo-naila/hub/
rsync -av proto/ hub@hub.local:robo-naila/proto/
rsync -av deploy/pi/ hub@hub.local:robo-naila/deploy/pi/
```

On the Pi:

```sh
cd ~/robo-naila
. "$HOME/.cargo/env"
cargo build --locked --release -p hub -j 1
sudo bash deploy/pi/install.sh --hub-only
sudoedit /etc/naila/hub.env  # NAILA_AI_GRPC_URL=http://MAC_LAN_IP:50051; RUST_LOG=info
sudo systemctl start naila-hub
journalctl -u naila-hub -f
```

`-j 1` limits build memory demand. Installation preserves an existing env file;
the template uses placeholders and tracing stays off by default. The service
starts at boot and restarts on failure; gRPC reconnect handles DHCP delay with
1–30s backoff. After env edits run `sudo systemctl restart naila-hub`.
Remove only the service, preserving env, with
`sudo bash deploy/pi/rollback.sh --hub-only`.

## Check the pair

On the Pi run `curl --fail --max-time 5 http://127.0.0.1:8080/health`; on the Mac
run `curl --fail --max-time 5 http://hub.local:8080/health`. Require
`grpc_connected: true` and healthy models under `ai_server`. Hub logs should show
the remote address, AI status and `bidirectional stream established`.
`status: ok` alone only proves HTTP is alive, so recheck after reconnect.

If gRPC remains disconnected, check the Mac's `grpc_server_started` log and
`lsof -nP -iTCP:50051 -sTCP:LISTEN`, its bind address/firewall and LAN isolation.
`127.0.0.1` on the Pi means the Pi, not the Mac. Refusal suggests no listener or
wrong address; timeouts suggest routing/firewall problems. Hardware labels such
as `mps` do not prove every model runs on GPU; check deployed configuration.

Allow client → Pi TCP 8080 and dynamic LAN UDP ICE/RTP; there is no STUN/TURN.
Health proves readiness, not an audio round trip. Use the
[voice benchmark](../../benchmarks/README.md) for the real audio path, and retain
health responses, logs and hardware/OS facts as deployment evidence.
