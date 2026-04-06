# WSL2 Development Setup

Run the full NAILA voice pipeline (audio-client → hub → ai-server) on a Windows machine using WSL2 and a USB headset.

## Prerequisites

- Windows 10 (build 19045+) or Windows 11 with WSL2
- WSLg enabled (default on recent Windows builds — provides PulseAudio via RDP)
- A USB headset or separate mic + speakers
- Ubuntu 24.04 (Noble) WSL distribution
- 16GB+ RAM recommended (AI models use ~6GB)

## 1. System Dependencies

```bash
# Audio libraries (PortAudio, Opus, FFmpeg)
sudo apt update
sudo apt install -y \
  portaudio19-dev libopus0 \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev \
  alsa-utils libasound2-plugins libpulse-dev \
  cmake build-essential protobuf-compiler \
  ffmpeg mosquitto mosquitto-clients

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

## 2. Fix Audio on WSL2

WSLg routes audio through PulseAudio over RDP, but the system PortAudio package is built without PulseAudio support. Two fixes are needed.

### 2a. Bridge ALSA to PulseAudio

```bash
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF
```

### 2b. Rebuild PortAudio with PulseAudio Support

The Ubuntu `libportaudio2` package only links ALSA and JACK — no PulseAudio. Rebuild from source:

```bash
cd /tmp
git clone https://github.com/PortAudio/portaudio.git
cd portaudio
cmake -B build -DPA_USE_PULSEAUDIO=ON -DPA_USE_ALSA=ON
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

Verify PulseAudio is linked:

```bash
ldd /usr/lib/x86_64-linux-gnu/libportaudio.so.2 | grep pulse
# Should show: libpulse.so.0
```

### 2c. Verify Audio Devices

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

You should see PulseAudio devices including `Default Sink`, `Default Source`, and RDP devices. If the list is empty, restart WSL (`wsl --shutdown` from PowerShell) and try again.

### 2d. Test Mic Capture

```bash
parecord --channels=1 --rate=48000 --format=s16le /tmp/test.raw &
echo "Speak now..." && sleep 3 && kill %1
python3 -c "
import numpy as np
a = np.fromfile('/tmp/test.raw', dtype=np.int16)
print(f'Peak: {np.max(np.abs(a))}/32767')
print('Mic working!' if np.max(np.abs(a)) > 500 else 'Silent — check Windows mic settings')
"
```

**Target:** Peak above 3000 when speaking normally. If too low:
- Windows Settings → Privacy & Security → Microphone → enable "Let desktop apps access your microphone"
- Windows Sound Settings → Input → select your headset mic, set volume to 80-100%
- Close apps with exclusive mic access (Discord, Teams, etc.)
- Restart WSL: `wsl --shutdown` from PowerShell

## 3. AI Server Setup

```bash
cd ai-server
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### Models

Models live in `ai-server/models/`. If not already downloaded:

```bash
bash download_models.sh
```

### GPU Acceleration (Optional)

CPU works but is slow (~2 tokens/s). For NVIDIA GPU with WSL2 CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python \
  --no-binary llama-cpp-python --force-reinstall --no-cache-dir
```

Verify: `python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`

**Note:** Even with CUDA for the LLM, the STT (faster-whisper) requires cuDNN. If you don't have cuDNN installed, force STT to CPU to avoid crashes (see Section 5).

## 4. Hub Setup

```bash
cd hub
cargo build --release
```

First build compiles proto files and all dependencies (~2-3 minutes).

## 5. Audio Client Setup

```bash
cd devices/audio-client
uv venv
source .venv/bin/activate
uv sync
```

## 6. Running the Full System

Open four terminals. Start each component in order:

### Terminal 1: MQTT Broker

```bash
sudo systemctl start mosquitto
```

### Terminal 2: AI Server

```bash
cd ai-server
source .venv/bin/activate
STT_VAD_FILTER=false STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 uv run python main.py
```

| Variable | Why |
|---|---|
| `STT_VAD_FILTER=false` | Hub already does VAD — double-filtering rejects short speech chunks |
| `STT_DEVICE=cpu` | Avoids cuDNN crashes if cuDNN is not installed |
| `STT_COMPUTE_TYPE=int8` | Best STT performance on CPU |

Wait for: `grpc_server_started`

### Terminal 3: Hub

```bash
cd hub
cargo run --release
```

Wait for: `signaling server listening addr=0.0.0.0:8080`

### Terminal 4: Audio Client

```bash
cd devices/audio-client
source .venv/bin/activate
uv run python -m audio_client --hub-url http://localhost:8080 --device-id dev-headset
```

Wait for: `WebRTC connected to hub`

### Verify

```bash
curl -s http://localhost:8080/health | python3 -m json.tool   # Hub + AI server status
curl -s http://localhost:8081/health | python3 -m json.tool   # Audio client status
```

## Troubleshooting

### No audio devices in `sounddevice.query_devices()`

- Confirm PulseAudio is running: `pactl info` (should show `Server String: unix:/mnt/wslg/PulseServer`)
- Confirm PortAudio has PulseAudio backend: `python3 -c "import sounddevice as sd; print(sd.query_hostapis())"` — look for `PulseAudio` entry
- If only ALSA/OSS appear, the PortAudio rebuild (Section 2b) didn't take effect. Run `sudo ldconfig` and retry.

### STT returns empty transcriptions

- Check mic volume (Section 2d) — peak should be above 3000
- Confirm `STT_VAD_FILTER=false` is set — the hub's VAD already filters silence
- Check AI server logs for `stt_transcription_text` at DEBUG level

### cuDNN errors / AI server crashes

- Set `STT_DEVICE=cpu` and `STT_COMPUTE_TYPE=int8`
- The cuDNN errors (`Unable to load libcudnn_ops.so`) mean faster-whisper can't find NVIDIA cuDNN libraries. CPU mode works fine.

### Audio client won't shut down with Ctrl+C

- Kill both processes: `ps aux | grep audio_client | grep -v grep` then `kill -9 <PIDs>`

### Hub gRPC reconnect loop

- AI server must be running first on port 50051
- Hub retries with exponential backoff (1s → 30s) — it will reconnect automatically once the AI server is up

### `pyproject.toml` warning about `extra-build-dependencies`

- Harmless `uv` warning — can be ignored
