# Pi Audio Device

WebRTC audio client for Raspberry Pi. Captures mic audio, runs acoustic echo cancellation, streams Opus to the hub via WebRTC, and plays TTS responses through the speaker.

## What it does

- Full-duplex audio via PortAudio (single callback for time-aligned mic + speaker)
- SpeexDSP echo cancellation so TTS playback doesn't feed back into mic
- aiortc WebRTC connection to hub (Opus, 48kHz, 20ms frames)
- Automatic reconnect with exponential backoff
- Health endpoint at `:8081/health` serving live metrics as JSON

## Setup

### System dependencies

```bash
sudo apt install libportaudio2 libspeexdsp-dev libopus0 \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev
```

### Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Configure

```bash
cp .env.example .env
# edit .env — set NAILA_HUB_URL and NAILA_DEVICE_ID at minimum
```

## Run

### Direct

```bash
uv run python -m pi_audio
```

Or with CLI overrides:

```bash
uv run python -m pi_audio --hub-url http://192.168.1.10:8080 --device-id pi-kitchen
```

### systemd

```bash
sudo cp pi-audio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pi-audio
```

Logs: `journalctl -u pi-audio -f`

## Tests

```bash
uv run pytest tests/
```

## Health

```bash
curl http://localhost:8081/health
```

Returns 200 when connected, 503 when disconnected. Same JSON body either way:

```json
{
  "connected": true,
  "uptime_seconds": 3421,
  "mic_frames_sent": 124000,
  "mic_frames_dropped": 0,
  "tts_frames_received": 82000,
  "tts_frames_played": 82000,
  "mic_queue_depth": 1,
  "speaker_queue_depth": 3,
  "audio_callback_duration_ms": 0.4,
  "connections": 3,
  "connection_failures": 2
}
```
