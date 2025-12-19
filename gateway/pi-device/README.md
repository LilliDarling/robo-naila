# Pi Device

WebRTC audio client for NAILA system. Captures audio from USB microphone and streams to Pi Gateway for AI processing.

## Setup

### Prerequisites

- Raspberry Pi (any model with WiFi)
- USB Microphone
- Speaker (optional, for TTS playback)
- Python 3.11+
- Network access to MQTT broker

### Hardware Setup

1. Plug in USB microphone
2. Verify detection: `arecord -l`
3. (Optional) Connect speaker for audio output

### Installation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git \
  libportaudio2 portaudio19-dev \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
  pkg-config ffmpeg

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Copy files to Pi (run from your local machine)
scp -r gateway/pi-device pi@192.168.50.151:~/

# SSH into Pi
ssh pi@192.168.50.151
cd ~/pi-device

# Setup virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Configuration

Set environment variables or create a `.env` file:

```bash
# MQTT Broker (Pi #1)
export MQTT_HOST=192.168.50.82
export MQTT_PORT=1883

# Device identity (unique per device)
export DEVICE_ID=pi-mic-01

# Audio device index (optional - auto-detect if not set)
export AUDIO_DEVICE=1

# Logging
export LOG_LEVEL=INFO
```

## Usage

### Running as a Service (Recommended)

To run the device client automatically on startup:

```bash
# Copy service file to systemd
sudo cp pi-device.service /etc/systemd/system/

# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable pi-device

# Start the service
sudo systemctl start pi-device

# Check service status
sudo systemctl status pi-device

# View logs
journalctl -u pi-device -f
```

Service management commands:
```bash
# Stop the service
sudo systemctl stop pi-device

# Restart the service
sudo systemctl restart pi-device

# Disable auto-start on boot
sudo systemctl disable pi-device
```

### Manual Mode

#### List Audio Devices

Find your USB microphone's device index:

```bash
python main.py --list-devices
```

Example output:
```
Available audio devices:
------------------------------------------------------------
  [0] bcm2835 Headphones (out:8)
  [1] USB Audio Device (in:1)
  [2] USB Audio Device (out:2)
------------------------------------------------------------

Set AUDIO_DEVICE environment variable to select input device
Example: AUDIO_DEVICE=1 python main.py
```

#### Start Client

```bash
# With specific audio device
AUDIO_DEVICE=1 python main.py

# Or with all config
MQTT_HOST=192.168.50.82 DEVICE_ID=living-room-mic AUDIO_DEVICE=1 python main.py
```

## Architecture

```
Device (Pi #3)     Gateway (Pi #2)      AI Server
     │                  │                   │
     │◄──── WebRTC ────►│◄───── TCP ───────►│
     │     (Opus)       │      (PCM)        │
     │                  │                   │
     └──────── MQTT Signaling ──────────────┘
```

## Connection Flow

```
Device                      MQTT                     Gateway
  │                          │                          │
  │── publish status ───────►│─────────────────────────►│
  │   (online: true)         │                          │
  │                          │     creates WebRTC PC    │
  │                          │                          │
  │◄─────────────────────────│◄── publish offer ────────│
  │                          │                          │
  │   creates answer         │                          │
  │                          │                          │
  │── publish answer ───────►│─────────────────────────►│
  │                          │                          │
  │◄─── ICE candidates ─────►│◄─── ICE candidates ─────►│
  │                          │                          │
  │◄═══════ WebRTC Audio Connected ═══════════════════►│
```

## MQTT Topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `naila/devices/{id}/status` | Device → Gateway | Device online/offline |
| `naila/devices/{id}/signaling/offer` | Gateway → Device | SDP offer |
| `naila/devices/{id}/signaling/answer` | Device → Gateway | SDP answer |
| `naila/devices/{id}/signaling/ice` | Bidirectional | ICE candidates |

## File Structure

```
pi-device/
├── main.py                 # Entry point
├── device/
│   ├── audio_client.py     # Main device client class
│   ├── audio_handler.py    # Audio capture/playback
│   └── webrtc.py           # WebRTC peer connection
├── mqtt/
│   ├── client.py           # MQTT connection
│   └── signaling.py        # WebRTC signaling handlers
├── config/
│   └── settings.py         # Configuration
└── requirements.txt
```

## Troubleshooting

### Microphone not detected

```bash
# Check USB devices
lsusb

# Check ALSA devices
arecord -l

# Test recording
arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -d 5 test.wav
aplay test.wav
```

### Permission denied

Add user to audio group:
```bash
sudo usermod -a -G audio $USER
# Then logout and login again
```

### No audio playback

1. Check speaker connection
2. Verify output device: `aplay -l`
3. Test output: `speaker-test -t wav`

### WebRTC connection fails

1. Check MQTT connectivity: `mosquitto_sub -h <broker> -t "#" -v`
2. Verify gateway is running and subscribed to device topics
3. Check ICE candidates are being exchanged
4. Enable DEBUG logging: `LOG_LEVEL=DEBUG python main.py`

### Audio not flowing

1. Enable DEBUG logging to see WebRTC track status
2. Verify microphone is capturing audio: use `--list-devices` and test with `arecord`
3. Check network connectivity between device and gateway
4. Verify gateway is forwarding audio to AI server
