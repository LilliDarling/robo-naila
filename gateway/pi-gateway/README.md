# Pi Gateway

WebRTC audio gateway for NAILA AI Server. Handles real-time audio streaming from devices and forwards to AI server for processing.

## Setup

### Prerequisites

- Raspberry Pi 4 (2GB+ RAM recommended)
- Python 3.11+
- Network access to MQTT broker and AI server

### Installation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
  pkg-config ffmpeg

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Copy files to Pi (run from your local machine)
scp -r gateway/pi-gateway pi2@192.168.50.150:~/

# SSH into Pi2
ssh pi2@192.168.50.150
cd ~/pi-gateway

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

# AI Server (optional for loopback testing)
export AI_SERVER_HOST=192.168.1.200
export AI_SERVER_PORT=9999

# Gateway identity
export GATEWAY_ID=gateway-01

# Logging
export LOG_LEVEL=INFO
```

## Usage

### Running as a Service (Recommended)

To run the gateway automatically on startup:

```bash
# Copy service file to systemd
sudo cp pi-gateway.service /etc/systemd/system/

# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable pi-gateway

# Start the service
sudo systemctl start pi-gateway

# Check service status
sudo systemctl status pi-gateway

# View logs
journalctl -u pi-gateway -f
```

Service management commands:
```bash
# Stop the service
sudo systemctl stop pi-gateway

# Restart the service
sudo systemctl restart pi-gateway

# Disable auto-start on boot
sudo systemctl disable pi-gateway
```

### Manual Mode

#### Loopback Mode (Testing)

Echo audio back without AI server - useful for testing WebRTC connection:

```bash
python main.py --loopback
```

#### Normal Mode

Requires AI server running:

```bash
python main.py
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

## MQTT Topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `naila/devices/{id}/status` | Device → Gateway | Device online/offline |
| `naila/devices/{id}/signaling/offer` | Gateway → Device | SDP offer |
| `naila/devices/{id}/signaling/answer` | Device → Gateway | SDP answer |
| `naila/devices/{id}/signaling/ice` | Bidirectional | ICE candidates |
| `naila/gateway/status` | Gateway → All | Gateway status |

## File Structure

```
pi-gateway/
├── main.py                 # Entry point
├── gateway/
│   ├── audio_gateway.py    # Main gateway class
│   ├── session.py          # Device session management
│   ├── vad.py              # Voice activity detection
│   └── tracks.py           # Audio track implementations
├── mqtt/
│   ├── client.py           # MQTT connection
│   └── signaling.py        # WebRTC signaling handlers
├── tcp/
│   └── ai_client.py        # TCP client to AI server
├── config/
│   └── settings.py         # Configuration
└── requirements.txt
```

## Troubleshooting

### WebRTC connection fails

1. Check MQTT connectivity: `mosquitto_sub -h <broker> -t "#" -v`
2. Verify device is publishing status with `capabilities: ["audio"]`
3. Check ICE candidates are being exchanged

### Audio not flowing

1. Enable DEBUG logging: `LOG_LEVEL=DEBUG python main.py`
2. Check VAD threshold - may need adjustment for your mic
3. Verify device microphone is working: `arecord -l`

### High latency

1. Use wired ethernet between gateway and AI server
2. Reduce VAD silence frames for faster utterance detection
3. Consider smaller Whisper model on AI server
