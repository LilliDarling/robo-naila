#!/usr/bin/env python3
"""
Pi Device - WebRTC audio client that connects to Pi Gateway.

This runs on a Raspberry Pi with a USB microphone and optionally a speaker.
It connects to the gateway via MQTT signaling and streams audio over WebRTC.

Usage:
    python main.py
    
    # List audio devices first:
    python main.py --list-devices
    
    # Specify audio device:
    AUDIO_DEVICE=1 python main.py
"""

import argparse
import asyncio
import fractions
import json
import logging
import signal
import sys
from typing import Optional

import aiomqtt
import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class MicrophoneTrack(MediaStreamTrack):
    """
    Audio track that captures from USB microphone.
    """
    
    kind = "audio"
    
    def __init__(self, device: Optional[int] = None, sample_rate: int = 48000, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = 960  # 20ms at 48kHz
        
        self._timestamp = 0
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._stream: Optional[sd.InputStream] = None
        self._device = device
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice when audio is available."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to int16
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        
        # Put in queue (non-blocking)
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(
                    lambda: self._queue.put_nowait(pcm) if not self._queue.full() else None
                )
            except Exception:
                pass
    
    def start(self):
        """Start audio capture."""
        self._loop = asyncio.get_event_loop()
        self._stream = sd.InputStream(
            device=self._device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self._stream.start()
        logger.info(f"Microphone started (device={self._device})")
    
    def stop(self):
        """Stop audio capture."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
    
    async def recv(self) -> AudioFrame:
        """Get next audio frame for WebRTC."""
        try:
            pcm = await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            pcm = np.zeros(self.frame_size, dtype=np.int16)
        
        # Shape: (channels, samples)
        pcm = pcm.reshape(1, -1)
        
        frame = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        
        self._timestamp += self.frame_size
        
        return frame


class AudioPlayer:
    """Simple audio output using sounddevice."""
    
    def __init__(self, device: Optional[int] = None, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.device = device
        self._stream: Optional[sd.OutputStream] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._running = False
    
    def start(self):
        """Start audio output."""
        self._running = True
        self._stream = sd.OutputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=960,
            dtype=np.float32
        )
        self._stream.start()
        logger.info("Audio player started")
    
    def stop(self):
        """Stop audio output."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
    
    def play(self, pcm: np.ndarray):
        """Play audio samples."""
        if self._stream and self._running:
            # Convert int16 to float32
            audio = pcm.astype(np.float32) / 32767.0
            self._stream.write(audio.reshape(-1, 1))


class DeviceClient:
    """Main device client that connects to gateway."""
    
    def __init__(self):
        self.pc: Optional[RTCPeerConnection] = None
        self.mqtt: Optional[aiomqtt.Client] = None
        self.mic_track: Optional[MicrophoneTrack] = None
        self.player: Optional[AudioPlayer] = None
        self._connected = False
    
    @property
    def prefix(self) -> str:
        return settings.topic_prefix
    
    @property
    def device_id(self) -> str:
        return settings.device_id
    
    async def connect_mqtt(self):
        """Connect to MQTT broker."""
        self.mqtt = aiomqtt.Client(
            hostname=settings.mqtt_host,
            port=settings.mqtt_port,
            username=settings.mqtt_username,
            password=settings.mqtt_password,
        )
        await self.mqtt.__aenter__()
        logger.info(f"Connected to MQTT broker at {settings.mqtt_host}:{settings.mqtt_port}")
    
    async def subscribe_signaling(self):
        """Subscribe to signaling topics."""
        # SDP offers from gateway
        await self.mqtt.subscribe(
            f"{self.prefix}/devices/{self.device_id}/signaling/offer"
        )
        # ICE candidates from gateway
        await self.mqtt.subscribe(
            f"{self.prefix}/devices/{self.device_id}/signaling/ice"
        )
        logger.info("Subscribed to signaling topics")
    
    async def publish_status(self, online: bool = True):
        """Publish device status."""
        await self.mqtt.publish(
            f"{self.prefix}/devices/{self.device_id}/status",
            json.dumps({
                "device_id": self.device_id,
                "online": online,
                "capabilities": ["audio", "speaker"]
            }).encode(),
            retain=True
        )
        logger.info(f"Published status: online={online}")
    
    async def handle_offer(self, sdp: str):
        """Handle SDP offer from gateway."""
        logger.info("Received SDP offer")
        
        # Close existing connection if any
        if self.pc:
            await self.pc.close()
        
        self.pc = RTCPeerConnection()
        
        # Create and add microphone track
        self.mic_track = MicrophoneTrack(
            device=settings.audio_device,
            sample_rate=settings.sample_rate
        )
        self.pc.addTrack(self.mic_track)
        
        # Handle incoming audio (TTS from gateway)
        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info("Receiving audio track from gateway")
                asyncio.create_task(self._play_incoming_audio(track))
        
        # Handle connection state
        @self.pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                self._connected = True
                self.mic_track.start()
            elif self.pc.connectionState in ("failed", "closed", "disconnected"):
                self._connected = False
                if self.mic_track:
                    self.mic_track.stop()
        
        # Handle ICE candidates
        @self.pc.on("icecandidate")
        async def on_ice(candidate):
            if candidate:
                await self.mqtt.publish(
                    f"{self.prefix}/devices/{self.device_id}/signaling/ice",
                    json.dumps({
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }).encode()
                )
        
        # Set remote description (offer)
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        
        # Create and set local description (answer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        # Send answer to gateway
        await self.mqtt.publish(
            f"{self.prefix}/devices/{self.device_id}/signaling/answer",
            json.dumps({
                "type": "answer",
                "sdp": self.pc.localDescription.sdp
            }).encode()
        )
        logger.info("Sent SDP answer")
    
    async def handle_ice_candidate(self, candidate_data: dict):
        """Handle ICE candidate from gateway."""
        if self.pc and candidate_data.get("candidate"):
            candidate = RTCIceCandidate(
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                candidate=candidate_data["candidate"]
            )
            await self.pc.addIceCandidate(candidate)
    
    async def _play_incoming_audio(self, track):
        """Play incoming audio from gateway."""
        if not self.player:
            self.player = AudioPlayer(sample_rate=settings.sample_rate)
            self.player.start()
        
        while True:
            try:
                frame = await track.recv()
                pcm = frame.to_ndarray().flatten().astype(np.int16)
                self.player.play(pcm)
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                break
    
    async def listen(self):
        """Listen for MQTT messages."""
        async for message in self.mqtt.messages:
            topic = str(message.topic)
            
            try:
                payload = json.loads(message.payload.decode())
            except json.JSONDecodeError:
                continue
            
            if "signaling/offer" in topic:
                await self.handle_offer(payload.get("sdp", ""))
            elif "signaling/ice" in topic:
                await self.handle_ice_candidate(payload)
    
    async def close(self):
        """Clean up resources."""
        await self.publish_status(online=False)
        
        if self.mic_track:
            self.mic_track.stop()
        
        if self.player:
            self.player.stop()
        
        if self.pc:
            await self.pc.close()
        
        if self.mqtt:
            await self.mqtt.__aexit__(None, None, None)


async def main():
    """Main entry point."""
    client = DeviceClient()
    
    logger.info(f"Starting device client: {settings.device_id}")
    
    # Connect to MQTT
    await client.connect_mqtt()
    await client.subscribe_signaling()
    
    # Publish online status (triggers gateway to send offer)
    await client.publish_status(online=True)
    
    # Handle shutdown
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(client.close())
        loop.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await client.listen()
    except asyncio.CancelledError:
        pass
    finally:
        await client.close()


def list_audio_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        input_ch = dev['max_input_channels']
        output_ch = dev['max_output_channels']
        if input_ch > 0 or output_ch > 0:
            io_str = []
            if input_ch > 0:
                io_str.append(f"in:{input_ch}")
            if output_ch > 0:
                io_str.append(f"out:{output_ch}")
            print(f"  [{i}] {dev['name']} ({', '.join(io_str)})")
    print("-" * 60)
    print(f"\nSet AUDIO_DEVICE environment variable to select input device")
    print(f"Example: AUDIO_DEVICE=1 python main.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi Device Client")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
