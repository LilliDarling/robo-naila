#!/usr/bin/env python3
"""
Pi Gateway - WebRTC Audio Gateway for NAILA AI Server

Usage:
    python main.py                    # Normal mode (requires AI server)
    python main.py --loopback         # Loopback mode (echo audio back)
"""

import argparse
import asyncio
import logging
import signal
import sys

from config import settings
from gateway import AudioGateway
from mqtt import MQTTClient, SignalingHandler
from tcp import AIServerClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class GatewayApp:
    """Main application class."""
    
    def __init__(self, loopback_mode: bool = False):
        self.loopback_mode = loopback_mode
        self.mqtt = MQTTClient()
        self.gateway = AudioGateway(loopback_mode=loopback_mode)
        self.signaling = SignalingHandler(self.mqtt, self.gateway)
        self.ai_client: AIServerClient | None = None
        self._shutdown = asyncio.Event()
    
    async def start(self):
        """Start the gateway."""
        logger.info(f"Starting Pi Gateway (loopback={self.loopback_mode})")
        
        # Connect to MQTT
        await self.mqtt.connect()
        await self.signaling.setup()
        
        # Connect to AI server (unless loopback mode)
        if not self.loopback_mode:
            self.ai_client = AIServerClient()
            try:
                await self.ai_client.connect()
                self.gateway.ai_client = self.ai_client
                self.gateway.on_utterance = self._handle_utterance
            except Exception as e:
                logger.warning(f"AI server not available: {e}")
                logger.warning("Running without AI server - audio will be discarded")
        
        # Publish gateway status
        await self.mqtt.publish(
            f"gateway/status",
            {
                "gateway_id": settings.gateway_id,
                "online": True,
                "loopback_mode": self.loopback_mode
            },
            retain=True
        )
        
        logger.info("Gateway ready")
        
        # Start MQTT listener
        try:
            await self.mqtt.listen()
        except asyncio.CancelledError:
            pass
    
    async def _handle_utterance(self, device_id: str, audio: "np.ndarray"):
        """Handle completed utterance from device."""
        if not self.ai_client or not self.ai_client.connected:
            logger.warning(f"[{device_id}] AI server not connected, dropping audio")
            return
        
        # Send to AI server
        logger.info(f"[{device_id}] Sending {len(audio)} samples to AI server")
        response_audio = await self.ai_client.process_audio(device_id, audio)
        
        if response_audio is not None and len(response_audio) > 0:
            logger.info(f"[{device_id}] Received {len(response_audio)} samples from AI server")
            await self.gateway.send_tts_audio(device_id, response_audio)
    
    async def stop(self):
        """Stop the gateway."""
        logger.info("Shutting down gateway...")
        
        # Publish offline status
        try:
            await self.mqtt.publish(
                f"gateway/status",
                {
                    "gateway_id": settings.gateway_id,
                    "online": False
                },
                retain=True
            )
        except Exception:
            pass
        
        # Close all sessions
        await self.gateway.close_all()
        
        # Disconnect from AI server
        if self.ai_client:
            await self.ai_client.disconnect()
        
        # Disconnect from MQTT
        await self.mqtt.disconnect()
        
        logger.info("Gateway stopped")


async def main(loopback_mode: bool = False):
    """Main entry point."""
    app = GatewayApp(loopback_mode=loopback_mode)
    
    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(app.stop())
        loop.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await app.start()
    except KeyboardInterrupt:
        pass
    finally:
        await app.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi Gateway")
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Enable loopback mode (echo audio back without AI server)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(loopback_mode=args.loopback))
    except KeyboardInterrupt:
        sys.exit(0)
