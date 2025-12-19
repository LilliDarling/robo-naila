import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway import AudioGateway
    from .client import MQTTClient

logger = logging.getLogger(__name__)


class SignalingHandler:
    """
    Handles WebRTC signaling over MQTT.
    
    Topics:
        - naila/devices/{device_id}/status - Device online/offline
        - naila/devices/{device_id}/signaling/offer - SDP offer (gateway → device)
        - naila/devices/{device_id}/signaling/answer - SDP answer (device → gateway)
        - naila/devices/{device_id}/signaling/ice - ICE candidates (bidirectional)
    """
    
    def __init__(self, mqtt: "MQTTClient", gateway: "AudioGateway"):
        self.mqtt = mqtt
        self.gateway = gateway
        
        # Wire up gateway ICE callback
        self.gateway.on_ice_candidate = self._on_gateway_ice_candidate
    
    async def setup(self):
        """Subscribe to signaling topics."""
        # Device status
        await self.mqtt.subscribe(
            "devices/+/status",
            self._handle_device_status
        )
        
        # SDP answers from devices
        await self.mqtt.subscribe(
            "devices/+/signaling/answer",
            self._handle_sdp_answer
        )
        
        # ICE candidates from devices
        await self.mqtt.subscribe(
            "devices/+/signaling/ice",
            self._handle_ice_candidate
        )
        
        logger.info("Signaling handler ready")
    
    async def _handle_device_status(self, topic: str, payload: dict):
        """Handle device coming online/offline."""
        # Extract device_id from topic: naila/devices/{device_id}/status
        parts = topic.split("/")
        device_id = parts[2] if len(parts) >= 3 else None
        
        if not device_id:
            return
        
        online = payload.get("online", False)
        capabilities = payload.get("capabilities", [])
        
        logger.info(f"Device {device_id}: online={online}, capabilities={capabilities}")
        
        if online and "audio" in capabilities:
            # Create WebRTC session and send offer
            await self._initiate_connection(device_id)
        elif not online:
            # Clean up session
            await self.gateway.close_session(device_id)
    
    async def _initiate_connection(self, device_id: str):
        """Initiate WebRTC connection to device."""
        session, sdp_offer = await self.gateway.create_session(device_id)
        
        # Send SDP offer to device
        await self.mqtt.publish(
            f"devices/{device_id}/signaling/offer",
            {
                "type": "offer",
                "sdp": sdp_offer
            }
        )
        logger.info(f"[{device_id}] SDP offer sent")
    
    async def _handle_sdp_answer(self, topic: str, payload: dict):
        """Handle SDP answer from device."""
        parts = topic.split("/")
        device_id = parts[2] if len(parts) >= 3 else None
        
        if not device_id:
            return
        
        sdp = payload.get("sdp")
        if sdp:
            await self.gateway.handle_answer(device_id, sdp)
            logger.info(f"[{device_id}] SDP answer processed")
    
    async def _handle_ice_candidate(self, topic: str, payload: dict):
        """Handle ICE candidate from device."""
        parts = topic.split("/")
        device_id = parts[2] if len(parts) >= 3 else None
        
        if not device_id:
            return
        
        await self.gateway.handle_ice_candidate(device_id, payload)
    
    async def _on_gateway_ice_candidate(self, device_id: str, candidate):
        """Send ICE candidate to device."""
        await self.mqtt.publish(
            f"devices/{device_id}/signaling/ice",
            {
                "candidate": candidate.candidate,
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex
            }
        )
