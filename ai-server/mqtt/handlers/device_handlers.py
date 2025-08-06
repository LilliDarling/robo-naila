import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage
from .base import BaseHandler


class DeviceHandlers(BaseHandler):
    """Handlers for device-related MQTT messages (sensors, audio, vision, status)"""
    
    def __init__(self, mqtt_service):
        super().__init__(mqtt_service)
        self._task_queue = asyncio.Queue(maxsize=1000)  # For high-frequency processing
    
    def register_handlers(self):
        """Register all device-related handlers"""
        handlers = {
            "naila/devices/+/sensors/+": self.handle_sensor_data,
            "naila/devices/+/audio/wake_word": self.handle_wake_word,
            "naila/devices/+/audio/stream": self.handle_audio_stream,
            "naila/devices/+/vision/event": self.handle_vision_event,
            "naila/devices/+/vision/frame": self.handle_vision_frame,
            "naila/devices/+/status/heartbeat": self.handle_device_heartbeat,
        }
        
        for topic, handler in handlers.items():
            self.mqtt_service.register_handler([topic], handler)
    
    async def handle_sensor_data(self, message: MQTTMessage):
        """Handle incoming sensor data from devices"""
        if not message.device_id or not message.subtype:
            return
        
        # TODO: PERFORMANCE OPTIMIZATION - High-Frequency Sensor Data
        # Some sensors (IMU, proximity) can send 10-100 Hz data streams
        # These would benefit from FlatBuffers for:
        # - Batch sensor readings in single message
        # - Zero-copy access to sensor arrays
        # - Smaller payloads for better wireless performance
        
        device_state = await self._get_or_create_device_state(message.device_id)
        await device_state.update_sensor(message.subtype, message.payload, message.timestamp)
        
        # Only process special sensor events, not all sensor data
        if message.subtype == "touch" and message.payload.get("state") == "pressed":
            await self._handle_touch_interaction(message.device_id, message.payload)
        elif message.subtype == "proximity" and message.payload.get("distance", 100) < 20:
            await self._handle_proximity_alert(message.device_id, message.payload)
        elif message.subtype == "battery" and message.payload.get("percentage", 100) < 20:
            await self._handle_low_battery(message.device_id, message.payload)
    
    async def handle_wake_word(self, message: MQTTMessage):
        """Handle wake word detection - fast response critical"""
        if not message.device_id or not message.payload.get("detected"):
            return
        
        confidence = message.payload.get("confidence", 0)
        if confidence < 0.7:
            return
        
        self.logger.info(f"Wake word detected: {message.device_id} ({confidence:.2f})")

        command_data = {
            "command_id": f"cmd_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "start_listening",
            "timeout_seconds": 10
        }
        
        # Non-blocking publish
        self.mqtt_service.publish_device_action(
            message.device_id, "audio", "control", command_data, qos=1
        )
    
    async def handle_audio_stream(self, message: MQTTMessage):
        """Handle incoming audio stream data"""
        if not message.device_id or not message.binary_data:
            return
        
        # TODO: PERFORMANCE OPTIMIZATION - Audio Stream FlatBuffers
        # High-frequency audio chunks (50-100 msg/sec) would benefit most from FlatBuffers:
        # - Zero-copy audio data access
        # - ~10x faster parsing than JSON
        # - Smaller payload size for better network utilization
        
        # Queue for async processing to avoid blocking
        try:
            await self._task_queue.put_nowait({
                "type": "audio_processing",
                "device_id": message.device_id,
                "data": message.binary_data,
                "timestamp": message.timestamp
            })
        except asyncio.QueueFull:
            self.logger.warning(f"Task queue full, dropping audio from {message.device_id}")
    
    async def handle_vision_event(self, message: MQTTMessage):
        """Handle vision detection events"""
        if not message.device_id:
            return
        
        event_type = message.payload.get("event_type")
        if event_type == "face_detected":
            context = await self._get_or_create_conversation_context(message.device_id)
            context.person_id = message.payload.get("person_id")
            context.detected_emotion = message.payload.get("emotion", "neutral")
            context.last_seen = message.timestamp
    
    async def handle_vision_frame(self, message: MQTTMessage):
        """Handle incoming vision frames - queue for async processing"""
        if not message.device_id or not message.binary_data:
            return
        
        # TODO: PERFORMANCE OPTIMIZATION - Vision Frame FlatBuffers
        # High-frequency vision frames (30 fps) are second-highest priority for FlatBuffers:
        # - Large binary image data (JPEG/raw pixels)
        # - Zero-copy access to image buffer
        # - Metadata (resolution, format, timestamp) without separate parsing

        try:
            await self._task_queue.put_nowait({
                "type": "vision_processing",
                "device_id": message.device_id,
                "data": message.binary_data,
                "timestamp": message.timestamp
            })
        except asyncio.QueueFull:
            self.logger.warning(f"Task queue full, dropping vision frame from {message.device_id}")
    
    async def handle_device_heartbeat(self, message: MQTTMessage):
        """Handle device heartbeat messages - minimal processing for performance"""
        if not message.device_id:
            return
        
        device_state = await self._get_or_create_device_state(message.device_id)
        device_state.last_heartbeat = message.timestamp
        device_state.status = message.payload.get("status", "unknown")
        device_state.uptime_seconds = message.payload.get("uptime_seconds", 0)
        device_state.firmware_version = message.payload.get("firmware_version", "unknown")
        
        # Only log status changes, not every heartbeat
        if device_state.status == "offline":
            self.logger.warning(f"Device {message.device_id} went offline")
    
    async def _handle_touch_interaction(self, device_id: str, touch_data: Dict[str, Any]):
        """Handle touch sensor interactions - fast response"""
        duration = touch_data.get("duration", 0)
        if duration < 100:  # Ignore very short touches (noise)
            return
        
        command_data = {
            "command_id": f"cmd_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expression": "happy",
            "intensity": 0.7,
            "duration_ms": 1500
        }
        
        self.mqtt_service.publish_device_action(
            device_id, "display", "expression", command_data, qos=1
        )
    
    async def _handle_proximity_alert(self, device_id: str, proximity_data: Dict[str, Any]):
        """Handle proximity sensor alerts - fast alert response"""
        distance = proximity_data.get("distance", 100)
        if distance > 15:  # Only very close objects
            return
        
        command_data = {
            "command_id": f"cmd_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expression": "alert",
            "intensity": 0.9,
            "duration_ms": 800
        }
        
        self.mqtt_service.publish_device_action(
            device_id, "display", "expression", command_data, qos=1
        )
    
    async def _handle_low_battery(self, device_id: str, battery_data: Dict[str, Any]):
        """Handle low battery alerts - system alert"""
        percentage = battery_data.get("percentage", 100)
        if percentage > 20:  # Only actual low battery
            return
        
        alert_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_type": "low_battery",
            "device_id": device_id,
            "battery_percentage": percentage,
            "severity": "critical" if percentage < 10 else "medium"
        }
        
        self.mqtt_service.publish_system_message("health", "alert", alert_data, qos=1)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get device handler performance statistics"""
        base_stats = await super().get_performance_stats()
        base_stats.update({
            "task_queue_size": self._task_queue.qsize(),
            "task_queue_max": self._task_queue.maxsize
        })
        return base_stats