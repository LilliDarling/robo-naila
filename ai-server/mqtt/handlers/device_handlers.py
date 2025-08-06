import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage
from .base import BaseHandler


# Topic patterns for device messages
TOPIC_DEVICE_SENSORS = "naila/devices/+/sensors/+"
TOPIC_DEVICE_WAKE_WORD = "naila/devices/+/audio/wake_word"
TOPIC_DEVICE_AUDIO_STREAM = "naila/devices/+/audio/stream"
TOPIC_DEVICE_VISION_EVENT = "naila/devices/+/vision/event"
TOPIC_DEVICE_VISION_FRAME = "naila/devices/+/vision/frame"
TOPIC_DEVICE_HEARTBEAT = "naila/devices/+/status/heartbeat"


class DeviceHandlers(BaseHandler):
    """Handlers for device-related MQTT messages (sensors, audio, vision, status)"""
    
    def __init__(self, mqtt_service):
        super().__init__(mqtt_service)
        self._task_queue = asyncio.Queue(maxsize=1000)  # For high-frequency processing
    
    def register_handlers(self):
        """Register all device-related handlers"""
        handlers = {
            TOPIC_DEVICE_SENSORS: self.handle_sensor_data,
            TOPIC_DEVICE_WAKE_WORD: self.handle_wake_word,
            TOPIC_DEVICE_AUDIO_STREAM: self.handle_audio_stream,
            TOPIC_DEVICE_VISION_EVENT: self.handle_vision_event,
            TOPIC_DEVICE_VISION_FRAME: self.handle_vision_frame,
            TOPIC_DEVICE_HEARTBEAT: self.handle_device_heartbeat,
        }
        
        for topic, handler in handlers.items():
            self.mqtt_service.register_handler([topic], handler)
    
    async def handle_sensor_data(self, message: MQTTMessage):
        """Handle incoming sensor data from devices"""
        if not message.device_id or not message.subtype:
            return
        
        device_state = await self._get_or_create_device_state(message.device_id)
        await device_state.update_sensor(message.subtype, message.payload, message.timestamp)
        
        # Store sensor data for AI context, but don't trigger direct actions
        # Battery monitoring could trigger AI notifications
        if message.subtype == "battery" and message.payload.get("percentage", 100) < 20:
            await self._handle_low_battery_for_ai(message.device_id, message.payload)
    
    async def handle_wake_word(self, message: MQTTMessage):
        """Handle wake word detection - trigger AI listening mode"""
        if not message.device_id or not message.payload.get("detected"):
            return
        
        confidence = message.payload.get("confidence", 0)
        if confidence < 0.7:
            return
        
        self.logger.info(f"Wake word detected: {message.device_id} ({confidence:.2f})")
        
        # Update conversation context to indicate listening mode
        context = await self._get_or_create_conversation_context(message.device_id)
        context.listening_mode = True
        context.wake_word_timestamp = message.timestamp
        
        # AI orchestration will handle the response
        orchestration_data = {
            "event": "wake_word_detected",
            "device_id": message.device_id,
            "confidence": confidence,
            "timestamp": message.timestamp
        }
        self.mqtt_service.publish_ai_orchestration("wake_word", orchestration_data, qos=1)
    
    async def handle_audio_stream(self, message: MQTTMessage):
        """Handle incoming audio stream data"""
        if not message.device_id or not message.binary_data:
            return
        
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
    
    async def _handle_low_battery_for_ai(self, device_id: str, battery_data: Dict[str, Any]):
        """Handle low battery - trigger AI to generate appropriate response"""
        percentage = battery_data.get("percentage", 100)
        if percentage > 20:  # Only actual low battery
            return
        
        # Trigger AI to generate a low battery notification/response
        ai_task_data = {
            "task_id": f"battery_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": device_id,
            "task_type": "battery_alert",
            "battery_percentage": percentage,
            "severity": "critical" if percentage < 10 else "warning",
            "priority": "high" if percentage < 10 else "normal"
        }
        
        # Let AI orchestration handle generating appropriate user notification
        self.mqtt_service.publish_ai_orchestration("system_alert", ai_task_data, qos=1)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get device handler performance statistics"""
        base_stats = await super().get_performance_stats()
        base_stats.update({
            "task_queue_size": self._task_queue.qsize(),
            "task_queue_max": self._task_queue.maxsize
        })
        return base_stats