import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

@dataclass
class DeviceState:
    """Thread-safe device state container"""
    device_id: str
    last_heartbeat: Optional[str] = None
    status: str = "unknown"
    sensors: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: int = 0
    firmware_version: str = "unknown"
    
    async def update_sensor(self, sensor_type: str, data: Any, timestamp: str):
        """Thread-safe sensor data update"""
        self.sensors[sensor_type] = {
            "data": data,
            "timestamp": timestamp
        }


@dataclass
class ConversationContext:
    """Thread-safe conversation context container"""
    device_id: str
    person_id: Optional[str] = None
    detected_emotion: str = "neutral"
    last_seen: Optional[str] = None
    visual_context: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None


class BaseHandler:
    """Base class for all MQTT message handlers"""
    
    def __init__(self, mqtt_service):
        self.mqtt_service = mqtt_service
        self.logger = logger
        
        # Shared state management with async locks
        self._device_states: Dict[str, DeviceState] = {}
        self._conversation_contexts: Dict[str, ConversationContext] = {}
        self._state_lock = asyncio.Lock()
        self._context_lock = asyncio.Lock()
        
        # Performance tracking
        self._active_devices: Set[str] = set()
    
    async def _get_or_create_device_state(self, device_id: str) -> DeviceState:
        """Thread-safe device state retrieval/creation"""
        async with self._state_lock:
            if device_id not in self._device_states:
                self._device_states[device_id] = DeviceState(device_id=device_id)
                self._active_devices.add(device_id)
            return self._device_states[device_id]
    
    async def _get_or_create_conversation_context(self, device_id: str) -> ConversationContext:
        """Thread-safe conversation context retrieval/creation"""
        async with self._context_lock:
            if device_id not in self._conversation_contexts:
                self._conversation_contexts[device_id] = ConversationContext(device_id=device_id)
            return self._conversation_contexts[device_id]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get handler performance statistics"""
        return {
            "active_devices": len(self._active_devices),
            "device_states": len(self._device_states),
            "conversation_contexts": len(self._conversation_contexts),
        }