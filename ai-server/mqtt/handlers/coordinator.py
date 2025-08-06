from typing import Dict, Any
from mqtt.core.service_coordinator import NailaMQTTService
from .device_handlers import DeviceHandlers
from .ai_handlers import AIHandlers
from .system_handlers import SystemHandlers


class ProtocolHandler:
    """Coordinates all MQTT protocol handlers - clean, focused entry point"""
    
    def __init__(self, mqtt_service: NailaMQTTService):
        self.mqtt_service = mqtt_service
        
        # Initialize specialized handler modules
        self.device_handlers = DeviceHandlers(mqtt_service)
        self.ai_handlers = AIHandlers(mqtt_service)
        self.system_handlers = SystemHandlers(mqtt_service)
    
    def register_all_handlers(self):
        """Register all protocol handlers with the MQTT service"""
        self.device_handlers.register_handlers()
        self.ai_handlers.register_handlers()
        self.system_handlers.register_handlers()
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics from all handlers"""
        device_stats = await self.device_handlers.get_performance_stats()
        ai_stats = await self.ai_handlers.get_performance_stats()
        system_stats = await self.system_handlers.get_performance_stats()
        
        return {
            "total_active_devices": device_stats["active_devices"],
            "device_states": device_stats["device_states"],
            "conversation_contexts": device_stats["conversation_contexts"],
            "task_queue_size": device_stats.get("task_queue_size", 0),
            "task_queue_max": device_stats.get("task_queue_max", 0),
            "handlers": {
                "device": len([m for m in dir(self.device_handlers) if m.startswith('handle_')]),
                "ai": len([m for m in dir(self.ai_handlers) if m.startswith('handle_')]),
                "system": len([m for m in dir(self.system_handlers) if m.startswith('handle_')])
            }
        }