import asyncio
import logging
from typing import Dict, Any, List, Callable
from datetime import datetime, timezone
from config.mqtt_config import MQTTConfig
from .connection import MQTTConnectionManager
from .routing import MessageRouter
from .publisher import MQTTPublisher
from .models import MQTTEventHandler, DeviceMessageType, SystemMessageType


class NailaMQTTService:
    """Modular MQTT service coordinator"""
    
    def __init__(self, config: MQTTConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.connection = MQTTConnectionManager(config)
        self.router = MessageRouter()
        self.publisher = MQTTPublisher(self.connection)
        
        # Service state
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Setup message callback
        self.connection.set_message_callback(self._handle_message)
    
    def register_handler(self, topics: List[str], handler: Callable) -> MQTTEventHandler:
        """Register an event handler for specific topics"""
        event_handler = self.router.register_handler(topics, handler)
        
        # Subscribe immediately if connected
        if self.connection.is_connected():
            for topic in topics:
                self.connection.subscribe(topic, self.config.qos)
        
        return event_handler
    
    def register_device_handler(self, device_id: str, message_type: DeviceMessageType, 
                               subtype: str, handler: Callable):
        """Register handler for device messages"""
        topic = f"naila/devices/{device_id}/{message_type.value}/{subtype}"
        return self.register_handler([topic], handler)
    
    def register_ai_processing_handler(self, processing_type: str, device_id: str, handler: Callable):
        """Register handler for AI processing results"""
        topic = f"naila/ai/processing/{processing_type}/{device_id}"
        return self.register_handler([topic], handler)
    
    def register_ai_orchestration_handler(self, orchestration_type: str, handler: Callable):
        """Register handler for AI orchestration messages"""
        topic = f"naila/ai/orchestration/{orchestration_type}"
        return self.register_handler([topic], handler)
    
    def register_system_handler(self, system_type: SystemMessageType, subtype: str, handler: Callable):
        """Register handler for system messages"""
        topic = f"naila/system/{system_type.value}/{subtype}"
        return self.register_handler([topic], handler)
    
    def _handle_message(self, topic: str, payload: str):
        """Internal message handler - delegates to router"""
        try:
            # Parse message using router
            message = self.router.parse_message(topic, payload)
            message.qos = self.config.qos

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.router.route_message(message))
                else:
                    self.logger.debug(f"Event loop not running, skipping message routing: {topic}")
            except RuntimeError:
                # No event loop available - this can happen during startup/shutdown
                self.logger.debug(f"No event loop available for message routing: {topic}")
                
        except Exception as e:
            self.logger.error(f"Message handling error for {topic}: {e}")
    
    def publish_device_action(self, device_id: str, action_type: str, subtype: str, 
                             data: dict, qos: int = None):
        """Publish action command to device"""
        return self.publisher.publish_device_action(device_id, action_type, subtype, data, qos)
    
    def publish_ai_processing_result(self, processing_type: str, device_id: str, 
                                   data: dict, qos: int = None):
        """Publish AI processing result"""
        return self.publisher.publish_ai_processing_result(processing_type, device_id, data, qos)
    
    def publish_ai_orchestration(self, orchestration_type: str, data: dict, qos: int = None):
        """Publish AI orchestration message"""
        return self.publisher.publish_ai_orchestration(orchestration_type, data, qos)
    
    def publish_ai_response(self, response_type: str, device_id: str, data: dict, qos: int = None):
        """Publish AI response to device"""
        return self.publisher.publish_ai_response(response_type, device_id, data, qos)
    
    def publish_system_message(self, system_type: str, subtype: str, data: dict, qos: int = None):
        """Publish system message"""
        return self.publisher.publish_system_message(system_type, subtype, data, qos)
    
    def publish_error(self, service_name: str, error: dict, qos: int = None):
        """Publish error message"""
        return self.publisher.publish_error(service_name, error, qos)
    
    def publish(self, topic: str, data: dict, qos: int = None):
        """Generic publish method"""
        return self.publisher.publish(topic, data, qos)
    
    async def start(self):
        """Start the MQTT service with all components"""
        if self._running:
            return
        
        self.logger.info("Starting modular MQTT service...")
        
        # Connect with retry logic
        await self.connection.connect()
        
        # Wait for connection to stabilize
        await asyncio.sleep(0.5)
        
        # Subscribe to all registered topics
        for topic in self.router.event_handlers.keys():
            self.connection.subscribe(topic, self.config.qos)
        
        self._running = True
        connection_stats = self.connection.get_connection_stats()
        self.logger.info(f"MQTT service started (connected to {connection_stats['broker_host']}:{connection_stats['broker_port']})")
    
    async def stop(self):
        """Graceful shutdown with resource cleanup"""
        if not self._running:
            return
        
        self.logger.info("Stopping modular MQTT service...")
        
        # Signal shutdown
        self._shutdown_event.set()
        self._running = False
        
        # Disconnect
        await self.connection.disconnect()
        
        # Clear router cache
        self.router._handler_cache.clear()
        
        # Log final stats
        stats = self.get_stats()
        self.logger.info(
            f"MQTT service stopped. Final stats: {stats['total_messages']} messages, "
            f"{stats['total_errors']} errors, {stats['cache_hit_rate']:.1%} cache hit rate"
        )
    
    def is_running(self) -> bool:
        """Check if service is running"""
        return self._running and self.connection.is_connected()
    
    def is_connected(self) -> bool:
        """Check if MQTT client is connected"""
        return self.connection.is_connected()
    
    @property
    def event_handlers(self) -> Dict:
        """Access to event handlers for compatibility"""
        return self.router.event_handlers
    
    def get_stats(self) -> dict:
        """Get comprehensive service statistics"""
        connection_stats = self.connection.get_connection_stats()
        routing_stats = self.router.get_routing_stats()
        publish_stats = self.publisher.get_publish_stats()
        
        # Calculate error rates safely
        total_messages = routing_stats["message_count"] + publish_stats["publish_count"]
        total_errors = routing_stats["error_count"] + publish_stats["publish_errors"]
        error_rate = total_errors / max(total_messages, 1)
        
        return {
            "running": self._running,
            "connected": connection_stats["connected"],
            "uptime_seconds": connection_stats["uptime_seconds"],
            
            # Message processing
            "message_count": routing_stats["message_count"],
            "error_count": routing_stats["error_count"],
            "error_rate": error_rate,  # Add missing error_rate
            "total_messages": total_messages,
            "total_errors": total_errors,
            
            # Performance
            "handlers_registered": routing_stats["handlers_registered"],
            "cache_size": routing_stats["cache_size"],
            "cache_hit_rate": routing_stats["cache_hit_rate"],
            
            # Publishing
            "publish_count": publish_stats["publish_count"],
            "publish_error_rate": publish_stats["publish_error_rate"],
            
            # Connection
            "connection_attempts": connection_stats["connection_attempts"],
            "broker_host": connection_stats["broker_host"],
            "broker_port": connection_stats["broker_port"]
        }