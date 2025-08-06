import logging
from config.mqtt_config import MQTTConfig
from mqtt.core.service_coordinator import NailaMQTTService
from mqtt.handlers.coordinator import ProtocolHandlerCoordinator
from .lifecycle import ServerLifecycleManager


logger = logging.getLogger(__name__)


class NailaAIServer:
    """NAILA AI Server with modular architecture"""
    
    def __init__(self):
        # Initialize core components
        self.config = MQTTConfig.from_env()
        self.mqtt_service = NailaMQTTService(self.config)
        self.protocol_handlers = ProtocolHandlerCoordinator(self.mqtt_service)
        
        # Initialize lifecycle manager
        self.lifecycle = ServerLifecycleManager(self.mqtt_service, self.protocol_handlers)
    
    async def start(self):
        """Start the AI server"""
        await self.lifecycle.start_server()
    
    async def stop(self):
        """Stop the AI server"""
        await self.lifecycle.stop_server()
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.lifecycle.is_running()