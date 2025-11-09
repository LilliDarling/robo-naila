from config.mqtt import MQTTConfig
from mqtt.core.service_coordinator import NailaMQTTService
from mqtt.handlers.coordinator import ProtocolHandler
from services.llm import LLMService
from services.stt import STTService
from .lifecycle import ServerLifecycleManager
from utils import get_logger


logger = get_logger(__name__)


class NailaAIServer:
    """NAILA AI Server with modular architecture"""

    def __init__(self):
        # Initialize core components
        self.config = MQTTConfig.from_env()
        self.mqtt_service = NailaMQTTService(self.config)
        self.protocol_handlers = ProtocolHandler(self.mqtt_service)

        # Initialize AI services
        self.llm_service = LLMService()
        self.stt_service = STTService()

        # Initialize lifecycle manager
        self.lifecycle = ServerLifecycleManager(
            self.mqtt_service,
            self.protocol_handlers,
            self.llm_service,
            self.stt_service
        )
    
    async def start(self):
        """Start the AI server"""
        await self.lifecycle.start_server()
    
    async def stop(self):
        """Stop the AI server"""
        await self.lifecycle.stop_server()
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.lifecycle.is_running()