from config.mqtt import MQTTConfig
from config.grpc import GRPCConfig
from mqtt.core.service_coordinator import NailaMQTTService
from mqtt.handlers.coordinator import ProtocolHandler
from rpc.server import GRPCServer
from rpc.service import NailaAIServicer
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService
from services.vision import VisionService
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

        # Initialize gRPC components
        self.grpc_config = GRPCConfig.from_env()
        self.grpc_servicer = NailaAIServicer()
        self.grpc_server = GRPCServer(self.grpc_config, self.grpc_servicer)

        # Initialize AI services
        self.llm_service = LLMService()
        self.stt_service = STTService()
        self.tts_service = TTSService()
        self.vision_service = VisionService()

        # Initialize lifecycle manager
        self.lifecycle = ServerLifecycleManager(
            self.mqtt_service,
            self.protocol_handlers,
            self.llm_service,
            self.stt_service,
            self.tts_service,
            self.vision_service,
            grpc_server=self.grpc_server,
            grpc_servicer=self.grpc_servicer,
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