import asyncio
from typing import Callable, Optional, List
from enum import Enum
from dataclasses import dataclass
from utils import get_logger


class TopicCategory(Enum):
    DEVICES = "devices"
    AI = "ai"
    SYSTEM = "system"


class DeviceMessageType(Enum):
    SENSORS = "sensors"
    AUDIO = "audio"
    VISION = "vision"
    STATUS = "status"
    ACTIONS = "actions"


class AIMessageType(Enum):
    PROCESSING = "processing"
    ORCHESTRATION = "orchestration"
    RESPONSES = "responses"


class SystemMessageType(Enum):
    HEALTH = "health"
    UPDATES = "updates"
    SECURITY = "security"
    ANALYTICS = "analytics"


@dataclass
class MQTTMessage:
    topic: str
    payload: dict
    category: TopicCategory
    device_id: Optional[str]
    message_type: str
    subtype: Optional[str]
    timestamp: str
    qos: int = 0
    binary_data: Optional[bytes] = None


class MQTTEventHandler:
    __slots__ = ['handler_func', 'topics', 'logger']

    def __init__(self, handler_func: Callable[[MQTTMessage], None], topics: List[str]):
        self.handler_func = handler_func
        self.topics = topics
        self.logger = get_logger(__name__)

    async def handle(self, message: MQTTMessage):
        """Fast async handler execution with error isolation"""
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                await self.handler_func(message)
            else:
                self.handler_func(message)
        except Exception as e:
            # Isolate handler errors - don't let one handler crash the service
            self.logger.error(
                "handler_error",
                topic=message.topic,
                error=str(e),
                error_type=type(e).__name__
            )