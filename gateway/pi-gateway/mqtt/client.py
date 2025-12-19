import asyncio
import json
import logging
from typing import Callable, Awaitable

import aiomqtt

from config import settings

logger = logging.getLogger(__name__)


class MQTTClient:
    """Async MQTT client wrapper."""
    
    def __init__(self):
        self.client: aiomqtt.Client | None = None
        self._handlers: dict[str, Callable[[str, dict], Awaitable[None]]] = {}
        self._connected = False
    
    @property
    def prefix(self) -> str:
        return settings.topic_prefix
    
    async def connect(self):
        """Connect to MQTT broker."""
        self.client = aiomqtt.Client(
            hostname=settings.mqtt.host,
            port=settings.mqtt.port,
            username=settings.mqtt.username,
            password=settings.mqtt.password,
        )
        await self.client.__aenter__()
        self._connected = True
        logger.info(f"Connected to MQTT broker at {settings.mqtt.host}:{settings.mqtt.port}")
    
    async def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            await self.client.__aexit__(None, None, None)
            self._connected = False
    
    async def subscribe(self, topic: str, handler: Callable[[str, dict], Awaitable[None]]):
        """Subscribe to a topic with a handler."""
        full_topic = f"{self.prefix}/{topic}"
        self._handlers[full_topic] = handler
        await self.client.subscribe(full_topic)
        logger.debug(f"Subscribed to {full_topic}")
    
    async def publish(self, topic: str, payload: dict, retain: bool = False):
        """Publish JSON payload to topic."""
        full_topic = f"{self.prefix}/{topic}"
        await self.client.publish(
            full_topic,
            json.dumps(payload).encode(),
            retain=retain
        )
        logger.debug(f"Published to {full_topic}")
    
    async def publish_raw(self, topic: str, payload: bytes, retain: bool = False):
        """Publish raw bytes to topic."""
        full_topic = f"{self.prefix}/{topic}"
        await self.client.publish(full_topic, payload, retain=retain)
    
    async def listen(self):
        """Listen for messages and dispatch to handlers."""
        async for message in self.client.messages:
            topic = str(message.topic)
            
            # Find matching handler (supports wildcards)
            handler = self._find_handler(topic)
            if handler:
                try:
                    payload = json.loads(message.payload.decode())
                    await handler(topic, payload)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on {topic}")
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}")
    
    def _find_handler(self, topic: str) -> Callable | None:
        """Find handler for topic, supporting + wildcard."""
        # Exact match
        if topic in self._handlers:
            return self._handlers[topic]
        
        # Wildcard match
        for pattern, handler in self._handlers.items():
            if self._matches_pattern(pattern, topic):
                return handler
        
        return None
    
    def _matches_pattern(self, pattern: str, topic: str) -> bool:
        """Check if topic matches MQTT pattern with + wildcards."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        
        if len(pattern_parts) != len(topic_parts):
            return False
        
        for p, t in zip(pattern_parts, topic_parts):
            if p == "+":
                continue
            if p != t:
                return False
        
        return True
