import asyncio
from datetime import datetime, timezone
from mqtt.client import NailaMQTTClient
from config.mqtt import MQTTConfig
from utils import get_logger


class MQTTConnectionManager:
    """Handles MQTT connection lifecycle with resilience and monitoring"""

    def __init__(self, config: MQTTConfig):
        self.config = config
        self.client = NailaMQTTClient(config)
        self.logger = get_logger(__name__)
        
        # Connection state
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._start_time = None
        
        # Stats
        self._connection_attempts = 0
        self._reconnection_count = 0
    
    async def connect(self, retry_attempts: int = 5) -> bool:
        """Connect with retry logic and exponential backoff"""
        if self._connected:
            return True

        async with self._connection_lock:
            self.logger.info("mqtt_connecting")
            self._start_time = datetime.now(timezone.utc)

            for attempt in range(retry_attempts):
                self._connection_attempts += 1

                try:
                    if self.client.connect():
                        self._connected = True
                        self.logger.info("mqtt_connected", broker_host=self.config.broker_host, broker_port=self.config.broker_port)
                        return True
                    else:
                        if attempt >= retry_attempts - 1:
                            raise ConnectionError("Failed to connect after all retries")

                        wait_time = min(2 ** attempt, 30)
                        self.logger.warning("mqtt_connection_failed", attempt=attempt + 1, retry_in=wait_time)
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = min(2 ** attempt, 30)
                        self.logger.warning("mqtt_connection_error", attempt=attempt + 1, error=str(e), error_type=type(e).__name__, retry_in=wait_time)
                        await asyncio.sleep(wait_time)
                    else:
                        raise ConnectionError(f"Failed to connect to MQTT broker: {e}") from e

            return False
    
    async def disconnect(self):
        """Graceful disconnection with cleanup"""
        if not self._connected:
            return

        async with self._connection_lock:
            self.logger.info("mqtt_disconnecting")

            if self._connected:
                self.client.disconnect()
                self._connected = False

            # Log final connection stats
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
            self.logger.info(
                "mqtt_disconnected",
                uptime_seconds=round(uptime, 1),
                connection_attempts=self._connection_attempts,
                reconnection_count=self._reconnection_count
            )
    
    def subscribe(self, topic: str, qos: int):
        """Subscribe to topic if connected"""
        if not self._connected:
            self.logger.debug("mqtt_not_connected_subscribe", topic=topic)
            return False

        if qos is None:
            qos = self.config.qos

        return self.client.subscribe(topic, qos)

    def publish(self, topic: str, payload: str, qos: int):
        """Publish message if connected"""
        if not self._connected:
            self.logger.debug("mqtt_not_connected_publish", topic=topic)
            return False

        if qos is None:
            qos = self.config.qos

        return self.client.publish(topic, payload, qos)
    
    def set_message_callback(self, callback):
        """Set message callback on the underlying client"""
        self.client.set_message_callback(callback)
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected
    
    def get_connection_stats(self) -> dict:
        """Get connection performance statistics"""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
        return {
            "connected": self._connected,
            "uptime_seconds": uptime,
            "connection_attempts": self._connection_attempts,
            "reconnection_count": self._reconnection_count,
            "broker_host": self.config.broker_host,
            "broker_port": self.config.broker_port,
            "client_id": self.config.client_id
        }