import asyncio
import logging
from datetime import datetime, timezone
from mqtt.client import NailaMQTTClient
from config.mqtt_config import MQTTConfig


class MQTTConnectionManager:
    """Handles MQTT connection lifecycle with resilience and monitoring"""
    
    def __init__(self, config: MQTTConfig):
        self.config = config
        self.client = NailaMQTTClient(config)
        self.logger = logging.getLogger(__name__)
        
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
            self.logger.info("Connecting to MQTT broker...")
            self._start_time = datetime.now(timezone.utc)
            
            for attempt in range(retry_attempts):
                self._connection_attempts += 1
                
                try:
                    if self.client.connect():
                        self._connected = True
                        self.logger.info(f"Connected to {self.config.broker_host}:{self.config.broker_port}")
                        return True
                    else:
                        if attempt < retry_attempts - 1:
                            wait_time = min(2 ** attempt, 30)
                            self.logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise ConnectionError("Failed to connect after all retries")
                            
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = min(2 ** attempt, 30)
                        self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise ConnectionError(f"Failed to connect to MQTT broker: {e}")
            
            return False
    
    async def disconnect(self):
        """Graceful disconnection with cleanup"""
        if not self._connected:
            return
        
        async with self._connection_lock:
            self.logger.info("Disconnecting from MQTT broker...")
            
            if self._connected:
                self.client.disconnect()
                self._connected = False
            
            # Log final connection stats
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() if self._start_time else 0
            self.logger.info(
                f"Disconnected. Connection stats: {uptime:.1f}s uptime, "
                f"{self._connection_attempts} attempts, {self._reconnection_count} reconnections"
            )
    
    def subscribe(self, topic: str, qos: int = None):
        """Subscribe to topic if connected"""
        if not self._connected:
            self.logger.debug(f"Not connected, cannot subscribe to {topic}")
            return False
        
        if qos is None:
            qos = self.config.qos
        
        return self.client.subscribe(topic, qos)
    
    def publish(self, topic: str, payload: str, qos: int = None):
        """Publish message if connected"""
        if not self._connected:
            self.logger.debug(f"Not connected, cannot publish to {topic}")
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