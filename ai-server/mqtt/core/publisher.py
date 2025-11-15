import json
from typing import Optional
from utils import get_logger


class MQTTPublisher:
    """Publishing interface for NAILA protocol messages"""

    def __init__(self, connection_manager):
        self.connection = connection_manager
        self.logger = get_logger(__name__)
        
        # Publishing stats
        self._publish_count = 0
        self._publish_errors = 0
    
    def publish_ai_response(self, response_type: str, device_id: str, data: dict, 
                          qos: Optional[int] = None) -> bool:
        """Publish AI-generated response to device (TTS audio, generated text, etc)"""
        topic = f"naila/ai/responses/{response_type}/{device_id}"
        return self._publish_fast(topic, data, qos)
    
    def publish_ai_processing_result(self, processing_type: str, device_id: str, 
                                   data: dict, qos: Optional[int] = None) -> bool:
        """Publish AI processing result"""
        topic = f"naila/ai/processing/{processing_type}/{device_id}"
        return self._publish_fast(topic, data, qos)
    
    def publish_ai_orchestration(self, orchestration_type: str, data: dict, 
                               qos: Optional[int] = None) -> bool:
        """Publish AI orchestration message"""
        topic = f"naila/ai/orchestration/{orchestration_type}"
        return self._publish_fast(topic, data, qos)
    
    def publish_system_message(self, system_type: str, subtype: str, data: dict, 
                             qos: Optional[int] = None) -> bool:
        """Publish system message"""
        topic = f"naila/system/{system_type}/{subtype}"
        return self._publish_fast(topic, data, qos)
    
    def publish_error(self, service_name: str, error: dict, qos: Optional[int] = None) -> bool:
        """Publish error message"""
        topic = f"naila/system/errors/{service_name}"
        return self._publish_fast(topic, error, qos)
    
    def publish(self, topic: str, data: dict, qos: Optional[int] = None) -> bool:
        """Generic publish method"""
        return self._publish_fast(topic, data, qos)
    
    def _publish_fast(self, topic: str, data: dict, qos: Optional[int] = None) -> bool:
        """Optimized publish method with minimal overhead"""
        if not self.connection.is_connected():
            self.logger.debug("mqtt_not_connected_publish", topic=topic)
            return False

        self._publish_count += 1

        try:
            payload = json.dumps(data, separators=(',', ':'))  # Compact JSON
            return self.connection.publish(topic, payload, qos)

        except Exception as e:
            self._publish_errors += 1
            self.logger.error("publish_error", topic=topic, error=str(e), error_type=type(e).__name__)
            return False
    
    def get_publish_stats(self) -> dict:
        """Get publishing performance statistics"""
        error_rate = self._publish_errors / max(self._publish_count, 1)
        
        return {
            "publish_count": self._publish_count,
            "publish_errors": self._publish_errors,
            "publish_error_rate": error_rate
        }