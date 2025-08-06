import json
import logging
from typing import Optional
from .models import DeviceMessageType, SystemMessageType


class MQTTPublisher:
    """Publishing interface for NAILA protocol messages"""
    
    def __init__(self, connection_manager):
        self.connection = connection_manager
        self.logger = logging.getLogger(__name__)
        
        # Publishing stats
        self._publish_count = 0
        self._publish_errors = 0
    
    def publish_device_action(self, device_id: str, action_type: str, subtype: str, 
                             data: dict, qos: Optional[int] = None) -> bool:
        """Publish action command to device"""
        topic = f"naila/devices/{device_id}/actions/{action_type}/{subtype}"
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
    
    def publish_ai_response(self, response_type: str, device_id: str, data: dict, 
                          qos: Optional[int] = None) -> bool:
        """Publish AI response to device"""
        topic = f"naila/ai/responses/{response_type}/{device_id}"
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
            self.logger.debug(f"Not connected, cannot publish to {topic}")
            return False
        
        self._publish_count += 1
        
        try:
            # TODO: PERFORMANCE OPTIMIZATION - FlatBuffers Serialization Point
            # For high-frequency topics (audio/vision streams), consider FlatBuffers:
            # if self._is_high_frequency_topic(topic):
            #     payload = self._serialize_flatbuffer(data, topic)
            # else:
            #     payload = json.dumps(data, separators=(',', ':'))  # Compact JSON for control
            
            payload = json.dumps(data, separators=(',', ':'))  # Compact JSON
            return self.connection.publish(topic, payload, qos)
            
        except Exception as e:
            self._publish_errors += 1
            self.logger.error(f"Publish error for {topic}: {e}")
            return False
    
    def get_publish_stats(self) -> dict:
        """Get publishing performance statistics"""
        error_rate = self._publish_errors / max(self._publish_count, 1)
        
        return {
            "publish_count": self._publish_count,
            "publish_errors": self._publish_errors,
            "publish_error_rate": error_rate
        }