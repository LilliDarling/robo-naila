import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
from .models import MQTTMessage, MQTTEventHandler, TopicCategory


class MessageRouter:
    """High-performance message routing with caching and topic parsing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Handler management
        self.event_handlers: Dict[str, List[MQTTEventHandler]] = {}
        self._handler_cache: Dict[str, List[MQTTEventHandler]] = {}
        
        # Performance metrics
        self._message_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def register_handler(self, topics: List[str], handler_func) -> MQTTEventHandler:
        """Register an event handler for specific topics"""
        event_handler = MQTTEventHandler(handler_func, topics)
        
        for topic in topics:
            if topic not in self.event_handlers:
                self.event_handlers[topic] = []
            self.event_handlers[topic].append(event_handler)
        
        # Clear cache when handlers change
        self._handler_cache.clear()
        return event_handler
    
    def parse_message(self, topic: str, payload: str) -> MQTTMessage:
        """Parse incoming message into structured format"""
        self._message_count += 1
        binary_data = None
        
        # Fast JSON parsing with fallback
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError:
            if payload.startswith('data:'):
                binary_data = payload.encode('utf-8')
                parsed_payload = {"binary": True, "size": len(binary_data)}
            else:
                parsed_payload = {"raw": payload}

        topic_info = self._parse_topic(topic)
        
        return MQTTMessage(
            topic=topic,
            payload=parsed_payload,
            category=topic_info["category"],
            device_id=topic_info["device_id"],
            message_type=topic_info["message_type"],
            subtype=topic_info["subtype"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            qos=0,
            binary_data=binary_data
        )
    
    def _parse_topic(self, topic: str) -> dict:
        """
        Parse MQTT topic according to NAILA protocol structure.
        
        Topic formats:
        - naila/devices/{device_id}/{message_type}/{subtype}
        - naila/ai/processing/{processing_type}/{device_id}
        - naila/ai/orchestration/{orchestration_type}
        - naila/ai/responses/{response_type}/{device_id}
        - naila/system/{system_type}/{subtype}
        """
        parts = topic.split('/')
        
        # Initialize result dictionary
        result = {
            "category": None,
            "device_id": None,
            "message_type": None,
            "subtype": None
        }
        
        # Validate basic structure: must start with "naila" and have at least 3 parts
        if len(parts) < 3 or parts[0] != "naila":
            return result
        
        # Topic structure indexes
        PROTOCOL_PREFIX = 0  # "naila"
        CATEGORY = 1         # "devices", "ai", "system"
        
        # Parse category
        category_str = parts[CATEGORY]
        try:
            category = TopicCategory(category_str)
        except ValueError:
            return result
        
        result["category"] = category
        
        # Parse based on category type
        if category == TopicCategory.DEVICES:
            # Format: naila/devices/{device_id}/{message_type}/{subtype}
            if len(parts) >= 5:
                result["device_id"] = parts[2]
                result["message_type"] = parts[3]
                result["subtype"] = parts[4]
                
        elif category == TopicCategory.AI:
            # AI topics have different structures based on message type
            if len(parts) >= 3:
                ai_type = parts[2]
                result["message_type"] = ai_type
                
                if ai_type == "processing" and len(parts) >= 5:
                    # Format: naila/ai/processing/{processing_type}/{device_id}
                    result["subtype"] = parts[3]
                    result["device_id"] = parts[4]
                    
                elif ai_type == "orchestration" and len(parts) >= 4:
                    # Format: naila/ai/orchestration/{orchestration_type}
                    result["subtype"] = parts[3]
                    
                elif ai_type == "responses" and len(parts) >= 5:
                    # Format: naila/ai/responses/{response_type}/{device_id}
                    result["subtype"] = parts[3]
                    result["device_id"] = parts[4]
                    
        elif category == TopicCategory.SYSTEM:
            # Format: naila/system/{system_type}/{subtype}
            if len(parts) >= 3:
                result["message_type"] = parts[2]
                if len(parts) >= 4:
                    result["subtype"] = parts[3]
        
        return result
    
    async def route_message(self, message: MQTTMessage):
        """Route message to appropriate handlers with caching"""
        handlers_found = False
        
        # Use cached handlers if available
        cached_handlers = self._handler_cache.get(message.topic)
        if cached_handlers is not None:
            self._cache_hits += 1
            handlers_found = len(cached_handlers) > 0
            tasks = [handler.handle(message) for handler in cached_handlers]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self._cache_misses += 1
            matching_handlers = []
            for topic_pattern, handlers in self.event_handlers.items():
                if self._topic_matches(message.topic, topic_pattern):
                    matching_handlers.extend(handlers)
                    handlers_found = True

            self._handler_cache[message.topic] = matching_handlers

            if matching_handlers:
                tasks = [handler.handle(message) for handler in matching_handlers]
                await asyncio.gather(*tasks, return_exceptions=True)
        
        if not handlers_found:
            self.logger.debug(f"No handler found for topic: {message.topic}")
    
    def _topic_matches(self, actual_topic: str, pattern: str) -> bool:
        """
        Check if an actual topic matches a pattern with MQTT wildcards.
        
        MQTT wildcards:
        - '+' matches exactly one topic level (e.g., naila/devices/+/sensors/temperature)
        - '#' matches zero or more topic levels, must be at end (e.g., naila/devices/#)
        
        Examples:
        - Pattern: naila/devices/+/sensors/+ matches naila/devices/robot1/sensors/temperature
        - Pattern: naila/devices/# matches naila/devices/robot1/sensors/temperature
        """
        # Exact match is fastest
        if pattern == actual_topic:
            return True
        
        # No wildcards means no match possible
        if '+' not in pattern and '#' not in pattern:
            return False
        
        pattern_parts = pattern.split('/')
        topic_parts = actual_topic.split('/')
        
        return self._match_topic_parts(topic_parts, pattern_parts)
    
    def _match_topic_parts(self, topic_parts: List[str], pattern_parts: List[str]) -> bool:
        """
        Match topic parts against pattern parts with wildcard support.
        
        Algorithm:
        - Iterate through both lists simultaneously
        - '#' wildcard matches everything remaining (multi-level wildcard)
        - '+' wildcard matches exactly one level (single-level wildcard)
        - Regular strings must match exactly
        - Both lists must be fully consumed for a match (unless '#' is encountered)
        
        Args:
            topic_parts: The actual topic split into parts (e.g., ['naila', 'devices', 'robot1'])
            pattern_parts: The pattern split into parts (e.g., ['naila', 'devices', '+'])
            
        Returns:
            True if the topic matches the pattern, False otherwise
        """
        i = j = 0
        
        while i < len(topic_parts) and j < len(pattern_parts):
            if pattern_parts[j] == '#':
                # Multi-level wildcard matches everything remaining
                return True
            elif pattern_parts[j] == '+':
                # Single-level wildcard matches exactly one level
                i += 1
                j += 1
            elif pattern_parts[j] == topic_parts[i]:
                # Exact string match required
                i += 1
                j += 1
            else:
                return False
        
        # Both lists must be fully consumed for a match
        return i == len(topic_parts) and j == len(pattern_parts)
    
    def get_routing_stats(self) -> dict:
        """Get routing performance statistics"""
        cache_hit_rate = self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        
        return {
            "message_count": self._message_count,
            "error_count": self._error_count,
            "handlers_registered": len(self.event_handlers),
            "cache_size": len(self._handler_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate
        }