"""Main orchestrator that integrates with MQTT"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone
from graphs.orchestration_graph import NAILAOrchestrationGraph
from memory.conversation_memory import memory_manager


logger = logging.getLogger(__name__)


class NAILAOrchestrator:
    """Main orchestrator for NAILA AI system"""
    
    def __init__(self, mqtt_service=None):
        self.graph = NAILAOrchestrationGraph()
        self.mqtt_service = mqtt_service
        self.memory = memory_manager
    
    async def process_task(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task from MQTT"""
        task_id = message_data.get("task_id", f"task_{datetime.now(timezone.utc).timestamp()}")
        device_id = message_data.get("device_id", "unknown")
        
        logger.info(f"Processing task {task_id} from {device_id}")
        
        # Get conversation context
        context = self.memory.get_context(device_id)
        
        # Build initial state
        initial_state = {
            "device_id": device_id,
            "task_id": task_id,
            "input_type": "text",
            "raw_input": message_data.get("transcription", ""),
            "context": context,
            "conversation_history": context.get("recent_exchanges", []),
            "confidence": message_data.get("confidence", 1.0)
        }
        
        # Run orchestration graph
        result = await self.graph.run(initial_state)
        
        # Update memory
        if result.get("processed_text") and result.get("response_text"):
            self.memory.add_exchange(
                device_id,
                result["processed_text"],
                result["response_text"],
                metadata={"intent": result.get("intent")}
            )
        
        # Publish response via MQTT if service available
        if self.mqtt_service and result.get("response_text"):
            await self._publish_response(device_id, task_id, result)
        
        return result
    
    async def _publish_response(self, device_id: str, task_id: str, result: Dict[str, Any]):
        """Publish AI response via MQTT - Command server will handle device commands"""
        
        # Build AI response
        ai_response = {
            "task_id": task_id,
            "source": "ai_server",
            "target_device": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": {
                "text": result.get("response_text", ""),
                "intent": result.get("intent", ""),
                "confidence": result.get("confidence", 1.0),
                "context": result.get("context", {})
            }
        }
        
        # Publish AI response - Command server subscribes to this
        from config.mqtt_topics import OUTPUT
        self.mqtt_service.publish(
            OUTPUT.ai_response_text,
            ai_response,
            qos=1
        )
        
        logger.info(f"Published AI response for task {task_id}")
    
    def cleanup(self):
        """Cleanup old conversations"""
        self.memory.cleanup_old_conversations()