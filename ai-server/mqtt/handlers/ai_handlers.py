import asyncio
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage
from .base import BaseHandler
from agents.orchestrator import NAILAOrchestrator


# Topic patterns for AI processing and orchestration
TOPIC_AI_STT_RESULT = "naila/ai/processing/stt/+"
TOPIC_AI_VISION_ANALYSIS = "naila/ai/processing/vision/+"
TOPIC_AI_MAIN_TASK = "naila/ai/orchestration/main/task"
TOPIC_AI_PERSONALITY_RESPONSE = "naila/ai/orchestration/personality/response"


class AIHandlers(BaseHandler):
    """Handlers for AI processing and orchestration messages"""
    
    def __init__(self, mqtt_service):
        super().__init__(mqtt_service)
        self.orchestrator = NAILAOrchestrator(mqtt_service)
    
    def register_handlers(self):
        """Register all AI-related handlers"""
        handlers = {
            TOPIC_AI_STT_RESULT: self.handle_stt_result,
            TOPIC_AI_VISION_ANALYSIS: self.handle_vision_analysis,
            TOPIC_AI_MAIN_TASK: self.handle_main_task,
            TOPIC_AI_PERSONALITY_RESPONSE: self.handle_personality_response,
        }
        
        for topic, handler in handlers.items():
            self.mqtt_service.register_handler([topic], handler)
    
    async def handle_stt_result(self, message: MQTTMessage):
        """Handle speech-to-text results - fast orchestration trigger"""
        if not message.device_id:
            return
        
        transcription = message.payload.get("transcription", "").strip()
        confidence = message.payload.get("confidence", 0)
        
        if not transcription or confidence < 0.7:
            return
        
        self.logger.info(f"STT: {message.device_id} -> '{transcription}' ({confidence:.2f})")
        
        # Fast task creation
        context = await self._get_or_create_conversation_context(message.device_id)
        task_data = {
            "task_id": f"task_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": message.device_id,
            "user_id": context.person_id or "unknown",
            "intent": "general_query",
            "transcription": transcription,
            "confidence": confidence,
            "priority": "normal"
        }
        
        # Non-blocking publish
        self.mqtt_service.publish_ai_orchestration("main/task", task_data, qos=1)
    
    async def handle_vision_analysis(self, message: MQTTMessage):
        """Handle computer vision analysis results"""
        if not message.device_id:
            return
        
        context = await self._get_or_create_conversation_context(message.device_id)
        context.visual_context = {
            "objects": message.payload.get("objects_detected", []),
            "scene": message.payload.get("scene_description", ""),
            "analyzed_at": message.timestamp
        }
    
    async def handle_main_task(self, message: MQTTMessage):
        """Handle main orchestration tasks with LangGraph integration"""
        task_id = message.payload.get("task_id")
        device_id = message.payload.get("device_id")
        transcription = message.payload.get("transcription", "")
        
        if not task_id or not device_id:
            return
        
        self.logger.info(f"Task {task_id}: {device_id} -> {transcription}")
        
        # Process with LangGraph orchestrator
        try:
            result = await self.orchestrator.process_task(message.payload)
            self.logger.info(f"Task {task_id} processed successfully")
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            # Fallback response
            response_data = {
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "response_text": "I encountered an issue processing your request. Please try again.",
                "audio_format": "wav",
                "sample_rate": 16000
            }
            self.mqtt_service.publish_ai_response("audio", device_id, response_data, qos=1)
    
    async def handle_personality_response(self, message: MQTTMessage):
        """Handle personality-adjusted responses - placeholder"""
        # TODO: Implement personality system integration
        pass