import asyncio
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage
from .base import BaseHandler


class AIHandlers(BaseHandler):
    """Handlers for AI processing and orchestration messages"""
    
    def register_handlers(self):
        """Register all AI-related handlers"""
        handlers = {
            "naila/ai/processing/stt/+": self.handle_stt_result,
            "naila/ai/processing/vision/+": self.handle_vision_analysis,
            "naila/ai/orchestration/main/task": self.handle_main_task,
            "naila/ai/orchestration/personality/response": self.handle_personality_response,
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
        """Handle main orchestration tasks - placeholder for AI agent integration"""
        task_id = message.payload.get("task_id")
        device_id = message.payload.get("device_id")
        transcription = message.payload.get("transcription", "")
        
        if not task_id or not device_id:
            return
        
        self.logger.info(f"Task {task_id}: {device_id} -> {transcription}")
        
        # TODO: Replace with actual AI agent integration
        # For now, simple pattern matching for demo
        response_text = self._generate_simple_response(transcription)
        
        # Fast response generation
        response_data = {
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_text": response_text,
            "audio_format": "wav",
            "sample_rate": 16000
        }
        
        # Non-blocking responses
        self.mqtt_service.publish_ai_response("audio", device_id, response_data, qos=1)
    
    def _generate_simple_response(self, transcription: str) -> str:
        """Fast pattern matching for common queries"""
        text_lower = transcription.lower()
        
        if "time" in text_lower:
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        elif "weather" in text_lower:
            return "I don't have access to weather data yet, but I'm working on it!"
        else:
            return "I heard you, but I'm still learning how to respond to that."
    
    async def handle_personality_response(self, message: MQTTMessage):
        """Handle personality-adjusted responses - placeholder"""
        # TODO: Implement personality system integration
        pass