import base64
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage
from .base import BaseHandler
from agents.orchestrator import NAILAOrchestrator


# Topic patterns for AI processing and orchestration
TOPIC_DEVICE_AUDIO = "naila/device/+/audio"
TOPIC_AI_STT_RESULT = "naila/ai/processing/stt/+"
TOPIC_AI_VISION_ANALYSIS = "naila/ai/processing/vision/+"
TOPIC_AI_MAIN_TASK = "naila/ai/orchestration/main/task"
TOPIC_AI_PERSONALITY_RESPONSE = "naila/ai/orchestration/personality/response"


class AIHandlers(BaseHandler):
    """Handlers for AI processing and orchestration messages"""

    def __init__(self, mqtt_service):
        super().__init__(mqtt_service)
        self.orchestrator = NAILAOrchestrator(mqtt_service)
        self.stt_service = None
        self.tts_service = None
        self.vision_service = None

    def set_llm_service(self, llm_service):
        """Set LLM service for the orchestrator"""
        self.orchestrator.graph.llm_service = llm_service
        self.orchestrator.graph.response_generator.llm_service = llm_service

    def set_stt_service(self, stt_service):
        """Set STT service for audio transcription"""
        self.stt_service = stt_service

    def set_tts_service(self, tts_service):
        """Set TTS service for audio synthesis"""
        self.tts_service = tts_service
        self.orchestrator.set_tts_service(tts_service)

    def set_vision_service(self, vision_service):
        """Set Vision service for image analysis"""
        self.vision_service = vision_service
        self.orchestrator.set_vision_service(vision_service)

    def register_handlers(self):
        """Register all AI-related handlers"""
        handlers = {
            TOPIC_DEVICE_AUDIO: self.handle_audio_input,
            TOPIC_AI_STT_RESULT: self.handle_stt_result,
            TOPIC_AI_VISION_ANALYSIS: self.handle_vision_analysis,
            TOPIC_AI_MAIN_TASK: self.handle_main_task,
            TOPIC_AI_PERSONALITY_RESPONSE: self.handle_personality_response,
        }

        for topic, handler in handlers.items():
            self.mqtt_service.register_handler([topic], handler)

    async def handle_audio_input(self, message: MQTTMessage):
        """Handle audio input from devices and transcribe using STT service"""
        if not message.device_id:
            self.logger.warning("Audio message missing device_id")
            return

        # Check if STT service is available
        if not self.stt_service or not self.stt_service.is_ready:
            self.logger.warning(f"STT service not available, cannot process audio from {message.device_id}")
            # Publish error message to MQTT topic for the device
            error_topic = f"devices/{message.device_id}/audio/error"
            error_payload = {
                "error": "stt_unavailable",
                "message": "Speech-to-text service is currently unavailable. Please try again later.",
                "device_id": message.device_id
            }
            await self.mqtt_service.publish(error_topic, error_payload)
            return

        try:
            # Extract audio data from message
            audio_base64 = message.payload.get("audio_data")
            audio_format = message.payload.get("format", "wav")
            duration_ms = message.payload.get("duration_ms", 0)

            if not audio_base64:
                self.logger.error("Audio message missing audio_data field")
                return

            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_base64)

            self.logger.info(f"Received audio from {message.device_id}: {len(audio_bytes)} bytes, {audio_format} format, {duration_ms}ms")

            # Transcribe audio using STT service
            result = await self.stt_service.transcribe_audio(
                audio_data=audio_bytes,
                format=audio_format,
                language=message.payload.get("metadata", {}).get("language")
            )

            if not result.text:
                self.logger.warning(f"Empty transcription for audio from {message.device_id}")
                return

            self.logger.info(f"Transcribed: '{result.text}' (confidence: {result.confidence:.2f})")

            # Publish STT result for orchestration
            stt_result_data = {
                "device_id": message.device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transcription": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "audio_duration_ms": result.duration_ms,
                "transcription_time_ms": result.transcription_time_ms,
            }

            # Publish to STT result topic (which triggers orchestration)
            self.mqtt_service.publish_ai_processing(
                f"stt/{message.device_id}",
                stt_result_data,
                qos=1
            )

        except Exception as e:
            self.logger.error(f"Error processing audio from {message.device_id}: {e}", exc_info=True)

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
        """Handle computer vision analysis requests from devices - routes through orchestration"""
        if not message.device_id:
            self.logger.warning("Vision message missing device_id")
            return

        if not self.vision_service or not self.vision_service.is_ready:
            self.logger.warning(f"Vision service not available, cannot process image from {message.device_id}")
            error_topic = f"naila/device/{message.device_id}/vision/error"
            error_payload = {
                "error": "vision_unavailable",
                "message": "Vision service is currently unavailable. Please try again later.",
                "device_id": message.device_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.mqtt_service.publish(error_topic, error_payload)
            return

        try:
            image_base64 = message.payload.get("image_data")
            query = message.payload.get("query", "What do you see?")

            if not image_base64:
                self.logger.error("Vision message missing image_data field")
                return

            # Decode base64 to raw image bytes
            image_bytes = base64.b64decode(image_base64)
            self.logger.info(f"Received image from {message.device_id}: {len(image_bytes)} bytes")

            # Route all vision requests through orchestration for consistent handling
            task_data = {
                "task_id": f"vision_task_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_id": message.device_id,
                "input_type": "vision",
                "query": query,
                "image_data": image_bytes,
                "confidence": 1.0,
                "priority": "normal"
            }
            self.mqtt_service.publish_ai_orchestration("main/task", task_data, qos=1)

        except Exception as e:
            self.logger.error(f"Error processing image from {message.device_id}: {e}", exc_info=True)
    
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