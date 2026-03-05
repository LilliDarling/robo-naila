"""Main orchestrator — transport-agnostic AI pipeline entry point"""

import base64
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone
from graphs.orchestration import NAILAOrchestrationGraph
from memory.conversation import memory_manager
from config.mqtt_topics import OUTPUT
from utils import get_logger


logger = get_logger(__name__)


class NAILAOrchestrator:
    """Main orchestrator for NAILA AI system.

    Both MQTT and gRPC route through this class to share the same
    LangGraph pipeline, conversation memory, and AI services.
    """

    def __init__(self, mqtt_service=None, llm_service=None, tts_service=None, vision_service=None):
        self.graph = NAILAOrchestrationGraph(llm_service=llm_service, tts_service=tts_service, vision_service=vision_service)
        self.mqtt_service = mqtt_service
        self.memory = memory_manager

    def set_llm_service(self, llm_service):
        """Set LLM service for the orchestration graph"""
        self.graph.llm_service = llm_service
        self.graph.response_generator.llm_service = llm_service

    def set_tts_service(self, tts_service):
        """Set TTS service for the orchestration graph"""
        self.graph.tts_service = tts_service
        self.graph.response_generator.tts_service = tts_service

    def set_vision_service(self, vision_service):
        """Set Vision service for the orchestration graph"""
        self.graph.vision_service = vision_service

    async def process_task(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task from MQTT — runs graph then publishes response via MQTT."""
        result = await self.process_task_with_callback(message_data)

        # Publish response via MQTT if service available
        task_id = message_data.get("task_id", "unknown")
        device_id = message_data.get("device_id", "unknown")
        if self.mqtt_service and result.get("response_text"):
            await self._publish_response(device_id, task_id, result)

        return result

    async def process_task_with_callback(
        self,
        message_data: Dict[str, Any],
        audio_delivery: Optional[Callable] = None,
        transport: str = "mqtt",
    ) -> Dict[str, Any]:
        """Process a task with optional transport-specific audio delivery.

        Args:
            message_data: Task payload (from MQTT message or gRPC request).
            audio_delivery: Async callback invoked by ResponseGenerator when
                TTS audio is ready. For gRPC this chunks and queues AudioOutput
                messages. For MQTT this is None (audio published separately).
            transport: Transport identifier ("mqtt" or "grpc").

        Returns:
            Final graph state dict with response_text, intent, etc.
        """
        task_id = message_data.get(
            "task_id", f"task_{datetime.now(timezone.utc).timestamp()}"
        )
        device_id = message_data.get("device_id", "unknown")

        logger.info("processing_task", task_id=task_id, device_id=device_id, transport=transport)

        # Get conversation context from shared memory
        context = self.memory.get_context(device_id)

        # Build initial state
        initial_state = {
            "device_id": device_id,
            "task_id": task_id,
            "input_type": message_data.get("input_type", "text"),
            "raw_input": message_data.get("transcription", message_data.get("query", "")),
            "context": context,
            "conversation_history": context.get("recent_exchanges", []),
            "confidence": message_data.get("confidence", 1.0),
            "image_data": message_data.get("image_data"),
            "visual_context": None,
        }

        # Build LangGraph config with transport callbacks
        config = None
        if audio_delivery or transport != "mqtt":
            config = {
                "configurable": {
                    "transport": transport,
                    "audio_delivery": audio_delivery,
                }
            }

        # Run orchestration graph
        result = await self.graph.run(initial_state, config=config)

        # Update shared memory
        if result.get("processed_text") and result.get("response_text"):
            self.memory.add_exchange(
                device_id,
                result["processed_text"],
                result["response_text"],
                metadata={"intent": result.get("intent")},
            )

        return result

    async def _publish_response(self, device_id: str, task_id: str, result: Dict[str, Any]):
        """Publish AI response via MQTT - Command server will handle device commands"""
        if not self.mqtt_service:
            return

        # Build AI text response
        ai_response = {
            "task_id": task_id,
            "source": "ai_server",
            "target_device": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": {
                "text": result.get("response_text", ""),
                "intent": result.get("intent", ""),
                "confidence": result.get("confidence", 1.0),
                "context": result.get("context", {}),
            },
        }

        # Publish text response - Command server subscribes to this
        self.mqtt_service.publish(
            OUTPUT.ai_response_text,
            ai_response,
            qos=1,
        )

        # Publish audio response if available
        if "response_audio" in result:
            audio_data = result["response_audio"]
            audio_response = {
                "task_id": task_id,
                "device_id": device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audio_data": base64.b64encode(audio_data.audio_bytes).decode("utf-8"),
                "format": audio_data.format,
                "sample_rate": audio_data.sample_rate,
                "duration_ms": audio_data.duration_ms,
                "text": audio_data.text,
                "metadata": {
                    "voice": "lessac",
                    "language": "en_US",
                    "synthesis_time_ms": audio_data.synthesis_time_ms,
                },
            }

            self.mqtt_service.publish(
                OUTPUT.ai_response_audio,
                audio_response,
                qos=1,
            )

            logger.info(
                "published_audio_response",
                duration_ms=audio_data.duration_ms,
                format=audio_data.format,
                task_id=task_id,
            )

        logger.info("published_ai_response", task_id=task_id)

    def cleanup(self):
        """Cleanup old conversations"""
        self.memory.cleanup_old_conversations()
