"""Main orchestrator — transport-agnostic AI pipeline entry point.

Runs the LangGraph pipeline and persists the completed turn. Transport-specific
concerns (MQTT publish, gRPC audio delivery) live in their respective layers —
this class only owns the pipeline lifecycle and memory persistence.
"""

from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone
from graphs.orchestration import NAILAOrchestrationGraph
from memory.conversation import ConversationMemory
from utils import get_logger


logger = get_logger(__name__)


class NAILAOrchestrator:
    """Main orchestrator for NAILA AI system.

    Both MQTT and gRPC route through this class to share the same
    LangGraph pipeline, conversation memory, and AI services. Transport-specific
    side effects (publishing, audio chunking) are the caller's responsibility.
    """

    def __init__(
        self,
        memory: ConversationMemory,
        llm_service=None,
        tts_service=None,
        vision_service=None,
    ):
        self.memory = memory
        self.graph = NAILAOrchestrationGraph(
            memory=memory,
            llm_service=llm_service,
            tts_service=tts_service,
            vision_service=vision_service,
        )

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

    async def process_task(
        self,
        message_data: Dict[str, Any],
        audio_delivery: Optional[Callable] = None,
        transport: str = "mqtt",
    ) -> Dict[str, Any]:
        """Run the graph for a single turn and persist the result.

        Args:
            message_data: Task payload (from MQTT message or gRPC request).
            audio_delivery: Async callback invoked by ResponseGenerator when
                TTS audio is ready. For gRPC this chunks and queues AudioOutput
                messages. For MQTT this is None (audio published separately).
            transport: Transport identifier ("mqtt" or "grpc"). Forwarded to
                the response generator via LangGraph config.

        Returns:
            Final graph state dict with response_text, intent, etc. Transport
            layers are responsible for any publish/delivery side effects.
        """
        task_id = message_data.get(
            "task_id", f"task_{datetime.now(timezone.utc).timestamp()}"
        )
        device_id = message_data.get("device_id", "unknown")

        logger.info("processing_task", task_id=task_id, device_id=device_id, transport=transport)

        # The graph's ``retrieve_context`` node is the single write site for
        # ``conversation_history`` — we initialize empty here and let the graph
        # populate from memory.
        initial_state = {
            "device_id": device_id,
            "task_id": task_id,
            "input_type": message_data.get("input_type", "text"),
            "raw_input": message_data.get("transcription", message_data.get("query", "")),
            "context": {},
            "conversation_history": [],
            "confidence": message_data.get("confidence", 1.0),
            "image_data": message_data.get("image_data"),
            "visual_context": None,
        }

        config = None
        if audio_delivery or transport != "mqtt":
            config = {
                "configurable": {
                    "transport": transport,
                    "audio_delivery": audio_delivery,
                }
            }

        result = await self.graph.run(initial_state, config=config)

        if result.get("processed_text") and result.get("response_text"):
            self.memory.commit_exchange(
                device_id,
                result["processed_text"],
                result["response_text"],
                intent=result.get("intent"),
                metadata=result.get("response_metadata", {}),
            )

        return result
