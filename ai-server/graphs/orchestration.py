"""Main orchestration graph for NAILA using LangGraph"""

from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from graphs.states import NAILAState
from agents.input_processor import InputProcessor
from agents.response_generator import ResponseGenerator
from memory.conversation import ConversationMemory
from utils import get_logger


logger = get_logger(__name__)

# Intents that don't benefit from history when we're already confident.
# ``time_query`` doesn't depend on prior turns; ``greeting`` is the same.
# Skipping recall for these saves a SQLite roundtrip per turn.
_RECALL_SKIP_INTENTS = frozenset({"time_query", "greeting"})
_RECALL_SKIP_CONFIDENCE = 0.8
_HISTORY_LIMIT = 10


class NAILAOrchestrationGraph:
    """Main orchestration workflow for NAILA"""

    def __init__(self, memory: ConversationMemory, llm_service=None, tts_service=None, vision_service=None):
        self.memory = memory
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.vision_service = vision_service
        self.input_processor = InputProcessor()
        self.response_generator = ResponseGenerator(llm_service=llm_service, tts_service=tts_service)
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with safe parallel processing

        Graph topology is static - vision node is always included but gracefully
        handles cases where vision service is unavailable or not needed.
        """
        workflow = StateGraph(NAILAState)

        # Add all nodes (static topology)
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("process_vision", self._process_vision)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("execute_actions", self._execute_actions)

        # Define sequential flow (static topology)
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "process_vision")
        workflow.add_edge("process_vision", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "execute_actions")
        workflow.add_edge("execute_actions", END)

        return workflow
    
    
    async def _process_vision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process vision input if image data is provided and vision service is available"""
        # Skip if no vision service or no image data
        if not self.vision_service or not state.get("image_data"):
            return state

        try:
            image_data = state.get("image_data")
            query = state.get("processed_text")

            # Analyze scene once (this performs the expensive YOLOv8 inference)
            scene = await self.vision_service.analyze_scene(image_data, query=query)

            # Clear image data from state to free memory
            state["image_data"] = None

            # Serialize scene to visual context
            state["visual_context"] = scene.to_dict()

            # If there's a query, generate a specific answer using the pre-computed scene
            if query:
                answer = await self.vision_service.answer_visual_query(query, scene)
                state["visual_context"]["answer"] = answer

            logger.info("vision_processed", num_detections=len(scene.detections))
        except Exception as e:
            logger.error("vision_processing_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(f"Vision processing failed: {str(e)}")
            state["image_data"] = None
        return state

    async def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Load recent exchanges from memory into ``state["conversation_history"]``.

        Memory returns newest-first (``ts DESC``); LLM consumers want
        chronological order, so we reverse before storing. Memory is canonical
        — every path here writes ``conversation_history`` directly so a stale
        caller-provided value can never leak into the turn.
        """
        device_id = state.get("device_id", "")
        intent = state.get("intent", "")
        confidence = state.get("confidence", 1.0)

        # Skip recall for high-confidence simple intents — they don't benefit
        # from history and the SQLite roundtrip is wasted work.
        if intent in _RECALL_SKIP_INTENTS and confidence > _RECALL_SKIP_CONFIDENCE:
            logger.debug("skipping_recall", intent=intent, reason="simple_query")
            state["conversation_history"] = []
            return state

        if not device_id:
            state["conversation_history"] = []
            return state

        try:
            recent = self.memory.recall_recent(device_id, n=_HISTORY_LIMIT)
            state["conversation_history"] = list(reversed(recent))
            logger.debug("history_recalled", device_id=device_id, count=len(recent))
        except Exception as e:
            logger.error(
                "context_retrieval_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            state["conversation_history"] = []

        return state
    
    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input node"""
        try:
            return await self.input_processor.process(state)
        except Exception as e:
            logger.error("input_processing_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(str(e))
            return state

    async def _generate_response(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """Generate response node — passes LangGraph config through for transport callbacks"""
        try:
            return await self.response_generator.process(state, config=config)
        except Exception as e:
            logger.error("response_generation_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(str(e))
            return state

    async def _execute_actions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions node - AI server only handles response, no device commands"""
        response_text = state.get("response_text", "")
        logger.info("processing_ai_response", task_id=state.get('task_id'), response_text=response_text)

        return state
    
    async def run(self, initial_state: Dict[str, Any], config: Optional[dict] = None) -> Dict[str, Any]:
        """Run the orchestration graph with optional transport config"""
        logger.info("starting_orchestration", task_id=initial_state.get('task_id'))

        # Ensure required fields
        initial_state.setdefault("context", {})
        initial_state.setdefault("conversation_history", [])
        initial_state.setdefault("errors", [])
        initial_state.setdefault("confidence", 1.0)

        # Run the graph — config carries transport callbacks (e.g. audio_delivery for gRPC)
        result = await self.app.ainvoke(initial_state, config=config)

        if result.get("errors"):
            logger.warning("orchestration_completed_with_errors", errors=result['errors'])
        else:
            logger.info("orchestration_completed", status="success")

        return result