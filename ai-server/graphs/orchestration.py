"""Main orchestration graph for NAILA using LangGraph"""

from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graphs.states import NAILAState
from agents.input_processor import InputProcessor
from agents.response_generator import ResponseGenerator
from utils import get_logger


logger = get_logger(__name__)


class NAILAOrchestrationGraph:
    """Main orchestration workflow for NAILA"""

    def __init__(self, llm_service=None, tts_service=None, vision_service=None):
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.vision_service = vision_service
        self.input_processor = InputProcessor()
        self.response_generator = ResponseGenerator(llm_service=llm_service, tts_service=tts_service)
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with safe parallel processing"""
        workflow = StateGraph(NAILAState)

        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("retrieve_context", self._retrieve_context)
        if self.vision_service:
            workflow.add_node("process_vision", self._process_vision)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("execute_actions", self._execute_actions)

        # Define sequential flow (safer than parallel state mutation)
        workflow.set_entry_point("process_input")
        if self.vision_service:
            workflow.add_edge("process_input", "process_vision")
            workflow.add_edge("process_vision", "retrieve_context")
        else:
            workflow.add_edge("process_input", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "execute_actions")
        workflow.add_edge("execute_actions", END)

        return workflow
    
    
    async def _process_vision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process vision input if image data is provided"""
        if not state.get("image_data"):
            return state
        try:
            image_data = state["image_data"]
            query = state.get("processed_text")
            if query:
                answer = await self.vision_service.answer_visual_query(image_data, query)
                scene = await self.vision_service.analyze_scene(image_data)
                state["visual_context"] = {
                    "answer": answer,
                    "description": scene.description,
                    "detections": [d.to_dict() for d in scene.detections],
                    "object_counts": scene.object_counts,
                    "main_objects": scene.main_objects,
                    "confidence": scene.confidence
                }
            else:
                scene = await self.vision_service.analyze_scene(image_data)
                state["visual_context"] = {
                    "description": scene.description,
                    "detections": [d.to_dict() for d in scene.detections],
                    "object_counts": scene.object_counts,
                    "main_objects": scene.main_objects,
                    "confidence": scene.confidence
                }
            logger.info("vision_processed", num_detections=len(scene.detections))
        except Exception as e:
            logger.error("vision_processing_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(f"Vision processing failed: {str(e)}")
        return state

    async def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve additional context safely"""
        device_id = state.get("device_id", "")
        intent = state.get("intent", "")
        confidence = state.get("confidence", 1.0)

        # Skip context retrieval for simple, high-confidence queries (performance optimization)
        if intent in ["time_query", "greeting"] and confidence > 0.8:
            logger.debug("skipping_context_retrieval", intent=intent, reason="simple_query")
            return state

        try:
            # Safe context enhancement - create new dict to avoid mutation race conditions
            enhanced_context = dict(state.get("context", {}))
            enhanced_context["context_retrieved"] = True
            enhanced_context["retrieval_timestamp"] = datetime.now().isoformat()

            # Could add more context sources here:
            # - Recent device interactions
            # - User preferences
            # - Environmental context

            state["context"] = enhanced_context
            logger.debug("context_enhanced", device_id=device_id)

        except Exception as e:
            logger.error("context_retrieval_error", device_id=device_id, error=str(e), error_type=type(e).__name__)
            # Don't fail the whole pipeline for context issues

        return state
    
    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input node"""
        try:
            return await self.input_processor.process(state)
        except Exception as e:
            logger.error("input_processing_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(str(e))
            return state

    async def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response node"""
        try:
            return await self.response_generator.process(state)
        except Exception as e:
            logger.error("response_generation_error", error=str(e), error_type=type(e).__name__)
            state.setdefault("errors", []).append(str(e))
            return state

    async def _execute_actions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions node - AI server only handles response, no device commands"""
        response_text = state.get("response_text", "")
        logger.info("processing_ai_response", task_id=state.get('task_id'), response_text=response_text)

        return state
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestration graph"""
        logger.info("starting_orchestration", task_id=initial_state.get('task_id'))

        # Ensure required fields
        initial_state.setdefault("context", {})
        initial_state.setdefault("conversation_history", [])
        initial_state.setdefault("errors", [])
        initial_state.setdefault("confidence", 1.0)

        # Run the graph
        result = await self.app.ainvoke(initial_state)

        if result.get("errors"):
            logger.warning("orchestration_completed_with_errors", errors=result['errors'])
        else:
            logger.info("orchestration_completed", status="success")

        return result