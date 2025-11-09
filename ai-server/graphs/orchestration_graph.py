"""Main orchestration graph for NAILA using LangGraph"""

import logging
from datetime import datetime
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graphs.states import NAILAState
from agents.input_processor import InputProcessor
from agents.response_generator import ResponseGenerator


logger = logging.getLogger(__name__)


class NAILAOrchestrationGraph:
    """Main orchestration workflow for NAILA"""

    def __init__(self, llm_service=None):
        self.input_processor = InputProcessor()
        self.response_generator = ResponseGenerator(llm_service=llm_service)
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph with safe parallel processing"""
        workflow = StateGraph(NAILAState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("execute_actions", self._execute_actions)
        
        # Define sequential flow (safer than parallel state mutation)
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")  
        workflow.add_edge("generate_response", "execute_actions")
        workflow.add_edge("execute_actions", END)
        
        return workflow
    
    
    async def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve additional context safely"""
        device_id = state.get("device_id", "")
        intent = state.get("intent", "")
        confidence = state.get("confidence", 1.0)
        
        # Skip context retrieval for simple, high-confidence queries (performance optimization)
        if intent in ["time_query", "greeting"] and confidence > 0.8:
            logger.debug(f"Skipping context retrieval for simple query: {intent}")
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
            logger.debug(f"Enhanced context for {device_id}")
            
        except Exception as e:
            logger.error(f"Context retrieval error for {device_id}: {e}")
            # Don't fail the whole pipeline for context issues
        
        return state
    
    async def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input node"""
        try:
            return await self.input_processor.process(state)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            state.setdefault("errors", []).append(str(e))
            return state
    
    async def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response node"""
        try:
            return await self.response_generator.process(state)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.setdefault("errors", []).append(str(e))
            return state
    
    async def _execute_actions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions node - AI server only handles response, no device commands"""
        response_text = state.get("response_text", "")
        logger.info(f"Processing AI response for task {state.get('task_id')}: '{response_text}'")
        
        return state
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the orchestration graph"""
        logger.info(f"Starting orchestration for task {initial_state.get('task_id')}")

        # Ensure required fields
        initial_state.setdefault("context", {})
        initial_state.setdefault("conversation_history", [])
        initial_state.setdefault("errors", [])
        initial_state.setdefault("confidence", 1.0)

        # Run the graph
        result = await self.app.ainvoke(initial_state)

        if result.get("errors"):
            logger.warning(f"Completed with errors: {result['errors']}")
        else:
            logger.info("Orchestration completed successfully")

        return result