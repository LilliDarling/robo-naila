"""Unit tests for NAILAOrchestrationGraph"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from graphs.orchestration import NAILAOrchestrationGraph


class TestNAILAOrchestrationGraph:
    """Test cases for NAILA orchestration workflow"""
    
    @pytest.fixture
    def mock_processors(self):
        """Mock input processor and response generator"""
        input_processor = Mock()
        response_generator = Mock()
        
        # Setup async mocks
        input_processor.process = AsyncMock()
        response_generator.process = AsyncMock()
        
        return input_processor, response_generator
    
    @pytest.fixture
    def orchestrator(self, mock_processors):
        """Create orchestrator with mocked components"""
        input_proc, response_gen = mock_processors
        
        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            return NAILAOrchestrationGraph()

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.input_processor is not None
        assert orchestrator.response_generator is not None
        assert orchestrator.workflow is not None
        assert orchestrator.app is not None

    def test_graph_structure(self, orchestrator):
        """Test that graph nodes and edges are properly configured"""
        workflow = orchestrator.workflow
        
        # Check nodes exist
        assert "process_input" in workflow.nodes
        assert "retrieve_context" in workflow.nodes 
        assert "generate_response" in workflow.nodes
        assert "execute_actions" in workflow.nodes
        
        # Check that workflow has proper structure
        assert len(workflow.nodes) >= 4

    @pytest.mark.asyncio
    async def test_process_input_node(self, orchestrator):
        """Test input processing node"""
        test_state = {"raw_input": "hello", "input_type": "text"}
        
        # Mock the processor response
        orchestrator.input_processor.process.return_value = {
            **test_state,
            "processed_text": "hello",
            "intent": "greeting",
            "confidence": 0.95
        }
        
        result = await orchestrator._process_input(test_state)
        
        assert result["processed_text"] == "hello"
        assert result["intent"] == "greeting"
        orchestrator.input_processor.process.assert_called_once_with(test_state)

    @pytest.mark.asyncio
    async def test_process_input_error_handling(self, orchestrator):
        """Test input processing error handling"""
        test_state = {"raw_input": "test"}
        
        # Mock processor to raise error
        orchestrator.input_processor.process.side_effect = Exception("Processing failed")
        
        result = await orchestrator._process_input(test_state)
        
        assert "errors" in result
        assert "Processing failed" in result["errors"]

    @pytest.mark.asyncio
    async def test_retrieve_context_simple_query(self, orchestrator):
        """Test context retrieval skip for simple queries"""
        state = {
            "intent": "time_query",
            "confidence": 0.9,
            "device_id": "robot_001",
            "context": {}
        }
        
        result = await orchestrator._retrieve_context(state)
        
        # Should skip enhancement for simple, high-confidence queries
        assert result == state

    @pytest.mark.asyncio
    async def test_retrieve_context_complex_query(self, orchestrator):
        """Test context retrieval for complex queries"""
        state = {
            "intent": "question",
            "confidence": 0.7,
            "device_id": "robot_001",
            "context": {"existing": "data"}
        }
        
        with patch('graphs.orchestration.datetime') as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2025-01-15T10:30:00"
            
            result = await orchestrator._retrieve_context(state)
        
        assert result["context"]["context_retrieved"] == True
        assert result["context"]["retrieval_timestamp"] == "2025-01-15T10:30:00"
        assert result["context"]["existing"] == "data"

    @pytest.mark.asyncio
    async def test_retrieve_context_error_handling(self, orchestrator):
        """Test context retrieval graceful error handling"""
        state = {
            "intent": "question",
            "confidence": 0.5,
            "device_id": "robot_001"
        }
        
        # Mock datetime to raise error
        with patch('graphs.orchestration.datetime', side_effect=Exception("Time error")):
            result = await orchestrator._retrieve_context(state)
        
        # Should not crash the pipeline
        assert result == state

    @pytest.mark.asyncio
    async def test_generate_response_node(self, orchestrator):
        """Test response generation node"""
        test_state = {
            "intent": "greeting",
            "processed_text": "hello",
            "context": {}
        }
        
        # Mock response generator
        orchestrator.response_generator.process.return_value = {
            **test_state,
            "response_text": "Hello! How can I help you?",
            "response_metadata": {"intent": "greeting"}
        }
        
        result = await orchestrator._generate_response(test_state)
        
        assert result["response_text"] == "Hello! How can I help you?"
        orchestrator.response_generator.process.assert_called_once_with(test_state)

    @pytest.mark.asyncio
    async def test_generate_response_error_handling(self, orchestrator):
        """Test response generation error handling"""
        test_state = {"intent": "greeting"}
        
        # Mock generator to raise error
        orchestrator.response_generator.process.side_effect = Exception("Generation failed")
        
        result = await orchestrator._generate_response(test_state)
        
        assert "errors" in result
        assert "Generation failed" in result["errors"]

    @pytest.mark.asyncio
    async def test_execute_actions_node(self, orchestrator):
        """Test action execution node"""
        test_state = {
            "task_id": "test_task_001",
            "response_text": "Hello there!"
        }
        
        result = await orchestrator._execute_actions(test_state)
        
        # Should return state unchanged (AI server doesn't execute device commands)
        assert result == test_state

    @pytest.mark.asyncio
    async def test_full_orchestration_flow(self, orchestrator):
        """Test complete orchestration workflow"""
        initial_state = {
            "task_id": "flow_test_001",
            "device_id": "robot_001",
            "raw_input": "What time is it?",
            "input_type": "text"
        }
        
        # Mock input processor
        orchestrator.input_processor.process.return_value = {
            **initial_state,
            "processed_text": "What time is it?",
            "intent": "time_query",
            "confidence": 0.95
        }
        
        # Mock response generator
        orchestrator.response_generator.process.return_value = {
            **initial_state,
            "processed_text": "What time is it?",
            "intent": "time_query", 
            "confidence": 0.95,
            "response_text": "The current time is 2:30 PM"
        }
        
        result = await orchestrator.run(initial_state)
        
        assert result["response_text"] == "The current time is 2:30 PM"
        assert result["intent"] == "time_query"
        assert result["task_id"] == "flow_test_001"
        
        # Verify processors were called
        orchestrator.input_processor.process.assert_called_once()
        orchestrator.response_generator.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_initializes_required_fields(self, orchestrator):
        """Test that run() ensures all required state fields exist"""
        minimal_state = {
            "task_id": "minimal_001",
            "raw_input": "test"
        }
        
        # Mock processors to return minimal responses
        orchestrator.input_processor.process.return_value = minimal_state
        orchestrator.response_generator.process.return_value = minimal_state
        
        result = await orchestrator.run(minimal_state)
        
        # Required fields should be initialized
        assert "context" in result
        assert "conversation_history" in result
        assert "errors" in result
        assert "confidence" in result
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_error_propagation_through_workflow(self, orchestrator):
        """Test that errors are collected from all nodes"""
        initial_state = {"task_id": "error_test", "raw_input": "test"}
        
        # Mock both processors to add errors
        orchestrator.input_processor.process.side_effect = Exception("Input error")
        orchestrator.response_generator.process.side_effect = Exception("Response error")
        
        result = await orchestrator.run(initial_state)
        
        # Both errors should be collected
        assert len(result["errors"]) >= 1
        assert any("Input error" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_state_immutability_in_context_retrieval(self, orchestrator):
        """Test that context retrieval doesn't mutate original state unsafely"""
        original_context = {"original": "data"}
        state = {
            "intent": "question",
            "confidence": 0.5,
            "context": original_context.copy()
        }
        
        with patch('graphs.orchestration.datetime') as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2025-01-15T10:30:00"
            
            result = await orchestrator._retrieve_context(state)
        
        # Original context should be preserved
        assert result["context"]["original"] == "data"
        # New fields should be added
        assert result["context"]["context_retrieved"] == True

    @pytest.mark.asyncio
    async def test_confidence_based_context_skip(self, orchestrator):
        """Test context skip logic for different confidence levels"""
        # High confidence greeting - should skip
        high_conf_state = {
            "intent": "greeting",
            "confidence": 0.9,
            "context": {"test": True}
        }
        
        result = await orchestrator._retrieve_context(high_conf_state)
        assert "context_retrieved" not in result.get("context", {})
        
        # Low confidence greeting - should enhance
        low_conf_state = {
            "intent": "greeting", 
            "confidence": 0.5,
            "context": {"test": True}
        }
        
        with patch('graphs.orchestration.datetime') as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2025-01-15T10:30:00"
            result = await orchestrator._retrieve_context(low_conf_state)
        
        assert result["context"]["context_retrieved"] == True

    @pytest.mark.asyncio
    async def test_workflow_with_empty_state(self, orchestrator):
        """Test workflow handles empty/minimal state gracefully"""
        empty_state = {}
        
        # Mock processors to handle empty state
        orchestrator.input_processor.process.return_value = {"processed": True}
        orchestrator.response_generator.process.return_value = {
            "processed": True,
            "response_text": "Empty state handled"
        }
        
        result = await orchestrator.run(empty_state)
        
        # Should complete without errors
        assert result["response_text"] == "Empty state handled"
        assert "errors" in result
        assert len(result.get("errors", [])) == 0