"""Unit tests for NAILAOrchestrationGraph"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from graphs.orchestration import NAILAOrchestrationGraph


def _mock_memory(history=None):
    """Memory double whose ``recall_recent`` returns the given (newest-first) list."""
    mem = Mock()
    mem.recall_recent.return_value = history or []
    return mem


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
    def memory(self):
        return _mock_memory()

    @pytest.fixture
    def orchestrator(self, mock_processors, memory):
        """Create orchestrator with mocked components"""
        input_proc, response_gen = mock_processors

        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            return NAILAOrchestrationGraph(memory=memory)

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
    async def test_retrieve_context_skip_for_high_confidence_time_query(
        self, orchestrator, memory
    ):
        """High-confidence ``time_query`` skips recall — it doesn't need history."""
        state = {
            "intent": "time_query",
            "confidence": 0.9,
            "device_id": "robot_001",
            "context": {},
        }

        result = await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_not_called()
        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_skip_for_high_confidence_greeting(
        self, orchestrator, memory
    ):
        """High-confidence ``greeting`` skips recall too."""
        state = {
            "intent": "greeting",
            "confidence": 0.95,
            "device_id": "robot_001",
            "context": {},
        }

        result = await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_not_called()
        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_calls_recall_for_complex_intents(
        self, orchestrator, memory
    ):
        """Non-trivial intents pull history from memory."""
        state = {
            "intent": "question",
            "confidence": 0.7,
            "device_id": "robot_001",
            "context": {},
        }

        await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_called_once_with("robot_001", n=10)

    @pytest.mark.asyncio
    async def test_retrieve_context_runs_for_low_confidence_simple_intents(
        self, orchestrator, memory
    ):
        """The skip is gated on confidence — a low-confidence ``time_query`` still recalls."""
        state = {
            "intent": "time_query",
            "confidence": 0.5,
            "device_id": "robot_001",
            "context": {},
        }

        await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_called_once_with("robot_001", n=10)

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_chronological_history(
        self, mock_processors
    ):
        """``recall_recent`` returns newest-first; state must hold chronological order
        because ``build_chat_messages`` slices the tail and assumes that orientation."""
        input_proc, response_gen = mock_processors
        memory = _mock_memory(history=[
            {"user": "third", "assistant": "3", "intent": None, "ts": 3, "metadata": {}},
            {"user": "second", "assistant": "2", "intent": None, "ts": 2, "metadata": {}},
            {"user": "first", "assistant": "1", "intent": None, "ts": 1, "metadata": {}},
        ])
        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            graph = NAILAOrchestrationGraph(memory=memory)

        state = {"intent": "question", "confidence": 0.7, "device_id": "d"}
        result = await graph._retrieve_context(state)

        assert [r["user"] for r in result["conversation_history"]] == [
            "first", "second", "third",
        ]

    @pytest.mark.asyncio
    async def test_retrieve_context_memory_error_does_not_crash(
        self, mock_processors
    ):
        """A flaky memory layer must not break the turn — fall back to empty history."""
        input_proc, response_gen = mock_processors
        memory = Mock()
        memory.recall_recent.side_effect = RuntimeError("db on fire")
        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            graph = NAILAOrchestrationGraph(memory=memory)

        state = {"intent": "question", "confidence": 0.7, "device_id": "d"}
        result = await graph._retrieve_context(state)

        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_missing_device_id_returns_empty_history(
        self, orchestrator, memory
    ):
        """No device_id → no recall, default to empty history."""
        state = {"intent": "question", "confidence": 0.7}

        result = await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_not_called()
        assert result["conversation_history"] == []

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
        orchestrator.response_generator.process.assert_called_once_with(test_state, config=None)

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
    async def test_retrieve_context_does_not_touch_context_dict(
        self, orchestrator, memory
    ):
        """``_retrieve_context`` writes to ``conversation_history`` only; the
        ``context`` dict is for unrelated per-turn metadata (visual context, etc.)
        and must come out identical to what went in."""
        original_context = {"original": "data", "visual_flag": True}
        state = {
            "intent": "question",
            "confidence": 0.5,
            "device_id": "robot_001",
            "context": dict(original_context),
        }

        result = await orchestrator._retrieve_context(state)

        assert result["context"] == original_context

    @pytest.mark.asyncio
    async def test_retrieve_context_skip_overwrites_stale_history(
        self, orchestrator, memory
    ):
        """Memory is canonical even on the skip path. If a stale
        ``conversation_history`` floats in from somewhere, the skip branch must
        still clear it so downstream consumers don't see ghost turns."""
        state = {
            "intent": "time_query",
            "confidence": 0.95,
            "device_id": "robot_001",
            "conversation_history": [
                {"user": "stale", "assistant": "ghost", "intent": None, "ts": 0, "metadata": {}}
            ],
        }

        result = await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_not_called()
        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_recall_failure_overwrites_stale_history(
        self, mock_processors
    ):
        """Same canonical-memory rule on the error path: don't fall back to
        a caller-provided history just because recall blew up."""
        input_proc, response_gen = mock_processors
        memory = Mock()
        memory.recall_recent.side_effect = RuntimeError("db on fire")
        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            graph = NAILAOrchestrationGraph(memory=memory)

        state = {
            "intent": "question",
            "confidence": 0.7,
            "device_id": "d",
            "conversation_history": [
                {"user": "stale", "assistant": "ghost", "intent": None, "ts": 0, "metadata": {}}
            ],
        }

        result = await graph._retrieve_context(state)

        assert result["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_retrieve_context_missing_device_id_overwrites_stale_history(
        self, orchestrator, memory
    ):
        """Same canonical rule when device_id is missing — clear, don't preserve."""
        state = {
            "intent": "question",
            "confidence": 0.7,
            "conversation_history": [
                {"user": "stale", "assistant": "ghost", "intent": None, "ts": 0, "metadata": {}}
            ],
        }

        result = await orchestrator._retrieve_context(state)

        memory.recall_recent.assert_not_called()
        assert result["conversation_history"] == []

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


class TestConfigPassthrough:
    """Test that LangGraph config (transport callbacks) passes through to nodes."""

    @pytest.fixture
    def mock_processors(self):
        input_processor = Mock()
        response_generator = Mock()
        input_processor.process = AsyncMock()
        response_generator.process = AsyncMock()
        return input_processor, response_generator

    @pytest.fixture
    def orchestrator(self, mock_processors):
        input_proc, response_gen = mock_processors
        with patch('graphs.orchestration.InputProcessor', return_value=input_proc), \
             patch('graphs.orchestration.ResponseGenerator', return_value=response_gen):
            return NAILAOrchestrationGraph(memory=_mock_memory())

    @pytest.mark.asyncio
    async def test_config_forwarded_to_generate_response(self, orchestrator):
        """Config with audio_delivery should reach the generate_response node."""
        callback_received = {}

        async def capture_config(state, config=None):
            callback_received["config"] = config
            return {**state, "response_text": "ok"}

        orchestrator.response_generator.process = AsyncMock(side_effect=capture_config)
        orchestrator.input_processor.process.return_value = {
            "processed_text": "test", "intent": "general"
        }

        config = {"configurable": {"transport": "grpc", "audio_delivery": lambda: None}}
        await orchestrator.run({"task_id": "cfg_test", "raw_input": "test"}, config=config)

        assert "config" in callback_received
        assert callback_received["config"]["configurable"]["transport"] == "grpc"

    @pytest.mark.asyncio
    async def test_no_config_still_works(self, orchestrator):
        """Graph should work fine without config (MQTT path)."""
        orchestrator.input_processor.process.return_value = {
            "processed_text": "hi", "intent": "greeting"
        }
        orchestrator.response_generator.process.return_value = {
            "response_text": "hello"
        }

        result = await orchestrator.run({"task_id": "no_cfg", "raw_input": "hi"})
        assert result["response_text"] == "hello"

    @pytest.mark.asyncio
    async def test_generate_response_passes_config(self, orchestrator):
        """_generate_response node should forward config to ResponseGenerator.process."""
        orchestrator.response_generator.process.return_value = {"response_text": "ok"}

        config = {"configurable": {"transport": "grpc"}}
        await orchestrator._generate_response({"intent": "general"}, config=config)

        orchestrator.response_generator.process.assert_called_once_with(
            {"intent": "general"}, config=config
        )