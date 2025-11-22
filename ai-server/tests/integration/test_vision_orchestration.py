"""Integration tests for vision orchestration"""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from graphs.orchestration import NAILAOrchestrationGraph
from services.vision import VisionService


@pytest.fixture
def sample_image_bytes():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def vision_service(event_loop):
    """Create and load vision service"""
    service = VisionService()
    if not service.model_path.exists():
        pytest.skip(f"Model not found at {service.model_path}")
    success = event_loop.run_until_complete(service.load_model())
    if not success:
        pytest.skip("Failed to load vision model")
    yield service
    service.unload_model()


class TestVisionOrchestration:
    """Test vision integration with orchestration graph"""

    @pytest.mark.asyncio
    async def test_orchestration_with_vision_service(self, vision_service, sample_image_bytes):
        """Test that orchestration graph accepts vision service"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)
        assert graph.vision_service is not None
        assert graph.vision_service.is_ready

    @pytest.mark.asyncio
    async def test_vision_node_in_workflow(self, vision_service):
        """Test that vision node is added when vision service provided"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)
        # Vision node should be in the workflow
        assert "process_vision" in list(graph.workflow.nodes)

    @pytest.mark.asyncio
    async def test_orchestration_without_vision_service(self):
        """Test that orchestration works without vision service

        Vision node is always present in the graph (static topology), but gracefully
        skips processing when vision service is unavailable.
        """
        graph = NAILAOrchestrationGraph(vision_service=None)
        assert graph.vision_service is None
        # Vision node is always present (static topology)
        assert "process_vision" in list(graph.workflow.nodes)

        # But it should gracefully skip processing without errors
        test_state = {
            "device_id": "test_device",
            "task_id": "test_task",
            "raw_input": "Hello",
            "processed_text": "Hello",
            "intent": "greeting",
            "confidence": 1.0,
            "context": {},
            "conversation_history": [],
            "image_data": None
        }
        result = await graph.run(test_state)
        # Should complete without vision processing
        assert "visual_context" not in result or result.get("visual_context") is None

    @pytest.mark.asyncio
    async def test_vision_processing_with_image_data(self, vision_service, sample_image_bytes):
        """Test vision processing node with actual image data"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)

        initial_state = {
            "device_id": "test_device",
            "task_id": "test_task",
            "input_type": "vision",
            "raw_input": "What do you see?",
            "processed_text": "What do you see?",
            "intent": "question",
            "confidence": 1.0,
            "context": {},
            "conversation_history": [],
            "image_data": sample_image_bytes,
            "visual_context": None,
            "response_text": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "errors": []
        }

        result = await graph.run(initial_state)

        # Visual context should be populated
        assert result.get("visual_context") is not None
        assert "description" in result["visual_context"]
        assert "detections" in result["visual_context"]
        assert "object_counts" in result["visual_context"]

    @pytest.mark.asyncio
    async def test_vision_query_with_answer(self, vision_service, sample_image_bytes):
        """Test vision query that generates an answer"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)

        initial_state = {
            "device_id": "test_device",
            "task_id": "test_task",
            "input_type": "vision",
            "raw_input": "How many objects do you see?",
            "processed_text": "How many objects do you see?",
            "intent": "question",
            "confidence": 1.0,
            "context": {},
            "conversation_history": [],
            "image_data": sample_image_bytes,
            "visual_context": None,
            "response_text": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "errors": []
        }

        result = await graph.run(initial_state)

        # Should have both answer and description
        assert result.get("visual_context") is not None
        assert "answer" in result["visual_context"]
        assert "description" in result["visual_context"]

    @pytest.mark.asyncio
    async def test_orchestration_without_image_data(self, vision_service):
        """Test that vision node is skipped when no image data"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)

        initial_state = {
            "device_id": "test_device",
            "task_id": "test_task",
            "input_type": "text",
            "raw_input": "Hello",
            "processed_text": "Hello",
            "intent": "greeting",
            "confidence": 1.0,
            "context": {},
            "conversation_history": [],
            "image_data": None,
            "visual_context": None,
            "response_text": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "errors": []
        }

        result = await graph.run(initial_state)

        # Should complete without visual context
        assert result.get("visual_context") is None
        assert result.get("response_text") is not None

    @pytest.mark.asyncio
    async def test_response_uses_visual_context(self, vision_service, sample_image_bytes):
        """Test that response generator uses visual context"""
        graph = NAILAOrchestrationGraph(vision_service=vision_service)

        initial_state = {
            "device_id": "test_device",
            "task_id": "test_task",
            "input_type": "vision",
            "raw_input": "Describe this image",
            "processed_text": "Describe this image",
            "intent": "question",
            "confidence": 1.0,
            "context": {},
            "conversation_history": [],
            "image_data": sample_image_bytes,
            "visual_context": None,
            "response_text": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "errors": []
        }

        result = await graph.run(initial_state)

        # Response should be generated from visual context
        assert result.get("response_text") is not None
        assert len(result["response_text"]) > 0
        # Response should contain visual information
        visual_ctx = result.get("visual_context", {})
        if visual_ctx.get("description"):
            assert visual_ctx["description"] in result["response_text"] or result["response_text"] in visual_ctx["description"]
