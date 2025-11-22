"""Integration tests for Vision Service - Requires actual model"""

import asyncio
import base64
from pathlib import Path
from io import BytesIO

import pytest
import numpy as np
from PIL import Image

from services.vision import VisionService, DetectionResult, SceneAnalysis
from config import vision as vision_config


@pytest.fixture
def vision_service():
    """Create a vision service instance"""
    return VisionService()


@pytest.fixture
def sample_image():
    """Create a simple test image"""
    # Create a simple RGB image (100x100 red square)
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes"""
    buffer = BytesIO()
    sample_image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def real_test_image():
    """Path to a real test image if available"""
    # You can place a test image in tests/fixtures/
    test_image_path = Path(__file__).parent.parent / "fixtures" / "test_image.jpg"
    return str(test_image_path) if test_image_path.exists() else None


class TestVisionServiceModelLoading:
    """Test actual model loading"""

    @pytest.mark.asyncio
    async def test_model_loading_success(self, vision_service):
        """Test that model loads successfully"""
        # Skip if model doesn't exist
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            # Load model
            success = await vision_service.load_model()

            # Verify
            assert success is True
            assert vision_service.is_ready is True
            assert vision_service.model is not None
            assert vision_service.device_name is not None

            # Check status
            status = vision_service.get_status()
            assert status["ready"] is True
            assert status["model_name"] == f"yolov8{vision_config.MODEL_SIZE}"

        finally:
            # Cleanup
            if vision_service.is_ready:
                vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_model_loading_with_hardware_detection(self, vision_service):
        """Test model loading with hardware detection"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            # Load with hardware detection
            success = await vision_service.load_model()

            assert success is True
            # Device should be detected
            assert vision_service.device_name in ['cpu', 'cuda:0', 'mps']

        finally:
            if vision_service.is_ready:
                vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_model_unload(self, vision_service):
        """Test model unloading"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        # Load model
        await vision_service.load_model()
        assert vision_service.is_ready is True

        # Unload model
        vision_service.unload_model()
        assert vision_service.is_ready is False
        assert vision_service.model is None


class TestObjectDetectionIntegration:
    """Test actual object detection with real model"""

    @pytest.mark.asyncio
    async def test_detect_objects_on_simple_image(self, vision_service, sample_image_bytes):
        """Test object detection on simple test image"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Run detection
            result = await vision_service.detect_objects(sample_image_bytes)

            # Verify result structure
            assert isinstance(result, DetectionResult)
            assert result.inference_time_ms > 0
            assert result.model_name == vision_service.model_name
            assert result.device in ['cpu', 'cuda:0', 'mps']
            assert result.image_size is not None

            # May or may not detect objects in a simple red square
            assert isinstance(result.detections, list)

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_detect_objects_with_custom_confidence(self, vision_service, sample_image_bytes):
        """Test detection with custom confidence threshold"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Run detection with high confidence
            result = await vision_service.detect_objects(
                sample_image_bytes,
                confidence=0.8
            )

            assert isinstance(result, DetectionResult)
            # All detections should meet confidence threshold
            for detection in result.detections:
                assert detection.confidence >= 0.8

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_detect_objects_on_real_image(self, vision_service, real_test_image):
        """Test detection on real image if available"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        if real_test_image is None:
            pytest.skip("No real test image available")

        try:
            await vision_service.load_model()

            # Run detection on real image
            result = await vision_service.detect_objects(real_test_image)

            assert isinstance(result, DetectionResult)
            assert result.inference_time_ms > 0

            # Real images should have some detections
            print(f"Detected {len(result.detections)} objects")
            for det in result.detections:
                print(f"  - {det.class_name}: {det.confidence:.2f}")

        finally:
            vision_service.unload_model()


class TestSceneAnalysisIntegration:
    """Test scene analysis with real model"""

    @pytest.mark.asyncio
    async def test_analyze_scene_basic(self, vision_service, sample_image_bytes):
        """Test basic scene analysis"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Analyze scene
            result = await vision_service.analyze_scene(sample_image_bytes)

            # Verify result structure
            assert isinstance(result, SceneAnalysis)
            assert isinstance(result.description, str)
            assert len(result.description) > 0
            assert isinstance(result.detections, list)
            assert isinstance(result.object_counts, dict)
            assert isinstance(result.main_objects, list)
            assert 0.0 <= result.confidence <= 1.0
            assert result.inference_time_ms > 0

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_analyze_scene_serialization(self, vision_service, sample_image_bytes):
        """Test that scene analysis can be serialized"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            result = await vision_service.analyze_scene(sample_image_bytes)

            # Convert to dict (for MQTT publishing)
            result_dict = result.to_dict()

            assert isinstance(result_dict, dict)
            assert "description" in result_dict
            assert "detections" in result_dict
            assert "object_counts" in result_dict
            assert "main_objects" in result_dict
            assert "confidence" in result_dict
            assert "inference_time_ms" in result_dict

        finally:
            vision_service.unload_model()


class TestVisualQueryIntegration:
    """Test visual query answering with real model"""

    @pytest.mark.asyncio
    async def test_answer_what_question(self, vision_service, sample_image_bytes):
        """Test answering 'what do you see' question"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Analyze scene first
            scene = await vision_service.analyze_scene(sample_image_bytes)

            answer = await vision_service.answer_visual_query(
                "What do you see?",
                scene
            )

            # Verify answer
            assert isinstance(answer, str)
            assert len(answer) > 0
            # Should contain "see" or "don't see"
            assert "see" in answer.lower()

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_answer_count_question(self, vision_service, sample_image_bytes):
        """Test answering count question"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Analyze scene first
            scene = await vision_service.analyze_scene(sample_image_bytes)

            answer = await vision_service.answer_visual_query(
                "How many objects are there?",
                scene
            )

            # Verify answer
            assert isinstance(answer, str)
            assert len(answer) > 0

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_answer_presence_question(self, vision_service, sample_image_bytes):
        """Test answering presence question"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Analyze scene first
            scene = await vision_service.analyze_scene(sample_image_bytes)

            answer = await vision_service.answer_visual_query(
                "Is there a person in the image?",
                scene
            )

            # Verify answer
            assert isinstance(answer, str)
            assert len(answer) > 0
            # Should contain yes or no
            assert "yes" in answer.lower() or "no" in answer.lower()

        finally:
            vision_service.unload_model()


class TestPerformanceMetrics:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_inference_time_reasonable(self, vision_service, sample_image_bytes):
        """Test that inference time is reasonable"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            result = await vision_service.detect_objects(sample_image_bytes)

            # Inference should complete in reasonable time
            # GPU: < 100ms, CPU: < 1000ms
            device_type = vision_service.hardware_info.get('device_type', 'cpu')
            if device_type == 'cuda':
                assert result.inference_time_ms < 100, f"GPU inference too slow: {result.inference_time_ms}ms"
            else:
                assert result.inference_time_ms < 1000, f"CPU inference too slow: {result.inference_time_ms}ms"

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_multiple_detections_performance(self, vision_service, sample_image_bytes):
        """Test running multiple detections"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            # Run multiple detections
            times = []
            for _ in range(5):
                result = await vision_service.detect_objects(sample_image_bytes)
                times.append(result.inference_time_ms)

            # Verify all completed successfully
            assert len(times) == 5
            assert all(t > 0 for t in times)

            # Average time should be consistent
            avg_time = sum(times) / len(times)
            print(f"Average inference time: {avg_time:.2f}ms")

        finally:
            vision_service.unload_model()


class TestImageFormats:
    """Test different image format handling"""

    @pytest.mark.asyncio
    async def test_detect_from_pil_image(self, vision_service, sample_image):
        """Test detection from PIL Image"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            result = await vision_service.detect_objects(sample_image)
            assert isinstance(result, DetectionResult)

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_detect_from_numpy_array(self, vision_service, sample_image):
        """Test detection from numpy array"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            img_array = np.array(sample_image)
            result = await vision_service.detect_objects(img_array)
            assert isinstance(result, DetectionResult)

        finally:
            vision_service.unload_model()

    @pytest.mark.asyncio
    async def test_detect_from_base64(self, vision_service, sample_image_bytes):
        """Test detection from base64 encoded image"""
        if not vision_service.model_path.exists():
            pytest.skip(f"Model not found at {vision_service.model_path}")

        try:
            await vision_service.load_model()

            base64_image = base64.b64encode(sample_image_bytes)
            result = await vision_service.detect_objects(base64_image)
            assert isinstance(result, DetectionResult)

        finally:
            vision_service.unload_model()
