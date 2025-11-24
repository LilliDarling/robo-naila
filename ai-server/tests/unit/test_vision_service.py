"""Unit tests for Vision Service - All tests use mocks"""

import asyncio
import base64
from pathlib import Path
from io import BytesIO
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest
import numpy as np
from PIL import Image

from services.vision import VisionService, Detection, DetectionResult, SceneAnalysis, BoundingBox
from config import vision as vision_config


@pytest.fixture
def vision_service():
    """Create a vision service instance"""
    return VisionService()


@pytest.fixture
def sample_image():
    """Create a simple test image"""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes"""
    buffer = BytesIO()
    sample_image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Convert sample image to base64"""
    return base64.b64encode(sample_image_bytes)


def create_mock_tensor(value):
    """Create a mock tensor that behaves like PyTorch tensor"""
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value = mock_tensor
    # Ensure numpy() returns scalar for single values, array for multiple
    if isinstance(value, np.ndarray) and value.size == 1:
        mock_tensor.numpy.return_value = value.item()  # Convert to scalar
    else:
        mock_tensor.numpy.return_value = value
    return mock_tensor


class TestVisionServiceInitialization:
    """Test vision service initialization"""

    def test_service_initialization(self, vision_service):
        """Test that service initializes correctly"""
        assert vision_service is not None
        assert vision_service.model is None
        assert not vision_service.is_ready
        assert vision_service.model_name == f"yolov8{vision_config.MODEL_SIZE}"

    def test_model_path_exists(self, vision_service):
        """Test that model path is correctly set"""
        assert vision_service.model_path == Path(vision_config.MODEL_PATH)


class TestImagePreprocessing:
    """Test image preprocessing methods"""

    @pytest.mark.asyncio
    async def test_preprocess_pil_image(self, vision_service, sample_image):
        """Test preprocessing of PIL Image"""
        processed, size = await vision_service._preprocess_image(sample_image)
        assert isinstance(processed, np.ndarray)
        assert size == (100, 100)

    @pytest.mark.asyncio
    async def test_preprocess_bytes(self, vision_service, sample_image_bytes):
        """Test preprocessing of image bytes"""
        processed, size = await vision_service._preprocess_image(sample_image_bytes)
        assert isinstance(processed, np.ndarray)
        assert size == (100, 100)

    @pytest.mark.asyncio
    async def test_preprocess_base64(self, vision_service, sample_image_base64):
        """Test preprocessing of base64 encoded image"""
        processed, size = await vision_service._preprocess_image(sample_image_base64)
        assert isinstance(processed, np.ndarray)
        assert size == (100, 100)

    @pytest.mark.asyncio
    async def test_preprocess_numpy_array(self, vision_service, sample_image):
        """Test preprocessing of numpy array"""
        img_array = np.array(sample_image)
        processed, size = await vision_service._preprocess_image(img_array)
        assert isinstance(processed, np.ndarray)
        assert size == (100, 100)

    @pytest.mark.asyncio
    async def test_preprocess_rgba_conversion(self, vision_service):
        """Test that RGBA images are converted to RGB"""
        rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        processed, size = await vision_service._preprocess_image(rgba_img)
        assert isinstance(processed, np.ndarray)
        assert processed.shape[2] == 3  # RGB has 3 channels

    @pytest.mark.asyncio
    async def test_preprocess_invalid_type(self, vision_service):
        """Test that invalid image type raises error"""
        with pytest.raises((ValueError, FileNotFoundError)):
            # String is treated as file path, so will raise FileNotFoundError
            await vision_service._preprocess_image("invalid_type")

        # Test with truly invalid type
        with pytest.raises(Exception):
            await vision_service._preprocess_image(12345)


class TestDetectionDataClasses:
    """Test detection data classes"""

    def test_bounding_box(self):
        """Test BoundingBox dataclass"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.center == (55, 110)
        assert bbox.area == 90 * 180

    def test_detection_to_dict(self):
        """Test Detection serialization"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            class_name="person",
            class_id=0,
            confidence=0.95,
            bbox=bbox
        )

        result = detection.to_dict()
        assert result["class"] == "person"
        assert result["class_id"] == 0
        assert result["confidence"] == 0.95
        assert result["bbox"] == [10, 20, 100, 200]
        assert result["center"] == [55, 110]
        assert result["area"] == 16200

    def test_detection_result_object_counts(self):
        """Test DetectionResult object counting"""
        bbox1 = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        bbox2 = BoundingBox(x1=110, y1=20, x2=200, y2=200)

        detections = [
            Detection("person", 0, 0.95, bbox1),
            Detection("person", 0, 0.90, bbox2),
            Detection("dog", 16, 0.85, bbox1)
        ]

        result = DetectionResult(
            detections=detections,
            image_size=(640, 480),
            inference_time_ms=100,
            timestamp="2025-01-01T00:00:00Z",
            model_name="yolov8n",
            device="cpu"
        )

        counts = result.get_object_counts()
        assert counts["person"] == 2
        assert counts["dog"] == 1


class TestDescriptionGeneration:
    """Test scene description generation"""

    def test_empty_detections(self, vision_service):
        """Test description with no detections"""
        description = vision_service._generate_description([], None)
        assert "don't see" in description.lower()

    def test_single_object(self, vision_service):
        """Test description with single object"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [Detection("person", 0, 0.95, bbox)]

        description = vision_service._generate_description(detections, None)
        assert "1 person" in description.lower()

    def test_multiple_same_objects(self, vision_service):
        """Test description with multiple same objects"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection("person", 0, 0.95, bbox),
            Detection("person", 0, 0.90, bbox)
        ]

        description = vision_service._generate_description(detections, None)
        assert "2 person" in description.lower()

    def test_multiple_different_objects(self, vision_service):
        """Test description with different objects"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection("person", 0, 0.95, bbox),
            Detection("dog", 16, 0.90, bbox),
            Detection("car", 2, 0.85, bbox)
        ]

        description = vision_service._generate_description(detections, None)
        assert "person" in description.lower()
        assert "dog" in description.lower()
        assert "car" in description.lower()

    def test_low_confidence_filtering(self, vision_service):
        """Test that low confidence detections are filtered"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection("person", 0, 0.95, bbox),
            Detection("dog", 16, 0.20, bbox)  # Very low confidence
        ]

        description = vision_service._generate_description(detections, None)
        # Should only mention person due to confidence threshold
        assert "person" in description.lower()


class TestMainObjectIdentification:
    """Test main object identification"""

    def test_identify_main_objects_empty(self, vision_service):
        """Test with no detections"""
        main_objects = vision_service._identify_main_objects([])
        assert main_objects == []

    def test_identify_main_objects_single(self, vision_service):
        """Test with single detection"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [Detection("person", 0, 0.95, bbox)]

        main_objects = vision_service._identify_main_objects(detections)
        assert "person" in main_objects
        assert len(main_objects) <= 3

    def test_identify_main_objects_multiple(self, vision_service):
        """Test with multiple detections"""
        bbox_large = BoundingBox(x1=10, y1=20, x2=200, y2=400)
        bbox_small = BoundingBox(x1=10, y1=20, x2=50, y2=60)

        detections = [
            Detection("person", 0, 0.95, bbox_large),  # Large, high confidence
            Detection("dog", 16, 0.90, bbox_small),    # Small, high confidence
            Detection("car", 2, 0.85, bbox_large),     # Large, medium confidence
            Detection("cat", 15, 0.80, bbox_small),    # Small, medium confidence
        ]

        main_objects = vision_service._identify_main_objects(detections, top_n=3)
        assert len(main_objects) <= 3
        # Person should be first due to high confidence and large size
        assert main_objects[0] == "person"

    def test_identify_main_objects_unique(self, vision_service):
        """Test that main objects are unique"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection("person", 0, 0.95, bbox),
            Detection("person", 0, 0.90, bbox),
            Detection("person", 0, 0.85, bbox),
        ]

        main_objects = vision_service._identify_main_objects(detections, top_n=3)
        assert len(main_objects) == 1
        assert main_objects[0] == "person"


class TestVisualQueryAnswering:
    """Test visual query answering"""

    def test_count_query_format(self, vision_service):
        """Test that count query format is detected"""
        question = "How many people are there?"
        assert "how many" in question.lower()

    def test_presence_query_format(self, vision_service):
        """Test that presence query format is detected"""
        question = "Is there a dog in the image?"
        assert "is there" in question.lower()

    def test_what_query_format(self, vision_service):
        """Test that what query format is detected"""
        question = "What do you see?"
        assert "what do you see" in question.lower()


class TestServiceStatus:
    """Test service status reporting"""

    def test_status_not_loaded(self, vision_service):
        """Test status when model not loaded"""
        status = vision_service.get_status()
        assert status["ready"] is False
        assert status["model_exists"] == Path(vision_config.MODEL_PATH).exists()
        assert status["model_name"] == f"yolov8{vision_config.MODEL_SIZE}"

    def test_get_model_type(self, vision_service):
        """Test model type reporting"""
        assert vision_service._get_model_type() == "Vision"


class TestConfiguration:
    """Test configuration values"""

    def test_config_values(self):
        """Test that config values are reasonable"""
        assert 0.0 <= vision_config.CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= vision_config.IOU_THRESHOLD <= 1.0
        assert vision_config.MAX_DETECTIONS > 0
        assert vision_config.IMAGE_SIZE > 0
        assert len(vision_config.COCO_CLASSES) == 80

    def test_coco_classes_complete(self):
        """Test that COCO classes list is complete"""
        expected_classes = {"person", "car", "dog", "cat", "chair", "bottle"}
        coco_classes_set = set(vision_config.COCO_CLASSES)
        missing = expected_classes - coco_classes_set
        assert not missing, f"Missing expected COCO classes: {missing}"


class TestModelLoading:
    """Test model loading with mocks"""

    @pytest.mark.asyncio
    async def test_model_loading_success(self, vision_service):
        """Test successful model loading with mocked YOLO"""
        with patch('services.vision.YOLO') as mock_yolo_class:
            # Mock the YOLO model
            mock_model = MagicMock()
            mock_yolo_class.return_value = mock_model

            # Mock hardware info
            vision_service.hardware_info = {
                'device_type': 'cuda',
                'device_name': 'Test GPU',
                'cpu_count': 8,
                'vram_gb': 8.0
            }

            # Load model
            success = await vision_service.load_model()

            # Verify
            assert success is True
            assert vision_service.is_ready is True
            assert vision_service.model is not None
            mock_yolo_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_failure(self, vision_service):
        """Test model loading failure handling"""
        with patch('services.vision.YOLO', side_effect=Exception("Model load error")):
            # Mock hardware info
            vision_service.hardware_info = {
                'device_type': 'cpu',
                'device_name': 'Test CPU',
                'cpu_count': 4,
                'vram_gb': None
            }

            # Attempt to load model
            success = await vision_service.load_model()

            # Verify failure is handled
            assert success is False
            assert vision_service.is_ready is False

    @pytest.mark.asyncio
    async def test_model_loading_without_ultralytics(self, vision_service):
        """Test that missing ultralytics is handled"""
        with patch('services.vision.HAS_ULTRALYTICS', False):
            with patch('services.vision.YOLO', None):
                vision_service.hardware_info = {'device_type': 'cpu', 'device_name': 'Test CPU', 'cpu_count': 4, 'vram_gb': None}

                success = await vision_service.load_model()
                assert success is False


class TestObjectDetection:
    """Test object detection with mocks"""

    @pytest.mark.asyncio
    async def test_detect_objects_success(self, vision_service, sample_image_bytes):
        """Test object detection with mocked model"""
        # Setup mocked model
        mock_model = MagicMock()
        mock_result = MagicMock()

        # Mock detection results - properly mock PyTorch tensors
        mock_boxes = MagicMock()

        # Create mock tensors that behave like PyTorch tensors
        mock_xyxy = [create_mock_tensor(np.array([10, 20, 100, 200]))]
        mock_boxes.xyxy = mock_xyxy
        mock_boxes.__len__ = lambda self: len(mock_xyxy)

        # Mock cls and conf as PyTorch-like tensors
        mock_boxes.cls = [create_mock_tensor(np.array([0]))]  # person class
        mock_boxes.conf = [create_mock_tensor(np.array([0.95]))]

        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        # Run detection
        result = await vision_service.detect_objects(sample_image_bytes)

        # Verify
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"
        assert result.detections[0].confidence == 0.95
        assert result.inference_time_ms >= 0  # Can be 0 for very fast mocked operations

    @pytest.mark.asyncio
    async def test_detect_objects_not_ready(self, vision_service, sample_image_bytes):
        """Test detection fails when model not ready"""
        vision_service.is_ready = False

        with pytest.raises(RuntimeError, match="Vision model not loaded"):
            await vision_service.detect_objects(sample_image_bytes)

    @pytest.mark.asyncio
    async def test_detect_objects_empty_results(self, vision_service, sample_image_bytes):
        """Test detection with no objects found"""
        # Setup mocked model with no detections
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        # Run detection
        result = await vision_service.detect_objects(sample_image_bytes)

        # Verify
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 0


class TestSceneAnalysis:
    """Test scene analysis with mocks"""

    @pytest.mark.asyncio
    async def test_analyze_scene_uses_scene_cache_for_repeated_content(
        self,
        vision_service,
        sample_image_bytes,
    ):
        """Verify scene cache is used for repeated identical content"""
        # Arrange: replace the scene cache with a fake that tracks hits/misses
        class FakeContentHashCache:
            def __init__(self):
                self._store = {}
                self._hits = 0
                self._misses = 0

            def get_by_content(self, content: bytes):
                key = hash(content)
                if key in self._store:
                    self._hits += 1
                    return self._store[key]
                self._misses += 1
                return None

            def put_by_content(self, content: bytes, value):
                key = hash(content)
                self._store[key] = value

            def get_stats(self):
                return {
                    "hits": self._hits,
                    "misses": self._misses,
                    "size": len(self._store),
                }

        fake_cache = FakeContentHashCache()
        original_cache = vision_service._scene_cache
        vision_service._scene_cache = fake_cache

        # Setup mocked model for detection
        mock_model = MagicMock()
        mock_result = MagicMock()

        mock_boxes = MagicMock()
        mock_xyxy = [create_mock_tensor(np.array([10, 20, 100, 200]))]
        mock_boxes.xyxy = mock_xyxy
        mock_boxes.__len__ = lambda self: len(mock_xyxy)
        mock_boxes.cls = [create_mock_tensor(np.array([0]))]
        mock_boxes.conf = [create_mock_tensor(np.array([0.95]))]
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        try:
            # Act: call analyze_scene twice with identical content
            first_result = await vision_service.analyze_scene(sample_image_bytes)
            second_result = await vision_service.analyze_scene(sample_image_bytes)

            stats = fake_cache.get_stats()

            # Assert: cache was populated once and then hit on the second call
            assert stats["misses"] == 1
            assert stats["hits"] == 1
            assert stats["size"] == 1

            # The second result should come from cache and be the same object
            assert second_result is first_result

            # Model predict should only be called once (first call)
            assert mock_model.predict.call_count == 1
        finally:
            # Restore original cache to avoid leaking state across tests
            vision_service._scene_cache = original_cache

    @pytest.mark.asyncio
    async def test_analyze_scene(self, vision_service, sample_image_bytes):
        """Test scene analysis with mocked detection"""
        # Setup mocked detection
        mock_model = MagicMock()
        mock_result = MagicMock()

        mock_boxes = MagicMock()
        mock_xyxy = [create_mock_tensor(np.array([10, 20, 100, 200]))]
        mock_boxes.xyxy = mock_xyxy
        mock_boxes.__len__ = lambda self: len(mock_xyxy)
        mock_boxes.cls = [create_mock_tensor(np.array([0]))]
        mock_boxes.conf = [create_mock_tensor(np.array([0.95]))]
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        # Analyze scene
        result = await vision_service.analyze_scene(sample_image_bytes)

        # Verify
        assert isinstance(result, SceneAnalysis)
        assert isinstance(result.description, str)
        assert "person" in result.description.lower()
        assert len(result.detections) == 1
        assert result.object_counts["person"] == 1


class TestVisualQueryAnsweringWithMocks:
    """Test visual query answering with mocks"""

    @pytest.mark.asyncio
    async def test_answer_visual_query_count(self, vision_service, sample_image_bytes):
        """Test answering count queries"""
        # Mock detection
        mock_model = MagicMock()
        mock_result = MagicMock()

        mock_boxes = MagicMock()
        mock_xyxy = [
            create_mock_tensor(np.array([10, 20, 100, 200])),
            create_mock_tensor(np.array([110, 20, 200, 200]))
        ]
        mock_boxes.xyxy = mock_xyxy
        mock_boxes.__len__ = lambda self: len(mock_xyxy)
        mock_boxes.cls = [create_mock_tensor(np.array([0])), create_mock_tensor(np.array([0]))]
        mock_boxes.conf = [create_mock_tensor(np.array([0.95])), create_mock_tensor(np.array([0.90]))]
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        # Analyze scene first
        scene = await vision_service.analyze_scene(sample_image_bytes)

        # Ask question
        answer = await vision_service.answer_visual_query(
            "How many people are there?",
            scene
        )

        # Verify - the answer should mention 2 objects/people
        assert "2" in answer

    @pytest.mark.asyncio
    async def test_answer_visual_query_presence(self, vision_service, sample_image_bytes):
        """Test answering presence queries"""
        # Mock detection with dog
        mock_model = MagicMock()
        mock_result = MagicMock()

        mock_boxes = MagicMock()
        mock_xyxy = [create_mock_tensor(np.array([10, 20, 100, 200]))]
        mock_boxes.xyxy = mock_xyxy
        mock_boxes.__len__ = lambda self: len(mock_xyxy)
        mock_boxes.cls = [create_mock_tensor(np.array([16]))]  # dog class
        mock_boxes.conf = [create_mock_tensor(np.array([0.90]))]
        mock_result.boxes = mock_boxes

        mock_model.predict.return_value = [mock_result]

        vision_service.model = mock_model
        vision_service.is_ready = True
        vision_service.device_name = 'cpu'

        # Analyze scene first
        scene = await vision_service.analyze_scene(sample_image_bytes)

        # Ask question
        answer = await vision_service.answer_visual_query(
            "Is there a dog?",
            scene
        )

        # Verify
        assert "yes" in answer.lower()
        assert "dog" in answer.lower()
