"""Vision Service for object detection and scene understanding using YOLOv8"""

import asyncio
import base64
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False

from config import vision as vision_config
from services.base import BaseAIService
from utils import ContentHashCache, get_logger


logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class Detection:
    """Object detection result"""
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "center": list(self.bbox.center),
            "area": self.bbox.area
        }


@dataclass
class DetectionResult:
    """Complete detection result with metadata"""
    detections: List[Detection]
    image_size: Tuple[int, int]
    inference_time_ms: int
    timestamp: str
    model_name: str
    device: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "object_counts": self.get_object_counts(),
            "image_size": list(self.image_size),
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "device": self.device
        }

    def get_object_counts(self) -> Dict[str, int]:
        """Get count of each detected object type"""
        counts = {}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts


@dataclass
class SceneAnalysis:
    """Scene understanding result"""
    description: str
    detections: List[Detection]
    object_counts: Dict[str, int]
    main_objects: List[str]
    confidence: float
    inference_time_ms: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "description": self.description,
            "detections": [d.to_dict() for d in self.detections],
            "object_counts": self.object_counts,
            "main_objects": self.main_objects,
            "confidence": round(self.confidence, 3),
            "inference_time_ms": self.inference_time_ms
        }


class VisionService(BaseAIService):
    """Service for object detection and scene understanding"""

    def __init__(self, cache_size: int = 32, cache_ttl: float = 60.0):
        super().__init__(str(vision_config.MODEL_PATH))
        self.model_name = f"yolov8{vision_config.MODEL_SIZE}"
        self.device_name = None
        self._scene_cache: ContentHashCache[SceneAnalysis] = ContentHashCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
            name="vision_scene_cache"
        )

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "Vision"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        logger.info(
            "vision_configuration",
            model=self.model_name,
            device=self.device_name,
            image_size=vision_config.IMAGE_SIZE,
            confidence_threshold=vision_config.CONFIDENCE_THRESHOLD,
            iou_threshold=vision_config.IOU_THRESHOLD
        )

    async def _load_model_impl(self) -> bool:
        """Vision-specific model loading logic"""
        try:
            # Check if ultralytics is available
            if not HAS_ULTRALYTICS or YOLO is None:
                logger.error(
                    "ultralytics_not_installed",
                    suggestion="Run: pip install ultralytics"
                )
                return False

            # Determine device
            device = self._get_device()
            self.device_name = device

            logger.info(
                "loading_yolo_model",
                model=self.model_name,
                device=device,
                half_precision=vision_config.HALF_PRECISION and device != "cpu"
            )

            # Load model (blocking operation, run in executor)
            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: YOLO(str(self.model_path))
                )

                # Set model to evaluation mode and move to device
                await loop.run_in_executor(
                    None,
                    lambda: self.model.to(device)
                )

                # Enable half precision if using GPU
                if vision_config.HALF_PRECISION and device != "cpu":
                    logger.debug("enabling_half_precision")
                    # YOLOv8 handles half precision internally

                logger.info(
                    "yolo_model_loaded",
                    classes=len(vision_config.COCO_CLASSES),
                    device=device
                )

                return True

            except Exception as e:
                logger.error(
                    "yolo_model_load_error",
                    error=str(e),
                    error_type=type(e).__name__
                )
                return False

        except Exception as e:
            logger.error(
                "vision_model_loading_exception",
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    def _get_device(self) -> str:
        """Determine optimal device based on hardware"""
        if vision_config.DEVICE != "auto":
            return vision_config.DEVICE

        # Use hardware info from base class
        if self.hardware_info:
            device_type = self.hardware_info.get('device_type', 'cpu')
            if device_type == 'cuda' and vision_config.ENABLE_GPU:
                return 'cuda:0'
            elif device_type == 'mps' and vision_config.ENABLE_GPU:
                return 'mps'

        return 'cpu'

    async def detect_objects(
        self,
        image: Union[bytes, np.ndarray, str, Image.Image],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> DetectionResult:
        """Detect objects in an image

        Args:
            image: Image as bytes, numpy array, file path, or PIL Image
            confidence: Confidence threshold (default from config)
            iou_threshold: IOU threshold for NMS (default from config)
            max_detections: Maximum number of detections (default from config)

        Returns:
            DetectionResult with all detected objects
        """
        if not self.is_ready or self.model is None:
            logger.error("vision_model_not_loaded")
            raise RuntimeError("Vision model not loaded")

        start_time = time.time()

        # Set parameters
        conf_threshold = confidence or vision_config.CONFIDENCE_THRESHOLD
        iou_thresh = iou_threshold or vision_config.IOU_THRESHOLD
        max_det = max_detections or vision_config.MAX_DETECTIONS

        try:
            # Preprocess image
            processed_image, original_size = await self._preprocess_image(image)

            # Run inference (blocking, run in executor)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.model.predict(
                    processed_image,
                    conf=conf_threshold,
                    iou=iou_thresh,
                    max_det=max_det,
                    imgsz=vision_config.IMAGE_SIZE,
                    verbose=False
                )
            )

            # Parse results
            detections = self._parse_detections(results[0], original_size)

            inference_time_ms = int((time.time() - start_time) * 1000)

            result = DetectionResult(
                detections=detections,
                image_size=original_size,
                inference_time_ms=inference_time_ms,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model_name=self.model_name,
                device=self.device_name
            )

            if vision_config.LOG_DETECTIONS:
                logger.info(
                    "object_detection_complete",
                    num_detections=len(detections),
                    inference_time_ms=inference_time_ms,
                    object_counts=result.get_object_counts()
                )

            return result

        except Exception as e:
            logger.error(
                "object_detection_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _preprocess_image(
        self,
        image: Union[bytes, np.ndarray, str, Image.Image]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for inference with memory-efficient handling

        Args:
            image: Input image in various formats

        Returns:
            Tuple of (processed_image, original_size)
        """
        pil_image = None
        buffer = None

        try:
            # Convert to PIL Image with explicit resource management
            if isinstance(image, bytes):
                # Try to open as raw bytes first, if that fails try base64 decode
                try:
                    buffer = BytesIO(image)
                    pil_image = Image.open(buffer)
                    pil_image.load()  # Force load to allow buffer reuse
                except Exception:
                    # Might be base64-encoded, try decoding
                    if buffer:
                        buffer.close()
                    try:
                        image_data = base64.b64decode(image)
                        buffer = BytesIO(image_data)
                        pil_image = Image.open(buffer)
                        pil_image.load()
                        del image_data  # Free decoded bytes
                    except Exception:
                        raise
            elif isinstance(image, str):
                pil_image = Image.open(image)
                pil_image.load()
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Convert to RGB if needed (creates new image, close old one)
            if pil_image.mode != 'RGB':
                rgb_image = pil_image.convert('RGB')
                if pil_image is not image:  # Don't close if it was passed in
                    pil_image.close()
                pil_image = rgb_image

            original_size = pil_image.size  # (width, height)

            # Convert to numpy array for YOLO
            image_array = np.array(pil_image)

            return image_array, original_size

        except Exception as e:
            logger.error(
                "image_preprocessing_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

        finally:
            # Clean up resources
            if buffer:
                buffer.close()
            if pil_image is not None and pil_image is not image:
                try:
                    pil_image.close()
                except Exception:
                    pass

    def _parse_detections(
        self,
        result,
        original_size: Tuple[int, int]
    ) -> List[Detection]:
        """Parse YOLO results into Detection objects

        Args:
            result: YOLO result object
            original_size: Original image size (width, height)

        Returns:
            List of Detection objects
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            # Get bounding box coordinates (xyxy format)
            bbox_coords = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox_coords)

            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())

            # Get class name
            class_name = vision_config.COCO_CLASSES[class_id] if class_id < len(vision_config.COCO_CLASSES) else f"class_{class_id}"

            # Apply class filter if configured
            if vision_config.ALLOWED_CLASSES and class_name not in vision_config.ALLOWED_CLASSES:
                continue

            detection = Detection(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            )
            detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    async def analyze_scene(
        self,
        image: Union[bytes, np.ndarray, str, Image.Image],
        query: Optional[str] = None
    ) -> SceneAnalysis:
        """Analyze a scene and generate description

        Args:
            image: Input image
            query: Optional specific query about the image

        Returns:
            SceneAnalysis with description and detected objects
        """
        # Check cache for bytes input (most common from MQTT)
        image_bytes: Optional[bytes] = None
        if isinstance(image, bytes):
            image_bytes = image
            cached = self._scene_cache.get_by_content(image_bytes)
            if cached is not None:
                logger.debug("scene_cache_hit", cache_stats=self._scene_cache.get_stats())
                return cached

        start_time = time.time()

        # Detect objects
        detection_result = await self.detect_objects(image)

        # Generate description
        description = self._generate_description(
            detection_result.detections,
            query
        )

        # Identify main objects (most prominent)
        main_objects = self._identify_main_objects(detection_result.detections)

        # Calculate overall confidence
        avg_confidence = (
            sum(d.confidence for d in detection_result.detections) / len(detection_result.detections)
            if detection_result.detections
            else 0.0
        )

        inference_time_ms = int((time.time() - start_time) * 1000)

        scene = SceneAnalysis(
            description=description,
            detections=detection_result.detections,
            object_counts=detection_result.get_object_counts(),
            main_objects=main_objects,
            confidence=avg_confidence,
            inference_time_ms=inference_time_ms
        )

        # Cache result for bytes input
        if image_bytes is not None:
            self._scene_cache.put_by_content(image_bytes, scene)
            logger.debug("scene_cached", cache_stats=self._scene_cache.get_stats())

        return scene

    def _generate_description(
        self,
        detections: List[Detection],
        query: Optional[str] = None
    ) -> str:
        """Generate natural language description of detected objects

        Args:
            detections: List of detected objects
            query: Optional specific query

        Returns:
            Natural language description
        """
        if not detections:
            return "I don't see any recognizable objects in this image."

        # Filter by confidence threshold
        filtered = [
            d for d in detections
            if d.confidence >= vision_config.DESCRIPTION_MIN_CONFIDENCE
        ][:vision_config.DESCRIPTION_MAX_OBJECTS]

        if not filtered:
            return "I can see some objects, but I'm not confident enough to identify them clearly."

        # Get counts
        counts = {}
        for det in filtered:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        # Build description
        parts = []
        for obj_name, count in counts.items():
            if count == 1:
                parts.append(f"1 {obj_name}")
            else:
                parts.append(f"{count} {obj_name}" if obj_name.endswith('s') else f"{count} {obj_name}s")

        if len(parts) == 1:
            description = f"I see {parts[0]}"
        elif len(parts) == 2:
            description = f"I see {parts[0]} and {parts[1]}"
        else:
            description = f"I see {', '.join(parts[:-1])}, and {parts[-1]}"

        description += " in this image."

        return description

    def _identify_main_objects(
        self,
        detections: List[Detection],
        top_n: int = 3
    ) -> List[str]:
        """Identify the main objects in the scene

        Args:
            detections: List of detected objects
            top_n: Number of main objects to return

        Returns:
            List of main object class names
        """
        if not detections:
            return []

        scored_detections = [
            (det.confidence * (det.bbox.area**0.5), det.class_name)
            for det in detections
        ]
        scored_detections.sort(reverse=True)

        # Get unique class names
        main_objects = []
        seen = set()
        for _, class_name in scored_detections:
            if class_name not in seen:
                main_objects.append(class_name)
                seen.add(class_name)
            if len(main_objects) >= top_n:
                break

        return main_objects

    async def answer_visual_query(
        self,
        question: str,
        scene: 'SceneAnalysis'
    ) -> str:
        """Answer a question about an image using pre-computed scene analysis

        Args:
            question: Question about the image
            scene: Pre-computed scene analysis from analyze_scene()

        Returns:
            Natural language answer
        """

        # For now, use rule-based answering
        # In the future, this can integrate with LLM for more sophisticated answers
        question_lower = question.lower()

        # Count questions
        if "how many" in question_lower:
            for obj_name in scene.object_counts:
                if obj_name in question_lower:
                    count = scene.object_counts[obj_name]
                    return f"I see {count} {obj_name}{'s' if count != 1 else ''} in the image."

            # General count
            total = sum(scene.object_counts.values())
            return f"I see {total} object{'s' if total != 1 else ''} in total."

        # Presence questions
        if "is there" in question_lower or "do you see" in question_lower:
            for obj_name in vision_config.COCO_CLASSES:
                if obj_name in question_lower:
                    if obj_name not in scene.object_counts:
                        return f"No, I don't see any {obj_name}s in the image."

                    count = scene.object_counts[obj_name]
                    return f"Yes, I see {count} {obj_name}{'s' if count != 1 else ''}."
        return scene.description

    def get_status(self) -> Dict:
        """Get current service status"""
        status = super().get_status()
        status.update({
            "model_name": self.model_name,
            "device": self.device_name,
            "image_size": vision_config.IMAGE_SIZE,
            "confidence_threshold": vision_config.CONFIDENCE_THRESHOLD,
            "scene_cache": self._scene_cache.get_stats(),
        })
        return status
