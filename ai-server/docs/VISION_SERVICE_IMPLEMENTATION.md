# Vision Service Implementation Plan

**Status**: Not Started
**Priority**: P2 - Medium
**Owner**: AI-Server Team
**Created**: 2025-11-04
**Model**: YOLOv8 Nano (yolov8n.pt)

---

## Overview

Implement the Vision service to analyze images from the robot's camera and detect objects, people, and scenes. This service will enable NAILA to understand its visual environment, answer questions about what it sees, and respond to visual queries in natural language.

---

## Current State Analysis

### What Exists
- ✅ Model downloaded: `models/vision/yolov8n.pt` (6.3MB)
- ✅ Hardware detection: `config/hardware_config.py`
- ✅ MQTT topics defined: `naila/ai/processing/vision/+`
- ✅ Orchestration pipeline ready
- ✅ LLM service for describing visual scenes

### What's Missing
- ❌ Vision service implementation
- ❌ Image preprocessing (resizing, normalization)
- ❌ Object detection and classification
- ❌ Scene description generation
- ❌ Integration with orchestration pipeline
- ❌ MQTT image message handling
- ❌ Visual query understanding
- ❌ Performance monitoring for inference

### Dependencies Required
```
ultralytics>=8.3.225          # YOLOv8 official library
pillow>=12.0.0                # Image processing
opencv-python>=4.11.0.86      # Computer vision utilities
numpy>=2.3.4                  # Array operations
torch>=2.9.0                  # PyTorch backend (already installed)
torchvision>=0.24.0           # Vision utilities
```

**Recommendation**: Use `ultralytics` for official YOLOv8 support with best performance.

---

## Implementation Plan

### Phase 1: Service Foundation (Create the Vision Service Class)

**Goal**: Create the basic vision service structure with model loading

**Files to Create**:
- `services/vision.py`
- `config/vision.py`

**Implementation Details**:

1. **Create `VisionService` class** with:
   - `__init__()` - Initialize configuration
   - `async load_model()` - Load YOLOv8 model
   - `async detect_objects()` - Detect objects in image
   - `async analyze_scene()` - Generate scene description
   - `async answer_visual_query()` - Answer questions about image
   - `unload_model()` - Clean up resources
   - `is_loaded()` - Check if model is ready

2. **Configuration Parameters** (`config/vision.py`):
   - Model path from `.env`
   - Confidence threshold (default: 0.25)
   - IOU threshold for NMS (default: 0.45)
   - Max detections per image (default: 100)
   - Image size for inference (default: 640x640)
   - Device (CPU/CUDA/MPS)
   - Class labels (COCO dataset - 80 classes)

3. **Hardware Optimization**:
   - Use `hardware_config.py` to determine optimal settings
   - Enable GPU acceleration if available (CUDA/Metal)
   - Set appropriate batch size
   - Configure half-precision (FP16) for speed

4. **Image Preprocessing**:
   - Decode base64 or load from file
   - Resize to model input size (640x640)
   - Normalize pixel values
   - Handle various image formats (JPEG, PNG, etc.)
   - Apply transforms (letterboxing, padding)

**Key Considerations**:
- YOLOv8 Nano is smallest/fastest YOLO variant
- Trained on COCO dataset (80 object classes)
- Model loading takes 1-2 seconds
- Inference time ~20-50ms on GPU, ~100-300ms on CPU
- Input size 640x640 is standard for YOLOv8

**Success Criteria**:
- [ ] VisionService class created with all methods
- [ ] Model loads successfully without errors
- [ ] Can detect objects in images accurately
- [ ] Hardware acceleration configured correctly
- [ ] Image preprocessing working

---

### Phase 2: Object Detection Engine

**Goal**: Implement accurate and efficient object detection

**Implementation Details**:

1. **Detection Method**:
   ```python
   async def detect_objects(
       image: Union[bytes, np.ndarray, str],
       confidence: float = 0.25,
       iou_threshold: float = 0.45,
       max_detections: int = 100
   ) -> DetectionResult:
       # Preprocess image
       # Run YOLOv8 inference
       # Apply NMS (Non-Maximum Suppression)
       # Post-process detections
       # Return structured result
   ```

2. **Detection Result Structure**:
   ```python
   @dataclass
   class Detection:
       class_name: str          # "person", "dog", "car", etc.
       class_id: int            # COCO class ID
       confidence: float        # 0.0-1.0
       bbox: BoundingBox        # x1, y1, x2, y2
       center: Tuple[int, int]  # Center point
       area: int                # Bounding box area

   @dataclass
   class BoundingBox:
       x1: int  # Top-left x
       y1: int  # Top-left y
       x2: int  # Bottom-right x
       y2: int  # Bottom-right y

   @dataclass
   class DetectionResult:
       detections: List[Detection]
       image_size: Tuple[int, int]
       inference_time_ms: int
       timestamp: str
       model_name: str
       device: str
   ```

3. **Object Classes** (COCO Dataset - 80 classes):
   - **People**: person
   - **Vehicles**: bicycle, car, motorcycle, bus, truck, train
   - **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
   - **Furniture**: chair, couch, bed, dining table, desk
   - **Electronics**: tv, laptop, mouse, keyboard, cell phone, remote
   - **Kitchen**: bottle, cup, bowl, fork, knife, spoon, microwave, oven
   - **Food**: banana, apple, sandwich, orange, broccoli, carrot, pizza, donut, cake
   - And 50+ more classes...

4. **Post-Processing**:
   - Filter by confidence threshold
   - Apply Non-Maximum Suppression (NMS)
   - Sort by confidence or size
   - Group nearby objects
   - Calculate spatial relationships

**Success Criteria**:
- [ ] Object detection working accurately
- [ ] Confidence thresholds effective
- [ ] NMS removes duplicate detections
- [ ] Bounding boxes accurate
- [ ] Inference time acceptable (<300ms CPU, <50ms GPU)

---

### Phase 3: Scene Understanding & Description

**Goal**: Generate natural language descriptions of visual scenes

**Implementation Details**:

1. **Scene Analysis Method**:
   ```python
   async def analyze_scene(
       image: Union[bytes, np.ndarray],
       query: Optional[str] = None
   ) -> SceneAnalysis:
       # Detect all objects
       # Analyze spatial relationships
       # Count objects by type
       # Identify main subjects
       # Generate description using LLM
       return scene_analysis
   ```

2. **Scene Analysis Structure**:
   ```python
   @dataclass
   class SceneAnalysis:
       description: str                    # Natural language description
       detections: List[Detection]         # All detected objects
       object_counts: Dict[str, int]       # {"person": 2, "car": 1}
       main_objects: List[str]             # Primary subjects
       spatial_info: Dict[str, Any]        # Spatial relationships
       confidence: float                   # Overall confidence
       inference_time_ms: int
   ```

3. **Description Generation**:
   ```python
   # Example scene descriptions:
   # Simple: "I see 2 people and 1 dog in this image."
   # Detailed: "There are two people standing near a car. A dog is sitting on the left side."
   # With context: "This appears to be a parking lot with people and their pet."
   ```

4. **Integration with LLM**:
   - Pass detected objects to LLM
   - Include spatial relationships
   - LLM generates natural description
   - Contextualize with conversation history

5. **Visual Query Answering**:
   ```python
   async def answer_visual_query(
       image: Union[bytes, np.ndarray],
       question: str
   ) -> str:
       # Detect objects in image
       # Analyze spatial relationships
       # Build context for LLM
       # Use LLM to answer question
       # Return natural language answer
   ```

   **Example Queries**:
   - "What do you see?" → "I see two people and a dog."
   - "How many people are there?" → "There are two people in the image."
   - "Is there a dog?" → "Yes, I see a dog sitting on the left."
   - "What color is the car?" → "I can detect a car, but I cannot determine colors with this model."

**Success Criteria**:
- [ ] Scene descriptions are accurate
- [ ] Natural language quality is good
- [ ] Visual queries answered correctly
- [ ] Spatial relationships identified
- [ ] LLM integration working

---

### Phase 4: MQTT Integration & Image Handling

**Goal**: Handle image messages via MQTT and publish results

**Files to Modify**:
- `mqtt/handlers/ai_handlers.py`
- `mqtt/handlers/coordinator.py`

**Implementation Details**:

1. **MQTT Image Message Format**:
   ```json
   {
       "device_id": "naila_001",
       "timestamp": "2025-11-04T12:34:56Z",
       "image_data": "<base64_encoded_image>",
       "format": "jpeg",
       "width": 1280,
       "height": 720,
       "query": "What do you see?",
       "metadata": {
           "source": "camera",
           "camera_id": "front"
       }
   }
   ```

2. **Vision Handler**:
   ```python
   # In ai_handlers.py
   async def handle_vision_query(self, message):
       # Extract image data from MQTT message
       # Decode base64 image
       # Validate image format

       # If query provided:
       #     answer_visual_query()
       # Else:
       #     analyze_scene()

       # Send results back via MQTT
   ```

3. **MQTT Topics**:
   - **Input**: `naila/device/{device_id}/camera/capture`
   - **Output**: `naila/ai/processing/vision/results`
   - **Queries**: `naila/device/{device_id}/vision/query`

4. **Vision Result Message**:
   ```json
   {
       "device_id": "naila_001",
       "timestamp": "2025-11-04T12:34:57Z",
       "description": "I see two people standing near a car with a dog.",
       "detections": [
           {
               "class": "person",
               "confidence": 0.92,
               "bbox": [120, 45, 280, 650]
           },
           {
               "class": "person",
               "confidence": 0.89,
               "bbox": [350, 60, 490, 640]
           },
           {
               "class": "dog",
               "confidence": 0.87,
               "bbox": [50, 400, 180, 600]
           },
           {
               "class": "car",
               "confidence": 0.95,
               "bbox": [500, 200, 900, 550]
           }
       ],
       "object_counts": {
           "person": 2,
           "dog": 1,
           "car": 1
       },
       "inference_time_ms": 145
   }
   ```

**Success Criteria**:
- [ ] MQTT image messages handled correctly
- [ ] Base64 decoding working
- [ ] Vision results published properly
- [ ] Error handling for bad images
- [ ] End-to-end image-to-description working

---

### Phase 5: Integration with Orchestration Pipeline

**Goal**: Connect vision service to the orchestration graph

**Files to Modify**:
- `agents/orchestrator.py`
- `graphs/orchestration_graph.py`

**Implementation Details**:

1. **Add Vision Node to Graph**:
   ```python
   # In orchestration_graph.py
   class NAILAOrchestrationGraph:
       def __init__(self, llm_service=None, vision_service=None):
           self.vision_service = vision_service

           # Add vision node
           if self.vision_service:
               workflow.add_node("vision_analysis", self._vision_node)
   ```

2. **Vision Context Injection**:
   ```python
   # When image is provided:
   # 1. Run vision analysis
   # 2. Add results to state context
   # 3. Pass to LLM for natural language response

   state["visual_context"] = {
       "detected_objects": [...],
       "scene_description": "...",
       "object_counts": {...}
   }
   ```

3. **Multimodal Query Handling**:
   ```
   User: "What do you see?" + [image]
      ↓
   Vision Service: Detects objects
      ↓
   LLM: Generates natural description
      ↓
   Response: "I see two people and a dog near a car."
   ```

4. **Context-Aware Visual Understanding**:
   ```
   User: "Is there a dog in this picture?" + [image]
      ↓
   Vision: Detects dog (confidence 0.87)
      ↓
   LLM with context: "Yes, I see a dog sitting on the left side."
   ```

**Success Criteria**:
- [ ] Vision integrated into orchestration
- [ ] Visual context flows to LLM
- [ ] Multimodal queries work
- [ ] Context-aware responses generated
- [ ] End-to-end visual Q&A working

---

### Phase 6: Server Lifecycle Integration

**Goal**: Load vision model during startup, handle gracefully during shutdown

**Files to Modify**:
- `server/lifecycle.py`
- `server/naila_server.py`

**Implementation Details**:

1. **Add Vision Loading Phase**:
   ```python
   # In lifecycle.py _load_ai_models()
   async def _load_ai_models(self):
       # Load LLM (existing)
       if self.llm_service:
           await self.llm_service.load_model()

       # Load STT (existing)
       if self.stt_service:
           await self.stt_service.load_model()

       # Load TTS (existing)
       if self.tts_service:
           await self.tts_service.load_model()

       # Load Vision (NEW)
       if self.vision_service:
           logger.info("Loading Vision model...")
           success = await self.vision_service.load_model()
           if success:
               logger.info(f"Vision model loaded: {self.vision_service.model_name}")
               self.protocol_handlers.set_vision_service(self.vision_service)
           else:
               logger.warning("Vision model failed to load - visual queries disabled")
   ```

2. **Initialization Sequence**:
   ```
   Phase 1: Initialize configuration
   Phase 2: Load AI models
       - Load LLM model
       - Load STT model
       - Load TTS model
       - Load Vision model (NEW)
   Phase 3: Register protocol handlers
   Phase 4: Start MQTT service
   Phase 5: Start health monitoring
   ```

3. **Dependency Injection**:
   - Store Vision service in `NailaAIServer`
   - Pass to `AIHandlers` for image processing
   - Pass to `Orchestrator` for multimodal queries
   - Make available via dependency injection

4. **Shutdown Handling**:
   - Unload vision model during graceful shutdown
   - Free GPU/memory resources
   - Cancel any in-flight inference

**Success Criteria**:
- [ ] Vision loads during server startup
- [ ] Loading status visible in logs
- [ ] Vision service accessible to components
- [ ] Graceful shutdown unloads vision properly
- [ ] Server handles vision load failures without crashing

---

### Phase 7: Performance Optimization & Monitoring

**Goal**: Ensure vision performs efficiently and track key metrics

**Implementation Details**:

1. **Performance Monitoring**:
   - Track inference time per request
   - Monitor detections per image
   - Log confidence scores
   - Track model memory usage
   - Count successful vs failed detections

2. **Add to Health Monitoring**:
   - Include vision status in health checks
   - Report vision metrics in `naila/ai/metrics` topic
   - Alert if inference times exceed thresholds

3. **Optimization Techniques**:
   - Use half-precision (FP16) on GPU
   - Optimize image size (trade quality vs speed)
   - Batch multiple images if needed
   - Cache common detections (future)
   - Use TensorRT for deployment (future)

4. **Logging & Debugging**:
   - Log image dimensions and format
   - Log detection results (debug mode)
   - Track inference parameters
   - Log any detection errors

**Metrics to Track**:
```python
{
    "vision_status": "loaded" | "unloaded" | "error",
    "model_name": "yolov8n",
    "inference_time_ms": 145,
    "image_width": 1280,
    "image_height": 720,
    "num_detections": 4,
    "avg_confidence": 0.91,
    "device": "cuda" | "cpu",
    "memory_usage_mb": 250
}
```

**Success Criteria**:
- [ ] Inference metrics tracked and logged
- [ ] Performance data included in health metrics
- [ ] Inference time reasonable (<300ms CPU, <50ms GPU)
- [ ] No memory leaks during extended operation

---

### Phase 8: Advanced Features

**Goal**: Enhance vision service with advanced capabilities

**Implementation Details**:

1. **Object Tracking** (Future):
   - Track objects across multiple frames
   - Maintain object IDs over time
   - Detect motion and direction

2. **Semantic Segmentation** (Future):
   - Pixel-level understanding
   - Scene segmentation
   - Background/foreground separation

3. **Face Detection & Recognition** (Future):
   - Detect faces in images
   - Recognize known individuals
   - Privacy-aware processing

4. **Color & Attribute Detection**:
   - Detect dominant colors
   - Identify object attributes
   - Enhance descriptions with details

5. **Spatial Reasoning**:
   - Understand spatial relationships
   - Detect "on", "under", "near", "between"
   - Answer spatial queries

6. **Multi-Camera Support**:
   - Handle multiple camera streams
   - Fuse information from different views
   - Panoramic understanding

**Success Criteria**:
- [ ] Advanced features working
- [ ] Spatial reasoning accurate
- [ ] Color detection functional
- [ ] Multi-camera support implemented

---

### Phase 9: Testing & Validation

**Goal**: Ensure vision service works correctly in all scenarios

**Test Cases**:

1. **Unit Tests** (`tests/unit/test_vision_service.py`):
   - [ ] Model loading succeeds
   - [ ] Detect objects in sample images
   - [ ] Image preprocessing works
   - [ ] Handle invalid images gracefully
   - [ ] Confidence filtering works
   - [ ] Unload model properly

2. **Integration Tests** (`tests/integration/test_vision_integration.py`):
   - [ ] Vision loads during server startup
   - [ ] AI handlers process image messages
   - [ ] Vision results reach orchestrator
   - [ ] End-to-end image-to-description flow
   - [ ] Fallback works when vision unavailable

3. **Detection Test Cases**:
   - [ ] Single object (person)
   - [ ] Multiple objects (people, pets, objects)
   - [ ] Small objects (cell phone, cup)
   - [ ] Large objects (car, truck)
   - [ ] Crowded scenes (many objects)
   - [ ] Empty scenes (no objects)
   - [ ] Low light images
   - [ ] Blurry images
   - [ ] Various image formats (JPEG, PNG, WEBP)

4. **Accuracy Testing**:
   - [ ] Detection accuracy > 80% (mAP@0.5)
   - [ ] False positive rate acceptable
   - [ ] False negative rate acceptable
   - [ ] Confidence scores reliable

5. **Manual Testing**:
   - [ ] Send real images via MQTT
   - [ ] Verify detection accuracy
   - [ ] Check response times
   - [ ] Test visual queries
   - [ ] Validate descriptions

**Success Criteria**:
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Detection accuracy meets expectations
- [ ] Processing time acceptable

---

## Technical Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                      NAILA AI Server                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │  MQTT Layer  │────────▶│  AI Handlers    │               │
│  │              │         └────────┬────────┘               │
│  │ Camera Topic │                  │                         │
│  └──────────────┘                  ▼                         │
│                          ┌──────────────────┐               │
│                          │ Vision Service   │◀──┐           │
│                          │  (NEW)           │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │ Image            │   │           │
│                          │ Preprocessing    │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │ YOLOv8 Inference │   │           │
│                          │ (PyTorch)        │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │ Object Detection │   │           │
│                          │ & NMS            │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   │ Detections  │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │  Orchestrator    │   │           │
│                          └────────┬─────────┘   │           │
│                                   │             │           │
│                                   ▼             │           │
│                          ┌──────────────────┐   │           │
│                          │  LLM Service     │───┘           │
│                          │  (Description)   │               │
│                          └──────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
1. Robot captures image from camera
   ↓
2. Image sent via MQTT (base64 encoded)
   ↓
3. AI Handler receives image message
   ↓
4. Vision Service:
   - Decodes image (base64 → bytes)
   - Preprocesses (resize, normalize)
   - Runs YOLOv8 inference
   - Applies NMS
   - Extracts detections
   ↓
5. Detections sent to Orchestrator
   ↓
6. If visual query:
   - Build context with detections
   - Pass to LLM for natural language answer
   ↓
7. Response sent back via MQTT:
   - Text description
   - Detection results (JSON)
   - Annotated image (optional)
   ↓
8. Robot displays/speaks response
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# Vision Configuration
VISION_MODEL_PATH=models/vision/yolov8n.pt
VISION_MODEL_SIZE=n  # n, s, m, l, x (nano to extra-large)
VISION_DEVICE=auto   # cpu, cuda, mps, auto

# Detection Settings
VISION_CONFIDENCE_THRESHOLD=0.25
VISION_IOU_THRESHOLD=0.45
VISION_MAX_DETECTIONS=100
VISION_IMAGE_SIZE=640

# Performance
VISION_HALF_PRECISION=true  # FP16 on GPU
VISION_BATCH_SIZE=1
VISION_ENABLE_GPU=true

# Output Settings
VISION_ANNOTATE_IMAGES=false
VISION_SAVE_DETECTIONS=false
VISION_ENABLE_TRACKING=false

# Class Filtering (optional)
VISION_ALLOWED_CLASSES=person,dog,cat,car,chair,bottle,cup
```

### Configuration Module (`config/vision.py`)
```python
"""Vision Service Configuration Constants"""

import os
from pathlib import Path

# Model Configuration
MODEL_PATH = os.getenv("VISION_MODEL_PATH", "models/vision/yolov8n.pt")
MODEL_SIZE = os.getenv("VISION_MODEL_SIZE", "n")
DEVICE = os.getenv("VISION_DEVICE", "auto")

# Detection Settings
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("VISION_IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("VISION_MAX_DETECTIONS", "100"))
IMAGE_SIZE = int(os.getenv("VISION_IMAGE_SIZE", "640"))

# Performance
HALF_PRECISION = os.getenv("VISION_HALF_PRECISION", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("VISION_BATCH_SIZE", "1"))
ENABLE_GPU = os.getenv("VISION_ENABLE_GPU", "true").lower() == "true"

# Output Settings
ANNOTATE_IMAGES = os.getenv("VISION_ANNOTATE_IMAGES", "false").lower() == "true"
SAVE_DETECTIONS = os.getenv("VISION_SAVE_DETECTIONS", "false").lower() == "true"
ENABLE_TRACKING = os.getenv("VISION_ENABLE_TRACKING", "false").lower() == "true"

# COCO Classes (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Allowed classes filter (empty = all classes)
ALLOWED_CLASSES = os.getenv("VISION_ALLOWED_CLASSES", "").split(",") if os.getenv("VISION_ALLOWED_CLASSES") else []

# Logging
LOG_DETECTIONS = os.getenv("VISION_LOG_DETECTIONS", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
```

---

## Risks & Mitigations

### Risk 1: Model Loading Time
- **Risk**: 1-2 second startup delay
- **Impact**: Slightly longer server startup
- **Mitigation**: Load in parallel with other models, show progress

### Risk 2: Inference Speed
- **Risk**: Slow inference on CPU (100-300ms)
- **Impact**: Delayed visual responses
- **Mitigation**:
  - Use YOLOv8 Nano (fastest variant)
  - Enable GPU acceleration
  - Use half-precision (FP16)
  - Optimize image size

### Risk 3: Detection Accuracy
- **Risk**: False positives/negatives, missed objects
- **Impact**: Incorrect descriptions, user frustration
- **Mitigation**:
  - Tune confidence threshold
  - Use appropriate model size (nano vs larger)
  - Clear error handling for low confidence
  - Allow user to provide corrections

### Risk 4: Limited Understanding
- **Risk**: Can only detect objects, not understand context
- **Impact**: Shallow visual understanding
- **Mitigation**:
  - Integrate with LLM for context
  - Analyze spatial relationships
  - Combine with conversation history
  - Set clear expectations

### Risk 5: Memory Usage
- **Risk**: Model + images use significant RAM
- **Impact**: Increased memory footprint
- **Mitigation**:
  - Use nano model (smallest)
  - Process images one at a time
  - Clean up after inference
  - Ensure sufficient RAM

### Risk 6: Image Quality Issues
- **Risk**: Blurry, dark, or poor quality images
- **Impact**: Failed or inaccurate detections
- **Mitigation**:
  - Image quality validation
  - Brightness/contrast adjustment
  - Clear error messages
  - Request better image from robot

---

## Success Metrics

### Functional Metrics
- [ ] Model loads successfully on server start
- [ ] Detects objects with >80% accuracy (mAP@0.5)
- [ ] Handles various image formats correctly
- [ ] Generates natural scene descriptions
- [ ] Answers visual queries accurately
- [ ] Server stable during extended operation

### Performance Metrics
- Model load time: < 3 seconds
- Inference time (GPU): < 50ms
- Inference time (CPU): < 300ms
- Memory usage: < 500MB
- No memory leaks over 24 hour operation

### Quality Metrics
- Detection accuracy (mAP@0.5): > 80%
- False positive rate: < 10%
- Confidence correlation: High confidence → High accuracy
- Description quality: Natural and accurate

---

## Timeline & Milestones

### Milestone 1: Basic Vision Service (3-4 hours)
- Create VisionService class
- Implement model loading
- Basic object detection working

### Milestone 2: Scene Understanding (2-3 hours)
- Scene description generation
- LLM integration
- Visual query answering

### Milestone 3: MQTT Integration (2-3 hours)
- Image message handling
- Result publishing
- Error handling

### Milestone 4: Orchestration Integration (2-3 hours)
- Add to orchestration graph
- Multimodal query support
- Context injection

### Milestone 5: Testing & Polish (2-3 hours)
- Write tests
- Performance tuning
- Accuracy validation

**Total Estimated Time**: 11-16 hours

---

## Dependencies & Prerequisites

### Required
- [x] YOLOv8 model downloaded
- [ ] `ultralytics` package installed
- [ ] `opencv-python` installed
- [ ] `pillow` installed
- [x] PyTorch installed (from LLM service)
- [x] Hardware detection working
- [x] MQTT infrastructure exists
- [x] LLM service exists (for descriptions)

### System Dependencies
```bash
# No additional system packages needed
# All dependencies are Python packages
```

### Optional (Future Enhancements)
- [ ] GPU available for acceleration
- [ ] TensorRT for optimized inference
- [ ] Larger YOLOv8 models for better accuracy
- [ ] Segmentation models

---

## Future Enhancements

### Short-term (Next Sprint)
1. **Object Tracking**: Track objects across frames
2. **Spatial Reasoning**: Understand "on", "under", "near"
3. **Color Detection**: Identify dominant colors
4. **Image Annotation**: Return annotated images with bounding boxes

### Long-term (Future Sprints)
1. **Semantic Segmentation**: Pixel-level understanding
2. **Face Recognition**: Identify known individuals
3. **Scene Classification**: Classify scene types (indoor, outdoor, etc.)
4. **Multi-Camera Fusion**: Combine multiple camera views
5. **3D Understanding**: Depth estimation and 3D reasoning
6. **Action Recognition**: Detect activities and actions
7. **Visual SLAM**: Simultaneous localization and mapping

---

## References

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Related Code
- `services/llm.py` - Similar service pattern to follow
- `mqtt/handlers/ai_handlers.py` - Where image messages will be handled
- `agents/orchestrator.py` - Where visual context will be integrated
- `config/hardware_config.py` - Hardware optimization
- `server/lifecycle.py` - Server startup/shutdown

---

## Status Tracking

### Phase 1: Service Foundation
- [ ] Create `services/vision.py`
- [ ] Create `config/vision.py`
- [ ] Implement `VisionService` class
- [ ] Implement `load_model()` method
- [ ] Implement `detect_objects()` method
- [ ] Hardware optimization configured
- [ ] Image preprocessing working

### Phase 2: Object Detection Engine
- [ ] Detection method implemented
- [ ] NMS working correctly
- [ ] Confidence filtering
- [ ] Post-processing pipeline
- [ ] COCO classes mapped

### Phase 3: Scene Understanding
- [ ] Scene analysis method
- [ ] Description generation
- [ ] LLM integration
- [ ] Visual query answering
- [ ] Spatial relationships

### Phase 4: MQTT Integration
- [ ] Image message parsing
- [ ] Base64 decoding
- [ ] Result publishing
- [ ] Error handling
- [ ] Topic subscription

### Phase 5: Orchestration Integration
- [ ] Add vision node to graph
- [ ] Context injection
- [ ] Multimodal queries
- [ ] End-to-end flow
- [ ] Integration testing

### Phase 6: Server Lifecycle
- [ ] Add vision loading phase
- [ ] Integrate into `lifecycle.py`
- [ ] Shutdown handling
- [ ] Error handling
- [ ] Status logging

### Phase 7: Performance & Monitoring
- [ ] Inference metrics tracked
- [ ] Health monitoring
- [ ] Performance logging
- [ ] Optimization applied
- [ ] Metrics validated

### Phase 8: Advanced Features
- [ ] Object tracking
- [ ] Color detection
- [ ] Spatial reasoning
- [ ] Multi-camera support

### Phase 9: Testing & Validation
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Accuracy validated
- [ ] Manual testing complete
- [ ] Quality meets expectations

---

**Last Updated**: 2025-11-04
**Next Review**: After Phase 1 completion
