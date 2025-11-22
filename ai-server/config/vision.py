"""Vision Service Configuration"""

import os
from pathlib import Path

# Model Configuration
MODEL_PATH = Path(os.getenv("VISION_MODEL_PATH", "models/vision/yolov8n.pt"))
MODEL_SIZE = os.getenv("VISION_MODEL_SIZE", "n")  # n, s, m, l, x
DEVICE = os.getenv("VISION_DEVICE", "auto")  # cpu, cuda, mps, auto

# Detection Settings
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("VISION_IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("VISION_MAX_DETECTIONS", "100"))
IMAGE_SIZE = int(os.getenv("VISION_IMAGE_SIZE", "640"))

# Performance Settings
HALF_PRECISION = os.getenv("VISION_HALF_PRECISION", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("VISION_BATCH_SIZE", "1"))
ENABLE_GPU = os.getenv("VISION_ENABLE_GPU", "true").lower() == "true"

# Output Settings
ANNOTATE_IMAGES = os.getenv("VISION_ANNOTATE_IMAGES", "false").lower() == "true"
SAVE_DETECTIONS = os.getenv("VISION_SAVE_DETECTIONS", "false").lower() == "true"
ENABLE_TRACKING = os.getenv("VISION_ENABLE_TRACKING", "false").lower() == "true"

# COCO Classes (80 classes from COCO dataset)
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

# Class filtering (empty list = all classes allowed)
ALLOWED_CLASSES_STR = os.getenv("VISION_ALLOWED_CLASSES", "")
ALLOWED_CLASSES = [c.strip() for c in ALLOWED_CLASSES_STR.split(",") if c.strip()] if ALLOWED_CLASSES_STR else []

# Logging
LOG_DETECTIONS = os.getenv("VISION_LOG_DETECTIONS", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True

# Scene description settings
DESCRIPTION_MAX_OBJECTS = int(os.getenv("VISION_DESCRIPTION_MAX_OBJECTS", "10"))
DESCRIPTION_MIN_CONFIDENCE = float(os.getenv("VISION_DESCRIPTION_MIN_CONFIDENCE", "0.5"))
