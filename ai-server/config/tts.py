"""TTS Service Configuration Constants"""

import os
from pathlib import Path


# Model Configuration
MODEL_PATH = os.getenv(
    "TTS_MODEL_PATH",
    "models/tts/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
VOICE = os.getenv("TTS_VOICE", "lessac")
LANGUAGE = os.getenv("TTS_LANGUAGE", "en_US")
SPEAKER_ID = int(os.getenv("TTS_SPEAKER_ID", "0"))

# Multi-Voice Support
ENABLE_MULTI_VOICE = os.getenv("TTS_ENABLE_MULTI_VOICE", "false").lower() == "true"
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "lessac")

# Voice Definitions
# Add voice configurations here as you download more Piper models
# Format: name -> (model_path, description, sample_rate, speaker_id)
AVAILABLE_VOICES = {
    "lessac": {
        "model_path": "models/tts/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "description": "Clear, professional female voice",
        "sample_rate": 22050,
        "speaker_id": 0,
        "language": "en_US"
    },
    # Add more voices here:
    # "amy": {
    #     "model_path": "models/tts/en/en_US/amy/medium/en_US-amy-medium.onnx",
    #     "description": "Friendly female voice",
    #     "sample_rate": 22050,
    #     "speaker_id": 0,
    #     "language": "en_US"
    # },
}

# Synthesis Parameters
SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "22050"))
LENGTH_SCALE = float(os.getenv("TTS_LENGTH_SCALE", "1.0"))  # Speaking rate
NOISE_SCALE = float(os.getenv("TTS_NOISE_SCALE", "0.667"))  # Pitch variation
NOISE_W = float(os.getenv("TTS_NOISE_W", "0.8"))  # Energy variation

# Output Settings
OUTPUT_FORMAT = os.getenv("TTS_OUTPUT_FORMAT", "mp3")  # wav, mp3, ogg, raw
MP3_BITRATE = int(os.getenv("TTS_MP3_BITRATE", "128"))  # kbps
OGG_QUALITY = int(os.getenv("TTS_OGG_QUALITY", "6"))  # 0-10

# Performance
ENABLE_GPU = os.getenv("TTS_ENABLE_GPU", "false").lower() == "true"
THREADS = int(os.getenv("TTS_THREADS", "2"))
CACHE_COMMON_PHRASES = os.getenv("TTS_CACHE_COMMON_PHRASES", "true").lower() == "true"
CACHE_INCLUDES_PARAMETERS = os.getenv("TTS_CACHE_INCLUDES_PARAMETERS", "true").lower() == "true"

# Text Processing
NORMALIZE_NUMBERS = os.getenv("TTS_NORMALIZE_NUMBERS", "true").lower() == "true"
NORMALIZE_DATES = os.getenv("TTS_NORMALIZE_DATES", "true").lower() == "true"
MAX_TEXT_LENGTH = int(os.getenv("TTS_MAX_TEXT_LENGTH", "500"))  # Characters
MIN_TEXT_LENGTH = int(os.getenv("TTS_MIN_TEXT_LENGTH", "1"))

# Common Phrases to Cache (if caching enabled)
COMMON_PHRASES = [
    "Hello",
    "Goodbye",
    "How can I help you?",
    "I don't understand",
    "Could you repeat that?",
    "One moment please",
    "I'm sorry",
    "Thank you",
    "You're welcome"
]

# Performance Thresholds
MAX_SYNTHESIS_TIME_SECONDS = 30.0  # Timeout
WARNING_RTF_THRESHOLD = 0.5  # Real-time factor (warn if slower)

# SSML Support
ENABLE_SSML = os.getenv("TTS_ENABLE_SSML", "true").lower() == "true"

# Logging
LOG_SYNTHESES = os.getenv("TTS_LOG_SYNTHESES", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
LOG_AUDIO_INFO = os.getenv("TTS_LOG_AUDIO_INFO", "true").lower() == "true"
