"""STT Service Configuration Constants"""

import os
from pathlib import Path


# Model Configuration
MODEL_PATH = os.getenv("STT_MODEL_PATH", "models/stt/ggml-small.en.bin")
LANGUAGE = os.getenv("STT_LANGUAGE", "en")
MODEL_SIZE = "small.en"  # Whisper model size

# Transcription Parameters
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
BEST_OF = int(os.getenv("STT_BEST_OF", "5"))
TEMPERATURE = float(os.getenv("STT_TEMPERATURE", "0.0"))  # 0 = deterministic
PATIENCE = float(os.getenv("STT_PATIENCE", "1.0"))

# Hardware Configuration
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")  # float16, int8, float32
DEVICE = os.getenv("STT_DEVICE", "auto")  # cpu, cuda, auto
THREADS = int(os.getenv("STT_THREADS", "0"))  # 0 = auto-detect
GPU_LAYERS = int(os.getenv("STT_GPU_LAYERS", "-1"))  # -1 = auto

# Audio Settings
SAMPLE_RATE = int(os.getenv("STT_SAMPLE_RATE", "16000"))  # Whisper requirement
CHANNELS = 1  # Mono audio required by Whisper

# VAD (Voice Activity Detection) Settings
VAD_FILTER = os.getenv("STT_VAD_FILTER", "true").lower() == "true"
VAD_THRESHOLD = float(os.getenv("STT_VAD_THRESHOLD", "0.5"))
MIN_SILENCE_DURATION_MS = int(os.getenv("STT_MIN_SILENCE_DURATION_MS", "500"))
SPEECH_PAD_MS = int(os.getenv("STT_SPEECH_PAD_MS", "400"))

# Quality Settings
MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.6"))
REJECT_LOW_CONFIDENCE = os.getenv("STT_REJECT_LOW_CONFIDENCE", "false").lower() == "true"
MIN_DURATION_MS = int(os.getenv("STT_MIN_DURATION_MS", "100"))
MAX_DURATION_MS = int(os.getenv("STT_MAX_DURATION_MS", "30000"))  # 30 seconds
MIN_TEXT_LENGTH = int(os.getenv("STT_MIN_TEXT_LENGTH", "1"))
MAX_TEXT_LENGTH = int(os.getenv("STT_MAX_TEXT_LENGTH", "500"))

# Supported Audio Formats
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a", "webm"]

# Performance Thresholds
MAX_TRANSCRIPTION_TIME_SECONDS = 30.0  # Timeout
WARNING_RTF_THRESHOLD = 1.0  # Real-time factor (warn if slower than real-time)
ENABLE_WARMUP = os.getenv("STT_ENABLE_WARMUP", "true").lower() == "true"
WARMUP_DURATION_MS = int(os.getenv("STT_WARMUP_DURATION_MS", "1000"))  # Duration of warm-up audio

# Response Formatting
STRIP_WHITESPACE = True
NORMALIZE_WHITESPACE = True  # Replace multiple spaces with single space
CAPITALIZE_FIRST = True  # Capitalize first letter

# Error Handling
ENABLE_FALLBACK = True  # Return empty on failure vs raise exception
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 0.5

# Logging
LOG_TRANSCRIPTIONS = os.getenv("STT_LOG_TRANSCRIPTIONS", "true").lower() == "true"
LOG_AUDIO_INFO = os.getenv("STT_LOG_AUDIO_INFO", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
