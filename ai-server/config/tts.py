"""TTS Service Configuration — Kokoro ONNX"""

import os


# Model Configuration
MODEL_PATH = os.getenv(
    "TTS_MODEL_PATH",
    "models/tts/kokoro/kokoro-v1.0.onnx"
)
VOICES_PATH = os.getenv(
    "TTS_VOICES_PATH",
    "models/tts/kokoro/voices-v1.0.bin"
)

# Voice Selection (see get_voices() for full list)
# American female: af_heart, af_bella, af_nicole, af_sarah, af_sky, af_nova, ...
# British female:  bf_emma, bf_isabella, bf_lily, bf_alice
# American male:   am_adam, am_michael, am_eric, ...
VOICE = os.getenv("TTS_VOICE", "bf_emma")
LANGUAGE = os.getenv("TTS_LANGUAGE", "en-us")

# Synthesis Parameters
SAMPLE_RATE = 24000  # Kokoro outputs at 24kHz (fixed)
SPEED = float(os.getenv("TTS_SPEED", "1.0"))  # Speaking rate (1.0 = normal)

# Text Processing
MAX_TEXT_LENGTH = int(os.getenv("TTS_MAX_TEXT_LENGTH", "500"))
MIN_TEXT_LENGTH = int(os.getenv("TTS_MIN_TEXT_LENGTH", "1"))
NORMALIZE_NUMBERS = os.getenv("TTS_NORMALIZE_NUMBERS", "true").lower() == "true"
NORMALIZE_DATES = os.getenv("TTS_NORMALIZE_DATES", "true").lower() == "true"

# Caching
CACHE_COMMON_PHRASES = os.getenv("TTS_CACHE_COMMON_PHRASES", "true").lower() == "true"
CACHE_INCLUDES_PARAMETERS = os.getenv("TTS_CACHE_INCLUDES_PARAMETERS", "true").lower() == "true"
MAX_CACHED_PHRASES = int(os.getenv("TTS_MAX_CACHED_PHRASES", "256"))
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

# Concurrency Control
MAX_CONCURRENT_REQUESTS = int(os.getenv("TTS_MAX_CONCURRENT_REQUESTS", "4"))
POOL_TIMEOUT_SECONDS = float(os.getenv("TTS_POOL_TIMEOUT_SECONDS", "30.0"))

# Performance Thresholds
MAX_SYNTHESIS_TIME_SECONDS = 30.0
WARNING_RTF_THRESHOLD = 0.5

# Logging
LOG_SYNTHESES = os.getenv("TTS_LOG_SYNTHESES", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
LOG_AUDIO_INFO = os.getenv("TTS_LOG_AUDIO_INFO", "true").lower() == "true"
