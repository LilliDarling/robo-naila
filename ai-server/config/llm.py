"""LLM Service Configuration Constants"""

import os
from pathlib import Path


# Model Configuration
MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
MODEL_TYPE = "llama"  # Model architecture type

# Context and Token Limits
CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
MAX_TOKENS_PER_RESPONSE = int(os.getenv("LLM_MAX_TOKENS", "512"))
CONTEXT_HISTORY_LIMIT = int(os.getenv("LLM_CONTEXT_HISTORY_LIMIT", "5"))
TOKEN_BUDGET_FOR_HISTORY = int(os.getenv("LLM_TOKEN_BUDGET_HISTORY", "2000"))

# Generation Parameters
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))

# Hardware Configuration
GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))  # 0 = CPU only, -1 = auto (use GPU if available), N = specific layers
THREADS = int(os.getenv("LLM_THREADS", "0"))  # 0 = auto-detect
BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "512"))
USE_MMAP = os.getenv("LLM_USE_MMAP", "true").lower() == "true"
USE_MLOCK = os.getenv("LLM_USE_MLOCK", "false").lower() == "true"

# Stop Sequences
STOP_SEQUENCES = [
    "<|eot_id|>",
    "<|end_of_text|>",
    "\n\nUser:",
    "\n\nHuman:",
]

# Llama 3.1 Special Tokens
LLAMA_3_BEGIN_OF_TEXT = "<|begin_of_text|>"
LLAMA_3_START_HEADER = "<|start_header_id|>"
LLAMA_3_END_HEADER = "<|end_header_id|>"
LLAMA_3_EOT = "<|eot_id|>"

# Performance Thresholds
MAX_INFERENCE_TIME_SECONDS = 30.0  # Timeout for generation
WARNING_INFERENCE_TIME_SECONDS = 10.0  # Log warning if slower

# Prompt Configuration
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system.txt"
FALLBACK_SYSTEM_PROMPT = "You are NAILA, a helpful robot assistant."

# Response Formatting
STRIP_WHITESPACE = True
MAX_RESPONSE_LENGTH = 1000  # Characters (safety limit)
MIN_RESPONSE_LENGTH = 1  # Characters

# Error Handling
ENABLE_FALLBACK_RESPONSES = True
MAX_RETRIES = 2  # Retry generation on failure
RETRY_DELAY_SECONDS = 1.0

# Logging
LOG_PROMPTS = os.getenv("LLM_LOG_PROMPTS", "false").lower() == "true"  # Debug only
LOG_RESPONSES = os.getenv("LLM_LOG_RESPONSES", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = True
