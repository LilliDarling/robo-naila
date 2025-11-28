"""LLM Service for text generation using Llama models"""

import asyncio
import time
from typing import Dict, List, Optional

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAS_LLAMA_CPP = False

from config import llm as llm_config
from services.base import BaseAIService
from utils.resource_pool import ResourcePool
from utils import get_logger


logger = get_logger(__name__)


class LLMService(BaseAIService):
    """Service for loading and running LLM inference"""

    def __init__(self):
        super().__init__(llm_config.MODEL_PATH)
        self.system_prompt: Optional[str] = None
        self._pool: Optional[ResourcePool] = None

    async def _load_system_prompt(self) -> str:
        """Load system prompt from file asynchronously"""
        loop = asyncio.get_running_loop()
        try:
            if llm_config.SYSTEM_PROMPT_FILE.exists():
                prompt = await loop.run_in_executor(
                    None,
                    lambda: llm_config.SYSTEM_PROMPT_FILE.read_text(encoding='utf-8').strip()
                )
                logger.info("system_prompt_loaded", prompt_file=str(llm_config.SYSTEM_PROMPT_FILE))
                return prompt
        except Exception as e:
            logger.warning("system_prompt_load_failed", error=str(e), error_type=type(e).__name__)

        logger.info("using_fallback_system_prompt")
        return llm_config.FALLBACK_SYSTEM_PROMPT

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "LLM"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        n_threads = self._get_thread_count(llm_config.THREADS)
        n_gpu_layers = self._get_gpu_layers()
        logger.info("llm_configuration", threads=n_threads, gpu_layers=n_gpu_layers)

    async def _load_model_impl(self) -> bool:
        """LLM-specific model loading logic"""
        try:
            # Check if llama-cpp-python is available
            if not HAS_LLAMA_CPP or Llama is None:
                logger.error("llama_cpp_not_installed", suggestion="Run: pip install llama-cpp-python")
                return False

            # Determine optimal settings
            n_threads = self._get_thread_count(llm_config.THREADS)
            n_gpu_layers = self._get_gpu_layers()

            # Load model (this is blocking, so run in executor)
            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: Llama(  # type: ignore[misc]
                        model_path=str(self.model_path),
                        n_ctx=llm_config.CONTEXT_SIZE,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=llm_config.BATCH_SIZE,
                        use_mmap=llm_config.USE_MMAP,
                        use_mlock=llm_config.USE_MLOCK,
                        verbose=False,
                    )
                )
            except MemoryError as e:
                logger.error(
                    "llm_out_of_memory",
                    context_size=llm_config.CONTEXT_SIZE,
                    gpu_layers=n_gpu_layers,
                    error=str(e),
                    suggestion="Reduce CONTEXT_SIZE or GPU_LAYERS"
                )
                return False
            except ValueError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg or "metal" in error_msg:
                    logger.error(
                        "llm_hardware_incompatibility",
                        error=str(e),
                        suggestion="Try setting GPU_LAYERS=0 for CPU-only mode"
                    )
                else:
                    logger.error("llm_invalid_configuration", error=str(e))
                return False
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.error(
                        "llm_runtime_oom",
                        context_size=llm_config.CONTEXT_SIZE,
                        gpu_layers=n_gpu_layers,
                        threads=n_threads,
                        error=str(e),
                        suggestion="Consider reducing context_size, gpu_layers, or threads"
                    )
                else:
                    logger.error("llm_runtime_error", error=str(e), error_type=type(e).__name__)
                return False

            # Load system prompt asynchronously
            self.system_prompt = await self._load_system_prompt()

            # Initialize resource pool for concurrency control
            self._pool = ResourcePool(
                max_concurrent=llm_config.MAX_CONCURRENT_REQUESTS,
                timeout=llm_config.POOL_TIMEOUT_SECONDS
            )
            logger.info("resource_pool_initialized", max_concurrent=llm_config.MAX_CONCURRENT_REQUESTS)

            return True

        except Exception as e:
            logger.error("llm_model_loading_exception", error=str(e), error_type=type(e).__name__)
            return False

    def _get_gpu_layers(self) -> int:
        """Determine how many layers to offload to GPU, based on hardware capabilities"""
        if llm_config.GPU_LAYERS != -1:
            return llm_config.GPU_LAYERS
        # Auto-detect: offload layers based on VRAM or device capabilities
        if self.hardware_info and self.hardware_info.get('acceleration') in ['cuda', 'metal']:
            vram_gb = self.hardware_info.get('vram_gb')
            if vram_gb is not None:
                # Heuristic: offload more layers for higher VRAM
                # Model-specific: Llama 3.1 8B typically has 32 layers
                # For other models, this should be adjusted
                estimated_layers = 32  # Default for Llama 3.1 8B

                if vram_gb >= 16:
                    logger.info("gpu_acceleration_enabled", vram_gb=round(vram_gb, 1), mode="full", layers="all")
                    return -1  # All layers
                elif vram_gb >= 8:
                    layers = int(estimated_layers * 0.75)
                    logger.info("gpu_acceleration_enabled", vram_gb=round(vram_gb, 1), mode="partial", layers=layers, percentage=75)
                    return layers
                elif vram_gb >= 4:
                    layers = int(estimated_layers * 0.5)
                    logger.info("gpu_acceleration_enabled", vram_gb=round(vram_gb, 1), mode="partial", layers=layers, percentage=50)
                    return layers
                else:
                    layers = int(estimated_layers * 0.25)
                    logger.info("gpu_acceleration_enabled", vram_gb=round(vram_gb, 1), mode="minimal", layers=layers, percentage=25)
                    return layers
            else:
                logger.info("gpu_detected_vram_unknown", mode="full_acceleration")
                return -1  # All layers
        return 0  # CPU only

    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate a chat completion from messages"""
        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot generate")
            return ""

        if self._pool is None:
            return await self._generate_chat_impl(messages, max_tokens, temperature, top_p)
        async with self._pool:
            return await self._generate_chat_impl(messages, max_tokens, temperature, top_p)

    async def _generate_chat_impl(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Internal implementation of chat generation"""
        try:
            start_time = time.time()

            # Build the prompt from messages
            prompt = self._format_chat_prompt(messages)

            if llm_config.LOG_PROMPTS:
                logger.debug("llm_prompt", prompt=prompt)

            # Set generation parameters
            max_tokens = max_tokens or llm_config.MAX_TOKENS_PER_RESPONSE
            temperature = temperature or llm_config.DEFAULT_TEMPERATURE
            top_p = top_p or llm_config.DEFAULT_TOP_P

            # Generate (blocking, run in executor)
            # Type assertion: model is guaranteed non-None after is_ready check
            model = self.model
            assert model is not None, "Model should be loaded"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=llm_config.DEFAULT_TOP_K,
                    repeat_penalty=llm_config.REPEAT_PENALTY,
                    stop=llm_config.STOP_SEQUENCES,
                    echo=False,
                )
            )

            # Extract generated text with validation
            if not response or not response.get('choices'):
                logger.error("llm_invalid_response_structure")
                return ""

            if 'text' not in response['choices'][0]:
                logger.error("llm_missing_text_field")
                return ""

            generated_text = response['choices'][0]['text']
            generated_text = self._clean_response(generated_text)

            # Log performance
            inference_time = time.time() - start_time
            tokens_generated = response['usage']['completion_tokens']
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            if llm_config.LOG_PERFORMANCE_METRICS:
                logger.info(
                    "llm_generation_performance",
                    inference_time_seconds=round(inference_time, 2),
                    tokens_generated=tokens_generated,
                    tokens_per_second=round(tokens_per_sec, 1)
                )

            if inference_time > llm_config.WARNING_INFERENCE_TIME_SECONDS:
                logger.warning("llm_slow_inference", inference_time_seconds=round(inference_time, 2))

            if llm_config.LOG_RESPONSES:
                logger.debug("llm_response", response=generated_text)

            return generated_text

        except Exception as e:
            logger.error("llm_generation_failed", error=str(e), error_type=type(e).__name__)
            return ""

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into Llama 3.1 chat format"""
        # Llama 3.1 Instruct format:
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # {system_message}<|eot_id|>
        # <|start_header_id|>user<|end_header_id|>
        # {user_message}<|eot_id|>
        # <|start_header_id|>assistant<|end_header_id|>
        # {assistant_message}<|eot_id|>

        # Don't add begin_of_text - llama-cpp-python adds it automatically
        prompt_parts = []

        for message in messages:
            role = message['role']
            content = message['content']

            prompt_parts.extend(
                (
                    f"{llm_config.LLAMA_3_START_HEADER}{role}{llm_config.LLAMA_3_END_HEADER}\n",
                    f"{content.strip()}{llm_config.LLAMA_3_EOT}",
                )
            )
        # Add assistant header to start generation
        prompt_parts.append(f"{llm_config.LLAMA_3_START_HEADER}assistant{llm_config.LLAMA_3_END_HEADER}\n")

        return "".join(prompt_parts)

    def _clean_response(self, text: str) -> str:
        """Clean up generated response"""
        if llm_config.STRIP_WHITESPACE:
            text = text.strip()

        # Remove any leaked special tokens only from the end of the response
        for token in llm_config.STOP_SEQUENCES:
            if text.endswith(token):
                text = text[: -len(token)]

        # Apply length limits
        if len(text) > llm_config.MAX_RESPONSE_LENGTH:
            logger.warning("llm_response_truncated", original_length=len(text), truncated_to=llm_config.MAX_RESPONSE_LENGTH)
            text = text[:llm_config.MAX_RESPONSE_LENGTH]

        return text

    def build_chat_messages(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """Build chat messages from query and history"""
        messages = []

        # Add system prompt
        if include_system:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add conversation history
        if conversation_history:
            history_limit = llm_config.CONTEXT_HISTORY_LIMIT
            recent_history = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history

            for exchange in recent_history:
                messages.extend(
                    (
                        {"role": "user", "content": exchange.get("user", "")},
                        {
                            "role": "assistant",
                            "content": exchange.get("assistant", ""),
                        },
                    )
                )
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })

        return messages

    def get_status(self) -> Dict:
        """Get current service status"""
        status = super().get_status()
        status.update({
            "context_size": llm_config.CONTEXT_SIZE,
            "max_tokens": llm_config.MAX_TOKENS_PER_RESPONSE,
        })
        if self._pool is not None:
            status["pool"] = self._pool.get_stats()
        return status
