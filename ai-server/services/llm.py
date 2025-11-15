"""LLM Service for text generation using Llama models"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import llm as llm_config
from services.base import BaseAIService


logger = logging.getLogger(__name__)


class LLMService(BaseAIService):
    """Service for loading and running LLM inference"""

    def __init__(self):
        super().__init__(llm_config.MODEL_PATH)
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            if llm_config.SYSTEM_PROMPT_FILE.exists():
                with open(llm_config.SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    logger.info(f"Loaded system prompt from {llm_config.SYSTEM_PROMPT_FILE}")
                    return prompt
        except Exception as e:
            logger.warning(f"Failed to load system prompt file: {e}")

        logger.info("Using fallback system prompt")
        return llm_config.FALLBACK_SYSTEM_PROMPT

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "LLM"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        n_threads = self._get_thread_count(llm_config.THREADS)
        n_gpu_layers = self._get_gpu_layers()
        logger.info(f"Configuration: threads={n_threads}, gpu_layers={n_gpu_layers}")

    async def _load_model_impl(self) -> bool:
        """LLM-specific model loading logic"""
        try:
            # Import llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError:
                logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
                return False

            # Determine optimal settings
            n_threads = self._get_thread_count(llm_config.THREADS)
            n_gpu_layers = self._get_gpu_layers()

            # Load model (this is blocking, so run in executor)
            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: Llama(
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
                    f"Out of memory while loading model. "
                    f"Try reducing CONTEXT_SIZE (current: {llm_config.CONTEXT_SIZE}) "
                    f"or GPU_LAYERS (current: {n_gpu_layers}). "
                    f"Error: {e}"
                )
                return False
            except ValueError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg or "metal" in error_msg:
                    logger.error(
                        f"Hardware incompatibility detected. "
                        f"GPU acceleration may not be available or supported. "
                        f"Try setting GPU_LAYERS=0 for CPU-only mode. "
                        f"Error: {e}"
                    )
                else:
                    logger.error(f"Invalid model configuration: {e}")
                return False
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.error(
                        f"Out of memory error during model loading. "
                        f"Current config: context_size={llm_config.CONTEXT_SIZE}, "
                        f"gpu_layers={n_gpu_layers}, threads={n_threads}. "
                        f"Consider reducing these values. Error: {e}"
                    )
                else:
                    logger.error(f"Runtime error loading model: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"Exception during model loading: {e}", exc_info=True)
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
                    logger.info(f"GPU detected with {vram_gb:.1f}GB VRAM, enabling full GPU acceleration")
                    return -1  # All layers
                elif vram_gb >= 8:
                    layers = int(estimated_layers * 0.75)
                    logger.info(f"GPU detected with {vram_gb:.1f}GB VRAM, enabling partial GPU acceleration ({layers} layers, ~75%)")
                    return layers
                elif vram_gb >= 4:
                    layers = int(estimated_layers * 0.5)
                    logger.info(f"GPU detected with {vram_gb:.1f}GB VRAM, enabling partial GPU acceleration ({layers} layers, ~50%)")
                    return layers
                else:
                    layers = int(estimated_layers * 0.25)
                    logger.info(f"GPU detected with {vram_gb:.1f}GB VRAM, enabling minimal GPU acceleration ({layers} layers, ~25%)")
                    return layers
            else:
                logger.info("GPU detected, VRAM unknown, enabling full GPU acceleration")
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

        try:
            start_time = time.time()

            # Build the prompt from messages
            prompt = self._format_chat_prompt(messages)

            if llm_config.LOG_PROMPTS:
                logger.debug(f"Prompt:\n{prompt}")

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
            if not response or 'choices' not in response or not response['choices']:
                logger.error("Invalid response structure from LLM")
                return ""

            if 'text' not in response['choices'][0]:
                logger.error("Missing 'text' field in LLM response")
                return ""

            generated_text = response['choices'][0]['text']
            generated_text = self._clean_response(generated_text)

            # Log performance
            inference_time = time.time() - start_time
            tokens_generated = response['usage']['completion_tokens']
            tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0

            if llm_config.LOG_PERFORMANCE_METRICS:
                logger.info(
                    f"Generation: {inference_time:.2f}s, "
                    f"{tokens_generated} tokens, "
                    f"{tokens_per_sec:.1f} tok/s"
                )

            if inference_time > llm_config.WARNING_INFERENCE_TIME_SECONDS:
                logger.warning(f"Slow inference: {inference_time:.2f}s")

            if llm_config.LOG_RESPONSES:
                logger.debug(f"Response: {generated_text}")

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
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
            logger.warning(f"Response truncated from {len(text)} to {llm_config.MAX_RESPONSE_LENGTH} chars")
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
        return status
