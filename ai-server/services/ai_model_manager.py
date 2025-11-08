"""AI Model Manager - Centralized model loading and lifecycle management"""

import logging
from typing import Optional
from services.llm import LLMService
from services.stt import STTService


logger = logging.getLogger(__name__)


class AIModelManager:
    """Manages loading, unloading, and lifecycle of AI models"""

    def __init__(self, llm_service: Optional[LLMService] = None, stt_service: Optional[STTService] = None):
        self.llm_service = llm_service
        self.stt_service = stt_service
        self._models_loaded = False

    async def load_models(self) -> bool:
        """Load all AI models during startup"""
        if self._models_loaded:
            logger.warning("Models already loaded")
            return True

        success = True

        # Load LLM model if configured
        if self.llm_service:
            logger.info("Loading LLM model...")
            llm_success = await self.llm_service.load_model()
            if llm_success:
                logger.info(f"LLM model loaded successfully: {self.llm_service.model_path.name}")
            else:
                logger.warning("LLM model failed to load - will use fallback responses")
                success = False
        else:
            logger.info("No LLM service configured - using pattern-based responses")

        # Load STT model if configured
        if self.stt_service:
            logger.info("Loading STT model...")
            stt_success = await self.stt_service.load_model()
            if stt_success:
                logger.info(f"STT model loaded successfully: {self.stt_service.model_path.name}")
            else:
                logger.warning("STT model failed to load - audio input will be unavailable")
                # Don't mark overall success as False - STT is optional
        else:
            logger.info("No STT service configured - audio input disabled")

        self._models_loaded = success
        return success

    def unload_models(self):
        """Unload all AI models during shutdown"""
        if not self._models_loaded:
            logger.debug("No models to unload")
            return

        # Unload LLM model
        if self.llm_service and self.llm_service.is_ready:
            self.llm_service.unload_model()
            logger.info("LLM model unloaded")

        # Unload STT model
        if self.stt_service and self.stt_service.is_ready:
            self.stt_service.unload_model()
            logger.info("STT model unloaded")

        self._models_loaded = False

    def get_llm_service(self) -> Optional[LLMService]:
        """Get the LLM service instance"""
        return self.llm_service

    def get_stt_service(self) -> Optional[STTService]:
        """Get the STT service instance"""
        return self.stt_service

    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self._models_loaded

    def get_status(self) -> dict:
        """Get status of all AI models"""
        status = {
            "models_loaded": self._models_loaded,
            "llm": None,
            "stt": None
        }

        if self.llm_service:
            status["llm"] = self.llm_service.get_status()

        if self.stt_service:
            status["stt"] = self.stt_service.get_status()

        return status
