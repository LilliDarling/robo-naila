"""AI Model Manager - Centralized model loading and lifecycle management"""

import asyncio
import logging
import os
from typing import Dict, Optional
from services.llm import LLMService
from services.stt import STTService
from config.hardware_config import HardwareOptimizer


logger = logging.getLogger(__name__)


class AIModelManager:
    """Manages loading, unloading, and lifecycle of AI models"""

    def __init__(self, llm_service: Optional[LLMService] = None, stt_service: Optional[STTService] = None):
        self.llm_service = llm_service
        self.stt_service = stt_service
        self._models_loaded = False
        self._hardware_info = None

    def _detect_hardware(self) -> Dict:
        """Detect hardware capabilities once and share across all services"""
        if self._hardware_info is None:
            logger.info("Detecting hardware capabilities...")
            hw_optimizer = HardwareOptimizer()
            self._hardware_info = {
                'device_type': hw_optimizer.hardware_info.device_type,
                'device_name': hw_optimizer.hardware_info.device_name,
                'acceleration': hw_optimizer.hardware_info.device_type,
                'cpu_count': os.cpu_count() or 4,
                'vram_gb': hw_optimizer.hardware_info.memory_gb
            }
            logger.info(f"Hardware detected: {self._hardware_info}")
        return self._hardware_info

    async def load_models(self) -> bool:
        """Load all AI models during startup with shared hardware detection and parallel loading"""
        if self._models_loaded:
            logger.warning("Models already loaded")
            return True

        success = True

        # Detect hardware once for all services
        hardware_info = self._detect_hardware()

        # Prepare loading tasks for parallel execution
        tasks = []
        task_names = []

        if self.llm_service:
            logger.info("Loading LLM model...")
            tasks.append(self.llm_service.load_model(hardware_info=hardware_info))
            task_names.append("LLM")
        else:
            logger.info("No LLM service configured - using pattern-based responses")

        if self.stt_service:
            logger.info("Loading STT model...")
            tasks.append(self.stt_service.load_model(hardware_info=hardware_info))
            task_names.append("STT")
        else:
            logger.info("No STT service configured - audio input disabled")

        # Load all models in parallel if there are any to load
        if tasks:
            logger.info(f"Loading {len(tasks)} model(s) in parallel...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result, name in zip(results, task_names):
                if isinstance(result, Exception):
                    logger.error(f"{name} model loading failed with exception: {result}")
                    if name == "LLM":
                        success = False  # LLM failure affects overall success
                elif result is True:
                    service = self.llm_service if name == "LLM" else self.stt_service
                    logger.info(f"{name} model loaded successfully: {service.model_path.name}")
                else:
                    logger.warning(f"{name} model failed to load")
                    if name == "LLM":
                        success = False  # LLM failure affects overall success
                        logger.warning("Will use fallback responses")
                    else:
                        logger.warning("Audio input will be unavailable")

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
