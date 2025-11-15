"""AI Model Manager - Centralized model loading and lifecycle management"""

import asyncio
import os
from typing import Dict, Optional
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService
from config.hardware import HardwareOptimizer
from utils import get_logger


logger = get_logger(__name__)


class AIModelManager:
    """Manages loading, unloading, and lifecycle of AI models"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        stt_service: Optional[STTService] = None,
        tts_service: Optional[TTSService] = None
    ):
        self.llm_service = llm_service
        self.stt_service = stt_service
        self.tts_service = tts_service
        self._models_loaded = False
        self._hardware_info = None

    def _detect_hardware(self) -> Dict:
        """Detect hardware capabilities once and share across all services"""
        if self._hardware_info is None:
            logger.info("detecting_hardware_capabilities")
            hw_optimizer = HardwareOptimizer()
            self._hardware_info = {
                'device_type': hw_optimizer.hardware_info.device_type,
                'device_name': hw_optimizer.hardware_info.device_name,
                'acceleration': hw_optimizer.hardware_info.device_type,
                'cpu_count': os.cpu_count() or 4,
                'vram_gb': hw_optimizer.hardware_info.memory_gb
            }
            logger.info(
                "hardware_detected",
                device_type=self._hardware_info['device_type'],
                device_name=self._hardware_info['device_name'],
                cpu_count=self._hardware_info['cpu_count'],
                vram_gb=self._hardware_info['vram_gb']
            )
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
            logger.info("loading_model", model_type="LLM")
            tasks.append(self.llm_service.load_model(hardware_info=hardware_info))
            task_names.append("LLM")
        else:
            logger.info("service_not_configured", service="LLM", fallback="pattern-based responses")

        if self.stt_service:
            logger.info("loading_model", model_type="STT")
            tasks.append(self.stt_service.load_model(hardware_info=hardware_info))
            task_names.append("STT")
        else:
            logger.info("service_not_configured", service="STT", impact="audio input disabled")

        if self.tts_service:
            logger.info("loading_model", model_type="TTS")
            tasks.append(self.tts_service.load_model(hardware_info=hardware_info))
            task_names.append("TTS")
        else:
            logger.info("service_not_configured", service="TTS", impact="audio output disabled")

        # Load all models in parallel if there are any to load
        if tasks:
            logger.info("loading_models_parallel", model_count=len(tasks))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result, name in zip(results, task_names):
                if isinstance(result, Exception):
                    logger.error("model_loading_failed", model_type=name, error=str(result), error_type=type(result).__name__)
                    if name == "LLM":
                        success = False  # LLM failure affects overall success
                elif result is True:
                    service = self.llm_service if name == "LLM" else (self.stt_service if name == "STT" else self.tts_service)
                    logger.info("model_loaded_successfully", model_type=name, model_path=service.model_path.name)
                else:
                    logger.warning("model_load_failed", model_type=name)
                    if name == "LLM":
                        success = False  # LLM failure affects overall success
                        logger.warning("using_fallback_responses")
                    else:
                        logger.warning("service_unavailable", service=name.lower())

        self._models_loaded = success
        return success

    def unload_models(self):
        """Unload all AI models during shutdown"""
        if not self._models_loaded:
            logger.debug("no_models_to_unload")
            return

        # Unload LLM model
        if self.llm_service and self.llm_service.is_ready:
            self.llm_service.unload_model()
            logger.info("model_unloaded", model_type="LLM")

        # Unload STT model
        if self.stt_service and self.stt_service.is_ready:
            self.stt_service.unload_model()
            logger.info("model_unloaded", model_type="STT")

        # Unload TTS model
        if self.tts_service and self.tts_service.is_ready:
            self.tts_service.unload_model()
            logger.info("model_unloaded", model_type="TTS")

        self._models_loaded = False

    def get_llm_service(self) -> Optional[LLMService]:
        """Get the LLM service instance"""
        return self.llm_service

    def get_stt_service(self) -> Optional[STTService]:
        """Get the STT service instance"""
        return self.stt_service

    def get_tts_service(self) -> Optional[TTSService]:
        """Get the TTS service instance"""
        return self.tts_service

    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self._models_loaded

    def get_status(self) -> dict:
        """Get status of all AI models"""
        status = {
            "models_loaded": self._models_loaded,
            "llm": None,
            "stt": None,
            "tts": None
        }

        if self.llm_service:
            status["llm"] = self.llm_service.get_status()

        if self.stt_service:
            status["stt"] = self.stt_service.get_status()

        if self.tts_service:
            status["tts"] = self.tts_service.get_status()

        return status
