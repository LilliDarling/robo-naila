"""Base class for AI model services with shared patterns"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from config.hardware import HardwareOptimizer
from utils import get_logger


logger = get_logger(__name__)


class BaseAIService(ABC):
    """Base class for AI model services (LLM, STT, etc.)

    Provides common functionality:
    - Model loading with hardware optimization
    - Thread count calculation
    - Hardware info management
    - Model unloading
    - Status reporting
    """

    def __init__(self, model_path: str):
        """Initialize the service

        Args:
            model_path: Path to the model file
        """
        self.model = None
        self.model_path = Path(model_path)
        self.is_ready = False
        self.hardware_info = None

    async def load_model(self, hardware_info: Optional[Dict] = None) -> bool:
        """Load the model with hardware optimization

        Args:
            hardware_info: Optional pre-detected hardware info. If None, will detect automatically.

        Returns:
            True if successful, False otherwise
        """
        if self.is_ready:
            logger.warning("model_already_loaded", service=self.__class__.__name__)
            return True

        try:
            # Verify model file exists
            if not self.model_path.exists():
                logger.error("model_file_not_found", model_path=str(self.model_path))
                return False

            logger.info("loading_model", model_type=self._get_model_type(), model_path=str(self.model_path))
            start_time = time.time()

            # Use provided hardware info or detect if not provided
            if hardware_info is not None:
                self.hardware_info = hardware_info
                logger.debug("using_shared_hardware_detection")
            else:
                # Fallback to individual detection (for backward compatibility)
                self.hardware_info = self._detect_hardware()
                logger.info(
                    "hardware_detected",
                    device_type=self.hardware_info['device_type'],
                    device_name=self.hardware_info['device_name'],
                    cpu_count=self.hardware_info['cpu_count'],
                    vram_gb=self.hardware_info['vram_gb']
                )

            # Call subclass-specific loading logic
            success = await self._load_model_impl()

            if success:
                load_time = time.time() - start_time
                self.is_ready = True
                logger.info("model_loaded_successfully", model_type=self._get_model_type(), load_time_seconds=round(load_time, 2))
                self._log_configuration()
                return True
            else:
                self.is_ready = False
                return False

        except Exception as e:
            logger.error("model_load_failed", model_type=self._get_model_type(), error=str(e), error_type=type(e).__name__)
            self.is_ready = False
            return False

    @abstractmethod
    async def _load_model_impl(self) -> bool:
        """Subclass implements specific model loading logic

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def _get_model_type(self) -> str:
        """Get the model type name for logging

        Returns:
            Model type string (e.g., "LLM", "STT")
        """
        pass

    @abstractmethod
    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        pass

    def _detect_hardware(self) -> Dict:
        """Detect hardware capabilities

        Returns:
            Dictionary with hardware information
        """
        hw_optimizer = HardwareOptimizer()
        return {
            'device_type': hw_optimizer.hardware_info.device_type,
            'device_name': hw_optimizer.hardware_info.device_name,
            'acceleration': hw_optimizer.hardware_info.device_type,
            'cpu_count': os.cpu_count() or 4,
            'vram_gb': hw_optimizer.hardware_info.memory_gb
        }

    def _get_thread_count(self, config_threads: int) -> int:
        """Determine optimal thread count based on hardware

        Args:
            config_threads: Configured thread count (0 = auto-detect)

        Returns:
            Optimal thread count
        """
        if config_threads > 0:
            return config_threads

        # Auto-detect based on CPU cores
        if self.hardware_info and 'cpu_count' in self.hardware_info:
            cpu_count = self.hardware_info['cpu_count']
            # Use 75% of available cores, minimum 2
            return max(2, int(cpu_count * 0.75))

        return 4  # Safe default

    def get_status(self) -> Dict:
        """Get current service status

        Returns:
            Dictionary with status information
        """
        return {
            "ready": self.is_ready,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "hardware": self.hardware_info,
        }

    def unload_model(self):
        """Unload the model and free resources"""
        if self.model:
            logger.info("unloading_model", model_type=self._get_model_type())
            try:
                # Try to call close() method if available for explicit cleanup
                if hasattr(self.model, 'close'):
                    self.model.close()
                    logger.debug("model_cleanup_called")
                else:
                    logger.debug("no_cleanup_method", fallback="garbage_collection")
            except Exception as e:
                logger.warning("model_cleanup_error", error=str(e), error_type=type(e).__name__)
            finally:
                del self.model
                self.model = None
                self.is_ready = False
