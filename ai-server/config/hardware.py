"""Hardware detection and optimization configuration"""

import multiprocessing
import os
import platform
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from utils import get_logger

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import psutil
except ImportError:
    psutil = None


logger = get_logger(__name__)


@dataclass
class HardwareInfo:
    """Hardware information and capabilities"""
    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    optimization_level: str = "standard"  # "minimal", "standard", "aggressive"


class HardwareOptimizer:
    """Hardware detection and optimization for AI models"""
    
    def __init__(self):
        self.hardware_info = self._detect_hardware()
        self.config = self._get_optimization_config()
    
    def _detect_hardware(self) -> HardwareInfo:
        """Comprehensive hardware detection"""
        try:
            # CUDA detection
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)

                # Get GPU memory
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Get compute capability
                compute_cap = torch.cuda.get_device_properties(0).major
                compute_capability = (
                    f"{compute_cap}.{torch.cuda.get_device_properties(0).minor}"
                )

                logger.info(
                    "cuda_gpu_detected",
                    device_name=device_name,
                    memory_gb=round(memory_gb, 1),
                    compute_capability=compute_capability
                )

                return HardwareInfo(
                    device_type="cuda",
                    device_name=device_name,
                    memory_gb=memory_gb,
                    compute_capability=compute_capability,
                    optimization_level=self._get_gpu_optimization_level(memory_gb)
                )

            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("mps_detected", device="Apple Silicon GPU")

                return HardwareInfo(
                    device_type="mps",
                    device_name="Apple Silicon GPU",
                    optimization_level="standard"
                )

            else:
                return self._detect_cpu_hardware()
        except Exception as e:
            logger.warning("hardware_detection_failed", error=str(e), error_type=type(e).__name__, fallback="minimal CPU config")
            return HardwareInfo(
                device_type="cpu",
                device_name="Unknown CPU",
                optimization_level="minimal"
            )

    def _detect_cpu_hardware(self):
        cpu_count = multiprocessing.cpu_count()
        if psutil:
            memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            logger.warning("psutil_unavailable", reason="memory detection unavailable")
            memory_gb = None

        # Detect CPU type
        cpu_info = self._get_cpu_info()

        logger.info(
            "cpu_detected",
            cpu_info=cpu_info,
            cpu_count=cpu_count,
            memory_gb=round(memory_gb, 1) if memory_gb else None
        )

        return HardwareInfo(
            device_type="cpu",
            device_name=cpu_info,
            memory_gb=memory_gb,
            optimization_level=self._get_cpu_optimization_level(cpu_count, memory_gb)
        )
    
    def _get_cpu_info(self) -> str:
        """Get CPU information"""
        try:
            if cpuinfo:
                return f"{cpuinfo.get_cpu_info()['brand_raw']} ({platform.machine()})"
            else:
                return f"CPU {platform.machine()} ({os.cpu_count() or 'unknown'} cores)"
        except Exception:
            return f"CPU ({os.cpu_count() or 'unknown'} cores)"
    
    def _get_gpu_optimization_level(self, memory_gb: float) -> str:
        """Determine GPU optimization level based on memory"""
        if memory_gb >= 8:
            return "aggressive"  # High-end GPU
        elif memory_gb >= 4:
            return "standard"    # Mid-range GPU
        else:
            return "minimal"     # Low-end GPU
    
    def _get_cpu_optimization_level(self, cpu_count: int, memory_gb: Optional[float]) -> str:
        """Determine CPU optimization level"""
        if memory_gb is None:
            return "minimal"

        if cpu_count >= 8 and memory_gb >= 16:
            return "aggressive"  # High-end CPU setup
        elif cpu_count >= 4 and memory_gb >= 8:
            return "standard"    # Standard CPU setup
        else:
            return "minimal"     # Low-end CPU setup
    
    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration based on hardware"""
        base_config = {
            "device": self.hardware_info.device_type,
            "batch_size": 1,  # Always 1 for real-time processing
            "precision": "float32",
            "compile_model": False,
            "use_cache": True,
            "thread_count": None
        }
        
        if self.hardware_info.device_type == "cuda":
            return {**base_config, **self._get_cuda_config()}
        elif self.hardware_info.device_type == "mps":
            return {**base_config, **self._get_mps_config()}
        else:
            return {**base_config, **self._get_cpu_config()}
    
    def _get_cuda_config(self) -> Dict[str, Any]:
        """CUDA-specific optimizations"""
        config = {
            "precision": "float16" if self.hardware_info.memory_gb and self.hardware_info.memory_gb > 4 else "float32",
            "compile_model": self.hardware_info.optimization_level == "aggressive",
        }
        
        # Memory optimization
        if self.hardware_info.memory_gb and self.hardware_info.memory_gb < 4:
            config["low_memory_mode"] = True
        
        return config
    
    def _get_mps_config(self) -> Dict[str, Any]:
        """Apple MPS-specific optimizations"""
        return {
            "precision": "float32",  # MPS works best with float32
            "compile_model": False,   # MPS compilation can be unstable
        }
    
    def _get_cpu_config(self) -> Dict[str, Any]:
        """CPU-specific optimizations"""
        config = {
            "precision": "float32",
            "thread_count": min(4, multiprocessing.cpu_count()),  # Limit threads for stability
        }
        
        # Intel CPU optimizations
        if "Intel" in self.hardware_info.device_name:
            config["use_intel_extension"] = True
        
        return config
    
    def get_model_config(self, model_type: str = "sentence_transformer") -> Dict[str, Any]:
        """Get model-specific configuration"""
        base = self.config.copy()
        
        if model_type == "sentence_transformer" and self.hardware_info.device_type == "cpu":
            base["convert_to_numpy"] = True

        return base
    
    def log_hardware_info(self):
        """Log detailed hardware information"""
        info = self.hardware_info
        logger.info(
            "hardware_configuration",
            device_type=info.device_type,
            device_name=info.device_name,
            memory_gb=round(info.memory_gb, 1) if info.memory_gb else None,
            optimization_level=info.optimization_level,
            config=self.config
        )


# Global hardware optimizer instance
hardware_optimizer = HardwareOptimizer()