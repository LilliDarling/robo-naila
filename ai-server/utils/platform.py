import platform
import signal
import sys
import asyncio
from typing import Optional, Callable

from utils import get_logger


logger = get_logger(__name__)


class CrossPlatformSignalHandler:
    """Cross-platform signal handling for graceful shutdown"""
    
    def __init__(self, shutdown_callback: Callable):
        self.shutdown_callback = shutdown_callback
        self._shutdown_initiated = False
        
    def setup_signals(self):
        """Setup signal handlers with platform-specific support"""
        try:
            if sys.platform == 'win32':
                # Windows support
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                logger.debug("signal_handlers_configured", platform="windows", signals="SIGINT, SIGTERM")
            else:
                # Unix-like systems (Linux, macOS)
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                if hasattr(signal, 'SIGHUP'):
                    signal.signal(signal.SIGHUP, self._signal_handler)
                logger.debug("signal_handlers_configured", platform="unix", signals="SIGINT, SIGTERM, SIGHUP")

        except Exception as e:
            logger.warning("signal_handler_setup_failed", error=str(e), error_type=type(e).__name__)
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals across platforms"""
        if self._shutdown_initiated:
            logger.warning("Force shutdown signal received")
            sys.exit(1)
        
        signal_names = {
            signal.SIGINT: "SIGINT",
            signal.SIGTERM: "SIGTERM",
        }
        if hasattr(signal, 'SIGHUP'):
            signal_names[signal.SIGHUP] = "SIGHUP"
        
        signal_name = signal_names.get(signum, f"Signal {signum}")
        logger.info("shutdown_signal_received", signal=signal_name, action="graceful_shutdown")

        self._shutdown_initiated = True
        
        # Schedule shutdown in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shutdown_callback())
        except RuntimeError:
            # No running loop, exit immediately
            logger.info("No event loop available, exiting immediately")
            sys.exit(0)


def get_platform_info() -> dict:
    """Get cross-platform system information"""
    return {
        "system": platform.system(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
    }


def check_python_version(min_version: tuple = (3, 8)) -> bool:
    """Check if Python version meets minimum requirements"""
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        min_version_str = ".".join(map(str, min_version))
        current_version_str = ".".join(map(str, current_version))
        logger.error("python_version_mismatch", required_version=min_version_str, current_version=current_version_str)
        return False
    
    return True