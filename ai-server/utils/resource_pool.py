"""Resource pool for managing concurrent access to AI models"""

import asyncio
from typing import Dict

from utils import get_logger


logger = get_logger(__name__)


class ResourcePool:
    """Manages concurrent access to resources with semaphore-based pooling"""

    def __init__(self, max_concurrent: int, timeout: float):
        """Initialize resource pool

        Args:
            max_concurrent: Maximum number of concurrent operations
            timeout: Timeout in seconds when waiting for a pool slot
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_requests = 0
        self._total_requests = 0
        self._pool_waits = 0

    async def __aenter__(self):
        """Acquire a pool slot with timeout (context manager)"""
        self._total_requests += 1

        if self._semaphore.locked():
            self._pool_waits += 1
            logger.debug("pool_slot_wait", active_requests=self._active_requests, max_concurrent=self.max_concurrent)

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.timeout
            )
            self._active_requests += 1
            return self
        except asyncio.TimeoutError as e:
            logger.error("pool_timeout", timeout_seconds=self.timeout, max_concurrent=self.max_concurrent)
            raise RuntimeError(
                f"Resource pool timeout: max {self.max_concurrent} concurrent requests exceeded"
            ) from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release pool slot (context manager)"""
        if self._active_requests > 0:
            self._active_requests -= 1
            self._semaphore.release()
        return False

    def get_stats(self) -> Dict:
        """Get pool statistics

        Returns:
            Dictionary with pool metrics
        """
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self._active_requests,
            "total_requests": self._total_requests,
            "pool_waits": self._pool_waits,
            "available_slots": self.max_concurrent - self._active_requests,
        }
