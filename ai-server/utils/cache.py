"""Generic caching utilities for AI services"""

import hashlib
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from utils.logging import get_logger


logger = get_logger(__name__)

T = TypeVar('T')


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL expiration.

    Useful for caching expensive computation results like model inference.
    """

    def __init__(
        self,
        max_size: int = 64,
        ttl_seconds: float = 60.0,
        name: str = "cache"
    ):
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._name = name

    def get(self, key: str) -> Optional[T]:
        """Get cached value if available and not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: T) -> None:
        """Cache a value."""
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get cached value or compute and cache it.

        Note: compute_fn is called outside the lock to avoid blocking.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute outside lock
        value = compute_fn()
        self.put(key, value)
        return value

    async def get_or_compute_async(
        self,
        key: str,
        compute_fn: Callable[[], Any]
    ) -> T:
        """Async version of get_or_compute."""
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute outside lock
        value = await compute_fn()
        self.put(key, value)
        return value

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            logger.debug("cache_cleared", cache_name=self._name)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "name": self._name,
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0
            }


class ContentHashCache(LRUCache[T]):
    """LRU cache that uses content hash as key.

    Useful for caching results based on input data like images or audio.
    """

    def __init__(
        self,
        max_size: int = 32,
        ttl_seconds: float = 60.0,
        name: str = "content_cache",
        hash_prefix_length: int = 16
    ):
        super().__init__(max_size, ttl_seconds, name)
        self._hash_prefix_length = hash_prefix_length

    def _compute_hash(self, data: bytes) -> str:
        """Compute truncated SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()[:self._hash_prefix_length]

    def get_by_content(self, data: bytes) -> Optional[T]:
        """Get cached value by content hash."""
        key = self._compute_hash(data)
        return self.get(key)

    def put_by_content(self, data: bytes, value: T) -> None:
        """Cache value by content hash."""
        key = self._compute_hash(data)
        self.put(key, value)

    async def get_or_compute_by_content(
        self,
        data: bytes,
        compute_fn: Callable[[], Any]
    ) -> T:
        """Get cached value by content or compute and cache it."""
        key = self._compute_hash(data)
        return await self.get_or_compute_async(key, compute_fn)
