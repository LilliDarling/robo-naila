"""Tests for TTS performance optimizations.

These tests verify the specific optimization behaviors of the Kokoro-based
TTSService — LRU phrase caching, cache-key construction, and parallel
warmup synthesis.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
import numpy as np

from services.tts import TTSService, TTSPhraseLRUCache


class TestLRUCacheEviction:
    """Test LRU cache evicts least recently used items correctly"""

    def test_evicts_oldest_when_full(self):
        """Verify oldest item evicted when cache exceeds maxsize"""
        cache = TTSPhraseLRUCache(maxsize=3)
        cache["first"] = 1
        cache["second"] = 2
        cache["third"] = 3

        cache["fourth"] = 4

        assert "first" not in cache
        assert len(cache) == 3
        assert list(cache.keys()) == ["second", "third", "fourth"]

    def test_access_refreshes_lru_order(self):
        """Verify accessing item moves it to most recent"""
        cache = TTSPhraseLRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        _ = cache["a"]

        cache["d"] = 4

        assert "a" in cache
        assert "b" not in cache
        assert list(cache.keys()) == ["c", "a", "d"]

    def test_update_refreshes_lru_order(self):
        """Verify updating existing key moves it to most recent"""
        cache = TTSPhraseLRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        cache["a"] = 99

        cache["d"] = 4

        assert cache["a"] == 99
        assert "b" not in cache


class TestCacheKeyDifferentiation:
    """Test cache keys properly differentiate between synthesis parameters"""

    @pytest.fixture
    def tts_service(self):
        return TTSService()

    def test_different_voices_different_keys(self, tts_service):
        """Same text with different voices produces different cache keys"""
        text = "hello"

        key_emma = tts_service._build_cache_key(text, voice="bf_emma")
        key_heart = tts_service._build_cache_key(text, voice="af_heart")

        assert key_emma != key_heart

    def test_different_speeds_different_keys(self, tts_service):
        """Speed parameter affects cache key"""
        text = "hello"

        key_normal = tts_service._build_cache_key(text, speed=1.0)
        key_fast = tts_service._build_cache_key(text, speed=1.3)

        assert key_normal != key_fast

    def test_default_params_match_explicit_config_values(self, tts_service):
        """Cache keys built without kwargs use config defaults — important for
        warmup hits, since _warmup_and_cache calls _build_cache_key(normalized)
        with no kwargs and synthesize() may also pass voice=None/speed=None.
        """
        from config import tts as tts_config

        key_implicit = tts_service._build_cache_key("hello")
        key_explicit = tts_service._build_cache_key(
            "hello",
            voice=tts_config.VOICE,
            speed=tts_config.SPEED,
        )

        assert key_implicit == key_explicit

    @patch('services.tts.tts_config')
    def test_parameter_caching_disabled(self, mock_config, tts_service):
        """When CACHE_INCLUDES_PARAMETERS=False, only text drives the key"""
        mock_config.CACHE_INCLUDES_PARAMETERS = False

        key1 = tts_service._build_cache_key("Hello", voice="bf_emma")
        key2 = tts_service._build_cache_key("Hello", voice="af_heart")

        assert key1 == "hello"
        assert key2 == "hello"


class TestParallelWarmupBehavior:
    """Test warmup parallelization reduces startup time"""

    @pytest.mark.asyncio
    @patch('services.tts.tts_config')
    async def test_warmup_synthesizes_concurrently(self, mock_config):
        """Warmup synthesizes phrases concurrently, not serially.

        Uses an event-driven concurrency probe rather than wall-clock timing,
        which is flaky on slow/loaded CI machines.
        """
        mock_config.COMMON_PHRASES = ["Hello", "Goodbye", "Thank you"]
        mock_config.CACHE_INCLUDES_PARAMETERS = True
        mock_config.VOICE = "bf_emma"
        mock_config.SPEED = 1.0
        mock_config.NORMALIZE_NUMBERS = False
        mock_config.NORMALIZE_DATES = False

        tts_service = TTSService()
        tts_service.model = Mock()

        tasks_started = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def track_synthesis(*args, **kwargs):
            nonlocal tasks_started, max_concurrent

            async with lock:
                tasks_started += 1
                if tasks_started > max_concurrent:
                    max_concurrent = tasks_started

            # Hold tasks "in flight" so concurrent ones overlap
            await asyncio.sleep(0.05)

            async with lock:
                tasks_started -= 1

            # _warmup_and_cache unpacks (samples, sample_rate)
            return (np.zeros(100, dtype=np.float32), 24000)

        with patch.object(tts_service, '_synthesize_to_audio', side_effect=track_synthesis):
            await tts_service._warmup_and_cache()

        assert max_concurrent == 3, (
            f"Expected 3 concurrent tasks, but max was {max_concurrent} — "
            "warmup is not running synthesis in parallel"
        )

    @pytest.mark.asyncio
    @patch('services.tts.tts_config')
    async def test_warmup_populates_phrase_cache(self, mock_config):
        """Successful warmup synthesis ends up in the phrase cache so that
        subsequent synthesize() calls for the same text hit the cache.
        """
        mock_config.COMMON_PHRASES = ["Hello", "Goodbye"]
        mock_config.CACHE_INCLUDES_PARAMETERS = True
        mock_config.VOICE = "bf_emma"
        mock_config.SPEED = 1.0
        mock_config.NORMALIZE_NUMBERS = False
        mock_config.NORMALIZE_DATES = False
        mock_config.MAX_CACHED_PHRASES = 32

        tts_service = TTSService()
        tts_service.model = Mock()
        # Re-create the cache against the patched config
        tts_service._phrase_cache = TTSPhraseLRUCache(
            maxsize=mock_config.MAX_CACHED_PHRASES
        )

        async def fake_synth(*args, **kwargs):
            return (np.zeros(100, dtype=np.float32), 24000)

        with patch.object(tts_service, '_synthesize_to_audio', side_effect=fake_synth):
            await tts_service._warmup_and_cache()

        assert len(tts_service._phrase_cache) == 2
