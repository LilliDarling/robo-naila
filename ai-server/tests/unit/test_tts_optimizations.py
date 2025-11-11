"""Tests for TTS performance optimizations

These tests verify the specific optimization behaviors, not just functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from services.tts import TTSService, LRUCache


class TestLRUCacheEviction:
    """Test LRU cache evicts least recently used items correctly"""

    def test_evicts_oldest_when_full(self):
        """Verify oldest item evicted when cache exceeds maxsize"""
        cache = LRUCache(maxsize=3)
        cache["first"] = 1
        cache["second"] = 2
        cache["third"] = 3

        # Add 4th item
        cache["fourth"] = 4

        # Oldest (first) should be evicted
        assert "first" not in cache
        assert len(cache) == 3
        assert list(cache.keys()) == ["second", "third", "fourth"]

    def test_access_refreshes_lru_order(self):
        """Verify accessing item moves it to most recent"""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Access "a" to refresh it
        _ = cache["a"]

        # Add new item - should evict "b" (now oldest)
        cache["d"] = 4

        assert "a" in cache
        assert "b" not in cache
        assert list(cache.keys()) == ["c", "a", "d"]

    def test_update_refreshes_lru_order(self):
        """Verify updating existing key moves it to most recent"""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Update "a"
        cache["a"] = 99

        # Add new item - should evict "b"
        cache["d"] = 4

        assert cache["a"] == 99
        assert "b" not in cache


class TestCacheKeyDifferentiation:
    """Test cache keys properly differentiate between synthesis parameters"""

    @pytest.fixture
    def tts_service(self):
        return TTSService()

    def test_different_emotions_different_keys(self, tts_service):
        """Verify same text with different emotions creates different cache keys"""
        text = "Hello"

        key_neutral = tts_service._build_cache_key(text, emotion=None)
        key_happy = tts_service._build_cache_key(text, emotion="happy")
        key_sad = tts_service._build_cache_key(text, emotion="sad")

        assert key_neutral != key_happy
        assert key_happy != key_sad
        assert key_neutral != key_sad

    def test_different_voices_different_keys(self, tts_service):
        """Verify same text with different voices creates different cache keys"""
        text = "Hello"

        key_voice1 = tts_service._build_cache_key(text, voice="lessac")
        key_voice2 = tts_service._build_cache_key(text, voice="amy")

        assert key_voice1 != key_voice2

    def test_different_prosody_different_keys(self, tts_service):
        """Verify prosody parameters affect cache key"""
        text = "Hello"

        key_normal = tts_service._build_cache_key(text, length_scale=1.0)
        key_fast = tts_service._build_cache_key(text, length_scale=0.8)

        assert key_normal != key_fast

    @patch('services.tts.tts_config')
    def test_parameter_caching_disabled(self, mock_config, tts_service):
        """Verify simple caching when CACHE_INCLUDES_PARAMETERS=False"""
        mock_config.CACHE_INCLUDES_PARAMETERS = False

        # With simple caching, all variations should produce same key
        key1 = tts_service._build_cache_key("Hello", emotion="happy")
        key2 = tts_service._build_cache_key("Hello", emotion="sad")

        # Both should just be lowercase text
        assert key1 == "hello"
        assert key2 == "hello"


class TestNormalizationCacheReuse:
    """Test normalization caching reduces redundant regex operations"""

    @pytest.fixture
    def tts_service(self):
        return TTSService()

    def test_normalization_cached_on_repeat(self, tts_service):
        """Verify repeated normalization uses cache"""
        text = "I have 100 dollars and 50 cents"

        # First call
        result1 = tts_service._normalize_text_cached(text)
        cache_info_1 = tts_service._normalize_text_cached.cache_info()

        # Second call - should hit cache
        result2 = tts_service._normalize_text_cached(text)
        cache_info_2 = tts_service._normalize_text_cached.cache_info()

        # Results identical
        assert result1 == result2
        # Cache hit count increased
        assert cache_info_2.hits > cache_info_1.hits

    def test_cache_cleared_with_service_cache(self, tts_service):
        """Verify clear_cache() clears normalization cache"""
        text = "Test 123"

        # Populate normalization cache
        tts_service._normalize_text_cached(text)
        assert tts_service._normalize_text_cached.cache_info().currsize > 0

        # Clear all caches
        tts_service.clear_cache()

        # Normalization cache should be empty
        assert tts_service._normalize_text_cached.cache_info().currsize == 0


class TestParallelWarmupBehavior:
    """Test warmup parallelization reduces startup time"""

    @pytest.mark.asyncio
    @patch('services.tts.tts_config')
    async def test_warmup_synthesizes_concurrently(self, mock_config):
        """Verify warmup synthesizes multiple phrases at once, not serially"""
        mock_config.COMMON_PHRASES = ["Hello", "Goodbye", "Thank you"]
        mock_config.CACHE_INCLUDES_PARAMETERS = True
        mock_config.VOICE = "lessac"
        mock_config.LENGTH_SCALE = 1.0
        mock_config.NOISE_SCALE = 0.5
        mock_config.NOISE_W = 0.6

        tts_service = TTSService()
        tts_service.model = Mock()

        synthesis_times = []

        async def track_synthesis(*args, **kwargs):
            """Track when synthesis happens"""
            synthesis_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)  # Simulate synthesis time
            return np.random.randn(100).astype('float32')

        with patch.object(tts_service, '_synthesize_to_audio', side_effect=track_synthesis):
            await tts_service._warmup_and_cache()

        # All 3 phrases should be synthesized
        assert len(synthesis_times) == 3

        # If parallel, all start times should be very close (< 10ms apart)
        time_diffs = [synthesis_times[i+1] - synthesis_times[i] for i in range(len(synthesis_times)-1)]
        assert all(diff < 0.01 for diff in time_diffs), \
            "Synthesis not parallel - time gaps too large"



class TestSSMLBatchOptimization:
    """Test SSML synthesis parallelizes segments with same voice"""

    @pytest.fixture
    def tts_service(self):
        service = TTSService()
        service.model = Mock()
        service.is_ready = True
        return service

    def test_segments_grouped_single_voice_mode(self, tts_service):
        """Verify single-voice mode groups all segments together"""
        from utils.ssml_parser import SSMLSegment

        # In single voice mode (multi_voice_enabled=False)
        tts_service.multi_voice_enabled = False

        segments = [
            SSMLSegment(text="Hello", voice="voice1"),
            SSMLSegment(text="World", voice="voice1"),
            SSMLSegment(text="Goodbye", voice="voice2"),
            SSMLSegment(text="Friend", voice="voice2"),
        ]

        groups = tts_service._group_segments_by_voice(segments)

        # Single voice mode: all segments in one group
        assert len(groups) == 1
        assert groups[0][0] is None  # No voice specified
        assert len(groups[0][1]) == 4  # All segments

    @pytest.mark.asyncio
    @patch('services.tts.tts_config')
    async def test_ssml_plain_text_concatenation(self, mock_config, tts_service):
        """Verify SSML with plain text (no tags) concatenates properly"""
        mock_config.SAMPLE_RATE = 22050
        mock_config.OUTPUT_FORMAT = "wav"
        mock_config.WARNING_RTF_THRESHOLD = 1.0  # Set to avoid comparison issues

        # Simple SSML - parser treats this as single segment
        ssml = "<speak>This is a single segment of text.</speak>"

        synthesis_count = 0

        async def track_synthesis(*args, **kwargs):
            nonlocal synthesis_count
            synthesis_count += 1
            return np.random.randn(1000).astype('float32')

        with patch.object(tts_service, '_synthesize_to_audio', side_effect=track_synthesis):
            with patch.object(tts_service, '_encode_audio', return_value=b'audio'):
                result = await tts_service._synthesize_ssml(ssml)

        # Should synthesize once (single segment)
        assert synthesis_count == 1
        assert result.audio_bytes == b'audio'


class TestAudioEncoderConversion:
    """Test audio encoder handles pre-conversion correctly"""

    def test_int16_passthrough(self):
        """Verify _to_int16 returns int16 arrays unchanged"""
        from utils.audio_encoder import AudioEncoder

        # Already int16
        audio_int16 = np.array([100, 200, 300], dtype=np.int16)
        result = AudioEncoder._to_int16(audio_int16)

        # Should return same array (no conversion)
        assert result.dtype == np.int16
        assert np.array_equal(result, audio_int16)

    def test_float32_conversion(self):
        """Verify _to_int16 converts float32 to int16"""
        from utils.audio_encoder import AudioEncoder

        # Float32 audio
        audio_float32 = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        result = AudioEncoder._to_int16(audio_float32)

        # Should be int16
        assert result.dtype == np.int16
        # Should be scaled properly
        assert result[0] == int(0.5 * 32767)
        assert result[2] == 32767
