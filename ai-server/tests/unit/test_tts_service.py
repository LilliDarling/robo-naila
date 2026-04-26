"""Unit tests for TTS service"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from services.tts import TTSService, AudioData, TTSPhraseLRUCache
from utils.text_normalizer import TextNormalizer


@pytest.fixture
def mock_model():
    """Mock Kokoro TTS model.

    Kokoro exposes ``model.create(text, voice, speed, lang)`` which returns
    a ``(samples, sample_rate)`` tuple of float32 PCM in [-1, 1].
    """
    model = Mock()
    model.create = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))
    model.get_voices = Mock(return_value=["bf_emma", "af_heart", "am_adam"])
    return model


@pytest.fixture
def tts_service():
    return TTSService()


@pytest.fixture
def text_normalizer():
    return TextNormalizer()


class TestTTSService:
    """Tests for TTSService"""

    @pytest.mark.asyncio
    async def test_service_initialization(self, tts_service):
        """Test service initializes correctly"""
        assert tts_service.model is None
        assert not tts_service.is_ready
        assert isinstance(tts_service.text_normalizer, TextNormalizer)
        assert isinstance(tts_service._phrase_cache, TTSPhraseLRUCache)

    def test_get_model_type(self, tts_service):
        """Test model type identification"""
        assert tts_service._get_model_type() == "TTS"

    @pytest.mark.asyncio
    async def test_empty_text_synthesis(self, tts_service):
        """Test synthesis with empty text"""
        tts_service.is_ready = True
        tts_service.model = Mock()

        result = await tts_service.synthesize("")
        assert result.audio_bytes == b""
        assert result.duration_ms == 0
        assert result.text == ""

    @pytest.mark.asyncio
    @patch('services.tts.tts_config')
    async def test_text_truncation(self, mock_config, tts_service, mock_model):
        """Test text is truncated if too long"""
        mock_config.MAX_TEXT_LENGTH = 10
        mock_config.SAMPLE_RATE = 24000
        mock_config.NORMALIZE_NUMBERS = False
        mock_config.NORMALIZE_DATES = False
        mock_config.LOG_SYNTHESES = False
        mock_config.LOG_PERFORMANCE_METRICS = False
        mock_config.WARNING_RTF_THRESHOLD = 1.0
        mock_config.CACHE_INCLUDES_PARAMETERS = True
        mock_config.VOICE = "bf_emma"
        mock_config.SPEED = 1.0

        tts_service.is_ready = True
        tts_service.model = mock_model

        long_text = "a" * 100
        with patch.object(tts_service, '_synthesize_to_audio', new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = (np.zeros(100, dtype=np.float32), 24000)
            with patch.object(tts_service, '_encode_audio', return_value=b'audio'):
                await tts_service.synthesize(long_text)
                # Text passed to synthesis should be truncated (plus the period appended for prosody)
                assert len(mock_synth.call_args[0][0]) <= mock_config.MAX_TEXT_LENGTH + 1

    @pytest.mark.asyncio
    async def test_phrase_caching(self, tts_service, mock_model):
        """Test cached phrases skip synthesis"""
        tts_service.is_ready = True
        tts_service.model = mock_model

        phrase = "Hello"
        audio_samples = np.zeros(1000, dtype=np.float32)
        normalized = tts_service._preprocess_text(phrase)
        cache_key = tts_service._build_cache_key(normalized)
        tts_service._phrase_cache[cache_key] = audio_samples

        with patch.object(tts_service, '_synthesize_to_audio') as mock_synth:
            with patch.object(tts_service, '_encode_audio', return_value=b'audio'):
                await tts_service.synthesize(phrase)
                mock_synth.assert_not_called()

    def test_clear_cache(self, tts_service):
        """Test cache clearing"""
        tts_service._phrase_cache["test"] = np.zeros(100)
        tts_service.clear_cache()
        assert len(tts_service._phrase_cache) == 0

    def test_get_status(self, tts_service):
        """Test status reporting"""
        status = tts_service.get_status()
        assert "ready" in status
        assert "model_path" in status
        assert "voice" in status
        assert "sample_rate" in status
        assert "speed" in status
        assert "cached_phrases" in status

    def test_get_available_voices_returns_empty_when_no_model(self, tts_service):
        """Without a loaded model, voice list is empty rather than crashing"""
        assert tts_service.get_available_voices() == []

    def test_get_available_voices_delegates_to_model(self, tts_service, mock_model):
        """With a model loaded, voice list comes from kokoro"""
        tts_service.model = mock_model
        voices = tts_service.get_available_voices()
        assert "bf_emma" in voices
        assert "af_heart" in voices


class TestTextNormalizer:
    """Tests for TextNormalizer"""

    def test_empty_text(self, text_normalizer):
        result = text_normalizer.normalize("")
        assert result == ""

    def test_abbreviation_expansion(self, text_normalizer):
        text = "Dr. Smith went to St. Louis."
        result = text_normalizer.normalize(text)
        assert "Doctor" in result or "Dr" in result
        assert "Saint" in result or "St" in result

    def test_url_removal(self, text_normalizer):
        text = "Visit https://example.com for more info"
        result = text_normalizer.normalize(text)
        assert "https://example.com" not in result
        assert "link" in result

    def test_email_handling(self, text_normalizer):
        text = "Contact test@example.com"
        result = text_normalizer.normalize(text)
        assert "test@example.com" not in result
        assert "email address" in result

    def test_currency_conversion(self, text_normalizer):
        text = "The cost is $50"
        result = text_normalizer.normalize(text)
        assert "$" not in result
        assert "dollar" in result.lower()

    def test_number_to_words(self, text_normalizer):
        text = "I have 5 apples"
        result = text_normalizer.normalize(text)
        assert "5" not in result
        assert "five" in result.lower()

    def test_whitespace_normalization(self, text_normalizer):
        text = "Hello    world  \n  test"
        result = text_normalizer.normalize(text)
        assert "    " not in result
        assert "\n" not in result

    def test_special_char_cleaning(self, text_normalizer):
        text = "Price: $100 & shipping"
        result = text_normalizer.normalize(text)
        assert "&" not in result
        assert "and" in result


class TestEncodeAudio:
    """Tests for the inline _encode_audio helper that replaced AudioEncoder."""

    def test_encode_raw_returns_int16_bytes(self, tts_service):
        """raw format returns 2 bytes per sample (int16 little-endian)"""
        samples = np.zeros(1000, dtype=np.float32)
        raw = tts_service._encode_audio(samples, "raw")
        assert isinstance(raw, bytes)
        assert len(raw) == 2000

    def test_encode_wav_has_riff_header(self, tts_service):
        """wav format produces a standard RIFF/WAVE header"""
        samples = np.zeros(1000, dtype=np.float32)
        wav = tts_service._encode_audio(samples, "wav")
        assert isinstance(wav, bytes)
        assert wav[:4] == b'RIFF'
        assert wav[8:12] == b'WAVE'

    def test_encode_clips_out_of_range_floats(self, tts_service):
        """Values outside [-1, 1] are clipped, not wrapped"""
        samples = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        raw = tts_service._encode_audio(samples, "raw")
        decoded = np.frombuffer(raw, dtype=np.int16)
        assert decoded[0] == 32767
        assert decoded[1] == -32768
        assert decoded[2] == 0


class TestAudioData:
    """Tests for AudioData dataclass"""

    def test_audio_data_creation(self):
        data = AudioData(
            audio_bytes=b'test',
            sample_rate=24000,
            format="wav",
            duration_ms=1000,
            synthesis_time_ms=100,
            text="Hello"
        )

        assert data.audio_bytes == b'test'
        assert data.sample_rate == 24000
        assert data.format == "wav"
        assert data.duration_ms == 1000
        assert data.synthesis_time_ms == 100
        assert data.text == "Hello"
        assert data.voice == "default"
        assert data.phonemes is None

    def test_audio_data_with_voice(self):
        data = AudioData(
            audio_bytes=b'test',
            sample_rate=24000,
            format="wav",
            duration_ms=1000,
            synthesis_time_ms=100,
            text="Hello",
            voice="bf_emma",
        )
        assert data.voice == "bf_emma"
