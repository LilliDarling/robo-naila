"""Unit tests for STT Service"""

import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open

import pytest

from services.stt import STTService, TranscriptionResult
from config import stt as stt_config


class TestSTTServiceInit:
    """Test STT Service initialization"""

    def test_init_sets_defaults(self):
        """Test that __init__ sets correct default values"""
        service = STTService()

        assert service.model is None
        assert service.is_ready is False
        assert service.hardware_info is None
        assert isinstance(service.model_path, Path)


class TestLoadModel:
    """Test model loading functionality"""

    @pytest.fixture
    def service(self):
        return STTService()

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, service):
        """Test that load_model returns True if already loaded"""
        service.is_ready = True

        result = await service.load_model()

        assert result is True

    @pytest.mark.asyncio
    async def test_load_model_file_not_found(self, service):
        """Test that load_model returns False when model file missing"""
        with patch('pathlib.Path.exists', return_value=False):
            result = await service.load_model()

        assert result is False
        assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_success(self, service):
        """Test successful model loading with proper configuration"""
        mock_whisper_model = MagicMock()
        hardware_info = {
            'device_type': 'cuda',
            'device_name': 'Test GPU',
            'acceleration': 'cuda',
            'cpu_count': 8,
            'vram_gb': 8.0
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('faster_whisper.WhisperModel', return_value=mock_whisper_model) as mock_whisper_class:

            result = await service.load_model(hardware_info=hardware_info)

            assert result is True
            assert service.is_ready is True
            assert service.model == mock_whisper_model
            assert service.hardware_info is not None
            assert service.hardware_info == hardware_info
            mock_whisper_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_success_without_hardware_info(self, service):
        """Test successful model loading with automatic hardware detection"""
        mock_whisper_model = MagicMock()
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'
        mock_hw_optimizer.hardware_info.device_name = 'Test GPU'
        mock_hw_optimizer.hardware_info.memory_gb = 8.0

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('faster_whisper.WhisperModel', return_value=mock_whisper_model) as mock_whisper_class:

            result = await service.load_model()

            assert result is True
            assert service.is_ready is True
            assert service.model == mock_whisper_model
            assert service.hardware_info is not None
            mock_whisper_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_import_error(self, service):
        """Test that load_model handles missing faster-whisper gracefully"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer'):

            # Mock the import to raise ImportError
            with patch.dict('sys.modules', {'faster_whisper': None}):
                with patch('builtins.__import__', side_effect=ImportError("No module")):
                    result = await service.load_model()

                    assert result is False
                    assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_memory_error(self, service):
        """Test that load_model handles out of memory errors"""
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cpu'
        mock_hw_optimizer.hardware_info.device_name = 'CPU'
        mock_hw_optimizer.hardware_info.memory_gb = 4.0

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('faster_whisper.WhisperModel', side_effect=MemoryError("OOM")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_gpu_incompatibility(self, service):
        """Test that load_model handles GPU incompatibility"""
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('faster_whisper.WhisperModel', side_effect=ValueError("CUDA error")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_with_warmup_enabled(self, service):
        """Test that model warm-up is called when enabled"""
        mock_whisper_model = MagicMock()
        hardware_info = {'device_type': 'cpu', 'cpu_count': 4}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('faster_whisper.WhisperModel', return_value=mock_whisper_model), \
             patch.object(stt_config, 'ENABLE_WARMUP', True), \
             patch.object(service, '_warmup_model', new_callable=AsyncMock) as mock_warmup:

            result = await service.load_model(hardware_info=hardware_info)

            assert result is True
            mock_warmup.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_with_warmup_disabled(self, service):
        """Test that model warm-up is skipped when disabled"""
        mock_whisper_model = MagicMock()
        hardware_info = {'device_type': 'cpu', 'cpu_count': 4}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('faster_whisper.WhisperModel', return_value=mock_whisper_model), \
             patch.object(stt_config, 'ENABLE_WARMUP', False), \
             patch.object(service, '_warmup_model', new_callable=AsyncMock) as mock_warmup:

            result = await service.load_model(hardware_info=hardware_info)

            assert result is True
            mock_warmup.assert_not_called()

    @pytest.mark.asyncio
    async def test_warmup_model_success(self, service):
        """Test successful model warm-up"""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        service.model = mock_model
        service.is_ready = True

        await service._warmup_model()

        # Verify transcribe was called with warm-up audio
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        warmup_audio = call_args[0][0]

        # Check warm-up audio is correct shape
        expected_samples = int(stt_config.SAMPLE_RATE * stt_config.WARMUP_DURATION_MS / 1000.0)
        assert len(warmup_audio) == expected_samples
        assert warmup_audio.dtype == np.float32

    @pytest.mark.asyncio
    async def test_warmup_model_failure_non_critical(self, service):
        """Test that warm-up failure doesn't prevent model from being ready"""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Warm-up error")
        service.model = mock_model
        service.is_ready = True

        # Should not raise, just log warning
        await service._warmup_model()

        # Model should still be ready despite warm-up failure
        assert service.is_ready is True


class TestDeviceSelection:
    """Test device selection logic"""

    @pytest.fixture
    def service(self):
        return STTService()

    def test_get_device_auto_cuda(self, service):
        """Test auto device selection with CUDA available"""
        service.hardware_info = {'acceleration': 'cuda'}

        device = service._get_device()

        assert device == 'cuda'

    def test_get_device_auto_cpu(self, service):
        """Test auto device selection with CPU only"""
        service.hardware_info = {'acceleration': 'cpu'}

        device = service._get_device()

        assert device == 'cpu'

    def test_get_device_explicit(self, service):
        """Test explicit device configuration"""
        with patch.object(stt_config, 'DEVICE', 'cpu'):
            device = service._get_device()

            assert device == 'cpu'


class TestComputeType:
    """Test compute type selection logic"""

    @pytest.fixture
    def service(self):
        return STTService()

    def test_get_compute_type_cpu(self, service):
        """Test compute type for CPU device"""
        service.hardware_info = {'acceleration': 'cpu'}

        with patch.object(service, '_get_device', return_value='cpu'):
            compute_type = service._get_compute_type()

            assert compute_type == 'int8'

    def test_get_compute_type_gpu_high_vram(self, service):
        """Test compute type for GPU with high VRAM"""
        service.hardware_info = {'acceleration': 'cuda', 'vram_gb': 8}

        with patch.object(service, '_get_device', return_value='cuda'):
            compute_type = service._get_compute_type()

            assert compute_type == 'float16'

    def test_get_compute_type_gpu_low_vram(self, service):
        """Test compute type for GPU with low VRAM"""
        service.hardware_info = {'acceleration': 'cuda', 'vram_gb': 2}

        with patch.object(service, '_get_device', return_value='cuda'):
            compute_type = service._get_compute_type()

            assert compute_type == 'int8'


class TestThreadCount:
    """Test thread count calculation"""

    @pytest.fixture
    def service(self):
        return STTService()

    def test_get_thread_count_configured(self, service):
        """Test that configured thread count is used"""
        count = service._get_thread_count(config_threads=8)
        assert count == 8

    def test_get_thread_count_auto_detect(self, service):
        """Test auto-detection of thread count"""
        service.hardware_info = {'cpu_count': 16}
        count = service._get_thread_count(config_threads=0)
        assert count == 12  # 75% of 16

    def test_get_thread_count_default(self, service):
        """Test default thread count fallback"""
        service.hardware_info = None
        count = service._get_thread_count(config_threads=0)
        assert count == 4


class TestAudioValidation:
    """Test audio validation"""

    @pytest.fixture
    def service(self):
        return STTService()

    def test_validate_audio_empty(self, service):
        """Test validation of empty audio data"""
        is_valid, error = service._validate_audio(b'', 'wav')

        assert is_valid is False
        assert 'Empty' in error

    def test_validate_audio_too_small(self, service):
        """Test validation of audio data that's too small"""
        is_valid, error = service._validate_audio(b'x' * 50, 'wav')

        assert is_valid is False
        assert 'too small' in error

    def test_validate_audio_unsupported_format(self, service):
        """Test validation of unsupported audio format"""
        is_valid, error = service._validate_audio(b'x' * 1000, 'xyz')

        assert is_valid is False
        assert 'Unsupported format' in error

    def test_validate_audio_success(self, service):
        """Test successful audio validation"""
        is_valid, error = service._validate_audio(b'x' * 1000, 'wav')

        assert is_valid is True
        assert error == ''


class TestAudioPreprocessing:
    """Test audio preprocessing"""

    @pytest.fixture
    def service(self):
        return STTService()

    @pytest.mark.asyncio
    async def test_preprocess_audio_wav(self, service):
        """Test preprocessing of WAV audio"""
        # Create mock audio file data
        mock_audio_array = np.random.randn(16000).astype('float32')  # 1 second at 16kHz
        mock_sample_rate = 16000

        # Mock soundfile module
        mock_sf = MagicMock()
        mock_sf.read.return_value = (mock_audio_array, mock_sample_rate)

        with patch.dict('sys.modules', {'soundfile': mock_sf}):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            assert isinstance(audio_array, np.ndarray)
            assert sample_rate == 16000
            assert duration_ms == 1000  # 1 second

    @pytest.mark.asyncio
    async def test_preprocess_audio_already_correct_format(self, service):
        """Test that preprocessing skips conversion for already correct format (16kHz mono)"""
        # Create mock audio that's already in correct format
        mock_audio_array = np.random.randn(16000).astype('float32')  # Mono, 16kHz
        mock_sample_rate = 16000

        mock_sf = MagicMock()
        mock_sf.read.return_value = (mock_audio_array, mock_sample_rate)

        # Mock resampy to verify it's NOT called
        mock_resampy = MagicMock()

        with patch.dict('sys.modules', {'soundfile': mock_sf, 'resampy': mock_resampy}):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            # Verify resampling was NOT called (audio already correct)
            assert not mock_resampy.resample.called
            assert sample_rate == 16000
            assert len(audio_array.shape) == 1  # Mono

    @pytest.mark.asyncio
    async def test_preprocess_audio_stereo_to_mono(self, service):
        """Test conversion of stereo audio to mono"""
        # Create mock stereo audio (2 channels)
        mock_stereo_audio = np.random.randn(16000, 2).astype('float32')
        mock_sample_rate = 16000

        mock_sf = MagicMock()
        mock_sf.read.return_value = (mock_stereo_audio, mock_sample_rate)

        with patch.dict('sys.modules', {'soundfile': mock_sf}):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            assert len(audio_array.shape) == 1  # Should be mono now

    @pytest.mark.asyncio
    async def test_preprocess_audio_resampling(self, service):
        """Test resampling of audio to 16kHz"""
        # Create mock audio at 44.1kHz
        mock_audio_array = np.random.randn(44100).astype('float32')
        mock_sample_rate = 44100

        mock_sf = MagicMock()
        mock_sf.read.return_value = (mock_audio_array, mock_sample_rate)
        mock_resampy = MagicMock()
        mock_resampy.resample = MagicMock(return_value=np.random.randn(16000).astype('float32'))

        with patch.dict('sys.modules', {'soundfile': mock_sf, 'resampy': mock_resampy}):

            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            mock_resampy.resample.assert_called_once()
            assert sample_rate == 16000

    @pytest.mark.asyncio
    async def test_preprocess_audio_truncate_long(self, service):
        """Test truncation of audio that's too long"""
        # Create audio longer than MAX_DURATION_MS
        long_duration_samples = int((stt_config.MAX_DURATION_MS / 1000.0) * 16000) + 10000
        mock_audio_array = np.random.randn(long_duration_samples).astype('float32')
        mock_sample_rate = 16000

        mock_sf = MagicMock()
        mock_sf.read.return_value = (mock_audio_array, mock_sample_rate)

        with patch.dict('sys.modules', {'soundfile': mock_sf}):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            assert duration_ms == stt_config.MAX_DURATION_MS


class TestTranscribeAudio:
    """Test audio transcription"""

    @pytest.fixture
    def service(self):
        service = STTService()
        service.is_ready = True
        service.model = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_transcribe_audio_model_not_ready(self):
        """Test that transcription fails gracefully when model not ready"""
        service = STTService()  # is_ready = False by default

        result = await service.transcribe_audio(b'fake_audio', 'wav')

        assert result.text == ''
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_audio(self, service):
        """Test transcription with invalid audio data"""
        result = await service.transcribe_audio(b'', 'wav')

        assert result.text == ''
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, service):
        """Test successful audio transcription"""
        # Mock preprocessing
        mock_audio_array = np.random.randn(16000).astype('float32')

        # Mock transcription result
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.1  # High confidence (close to 0)

        mock_info = MagicMock()
        mock_info.language = 'en'

        service.model.transcribe.return_value = ([mock_segment], mock_info)

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(mock_audio_array, 16000, 1000)):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            assert result.text == 'Hello world'  # Cleaned and stripped
            assert result.language == 'en'
            assert result.confidence > 0.0
            assert result.duration_ms == 1000

    @pytest.mark.asyncio
    async def test_transcribe_audio_empty_result(self, service):
        """Test transcription that produces empty text"""
        mock_audio_array = np.random.randn(16000).astype('float32')
        mock_info = MagicMock()
        mock_info.language = 'en'

        # Return empty list of segments
        service.model.transcribe.return_value = (iter([]), mock_info)

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(mock_audio_array, 16000, 1000)):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            assert result.text == ''
            assert result.confidence == 0.0  # Fixed: empty transcription now returns 0.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_low_confidence_warning(self, service):
        """Test that low confidence transcriptions log a warning but still return text"""
        mock_audio_array = np.random.randn(16000).astype('float32')

        # Mock low confidence segment
        mock_segment = MagicMock()
        mock_segment.text = "Unclear audio"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -2.0  # Very low confidence

        mock_info = MagicMock()
        mock_info.language = 'en'

        service.model.transcribe.return_value = ([mock_segment], mock_info)

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(mock_audio_array, 16000, 1000)), \
             patch.object(stt_config, 'REJECT_LOW_CONFIDENCE', False):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            # Should still return text even with low confidence
            assert result.text == 'Unclear audio'
            assert result.confidence < stt_config.MIN_CONFIDENCE

    @pytest.mark.asyncio
    async def test_transcribe_audio_reject_low_confidence(self, service):
        """Test that low confidence transcriptions are rejected when configured"""
        mock_audio_array = np.random.randn(16000).astype('float32')

        # Mock low confidence segment
        mock_segment = MagicMock()
        mock_segment.text = "Unclear audio"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -2.0  # Very low confidence

        mock_info = MagicMock()
        mock_info.language = 'en'

        service.model.transcribe.return_value = ([mock_segment], mock_info)

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(mock_audio_array, 16000, 1000)), \
             patch.object(stt_config, 'REJECT_LOW_CONFIDENCE', True):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            # Should return empty text due to low confidence rejection
            assert result.text == ''
            assert result.confidence < stt_config.MIN_CONFIDENCE
            assert result.language == 'en'  # Other fields should still be populated


class TestCleanTranscription:
    """Test transcription cleaning"""

    @pytest.fixture
    def service(self):
        return STTService()

    def test_clean_transcription_strip_whitespace(self, service):
        """Test that whitespace is stripped"""
        text = service._clean_transcription("  hello world  ")

        assert text == "Hello world"  # Also capitalized

    def test_clean_transcription_normalize_whitespace(self, service):
        """Test that multiple spaces are normalized"""
        text = service._clean_transcription("hello    world\n\ntest")

        assert text == "Hello world test"

    def test_clean_transcription_capitalize_first(self, service):
        """Test that first letter is capitalized"""
        text = service._clean_transcription("hello world")

        assert text[0].isupper()
        assert text == "Hello world"

    def test_clean_transcription_truncate_long(self, service):
        """Test that overly long text is truncated"""
        long_text = "a" * (stt_config.MAX_TEXT_LENGTH + 100)
        text = service._clean_transcription(long_text)

        assert len(text) == stt_config.MAX_TEXT_LENGTH


class TestTranscribeFile:
    """Test file transcription"""

    @pytest.fixture
    def service(self):
        service = STTService()
        service.is_ready = True
        service.model = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, service):
        """Test transcription of non-existent file"""
        result = await service.transcribe_file('/fake/path.wav')

        assert result.text == ''
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_file_success(self, service):
        """Test successful file transcription"""
        mock_result = TranscriptionResult(
            text="Test transcription",
            language="en",
            confidence=0.95,
            duration_ms=1000,
            transcription_time_ms=300
        )

        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=b'fake_audio')), \
             patch.object(service, 'transcribe_audio', return_value=mock_result) as mock_transcribe:

            result = await service.transcribe_file('/fake/path.wav')

            assert result.text == "Test transcription"
            mock_transcribe.assert_called_once()


class TestTranscribeBatch:
    """Test batch transcription"""

    @pytest.fixture
    def service(self):
        service = STTService()
        service.is_ready = True
        service.model = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_transcribe_batch_empty(self, service):
        """Test batch transcription with empty input"""
        results = await service.transcribe_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_transcribe_batch_model_not_ready(self):
        """Test batch transcription when model not ready"""
        service = STTService()  # is_ready = False
        batch = [(b'audio1', 'wav'), (b'audio2', 'wav')]

        results = await service.transcribe_batch(batch)

        assert len(results) == 2
        assert all(r.text == '' for r in results)
        assert all(r.confidence == 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_transcribe_batch_success(self, service):
        """Test successful batch transcription"""
        # Mock preprocessing
        mock_audio1 = np.random.randn(16000).astype('float32')
        mock_audio2 = np.random.randn(16000).astype('float32')

        # Mock segments
        mock_segment1 = MagicMock()
        mock_segment1.text = "First audio"
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.avg_logprob = -0.1

        mock_segment2 = MagicMock()
        mock_segment2.text = "Second audio"
        mock_segment2.start = 0.0
        mock_segment2.end = 1.0
        mock_segment2.avg_logprob = -0.2

        mock_info = MagicMock()
        mock_info.language = 'en'

        # Mock preprocessing to return different arrays
        async def mock_preprocess_safe(audio_data, fmt):
            if audio_data == b'audio1':
                return mock_audio1, 16000, 1000
            elif audio_data == b'audio2':
                return mock_audio2, 16000, 1000
            return None, 0, 0

        service._preprocess_audio_safe = mock_preprocess_safe

        # Mock transcription to return different results
        async def mock_run_transcription(audio_array, language):
            if np.array_equal(audio_array, mock_audio1):
                return ([mock_segment1], mock_info)
            elif np.array_equal(audio_array, mock_audio2):
                return ([mock_segment2], mock_info)

        service._run_transcription = mock_run_transcription

        batch = [(b'audio1', 'wav'), (b'audio2', 'wav')]
        results = await service.transcribe_batch(batch)

        assert len(results) == 2
        assert results[0].text == 'First audio'
        assert results[1].text == 'Second audio'
        assert all(r.language == 'en' for r in results)
        assert all(r.confidence > 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_transcribe_batch_partial_failure(self, service):
        """Test batch transcription with some items failing"""
        mock_audio = np.random.randn(16000).astype('float32')

        mock_segment = MagicMock()
        mock_segment.text = "Valid audio"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.avg_logprob = -0.1

        mock_info = MagicMock()
        mock_info.language = 'en'

        # First item succeeds, second fails preprocessing
        async def mock_preprocess_safe(audio_data, fmt):
            if audio_data == b'valid_audio':
                return mock_audio, 16000, 1000
            return None, 0, 0  # Failure

        service._preprocess_audio_safe = mock_preprocess_safe
        service._run_transcription = AsyncMock(return_value=([mock_segment], mock_info))

        batch = [(b'valid_audio', 'wav'), (b'invalid_audio', 'wav')]
        results = await service.transcribe_batch(batch)

        assert len(results) == 2
        assert results[0].text == 'Valid audio'
        assert results[1].text == ''  # Failed item
        assert results[1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_batch_all_preprocessing_failed(self, service):
        """Test batch transcription when all preprocessing fails"""
        async def mock_preprocess_safe(audio_data, fmt):
            return None, 0, 0  # All fail

        service._preprocess_audio_safe = mock_preprocess_safe

        batch = [(b'audio1', 'wav'), (b'audio2', 'wav')]
        results = await service.transcribe_batch(batch)

        assert len(results) == 2
        assert all(r.text == '' for r in results)
        assert all(r.confidence == 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_transcribe_batch_transcription_exception(self, service):
        """Test batch transcription when transcription raises exception"""
        mock_audio = np.random.randn(16000).astype('float32')

        async def mock_preprocess_safe(audio_data, fmt):
            return mock_audio, 16000, 1000

        service._preprocess_audio_safe = mock_preprocess_safe
        service._run_transcription = AsyncMock(side_effect=Exception("Transcription error"))

        batch = [(b'audio1', 'wav')]
        results = await service.transcribe_batch(batch)

        assert len(results) == 1
        assert results[0].text == ''
        assert results[0].confidence == 0.0


class TestGetStatus:
    """Test status reporting"""

    def test_get_status(self):
        """Test that get_status returns correct information"""
        service = STTService()
        service.is_ready = True
        service.hardware_info = {'device_type': 'cpu'}

        status = service.get_status()

        assert status['ready'] is True
        assert 'model_path' in status
        assert 'hardware' in status
        assert status['sample_rate'] == stt_config.SAMPLE_RATE
        assert status['language'] == stt_config.LANGUAGE


class TestUnloadModel:
    """Test model unloading"""

    def test_unload_model_success(self):
        """Test successful model unloading"""
        service = STTService()
        service.model = MagicMock()
        service.is_ready = True

        service.unload_model()

        assert service.model is None
        assert service.is_ready is False

    def test_unload_model_with_close_method(self):
        """Test model unloading when model has close() method"""
        service = STTService()
        mock_model = MagicMock()
        mock_model.close = MagicMock()
        service.model = mock_model
        service.is_ready = True

        service.unload_model()

        # Note: model is deleted, so we check the mock before deletion
        mock_model.close.assert_called_once()

    def test_unload_model_already_unloaded(self):
        """Test unloading when no model is loaded"""
        service = STTService()

        # Should not raise an exception
        service.unload_model()

        assert service.model is None
        assert service.is_ready is False


class TestRetryLogic:
    """Test retry and error recovery behavior"""

    @pytest.fixture
    def service(self):
        service = STTService()
        service.is_ready = True
        service.model = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_run_transcription_success_first_try(self, service):
        """Test that successful transcription doesn't retry"""
        mock_segment = MagicMock()
        mock_segment.text = "Success"
        mock_info = MagicMock()
        mock_info.language = 'en'

        service.model.transcribe.return_value = ([mock_segment], mock_info)

        audio_array = np.random.randn(16000).astype('float32')
        result = await service._run_transcription(audio_array, None)

        # Should be called exactly once (no retries)
        assert service.model.transcribe.call_count == 1
        assert result == ([mock_segment], mock_info)

    @pytest.mark.asyncio
    async def test_run_transcription_retry_then_success(self, service):
        """Test that transcription retries on failure then succeeds"""
        mock_segment = MagicMock()
        mock_segment.text = "Success"
        mock_info = MagicMock()
        mock_info.language = 'en'

        # Fail twice, then succeed
        service.model.transcribe.side_effect = [
            RuntimeError("Temporary error"),
            RuntimeError("Another error"),
            ([mock_segment], mock_info)
        ]

        audio_array = np.random.randn(16000).astype('float32')

        with patch.object(stt_config, 'MAX_RETRIES', 2), \
             patch.object(stt_config, 'RETRY_DELAY_SECONDS', 0.01):  # Fast retry for testing

            result = await service._run_transcription(audio_array, None)

            # Should be called 3 times (initial + 2 retries)
            assert service.model.transcribe.call_count == 3
            assert result == ([mock_segment], mock_info)

    @pytest.mark.asyncio
    async def test_run_transcription_all_retries_fail(self, service):
        """Test that transcription raises after all retries fail"""
        # Always fail
        service.model.transcribe.side_effect = RuntimeError("Persistent error")

        audio_array = np.random.randn(16000).astype('float32')

        with patch.object(stt_config, 'MAX_RETRIES', 2), \
             patch.object(stt_config, 'RETRY_DELAY_SECONDS', 0.01):

            with pytest.raises(RuntimeError, match="Persistent error"):
                await service._run_transcription(audio_array, None)

            # Should be called 3 times (initial + 2 retries)
            assert service.model.transcribe.call_count == 3

    @pytest.mark.asyncio
    async def test_run_transcription_exponential_backoff(self, service):
        """Test that retry delay uses exponential backoff"""
        service.model.transcribe.side_effect = RuntimeError("Error")

        audio_array = np.random.randn(16000).astype('float32')
        sleep_times = []

        original_sleep = asyncio.sleep

        async def track_sleep(delay):
            sleep_times.append(delay)
            await original_sleep(0)  # Don't actually wait

        with patch.object(stt_config, 'MAX_RETRIES', 2), \
             patch.object(stt_config, 'RETRY_DELAY_SECONDS', 0.5), \
             patch('asyncio.sleep', side_effect=track_sleep):

            try:
                await service._run_transcription(audio_array, None)
            except RuntimeError:
                pass

            # Should have exponential backoff: 0.5s, 1.0s (0.5 * 2^0, 0.5 * 2^1)
            assert len(sleep_times) == 2
            assert sleep_times[0] == 0.5  # First retry: delay * 2^0
            assert sleep_times[1] == 1.0  # Second retry: delay * 2^1

    @pytest.mark.asyncio
    async def test_transcribe_audio_handles_retry_failure(self, service):
        """Test that transcribe_audio returns empty result when retries exhausted"""
        # Make _run_transcription always fail
        service.model.transcribe.side_effect = RuntimeError("Persistent error")

        audio_array = np.random.randn(16000).astype('float32')

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(audio_array, 16000, 1000)), \
             patch.object(stt_config, 'MAX_RETRIES', 1), \
             patch.object(stt_config, 'RETRY_DELAY_SECONDS', 0.01):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            # Should return empty result instead of raising
            assert result.text == ''
            assert result.confidence == 0.0
