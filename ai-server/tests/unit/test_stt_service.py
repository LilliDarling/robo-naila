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
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'
        mock_hw_optimizer.hardware_info.device_name = 'Test GPU'
        mock_hw_optimizer.hardware_info.memory_gb = 8.0

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.stt.HardwareOptimizer', return_value=mock_hw_optimizer), \
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
             patch('services.stt.HardwareOptimizer'):

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
             patch('services.stt.HardwareOptimizer', return_value=mock_hw_optimizer), \
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
             patch('services.stt.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('faster_whisper.WhisperModel', side_effect=ValueError("CUDA error")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False


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
        with patch.object(stt_config, 'THREADS', 8):
            count = service._get_thread_count()

            assert count == 8

    def test_get_thread_count_auto_detect(self, service):
        """Test auto-detection of thread count"""
        service.hardware_info = {'cpu_count': 16}

        with patch.object(stt_config, 'THREADS', 0):
            count = service._get_thread_count()

            assert count == 12  # 75% of 16

    def test_get_thread_count_default(self, service):
        """Test default thread count fallback"""
        service.hardware_info = None

        with patch.object(stt_config, 'THREADS', 0):
            count = service._get_thread_count()

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

        with patch('soundfile.read', return_value=(mock_audio_array, mock_sample_rate)):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            assert isinstance(audio_array, np.ndarray)
            assert sample_rate == 16000
            assert duration_ms == 1000  # 1 second

    @pytest.mark.asyncio
    async def test_preprocess_audio_stereo_to_mono(self, service):
        """Test conversion of stereo audio to mono"""
        # Create mock stereo audio (2 channels)
        mock_stereo_audio = np.random.randn(16000, 2).astype('float32')
        mock_sample_rate = 16000

        with patch('soundfile.read', return_value=(mock_stereo_audio, mock_sample_rate)):
            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            assert len(audio_array.shape) == 1  # Should be mono now

    @pytest.mark.asyncio
    async def test_preprocess_audio_resampling(self, service):
        """Test resampling of audio to 16kHz"""
        # Create mock audio at 44.1kHz
        mock_audio_array = np.random.randn(44100).astype('float32')
        mock_sample_rate = 44100

        with patch('soundfile.read', return_value=(mock_audio_array, mock_sample_rate)), \
             patch('resampy.resample', return_value=np.random.randn(16000).astype('float32')) as mock_resample:

            audio_array, sample_rate, duration_ms = await service._preprocess_audio(b'fake_wav_data', 'wav')

            mock_resample.assert_called_once()
            assert sample_rate == 16000

    @pytest.mark.asyncio
    async def test_preprocess_audio_truncate_long(self, service):
        """Test truncation of audio that's too long"""
        # Create audio longer than MAX_DURATION_MS
        long_duration_samples = int((stt_config.MAX_DURATION_MS / 1000.0) * 16000) + 10000
        mock_audio_array = np.random.randn(long_duration_samples).astype('float32')
        mock_sample_rate = 16000

        with patch('soundfile.read', return_value=(mock_audio_array, mock_sample_rate)):
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

        service.model.transcribe.return_value = ([], mock_info)

        with patch.object(service, '_validate_audio', return_value=(True, '')), \
             patch.object(service, '_preprocess_audio', return_value=(mock_audio_array, 16000, 1000)):

            result = await service.transcribe_audio(b'fake_audio', 'wav')

            assert result.text == ''
            assert result.confidence == 0.0


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
        service.model = MagicMock()
        service.model.close = MagicMock()
        service.is_ready = True

        service.unload_model()

        service.model.close.assert_called_once()

    def test_unload_model_already_unloaded(self):
        """Test unloading when no model is loaded"""
        service = STTService()

        # Should not raise an exception
        service.unload_model()

        assert service.model is None
        assert service.is_ready is False
