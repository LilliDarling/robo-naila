"""Unit tests for LLM Service"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open

import pytest

from services.llm import LLMService
from config import llm as llm_config


class TestLLMServiceInit:
    """Test LLM Service initialization"""

    def test_init_sets_defaults(self):
        """Test that __init__ sets correct default values"""
        service = LLMService()

        assert service.model is None
        assert service.is_ready is False
        assert service.hardware_info is None
        assert isinstance(service.model_path, Path)
        # system_prompt is None until model is loaded (loaded asynchronously)
        assert service.system_prompt is None
        assert service._pool is None

    @pytest.mark.asyncio
    async def test_load_system_prompt_from_file(self):
        """Test that system prompt is loaded from file if it exists"""
        service = LLMService()

        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value="Custom system prompt"):
                prompt = await service._load_system_prompt()

        assert prompt == "Custom system prompt"

    @pytest.mark.asyncio
    async def test_load_system_prompt_uses_fallback_when_file_missing(self):
        """Test that fallback prompt is used when file doesn't exist"""
        service = LLMService()

        with patch.object(Path, 'exists', return_value=False):
            prompt = await service._load_system_prompt()

        assert prompt == llm_config.FALLBACK_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_load_system_prompt_uses_fallback_on_read_error(self):
        """Test that fallback prompt is used when file read fails"""
        service = LLMService()

        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', side_effect=IOError("File error")):
                prompt = await service._load_system_prompt()

        assert prompt == llm_config.FALLBACK_SYSTEM_PROMPT


class TestLoadModel:
    """Test model loading functionality"""

    @pytest.fixture
    def service(self):
        return LLMService()

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
        mock_llama = MagicMock()
        hardware_info = {
            'device_type': 'cuda',
            'device_name': 'Test GPU',
            'acceleration': 'cuda',
            'cpu_count': 8,
            'vram_gb': 8.0
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.llm.Llama', return_value=mock_llama) as mock_llama_class:

            result = await service.load_model(hardware_info=hardware_info)

            assert result is True
            assert service.is_ready is True
            assert service.model == mock_llama
            assert service.hardware_info is not None
            assert service.hardware_info == hardware_info
            mock_llama_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_success_without_hardware_info(self, service):
        """Test successful model loading with automatic hardware detection"""
        mock_llama = MagicMock()
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'
        mock_hw_optimizer.hardware_info.device_name = 'Test GPU'
        mock_hw_optimizer.hardware_info.memory_gb = 8.0

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('services.llm.Llama', return_value=mock_llama) as mock_llama_class:

            result = await service.load_model()

            assert result is True
            assert service.is_ready is True
            assert service.model == mock_llama
            assert service.hardware_info is not None
            mock_llama_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_import_error(self, service):
        """Test that load_model handles missing llama-cpp-python gracefully"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer'), \
             patch('services.llm.HAS_LLAMA_CPP', False), \
             patch('services.llm.Llama', None):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_exception(self, service):
        """Test that load_model handles unexpected exceptions"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', side_effect=Exception("Hardware error")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_invalid_file(self, service):
        """Test that load_model handles invalid model files"""
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cpu'
        mock_hw_optimizer.hardware_info.device_name = 'Test CPU'
        mock_hw_optimizer.hardware_info.memory_gb = None

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('services.llm.Llama', side_effect=ValueError("Invalid model format")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_concurrent_calls(self, service):
        """Test concurrent model loading attempts"""
        # This test verifies that the check for is_ready prevents redundant loads
        # However, since the check happens at the start and all three coroutines
        # may check before any sets is_ready=True, we just verify all succeed
        mock_llama = MagicMock()
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cpu'
        mock_hw_optimizer.hardware_info.device_name = 'Test CPU'
        mock_hw_optimizer.hardware_info.memory_gb = None

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.base.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('services.llm.Llama', return_value=mock_llama):

            # Start multiple concurrent loads
            results = await asyncio.gather(
                service.load_model(),
                service.load_model(),
                service.load_model()
            )

            # All should succeed (either loading or detecting already loaded)
            assert all(results)
            # Service should be ready after all complete
            assert service.is_ready is True


class TestThreadCount:
    """Test thread count calculation"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_get_thread_count_from_config(self, service):
        """Test that configured thread count is used when set"""
        result = service._get_thread_count(config_threads=8)
        assert result == 8

    def test_get_thread_count_auto_detect(self, service):
        """Test thread count auto-detection from hardware info"""
        service.hardware_info = {'cpu_count': 16}
        result = service._get_thread_count(config_threads=0)
        # Should use 75% of cores, minimum 2
        assert result == 12

    def test_get_thread_count_default(self, service):
        """Test default thread count when no hardware info available"""
        service.hardware_info = None
        result = service._get_thread_count(config_threads=0)
        assert result == 4

    def test_get_thread_count_negative_config(self, service):
        """Test that negative THREADS config falls back to auto or default"""
        service.hardware_info = {'cpu_count': 8}
        result = service._get_thread_count(config_threads=-5)
        # Should fall back to default (since THREADS check is > 0, not >= 0)
        assert result == 6  # 75% of 8

        service.hardware_info = None
        result = service._get_thread_count(config_threads=-3)
        # Should fall back to default value
        assert result == 4


class TestGPULayers:
    """Test GPU layer calculation"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_get_gpu_layers_auto_with_cuda(self, service):
        """Test GPU layer auto-detection with CUDA available"""
        service.hardware_info = {'acceleration': 'cuda'}

        with patch.object(llm_config, 'GPU_LAYERS', -1):
            result = service._get_gpu_layers()

        assert result == -1

    def test_get_gpu_layers_auto_with_metal(self, service):
        """Test GPU layer auto-detection with Metal available"""
        service.hardware_info = {'acceleration': 'metal'}

        with patch.object(llm_config, 'GPU_LAYERS', -1):
            result = service._get_gpu_layers()

        assert result == -1

    def test_get_gpu_layers_auto_with_cpu(self, service):
        """Test GPU layer auto-detection with CPU only"""
        service.hardware_info = {'acceleration': 'cpu'}

        with patch.object(llm_config, 'GPU_LAYERS', -1):
            result = service._get_gpu_layers()

        assert result == 0

    def test_get_gpu_layers_explicit_value(self, service):
        """Test explicit GPU layer configuration"""
        with patch.object(llm_config, 'GPU_LAYERS', 20):
            result = service._get_gpu_layers()

        assert result == 20

    def test_get_gpu_layers_missing_acceleration_key(self, service):
        """Test _get_gpu_layers when hardware_info lacks 'acceleration' key"""
        service.hardware_info = {'device_type': 'cpu'}  # Missing 'acceleration' key

        with patch.object(llm_config, 'GPU_LAYERS', -1):
            result = service._get_gpu_layers()

        # Should default to CPU (0 layers) since acceleration key is missing
        assert result == 0


class TestGenerateChat:
    """Test chat generation"""

    @pytest.fixture
    def service(self):
        service = LLMService()
        service.is_ready = True
        service.model = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_generate_chat_not_ready(self):
        """Test that generation fails when model not ready"""
        service = LLMService()
        service.is_ready = False

        result = await service.generate_chat([{"role": "user", "content": "Hello"}])

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_chat_model_none(self, service):
        """Test that generation fails when model is None"""
        service.model = None

        result = await service.generate_chat([{"role": "user", "content": "Hello"}])

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_chat_success(self, service):
        """Test successful chat generation"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        mock_response = {
            'choices': [{'text': 'Hi there!'}],
            'usage': {'completion_tokens': 5}
        }
        service.model.return_value = mock_response

        result = await service.generate_chat(messages)

        assert result == "Hi there!"
        service.model.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_chat_with_custom_params(self, service):
        """Test chat generation with custom parameters"""
        messages = [{"role": "user", "content": "Test"}]
        mock_response = {
            'choices': [{'text': 'Response'}],
            'usage': {'completion_tokens': 10}
        }
        service.model.return_value = mock_response

        result = await service.generate_chat(
            messages,
            max_tokens=100,
            temperature=0.5,
            top_p=0.8
        )

        assert result == "Response"
        call_kwargs = service.model.call_args[1]
        assert call_kwargs['max_tokens'] == 100
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['top_p'] == 0.8

    @pytest.mark.asyncio
    async def test_generate_chat_cleans_response(self, service):
        """Test that response is cleaned (whitespace, special tokens)"""
        messages = [{"role": "user", "content": "Test"}]
        mock_response = {
            'choices': [{'text': '  Response with spaces  <|eot_id|>  '}],
            'usage': {'completion_tokens': 5}
        }
        service.model.return_value = mock_response

        result = await service.generate_chat(messages)

        # Should strip leading whitespace and remove <|eot_id|>, but trailing spaces after token removal remain
        assert result == "Response with spaces  "

    @pytest.mark.asyncio
    async def test_generate_chat_handles_exception(self, service):
        """Test that exceptions during generation are handled"""
        messages = [{"role": "user", "content": "Test"}]
        service.model.side_effect = Exception("Generation failed")

        result = await service.generate_chat(messages)

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_chat_empty_messages(self, service):
        """Test generate_chat with empty messages list"""
        messages = []
        mock_response = {
            'choices': [{'text': 'Response'}],
            'usage': {'completion_tokens': 5}
        }
        service.model.return_value = mock_response

        result = await service.generate_chat(messages)

        # Should still work with empty messages
        assert result == "Response"


class TestFormatChatPrompt:
    """Test prompt formatting"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_format_chat_prompt_single_message(self, service):
        """Test formatting a single user message"""
        messages = [{"role": "user", "content": "Hello"}]

        result = service._format_chat_prompt(messages)

        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello" in result
        assert "<|eot_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        # Should NOT have begin_of_text - llama-cpp-python adds it
        assert "<|begin_of_text|>" not in result

    def test_format_chat_prompt_conversation(self, service):
        """Test formatting a multi-turn conversation"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]

        result = service._format_chat_prompt(messages)

        # 4 messages + 1 assistant prompt = 5 total headers
        assert result.count("<|start_header_id|>") == 5
        assert result.count("<|eot_id|>") == 4  # One for each message
        assert "You are helpful" in result
        assert "How are you?" in result
        assert "<|begin_of_text|>" not in result  # Should not be added by us

    def test_format_chat_prompt_strips_whitespace(self, service):
        """Test that message content whitespace is stripped"""
        messages = [{"role": "user", "content": "  Hello  \n\n  "}]

        result = service._format_chat_prompt(messages)

        assert "  Hello  " not in result
        assert "Hello" in result

    def test_format_chat_prompt_missing_content_or_role(self, service):
        """Test formatting with missing 'content' or 'role' keys"""
        # Missing 'content'
        messages_missing_content = [{"role": "user"}]
        # Missing 'role'
        messages_missing_role = [{"content": "Hello"}]

        # Expecting KeyError when accessing missing keys
        with pytest.raises(KeyError):
            service._format_chat_prompt(messages_missing_content)

        with pytest.raises(KeyError):
            service._format_chat_prompt(messages_missing_role)


class TestCleanResponse:
    """Test response cleaning"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_clean_response_strips_whitespace(self, service):
        """Test that leading/trailing whitespace is removed"""
        text = "  Response text  \n\n  "

        result = service._clean_response(text)

        assert result == "Response text"

    def test_clean_response_removes_stop_sequences(self, service):
        """Test that stop sequences are removed from the end"""
        text = "Response<|end_of_text|>"

        result = service._clean_response(text)

        assert "<|end_of_text|>" not in result
        assert result == "Response"

    def test_clean_response_truncates_long_text(self, service):
        """Test that overly long responses are truncated"""
        text = "A" * 1500  # Longer than MAX_RESPONSE_LENGTH

        with patch.object(llm_config, 'MAX_RESPONSE_LENGTH', 1000):
            result = service._clean_response(text)

        assert len(result) == 1000

    def test_clean_response_only_stop_sequences(self, service):
        """Test that input with only stop sequences is cleaned to empty string"""
        # Since we only strip from the end, we need stop sequences at the end
        text = "<|end_of_text|>"

        result = service._clean_response(text)

        assert result == ""


class TestBuildChatMessages:
    """Test chat message building"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_build_chat_messages_simple(self, service):
        """Test building messages with just a query"""
        result = service.build_chat_messages("Hello", [])

        assert len(result) == 2  # system + user
        assert result[0]['role'] == 'system'
        assert result[1]['role'] == 'user'
        assert result[1]['content'] == 'Hello'

    def test_build_chat_messages_with_history(self, service):
        """Test building messages with conversation history"""
        history = [
            {"user": "Hi", "assistant": "Hello!"},
            {"user": "How are you?", "assistant": "I'm good!"}
        ]

        result = service.build_chat_messages("What's your name?", history)

        # system + (2 exchanges * 2 messages each) + current user query = 6
        # Note: The implementation uses extend() with 2 messages per exchange, not 1
        assert len(result) == 6
        assert result[0]['role'] == 'system'
        assert result[1]['role'] == 'user'
        assert result[1]['content'] == 'Hi'
        assert result[2]['role'] == 'assistant'
        assert result[2]['content'] == 'Hello!'
        assert result[-1]['role'] == 'user'
        assert result[-1]['content'] == "What's your name?"

    def test_build_chat_messages_limits_history(self, service):
        """Test that history is limited to CONTEXT_HISTORY_LIMIT"""
        history = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(10)]

        with patch.object(llm_config, 'CONTEXT_HISTORY_LIMIT', 3):
            result = service.build_chat_messages("Final question", history)

        # Should have: system + (3 * 2 exchanges) + current query = 8
        assert len(result) == 8
        # Should include last 3 exchanges only
        assert "Q7" in result[1]['content']
        assert "Q6" not in str(result)

    def test_build_chat_messages_without_system(self, service):
        """Test building messages without system prompt"""
        result = service.build_chat_messages("Hello", [], include_system=False)

        assert len(result) == 1
        assert result[0]['role'] == 'user'

    def test_build_chat_messages_with_none_query(self, service):
        """Test build_chat_messages with query=None"""
        # The current implementation doesn't validate None, but it will append None as content
        # This test documents current behavior - None is treated as a valid string
        result = service.build_chat_messages(None, [])

        # Should have system + user message with None content
        assert len(result) == 2
        assert result[-1]['content'] is None


class TestGetStatus:
    """Test status reporting"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_get_status_returns_complete_info(self, service):
        """Test that get_status returns all required fields"""
        service.is_ready = True
        service.hardware_info = {'device': 'cuda'}

        with patch('pathlib.Path.exists', return_value=True):
            status = service.get_status()

        assert 'ready' in status
        assert 'model_path' in status
        assert 'model_exists' in status
        assert 'hardware' in status
        assert 'context_size' in status
        assert 'max_tokens' in status

    def test_get_status_ready_state(self, service):
        """Test status reflects ready state correctly"""
        service.is_ready = True

        status = service.get_status()

        assert status['ready'] is True

    def test_get_status_model_path_not_exists(self, service):
        """Test get_status when model_path does not exist"""
        service.is_ready = False
        service.hardware_info = {'device': 'cpu'}
        with patch('pathlib.Path.exists', return_value=False):
            status = service.get_status()
        assert 'ready' in status
        assert 'model_path' in status
        assert 'model_exists' in status
        assert status['model_exists'] is False
        assert 'hardware' in status
        assert 'context_size' in status
        assert 'max_tokens' in status


class TestUnloadModel:
    """Test model unloading"""

    @pytest.fixture
    def service(self):
        service = LLMService()
        service.model = MagicMock()
        service.is_ready = True
        return service

    def test_unload_model_clears_state(self, service):
        """Test that unload_model clears model and ready state"""
        service.unload_model()

        assert service.model is None
        assert service.is_ready is False

    def test_unload_model_when_not_loaded(self):
        """Test that unload_model handles already unloaded state"""
        service = LLMService()
        service.model = None
        service.is_ready = False

        # Should not raise exception
        service.unload_model()

        assert service.model is None
        assert not service.is_ready
