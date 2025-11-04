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
        assert service.system_prompt is not None

    @patch('builtins.open', new_callable=mock_open, read_data="Custom system prompt")
    @patch.object(Path, 'exists', return_value=True)
    def test_init_loads_system_prompt_from_file(self, mock_exists, mock_file):
        """Test that system prompt is loaded from file if it exists"""
        service = LLMService()

        assert service.system_prompt == "Custom system prompt"
        mock_file.assert_called_once()

    @patch.object(Path, 'exists', return_value=False)
    def test_init_uses_fallback_prompt_when_file_missing(self, mock_exists):
        """Test that fallback prompt is used when file doesn't exist"""
        service = LLMService()

        assert service.system_prompt == llm_config.FALLBACK_SYSTEM_PROMPT

    @patch('builtins.open', side_effect=IOError("File error"))
    @patch.object(Path, 'exists', return_value=True)
    def test_init_uses_fallback_prompt_on_read_error(self, mock_exists, mock_file):
        """Test that fallback prompt is used when file read fails"""
        service = LLMService()

        assert service.system_prompt == llm_config.FALLBACK_SYSTEM_PROMPT


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
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'
        mock_hw_optimizer.hardware_info.device_name = 'Test GPU'

        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.llm.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('llama_cpp.Llama', return_value=mock_llama) as mock_llama_class:

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
             patch('services.llm.HardwareOptimizer'), \
             patch('builtins.__import__', side_effect=ImportError("llama_cpp not found")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_load_model_exception(self, service):
        """Test that load_model handles unexpected exceptions"""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('services.llm.HardwareOptimizer', side_effect=Exception("Hardware error")):

            result = await service.load_model()

            assert result is False
            assert service.is_ready is False


class TestThreadCount:
    """Test thread count calculation"""

    @pytest.fixture
    def service(self):
        return LLMService()

    def test_get_thread_count_from_config(self, service):
        """Test that configured thread count is used when set"""
        with patch.object(llm_config, 'THREADS', 8):
            result = service._get_thread_count()

        assert result == 8

    def test_get_thread_count_auto_detect(self, service):
        """Test thread count auto-detection from hardware info"""
        service.hardware_info = {'cpu_count': 16}

        with patch.object(llm_config, 'THREADS', 0):
            result = service._get_thread_count()

        # Should use 75% of cores, minimum 2
        assert result == 12

    def test_get_thread_count_default(self, service):
        """Test default thread count when no hardware info available"""
        service.hardware_info = None

        with patch.object(llm_config, 'THREADS', 0):
            result = service._get_thread_count()

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
        """Test that stop sequences are removed"""
        text = "Response<|eot_id|><|end_of_text|>"

        result = service._clean_response(text)

        assert "<|eot_id|>" not in result
        assert "<|end_of_text|>" not in result
        assert result == "Response"

    def test_clean_response_truncates_long_text(self, service):
        """Test that overly long responses are truncated"""
        text = "A" * 1500  # Longer than MAX_RESPONSE_LENGTH

        with patch.object(llm_config, 'MAX_RESPONSE_LENGTH', 1000):
            result = service._clean_response(text)

        assert len(result) == 1000


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
