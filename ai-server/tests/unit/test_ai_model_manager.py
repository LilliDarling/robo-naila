"""Unit tests for AI Model Manager"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.ai_model_manager import AIModelManager
from services.llm import LLMService
from services.stt import STTService


class TestAIModelManager:
    """Test AI Model Manager functionality"""

    @pytest.fixture
    def llm_service(self):
        service = MagicMock()
        model_path = MagicMock()
        model_path.name = "test_llm.gguf"
        service.model_path = model_path
        service.load_model = AsyncMock(return_value=True)
        service.is_ready = False
        service.get_status = MagicMock(return_value={'ready': False})
        return service

    @pytest.fixture
    def stt_service(self):
        service = MagicMock()
        model_path = MagicMock()
        model_path.name = "test_stt.bin"
        service.model_path = model_path
        service.load_model = AsyncMock(return_value=True)
        service.is_ready = False
        service.get_status = MagicMock(return_value={'ready': False})
        return service

    @pytest.fixture
    def manager(self, llm_service, stt_service):
        return AIModelManager(llm_service, stt_service)

    @pytest.mark.asyncio
    async def test_hardware_detection_called_once(self, manager, llm_service, stt_service):
        """Test that hardware detection is only called once for all services"""
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cuda'
        mock_hw_optimizer.hardware_info.device_name = 'Test GPU'
        mock_hw_optimizer.hardware_info.memory_gb = 8.0

        with patch('services.ai_model_manager.HardwareOptimizer', return_value=mock_hw_optimizer) as mock_hw_class:
            await manager.load_models()

            # Hardware detection should be called only once
            assert mock_hw_class.call_count == 1

            # Both services should receive the hardware info
            assert llm_service.load_model.called
            assert stt_service.load_model.called

            # Verify hardware_info was passed to both services
            llm_call_kwargs = llm_service.load_model.call_args.kwargs
            stt_call_kwargs = stt_service.load_model.call_args.kwargs

            assert 'hardware_info' in llm_call_kwargs
            assert 'hardware_info' in stt_call_kwargs
            assert llm_call_kwargs['hardware_info'] == stt_call_kwargs['hardware_info']

    @pytest.mark.asyncio
    async def test_parallel_model_loading(self, manager, llm_service, stt_service):
        """Test that models are loaded in parallel using asyncio.gather"""
        import asyncio

        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cpu'
        mock_hw_optimizer.hardware_info.device_name = 'CPU'
        mock_hw_optimizer.hardware_info.memory_gb = 16.0

        with patch('services.ai_model_manager.HardwareOptimizer', return_value=mock_hw_optimizer), \
             patch('services.ai_model_manager.asyncio.gather', new_callable=AsyncMock) as mock_gather:

            # Make gather return successful results
            mock_gather.return_value = [True, True]

            await manager.load_models()

            # Verify asyncio.gather was called
            assert mock_gather.called

            # Verify it was called with both model loading tasks
            call_args = mock_gather.call_args
            assert len(call_args[0]) == 2  # Two tasks (LLM and STT)

            # Verify return_exceptions=True was passed
            assert call_args[1].get('return_exceptions') is True

    @pytest.mark.asyncio
    async def test_hardware_info_cached(self, manager):
        """Test that hardware info is cached after first detection"""
        mock_hw_optimizer = MagicMock()
        mock_hw_optimizer.hardware_info.device_type = 'cpu'
        mock_hw_optimizer.hardware_info.device_name = 'CPU'
        mock_hw_optimizer.hardware_info.memory_gb = 16.0

        with patch('services.ai_model_manager.HardwareOptimizer', return_value=mock_hw_optimizer) as mock_hw_class:
            # First detection
            hardware_info_1 = manager._detect_hardware()

            # Second detection should use cache
            hardware_info_2 = manager._detect_hardware()

            # Should only be called once
            assert mock_hw_class.call_count == 1

            # Results should be identical
            assert hardware_info_1 is hardware_info_2

    @pytest.mark.asyncio
    async def test_load_models_with_llm_only(self, llm_service):
        """Test loading models with only LLM service"""
        manager = AIModelManager(llm_service=llm_service)

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            assert result is True
            assert llm_service.load_model.called

    @pytest.mark.asyncio
    async def test_load_models_with_stt_only(self, stt_service):
        """Test loading models with only STT service"""
        manager = AIModelManager(stt_service=stt_service)

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            # STT is optional, so success depends on whether we consider it required
            assert stt_service.load_model.called

    @pytest.mark.asyncio
    async def test_load_models_llm_failure(self, manager, llm_service, stt_service):
        """Test that LLM failure affects overall success"""
        llm_service.load_model = AsyncMock(return_value=False)

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            assert result is False  # LLM failure means overall failure

    @pytest.mark.asyncio
    async def test_load_models_stt_failure(self, manager, llm_service, stt_service):
        """Test that STT failure doesn't affect overall success"""
        stt_service.load_model = AsyncMock(return_value=False)

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            # STT is optional, so overall should still succeed if LLM loaded
            assert result is True

    @pytest.mark.asyncio
    async def test_load_models_exception_handling(self, manager, llm_service, stt_service):
        """Test that exceptions during loading are handled gracefully"""
        # Make STT raise an exception
        stt_service.load_model = AsyncMock(side_effect=RuntimeError("Model corrupted"))

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            # Should handle exception and still succeed if LLM loaded
            assert result is True  # LLM loaded successfully

    @pytest.mark.asyncio
    async def test_load_models_llm_exception(self, manager, llm_service, stt_service):
        """Test that LLM exception affects overall result"""
        # Make LLM raise an exception
        llm_service.load_model = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with patch('services.ai_model_manager.HardwareOptimizer'):
            result = await manager.load_models()

            # Should fail overall if LLM throws exception
            assert result is False

    def test_get_llm_service(self, manager, llm_service):
        """Test getting LLM service"""
        assert manager.get_llm_service() == llm_service

    def test_get_stt_service(self, manager, stt_service):
        """Test getting STT service"""
        assert manager.get_stt_service() == stt_service

    def test_get_status(self, manager, llm_service, stt_service):
        """Test getting status of all models"""
        llm_service.get_status.return_value = {'ready': True}
        stt_service.get_status.return_value = {'ready': True}

        status = manager.get_status()

        assert 'models_loaded' in status
        assert 'llm' in status
        assert 'stt' in status
        assert llm_service.get_status.called
        assert stt_service.get_status.called
