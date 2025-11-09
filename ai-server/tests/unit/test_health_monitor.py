"""Unit tests for Health Monitor"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from server.health_monitor import HealthMonitor


class TestHealthMonitor:
    """Test Health Monitor functionality"""

    @pytest.fixture
    def mqtt_service(self):
        """Mock MQTT service"""
        service = MagicMock()
        service.is_connected = MagicMock(return_value=True)
        service.publish_system_message = MagicMock()
        service.get_stats = MagicMock(return_value={
            'connected': True,
            'message_count': 100,
            'error_count': 5,
            'error_rate': 0.05,
            'handlers_registered': 10,
            'cache_size': 8
        })
        return service

    @pytest.fixture
    def protocol_handlers(self):
        """Mock protocol handlers"""
        handlers = MagicMock()
        handlers.get_performance_stats = AsyncMock(return_value={
            'active_devices': 2,
            'device_states': {'device1': 'ready', 'device2': 'ready'},
            'conversation_contexts': 1,
            'task_queue_size': 3,
            'task_queue_max': 10
        })
        return handlers

    @pytest.fixture
    def ai_model_manager(self):
        """Mock AI model manager"""
        manager = MagicMock()
        manager.get_status = MagicMock(return_value={
            'models_loaded': True,
            'llm': {
                'ready': True,
                'model_path': '/path/to/model.gguf',
                'hardware': {'device_type': 'cuda'}
            },
            'stt': {
                'ready': True,
                'model_path': '/path/to/stt.bin',
                'hardware': {'device_type': 'cuda'}
            },
            'tts': {
                'ready': True,
                'model_path': '/path/to/tts.onnx',
                'voice': 'lessac',
                'sample_rate': 22050,
                'output_format': 'mp3',
                'cached_phrases': 10
            }
        })
        return manager

    @pytest.fixture
    def health_monitor(self, mqtt_service, protocol_handlers):
        """Health monitor without AI model manager"""
        return HealthMonitor(mqtt_service, protocol_handlers)

    @pytest.fixture
    def health_monitor_with_ai(self, mqtt_service, protocol_handlers, ai_model_manager):
        """Health monitor with AI model manager"""
        return HealthMonitor(mqtt_service, protocol_handlers, ai_model_manager)

    @pytest.mark.asyncio
    async def test_start_monitoring(self, health_monitor):
        """Test starting health monitoring"""
        await health_monitor.start_monitoring(interval=30)

        assert health_monitor._monitoring_task is not None
        assert isinstance(health_monitor._monitoring_task, asyncio.Task)

        # Cleanup
        await health_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, health_monitor):
        """Test stopping health monitoring"""
        await health_monitor.start_monitoring(interval=30)
        await health_monitor.stop_monitoring()

        assert health_monitor._monitoring_task is None or health_monitor._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_publish_health_status_mqtt_connected(self, health_monitor_with_ai, mqtt_service, ai_model_manager):
        """Test publishing health status when MQTT is connected"""
        await health_monitor_with_ai._publish_health_status()

        # Verify MQTT publish was called
        assert mqtt_service.publish_system_message.called

        # Verify the message structure
        call_args = mqtt_service.publish_system_message.call_args
        assert call_args[0][0] == "health"
        assert call_args[0][1] == "services"

        health_data = call_args[0][2]
        assert 'timestamp' in health_data
        assert 'system' in health_data
        assert 'mqtt_service' in health_data
        assert 'protocol_handlers' in health_data
        assert 'ai_services' in health_data

    @pytest.mark.asyncio
    async def test_publish_health_status_mqtt_disconnected(self, health_monitor, mqtt_service):
        """Test that health status is not published when MQTT is disconnected"""
        mqtt_service.is_connected.return_value = False

        await health_monitor._publish_health_status()

        # Verify publish was not called
        assert not mqtt_service.publish_system_message.called

    @pytest.mark.asyncio
    async def test_health_status_system_info(self, health_monitor_with_ai):
        """Test system information in health status"""
        with patch('server.health_monitor.HealthMonitor._get_memory_usage', return_value=256.5):
            await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        system_info = health_data['system']
        assert 'uptime_seconds' in system_info
        assert 'status' in system_info
        assert system_info['status'] == 'healthy'
        assert 'memory_usage_mb' in system_info
        assert system_info['memory_usage_mb'] == 256.5
        assert 'python_version' in system_info
        assert 'platform' in system_info

    @pytest.mark.asyncio
    async def test_health_status_mqtt_service_info(self, health_monitor_with_ai):
        """Test MQTT service information in health status"""
        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        mqtt_info = health_data['mqtt_service']
        assert mqtt_info['status'] == 'healthy'
        assert mqtt_info['message_count'] == 100
        assert mqtt_info['error_count'] == 5
        assert mqtt_info['error_rate'] == 0.05
        assert mqtt_info['handlers_registered'] == 10
        assert 'cache_efficiency' in mqtt_info

    @pytest.mark.asyncio
    async def test_health_status_protocol_handlers_info(self, health_monitor_with_ai):
        """Test protocol handlers information in health status"""
        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        handlers_info = health_data['protocol_handlers']
        assert handlers_info['status'] == 'healthy'
        assert handlers_info['active_devices'] == 2
        assert handlers_info['conversation_contexts'] == 1
        assert 'task_queue_utilization' in handlers_info
        assert handlers_info['task_queue_utilization'] == 0.3  # 3/10

    @pytest.mark.asyncio
    async def test_health_status_ai_services_not_configured(self, health_monitor):
        """Test AI services status when not configured"""
        await health_monitor._publish_health_status()

        call_args = health_monitor.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        ai_services = health_data['ai_services']
        assert ai_services['status'] == 'not_configured'

    @pytest.mark.asyncio
    async def test_health_status_ai_services_with_llm(self, health_monitor_with_ai):
        """Test AI services status includes LLM information"""
        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        ai_services = health_data['ai_services']
        assert ai_services['status'] == 'healthy'
        assert ai_services['models_loaded'] is True
        assert 'llm' in ai_services['services']

        llm_info = ai_services['services']['llm']
        assert llm_info['status'] == 'ready'
        assert llm_info['model'] == 'model.gguf'
        assert llm_info['hardware'] == 'cuda'

    @pytest.mark.asyncio
    async def test_health_status_ai_services_with_stt(self, health_monitor_with_ai):
        """Test AI services status includes STT information"""
        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        ai_services = health_data['ai_services']
        assert 'stt' in ai_services['services']

        stt_info = ai_services['services']['stt']
        assert stt_info['status'] == 'ready'
        assert stt_info['model'] == 'stt.bin'
        assert stt_info['hardware'] == 'cuda'

    @pytest.mark.asyncio
    async def test_health_status_ai_services_with_tts(self, health_monitor_with_ai):
        """Test AI services status includes TTS information"""
        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        ai_services = health_data['ai_services']
        assert 'tts' in ai_services['services']

        tts_info = ai_services['services']['tts']
        assert tts_info['status'] == 'ready'
        assert tts_info['model'] == 'tts.onnx'
        assert tts_info['voice'] == 'lessac'
        assert tts_info['sample_rate'] == 22050
        assert tts_info['output_format'] == 'mp3'
        assert tts_info['cached_phrases'] == 10

    @pytest.mark.asyncio
    async def test_health_status_ai_services_not_ready(self, health_monitor_with_ai, ai_model_manager):
        """Test AI services status when models are not ready"""
        ai_model_manager.get_status.return_value = {
            'models_loaded': False,
            'tts': {
                'ready': False,
                'model_path': '',
                'voice': 'unknown',
                'sample_rate': 0,
                'output_format': 'unknown',
                'cached_phrases': 0
            }
        }

        await health_monitor_with_ai._publish_health_status()

        call_args = health_monitor_with_ai.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        ai_services = health_data['ai_services']
        assert ai_services['status'] == 'unhealthy'
        assert ai_services['models_loaded'] is False

        tts_info = ai_services['services']['tts']
        assert tts_info['status'] == 'not_ready'

    @pytest.mark.asyncio
    async def test_get_ai_services_status_exception_handling(self, health_monitor_with_ai, ai_model_manager):
        """Test that exceptions in AI services status are handled gracefully"""
        ai_model_manager.get_status.side_effect = RuntimeError("Model manager crashed")

        ai_services = health_monitor_with_ai._get_ai_services_status()

        assert ai_services['status'] == 'error'
        assert 'error' in ai_services
        assert 'Model manager crashed' in ai_services['error']

    def test_calculate_cache_efficiency(self, health_monitor):
        """Test cache efficiency calculation"""
        stats = {'cache_size': 8, 'handlers_registered': 10}
        efficiency = health_monitor._calculate_cache_efficiency(stats)
        assert efficiency == 0.8

    def test_calculate_cache_efficiency_over_100_percent(self, health_monitor):
        """Test cache efficiency caps at 100%"""
        stats = {'cache_size': 15, 'handlers_registered': 10}
        efficiency = health_monitor._calculate_cache_efficiency(stats)
        assert efficiency == 1.0

    def test_calculate_cache_efficiency_no_handlers(self, health_monitor):
        """Test cache efficiency with zero handlers"""
        stats = {'cache_size': 5, 'handlers_registered': 0}
        efficiency = health_monitor._calculate_cache_efficiency(stats)
        assert efficiency == 1.0  # min(5/1, 1.0)

    def test_get_memory_usage_with_psutil(self, health_monitor):
        """Test memory usage calculation with psutil available"""
        # Mock psutil at import time
        mock_psutil = MagicMock()
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 536870912  # 512 MB in bytes
        mock_psutil.Process.return_value = mock_process

        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            memory_mb = health_monitor._get_memory_usage()
            assert memory_mb == 512.0

    def test_get_memory_usage_without_psutil(self, health_monitor):
        """Test memory usage fallback when psutil is not available"""
        # Simulate ImportError when trying to import psutil
        import sys
        original_psutil = sys.modules.get('psutil')

        # Remove psutil from sys.modules to simulate it not being installed
        if 'psutil' in sys.modules:
            del sys.modules['psutil']

        # Make import fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("No module named 'psutil'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', mock_import):
            memory_mb = health_monitor._get_memory_usage()
            assert memory_mb == 0.0

        # Restore psutil if it was there
        if original_psutil is not None:
            sys.modules['psutil'] = original_psutil

    def test_get_memory_usage_exception(self, health_monitor):
        """Test memory usage calculation handles exceptions gracefully"""
        mock_psutil = MagicMock()
        mock_psutil.Process.side_effect = Exception("Process error")

        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            memory_mb = health_monitor._get_memory_usage()
            assert memory_mb == 0.0

    @pytest.mark.asyncio
    async def test_monitor_loop_runs_periodically(self, health_monitor):
        """Test that monitor loop publishes health status periodically"""
        # Start monitoring with short interval
        await health_monitor.start_monitoring(interval=0.1)

        # Wait for at least one cycle
        await asyncio.sleep(0.15)

        # Verify health status was published
        assert health_monitor.mqtt_service.publish_system_message.called

        # Cleanup
        await health_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_monitor_loop_handles_exceptions(self, health_monitor, protocol_handlers):
        """Test that monitor loop handles exceptions and continues"""
        # Make get_performance_stats fail once, then succeed
        protocol_handlers.get_performance_stats.side_effect = [
            RuntimeError("Temporary error"),
            {
                'active_devices': 1,
                'device_states': {},
                'conversation_contexts': 0,
                'task_queue_size': 0,
                'task_queue_max': 10
            }
        ]

        await health_monitor.start_monitoring(interval=0.1)

        # Wait for multiple cycles
        await asyncio.sleep(0.25)

        # Loop should continue despite error
        assert not health_monitor._monitoring_task.done()

        # Cleanup
        await health_monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_mqtt_unhealthy_status(self, health_monitor, mqtt_service):
        """Test MQTT service status when disconnected"""
        mqtt_service.get_stats.return_value['connected'] = False
        mqtt_service.is_connected.return_value = True  # Still publish even if stats show disconnected

        await health_monitor._publish_health_status()

        call_args = mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        mqtt_info = health_data['mqtt_service']
        assert mqtt_info['status'] == 'unhealthy'

    @pytest.mark.asyncio
    async def test_health_data_qos_level(self, health_monitor):
        """Test that health data is published with QoS level 1"""
        await health_monitor._publish_health_status()

        call_args = health_monitor.mqtt_service.publish_system_message.call_args
        assert call_args[1]['qos'] == 1

    @pytest.mark.asyncio
    async def test_health_status_includes_timestamp(self, health_monitor):
        """Test that health status includes ISO format timestamp"""
        await health_monitor._publish_health_status()

        call_args = health_monitor.mqtt_service.publish_system_message.call_args
        health_data = call_args[0][2]

        assert 'timestamp' in health_data
        # Verify it's a valid ISO format timestamp
        timestamp = datetime.fromisoformat(health_data['timestamp'].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
