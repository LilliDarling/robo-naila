"""Unit tests for ServerLifecycleManager shutdown behavior.

The lifecycle owns shutdown cleanup of long-lived resources. The memory
connection is one of those — it must be closed during graceful shutdown.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from server.lifecycle import ServerLifecycleManager


@pytest.fixture
def mqtt_service():
    service = MagicMock()
    service.is_connected = MagicMock(return_value=True)
    service.publish_system_message = MagicMock()
    service.stop = AsyncMock()
    return service


@pytest.fixture
def protocol_handlers():
    return MagicMock()


@pytest.fixture
def orchestrator():
    """Orchestrator with a mock memory whose close() we can assert on."""
    orch = MagicMock()
    orch.memory = MagicMock()
    orch.memory.close = MagicMock()
    return orch


@pytest.fixture
def lifecycle(mqtt_service, protocol_handlers, orchestrator):
    mgr = ServerLifecycleManager(
        mqtt_service=mqtt_service,
        protocol_handlers=protocol_handlers,
        orchestrator=orchestrator,
    )
    # Stub the heavy collaborators so stop_server doesn't try to talk to
    # real subsystems.
    mgr.health_monitor = MagicMock()
    mgr.health_monitor.stop_monitoring = AsyncMock()
    mgr.ai_model_manager = MagicMock()
    mgr.ai_model_manager.unload_models = MagicMock()
    return mgr


class TestShutdownClosesMemory:
    @pytest.mark.asyncio
    async def test_stop_server_closes_memory_connection(self, lifecycle, orchestrator):
        await lifecycle.stop_server()
        orchestrator.memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_close_after_mqtt_stop(self, lifecycle, orchestrator, mqtt_service):
        # Memory must close AFTER MQTT — otherwise an in-flight handler can
        # still try to commit_exchange on a closed connection.
        call_order: list[str] = []
        mqtt_service.stop = AsyncMock(side_effect=lambda: call_order.append("mqtt_stop"))
        orchestrator.memory.close = MagicMock(side_effect=lambda: call_order.append("memory_close"))

        await lifecycle.stop_server()

        assert call_order.index("mqtt_stop") < call_order.index("memory_close")

    @pytest.mark.asyncio
    async def test_memory_close_runs_even_if_earlier_stage_errors(
        self, lifecycle, orchestrator
    ):
        # Even if health monitoring shutdown blows up, the connection must
        # still be closed. Resource leaks on unhappy paths are still leaks.
        lifecycle.health_monitor.stop_monitoring = AsyncMock(
            side_effect=RuntimeError("health monitor exploded")
        )

        await lifecycle.stop_server()

        orchestrator.memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_orchestrator_means_no_close_attempt(
        self, mqtt_service, protocol_handlers
    ):
        # Defensive: if for some reason there's no orchestrator (e.g. partial
        # init failure), shutdown shouldn't raise AttributeError.
        mgr = ServerLifecycleManager(
            mqtt_service=mqtt_service,
            protocol_handlers=protocol_handlers,
            orchestrator=None,
        )
        mgr.health_monitor = MagicMock()
        mgr.health_monitor.stop_monitoring = AsyncMock()
        mgr.ai_model_manager = MagicMock()
        mgr.ai_model_manager.unload_models = MagicMock()

        await mgr.stop_server()  # must not raise
