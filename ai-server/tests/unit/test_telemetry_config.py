"""Tests for trace export configuration and the benchmark profile."""

import importlib

import pytest


@pytest.fixture
def reload_telemetry_config(monkeypatch):
    """Reload ``config.telemetry`` after env mutation so the benchmark switch re-evaluates."""

    def _reload(env: dict | None = None):
        for key in ("NAILA_BENCHMARK", "NAILA_OTEL_ENABLED", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            monkeypatch.delenv(key, raising=False)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        import config.benchmark as benchmark
        import config.telemetry as mod

        importlib.reload(benchmark)
        return importlib.reload(mod).TelemetryConfig.from_env()

    return _reload


class TestTelemetryConfig:
    def test_disabled_by_default(self, reload_telemetry_config):
        config = reload_telemetry_config()
        assert not config.enabled
        assert config.traces_endpoint is None

    def test_enabled_flag(self, reload_telemetry_config):
        config = reload_telemetry_config({
            "NAILA_OTEL_ENABLED": "1",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://collector:4318/v1/traces",
        })
        assert config.enabled
        assert config.traces_endpoint == "http://collector:4318/v1/traces"

    def test_benchmark_mode_enables_tracing(self, reload_telemetry_config):
        assert reload_telemetry_config({"NAILA_BENCHMARK": "1"}).enabled

    def test_empty_endpoint_is_unset(self, reload_telemetry_config):
        config = reload_telemetry_config({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": ""})
        assert config.traces_endpoint is None
