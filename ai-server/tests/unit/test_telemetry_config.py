"""Trace settings are resolved once, including benchmark opt-in."""
import importlib

import pytest


@pytest.mark.parametrize('env,enabled,endpoint', [
    ({}, False, None),
    ({'NAILA_OTEL_ENABLED': '1', 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT': 'http://collector:4318/v1/traces'}, True, 'http://collector:4318/v1/traces'),
    ({'NAILA_BENCHMARK': '1'}, True, None),
    ({'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT': ''}, False, None),
])
def test_telemetry_config(monkeypatch, env, enabled, endpoint):
    import config.benchmark as benchmark
    from config.telemetry import TelemetryConfig
    for key in ('NAILA_BENCHMARK', 'NAILA_OTEL_ENABLED', 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(benchmark)
    config = TelemetryConfig.from_env()
    assert (config.enabled, config.traces_endpoint) == (enabled, endpoint)
