"""Trace export configuration"""

import os
from dataclasses import dataclass

from config import benchmark


@dataclass
class TelemetryConfig:
    enabled: bool = False
    traces_endpoint: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("NAILA_OTEL_ENABLED") == "1" or benchmark.ENABLED,
            traces_endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or None,
        )
