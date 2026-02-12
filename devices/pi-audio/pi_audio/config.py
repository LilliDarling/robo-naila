from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceConfig:
    hub_url: str = "http://localhost:8080"
    device_id: str = "pi-audio-01"
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 48_000
    frame_duration_ms: int = 20
    channels: int = 1
    reconnect_delay: float = 3.0
    max_reconnect_delay: float = 30.0
    log_level: str = "INFO"

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000

    @classmethod
    def from_env_and_args(cls) -> DeviceConfig:
        parser = argparse.ArgumentParser(description="Pi Audio Device Client")
        parser.add_argument("--hub-url", default=None)
        parser.add_argument("--device-id", default=None)
        parser.add_argument("--input-device", default=None)
        parser.add_argument("--output-device", default=None)
        parser.add_argument("--sample-rate", type=int, default=None)
        parser.add_argument("--frame-duration-ms", type=int, default=None)
        parser.add_argument("--channels", type=int, default=None)
        parser.add_argument("--reconnect-delay", type=float, default=None)
        parser.add_argument("--max-reconnect-delay", type=float, default=None)
        parser.add_argument("--log-level", default=None)
        args = parser.parse_args()

        def _resolve(env_key: str, arg_val, default, coerce: type = str):
            if arg_val is not None:
                return arg_val
            env = os.environ.get(env_key)
            if env is not None:
                try:
                    return coerce(env)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid value for {env_key}: {env!r}") from exc
            return default

        def _resolve_device(env_key: str, arg_val):
            if arg_val is not None:
                try:
                    return int(arg_val)
                except ValueError:
                    return arg_val
            env = os.environ.get(env_key)
            if env is not None:
                try:
                    return int(env)
                except ValueError:
                    return env
            return None

        return cls(
            hub_url=_resolve("NAILA_HUB_URL", args.hub_url, cls.hub_url),
            device_id=_resolve("NAILA_DEVICE_ID", args.device_id, cls.device_id),
            input_device=_resolve_device("NAILA_INPUT_DEVICE", args.input_device),
            output_device=_resolve_device("NAILA_OUTPUT_DEVICE", args.output_device),
            sample_rate=_resolve("NAILA_SAMPLE_RATE", args.sample_rate, cls.sample_rate, int),
            frame_duration_ms=_resolve("NAILA_FRAME_DURATION_MS", args.frame_duration_ms, cls.frame_duration_ms, int),
            channels=_resolve("NAILA_CHANNELS", args.channels, cls.channels, int),
            reconnect_delay=_resolve("NAILA_RECONNECT_DELAY", args.reconnect_delay, cls.reconnect_delay, float),
            max_reconnect_delay=_resolve("NAILA_MAX_RECONNECT_DELAY", args.max_reconnect_delay, cls.max_reconnect_delay, float),
            log_level=_resolve("NAILA_LOG_LEVEL", args.log_level, cls.log_level),
        )
