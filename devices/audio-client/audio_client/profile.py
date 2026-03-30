from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AudioCodec(Enum):
    PCM_S16LE = "pcm_s16le"
    OPUS = "opus"


@dataclass(frozen=True)
class DeviceProfile:
    """Audio capabilities and defaults for a device type.

    Used by the client to configure its audio pipeline, and sent to the hub
    in the connect request so the server can adapt its output format.
    """

    name: str
    sample_rate: int
    channels: int
    frame_duration_ms: int
    preferred_output_sample_rate: int
    supported_output_codecs: tuple[AudioCodec, ...]
    aec_enabled: bool


PROFILES: dict[str, DeviceProfile] = {
    "mac": DeviceProfile(
        name="mac",
        sample_rate=48_000,
        channels=1,
        frame_duration_ms=20,
        preferred_output_sample_rate=48_000,
        supported_output_codecs=(AudioCodec.OPUS, AudioCodec.PCM_S16LE),
        aec_enabled=False,
    ),
}

DEFAULT_PROFILE = "mac"


def get_profile(name: str) -> DeviceProfile:
    try:
        return PROFILES[name]
    except KeyError:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown device profile {name!r} (available: {available})")
