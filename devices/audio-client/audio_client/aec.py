from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Filter tail: 100ms at 48kHz = 4800 samples.
# Covers typical desk-robot acoustic path.
_FILTER_TAIL_MS = 100


class EchoCanceller:
    """Wraps speexdsp.EchoState for acoustic echo cancellation."""

    def __init__(self, sample_rate: int, frame_size: int) -> None:
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._filter_tail = sample_rate * _FILTER_TAIL_MS // 1000
        self._echo: object | None = None
        self._init_echo()

    def _init_echo(self) -> None:
        try:
            import speexdsp

            self._echo = speexdsp.EchoState.create(
                self._frame_size, self._filter_tail
            )
            log.info(
                "AEC initialised: frame=%d tail=%d",
                self._frame_size,
                self._filter_tail,
            )
        except Exception:
            log.warning("speexdsp unavailable — AEC disabled", exc_info=True)
            self._echo = None

    def process(
        self, mic: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Cancel echo from *mic* using *reference* (speaker output).

        Both arrays must be int16, length == frame_size.
        Returns cleaned int16 array of the same length.
        """
        if self._echo is None:
            return mic

        mic_bytes = mic.astype(np.int16).tobytes()
        ref_bytes = reference.astype(np.int16).tobytes()
        out_bytes = self._echo.process(mic_bytes, ref_bytes)
        return np.frombuffer(out_bytes, dtype=np.int16).copy()
