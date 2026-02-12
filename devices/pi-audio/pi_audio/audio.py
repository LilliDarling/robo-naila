from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from .aec import EchoCanceller

if TYPE_CHECKING:
    from .metrics import DeviceMetrics

log = logging.getLogger(__name__)

# Maximum queued frames before we start dropping.
_MIC_QUEUE_MAX = 50
_SPEAKER_QUEUE_MAX = 50


class StoppedError(Exception):
    """Raised when a read is attempted on a stopped pipeline."""


class AudioPipeline:
    """Full-duplex audio via sounddevice.Stream + AEC.

    The PortAudio callback runs on a dedicated audio thread and handles
    both directions in one call, giving time-aligned mic capture and
    speaker output — exactly what AEC needs.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        channels: int,
        input_device: int | str | None,
        output_device: int | str | None,
        metrics: DeviceMetrics | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._channels = channels
        self._input_device = input_device
        self._output_device = output_device
        self._metrics = metrics

        self._aec = EchoCanceller(sample_rate, frame_size)

        # Thread-safe queues bridging PortAudio thread ↔ asyncio thread.
        self._mic_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=_MIC_QUEUE_MAX)
        self._speaker_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_SPEAKER_QUEUE_MAX)

        self._stream: sd.Stream | None = None
        self._silence = np.zeros(frame_size, dtype=np.int16)

    # ------------------------------------------------------------------
    # PortAudio callback (audio thread)
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        t0 = time.monotonic()

        if status:
            log.warning("portaudio status: %s", status)

        # 1. Speaker output — dequeue or silence.
        try:
            speaker_samples = self._speaker_queue.get_nowait()
            if self._metrics:
                self._metrics.tts_frames_played += 1
        except queue.Empty:
            speaker_samples = self._silence

        outdata[:, 0] = speaker_samples

        # 2. Mic capture + AEC.
        mic_raw = indata[:, 0].astype(np.int16)
        cleaned = self._aec.process(mic_raw, speaker_samples)

        if self._metrics:
            self._metrics.mic_frames_captured += 1
            self._metrics.aec_frames_processed += 1

        # 3. Push to mic queue (drop oldest on overflow).
        try:
            self._mic_queue.put_nowait(cleaned)
        except queue.Full:
            # Drop the oldest frame and push the new one.
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mic_queue.put_nowait(cleaned)
            except queue.Full:
                pass
            if self._metrics:
                self._metrics.mic_frames_dropped += 1

        if self._metrics:
            elapsed = (time.monotonic() - t0) * 1000.0
            self._metrics.audio_callback_duration_ms = elapsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stream = sd.Stream(
            samplerate=self._sample_rate,
            blocksize=self._frame_size,
            channels=self._channels,
            dtype="int16",
            device=(self._input_device, self._output_device),
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "audio pipeline started: sr=%d frame=%d",
            self._sample_rate,
            self._frame_size,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            # Unblock any executor thread waiting in read_mic_frame().
            self._mic_queue.put(None)
            log.info("audio pipeline stopped")

    async def read_mic_frame(self) -> np.ndarray:
        """Async bridge: block in executor until a mic frame is available."""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self._mic_queue.get)
        if frame is None:
            raise StoppedError
        return frame

    def queue_playback(self, samples: np.ndarray) -> None:
        """Non-blocking enqueue of TTS samples for playback."""
        try:
            self._speaker_queue.put_nowait(samples)
        except queue.Full:
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._speaker_queue.put_nowait(samples)
            except queue.Full:
                pass
            if self._metrics:
                self._metrics.tts_frames_dropped += 1

    @property
    def mic_queue_depth(self) -> int:
        return self._mic_queue.qsize()

    @property
    def speaker_queue_depth(self) -> int:
        return self._speaker_queue.qsize()
