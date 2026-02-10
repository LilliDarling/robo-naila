from __future__ import annotations

import asyncio
import logging
import threading
import time

from aiohttp import web

log = logging.getLogger(__name__)

_LOG_INTERVAL = 10  # seconds
_HEALTH_PORT = 8081


class DeviceMetrics:
    """Thread-safe counters and gauges for the pi-audio device.

    Accessed from both the PortAudio audio thread and the asyncio thread.
    Uses simple locking; at 50 fps the contention is negligible.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()

        # Counters (lifetime totals).
        self.mic_frames_captured: int = 0
        self.mic_frames_sent: int = 0
        self.mic_frames_dropped: int = 0
        self.tts_frames_received: int = 0
        self.tts_frames_played: int = 0
        self.tts_frames_dropped: int = 0
        self.aec_frames_processed: int = 0
        self.connections: int = 0
        self.connection_failures: int = 0

        # Gauges (point-in-time).
        self.mic_queue_depth: int = 0
        self.speaker_queue_depth: int = 0
        self.connected: bool = False
        self.audio_callback_duration_ms: float = 0.0

        # Delta tracking for periodic log (rates per interval).
        self._prev_mic_sent: int = 0
        self._prev_mic_dropped: int = 0
        self._prev_tts_received: int = 0
        self._prev_tts_played: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start

    def snapshot(self) -> dict:
        return {
            "connected": self.connected,
            "uptime_seconds": int(self.uptime_seconds),
            "mic_frames_sent": self.mic_frames_sent,
            "mic_frames_dropped": self.mic_frames_dropped,
            "tts_frames_received": self.tts_frames_received,
            "tts_frames_played": self.tts_frames_played,
            "mic_queue_depth": self.mic_queue_depth,
            "speaker_queue_depth": self.speaker_queue_depth,
            "audio_callback_duration_ms": round(self.audio_callback_duration_ms, 2),
            "connections": self.connections,
            "connection_failures": self.connection_failures,
        }

    def log_periodic(self) -> None:
        """Log deltas since last call, then reset delta counters."""
        d_sent = self.mic_frames_sent - self._prev_mic_sent
        d_drop = self.mic_frames_dropped - self._prev_mic_dropped
        d_tts_r = self.tts_frames_received - self._prev_tts_received
        d_tts_p = self.tts_frames_played - self._prev_tts_played

        self._prev_mic_sent = self.mic_frames_sent
        self._prev_mic_dropped = self.mic_frames_dropped
        self._prev_tts_received = self.tts_frames_received
        self._prev_tts_played = self.tts_frames_played

        log.info(
            "metrics | mic_sent=%d mic_dropped=%d tts_received=%d tts_played=%d "
            "mic_q=%d spk_q=%d cb_ms=%.1f connected=%s",
            d_sent,
            d_drop,
            d_tts_r,
            d_tts_p,
            self.mic_queue_depth,
            self.speaker_queue_depth,
            self.audio_callback_duration_ms,
            self.connected,
        )


# ------------------------------------------------------------------
# Health HTTP server
# ------------------------------------------------------------------


async def _health_handler(request: web.Request) -> web.Response:
    metrics: DeviceMetrics = request.app["metrics"]
    snap = metrics.snapshot()
    status = 200 if metrics.connected else 503
    return web.json_response(snap, status=status)


async def start_health_server(metrics: DeviceMetrics) -> web.AppRunner:
    app = web.Application()
    app["metrics"] = metrics
    app.router.add_get("/health", _health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", _HEALTH_PORT)
    await site.start()
    log.info("health server listening on :%d", _HEALTH_PORT)
    return runner


# ------------------------------------------------------------------
# Periodic logging task
# ------------------------------------------------------------------


async def periodic_log(metrics: DeviceMetrics, pipeline=None) -> None:
    """Log metrics every _LOG_INTERVAL seconds. Runs as a background task."""
    while True:
        await asyncio.sleep(_LOG_INTERVAL)
        # Update queue gauges from pipeline if available.
        if pipeline is not None:
            metrics.mic_queue_depth = pipeline.mic_queue_depth
            metrics.speaker_queue_depth = pipeline.speaker_queue_depth
        metrics.log_periodic()
