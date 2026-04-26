from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

import aiohttp

from .audio import AudioPipeline
from .config import DeviceConfig
from .metrics import DeviceMetrics, periodic_log, start_health_server
from .webrtc import WebRTCClient

log = logging.getLogger(__name__)

# Audio recovery tunables. The watchdog in metrics warns at 2.0s of callback
# silence; the recovery loop waits a little longer before acting so a one-off
# scheduling hiccup or buffer underrun doesn't trigger a teardown.
_RECOVERY_STALL_THRESHOLD = 3.0
_RECOVERY_CHECK_INTERVAL = 1.0
_RECOVERY_INITIAL_BACKOFF = 1.0
_RECOVERY_MAX_BACKOFF = 10.0
_RECOVERY_MAX_ATTEMPTS = 20


async def audio_recovery_loop(
    pipeline: AudioPipeline,
    metrics: DeviceMetrics,
    shutdown: asyncio.Event,
) -> None:
    """Restart the AudioPipeline if its PortAudio callback stops firing.

    On WSLg the underlying PulseAudio bridge can disappear at any time
    (Windows audio device changes, RDP renegotiation, WSLg restart). When it
    does, PortAudio's callback silently stops and `sd.Stream.stop()` blocks.
    This loop watches for the stall, tears down the wedged stream, and
    creates a fresh one. The WebRTC connection survives the gap because
    MicTrack and the TTS receive task hold the AudioPipeline by reference,
    not its underlying sd.Stream.
    """
    loop = asyncio.get_running_loop()
    while not shutdown.is_set():
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=_RECOVERY_CHECK_INTERVAL)
            return  # shutdown signalled
        except asyncio.TimeoutError:
            pass

        last = metrics.last_callback_monotonic
        if last == 0.0:
            continue  # callback never fired yet — startup, not a stall
        stale = time.monotonic() - last
        if stale < _RECOVERY_STALL_THRESHOLD:
            continue

        log.warning(
            "audio recovery: callback stalled for %.1fs, attempting pipeline restart",
            stale,
        )

        backoff = _RECOVERY_INITIAL_BACKOFF
        for attempt in range(1, _RECOVERY_MAX_ATTEMPTS + 1):
            if shutdown.is_set():
                return
            # Run the blocking restart in an executor with a hard timeout so a
            # truly unrecoverable PortAudio state (e.g. _terminate hanging)
            # can't deadlock this task.
            try:
                ok = await asyncio.wait_for(
                    loop.run_in_executor(None, pipeline.restart),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                log.warning("audio recovery: restart attempt %d timed out", attempt)
                ok = False
            except Exception:
                log.exception("audio recovery: restart attempt %d raised", attempt)
                ok = False

            if ok:
                # The restart returned True but the callback still has to
                # actually fire to confirm the new stream is alive. Wait
                # briefly and check.
                await asyncio.sleep(0.5)
                if metrics.last_callback_monotonic > last:
                    log.info("audio recovery: pipeline alive again after %d attempts", attempt)
                    break
                log.warning("audio recovery: stream restarted but callback still silent")

            try:
                await asyncio.wait_for(shutdown.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, _RECOVERY_MAX_BACKOFF)
        else:
            log.error(
                "audio recovery: gave up after %d attempts — restart the client manually",
                _RECOVERY_MAX_ATTEMPTS,
            )
            return


async def run(config: DeviceConfig) -> None:
    """Main reconnect loop with exponential backoff."""
    metrics = DeviceMetrics()
    pipeline = AudioPipeline(
        sample_rate=config.sample_rate,
        frame_size=config.samples_per_frame,
        channels=config.channels,
        input_device=config.input_device,
        output_device=config.output_device,
        metrics=metrics,
    )

    # Start health server + periodic metrics logger.
    health_runner = await start_health_server(metrics)
    log_task = asyncio.create_task(periodic_log(metrics, pipeline))

    # Shutdown flag set by signal handlers.
    # Second Ctrl+C force-exits via os._exit, bypassing asyncio cleanup.
    # sys.exit(1) inside a signal handler running on the asyncio loop raises
    # SystemExit through the runner — which then tries to await cleanup that
    # may itself be wedged (e.g. PortAudio.stop() blocked on a dead audio
    # device). os._exit terminates immediately, no cleanup, no traceback.
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal():
        if shutdown.is_set():
            log.warning("forced exit on second signal")
            os._exit(130)  # 128 + SIGINT
        log.info("shutdown requested (Ctrl+C again to force-exit)")
        shutdown.set()

    # Asyncio signal handlers are Unix-only. On Windows fall back to
    # signal.signal which fires on the main thread; we hop back onto the
    # event loop via call_soon_threadsafe so _handle_signal sees a normal
    # asyncio context. Some signals (SIGTERM behaviour) are not portable
    # on Windows — best-effort.
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)
    except NotImplementedError:
        def _signal_thunk(signum, _frame):
            loop.call_soon_threadsafe(_handle_signal)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _signal_thunk)
            except (ValueError, OSError):
                pass

    pipeline.start()
    # Start the audio-pipeline watchdog/recovery task now that pipeline is up
    # and the shutdown event exists. It restarts the sd.Stream if PortAudio
    # callbacks stop firing, without tearing down WebRTC.
    recovery_task = asyncio.create_task(audio_recovery_loop(pipeline, metrics, shutdown))
    delay = config.reconnect_delay

    async with aiohttp.ClientSession() as session:
        try:
            while not shutdown.is_set():
                client = WebRTCClient(
                    hub_url=config.hub_url,
                    device_id=config.device_id,
                    pipeline=pipeline,
                    session=session,
                    metrics=metrics,
                    profile=config.profile,
                )
                try:
                    await client.connect()
                    delay = config.reconnect_delay  # Reset backoff on success.
                    # Wait for disconnect or shutdown.
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(client.wait_closed()),
                            asyncio.create_task(shutdown.wait()),
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except Exception:
                    metrics.connection_failures += 1
                    log.exception("connection failed")
                finally:
                    await client.close()

                if shutdown.is_set():
                    break

                log.info("reconnecting in %.1fs", delay)
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=delay)
                    break  # Shutdown signalled during backoff.
                except asyncio.TimeoutError:
                    pass  # Backoff elapsed, retry.
                delay = min(delay * 2, config.max_reconnect_delay)
        finally:
            # PortAudio's stop()/close() are blocking C calls. When the audio
            # device dies underneath them (PulseAudio drops out on WSLg, USB
            # device unplugged, etc.) they can block indefinitely. Run them in
            # an executor with a hard timeout so Ctrl+C can always exit.
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, pipeline.stop),
                    timeout=3.0,
                )
            except asyncio.TimeoutError:
                log.warning("pipeline stop timed out — audio device may be wedged; exiting anyway")
            except Exception:
                log.exception("pipeline stop raised")

            log_task.cancel()
            recovery_task.cancel()
            try:
                await asyncio.wait_for(health_runner.cleanup(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            log.info("shutdown complete")


def cli() -> None:
    """Entry point: ``python -m audio_client`` or ``audio-client`` script."""
    config = DeviceConfig.from_env_and_args()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("starting audio-client: profile=%s hub=%s id=%s", config.profile.name, config.hub_url, config.device_id)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        log.info("interrupted, exiting")


if __name__ == "__main__":
    cli()
