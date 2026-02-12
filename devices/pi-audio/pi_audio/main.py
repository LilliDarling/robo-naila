from __future__ import annotations

import asyncio
import logging
import signal
import sys

import aiohttp

from .audio import AudioPipeline
from .config import DeviceConfig
from .metrics import DeviceMetrics, periodic_log, start_health_server
from .webrtc import WebRTCClient

log = logging.getLogger(__name__)


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
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    pipeline.start()
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
            pipeline.stop()
            log_task.cancel()
            await health_runner.cleanup()
            log.info("shutdown complete")


def cli() -> None:
    """Entry point: ``python -m pi_audio`` or ``pi-audio`` script."""
    config = DeviceConfig.from_env_and_args()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("starting pi-audio device: hub=%s id=%s", config.hub_url, config.device_id)
    asyncio.run(run(config))


if __name__ == "__main__":
    cli()
