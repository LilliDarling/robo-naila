from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError

from .http import exchange_sdp

if TYPE_CHECKING:
    from .audio import AudioPipeline
    from .metrics import DeviceMetrics

log = logging.getLogger(__name__)

# Must match hub/src/webrtc.rs constants.
OPUS_SAMPLE_RATE = 48_000
OPUS_FRAME_MS = 20
SAMPLES_PER_FRAME = 960


class MicTrack(MediaStreamTrack):
    """Outbound audio track that reads cleaned mic frames from the AudioPipeline.

    aiortc calls recv() to pull the next audio frame for Opus encoding + RTP.
    """

    kind = "audio"

    def __init__(self, pipeline: AudioPipeline, metrics: DeviceMetrics | None = None) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._metrics = metrics
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        samples = await self._pipeline.read_mic_frame()

        # Build av.AudioFrame: s16 mono 48kHz, 960 samples.
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = OPUS_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = f"1/{OPUS_SAMPLE_RATE}"
        self._pts += SAMPLES_PER_FRAME

        if self._metrics:
            self._metrics.mic_frames_sent += 1

        return frame


class WebRTCClient:
    """Manages a single aiortc peer connection to the hub."""

    def __init__(
        self,
        hub_url: str,
        device_id: str,
        pipeline: AudioPipeline,
        metrics: DeviceMetrics | None = None,
    ) -> None:
        self._hub_url = hub_url
        self._device_id = device_id
        self._pipeline = pipeline
        self._metrics = metrics
        self._pc: RTCPeerConnection | None = None
        self._closed = asyncio.Event()
        self._recv_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Create offer, exchange SDP with hub, start streaming."""
        # No ICE servers — local network only, matching hub RTCConfiguration.
        self._pc = RTCPeerConnection(configuration={"iceServers": []})

        # Outbound: mic → hub.
        mic_track = MicTrack(self._pipeline, self._metrics)
        self._pc.addTrack(mic_track)

        # Inbound: hub TTS → speaker.
        @self._pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            log.info("received remote track: kind=%s", track.kind)
            if track.kind == "audio":
                self._recv_task = asyncio.create_task(self._recv_tts(track))

        # Connection state monitoring.
        @self._pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = self._pc.connectionState
            log.info("connection state: %s", state)
            if state in ("failed", "closed"):
                self._closed.set()

        # SDP exchange.
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # Wait for ICE gathering to complete (matching hub behaviour).
        await self._wait_ice_gathering()

        answer_sdp = await exchange_sdp(
            self._hub_url, self._device_id, self._pc.localDescription.sdp
        )
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self._pc.setRemoteDescription(answer)

        if self._metrics:
            self._metrics.connections += 1
            self._metrics.connected = True

        log.info("WebRTC connected to hub")

    async def _wait_ice_gathering(self) -> None:
        """Block until ICE gathering is complete."""
        if self._pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def _check() -> None:
            if self._pc.iceGatheringState == "complete":
                done.set()

        await done.wait()

    async def _recv_tts(self, track: MediaStreamTrack) -> None:
        """Receive inbound TTS audio frames and enqueue for playback."""
        log.info("TTS receive loop started")
        try:
            while True:
                frame: av.AudioFrame = await track.recv()
                # Decode to int16 numpy array.
                pcm = frame.to_ndarray().flatten().astype(np.int16)
                self._pipeline.queue_playback(pcm)
                if self._metrics:
                    self._metrics.tts_frames_received += 1
        except MediaStreamError:
            log.info("TTS track ended")
        except Exception:
            log.exception("TTS receive error")
        finally:
            self._closed.set()

    async def wait_closed(self) -> None:
        """Wait until the connection drops."""
        await self._closed.wait()

    async def close(self) -> None:
        if self._metrics:
            self._metrics.connected = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        log.info("WebRTC connection closed")
