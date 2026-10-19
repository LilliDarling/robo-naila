"""File replay and discard sink independent of production capture/playback."""
import asyncio
import hashlib
import io
import time
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from aiortc import AudioStreamTrack
from av import AudioFrame


@dataclass(frozen=True)
class Fixture:
    pcm: bytes
    sha256: str
    padding_samples: int

    @classmethod
    def load(cls, path):
        data = Path(path).read_bytes()
        with wave.open(io.BytesIO(data)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 48000, 'NONE'):
                raise ValueError('WAV must be mono PCM16 48 kHz')
            count = wav.getnframes()
            if not 24000 <= count <= 1440000:
                raise ValueError('WAV duration must be 0.5–30 seconds')
            pcm = wav.readframes(count)
            if len(pcm) != count * 2:
                raise ValueError('truncated WAV')
        padding = (-count) % 960
        return cls(pcm + bytes(padding * 2), hashlib.sha256(data).hexdigest(), padding)


class ReplayTrack(AudioStreamTrack):
    # aiortc 1.14 AudioStreamTrack.recv supplies silence/pacing, not file replay:
    # https://github.com/aiortc/aiortc/blob/1.14.0/src/aiortc/mediastreams.py
    def __init__(self, fixture, clock=time.monotonic, sleep=asyncio.sleep):
        super().__init__()
        self.fixture, self.clock, self.sleep = fixture, clock, sleep
        self.ready_at = None
        self.cursor = self.pts = 0
        self.next_at = clock()
        self.first = self.end = None
        self.finished = asyncio.Event()

    def connected(self):
        self.ready_at = self.clock() + .5

    async def recv(self):
        await self.sleep(max(0, self.next_at - self.clock()))
        now = self.clock()
        pcm = bytes(1920)
        scored = self.ready_at is not None and now >= self.ready_at and self.cursor < len(self.fixture.pcm)
        if scored:
            pcm = self.fixture.pcm[self.cursor:self.cursor + 1920]
        frame = AudioFrame(format='s16', layout='mono', samples=960)
        frame.planes[0].update(pcm)
        frame.sample_rate, frame.time_base, frame.pts = 48000, Fraction(1, 48000), self.pts
        # No await between cursor commit and handoff; cancellation consumes nothing.
        if scored:
            self.first = now if self.first is None else self.first
            self.cursor += 1920
            if self.cursor == len(self.fixture.pcm):
                self.end = now + .02
                self.finished.set()
        self.pts += 960
        self.next_at = max(self.next_at + .02, now + .02)
        return frame


class Sink:
    def __init__(self):
        self.first = None
        self.count = 0

    async def consume(self, track):
        while True:
            await track.recv()
            if self.first is None:
                self.first = time.monotonic()
            self.count += 1
