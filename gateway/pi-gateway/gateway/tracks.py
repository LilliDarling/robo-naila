import asyncio
import fractions

import numpy as np
from av import AudioFrame
from aiortc.mediastreams import MediaStreamTrack


class TTSOutputTrack(MediaStreamTrack):
    """
    Audio track that streams TTS output to device.
    Pulls audio from an asyncio queue.
    """
    
    kind = "audio"
    
    def __init__(self, queue: asyncio.Queue, sample_rate: int = 48000, channels: int = 1):
        super().__init__()
        self.queue = queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = 960  # 20ms at 48kHz
        
        # Timing
        self._timestamp = 0
        self._start_time = None
    
    async def recv(self) -> AudioFrame:
        """Get next audio frame for WebRTC."""
        # Try to get audio from queue, return silence if empty
        try:
            pcm = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pcm = np.zeros(self.frame_size, dtype=np.int16)
        
        # Ensure correct shape: (channels, samples)
        if pcm.ndim == 1:
            pcm = pcm.reshape(1, -1)
        
        # Create frame
        frame = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        
        self._timestamp += self.frame_size
        
        return frame


class LoopbackTrack(MediaStreamTrack):
    """
    Audio track that echoes received audio back.
    Useful for testing without AI server.
    """
    
    kind = "audio"
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = 960
        self.buffer: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._timestamp = 0
    
    async def feed(self, pcm: np.ndarray):
        """Feed audio to be echoed back."""
        try:
            self.buffer.put_nowait(pcm)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(pcm)
            except asyncio.QueueEmpty:
                pass
    
    async def recv(self) -> AudioFrame:
        """Get next audio frame."""
        try:
            pcm = self.buffer.get_nowait()
        except asyncio.QueueEmpty:
            pcm = np.zeros(self.frame_size, dtype=np.int16)
        
        if pcm.ndim == 1:
            pcm = pcm.reshape(1, -1)
        
        frame = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        
        self._timestamp += len(pcm.flatten())
        
        return frame
