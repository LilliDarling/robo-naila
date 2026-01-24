import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from aiortc import RTCPeerConnection


@dataclass
class DeviceSession:
    """Represents a connected device's WebRTC session."""
    
    device_id: str
    pc: RTCPeerConnection
    
    # Audio buffering
    audio_buffer: np.ndarray = field(default=None)
    buffer_pos: int = 0
    
    # VAD state
    is_speaking: bool = False
    silence_frames: int = 0
    
    # TTS playback queue
    tts_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tts_playing: bool = False
    
    # Request tracking
    pending_requests: dict[str, Any] = field(default_factory=dict)
    
    # Session state
    connected: bool = False
    
    def __post_init__(self):
        if self.audio_buffer is None:
            # 30 seconds buffer at 48kHz mono
            max_samples = 48000 * 30
            self.audio_buffer = np.zeros(max_samples, dtype=np.int16)
    
    def reset_buffer(self):
        """Clear the audio buffer."""
        self.buffer_pos = 0
        self.is_speaking = False
        self.silence_frames = 0
    
    def append_audio(self, pcm: np.ndarray) -> bool:
        """
        Append audio to buffer.
        Returns True if buffer has space, False if full.
        """
        samples = len(pcm)
        if self.buffer_pos + samples > len(self.audio_buffer):
            return False
        
        self.audio_buffer[self.buffer_pos:self.buffer_pos + samples] = pcm
        self.buffer_pos += samples
        return True
    
    def get_buffered_audio(self) -> np.ndarray:
        """Get the buffered audio and reset."""
        audio = self.audio_buffer[:self.buffer_pos].copy()
        self.reset_buffer()
        return audio
    
    async def queue_tts_audio(self, pcm: np.ndarray, chunk_size: int = 960):
        """Queue TTS audio for playback in chunks."""
        self.tts_playing = True
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk
                padded = np.zeros(chunk_size, dtype=np.int16)
                padded[:len(chunk)] = chunk
                chunk = padded
            await self.tts_queue.put(chunk)
    
    def clear_tts_queue(self):
        """Clear pending TTS audio (for interruption)."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.tts_playing = False
    
    async def close(self):
        """Clean up session resources."""
        self.connected = False
        self.clear_tts_queue()
        if self.pc:
            await self.pc.close()
