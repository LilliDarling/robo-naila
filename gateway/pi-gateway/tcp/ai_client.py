import asyncio
import logging
import uuid

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class AIServerClient:
    """
    TCP client for communicating with AI server.
    
    Protocol:
        Request:  {device_id}|{audio_length}|{sample_rate}|{request_id}\n<pcm bytes>
        Response: {device_id}|{audio_length}|{sample_rate}|{request_id}\n<pcm bytes>
    """
    
    def __init__(self):
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    async def connect(self):
        """Connect to AI server."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                settings.ai_server.host,
                settings.ai_server.port
            )
            self._connected = True
            logger.info(
                f"Connected to AI server at "
                f"{settings.ai_server.host}:{settings.ai_server.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to AI server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from AI server."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self._connected = False
    
    async def process_audio(
        self,
        device_id: str,
        pcm: np.ndarray,
        sample_rate: int = 48000
    ) -> np.ndarray | None:
        """
        Send audio to AI server and get TTS response.
        
        Args:
            device_id: Device identifier
            pcm: Audio as int16 numpy array
            sample_rate: Sample rate
            
        Returns:
            TTS audio as int16 numpy array, or None on error
        """
        if not self._connected:
            logger.warning("AI server not connected")
            return None
        
        async with self._lock:
            try:
                return await self._send_receive(device_id, pcm, sample_rate)
            except Exception as e:
                logger.error(f"AI server communication error: {e}")
                self._connected = False
                return None
    
    async def _send_receive(
        self,
        device_id: str,
        pcm: np.ndarray,
        sample_rate: int
    ) -> np.ndarray | None:
        """Send request and receive response."""
        request_id = str(uuid.uuid4())[:8]
        pcm_bytes = pcm.tobytes()
        
        # Build header
        header = f"{device_id}|{len(pcm_bytes)}|{sample_rate}|{request_id}\n"
        
        # Send
        self.writer.write(header.encode() + pcm_bytes)
        await self.writer.drain()
        
        logger.debug(f"[{device_id}] Sent {len(pcm_bytes)} bytes to AI server")
        
        # Receive response
        response_header = await asyncio.wait_for(
            self.reader.readline(),
            timeout=settings.ai_server.timeout
        )
        
        if not response_header:
            logger.warning("Empty response from AI server")
            return None
        
        # Parse header
        parts = response_header.decode().strip().split("|")
        if len(parts) != 4:
            logger.warning(f"Invalid response header: {response_header}")
            return None
        
        resp_device_id, audio_len, resp_sample_rate, resp_request_id = parts
        audio_len = int(audio_len)
        
        if audio_len == 0:
            logger.debug(f"[{device_id}] Empty audio response (silence/noise)")
            return None
        
        # Read audio bytes
        audio_bytes = await asyncio.wait_for(
            self.reader.readexactly(audio_len),
            timeout=settings.ai_server.timeout
        )
        
        logger.debug(f"[{device_id}] Received {audio_len} bytes from AI server")
        
        # Convert to numpy array
        return np.frombuffer(audio_bytes, dtype=np.int16)
    
    async def reconnect(self, max_retries: int = 5, delay: float = 2.0):
        """Attempt to reconnect with backoff."""
        for attempt in range(max_retries):
            try:
                await self.connect()
                return True
            except Exception:
                if attempt < max_retries - 1:
                    logger.info(f"Reconnect attempt {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(delay * (attempt + 1))
        
        logger.error("Failed to reconnect to AI server")
        return False
