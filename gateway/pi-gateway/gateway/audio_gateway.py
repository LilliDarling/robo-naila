import asyncio
import logging

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

from config import settings
from .session import DeviceSession
from .vad import create_vad
from .tracks import TTSOutputTrack, LoopbackTrack

logger = logging.getLogger(__name__)


class AudioGateway:
    """
    Main gateway class that manages WebRTC connections to devices.
    """
    
    def __init__(self, loopback_mode: bool = False):
        self.sessions: dict[str, DeviceSession] = {}
        self.vad = create_vad(
            mode=settings.audio.vad_mode,
            threshold=settings.audio.vad_threshold,
            sample_rate=settings.audio.sample_rate
        )
        self.loopback_mode = loopback_mode
        
        # Callbacks
        self.on_utterance: callable = None  # Called when speech ends
        self.on_ice_candidate: callable = None  # Called for ICE candidates
        
        # TCP client to AI server (set externally)
        self.ai_client = None
    
    async def create_session(self, device_id: str) -> tuple[DeviceSession, str]:
        """
        Create a new WebRTC session for a device.
        Returns session and SDP offer.
        """
        # Clean up existing session if any
        if device_id in self.sessions:
            await self.sessions[device_id].close()
        
        pc = RTCPeerConnection()
        session = DeviceSession(device_id=device_id, pc=pc)
        
        # Set up output track
        if self.loopback_mode:
            output_track = LoopbackTrack(sample_rate=settings.audio.sample_rate)
            session._loopback_track = output_track
        else:
            output_track = TTSOutputTrack(
                queue=session.tts_queue,
                sample_rate=settings.audio.sample_rate
            )
        
        pc.addTrack(output_track)
        
        # Handle incoming audio track
        @pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info(f"[{device_id}] Audio track received")
                asyncio.create_task(self._handle_audio_track(session, track))
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connection_state():
            logger.info(f"[{device_id}] Connection state: {pc.connectionState}")
            if pc.connectionState == "connected":
                session.connected = True
            elif pc.connectionState in ("failed", "closed", "disconnected"):
                session.connected = False
        
        # Handle ICE candidates
        @pc.on("icecandidate")
        async def on_ice(candidate):
            if candidate and self.on_ice_candidate:
                await self.on_ice_candidate(device_id, candidate)
        
        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        self.sessions[device_id] = session
        logger.info(f"[{device_id}] Session created")
        
        return session, pc.localDescription.sdp
    
    async def handle_answer(self, device_id: str, sdp: str):
        """Process SDP answer from device."""
        session = self.sessions.get(device_id)
        if not session:
            logger.warning(f"[{device_id}] No session found for answer")
            return
        
        answer = RTCSessionDescription(sdp=sdp, type="answer")
        await session.pc.setRemoteDescription(answer)
        logger.info(f"[{device_id}] Answer processed")
    
    async def handle_ice_candidate(self, device_id: str, candidate_data: dict):
        """Process ICE candidate from device."""
        session = self.sessions.get(device_id)
        if not session:
            return
        
        if candidate_data.get("candidate"):
            candidate = RTCIceCandidate(
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                candidate=candidate_data["candidate"]
            )
            await session.pc.addIceCandidate(candidate)
    
    async def _handle_audio_track(self, session: DeviceSession, track):
        """Process incoming audio from device."""
        logger.info(f"[{session.device_id}] Starting audio processing")
        
        while True:
            try:
                frame = await track.recv()
            except Exception as e:
                logger.error(f"[{session.device_id}] Track recv error: {e}")
                break
            
            # Convert to PCM int16
            pcm = frame.to_ndarray().flatten().astype(np.int16)
            
            # Loopback mode: echo back immediately
            if self.loopback_mode and hasattr(session, '_loopback_track'):
                await session._loopback_track.feed(pcm)
                continue
            
            # VAD processing
            await self._process_vad(session, pcm)
    
    async def _process_vad(self, session: DeviceSession, pcm: np.ndarray):
        """Voice activity detection and utterance buffering."""
        is_speech = self.vad.is_speech(pcm)
        
        # Interruption detection: user speaks while TTS playing
        if is_speech and session.tts_playing:
            logger.info(f"[{session.device_id}] Interruption detected")
            session.clear_tts_queue()
        
        if is_speech:
            session.is_speaking = True
            session.silence_frames = 0
            session.append_audio(pcm)
        else:
            if session.is_speaking:
                session.silence_frames += 1
                session.append_audio(pcm)  # Include trailing silence
                
                # Check if utterance ended (300ms silence)
                if session.silence_frames >= settings.audio.silence_frames:
                    await self._handle_utterance_end(session)
    
    async def _handle_utterance_end(self, session: DeviceSession):
        """Called when VAD detects end of utterance."""
        audio = session.get_buffered_audio()
        
        if len(audio) < settings.audio.sample_rate * 0.3:  # Less than 300ms
            logger.debug(f"[{session.device_id}] Utterance too short, ignoring")
            return
        
        logger.info(f"[{session.device_id}] Utterance complete: {len(audio)} samples")
        
        if self.on_utterance:
            await self.on_utterance(session.device_id, audio)
    
    async def send_tts_audio(self, device_id: str, pcm: np.ndarray):
        """Send TTS audio to device."""
        session = self.sessions.get(device_id)
        if not session or not session.connected:
            logger.warning(f"[{device_id}] Cannot send TTS: not connected")
            return
        
        await session.queue_tts_audio(pcm, chunk_size=settings.audio.frame_size)
    
    async def close_session(self, device_id: str):
        """Close a device session."""
        session = self.sessions.pop(device_id, None)
        if session:
            await session.close()
            logger.info(f"[{device_id}] Session closed")
    
    async def close_all(self):
        """Close all sessions."""
        for device_id in list(self.sessions.keys()):
            await self.close_session(device_id)
