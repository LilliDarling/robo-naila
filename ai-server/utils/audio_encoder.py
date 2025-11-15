"""Audio encoding utilities for TTS output formatting"""

import io
import wave
from typing import Optional

import numpy as np

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    AudioSegment = None  # type: ignore[assignment,misc]
    HAS_PYDUB = False

from utils import get_logger


logger = get_logger(__name__)


class AudioEncoder:
    """Encode audio samples to various output formats

    Supports:
    - WAV (uncompressed)
    - MP3 (compressed, requires pydub)
    - OGG (compressed, requires pydub)
    - RAW PCM (raw audio data)
    """

    @staticmethod
    def encode_wav(audio_samples: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio as WAV format

        Args:
            audio_samples: Audio data as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz

        Returns:
            WAV file bytes
        """
        # Convert float32 to int16
        audio_int16 = (audio_samples * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()

    @staticmethod
    def encode_mp3(
        audio_samples: np.ndarray,
        sample_rate: int,
        bitrate: int = 128
    ) -> bytes:
        """Encode audio as MP3 format

        Args:
            audio_samples: Audio data as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            bitrate: MP3 bitrate in kbps

        Returns:
            MP3 file bytes
        """
        if not HAS_PYDUB or AudioSegment is None:
            logger.error("pydub_not_installed", format="mp3", suggestion="Run: uv add pydub")
            raise ImportError("pydub is required for MP3 encoding")

        # Convert float32 to int16
        audio_int16 = (audio_samples * 32767).astype(np.int16)

        # Create AudioSegment from raw data
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # Mono
        )

        # Export as MP3
        mp3_buffer = io.BytesIO()
        audio_segment.export(
            mp3_buffer,
            format="mp3",
            bitrate=f"{bitrate}k"
        )

        return mp3_buffer.getvalue()

    @staticmethod
    def encode_ogg(
        audio_samples: np.ndarray,
        sample_rate: int,
        quality: int = 6
    ) -> bytes:
        """Encode audio as OGG Vorbis format

        Args:
            audio_samples: Audio data as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            quality: OGG quality level (0-10, higher is better)

        Returns:
            OGG file bytes
        """
        if not HAS_PYDUB or AudioSegment is None:
            logger.error("pydub_not_installed", format="ogg", suggestion="Run: uv add pydub")
            raise ImportError("pydub is required for OGG encoding")

        # Convert float32 to int16
        audio_int16 = (audio_samples * 32767).astype(np.int16)

        # Create AudioSegment from raw data
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # Mono
        )

        # Export as OGG
        ogg_buffer = io.BytesIO()
        audio_segment.export(
            ogg_buffer,
            format="ogg",
            codec="libvorbis",
            parameters=["-q:a", str(quality)]
        )

        return ogg_buffer.getvalue()

    @staticmethod
    def encode_raw(audio_samples: np.ndarray) -> bytes:
        """Return raw PCM audio data

        Args:
            audio_samples: Audio data as numpy array (float32, -1.0 to 1.0)

        Returns:
            Raw PCM bytes (int16)
        """
        # Convert float32 to int16
        audio_int16 = (audio_samples * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def encode(
        audio_samples: np.ndarray,
        sample_rate: int,
        format: str,
        bitrate: Optional[int] = None,
        quality: Optional[int] = None
    ) -> bytes:
        """Encode audio to specified format

        Args:
            audio_samples: Audio data as numpy array (float32, -1.0 to 1.0)
            sample_rate: Sample rate in Hz
            format: Output format ("wav", "mp3", "ogg", "raw")
            bitrate: MP3 bitrate in kbps (optional, for MP3)
            quality: OGG quality level (optional, for OGG)

        Returns:
            Encoded audio bytes

        Raises:
            ValueError: If format is unsupported
        """
        format_lower = format.lower()

        if format_lower == "mp3":
            return AudioEncoder.encode_mp3(audio_samples, sample_rate, bitrate or 128)
        elif format_lower == "ogg":
            return AudioEncoder.encode_ogg(audio_samples, sample_rate, quality or 6)
        elif format_lower == "raw":
            return AudioEncoder.encode_raw(audio_samples)
        elif format_lower == "wav":
            return AudioEncoder.encode_wav(audio_samples, sample_rate)
        else:
            raise ValueError(f"Unsupported audio format: {format}")
