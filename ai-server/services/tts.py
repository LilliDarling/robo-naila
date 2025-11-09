"""TTS Service for text-to-speech synthesis using Piper TTS"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import tts as tts_config
from services.audio_encoder import AudioEncoder
from services.base import BaseAIService
from services.text_normalizer import TextNormalizer


logger = logging.getLogger(__name__)


@dataclass
class AudioData:
    """Result of text-to-speech synthesis"""
    audio_bytes: bytes
    sample_rate: int
    format: str
    duration_ms: int
    synthesis_time_ms: int
    text: str
    phonemes: Optional[str] = None


class TTSService(BaseAIService):
    """Service for loading and running TTS inference with Piper"""

    def __init__(self):
        super().__init__(tts_config.MODEL_PATH)
        self.text_normalizer = TextNormalizer(language="en")
        self.audio_encoder = AudioEncoder()
        self._phrase_cache: Dict[str, np.ndarray] = {}

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "TTS"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        logger.info(
            f"Configuration: voice={tts_config.VOICE}, "
            f"sample_rate={tts_config.SAMPLE_RATE}Hz, "
            f"output_format={tts_config.OUTPUT_FORMAT}"
        )

    async def _load_model_impl(self) -> bool:
        """TTS-specific model loading logic"""
        try:
            # Import piper-tts
            try:
                from piper import PiperVoice
            except ImportError:
                logger.error("piper-tts not installed. Run: uv add piper-tts")
                return False

            # Load model (blocking, run in executor)
            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: PiperVoice.load(
                        str(self.model_path),
                        use_cuda=tts_config.ENABLE_GPU
                    )
                )
            except FileNotFoundError as e:
                logger.error(
                    f"Model file not found: {self.model_path}. "
                    f"Ensure the model is downloaded. Error: {e}"
                )
                return False
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg:
                    logger.error(
                        f"GPU error during model loading. "
                        f"Try setting ENABLE_GPU=false for CPU-only mode. "
                        f"Error: {e}"
                    )
                else:
                    logger.error(f"Runtime error loading model: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to load Piper model: {e}")
                return False

            # Warm up model if enabled (cache common phrases)
            if tts_config.CACHE_COMMON_PHRASES:
                await self._warmup_and_cache()

            return True

        except Exception as e:
            logger.error(f"Exception during model loading: {e}", exc_info=True)
            return False

    async def _warmup_and_cache(self):
        """Warm up model and cache common phrases"""
        try:
            logger.info("Warming up TTS model and caching common phrases...")
            start_time = time.time()

            for phrase in tts_config.COMMON_PHRASES:
                try:
                    # Synthesize and cache
                    audio_samples = await self._synthesize_to_audio(phrase)
                    self._phrase_cache[phrase.lower()] = audio_samples
                except Exception as e:
                    logger.warning(f"Failed to cache phrase '{phrase}': {e}")

            warmup_time = time.time() - start_time
            logger.info(
                f"Model warm-up completed in {warmup_time:.2f}s. "
                f"Cached {len(self._phrase_cache)} phrases."
            )

        except Exception as e:
            logger.warning(f"Model warm-up failed (non-critical): {e}")

    async def synthesize(
        self,
        text: str,
        output_format: Optional[str] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> AudioData:
        """Synthesize speech from text

        Args:
            text: Text to synthesize
            output_format: Output format (wav, mp3, ogg, raw). Defaults to config.
            length_scale: Speaking rate (1.0 = normal). Defaults to config.
            noise_scale: Pitch variation. Defaults to config.
            noise_w: Energy variation. Defaults to config.

        Returns:
            AudioData with synthesized speech
        """
        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot synthesize")
            return self._empty_result()

        try:
            start_time = time.time()

            # Validate input
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for synthesis")
                return self._empty_result()

            if len(text) > tts_config.MAX_TEXT_LENGTH:
                logger.warning(
                    f"Text too long ({len(text)} chars), truncating to {tts_config.MAX_TEXT_LENGTH}"
                )
                text = text[:tts_config.MAX_TEXT_LENGTH]

            # Normalize text
            normalized_text = self._preprocess_text(text)

            if tts_config.LOG_SYNTHESES:
                logger.debug(f"Synthesizing: '{normalized_text}'")

            # Check cache for common phrases
            cache_key = normalized_text.lower()
            if cache_key in self._phrase_cache:
                logger.debug(f"Using cached audio for: '{normalized_text}'")
                audio_samples = self._phrase_cache[cache_key]
            else:
                # Synthesize audio
                audio_samples = await self._synthesize_to_audio(
                    normalized_text,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w=noise_w
                )

            # Calculate duration
            duration_ms = int((len(audio_samples) / tts_config.SAMPLE_RATE) * 1000)

            # Encode to requested format
            output_format = output_format or tts_config.OUTPUT_FORMAT
            audio_bytes = self._encode_audio(audio_samples, output_format)

            # Calculate metrics
            synthesis_time_ms = int((time.time() - start_time) * 1000)
            self._log_performance(duration_ms, synthesis_time_ms, normalized_text)

            return AudioData(
                audio_bytes=audio_bytes,
                sample_rate=tts_config.SAMPLE_RATE,
                format=output_format,
                duration_ms=duration_ms,
                synthesis_time_ms=synthesis_time_ms,
                text=normalized_text,
                phonemes=None  # Could be populated if needed
            )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return self._empty_result()

    async def _synthesize_to_audio(
        self,
        text: str,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> np.ndarray:
        """Internal synthesis to audio samples

        Args:
            text: Normalized text to synthesize
            length_scale: Speaking rate
            noise_scale: Pitch variation
            noise_w: Energy variation

        Returns:
            Audio samples as numpy array (float32)
        """
        model = self.model
        assert model is not None, "Model should be loaded"

        # Use config defaults if not specified
        length_scale = length_scale or tts_config.LENGTH_SCALE
        noise_scale = noise_scale or tts_config.NOISE_SCALE
        noise_w = noise_w or tts_config.NOISE_W

        # Run synthesis in executor (blocking operation)
        loop = asyncio.get_event_loop()
        audio_samples = await loop.run_in_executor(
            None,
            lambda: model.synthesize(
                text,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                speaker_id=tts_config.SPEAKER_ID
            )
        )

        return audio_samples

    def _preprocess_text(self, text: str) -> str:
        """Preprocess and normalize text for synthesis

        Args:
            text: Raw text

        Returns:
            Normalized text ready for synthesis
        """
        # Strip whitespace
        text = text.strip()

        # Apply normalization if enabled
        if tts_config.NORMALIZE_NUMBERS or tts_config.NORMALIZE_DATES:
            text = self.text_normalizer.normalize(text)

        # Ensure proper sentence ending for better prosody
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def _encode_audio(self, audio_samples: np.ndarray, format: str) -> bytes:
        """Encode audio samples to specified format

        Args:
            audio_samples: Raw audio samples (float32)
            format: Output format (wav, mp3, ogg, raw)

        Returns:
            Encoded audio bytes
        """
        try:
            return self.audio_encoder.encode(
                audio_samples,
                tts_config.SAMPLE_RATE,
                format,
                bitrate=tts_config.MP3_BITRATE,
                quality=tts_config.OGG_QUALITY
            )
        except Exception as e:
            logger.error(f"Audio encoding failed for format '{format}': {e}")
            # Fallback to WAV if encoding fails
            logger.info("Falling back to WAV format")
            return self.audio_encoder.encode_wav(audio_samples, tts_config.SAMPLE_RATE)

    def _log_performance(self, duration_ms: int, synthesis_time_ms: int, text: str):
        """Log performance metrics"""
        synthesis_time = synthesis_time_ms / 1000.0
        rtf = synthesis_time / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if tts_config.LOG_PERFORMANCE_METRICS:
            logger.info(
                f"Synthesis: {synthesis_time:.2f}s, "
                f"audio={duration_ms}ms, "
                f"RTF={rtf:.3f}, "
                f"text_len={len(text)}"
            )

        if rtf > tts_config.WARNING_RTF_THRESHOLD:
            logger.warning(f"Slow synthesis: RTF={rtf:.3f} (threshold: {tts_config.WARNING_RTF_THRESHOLD})")

    def _empty_result(self) -> AudioData:
        """Return an empty audio result"""
        return AudioData(
            audio_bytes=b"",
            sample_rate=tts_config.SAMPLE_RATE,
            format=tts_config.OUTPUT_FORMAT,
            duration_ms=0,
            synthesis_time_ms=0,
            text=""
        )

    async def synthesize_to_file(self, text: str, file_path: str, output_format: Optional[str] = None) -> bool:
        """Synthesize speech and save to file

        Args:
            text: Text to synthesize
            file_path: Path to save audio file
            output_format: Output format (inferred from file extension if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Infer format from file extension if not specified
            if output_format is None:
                output_format = Path(file_path).suffix.lstrip('.').lower()
                if not output_format:
                    output_format = tts_config.OUTPUT_FORMAT

            # Synthesize
            audio_data = await self.synthesize(text, output_format=output_format)

            if not audio_data.audio_bytes:
                logger.error("Synthesis failed, no audio generated")
                return False

            # Save to file
            with open(file_path, 'wb') as f:
                f.write(audio_data.audio_bytes)

            logger.info(f"Audio saved to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio file: {e}", exc_info=True)
            return False

    def clear_cache(self):
        """Clear the phrase cache"""
        self._phrase_cache.clear()
        logger.info("Phrase cache cleared")

    def get_status(self) -> Dict:
        """Get current service status"""
        status = super().get_status()
        status.update({
            "voice": tts_config.VOICE,
            "sample_rate": tts_config.SAMPLE_RATE,
            "output_format": tts_config.OUTPUT_FORMAT,
            "cached_phrases": len(self._phrase_cache),
        })
        return status
