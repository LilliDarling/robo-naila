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
from services.voice_manager import VoiceManager, VoiceConfig
from services.emotion_presets import get_emotion_parameters, list_emotions


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
    voice: str = "default"
    phonemes: Optional[str] = None


class TTSService(BaseAIService):
    """Service for loading and running TTS inference with Piper"""

    def __init__(self):
        super().__init__(tts_config.MODEL_PATH)
        self.text_normalizer = TextNormalizer(language="en")
        self.audio_encoder = AudioEncoder()
        self._phrase_cache: Dict[str, np.ndarray] = {}

        # Multi-voice support
        self.voice_manager = VoiceManager()
        self.multi_voice_enabled = tts_config.ENABLE_MULTI_VOICE

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

            if self.multi_voice_enabled:
                # Multi-voice mode: Register and load available voices
                logger.info("Multi-voice mode enabled")

                # Register all available voices
                for voice_name, voice_config in tts_config.AVAILABLE_VOICES.items():
                    config = VoiceConfig(
                        name=voice_name,
                        model_path=Path(voice_config["model_path"]),
                        language=voice_config.get("language", "en_US"),
                        sample_rate=voice_config.get("sample_rate", 22050),
                        description=voice_config.get("description", ""),
                        speaker_id=voice_config.get("speaker_id", 0)
                    )
                    self.voice_manager.register_voice(config)

                # Load default voice
                default_voice = tts_config.DEFAULT_VOICE
                if not await self.voice_manager.load_voice(default_voice, tts_config.ENABLE_GPU):
                    logger.error(f"Failed to load default voice: {default_voice}")
                    return False

                # Set model to the current voice for compatibility
                self.model = self.voice_manager.get_current_voice()

            else:
                # Single-voice mode: Load model directly (legacy behavior)
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
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> AudioData:
        """Synthesize speech from text

        Args:
            text: Text to synthesize
            output_format: Output format (wav, mp3, ogg, raw). Defaults to config.
            length_scale: Speaking rate (1.0 = normal). Defaults to config.
            noise_scale: Pitch variation. Defaults to config.
            noise_w: Energy variation. Defaults to config.
            voice: Voice name to use (only if multi-voice enabled). Defaults to current voice.
            emotion: Emotion/tone preset (happy, sad, calm, etc.). Overrides length/noise params.

        Returns:
            AudioData with synthesized speech
        """
        # Apply emotion preset if specified
        if emotion:
            emotion_params = get_emotion_parameters(emotion)
            if emotion_params:
                # Emotion preset overrides individual parameters
                length_scale = emotion_params["length_scale"]
                noise_scale = emotion_params["noise_scale"]
                noise_w = emotion_params["noise_w"]
                logger.debug(f"Applied emotion preset: {emotion}")
            else:
                logger.warning(f"Unknown emotion preset: {emotion}, using defaults")

        # Handle voice switching if multi-voice is enabled
        if voice and self.multi_voice_enabled:
            await self._switch_voice(voice)

        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot synthesize")
            return self._empty_result()

        try:
            start_time = time.time()

            # Validate input
            if not text or not text.strip():
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

            # Determine voice name
            voice_name = voice or (
                self.voice_manager.get_current_voice_name()
                if self.multi_voice_enabled
                else tts_config.VOICE
            )

            return AudioData(
                audio_bytes=audio_bytes,
                sample_rate=tts_config.SAMPLE_RATE,
                format=output_format,
                duration_ms=duration_ms,
                synthesis_time_ms=synthesis_time_ms,
                text=normalized_text,
                voice=voice_name or "default",
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
        from piper.config import SynthesisConfig

        model = self.model
        assert model is not None, "Model should be loaded"

        # Use config defaults if not specified
        length_scale = length_scale or tts_config.LENGTH_SCALE
        noise_scale = noise_scale or tts_config.NOISE_SCALE
        noise_w = noise_w or tts_config.NOISE_W

        # Create synthesis config
        syn_config = SynthesisConfig(
            speaker_id=tts_config.SPEAKER_ID,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )

        # Run synthesis in executor (blocking operation)
        loop = asyncio.get_event_loop()

        def _synthesize():
            # Synthesize returns an iterable of AudioChunk objects
            audio_chunks = []
            for chunk in model.synthesize(text, syn_config):
                # AudioChunk has audio_int16_array which is already a numpy array
                audio_chunks.append(chunk.audio_int16_array)
            # Concatenate all chunks into a single array
            if audio_chunks:
                # Convert int16 to float32 in range [-1.0, 1.0]
                audio_int16 = np.concatenate(audio_chunks)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                return audio_float32
            return np.array([], dtype=np.float32)

        return await loop.run_in_executor(None, _synthesize)

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
                output_format = Path(file_path).suffix.lstrip('.').lower() or tts_config.OUTPUT_FORMAT

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

    async def _switch_voice(self, voice_name: str) -> bool:
        """Switch to a different voice

        Args:
            voice_name: Name of the voice to switch to

        Returns:
            True if successful, False otherwise
        """
        if not self.multi_voice_enabled:
            logger.warning("Multi-voice not enabled, cannot switch voice")
            return False

        # Check if voice is already current
        if self.voice_manager.get_current_voice_name() == voice_name:
            return True

        # Load voice if not already loaded
        if not self.voice_manager.is_voice_loaded(voice_name):
            if not await self.voice_manager.load_voice(voice_name, tts_config.ENABLE_GPU):
                logger.error(f"Failed to load voice: {voice_name}")
                return False

        # Switch to voice
        if self.voice_manager.set_current_voice(voice_name):
            self.model = self.voice_manager.get_current_voice()
            logger.info(f"Switched to voice: {voice_name}")
            return True

        return False

    async def load_voice(self, voice_name: str) -> bool:
        """Load an additional voice (multi-voice mode only)

        Args:
            voice_name: Name of the voice to load

        Returns:
            True if successful, False otherwise
        """
        if not self.multi_voice_enabled:
            logger.warning("Multi-voice not enabled")
            return False

        return await self.voice_manager.load_voice(voice_name, tts_config.ENABLE_GPU)

    def unload_voice(self, voice_name: str):
        """Unload a voice (multi-voice mode only)

        Args:
            voice_name: Name of the voice to unload
        """
        if not self.multi_voice_enabled:
            logger.warning("Multi-voice not enabled")
            return

        self.voice_manager.unload_voice(voice_name)

    def get_available_voices(self) -> list:
        """Get list of available voice names

        Returns:
            List of voice names
        """
        if self.multi_voice_enabled:
            return self.voice_manager.get_registered_voices()
        else:
            return [tts_config.VOICE]

    def get_loaded_voices(self) -> list:
        """Get list of currently loaded voice names

        Returns:
            List of voice names
        """
        if self.multi_voice_enabled:
            return self.voice_manager.get_loaded_voices()
        else:
            return [tts_config.VOICE] if self.is_ready else []

    def get_current_voice(self) -> str:
        """Get name of current active voice

        Returns:
            Voice name
        """
        if self.multi_voice_enabled:
            return self.voice_manager.get_current_voice_name() or tts_config.DEFAULT_VOICE
        else:
            return tts_config.VOICE

    def get_available_emotions(self) -> list[str]:
        """Get list of available emotion presets

        Returns:
            List of emotion names
        """
        return list_emotions()

    def get_status(self) -> Dict:
        """Get current service status"""
        status = super().get_status()

        if self.multi_voice_enabled:
            # Multi-voice status
            voice_status = self.voice_manager.get_status()
            status.update({
                "multi_voice_enabled": True,
                "current_voice": voice_status["current_voice"],
                "loaded_voices": voice_status["loaded_voices"],
                "available_voices": voice_status["registered_voices"],
                "sample_rate": tts_config.SAMPLE_RATE,
                "output_format": tts_config.OUTPUT_FORMAT,
                "cached_phrases": len(self._phrase_cache),
            })
        else:
            # Single-voice status (legacy)
            status.update({
                "multi_voice_enabled": False,
                "voice": tts_config.VOICE,
                "sample_rate": tts_config.SAMPLE_RATE,
                "output_format": tts_config.OUTPUT_FORMAT,
                "cached_phrases": len(self._phrase_cache),
            })

        return status
