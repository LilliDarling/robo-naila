"""TTS Service for text-to-speech synthesis using Piper TTS"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from piper.config import SynthesisConfig

import numpy as np

try:
    from piper import PiperVoice
    HAS_PIPER = True
except ImportError:
    PiperVoice = None
    HAS_PIPER = False

from config import tts as tts_config
from utils.audio_encoder import AudioEncoder
from services.base import BaseAIService
from utils.text_normalizer import TextNormalizer
from managers.voice import VoiceManager, VoiceConfig
from utils.emotion_presets import get_emotion_parameters, list_emotions
from utils.ssml_parser import SSMLParser
from utils import get_logger


logger = get_logger(__name__)


class LRUCache(OrderedDict):
    """LRU cache with size limit for phrase caching"""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            # Move to end (most recently used)
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            # Remove oldest item
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key):
        # Move to end on access
        self.move_to_end(key)
        return super().__getitem__(key)


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
        self._phrase_cache = LRUCache(maxsize=tts_config.MAX_CACHED_PHRASES)

        # Multi-voice support
        self.voice_manager = VoiceManager()
        self.multi_voice_enabled = tts_config.ENABLE_MULTI_VOICE

        # SSML support (lazy initialization)
        self._ssml_parser: Optional[SSMLParser] = None
        self.ssml_enabled = tts_config.ENABLE_SSML

    @property
    def ssml_parser(self) -> SSMLParser:
        """Lazy-load SSML parser on first use"""
        if self._ssml_parser is None:
            self._ssml_parser = SSMLParser()
        return self._ssml_parser

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "TTS"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        logger.info(
            "tts_configuration",
            voice=tts_config.VOICE,
            sample_rate=tts_config.SAMPLE_RATE,
            output_format=tts_config.OUTPUT_FORMAT
        )

    async def _load_model_impl(self) -> bool:
        """TTS-specific model loading logic"""
        try:
            # Check if piper-tts is available
            if not HAS_PIPER or PiperVoice is None:
                logger.error("piper_not_installed", suggestion="Run: uv add piper-tts")
                return False

            # Type assertion for type checker
            assert PiperVoice is not None, "PiperVoice should be available after import check"

            if self.multi_voice_enabled:
                # Multi-voice mode: Register and load available voices
                logger.info("tts_multi_voice_enabled")

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
                    logger.error("tts_default_voice_load_failed", voice_name=default_voice)
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
                        "tts_model_file_not_found",
                        model_path=str(self.model_path),
                        error=str(e),
                        suggestion="Ensure the model is downloaded"
                    )
                    return False
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cuda" in error_msg or "gpu" in error_msg:
                        logger.error(
                            "tts_gpu_error",
                            error=str(e),
                            suggestion="Try setting ENABLE_GPU=false for CPU-only mode"
                        )
                    else:
                        logger.error("tts_runtime_error", error=str(e), error_type=type(e).__name__)
                    return False
                except Exception as e:
                    logger.error("tts_piper_load_failed", error=str(e), error_type=type(e).__name__)
                    return False

            # Warm up model if enabled (cache common phrases)
            if tts_config.CACHE_COMMON_PHRASES:
                await self._warmup_and_cache()

            return True

        except Exception as e:
            logger.error("tts_model_loading_exception", error=str(e), error_type=type(e).__name__)
            return False

    async def _warmup_and_cache(self):
        """Warm up model and cache common phrases (parallelized)"""
        try:
            logger.info("tts_warmup_starting")
            start_time = time.time()

            # Prepare synthesis tasks for parallel execution
            async def synthesize_phrase(phrase: str):
                """Synthesize and return phrase with cache key"""
                try:
                    normalized = self._preprocess_text(phrase)
                    audio_samples = await self._synthesize_to_audio(normalized)
                    cache_key = self._build_cache_key(normalized)
                    return (cache_key, audio_samples, phrase)
                except Exception as e:
                    logger.warning("phrase_cache_failed", phrase=phrase, error=str(e), error_type=type(e).__name__)
                    return None

            # Synthesize all phrases in parallel
            tasks = [synthesize_phrase(phrase) for phrase in tts_config.COMMON_PHRASES]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Cache successful results
            for result in results:
                if result and not isinstance(result, Exception):
                    cache_key, audio_samples, phrase = result
                    self._phrase_cache[cache_key] = audio_samples

            warmup_time = time.time() - start_time
            logger.info(
                "tts_warmup_completed",
                warmup_time_seconds=round(warmup_time, 2),
                cached_phrases=len(self._phrase_cache)
            )

        except Exception as e:
            logger.warning("tts_warmup_failed", error=str(e), severity="non-critical")

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
        """Synthesize speech from text (supports SSML)

        Args:
            text: Text to synthesize (plain text or SSML markup)
            output_format: Output format (wav, mp3, ogg, raw). Defaults to config.
            length_scale: Speaking rate (1.0 = normal). Defaults to config.
            noise_scale: Pitch variation. Defaults to config.
            noise_w: Energy variation. Defaults to config.
            voice: Voice name to use (only if multi-voice enabled). Defaults to current voice.
            emotion: Emotion/tone preset (happy, sad, calm, etc.). Overrides length/noise params.

        Returns:
            AudioData with synthesized speech
        """
        # Check if input is SSML and SSML is enabled
        if self.ssml_enabled and self.ssml_parser.is_ssml(text):
            return await self._synthesize_ssml(text, output_format)

        # Plain text synthesis (original path)
        # Apply emotion preset if specified
        if emotion:
            if emotion_params := get_emotion_parameters(emotion):
                # Emotion preset overrides individual parameters
                length_scale = emotion_params["length_scale"]
                noise_scale = emotion_params["noise_scale"]
                noise_w = emotion_params["noise_w"]
                logger.debug("emotion_preset_applied", emotion=emotion)
            else:
                logger.warning("unknown_emotion_preset", emotion=emotion, fallback="defaults")

        # Handle voice switching if multi-voice is enabled
        if voice and self.multi_voice_enabled:
            await self._switch_voice(voice)

        if not self.is_ready or self.model is None:
            logger.error("tts_model_not_loaded")
            return self._empty_result()

        try:
            start_time = time.time()

            # Validate input
            if not text or not text.strip():
                logger.warning("tts_empty_text")
                return self._empty_result()

            if len(text) > tts_config.MAX_TEXT_LENGTH:
                logger.warning(
                    "tts_text_too_long",
                    text_length=len(text),
                    max_length=tts_config.MAX_TEXT_LENGTH,
                    action="truncating"
                )
                text = text[:tts_config.MAX_TEXT_LENGTH]

            # Normalize text
            normalized_text = self._preprocess_text(text)

            if tts_config.LOG_SYNTHESES:
                logger.debug("tts_synthesizing", text=normalized_text)

            # Check cache for common phrases
            cache_key = self._build_cache_key(
                normalized_text,
                voice=voice,
                emotion=emotion,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w
            )
            if cache_key in self._phrase_cache:
                logger.debug("tts_using_cached_audio", text=normalized_text)
                audio_samples = self._phrase_cache[cache_key]
            else:
                # Synthesize audio
                audio_samples = await self._synthesize_to_audio(
                    normalized_text,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w=noise_w
                )
                # Cache the result
                self._phrase_cache[cache_key] = audio_samples

            # Calculate duration
            duration_ms = int((len(audio_samples) / tts_config.SAMPLE_RATE) * 1000)

            # Encode to requested format
            output_format = output_format or tts_config.OUTPUT_FORMAT
            audio_bytes = await self._encode_audio(audio_samples, output_format)

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
            logger.error("tts_synthesis_failed", error=str(e), error_type=type(e).__name__)
            return self._empty_result()

    def _group_segments_by_voice(self, segments):
        """Group SSML segments by voice to enable batch synthesis

        Args:
            segments: List of SSMLSegment objects

        Returns:
            List of tuples (voice_name, segments_for_voice)
        """
        if not self.multi_voice_enabled:
            # Single voice mode - all segments use same voice
            return [(None, segments)]

        # Group consecutive segments with same voice
        groups = []
        current_voice = None
        current_group = []

        for segment in segments:
            segment_voice = segment.voice or self.voice_manager.get_current_voice_name()

            if segment_voice != current_voice and current_group:
                # Voice changed, save current group
                groups.append((current_voice, current_group))
                current_group = []

            current_voice = segment_voice
            current_group.append(segment)

        # Add final group
        if current_group:
            groups.append((current_voice, current_group))

        return groups

    async def _synthesize_ssml(self, ssml_text: str, output_format: Optional[str] = None) -> AudioData:
        """Synthesize speech from SSML markup

        Args:
            ssml_text: SSML markup text
            output_format: Output format. Defaults to config.

        Returns:
            AudioData with synthesized speech
        """
        try:
            start_time = time.time()

            # Parse SSML into segments
            segments = self.ssml_parser.parse(ssml_text)

            if not segments:
                logger.warning("No segments parsed from SSML")
                return self._empty_result()

            logger.debug("ssml_segments_parsed", segment_count=len(segments))

            # Group segments by voice for batch synthesis
            voice_groups = self._group_segments_by_voice(segments)

            # Synthesize each segment (with parallel synthesis for same-voice segments)
            all_audio_samples = []
            total_duration_ms = 0
            sample_rate = tts_config.SAMPLE_RATE

            for voice_name, group_segments in voice_groups:
                # Switch voice once per group
                if voice_name and self.multi_voice_enabled:
                    await self._switch_voice(voice_name)

                # Prepare synthesis tasks for this voice group
                synthesis_tasks = []
                segment_indices = []

                for i, segment in enumerate(group_segments):
                    if segment.text.strip():
                        # Prepare parameters
                        length_scale = segment.length_scale
                        noise_scale = segment.noise_scale
                        noise_w = segment.noise_w

                        if segment.emotion:
                            if emotion_params := get_emotion_parameters(segment.emotion):
                                length_scale = emotion_params["length_scale"]
                                noise_scale = emotion_params["noise_scale"]
                                noise_w = emotion_params["noise_w"]

                        normalized_text = self._preprocess_text(segment.text)
                        synthesis_tasks.append(
                            self._synthesize_to_audio(
                                normalized_text,
                                length_scale=length_scale,
                                noise_scale=noise_scale,
                                noise_w=noise_w
                            )
                        )
                        segment_indices.append((i, segment))

                # Synthesize all segments in parallel for this voice
                if synthesis_tasks:
                    audio_results = await asyncio.gather(*synthesis_tasks)

                    # Interleave results with pauses
                    result_idx = 0
                    for i, segment in enumerate(group_segments):
                        # Add pause before segment
                        if segment.pause_before > 0:
                            silence_samples = int(segment.pause_before * sample_rate)
                            all_audio_samples.append(np.zeros(silence_samples, dtype=np.float32))
                            total_duration_ms += int(segment.pause_before * 1000)

                        # Add synthesized audio
                        if segment.text.strip() and result_idx < len(audio_results):
                            audio_samples = audio_results[result_idx]
                            all_audio_samples.append(audio_samples)
                            segment_duration = int((len(audio_samples) / sample_rate) * 1000)
                            total_duration_ms += segment_duration
                            result_idx += 1

                        # Add pause after segment
                        if segment.pause_after > 0:
                            silence_samples = int(segment.pause_after * sample_rate)
                            all_audio_samples.append(np.zeros(silence_samples, dtype=np.float32))
                            total_duration_ms += int(segment.pause_after * 1000)

            # Concatenate all segments
            if all_audio_samples:
                final_audio = np.concatenate(all_audio_samples)
            else:
                logger.warning("No audio samples generated from SSML")
                return self._empty_result()

            # Encode to requested format
            output_format = output_format or tts_config.OUTPUT_FORMAT
            audio_bytes = await self._encode_audio(final_audio, output_format)

            # Calculate metrics
            synthesis_time_ms = int((time.time() - start_time) * 1000)

            # Extract plain text for logging
            plain_text = " ".join(seg.text for seg in segments if seg.text.strip())
            self._log_performance(total_duration_ms, synthesis_time_ms, plain_text[:50])

            # Determine voice name
            voice_name = (
                self.voice_manager.get_current_voice_name()
                if self.multi_voice_enabled
                else tts_config.VOICE
            )

            return AudioData(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                format=output_format,
                duration_ms=total_duration_ms,
                synthesis_time_ms=synthesis_time_ms,
                text=plain_text,
                voice=voice_name or "default",
                phonemes=None
            )

        except Exception as e:
            logger.error("ssml_synthesis_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
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

    def _build_cache_key(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None
    ) -> str:
        """Build cache key from text and synthesis parameters

        Args:
            text: Normalized text
            voice: Voice name
            emotion: Emotion preset
            length_scale: Speaking rate
            noise_scale: Pitch variation
            noise_w: Energy variation

        Returns:
            Cache key string
        """
        if tts_config.CACHE_INCLUDES_PARAMETERS:
            # Include parameters for more precise caching
            voice_key = voice or (
                self.voice_manager.get_current_voice_name()
                if self.multi_voice_enabled
                else tts_config.VOICE
            )
            emotion_key = emotion or "neutral"
            length_key = length_scale or tts_config.LENGTH_SCALE
            noise_key = noise_scale or tts_config.NOISE_SCALE
            noise_w_key = noise_w or tts_config.NOISE_W
            return f"{text.lower()}:{voice_key}:{emotion_key}:{length_key}:{noise_key}:{noise_w_key}"
        else:
            # Simple text-only caching (legacy behavior)
            return text.lower()

    @lru_cache(maxsize=256)
    def _normalize_text_cached(self, text: str) -> str:
        """Cache normalized text to avoid redundant processing

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text
        """
        return self.text_normalizer.normalize(text)

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
            text = self._normalize_text_cached(text)

        # Ensure proper sentence ending for better prosody
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    async def _encode_audio(self, audio_samples: np.ndarray, format: str) -> bytes:
        """Encode audio samples to specified format

        Args:
            audio_samples: Raw audio samples (float32)
            format: Output format (wav, mp3, ogg, raw)

        Returns:
            Encoded audio bytes
        """
        try:
            # MP3/OGG encoding uses ffmpeg (blocking I/O), run in executor
            if format.lower() in ('mp3', 'ogg'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self.audio_encoder.encode,
                    audio_samples,
                    tts_config.SAMPLE_RATE,
                    format,
                    tts_config.MP3_BITRATE,
                    tts_config.OGG_QUALITY
                )
            else:
                # WAV/RAW are fast, no executor needed
                return self.audio_encoder.encode(
                    audio_samples,
                    tts_config.SAMPLE_RATE,
                    format,
                    bitrate=tts_config.MP3_BITRATE,
                    quality=tts_config.OGG_QUALITY
                )
        except Exception as e:
            logger.error("audio_encoding_failed", format=format, error=str(e), error_type=type(e).__name__)
            # Fallback to WAV if encoding fails
            logger.info("Falling back to WAV format")
            return self.audio_encoder.encode_wav(audio_samples, tts_config.SAMPLE_RATE)

    def _log_performance(self, duration_ms: int, synthesis_time_ms: int, text: str):
        """Log performance metrics"""
        synthesis_time = synthesis_time_ms / 1000.0
        rtf = synthesis_time / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if tts_config.LOG_PERFORMANCE_METRICS:
            logger.info(
                "tts_synthesis_performance",
                synthesis_time_seconds=round(synthesis_time, 2),
                audio_duration_ms=duration_ms,
                rtf=round(rtf, 3),
                text_length=len(text)
            )

        if rtf > tts_config.WARNING_RTF_THRESHOLD:
            logger.warning("tts_slow_synthesis", rtf=round(rtf, 3), threshold=tts_config.WARNING_RTF_THRESHOLD)

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

            logger.info("audio_saved_to_file", file_path=file_path)
            return True

        except Exception as e:
            logger.error("audio_file_save_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
            return False

    def clear_cache(self):
        """Clear all caches (phrase and normalization)"""
        self._phrase_cache.clear()
        self._normalize_text_cached.cache_clear()
        logger.info("All caches cleared")

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
        if not self.voice_manager.is_voice_loaded(voice_name) and not await self.voice_manager.load_voice(voice_name, tts_config.ENABLE_GPU):
            logger.error("voice_load_failed", voice_name=voice_name)
            return False

        # Switch to voice
        if self.voice_manager.set_current_voice(voice_name):
            self.model = self.voice_manager.get_current_voice()
            logger.info("voice_switched", voice_name=voice_name)
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
