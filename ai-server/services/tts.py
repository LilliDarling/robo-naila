"""TTS Service for text-to-speech synthesis using Kokoro ONNX"""

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from kokoro_onnx import Kokoro
    HAS_KOKORO = True
except ImportError:
    Kokoro = None
    HAS_KOKORO = False

from config import tts as tts_config
from services.base import BaseAIService
from utils.text_normalizer import TextNormalizer
from utils.resource_pool import ResourcePool
from utils import get_logger


logger = get_logger(__name__)


class TTSPhraseLRUCache(OrderedDict):
    """LRU cache with size limit for phrase caching"""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key):
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
    """Service for TTS inference with Kokoro ONNX"""

    def __init__(self):
        super().__init__(tts_config.MODEL_PATH)
        self.text_normalizer = TextNormalizer(language="en")
        self._phrase_cache = TTSPhraseLRUCache(maxsize=tts_config.MAX_CACHED_PHRASES)
        self._pool: Optional[ResourcePool] = None
        self._voices_path = tts_config.VOICES_PATH

    def _get_model_type(self) -> str:
        return "TTS"

    def _validate_model_path(self) -> bool:
        return Path(self.model_path).exists()

    def _log_configuration(self):
        logger.info(
            "tts_configuration",
            voice=tts_config.VOICE,
            sample_rate=tts_config.SAMPLE_RATE,
            speed=tts_config.SPEED,
        )

    async def _load_model_impl(self) -> bool:
        try:
            if not HAS_KOKORO or Kokoro is None:
                logger.error("kokoro_not_installed", suggestion="Run: uv add kokoro-onnx")
                return False

            voices_path = Path(self._voices_path)
            if not voices_path.exists():
                logger.error("voices_file_not_found", path=str(voices_path))
                return False

            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: Kokoro(str(self.model_path), str(voices_path)),
                )
            except FileNotFoundError as e:
                logger.error("tts_model_file_not_found", model_path=str(self.model_path), error=str(e))
                return False
            except Exception as e:
                logger.error("tts_load_failed", error=str(e), error_type=type(e).__name__)
                return False

            available = self.model.get_voices()
            logger.info("tts_voices_available", count=len(available), voices=available[:10])

            if tts_config.VOICE not in available:
                logger.warning("tts_voice_not_found", voice=tts_config.VOICE, fallback=available[0])

            # Warm up with common phrases
            if tts_config.CACHE_COMMON_PHRASES:
                await self._warmup_and_cache()

            self._pool = ResourcePool(
                max_concurrent=tts_config.MAX_CONCURRENT_REQUESTS,
                timeout=tts_config.POOL_TIMEOUT_SECONDS,
            )
            logger.info("resource_pool_initialized", max_concurrent=tts_config.MAX_CONCURRENT_REQUESTS)

            return True

        except Exception as e:
            logger.error("tts_model_loading_exception", error=str(e), error_type=type(e).__name__)
            return False

    async def _warmup_and_cache(self):
        try:
            logger.info("tts_warmup_starting")
            start_time = time.time()

            async def synthesize_phrase(phrase: str):
                try:
                    normalized = self._preprocess_text(phrase)
                    audio_samples, _ = await self._synthesize_to_audio(normalized)
                    cache_key = self._build_cache_key(normalized)
                    return (cache_key, audio_samples, phrase)
                except Exception as e:
                    logger.warning("phrase_cache_failed", phrase=phrase, error=str(e))
                    return None

            tasks = [synthesize_phrase(p) for p in tts_config.COMMON_PHRASES]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result and not isinstance(result, Exception):
                    cache_key, audio_samples, _ = result
                    self._phrase_cache[cache_key] = audio_samples

            warmup_time = time.time() - start_time
            logger.info("tts_warmup_completed", warmup_time_seconds=round(warmup_time, 2), cached_phrases=len(self._phrase_cache))

        except Exception as e:
            logger.warning("tts_warmup_failed", error=str(e), severity="non-critical")

    async def synthesize(
        self,
        text: str,
        output_format: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ) -> AudioData:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            output_format: Output format ("raw" for PCM, "wav" otherwise).
            voice: Voice name. Defaults to config voice.
            speed: Speaking rate. Defaults to config speed.

        Returns:
            AudioData with synthesized speech.
        """
        if not self.is_ready or self.model is None:
            logger.error("tts_model_not_loaded")
            return self._empty_result()

        try:
            start_time = time.time()

            if not text or not text.strip():
                logger.warning("tts_empty_text")
                return self._empty_result()

            if len(text) > tts_config.MAX_TEXT_LENGTH:
                logger.warning("tts_text_too_long", text_length=len(text), max_length=tts_config.MAX_TEXT_LENGTH)
                text = text[:tts_config.MAX_TEXT_LENGTH]

            normalized_text = self._preprocess_text(text)

            if tts_config.LOG_SYNTHESES:
                logger.debug("tts_synthesizing", text=normalized_text)

            # Check cache
            cache_key = self._build_cache_key(normalized_text, voice=voice, speed=speed)
            if cache_key in self._phrase_cache:
                logger.debug("tts_using_cached_audio", text=normalized_text)
                audio_samples = self._phrase_cache[cache_key]
            else:
                audio_samples, _ = await self._synthesize_to_audio(
                    normalized_text, voice=voice, speed=speed,
                )
                self._phrase_cache[cache_key] = audio_samples

            duration_ms = int((len(audio_samples) / tts_config.SAMPLE_RATE) * 1000)

            # Encode output
            output_format = output_format or "wav"
            audio_bytes = self._encode_audio(audio_samples, output_format)

            synthesis_time_ms = int((time.time() - start_time) * 1000)
            self._log_performance(duration_ms, synthesis_time_ms, normalized_text)

            return AudioData(
                audio_bytes=audio_bytes,
                sample_rate=tts_config.SAMPLE_RATE,
                format=output_format,
                duration_ms=duration_ms,
                synthesis_time_ms=synthesis_time_ms,
                text=normalized_text,
                voice=voice or tts_config.VOICE,
            )

        except Exception as e:
            logger.error("tts_synthesis_failed", error=str(e), error_type=type(e).__name__)
            return self._empty_result()

    async def _synthesize_to_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Internal synthesis returning (audio_samples_float32, sample_rate)."""
        if self._pool is None:
            return await self._synthesize_impl(text, voice, speed)
        async with self._pool:
            return await self._synthesize_impl(text, voice, speed)

    async def _synthesize_impl(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        model = self.model
        assert model is not None

        voice = voice or tts_config.VOICE
        speed = speed or tts_config.SPEED

        logger.info("tts_synthesizing_voice", voice=voice, speed=speed, text_len=len(text))

        loop = asyncio.get_event_loop()
        samples, sr = await loop.run_in_executor(
            None,
            lambda: model.create(text, voice=voice, speed=speed, lang=tts_config.LANGUAGE),
        )
        return samples, sr

    def _encode_audio(self, audio_samples: np.ndarray, fmt: str) -> bytes:
        """Encode float32 audio samples to bytes."""
        logger.info("tts_encode_debug",
            dtype=str(audio_samples.dtype),
            shape=audio_samples.shape,
            min=float(audio_samples.min()),
            max=float(audio_samples.max()),
            fmt=fmt,
        )
        # Kokoro outputs float32 in [-1, 1]. Convert to int16 PCM.
        scaled = audio_samples * 32767.0
        audio_int16 = np.clip(scaled, -32768, 32767).astype(np.int16)
        logger.info("tts_encode_post",
            scaled_range=f"[{scaled.min():.1f}, {scaled.max():.1f}]",
            int16_range=f"[{audio_int16.min()}, {audio_int16.max()}]",
            raw_bytes_len=len(audio_int16.tobytes()),
        )

        if fmt == "raw":
            return audio_int16.tobytes()

        # WAV format
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_config.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def _build_cache_key(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> str:
        if tts_config.CACHE_INCLUDES_PARAMETERS:
            v = voice or tts_config.VOICE
            s = speed or tts_config.SPEED
            return f"{text.lower()}:{v}:{s}"
        return text.lower()

    def _preprocess_text(self, text: str) -> str:
        text = text.strip()
        if tts_config.NORMALIZE_NUMBERS or tts_config.NORMALIZE_DATES:
            text = self.text_normalizer.normalize(text)
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def _log_performance(self, duration_ms: int, synthesis_time_ms: int, text: str):
        synthesis_time = synthesis_time_ms / 1000.0
        rtf = synthesis_time / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if tts_config.LOG_PERFORMANCE_METRICS:
            logger.info(
                "tts_synthesis_performance",
                synthesis_time_seconds=round(synthesis_time, 2),
                audio_duration_ms=duration_ms,
                rtf=round(rtf, 3),
                text_length=len(text),
            )

        if rtf > tts_config.WARNING_RTF_THRESHOLD:
            logger.warning("tts_slow_synthesis", rtf=round(rtf, 3), threshold=tts_config.WARNING_RTF_THRESHOLD)

    def _empty_result(self) -> AudioData:
        return AudioData(
            audio_bytes=b"",
            sample_rate=tts_config.SAMPLE_RATE,
            format="raw",
            duration_ms=0,
            synthesis_time_ms=0,
            text="",
        )

    def get_available_voices(self) -> list:
        if self.model is not None:
            return self.model.get_voices()
        return []

    def get_current_voice(self) -> str:
        return tts_config.VOICE

    def get_status(self) -> Dict:
        status = super().get_status()
        status.update({
            "voice": tts_config.VOICE,
            "sample_rate": tts_config.SAMPLE_RATE,
            "speed": tts_config.SPEED,
            "cached_phrases": len(self._phrase_cache),
            "available_voices": len(self.get_available_voices()),
        })
        if self._pool is not None:
            status["pool"] = self._pool.get_stats()
        return status

    def clear_cache(self):
        self._phrase_cache.clear()
        logger.info("tts_cache_cleared")
