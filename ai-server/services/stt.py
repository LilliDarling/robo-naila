"""STT Service for speech-to-text transcription using Whisper models"""

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import stt as stt_config
from services.base import BaseAIService
from services.resource_pool import ResourcePool


logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: Optional[int] = None, delay: Optional[float] = None):
    """Decorator to retry async functions on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts (uses config default if None)
        delay: Initial delay between retries in seconds (uses config default if None)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = max_retries if max_retries is not None else stt_config.MAX_RETRIES
            retry_delay = delay if delay is not None else stt_config.RETRY_DELAY_SECONDS

            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < retries:
                        # Exponential backoff: delay * 2^attempt
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{retries + 1}): {e}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {retries + 1} attempts: {e}"
                        )

            # If all retries failed, raise the last exception
            raise last_exception

        return wrapper
    return decorator


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    language: str
    confidence: float
    duration_ms: int
    transcription_time_ms: int
    segments: Optional[List[Dict]] = None


class STTService(BaseAIService):
    """Service for loading and running STT inference with resource pooling"""

    def __init__(self):
        super().__init__(stt_config.MODEL_PATH)
        self._pool: Optional[ResourcePool] = None

    def _get_model_type(self) -> str:
        """Get the model type name for logging"""
        return "STT"

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        device = self._get_device()
        compute_type = self._get_compute_type()
        cpu_threads = self._get_thread_count(stt_config.THREADS)
        logger.info(f"Configuration: device={device}, compute_type={compute_type}, cpu_threads={cpu_threads}")

    async def _load_model_impl(self) -> bool:
        """STT-specific model loading logic"""
        try:
            # Import faster-whisper
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.error("faster-whisper not installed. Run: pip install faster-whisper")
                return False

            # Determine optimal settings
            device = self._get_device()
            compute_type = self._get_compute_type()
            cpu_threads = self._get_thread_count(stt_config.THREADS)

            # Load model (this is blocking, so run in executor)
            loop = asyncio.get_event_loop()
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperModel(
                        str(self.model_path),
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        num_workers=1,
                    )
                )
            except MemoryError as e:
                logger.error(
                    f"Out of memory while loading model. "
                    f"Try using a smaller model or setting COMPUTE_TYPE=int8. "
                    f"Current config: compute_type={compute_type}, device={device}. "
                    f"Error: {e}"
                )
                return False
            except ValueError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg:
                    logger.error(
                        f"GPU incompatibility detected. "
                        f"Try setting DEVICE=cpu for CPU-only mode. "
                        f"Error: {e}"
                    )
                else:
                    logger.error(f"Invalid model configuration: {e}")
                return False
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.error(
                        f"Out of memory error during model loading. "
                        f"Current config: device={device}, compute_type={compute_type}, "
                        f"cpu_threads={cpu_threads}. "
                        f"Consider using CPU mode or int8 compute type. Error: {e}"
                    )
                else:
                    logger.error(f"Runtime error loading model: {e}")
                return False

            # Initialize resource pool
            self._pool = ResourcePool(
                max_concurrent=stt_config.MAX_CONCURRENT_REQUESTS,
                timeout=stt_config.POOL_TIMEOUT_SECONDS
            )
            logger.info(f"Resource pool initialized with max {stt_config.MAX_CONCURRENT_REQUESTS} concurrent requests")

            # Warm up the model if enabled
            if stt_config.ENABLE_WARMUP:
                await self._warmup_model()

            return True

        except Exception as e:
            logger.error(f"Exception during model loading: {e}", exc_info=True)
            return False

    async def _warmup_model(self):
        """Warm up the model with a dummy transcription to reduce first-inference latency"""
        try:
            logger.info("Warming up STT model...")
            start_time = time.time()

            # Generate silent audio for warm-up
            duration_seconds = stt_config.WARMUP_DURATION_MS / 1000.0
            sample_count = int(stt_config.SAMPLE_RATE * duration_seconds)

            # Create very quiet white noise (better for warm-up than pure silence)
            warmup_audio = np.random.randn(sample_count).astype('float32') * 0.001

            # Run a quick transcription
            model = self.model
            assert model is not None, "Model should be loaded"

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: model.transcribe(
                    warmup_audio,
                    language=stt_config.LANGUAGE,
                    beam_size=1,  # Minimal beam for warm-up
                    vad_filter=False,  # Skip VAD for warm-up
                )
            )

            warmup_time = time.time() - start_time
            logger.info(f"Model warm-up completed in {warmup_time:.2f}s")

        except Exception as e:
            logger.warning(f"Model warm-up failed (non-critical): {e}")

    def _get_device(self) -> str:
        """Determine optimal device based on hardware"""
        if stt_config.DEVICE != "auto":
            return stt_config.DEVICE

        # Auto-detect based on hardware
        if self.hardware_info and self.hardware_info.get('acceleration') == 'cuda':
            logger.info("CUDA detected, using GPU acceleration")
            return "cuda"

        logger.info("No GPU detected, using CPU")
        return "cpu"

    def _get_compute_type(self) -> str:
        """Determine optimal compute type based on hardware and device"""
        device = self._get_device()

        # For CPU, use int8 for best performance
        if device == "cpu":
            return "int8"

        # For GPU, use float16 if available, otherwise int8
        if self.hardware_info and self.hardware_info.get('acceleration') == 'cuda':
            vram_gb = self.hardware_info.get('vram_gb', 0)
            if vram_gb >= 4:
                return "float16"

        return "int8"


    async def transcribe_audio(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio data to text with resource pooling"""
        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot transcribe")
            return self._empty_result()

        if self._pool is None:
            return await self._transcribe_impl(audio_data, format, language)
        async with self._pool:
            return await self._transcribe_impl(audio_data, format, language)

    async def _transcribe_impl(
        self,
        audio_data: bytes,
        format: str,
        language: Optional[str],
    ) -> TranscriptionResult:
        """Internal transcription implementation"""
        try:
            start_time = time.time()

            # Validate and preprocess audio
            is_valid, error_msg = self._validate_audio(audio_data, format)
            if not is_valid:
                logger.error(f"Audio validation failed: {error_msg}")
                return self._empty_result()

            audio_array, sample_rate, duration_ms = await self._preprocess_audio(audio_data, format)

            if stt_config.LOG_AUDIO_INFO:
                logger.debug(f"Audio preprocessed: {duration_ms}ms, {sample_rate}Hz, {len(audio_array)} samples")

            # Run transcription
            segments_list, info = await self._run_transcription(audio_array, language)

            # Process segments and calculate confidence
            segments, transcribed_text, avg_confidence = self._process_segments(segments_list)

            # Calculate performance metrics
            transcription_time_ms = int((time.time() - start_time) * 1000)
            self._log_performance(duration_ms, transcription_time_ms, avg_confidence, transcribed_text)

            # Check confidence threshold and build result
            return self._build_result(
                transcribed_text, info.language, avg_confidence,
                duration_ms, transcription_time_ms, segments
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return self._empty_result()

    async def transcribe_batch(
        self,
        audio_batch: List[Tuple[bytes, str]],
        language: Optional[str] = None,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio chunks efficiently

        Args:
            audio_batch: List of (audio_data, format) tuples
            language: Optional language code for all audio chunks

        Returns:
            List of TranscriptionResult objects in the same order as input
        """
        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot transcribe batch")
            return [self._empty_result() for _ in audio_batch]

        if not audio_batch:
            return []

        try:
            start_time = time.time()
            results = []

            # Preprocess all audio in parallel
            preprocess_tasks = [
                self._preprocess_audio_safe(audio_data, fmt)
                for audio_data, fmt in audio_batch
            ]
            preprocessed = await asyncio.gather(*preprocess_tasks)

            # Filter out failed preprocessing
            valid_items = [
                (i, audio_array, sample_rate, duration_ms)
                for i, (audio_array, sample_rate, duration_ms) in enumerate(preprocessed)
                if audio_array is not None
            ]

            if not valid_items:
                logger.warning("All audio in batch failed preprocessing")
                return [self._empty_result() for _ in audio_batch]

            # Transcribe all valid audio
            transcription_tasks = [
                self._run_transcription(audio_array, language)
                for _, audio_array, _, _ in valid_items
            ]
            transcriptions = await asyncio.gather(*transcription_tasks, return_exceptions=True)

            # Build results maintaining original order
            result_map = {}
            for (idx, _, _, duration_ms), transcription in zip(valid_items, transcriptions):
                if isinstance(transcription, Exception):
                    logger.error(f"Batch item {idx} failed: {transcription}")
                    result_map[idx] = self._empty_result()
                else:
                    segments_list, info = transcription
                    segments, text, confidence = self._process_segments(segments_list)
                    result_map[idx] = TranscriptionResult(
                        text=text,
                        language=info.language,
                        confidence=confidence,
                        duration_ms=duration_ms,
                        transcription_time_ms=0,  # Updated below
                        segments=segments,
                    )

            # Fill in results for failed preprocessing
            for i in range(len(audio_batch)):
                if i not in result_map:
                    result_map[i] = self._empty_result()

            results = [result_map[i] for i in range(len(audio_batch))]

            # Log batch performance
            total_time = time.time() - start_time
            total_audio_duration = sum(r.duration_ms for r in results) / 1000.0
            logger.info(
                f"Batch transcription: {len(audio_batch)} items, "
                f"{total_time:.2f}s total, "
                f"{total_audio_duration:.1f}s audio, "
                f"RTF={total_time/total_audio_duration:.2f}"
            )

            return results

        except Exception as e:
            logger.error(f"Batch transcription failed: {e}", exc_info=True)
            return [self._empty_result() for _ in audio_batch]

    async def _preprocess_audio_safe(
        self, audio_data: bytes, format: str
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """Safely preprocess audio, returning None on failure"""
        try:
            is_valid, error_msg = self._validate_audio(audio_data, format)
            if not is_valid:
                logger.warning(f"Audio validation failed: {error_msg}")
                return None, 0, 0

            return await self._preprocess_audio(audio_data, format)
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return None, 0, 0

    def _empty_result(self) -> TranscriptionResult:
        """Return an empty transcription result"""
        return TranscriptionResult(
            text="",
            language="",
            confidence=0.0,
            duration_ms=0,
            transcription_time_ms=0
        )

    @retry_on_failure()
    async def _run_transcription(self, audio_array: np.ndarray, language: Optional[str]):
        """Run the Whisper transcription model with retry logic"""
        model = self.model
        assert model is not None, "Model should be loaded"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_array,
                language=language or stt_config.LANGUAGE,
                beam_size=stt_config.BEAM_SIZE,
                best_of=stt_config.BEST_OF,
                temperature=stt_config.TEMPERATURE,
                vad_filter=stt_config.VAD_FILTER,
                vad_parameters={
                    "threshold": stt_config.VAD_THRESHOLD,
                    "min_silence_duration_ms": stt_config.MIN_SILENCE_DURATION_MS,
                    "speech_pad_ms": stt_config.SPEECH_PAD_MS,
                } if stt_config.VAD_FILTER else None,
            )
        )

    def _process_segments(self, segments_list) -> Tuple[List[Dict], str, float]:
        """Process transcription segments and calculate confidence"""
        segments = []
        text_parts = []
        total_logprob = 0.0
        segment_count = 0

        for segment in segments_list:
            logprob = getattr(segment, 'avg_logprob', 0.0)
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": logprob,
            })
            text_parts.append(segment.text)
            total_logprob += logprob
            segment_count += 1

        # Build text and calculate confidence
        transcribed_text = self._clean_transcription(" ".join(text_parts).strip())
        avg_confidence = self._calculate_confidence(total_logprob, segment_count)

        return segments, transcribed_text, avg_confidence

    def _calculate_confidence(self, total_logprob: float, segment_count: int) -> float:
        """Convert log probability to confidence score (0-1 scale)"""
        if segment_count == 0:
            return 0.0

        # Whisper's avg_logprob is typically in range [-1, 0]
        # Log probs closer to 0 are better (higher confidence)
        avg_logprob = total_logprob / segment_count
        return min(1.0, max(0.0, np.exp(avg_logprob)))

    def _log_performance(self, duration_ms: int, transcription_time_ms: int,
                         avg_confidence: float, transcribed_text: str):
        """Log performance metrics and transcription results"""
        transcription_time = transcription_time_ms / 1000.0
        rtf = transcription_time / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if stt_config.LOG_PERFORMANCE_METRICS:
            logger.info(
                f"Transcription: {transcription_time:.2f}s, "
                f"audio={duration_ms}ms, "
                f"RTF={rtf:.2f}, "
                f"confidence={avg_confidence:.2f}"
            )

        if rtf > stt_config.WARNING_RTF_THRESHOLD:
            logger.warning(f"Slow transcription: RTF={rtf:.2f} (slower than real-time)")

        if stt_config.LOG_TRANSCRIPTIONS:
            logger.debug(f"Transcription: {transcribed_text}")

    def _build_result(self, text: str, language: str, confidence: float,
                      duration_ms: int, transcription_time_ms: int,
                      segments: List[Dict]) -> TranscriptionResult:
        """Build final transcription result, checking confidence threshold"""
        # Check confidence threshold
        if confidence < stt_config.MIN_CONFIDENCE:
            logger.warning(f"Low confidence transcription: {confidence:.2f}")
            if stt_config.REJECT_LOW_CONFIDENCE:
                logger.info(f"Rejecting low confidence transcription (threshold: {stt_config.MIN_CONFIDENCE})")
                text = ""  # Reject the transcription

        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            duration_ms=duration_ms,
            transcription_time_ms=transcription_time_ms,
            segments=segments,
        )

    def _validate_audio(self, audio_data: bytes, format: str) -> Tuple[bool, str]:
        """Validate audio data"""
        if not audio_data:
            return False, "Empty audio data"

        if len(audio_data) < 100:  # Minimum viable audio size
            return False, "Audio data too small"

        if format.lower() not in stt_config.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {format}. Supported: {stt_config.SUPPORTED_FORMATS}"

        return True, ""

    async def _preprocess_audio(self, audio_data: bytes, format: str) -> Tuple[np.ndarray, int, int]:
        """Preprocess audio data for Whisper with optimization for correct format"""
        try:
            # Import audio processing libraries
            import soundfile as sf
            from io import BytesIO

            # Load audio using soundfile
            audio_io = BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_io, dtype='float32')

            # Track if any conversion is needed
            needs_conversion = False

            # Check if stereo to mono conversion needed
            is_mono = len(audio_array.shape) == 1
            if not is_mono:
                needs_conversion = True
                audio_array = audio_array.mean(axis=1)
                logger.debug("Converted stereo to mono")

            # Check if resampling needed
            correct_sample_rate = sample_rate == stt_config.SAMPLE_RATE
            if not correct_sample_rate:
                needs_conversion = True
                import resampy
                audio_array = resampy.resample(audio_array, sample_rate, stt_config.SAMPLE_RATE)
                sample_rate = stt_config.SAMPLE_RATE
                logger.debug(f"Resampled from {sample_rate}Hz to {stt_config.SAMPLE_RATE}Hz")

            # Log optimization when no conversion needed
            if not needs_conversion:
                logger.debug("Audio already in correct format (16kHz mono), skipping conversion")

            # Calculate duration
            duration_ms = int((len(audio_array) / sample_rate) * 1000)

            # Validate duration
            if duration_ms < stt_config.MIN_DURATION_MS:
                logger.warning(f"Audio too short: {duration_ms}ms < {stt_config.MIN_DURATION_MS}ms")

            if duration_ms > stt_config.MAX_DURATION_MS:
                logger.warning(f"Audio too long: {duration_ms}ms > {stt_config.MAX_DURATION_MS}ms, truncating")
                max_samples = int((stt_config.MAX_DURATION_MS / 1000.0) * sample_rate)
                audio_array = audio_array[:max_samples]
                duration_ms = stt_config.MAX_DURATION_MS

            return audio_array, sample_rate, duration_ms

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}", exc_info=True)
            raise

    def _clean_transcription(self, text: str) -> str:
        """Clean up transcribed text"""
        if stt_config.STRIP_WHITESPACE:
            text = text.strip()

        if stt_config.NORMALIZE_WHITESPACE:
            # Replace multiple spaces/newlines with single space
            import re
            text = re.sub(r'\s+', ' ', text)

        if stt_config.CAPITALIZE_FIRST and text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Apply length limits
        if len(text) > stt_config.MAX_TEXT_LENGTH:
            logger.warning(f"Transcription truncated from {len(text)} to {stt_config.MAX_TEXT_LENGTH} chars")
            text = text[:stt_config.MAX_TEXT_LENGTH]

        return text

    async def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe audio file"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Audio file not found: {file_path}")
                return TranscriptionResult(
                    text="",
                    language="",
                    confidence=0.0,
                    duration_ms=0,
                    transcription_time_ms=0
                )

            # Read file
            with open(path, 'rb') as f:
                audio_data = f.read()

            # Determine format from extension
            ext_format = path.suffix.lstrip('.').lower()

            return await self.transcribe_audio(audio_data, ext_format)

        except Exception as e:
            logger.error(f"File transcription failed: {e}", exc_info=True)
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration_ms=0,
                transcription_time_ms=0
            )

    def get_status(self) -> Dict:
        """Get current service status including pool stats"""
        status = super().get_status()
        status.update({
            "sample_rate": stt_config.SAMPLE_RATE,
            "supported_formats": stt_config.SUPPORTED_FORMATS,
            "language": stt_config.LANGUAGE,
        })

        # Add pool stats if pool is initialized
        if self._pool is not None:
            status["pool"] = self._pool.get_stats()

        return status
