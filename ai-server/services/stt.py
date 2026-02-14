"""STT Service for speech-to-text transcription using Whisper models"""
import re
import resampy
import soundfile as sf
from io import BytesIO
import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    WhisperModel = None
    HAS_FASTER_WHISPER = False

from config import stt as stt_config
from services.base import BaseAIService
from utils.retry import retry_on_failure
from utils.resource_pool import ResourcePool
from utils import get_logger


logger = get_logger(__name__)


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

    def _validate_model_path(self) -> bool:
        """Accept both local directories and model size strings like 'small.en'."""
        if self.model_path.exists():
            return True
        # faster_whisper accepts model size strings (e.g. "small.en") that it
        # downloads from HuggingFace — these aren't filesystem paths.
        model_str = str(self.model_path)
        return "/" not in model_str and "\\" not in model_str

    def _log_configuration(self):
        """Log model-specific configuration after successful load"""
        device = self._get_device()
        compute_type = self._get_compute_type()
        cpu_threads = self._get_thread_count(stt_config.THREADS)
        logger.info("stt_configuration", device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    async def _load_model_impl(self) -> bool:
        """STT-specific model loading logic"""
        try:
            # Check if faster-whisper is available
            if not HAS_FASTER_WHISPER or WhisperModel is None:
                logger.error("faster_whisper_not_installed", suggestion="Run: pip install faster-whisper")
                return False

            # Type assertion for type checker
            assert WhisperModel is not None, "WhisperModel should be available after import check"

            # Determine optimal settings
            device = self._get_device()
            compute_type = self._get_compute_type()
            cpu_threads = self._get_thread_count(stt_config.THREADS)

            # Resolve download root for HuggingFace model caching
            download_root = str(Path(stt_config.MODEL_DOWNLOAD_ROOT).resolve())

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
                        download_root=download_root,
                    )
                )
            except MemoryError as e:
                logger.error(
                    "stt_out_of_memory",
                    compute_type=compute_type,
                    device=device,
                    error=str(e),
                    suggestion="Try using a smaller model or setting COMPUTE_TYPE=int8"
                )
                return False
            except ValueError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg:
                    logger.error(
                        "stt_gpu_incompatibility",
                        error=str(e),
                        suggestion="Try setting DEVICE=cpu for CPU-only mode"
                    )
                else:
                    logger.error("stt_invalid_configuration", error=str(e))
                return False
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.error(
                        "stt_runtime_oom",
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        error=str(e),
                        suggestion="Consider using CPU mode or int8 compute type"
                    )
                else:
                    logger.error("stt_runtime_error", error=str(e), error_type=type(e).__name__)
                return False

            # Initialize resource pool
            self._pool = ResourcePool(
                max_concurrent=stt_config.MAX_CONCURRENT_REQUESTS,
                timeout=stt_config.POOL_TIMEOUT_SECONDS
            )
            logger.info("resource_pool_initialized", max_concurrent=stt_config.MAX_CONCURRENT_REQUESTS)

            # Warm up the model if enabled
            if stt_config.ENABLE_WARMUP:
                await self._warmup_model()

            return True

        except Exception as e:
            logger.error("stt_model_loading_exception", error=str(e), error_type=type(e).__name__)
            return False

    async def _warmup_model(self):
        """Warm up the model with a dummy transcription to reduce first-inference latency"""
        try:
            logger.info("stt_warmup_starting")
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
            logger.info("stt_warmup_completed", warmup_time_seconds=round(warmup_time, 2))

        except Exception as e:
            logger.warning("stt_warmup_failed", error=str(e), severity="non-critical")

    def _get_device(self) -> str:
        """Determine optimal device based on hardware"""
        if stt_config.DEVICE != "auto":
            return stt_config.DEVICE

        # Auto-detect based on hardware
        if self.hardware_info and self.hardware_info.get('acceleration') == 'cuda':
            logger.info("stt_device_selected", device="cuda", reason="CUDA detected")
            return "cuda"

        logger.info("stt_device_selected", device="cpu", reason="No GPU detected")
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
            logger.error("stt_model_not_loaded")
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
                logger.error("audio_validation_failed", error=error_msg)
                return self._empty_result()

            audio_array, sample_rate, duration_ms = await self._preprocess_audio(audio_data, format)

            if stt_config.LOG_AUDIO_INFO:
                logger.debug("audio_preprocessed", duration_ms=duration_ms, sample_rate=sample_rate, sample_count=len(audio_array))

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
            logger.error("transcription_failed", error=str(e), error_type=type(e).__name__)
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
            logger.error("stt_model_not_loaded_batch")
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
                logger.warning("batch_preprocessing_all_failed")
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
                    logger.error("batch_item_failed", item_index=idx, error=str(transcription), error_type=type(transcription).__name__)
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
            rtf = total_time/total_audio_duration if total_audio_duration > 0 else 0
            logger.info(
                "batch_transcription_completed",
                item_count=len(audio_batch),
                total_time_seconds=round(total_time, 2),
                total_audio_seconds=round(total_audio_duration, 1),
                rtf=round(rtf, 2)
            )

            return results

        except Exception as e:
            logger.error("batch_transcription_failed", error=str(e), error_type=type(e).__name__)
            return [self._empty_result() for _ in audio_batch]

    async def _preprocess_audio_safe(
        self, audio_data: bytes, format: str
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """Safely preprocess audio, returning None on failure"""
        try:
            is_valid, error_msg = self._validate_audio(audio_data, format)
            if not is_valid:
                logger.warning("audio_validation_failed_safe", error=error_msg)
                return None, 0, 0

            return await self._preprocess_audio(audio_data, format)
        except Exception as e:
            logger.warning("audio_preprocessing_failed_safe", error=str(e), error_type=type(e).__name__)
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
                "stt_transcription_performance",
                transcription_time_seconds=round(transcription_time, 2),
                audio_duration_ms=duration_ms,
                rtf=round(rtf, 2),
                confidence=round(avg_confidence, 2)
            )

        if rtf > stt_config.WARNING_RTF_THRESHOLD:
            logger.warning("stt_slow_transcription", rtf=round(rtf, 2), threshold=stt_config.WARNING_RTF_THRESHOLD)

        if stt_config.LOG_TRANSCRIPTIONS:
            logger.debug("stt_transcription_text", text=transcribed_text)

    def _build_result(self, text: str, language: str, confidence: float,
                      duration_ms: int, transcription_time_ms: int,
                      segments: List[Dict]) -> TranscriptionResult:
        """Build final transcription result, checking confidence threshold"""
        # Check confidence threshold
        if confidence < stt_config.MIN_CONFIDENCE:
            logger.warning("stt_low_confidence", confidence=round(confidence, 2), threshold=stt_config.MIN_CONFIDENCE)
            if stt_config.REJECT_LOW_CONFIDENCE:
                logger.info("stt_rejecting_low_confidence", threshold=stt_config.MIN_CONFIDENCE)
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
                logger.debug("audio_converted_to_mono")

            # Check if resampling needed
            correct_sample_rate = sample_rate == stt_config.SAMPLE_RATE
            if not correct_sample_rate:
                needs_conversion = True
                original_rate = sample_rate
                audio_array = resampy.resample(audio_array, sample_rate, stt_config.SAMPLE_RATE)
                sample_rate = stt_config.SAMPLE_RATE
                logger.debug("audio_resampled", from_hz=original_rate, to_hz=stt_config.SAMPLE_RATE)

            # Log optimization when no conversion needed
            if not needs_conversion:
                logger.debug("audio_no_conversion_needed", format="16kHz_mono")

            # Calculate duration
            duration_ms = int((len(audio_array) / sample_rate) * 1000)

            # Validate duration
            if duration_ms < stt_config.MIN_DURATION_MS:
                logger.warning("audio_too_short", duration_ms=duration_ms, min_duration_ms=stt_config.MIN_DURATION_MS)

            if duration_ms > stt_config.MAX_DURATION_MS:
                logger.warning("audio_too_long_truncating", duration_ms=duration_ms, max_duration_ms=stt_config.MAX_DURATION_MS)
                max_samples = int((stt_config.MAX_DURATION_MS / 1000.0) * sample_rate)
                audio_array = audio_array[:max_samples]
                duration_ms = stt_config.MAX_DURATION_MS

            return audio_array, sample_rate, duration_ms

        except Exception as e:
            logger.error("audio_preprocessing_exception", error=str(e), error_type=type(e).__name__)
            raise

    def _clean_transcription(self, text: str) -> str:
        """Clean up transcribed text"""
        if stt_config.STRIP_WHITESPACE:
            text = text.strip()

        if stt_config.NORMALIZE_WHITESPACE:
            # Replace multiple spaces/newlines with single space
            text = re.sub(r'\s+', ' ', text)

        if stt_config.CAPITALIZE_FIRST and text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Apply length limits
        if len(text) > stt_config.MAX_TEXT_LENGTH:
            logger.warning("transcription_truncated", original_length=len(text), truncated_to=stt_config.MAX_TEXT_LENGTH)
            text = text[:stt_config.MAX_TEXT_LENGTH]

        return text

    async def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe audio file"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error("audio_file_not_found", file_path=file_path)
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
            logger.error("file_transcription_failed", file_path=file_path, error=str(e), error_type=type(e).__name__)
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
