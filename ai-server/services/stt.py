"""STT Service for speech-to-text transcription using Whisper models"""

import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import stt as stt_config
from config.hardware_config import HardwareOptimizer


logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    language: str
    confidence: float
    duration_ms: int
    transcription_time_ms: int
    segments: Optional[List[Dict]] = None


class STTService:
    """Service for loading and running STT inference"""

    def __init__(self):
        self.model = None
        self.model_path = Path(stt_config.MODEL_PATH)
        self.is_ready = False
        self.hardware_info = None

    async def load_model(self, hardware_info: Optional[Dict] = None) -> bool:
        """Load the STT model with hardware optimization

        Args:
            hardware_info: Optional pre-detected hardware info. If None, will detect automatically.
        """
        if self.is_ready:
            logger.warning("Model already loaded")
            return True

        try:
            # Verify model file exists
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False

            logger.info(f"Loading STT model: {self.model_path}")
            start_time = time.time()

            # Use provided hardware info or detect if not provided
            if hardware_info is not None:
                self.hardware_info = hardware_info
                logger.debug("Using shared hardware detection")
            else:
                # Fallback to individual detection (for backward compatibility)
                hw_optimizer = HardwareOptimizer()
                self.hardware_info = {
                    'device_type': hw_optimizer.hardware_info.device_type,
                    'device_name': hw_optimizer.hardware_info.device_name,
                    'acceleration': hw_optimizer.hardware_info.device_type,
                    'cpu_count': os.cpu_count() or 4,
                    'vram_gb': hw_optimizer.hardware_info.memory_gb
                }
                logger.info(f"Hardware detected: {self.hardware_info}")

            # Import faster-whisper
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.error("faster-whisper not installed. Run: pip install faster-whisper")
                return False

            # Determine optimal settings
            device = self._get_device()
            compute_type = self._get_compute_type()
            cpu_threads = self._get_thread_count()

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
                self.is_ready = False
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
                self.is_ready = False
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
                self.is_ready = False
                return False

            load_time = time.time() - start_time
            self.is_ready = True

            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Configuration: device={device}, compute_type={compute_type}, cpu_threads={cpu_threads}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.is_ready = False
            return False

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

    def _get_thread_count(self) -> int:
        """Determine optimal thread count based on hardware"""
        if stt_config.THREADS > 0:
            return stt_config.THREADS

        # Auto-detect based on CPU cores
        if self.hardware_info and 'cpu_count' in self.hardware_info:
            cpu_count = self.hardware_info['cpu_count']
            # Use 75% of available cores, minimum 2
            return max(2, int(cpu_count * 0.75))

        return 4  # Safe default

    async def transcribe_audio(
        self,
        audio_data: bytes,
        format: str = "wav",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio data to text"""
        if not self.is_ready or self.model is None:
            logger.error("Model not loaded, cannot transcribe")
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration_ms=0,
                transcription_time_ms=0
            )

        try:
            start_time = time.time()

            # Validate audio
            is_valid, error_msg = self._validate_audio(audio_data, format)
            if not is_valid:
                logger.error(f"Audio validation failed: {error_msg}")
                return TranscriptionResult(
                    text="",
                    language="",
                    confidence=0.0,
                    duration_ms=0,
                    transcription_time_ms=0
                )

            # Preprocess audio
            audio_array, sample_rate, duration_ms = await self._preprocess_audio(audio_data, format)

            if stt_config.LOG_AUDIO_INFO:
                logger.debug(f"Audio preprocessed: {duration_ms}ms, {sample_rate}Hz, {len(audio_array)} samples")

            # Transcribe (blocking, run in executor)
            model = self.model
            assert model is not None, "Model should be loaded"

            loop = asyncio.get_event_loop()
            segments_list, info = await loop.run_in_executor(
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

            # Extract segments and build text
            segments = []
            text_parts = []
            total_confidence = 0.0
            segment_count = 0

            for segment in segments_list:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": getattr(segment, 'avg_logprob', 0.0),
                })
                text_parts.append(segment.text)
                total_confidence += getattr(segment, 'avg_logprob', 0.0)
                segment_count += 1

            # Build final text and calculate average confidence
            transcribed_text = " ".join(text_parts).strip()
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

            # Convert log probability to confidence (0-1 scale)
            # avg_logprob is typically negative, so we use exp() to convert
            avg_confidence = min(1.0, max(0.0, np.exp(avg_confidence)))

            # Clean the transcription
            transcribed_text = self._clean_transcription(transcribed_text)

            # Calculate metrics
            transcription_time = time.time() - start_time
            transcription_time_ms = int(transcription_time * 1000)
            rtf = transcription_time / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

            # Log performance
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

            # Check confidence threshold
            if avg_confidence < stt_config.MIN_CONFIDENCE:
                logger.warning(f"Low confidence transcription: {avg_confidence:.2f}")

            return TranscriptionResult(
                text=transcribed_text,
                language=info.language,
                confidence=avg_confidence,
                duration_ms=duration_ms,
                transcription_time_ms=transcription_time_ms,
                segments=segments,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration_ms=0,
                transcription_time_ms=0
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
        """Preprocess audio data for Whisper"""
        try:
            # Import audio processing libraries
            import soundfile as sf
            from io import BytesIO

            # Load audio using soundfile
            audio_io = BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_io, dtype='float32')

            # Convert stereo to mono if necessary
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Resample to 16kHz if necessary (Whisper requirement)
            if sample_rate != stt_config.SAMPLE_RATE:
                import resampy
                audio_array = resampy.resample(audio_array, sample_rate, stt_config.SAMPLE_RATE)
                sample_rate = stt_config.SAMPLE_RATE

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
        """Get current service status"""
        return {
            "ready": self.is_ready,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "hardware": self.hardware_info,
            "sample_rate": stt_config.SAMPLE_RATE,
            "supported_formats": stt_config.SUPPORTED_FORMATS,
            "language": stt_config.LANGUAGE,
        }

    def unload_model(self):
        """Unload the model and free resources"""
        if self.model:
            logger.info("Unloading STT model")
            try:
                # faster-whisper doesn't have explicit cleanup, rely on garbage collection
                if hasattr(self.model, 'close'):
                    self.model.close()
                    logger.debug("Model cleanup method called successfully")
                else:
                    logger.debug("Model does not have a close() method; relying on garbage collection")
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
            finally:
                del self.model
                self.model = None
                self.is_ready = False
