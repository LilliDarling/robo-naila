"""gRPC NailaAI service implementation — voice conversation pipeline.

Handles bidirectional streaming between the Hub and AI processing services.
Audio flows: Hub -> STT -> LLM -> TTS -> Hub
"""

import asyncio
import io
import wave
from typing import Optional

from rpc.generated import naila_pb2, naila_pb2_grpc
from utils import get_logger


logger = get_logger(__name__)

# TTS output chunking: ~100ms at 22050 Hz, 16-bit mono = 4410 bytes
TTS_CHUNK_BYTES = 4410


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM S16LE bytes in a WAV header for soundfile compatibility.

    The STT service uses soundfile.read() which requires headers — it cannot
    read raw PCM directly. This adds a minimal WAV header around the PCM data.
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


class NailaAIServicer(naila_pb2_grpc.NailaAIServicer):
    """Implements the NailaAI gRPC service.

    Services are injected after model loading via set_*_service() methods,
    matching the pattern used by MQTT ProtocolHandler.
    """

    def __init__(self):
        self.stt_service = None
        self.llm_service = None
        self.tts_service = None

    def set_stt_service(self, stt_service):
        self.stt_service = stt_service

    def set_llm_service(self, llm_service):
        self.llm_service = llm_service

    def set_tts_service(self, tts_service):
        self.tts_service = tts_service

    async def StreamConversation(self, request_iterator, context):
        """Bidirectional streaming voice conversation.

        Receives AudioInput messages from the Hub, buffers speech audio,
        and processes complete utterances through the STT -> LLM -> TTS pipeline.
        Streams TTS audio back as AudioOutput chunks.
        """
        audio_buffer = bytearray()
        device_id = ""
        conversation_id = ""
        input_sample_rate = 48000
        processing_task: Optional[asyncio.Task] = None

        peer = context.peer()
        logger.info("stream_opened", peer=peer)

        try:
            async for audio_input in request_iterator:
                device_id = audio_input.device_id
                conversation_id = audio_input.conversation_id
                input_sample_rate = audio_input.sample_rate or 48000

                # Extract PCM audio from oneof
                pcm_data = self._extract_audio(audio_input)

                event = audio_input.event

                if event == naila_pb2.SPEECH_EVENT_START:
                    audio_buffer.clear()
                    if pcm_data:
                        audio_buffer.extend(pcm_data)
                    logger.debug(
                        "speech_start",
                        device_id=device_id,
                        conversation_id=conversation_id,
                    )

                elif event == naila_pb2.SPEECH_EVENT_CONTINUE:
                    if pcm_data:
                        audio_buffer.extend(pcm_data)

                elif event == naila_pb2.SPEECH_EVENT_END:
                    if pcm_data:
                        audio_buffer.extend(pcm_data)

                    duration_ms = _pcm_duration_ms(
                        len(audio_buffer), input_sample_rate
                    )
                    logger.info(
                        "speech_end",
                        device_id=device_id,
                        conversation_id=conversation_id,
                        audio_bytes=len(audio_buffer),
                        duration_ms=duration_ms,
                    )

                    # Process the complete utterance
                    async for output in self._process_utterance(
                        bytes(audio_buffer),
                        input_sample_rate,
                        device_id,
                        conversation_id,
                    ):
                        yield output

                    audio_buffer.clear()

                elif event == naila_pb2.SPEECH_EVENT_INTERRUPT:
                    logger.info(
                        "speech_interrupt",
                        device_id=device_id,
                        conversation_id=conversation_id,
                    )
                    if processing_task and not processing_task.done():
                        processing_task.cancel()
                    audio_buffer.clear()

        except asyncio.CancelledError:
            logger.info("stream_cancelled", device_id=device_id)
        except Exception as e:
            logger.error(
                "stream_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__,
            )
        finally:
            logger.info(
                "stream_closed",
                device_id=device_id,
                conversation_id=conversation_id,
            )

    async def _process_utterance(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        device_id: str,
        conversation_id: str,
    ):
        """Run the STT -> LLM -> TTS pipeline on a complete utterance.

        Yields AudioOutput messages as an async generator.
        """
        # ── STT ──────────────────────────────────────────────────────────
        if not self.stt_service or not self.stt_service.is_ready:
            logger.error("stt_service_not_available")
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_STT_FAILED, "STT service not available",
            )
            return

        try:
            wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)
            stt_result = await self.stt_service.transcribe_audio(
                wav_bytes, format="wav"
            )
        except Exception as e:
            logger.error("stt_failed", error=str(e), error_type=type(e).__name__)
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_STT_FAILED, str(e),
            )
            return

        transcription = stt_result.text.strip()
        if not transcription:
            logger.info("stt_empty_transcription", device_id=device_id)
            yield naila_pb2.AudioOutput(
                device_id=device_id,
                conversation_id=conversation_id,
                is_final=True,
            )
            return

        logger.info(
            "stt_result",
            device_id=device_id,
            text=transcription,
            confidence=round(stt_result.confidence, 2),
            duration_ms=stt_result.transcription_time_ms,
        )

        # ── LLM ──────────────────────────────────────────────────────────
        if not self.llm_service or not self.llm_service.is_ready:
            logger.error("llm_service_not_available")
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_LLM_FAILED, "LLM service not available",
            )
            return

        try:
            messages = self.llm_service.build_chat_messages(transcription)
            response_text = await self.llm_service.generate_chat(messages)
        except Exception as e:
            logger.error("llm_failed", error=str(e), error_type=type(e).__name__)
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_LLM_FAILED, str(e),
            )
            return

        if not response_text.strip():
            logger.warning("llm_empty_response", device_id=device_id)
            yield naila_pb2.AudioOutput(
                device_id=device_id,
                conversation_id=conversation_id,
                is_final=True,
                final_stt=transcription,
            )
            return

        logger.info(
            "llm_result",
            device_id=device_id,
            response_length=len(response_text),
        )

        # ── TTS ──────────────────────────────────────────────────────────
        if not self.tts_service or not self.tts_service.is_ready:
            logger.error("tts_service_not_available")
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_TTS_FAILED, "TTS service not available",
            )
            return

        try:
            tts_result = await self.tts_service.synthesize(
                response_text, output_format="raw"
            )
        except Exception as e:
            logger.error("tts_failed", error=str(e), error_type=type(e).__name__)
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_TTS_FAILED, str(e),
            )
            return

        if not tts_result.audio_bytes:
            logger.warning("tts_empty_audio", device_id=device_id)
            yield naila_pb2.AudioOutput(
                device_id=device_id,
                conversation_id=conversation_id,
                is_final=True,
                final_stt=transcription,
            )
            return

        logger.info(
            "tts_result",
            device_id=device_id,
            audio_bytes=len(tts_result.audio_bytes),
            sample_rate=tts_result.sample_rate,
            duration_ms=tts_result.duration_ms,
        )

        # ── Stream TTS audio back in chunks ──────────────────────────────
        audio_bytes = tts_result.audio_bytes
        tts_sample_rate = tts_result.sample_rate
        for sequence, offset in enumerate(range(0, len(audio_bytes), TTS_CHUNK_BYTES)):
            chunk = audio_bytes[offset : offset + TTS_CHUNK_BYTES]
            is_last = offset + TTS_CHUNK_BYTES >= len(audio_bytes)

            yield naila_pb2.AudioOutput(
                device_id=device_id,
                conversation_id=conversation_id,
                audio_pcm=chunk,
                sample_rate=tts_sample_rate,
                sequence_num=sequence,
                is_final=is_last,
                final_stt=transcription if sequence == 0 else "",
            )

    @staticmethod
    def _extract_audio(audio_input) -> Optional[bytes]:
        """Extract PCM bytes from AudioInput oneof field."""
        which = audio_input.WhichOneof("audio")
        if which == "audio_pcm":
            return audio_input.audio_pcm
        if which == "audio_opus":
            logger.warning("opus_audio_not_supported")
            return None
        return None

    @staticmethod
    def _error_output(
        device_id: str,
        conversation_id: str,
        error_code: int,
        error_message: str,
    ) -> naila_pb2.AudioOutput:
        """Build an error AudioOutput message."""
        return naila_pb2.AudioOutput(
            device_id=device_id,
            conversation_id=conversation_id,
            error_code=error_code,
            error_message=error_message,
            is_final=True,
        )


def _pcm_duration_ms(byte_count: int, sample_rate: int, channels: int = 1, sample_width: int = 2) -> int:
    """Calculate duration in ms from PCM byte count."""
    if sample_rate == 0:
        return 0
    samples = byte_count // (channels * sample_width)
    return int((samples / sample_rate) * 1000)
