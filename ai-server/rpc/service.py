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

        Input reading and utterance processing run concurrently so that
        SPEECH_EVENT_INTERRUPT can cancel an in-flight processing task.
        """
        audio_buffer = bytearray()
        device_id = ""
        conversation_id = ""
        input_sample_rate = 48000
        processing_task: Optional[asyncio.Task] = None
        output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _SENTINEL = object()

        peer = context.peer()
        logger.info("stream_opened", peer=peer)

        async def _read_inputs():
            """Read the input stream and dispatch speech events."""
            nonlocal audio_buffer, device_id, conversation_id
            nonlocal input_sample_rate, processing_task

            try:
                async for audio_input in request_iterator:
                    device_id = audio_input.device_id
                    conversation_id = audio_input.conversation_id
                    input_sample_rate = audio_input.sample_rate or 48000

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

                        # Cancel any prior processing before starting a new one
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await processing_task
                            except (asyncio.CancelledError, Exception):
                                pass

                        frozen_audio = bytes(audio_buffer)
                        audio_buffer.clear()

                        processing_task = asyncio.create_task(
                            self._enqueue_utterance(
                                output_queue,
                                frozen_audio,
                                input_sample_rate,
                                device_id,
                                conversation_id,
                                context,
                            )
                        )

                    elif event == naila_pb2.SPEECH_EVENT_INTERRUPT:
                        logger.info(
                            "speech_interrupt",
                            device_id=device_id,
                            conversation_id=conversation_id,
                        )
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await processing_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            processing_task = None
                        audio_buffer.clear()
            finally:
                # Wait for any in-flight processing to finish before closing
                if processing_task and not processing_task.done():
                    try:
                        await processing_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await output_queue.put(_SENTINEL)

        reader_task = asyncio.create_task(_read_inputs())

        try:
            while True:
                item = await output_queue.get()
                if item is _SENTINEL:
                    break
                yield item
        except asyncio.CancelledError:
            logger.info("stream_cancelled", device_id=device_id)
            reader_task.cancel()
        except Exception as e:
            logger.error(
                "stream_error",
                device_id=device_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            reader_task.cancel()
        finally:
            if not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info(
                "stream_closed",
                device_id=device_id,
                conversation_id=conversation_id,
            )

    async def _enqueue_utterance(
        self,
        queue: asyncio.Queue,
        pcm_bytes: bytes,
        sample_rate: int,
        device_id: str,
        conversation_id: str,
        context=None,
    ):
        """Run the utterance pipeline and push results onto *queue*.

        Designed to run as an asyncio.Task so it can be cancelled on interrupt.
        Checks context.is_active() before each stage to avoid wasting compute
        after a client disconnect.
        """
        async for output in self._process_utterance(
            pcm_bytes, sample_rate, device_id, conversation_id, context,
        ):
            if context and not context.is_active():
                logger.info("client_disconnected_during_enqueue", device_id=device_id)
                return
            await queue.put(output)

    async def _process_utterance(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        device_id: str,
        conversation_id: str,
        context=None,
    ):
        """Run the STT -> LLM -> TTS pipeline on a complete utterance.

        Yields AudioOutput messages as an async generator.
        Checks context.is_active() between pipeline stages to bail out
        early if the client has disconnected.
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

        if context and not context.is_active():
            logger.info("client_disconnected_after_stt", device_id=device_id)
            return

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

        if context and not context.is_active():
            logger.info("client_disconnected_after_llm", device_id=device_id)
            return

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

        if context and not context.is_active():
            logger.info("client_disconnected_after_tts", device_id=device_id)
            return

        # ── Stream TTS audio back in chunks ──────────────────────────────
        audio_bytes = tts_result.audio_bytes
        tts_sample_rate = tts_result.sample_rate
        for sequence, offset in enumerate(range(0, len(audio_bytes), TTS_CHUNK_BYTES)):
            if context and not context.is_active():
                logger.info("client_disconnected_during_tts_stream", device_id=device_id)
                return
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
