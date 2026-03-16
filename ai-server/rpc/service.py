"""gRPC NailaAI service implementation — voice conversation pipeline.

Handles bidirectional streaming between the Hub and AI processing services.
Audio flows: Hub -> STT -> Orchestration Graph (intent, context, LLM, TTS) -> Hub

The gRPC servicer handles transport-specific concerns (audio buffering, speech
events, PCM chunking) while delegating AI logic to the shared orchestrator,
which runs the same LangGraph pipeline used by MQTT.
"""

import asyncio
import io
import time
import wave
from typing import Optional

from typing import TYPE_CHECKING

from rpc.generated import naila_pb2, naila_pb2_grpc
from utils import get_logger

if TYPE_CHECKING:
    from agents.orchestrator import NAILAOrchestrator


logger = get_logger(__name__)

# TTS output chunking: ~100ms at 22050 Hz, 16-bit mono = 4410 bytes
TTS_CHUNK_BYTES = 4410

# Max audio buffer: ~60s of 16-bit mono at 48kHz = ~5.76 MB
MAX_AUDIO_BUFFER_BYTES = 6 * 1024 * 1024


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


def _pcm_duration_ms(byte_count: int, sample_rate: int, channels: int = 1, sample_width: int = 2) -> int:
    """Calculate duration in ms from PCM byte count."""
    if sample_rate == 0:
        return 0
    samples = byte_count // (channels * sample_width)
    return int((samples / sample_rate) * 1000)


class NailaAIServicer(naila_pb2_grpc.NailaAIServicer):
    """Implements the NailaAI gRPC service.

    STT is called directly (transport-specific audio buffering). All other
    AI logic (intent detection, context, LLM, TTS) goes through the shared
    orchestrator which runs the LangGraph pipeline.
    """

    def __init__(self):
        self.stt_service = None
        self.orchestrator: Optional[NAILAOrchestrator] = None

    def set_stt_service(self, stt_service):
        self.stt_service = stt_service

    def set_orchestrator(self, orchestrator):
        self.orchestrator = orchestrator

    async def StreamConversation(self, request_iterator, context):
        """Bidirectional streaming voice conversation.

        Receives AudioInput messages from the Hub, buffers speech audio,
        and processes complete utterances through STT then the orchestration
        graph. Streams TTS audio back as AudioOutput chunks.

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
                            if len(audio_buffer) + len(pcm_data) > MAX_AUDIO_BUFFER_BYTES:
                                logger.warning(
                                    "audio_buffer_overflow",
                                    device_id=device_id,
                                    conversation_id=conversation_id,
                                    buffer_size=len(audio_buffer),
                                    incoming_size=len(pcm_data),
                                    max_buffer_bytes=MAX_AUDIO_BUFFER_BYTES,
                                    buffer_duration_ms=_pcm_duration_ms(
                                        len(audio_buffer), input_sample_rate
                                    ),
                                    reason="buffer_limit_exceeded",
                                )
                                audio_buffer.clear()
                                await output_queue.put(self._error_output(
                                    device_id, conversation_id,
                                    naila_pb2.ERROR_INTERNAL,
                                    "Audio buffer limit exceeded, utterance discarded",
                                ))
                                continue
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
        """Run STT then delegate to the orchestration graph.

        STT is transport-specific (gRPC buffers raw PCM chunks with speech
        events). Everything after transcription goes through the shared
        LangGraph pipeline via the orchestrator.

        Yields AudioOutput messages as an async generator.
        """
        # ── STT (transport-specific) ─────────────────────────────────────
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

        # ── Orchestration (shared graph) ─────────────────────────────────
        if not self.orchestrator:
            logger.error("orchestrator_not_available")
            yield self._error_output(
                device_id, conversation_id,
                naila_pb2.ERROR_INTERNAL, "Orchestrator not available",
            )
            return

        # Stream TTS chunks to the client as they are produced instead of
        # buffering the full response in memory.
        _DONE = object()
        tts_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        audio_produced = False

        async def _grpc_audio_delivery(audio_data, text: str, is_final: bool = True):
            """Callback invoked by ResponseGenerator when TTS audio is ready.

            Chunks the raw PCM into ~100ms AudioOutput messages and streams
            them into an asyncio.Queue for immediate delivery to the client.
            """
            if not audio_data or not audio_data.audio_bytes:
                return
            raw_bytes = audio_data.audio_bytes
            tts_sample_rate = audio_data.sample_rate
            for seq, offset in enumerate(range(0, len(raw_bytes), TTS_CHUNK_BYTES)):
                chunk = raw_bytes[offset : offset + TTS_CHUNK_BYTES]
                is_last = offset + TTS_CHUNK_BYTES >= len(raw_bytes)
                await tts_queue.put(naila_pb2.AudioOutput(
                    device_id=device_id,
                    conversation_id=conversation_id,
                    audio_pcm=chunk,
                    sample_rate=tts_sample_rate,
                    sequence_num=seq,
                    is_final=is_last and is_final,
                    final_stt=transcription if seq == 0 else "",
                ))

        task_data = {
            "task_id": f"grpc_{conversation_id}_{int(time.time() * 1000)}",
            "device_id": device_id,
            "input_type": "audio",
            "transcription": transcription,
            "confidence": stt_result.confidence,
        }

        async def _run_orchestration():
            try:
                await self.orchestrator.process_task_with_callback(
                    task_data,
                    audio_delivery=_grpc_audio_delivery,
                    transport="grpc",
                )
            except Exception as e:
                logger.error("orchestration_failed", error=str(e), error_type=type(e).__name__)
                await tts_queue.put(self._error_output(
                    device_id, conversation_id,
                    naila_pb2.ERROR_INTERNAL, str(e),
                ))
            finally:
                await tts_queue.put(_DONE)

        orchestration_task = asyncio.create_task(_run_orchestration())

        try:
            while True:
                item = await tts_queue.get()
                if item is _DONE:
                    break
                if context and not context.is_active():
                    logger.info("client_disconnected_during_tts_stream", device_id=device_id)
                    orchestration_task.cancel()
                    return
                audio_produced = True
                yield item
        except asyncio.CancelledError:
            orchestration_task.cancel()
            raise

        # If no audio was produced (TTS unavailable), send text-only final
        if not audio_produced:
            yield naila_pb2.AudioOutput(
                device_id=device_id,
                conversation_id=conversation_id,
                is_final=True,
                final_stt=transcription,
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
