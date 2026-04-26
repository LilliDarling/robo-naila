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
from fractions import Fraction
from typing import Optional

import av
import numpy as np
import resampy

from typing import TYPE_CHECKING

import psutil

from rpc.generated import naila_pb2, naila_pb2_grpc
from utils import get_logger

if TYPE_CHECKING:
    from agents.orchestrator import NAILAOrchestrator
    from managers.ai_model import AIModelManager


logger = get_logger(__name__)

# TTS output chunking: ~100ms at 24000 Hz, 16-bit mono = 4800 bytes
TTS_CHUNK_BYTES = 4800

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

    # Services considered critical for the voice pipeline.
    # If any of these are down, overall health is UNHEALTHY.
    _CRITICAL_SERVICES = {"stt", "llm"}

    def __init__(self):
        self.stt_service = None
        self.orchestrator: Optional[NAILAOrchestrator] = None
        self._ai_model_manager: Optional[AIModelManager] = None
        self._start_time: Optional[float] = None
        self._server_version: str = ""
        self._max_concurrent_streams: int = 0

    def set_stt_service(self, stt_service):
        self.stt_service = stt_service

    def set_orchestrator(self, orchestrator):
        self.orchestrator = orchestrator

    def set_ai_model_manager(self, manager: "AIModelManager"):
        self._ai_model_manager = manager

    def set_server_info(
        self,
        start_time: Optional[float] = None,
        server_version: str = "",
        max_concurrent_streams: int = 0,
    ):
        self._start_time = start_time
        self._server_version = server_version
        self._max_concurrent_streams = max_concurrent_streams

    async def GetStatus(self, request, context):
        """Return server health, capabilities, and optional model/metrics info."""
        uptime = int(time.time() - self._start_time) if self._start_time else 0

        # Build component health and determine overall health
        components = []
        health = naila_pb2.SERVER_HEALTH_HEALTHY
        status = None

        if self._ai_model_manager:
            status = self._ai_model_manager.get_status()

            if not status.get("models_loaded"):
                health = naila_pb2.SERVER_HEALTH_UNHEALTHY

            for name in ("stt", "llm", "tts", "vision"):
                svc = status.get(name)
                if svc is None:
                    continue
                ready = svc.get("ready", False)
                svc_health = (
                    naila_pb2.SERVER_HEALTH_HEALTHY
                    if ready
                    else naila_pb2.SERVER_HEALTH_UNHEALTHY
                )
                components.append(naila_pb2.ComponentHealth(
                    name=name,
                    health=svc_health,
                    message="ready" if ready else "not ready",
                ))
                if not ready:
                    if name in self._CRITICAL_SERVICES:
                        health = naila_pb2.SERVER_HEALTH_UNHEALTHY
                    elif health == naila_pb2.SERVER_HEALTH_HEALTHY:
                        health = naila_pb2.SERVER_HEALTH_DEGRADED
        else:
            health = naila_pb2.SERVER_HEALTH_UNHEALTHY

        response = naila_pb2.StatusResponse(
            health=health,
            server_version=self._server_version,
            uptime_seconds=uptime,
            supported_input_codecs=[
                naila_pb2.AUDIO_CODEC_PCM_S16LE,
                naila_pb2.AUDIO_CODEC_OPUS,
            ],
            supported_output_codecs=[
                naila_pb2.AUDIO_CODEC_PCM_S16LE,
            ],
            max_concurrent_streams=self._max_concurrent_streams,
            components=components,
        )

        # Model info (only when requested — reuse `status` from above)
        if request.include_model_info and status:
            for name, field in [
                ("stt", "stt_model"),
                ("llm", "llm_model"),
                ("tts", "tts_model"),
                ("vision", "vision_model"),
            ]:
                svc = status.get(name)
                if svc is None:
                    continue
                model_path = svc.get("model_path", "")
                hw = svc.get("hardware") or {}
                getattr(response, field).CopyFrom(naila_pb2.ModelInfo(
                    model_id=model_path.rsplit("/", 1)[-1] if model_path else "",
                    loaded=svc.get("ready", False),
                    device=hw.get("device_type", ""),
                ))

        # Metrics (only when requested)
        if request.include_metrics:
            cpu = 0.0
            mem = 0.0
            try:
                cpu = psutil.cpu_percent(interval=0.0) / 100.0
                mem = psutil.virtual_memory().percent / 100.0
            except Exception:
                logger.debug("metrics_collection_failed")
            response.metrics.CopyFrom(naila_pb2.ServerMetrics(
                cpu_utilization=cpu,
                memory_utilization=mem,
            ))

        return response

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
        target_output_sample_rate: Optional[int] = None
        processing_task: Optional[asyncio.Task] = None
        output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _SENTINEL = object()

        peer = context.peer()
        logger.info("stream_opened", peer=peer)

        # Send initial metadata immediately so the tonic client unblocks.
        # Without this, grpc.aio may delay response headers until the first
        # yield, causing a deadlock with bidirectional streaming clients.
        await context.send_initial_metadata(())

        async def _read_inputs():
            """Read the input stream and dispatch speech events."""
            nonlocal audio_buffer, device_id, conversation_id
            nonlocal input_sample_rate, target_output_sample_rate, processing_task

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

                        # Read SessionConfig from the first START message.
                        if target_output_sample_rate is None and audio_input.HasField("session_config"):
                            sc = audio_input.session_config
                            if sc.preferred_output_sample_rate:
                                target_output_sample_rate = sc.preferred_output_sample_rate
                                logger.info(
                                    "session_config",
                                    device_id=device_id,
                                    preferred_output_sample_rate=target_output_sample_rate,
                                    supported_output_codecs=[
                                        naila_pb2.AudioCodec.Name(c)
                                        for c in sc.supported_output_codecs
                                    ],
                                )

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
                                target_output_sample_rate=target_output_sample_rate,
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
        target_output_sample_rate: Optional[int] = None,
    ):
        """Run the utterance pipeline and push results onto *queue*.

        Designed to run as an asyncio.Task so it can be cancelled on interrupt.
        Checks context.cancelled() before each stage to avoid wasting compute
        after a client disconnect.
        """
        async for output in self._process_utterance(
            pcm_bytes, sample_rate, device_id, conversation_id, context,
            target_output_sample_rate=target_output_sample_rate,
        ):
            if context and context.cancelled():
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
        target_output_sample_rate: Optional[int] = None,
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

        if context and context.cancelled():
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

            Pre-encodes TTS audio as Opus frames using PyAV and sends them
            via the audio_opus field. The hub forwards these directly as RTP
            packets without re-encoding, avoiding the Rust opus crate's
            amplitude reduction bug.
            """
            if not audio_data or not audio_data.audio_bytes:
                return
            raw_bytes = audio_data.audio_bytes
            tts_sample_rate = audio_data.sample_rate

            # Resample to 48kHz for Opus (WebRTC standard).
            target_rate = 48000
            if tts_sample_rate != target_rate:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                resampled = resampy.resample(samples, tts_sample_rate, target_rate)
                pcm_48k = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)
            else:
                pcm_48k = np.frombuffer(raw_bytes, dtype=np.int16)

            # Prepend ~100ms of silence so the receive path (jitter buffer,
            # opus decoder cold-start, playback callback timing) reaches steady
            # state before real audio begins. Without this the first ~20-40ms
            # of every TTS response is clipped or muffled — listener hears
            # "...m here to help" instead of "I'm here to help". 100ms at 48kHz
            # = 4800 samples = 5 Opus frames; a barely-perceptible extra pause
            # before the response, much better than a chopped first syllable.
            lead_in_silence = np.zeros(4800, dtype=np.int16)
            pcm_48k = np.concatenate([lead_in_silence, pcm_48k])

            # Encode 20ms Opus frames (960 samples at 48kHz) using PyAV.
            opus_enc = av.CodecContext.create('libopus', 'w')
            opus_enc.sample_rate = 48000
            opus_enc.layout = 'mono'
            opus_enc.format = av.AudioFormat('s16')
            opus_enc.bit_rate = 64000
            opus_enc.open()

            seq = 0
            frame_size = 960  # 20ms at 48kHz
            frame_duration_s = frame_size / 48000  # 0.020s

            # Real-time pacing. Without this, the encoder produces ~90 frames
            # in well under a second and gRPC ships them all to the hub at
            # full speed; the hub forwards them at full speed to WebRTC; the
            # audio client receives them in a burst far faster than the 50
            # frames/sec real-time playback rate. Its bounded speaker queue
            # then overflows and drops frames mid-response, which sounds like
            # "speeding up" to the listener.
            #
            # Strategy: send the first INITIAL_BURST_FRAMES as fast as we can
            # so the receiver has a small jitter buffer to start with, then
            # pace the remainder so each frame leaves at its real-time slot.
            # The math: target_time for frame N = start + N * 20ms. If we're
            # ahead, sleep until then; if behind, send immediately.
            INITIAL_BURST_FRAMES = 10  # 200ms head-start for jitter buffer
            pacing_start = time.monotonic()

            # Opus is a buffered encoder: encode(frame) may yield 0+ packets,
            # and encode(None) flushes any remaining. We can't know during the
            # loop which packet will be the actual last one, so we hold a
            # one-packet lookahead — the previous packet flushes out as
            # is_final=False once we have a successor, and only the truly
            # final packet (after the encoder is fully flushed) carries
            # is_final=is_final. Without this, multiple packets ended up
            # tagged is_final=True (the last in-loop packet plus every flush
            # packet), which silently violated the "last frame of this
            # response" contract documented in hub::audio.
            pending: Optional[bytes] = None

            async def _send_packet(payload: bytes, final: bool) -> None:
                nonlocal seq
                await tts_queue.put(naila_pb2.AudioOutput(
                    device_id=device_id,
                    conversation_id=conversation_id,
                    audio_opus=payload,
                    sample_rate=48000,
                    sequence_num=seq,
                    is_final=final,
                    final_stt=transcription if seq == 0 else "",
                ))
                seq += 1
                if seq > INITIAL_BURST_FRAMES:
                    target = pacing_start + seq * frame_duration_s
                    delay = target - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

            for i in range(0, len(pcm_48k) - frame_size + 1, frame_size):
                chunk = pcm_48k[i : i + frame_size]
                frame = av.AudioFrame.from_ndarray(
                    chunk.reshape(1, -1), format='s16', layout='mono',
                )
                frame.sample_rate = 48000
                frame.pts = i
                frame.time_base = Fraction(1, 48000)

                for pkt in opus_enc.encode(frame):
                    if pending is not None:
                        await _send_packet(pending, final=False)
                    pending = bytes(pkt)

            # Flush encoder
            for pkt in opus_enc.encode(None):
                if pending is not None:
                    await _send_packet(pending, final=False)
                pending = bytes(pkt)

            # Send the genuinely-last packet (or nothing if no audio was produced).
            if pending is not None:
                await _send_packet(pending, final=is_final)

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
                if context and context.cancelled():
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
