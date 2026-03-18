"""Unit tests for the gRPC NailaAI service — StreamConversation, GetStatus, and helpers."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rpc.generated import naila_pb2
from rpc.service import NailaAIServicer, _pcm_duration_ms, _pcm_to_wav, TTS_CHUNK_BYTES


# ── helpers ──────────────────────────────────────────────────────────────────

def _audio_input(event, pcm=b"", device_id="dev1", conversation_id="conv1", sample_rate=16000):
    """Build a minimal AudioInput-like object."""
    msg = MagicMock()
    msg.event = event
    msg.device_id = device_id
    msg.conversation_id = conversation_id
    msg.sample_rate = sample_rate
    msg.WhichOneof.return_value = "audio_pcm" if pcm else None
    msg.audio_pcm = pcm or None
    return msg


async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


def _make_orchestrator_result(response_text="hi there", intent="greeting"):
    """Build a mock orchestrator.process_task_with_callback result."""
    return {
        "response_text": response_text,
        "intent": intent,
        "confidence": 0.95,
        "processed_text": "hello",
    }


def _make_servicer(stt_text="hello", response_text="hi there", tts_bytes=b"\x00" * 100):
    """Return a NailaAIServicer with mocked STT service and orchestrator.

    The orchestrator mock simulates the real flow: when process_task_with_callback
    is called with an audio_delivery callback, it invokes the callback with TTS
    audio data (as the graph's ResponseGenerator would).
    """
    stt_result = SimpleNamespace(text=stt_text, confidence=0.95, transcription_time_ms=42)
    tts_audio = SimpleNamespace(audio_bytes=tts_bytes, sample_rate=22050, duration_ms=100)

    stt = MagicMock(is_ready=True)
    stt.transcribe_audio = AsyncMock(return_value=stt_result)

    async def mock_process_task_with_callback(message_data, audio_delivery=None, transport="mqtt"):
        """Simulate the orchestrator running the graph and calling the audio delivery callback."""
        result = _make_orchestrator_result(response_text=response_text)
        if audio_delivery and tts_bytes:
            await audio_delivery(tts_audio, response_text, is_final=True)
        return result

    orchestrator = MagicMock()
    orchestrator.process_task_with_callback = AsyncMock(side_effect=mock_process_task_with_callback)

    servicer = NailaAIServicer()
    servicer.set_stt_service(stt)
    servicer.set_orchestrator(orchestrator)
    return servicer


def _grpc_context():
    ctx = MagicMock()
    ctx.peer.return_value = "ipv4:127.0.0.1:9999"
    return ctx


# ── _pcm_duration_ms / _pcm_to_wav ──────────────────────────────────────────

class TestPcmHelpers:
    def test_pcm_duration_ms(self):
        # 16000 samples/s, 16-bit mono -> 2 bytes/sample -> 32000 bytes = 1 s
        assert _pcm_duration_ms(32000, 16000) == 1000

    def test_pcm_duration_ms_zero_rate(self):
        assert _pcm_duration_ms(100, 0) == 0

    def test_pcm_to_wav_roundtrip(self):
        pcm = b"\x00\x01" * 100
        wav = _pcm_to_wav(pcm, 16000)
        assert wav[:4] == b"RIFF"


# ── StreamConversation — happy path ──────────────────────────────────────────

class TestStreamConversation:
    @pytest.mark.asyncio
    async def test_full_utterance_produces_output(self):
        """START -> CONTINUE -> END should yield TTS audio output via orchestrator."""
        servicer = _make_servicer()
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_CONTINUE, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) >= 1
        assert outputs[-1].is_final is True
        # Verify orchestrator was called (not direct LLM/TTS)
        servicer.orchestrator.process_task_with_callback.assert_called_once()
        call_kwargs = servicer.orchestrator.process_task_with_callback.call_args
        assert call_kwargs.kwargs["transport"] == "grpc"
        assert call_kwargs.kwargs["audio_delivery"] is not None

    @pytest.mark.asyncio
    async def test_orchestrator_receives_transcription(self):
        """Orchestrator should receive the STT transcription in task data."""
        servicer = _make_servicer(stt_text="what time is it")
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        async for _ in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            pass

        call_args = servicer.orchestrator.process_task_with_callback.call_args
        task_data = call_args.args[0] if call_args.args else call_args.kwargs.get("message_data", {})
        assert task_data["transcription"] == "what time is it"
        assert task_data["device_id"] == "dev1"

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        servicer = _make_servicer()
        outputs = []
        async for out in servicer.StreamConversation(_async_iter([]), _grpc_context()):
            outputs.append(out)
        assert outputs == []

    @pytest.mark.asyncio
    async def test_no_tts_audio_yields_text_only_final(self):
        """When TTS produces no audio, a text-only final message should be sent."""
        servicer = _make_servicer(tts_bytes=b"")
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].is_final is True
        assert outputs[0].final_stt == "hello"


# ── Interrupt cancellation ───────────────────────────────────────────────────

class TestInterruptCancellation:
    @pytest.mark.asyncio
    async def test_interrupt_cancels_in_flight_processing(self):
        """An INTERRUPT arriving while processing should cancel the task."""
        processing_started = asyncio.Event()
        processing_cancelled = asyncio.Event()

        original_enqueue = NailaAIServicer._enqueue_utterance

        async def slow_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id, context=None):
            processing_started.set()
            try:
                # Simulate long processing
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                processing_cancelled.set()
                raise

        servicer = _make_servicer()

        async def input_stream():
            yield _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100)
            yield _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100)
            # Wait until processing has started before sending interrupt
            await asyncio.wait_for(processing_started.wait(), timeout=2.0)
            yield _audio_input(naila_pb2.SPEECH_EVENT_INTERRUPT)

        with patch.object(NailaAIServicer, "_enqueue_utterance", slow_enqueue):
            outputs = []
            async for out in servicer.StreamConversation(input_stream(), _grpc_context()):
                outputs.append(out)

        assert processing_cancelled.is_set(), "Processing task was not cancelled on interrupt"

    @pytest.mark.asyncio
    async def test_interrupt_without_processing_is_safe(self):
        """INTERRUPT when nothing is processing should not raise."""
        servicer = _make_servicer()
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_INTERRUPT),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)
        assert outputs == []

    @pytest.mark.asyncio
    async def test_interrupt_clears_audio_buffer(self):
        """After INTERRUPT, buffered audio should be discarded."""
        servicer = _make_servicer()
        processing_started = asyncio.Event()

        async def slow_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id, context=None):
            processing_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        async def input_stream():
            # First utterance: start buffering, then end to trigger processing
            yield _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x01" * 100)
            yield _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x01" * 100)
            await asyncio.wait_for(processing_started.wait(), timeout=2.0)
            # Interrupt, then start new utterance
            yield _audio_input(naila_pb2.SPEECH_EVENT_INTERRUPT)
            # New utterance after interrupt
            yield _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x02" * 50)
            yield _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x02" * 50)

        # Track what PCM was passed to the pipeline on the second call
        captured_pcm = []
        original_enqueue = NailaAIServicer._enqueue_utterance
        call_count = 0

        async def tracking_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id, context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: simulate slow processing
                processing_started.set()
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    raise
            else:
                # Second call: run normally
                captured_pcm.append(pcm_bytes)
                await original_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id, context)

        with patch.object(NailaAIServicer, "_enqueue_utterance", tracking_enqueue):
            outputs = []
            async for out in servicer.StreamConversation(input_stream(), _grpc_context()):
                outputs.append(out)

        # Second utterance should only have the \x02 audio (START + END = 100 bytes), not leftover \x01
        assert len(captured_pcm) == 1
        assert captured_pcm[0] == b"\x02" * 100
        assert b"\x01" not in captured_pcm[0]


# ── Error handling ───────────────────────────────────────────────────────────

class TestServiceErrors:
    @pytest.mark.asyncio
    async def test_stt_not_ready_yields_error(self):
        servicer = _make_servicer()
        servicer.stt_service.is_ready = False
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].error_code == naila_pb2.ERROR_STT_FAILED
        assert outputs[0].is_final is True

    @pytest.mark.asyncio
    async def test_orchestrator_not_available_yields_error(self):
        """When orchestrator is not set, should yield an internal error."""
        servicer = _make_servicer()
        servicer.orchestrator = None
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].error_code == naila_pb2.ERROR_INTERNAL

    @pytest.mark.asyncio
    async def test_orchestrator_exception_yields_error(self):
        """When orchestrator raises, should yield an internal error."""
        servicer = _make_servicer()
        servicer.orchestrator.process_task_with_callback = AsyncMock(
            side_effect=RuntimeError("graph failed")
        )
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].error_code == naila_pb2.ERROR_INTERNAL

    @pytest.mark.asyncio
    async def test_empty_transcription_yields_final(self):
        """Empty STT result should yield a final-only message."""
        servicer = _make_servicer(stt_text="")
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].is_final is True
        # Orchestrator should NOT be called for empty transcriptions
        servicer.orchestrator.process_task_with_callback.assert_not_called()


# ── _extract_audio ───────────────────────────────────────────────────────────

class TestExtractAudio:
    def test_pcm_audio(self):
        msg = MagicMock()
        msg.WhichOneof.return_value = "audio_pcm"
        msg.audio_pcm = b"\x00" * 50
        assert NailaAIServicer._extract_audio(msg) == b"\x00" * 50

    def test_opus_returns_none(self):
        msg = MagicMock()
        msg.WhichOneof.return_value = "audio_opus"
        assert NailaAIServicer._extract_audio(msg) is None

    def test_no_audio_returns_none(self):
        msg = MagicMock()
        msg.WhichOneof.return_value = None
        assert NailaAIServicer._extract_audio(msg) is None


# ── GetStatus helpers ───────────────────────────────────────────────────────

def _make_ai_model_manager(
    models_loaded=True,
    llm_ready=True,
    stt_ready=True,
    tts_ready=True,
    vision_ready=True,
):
    """Build a mock AIModelManager with configurable service readiness."""
    manager = MagicMock()

    def _svc_status(name, ready, model_path, **extras):
        status = {
            "ready": ready,
            "model_path": f"/models/{model_path}",
            "model_exists": ready,
            "hardware": {"device_type": "cuda", "device_name": "NVIDIA RTX 4090"},
        }
        status.update(extras)
        return status

    manager.get_status.return_value = {
        "models_loaded": models_loaded,
        "llm": _svc_status("llm", llm_ready, "llama-3-8b.gguf", context_size=8192, max_tokens=512),
        "stt": _svc_status("stt", stt_ready, "whisper-small-en", sample_rate=16000),
        "tts": _svc_status("tts", tts_ready, "lessac.onnx", sample_rate=22050, voice="lessac"),
        "vision": _svc_status("vision", vision_ready, "yolov8n.pt", model_name="yolov8n"),
    }
    return manager


def _make_status_servicer(
    models_loaded=True,
    llm_ready=True,
    stt_ready=True,
    tts_ready=True,
    vision_ready=True,
    start_time=None,
    max_concurrent_streams=4,
):
    """Return a NailaAIServicer configured for GetStatus tests."""
    servicer = NailaAIServicer()
    servicer.set_ai_model_manager(
        _make_ai_model_manager(models_loaded, llm_ready, stt_ready, tts_ready, vision_ready)
    )
    servicer.set_server_info(
        start_time=start_time or time.time(),
        server_version="1.0.0",
        max_concurrent_streams=max_concurrent_streams,
    )
    return servicer


# ── GetStatus — happy path ──────────────────────────────────────────────────

class TestGetStatus:
    @pytest.mark.asyncio
    async def test_healthy_response(self):
        """All services ready should return HEALTHY."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_HEALTHY
        assert response.server_version == "1.0.0"
        assert response.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_supported_codecs(self):
        """Response should advertise PCM and Opus input, PCM output."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert naila_pb2.AUDIO_CODEC_PCM_S16LE in response.supported_input_codecs
        assert naila_pb2.AUDIO_CODEC_PCM_S16LE in response.supported_output_codecs

    @pytest.mark.asyncio
    async def test_max_concurrent_streams(self):
        """Response should reflect the configured max concurrent streams."""
        servicer = _make_status_servicer(max_concurrent_streams=8)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.max_concurrent_streams == 8

    @pytest.mark.asyncio
    async def test_uptime_calculation(self):
        """Uptime should reflect time since server start."""
        start = time.time() - 120  # started 2 minutes ago
        servicer = _make_status_servicer(start_time=start)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.uptime_seconds >= 119  # allow 1s tolerance

    @pytest.mark.asyncio
    async def test_component_health_all_ready(self):
        """When all services are ready, all components should be HEALTHY."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        component_names = [c.name for c in response.components]
        assert "stt" in component_names
        assert "llm" in component_names
        assert "tts" in component_names
        assert "vision" in component_names

        for component in response.components:
            assert component.health == naila_pb2.SERVER_HEALTH_HEALTHY

    @pytest.mark.asyncio
    async def test_component_health_partial_failure(self):
        """When a service is not ready, its component should be UNHEALTHY."""
        servicer = _make_status_servicer(tts_ready=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        tts_component = next(c for c in response.components if c.name == "tts")
        assert tts_component.health == naila_pb2.SERVER_HEALTH_UNHEALTHY

        llm_component = next(c for c in response.components if c.name == "llm")
        assert llm_component.health == naila_pb2.SERVER_HEALTH_HEALTHY


# ── GetStatus — degraded / unhealthy ────────────────────────────────────────

class TestGetStatusHealth:
    @pytest.mark.asyncio
    async def test_degraded_when_non_critical_service_down(self):
        """When vision is down but core services are up, health should be DEGRADED."""
        servicer = _make_status_servicer(vision_ready=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_DEGRADED

    @pytest.mark.asyncio
    async def test_degraded_when_tts_down(self):
        """When TTS is down, health should be DEGRADED (voice still works inbound)."""
        servicer = _make_status_servicer(tts_ready=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_when_llm_down(self):
        """When LLM is down, health should be UNHEALTHY (can't generate responses)."""
        servicer = _make_status_servicer(llm_ready=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_UNHEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_when_stt_down(self):
        """When STT is down, health should be UNHEALTHY (can't process voice input)."""
        servicer = _make_status_servicer(stt_ready=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_UNHEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_when_models_not_loaded(self):
        """When models_loaded is False, overall health should be UNHEALTHY."""
        servicer = _make_status_servicer(models_loaded=False)
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_UNHEALTHY


# ── GetStatus — model info ──────────────────────────────────────────────────

class TestGetStatusModelInfo:
    @pytest.mark.asyncio
    async def test_model_info_included_when_requested(self):
        """Model info should be populated when include_model_info=True."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_model_info=True)
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.llm_model.loaded is True
        assert response.llm_model.model_id != ""
        assert response.stt_model.loaded is True
        assert response.tts_model.loaded is True
        assert response.vision_model.loaded is True

    @pytest.mark.asyncio
    async def test_model_info_has_device(self):
        """Model info should include the hardware device."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_model_info=True)
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.llm_model.device != ""

    @pytest.mark.asyncio
    async def test_model_info_not_included_when_not_requested(self):
        """Model info should be empty when include_model_info=False."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_model_info=False)
        response = await servicer.GetStatus(request, _grpc_context())

        # Protobuf default: unset sub-messages have all-default fields
        assert response.llm_model.model_id == ""
        assert response.stt_model.model_id == ""

    @pytest.mark.asyncio
    async def test_model_info_partial_services(self):
        """Model info for unavailable services should show loaded=False."""
        servicer = _make_status_servicer(tts_ready=False)
        request = naila_pb2.StatusRequest(include_model_info=True)
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.tts_model.loaded is False
        assert response.llm_model.loaded is True


# ── GetStatus — metrics ─────────────────────────────────────────────────────

class TestGetStatusMetrics:
    @pytest.mark.asyncio
    async def test_metrics_included_when_requested(self):
        """Server metrics should be populated when include_metrics=True."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_metrics=True)

        with patch("rpc.service.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.0
            mock_psutil.virtual_memory.return_value = SimpleNamespace(percent=62.5)
            response = await servicer.GetStatus(request, _grpc_context())

        assert response.metrics.cpu_utilization == pytest.approx(0.45, abs=0.01)
        assert response.metrics.memory_utilization == pytest.approx(0.625, abs=0.01)

    @pytest.mark.asyncio
    async def test_metrics_not_included_when_not_requested(self):
        """Metrics should be empty when include_metrics=False."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_metrics=False)
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.metrics.cpu_utilization == 0.0

    @pytest.mark.asyncio
    async def test_metrics_handles_psutil_failure(self):
        """Metrics should gracefully handle psutil errors."""
        servicer = _make_status_servicer()
        request = naila_pb2.StatusRequest(include_metrics=True)

        with patch("rpc.service.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = RuntimeError("no access")
            mock_psutil.virtual_memory.side_effect = RuntimeError("no access")
            response = await servicer.GetStatus(request, _grpc_context())

        # Should still return a valid response with zeroed metrics
        assert response.health != naila_pb2.SERVER_HEALTH_UNKNOWN
        assert response.metrics.cpu_utilization == 0.0


# ── GetStatus — no manager configured ───────────────────────────────────────

class TestGetStatusNoManager:
    @pytest.mark.asyncio
    async def test_no_manager_returns_unhealthy(self):
        """When AIModelManager is not set, server should report UNHEALTHY."""
        servicer = NailaAIServicer()
        servicer.set_server_info(start_time=time.time(), server_version="1.0.0")
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.health == naila_pb2.SERVER_HEALTH_UNHEALTHY
        assert response.server_version == "1.0.0"
        assert len(response.components) == 0

    @pytest.mark.asyncio
    async def test_no_server_info_uses_defaults(self):
        """When server info is not set, defaults should be used."""
        servicer = NailaAIServicer()
        servicer.set_ai_model_manager(_make_ai_model_manager())
        request = naila_pb2.StatusRequest()
        response = await servicer.GetStatus(request, _grpc_context())

        assert response.server_version == ""
        assert response.uptime_seconds == 0
