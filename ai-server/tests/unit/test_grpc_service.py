"""Unit tests for the gRPC NailaAI service — StreamConversation and helpers."""

import asyncio
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


def _make_servicer(stt_text="hello", llm_text="hi there", tts_bytes=b"\x00" * 100):
    """Return a NailaAIServicer with mocked sub-services."""
    stt_result = SimpleNamespace(text=stt_text, confidence=0.95, transcription_time_ms=42)
    tts_result = SimpleNamespace(audio_bytes=tts_bytes, sample_rate=22050, duration_ms=100)

    stt = MagicMock(is_ready=True)
    stt.transcribe_audio = AsyncMock(return_value=stt_result)

    llm = MagicMock(is_ready=True)
    llm.build_chat_messages = MagicMock(return_value=[{"role": "user", "content": stt_text}])
    llm.generate_chat = AsyncMock(return_value=llm_text)

    tts = MagicMock(is_ready=True)
    tts.synthesize = AsyncMock(return_value=tts_result)

    servicer = NailaAIServicer()
    servicer.set_stt_service(stt)
    servicer.set_llm_service(llm)
    servicer.set_tts_service(tts)
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
        """START -> CONTINUE -> END should yield TTS audio output."""
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

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        servicer = _make_servicer()
        outputs = []
        async for out in servicer.StreamConversation(_async_iter([]), _grpc_context()):
            outputs.append(out)
        assert outputs == []


# ── Interrupt cancellation ───────────────────────────────────────────────────

class TestInterruptCancellation:
    @pytest.mark.asyncio
    async def test_interrupt_cancels_in_flight_processing(self):
        """An INTERRUPT arriving while processing should cancel the task."""
        processing_started = asyncio.Event()
        processing_cancelled = asyncio.Event()

        original_enqueue = NailaAIServicer._enqueue_utterance

        async def slow_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id):
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

        async def slow_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id):
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

        async def tracking_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id):
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
                await original_enqueue(self, queue, pcm_bytes, sample_rate, device_id, conversation_id)

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
    async def test_llm_not_ready_yields_error(self):
        servicer = _make_servicer()
        servicer.llm_service.is_ready = False
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].error_code == naila_pb2.ERROR_LLM_FAILED

    @pytest.mark.asyncio
    async def test_tts_not_ready_yields_error(self):
        servicer = _make_servicer()
        servicer.tts_service.is_ready = False
        inputs = [
            _audio_input(naila_pb2.SPEECH_EVENT_START, pcm=b"\x00" * 100),
            _audio_input(naila_pb2.SPEECH_EVENT_END, pcm=b"\x00" * 100),
        ]
        outputs = []
        async for out in servicer.StreamConversation(_async_iter(inputs), _grpc_context()):
            outputs.append(out)

        assert len(outputs) == 1
        assert outputs[0].error_code == naila_pb2.ERROR_TTS_FAILED


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
