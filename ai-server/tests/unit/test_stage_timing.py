"""Controlled model/pool delays verify stage boundaries without model downloads."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry as otel
from services import timing
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService


@pytest.fixture
def capture(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    exporter.finished = {i: threading.Event() for i in (0, 1, 2)}
    export = exporter.export
    def notify(spans):
        result = export(spans)
        for span in spans:
            exporter.finished.get(span.context.trace_id, exporter.finished[0]).set()
        return result
    monkeypatch.setattr(exporter, "export", notify)
    yield exporter
    provider.shutdown()


@pytest.mark.asyncio
async def test_stt_includes_pool_and_work_with_monotonic_clock(capture, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    class Pool:
        async def __aenter__(self):
            clock[0] += .020
        async def __aexit__(self, *args):
            clock[0] += .003
    service = STTService()
    service.is_ready, service.model, service._pool = True, object(), Pool()
    async def work(*args):
        clock[0] += .030
        monkeypatch.setattr(timing.time, "time", lambda: -100000.0)  # Wall-clock jump.
        return SimpleNamespace(text="recognized", transcription_time_ms=7)
    monkeypatch.setattr(service, "_transcribe_impl", work)
    result = await service.transcribe_audio(b"unchanged")
    span, = capture.get_finished_spans()
    assert result.text == "recognized"
    assert span.attributes["duration_ms"] == pytest.approx(53)
    assert span.attributes["transcription_time_ms"] == 7
    assert span.attributes["outcome"] == "success"


@pytest.mark.asyncio
async def test_tts_actual_cache_branch_and_empty_error(capture, monkeypatch):
    service = TTSService()
    service.is_ready, service.model = True, object()
    service._synthesize_to_audio = AsyncMock(return_value=(np.ones(2400, dtype=np.float32) * .1, 24000))
    # Cache state is selected after normalization, not by a preflight timing probe.
    preprocess = service._preprocess_text
    def normalize(text):
        result = preprocess(text)
        if text == "Second sentence.":
            service._phrase_cache[service._build_cache_key(result)] = np.ones(2400, dtype=np.float32)
        return result
    monkeypatch.setattr(service, "_preprocess_text", normalize)
    first = await service.synthesize("First sentence.", output_format="raw")
    second = await service.synthesize("Second sentence.", output_format="raw")
    empty = await service.synthesize(" ", output_format="raw")
    encode = service._encode_audio
    # Isolate an encoder that permits an empty WAV (the current debug logger rejects it).
    monkeypatch.setattr(service, "_encode_audio", lambda samples, fmt: b"RIFF header" if not len(samples) else encode(samples, fmt))
    service._synthesize_to_audio.return_value = (np.array([], dtype=np.float32), 24000)
    header_only = await service.synthesize("No samples.", output_format="wav")
    assert header_only.audio_bytes  # A WAV header is not synthesized speech.
    service._synthesize_to_audio.side_effect = RuntimeError("not exported")
    failed = await service.synthesize("Failure sentence.", output_format="raw")
    spans = capture.get_finished_spans()
    assert first.audio_bytes and second.audio_bytes
    assert not empty.audio_bytes and not failed.audio_bytes
    assert [s.attributes.get("cache_hit") for s in spans] == [False, True, None, False, False]
    assert [s.attributes["outcome"] for s in spans] == ["success", "success", "empty", "empty_output", "error"]
    assert all("synthesis_time_ms" in s.attributes for s in spans)
    assert service._synthesize_to_audio.await_count == 3


@pytest.mark.asyncio
async def test_stream_producer_ends_before_slow_consumer(capture, monkeypatch):
    service = LLMService()
    service.is_ready = True
    clock = [10.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    class Pool:
        async def __aenter__(self):
            clock[0] += .020
        async def __aexit__(self, *args):
            pass
    service._pool = Pool()
    def model(*args, **kwargs):
        clock[0] += .030
        yield {"choices": [{"text": "First sentence. "}]}
        clock[0] += .010
        yield {"choices": [{"text": "Second sentence."}]}
    service.model = model
    stream = service.generate_chat_stream([])
    first = await anext(stream)
    assert not trace.get_current_span().get_span_context().is_valid
    assert await asyncio.to_thread(capture.finished[0].wait, 2)
    span, = capture.get_finished_spans()
    assert span.attributes["outcome"] == "success"
    assert span.attributes["sentence_count"] == 2
    assert 0 <= span.attributes["llm.first_output_ms"] <= span.attributes["duration_ms"]
    assert span.attributes["llm.first_output_ms"] == pytest.approx(50)
    assert span.attributes["duration_ms"] == pytest.approx(60)
    clock[0] += 5  # Slow TTS/delivery after producer completion must not inflate timing.
    assert first == "First sentence."
    assert [s async for s in stream] == ["Second sentence."]
    assert capture.get_finished_spans() == (span,)


@pytest.mark.asyncio
async def test_distinct_producer_parents_cancel_does_not_end_span(capture):
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    async def consume(index):
        service = LLMService()
        service.is_ready = True
        def model(*args, **kwargs):
            assert trace.get_current_span().get_span_context().trace_id == index + 1
            started[index].set()
            assert release.wait(3)
            yield {"choices": [{"text": "Completed sentence."}]}
        service.model = model
        parent = f"00-{index + 1:032x}-{index + 1:016x}-01"
        with otel.attached(otel.extract(parent)):
            return [s async for s in service.generate_chat_stream([])]
    tasks = [asyncio.create_task(consume(i)) for i in range(2)]
    try:
        for event in started:
            assert await asyncio.to_thread(event.wait, 2)
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[0], .5)
        assert capture.get_finished_spans() == ()
    finally:
        release.set()
    await tasks[1]
    assert await asyncio.to_thread(capture.finished[1].wait, 2)
    spans = capture.get_finished_spans()
    assert len(spans) == 2
    assert {s.context.trace_id for s in spans} == {1, 2}
    cancelled, = [s for s in spans if s.context.trace_id == 1]
    assert cancelled.attributes["caller_cancelled"] is True
    assert cancelled.attributes["outcome"] == "cancelled"
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("empty", "empty"), ("error", "error"), ("unavailable", "unavailable")])
async def test_llm_unusable_output_cannot_pass(capture, mode, expected):
    service = LLMService()
    service.is_ready = mode != "unavailable"
    def model(*args, **kwargs):
        if mode == "error":
            raise RuntimeError("not exported")
        return iter(()) if kwargs.get("stream") else {"choices": []}
    service.model = model
    assert [s async for s in service.generate_chat_stream([])] == []
    assert await service.generate_chat([]) == ""
    spans = capture.get_finished_spans()
    assert len(spans) == 2
    assert all(s.attributes["outcome"] == expected for s in spans)
    assert all("llm.first_output_ms" not in s.attributes for s in spans)
    assert all(not s.events for s in spans)  # No exception messages or prompts.


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "empty_output", "error", "fallback", "cancelled"])
async def test_turn_ends_after_enqueue_completion(capture, outcome):
    from rpc.service import NailaAIServicer
    from rpc.generated import naila_pb2 as pb
    service = NailaAIServicer()
    finish = asyncio.Event()
    async def process(*args, **kwargs):
        if outcome == "fallback":
            timing.turn_outcome("fallback")
        yield pb.AudioOutput(audio_opus=b"audio" if outcome != "empty_output" else b"", error_code=1 if outcome == "error" else 0)
        await finish.wait()
    service._process_utterance = process
    turn = timing.Stage("ai.turn")
    queue = asyncio.Queue()
    task = asyncio.create_task(service._timed_utterance(turn, queue, b"pcm", 48000, "device", "conversation"))
    await asyncio.wait_for(queue.get(), 2)
    assert not capture.get_finished_spans()  # Queued output alone is not completed orchestration.
    if outcome == "cancelled":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        finish.set()
        await task
    span, = capture.get_finished_spans()
    assert span.attributes["outcome"] == outcome


@pytest.mark.asyncio
async def test_stream_cancellation_while_waiting_for_pool(capture):
    waiting = asyncio.Event()
    class Pool:
        async def __aenter__(self):
            waiting.set()
            await asyncio.Event().wait()
        async def __aexit__(self, *args):
            pass
    service = LLMService()
    service.is_ready, service.model, service._pool = True, object(), Pool()
    async def consume():
        return [sentence async for sentence in service.generate_chat_stream([])]
    task = asyncio.create_task(consume())
    await asyncio.wait_for(waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    span, = capture.get_finished_spans()
    assert span.attributes["outcome"] == "cancelled"
    assert "llm.first_output_ms" not in span.attributes
