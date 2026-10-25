"""Controlled model/pool delays verify stage boundaries without model downloads."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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

pytestmark = pytest.mark.asyncio


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


@pytest.mark.parametrize('kind', ['stt', 'llm'])
async def test_stage_includes_pool_and_work_excludes_consumer_delay(capture, monkeypatch, kind):
    clock = [10.0]
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    class Pool:
        async def __aenter__(self):
            clock[0] += .020
        async def __aexit__(self, *args):
            clock[0] += .003
    service = STTService() if kind == 'stt' else LLMService()
    service.is_ready, service._pool = True, Pool()
    if kind == 'stt':
        service.model = object()
        async def work(*args):
            clock[0] += .030
            monkeypatch.setattr(timing.time, 'time', lambda: -100000.0)  # Wall-clock jump.
            return SimpleNamespace(text='recognized', transcription_time_ms=7)
        monkeypatch.setattr(service, '_transcribe_impl', work)
        assert (await service.transcribe_audio(b'unchanged')).text == 'recognized'
    else:
        def model(*args, **kwargs):
            clock[0] += .030
            yield {'choices': [{'text': 'First sentence. '}]}
            clock[0] += .010
            yield {'choices': [{'text': 'Second sentence.'}]}
        service.model = model
        stream = service.generate_chat_stream([])
        assert await anext(stream) == 'First sentence.'
        assert not trace.get_current_span().get_span_context().is_valid
        assert await asyncio.to_thread(capture.finished[0].wait, 2)
    span, = capture.get_finished_spans()
    assert span.attributes['outcome'] == 'success'
    assert span.attributes['duration_ms'] == pytest.approx(53 if kind == 'stt' else 60)
    if kind == 'stt':
        assert span.attributes['transcription_time_ms'] == 7
    else:
        assert span.attributes['sentence_count'] == 2
        assert span.attributes['llm.first_output_ms'] == pytest.approx(50)
        assert 0 <= span.attributes['llm.first_output_ms'] <= span.attributes['duration_ms']
        clock[0] += 5  # Slow TTS/delivery cannot inflate producer timing.
        assert [s async for s in stream] == ['Second sentence.']
        assert capture.get_finished_spans() == (span,)


@pytest.mark.parametrize('mode,cache_hit,outcome', [
    ('synth', False, 'success'), ('cache', True, 'success'), ('blank', None, 'empty'),
    ('header', False, 'empty_output'), ('error', False, 'error'),
])
async def test_tts_cache_normalization_and_output(capture, monkeypatch, mode, cache_hit, outcome):
    service = TTSService()
    service.is_ready, service.model = True, object()
    audio = np.array([], dtype=np.float32) if mode == 'header' else np.ones(2400, dtype=np.float32) * .1
    service._synthesize_to_audio = AsyncMock(return_value=(audio, 24000))
    if mode == 'error':
        service._synthesize_to_audio.side_effect = RuntimeError('not exported')
    preprocess = service._preprocess_text
    def normalize(text):
        result = preprocess(text)
        if mode == 'cache':  # Cache may appear during normalization, not preflight.
            service._phrase_cache[service._build_cache_key(result)] = audio
        return result
    monkeypatch.setattr(service, '_preprocess_text', normalize)
    encode = service._encode_audio
    monkeypatch.setattr(service, '_encode_audio', lambda samples, fmt: b'RIFF header' if not len(samples) else encode(samples, fmt))
    result = await service.synthesize(' ' if mode == 'blank' else 'A sentence.', output_format='wav' if mode == 'header' else 'raw')
    span, = capture.get_finished_spans()
    assert bool(result.audio_bytes) is (mode in ('synth', 'cache', 'header'))
    assert span.attributes.get('cache_hit') is cache_hit
    assert span.attributes['outcome'] == outcome and 'synthesis_time_ms' in span.attributes
    assert service._synthesize_to_audio.await_count == int(mode in ('synth', 'header', 'error'))


@pytest.mark.parametrize("backend", ["helper", "llm"])
async def test_distinct_producer_parents_cancel_does_not_end_span(capture, backend):
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    pool = ThreadPoolExecutor(2)
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
            if backend == "helper":
                def produce():
                    with otel.span("llm"):
                        return list(model())
                return await asyncio.get_running_loop().run_in_executor(pool, otel.executor_call(produce))
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
    pool.shutdown()
    spans = capture.get_finished_spans()
    assert len(spans) == 2
    assert {s.context.trace_id for s in spans} == {1, 2}
    cancelled, = [s for s in spans if s.context.trace_id == 1]
    if backend == "llm":
        assert cancelled.attributes["caller_cancelled"] is True
        assert cancelled.attributes["outcome"] == "cancelled"
    assert not trace.get_current_span().get_span_context().is_valid


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


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('mode,usage', [
    ('success', None), ('success', {'completion_tokens': 17, 'prompt_tokens': 43}),
    ('success', {'completion_tokens': -1, 'prompt_tokens': True}),
    ('empty', None), ('error', None), ('unavailable', None),
])
async def test_llm_outcome_usage_and_payload_privacy(capture, streaming, mode, usage):
    service = LLMService()
    service.is_ready = mode != 'unavailable'
    def model(*args, **options):
        assert options['temperature'] == .7
        if mode == 'error':
            raise RuntimeError('not exported')
        choices = [{'text': 'Several tokens in one chunk.'}] if mode == 'success' else []
        return iter([{'choices': choices}, {'choices': [], 'usage': usage}]) if streaming else {'choices': choices, 'usage': usage}
    service.model = model
    result = ' '.join([s async for s in service.generate_chat_stream([])]) if streaming else await service.generate_chat([])
    assert result == ('Several tokens in one chunk.' if mode == 'success' else '')
    span, = capture.get_finished_spans()
    assert span.attributes['outcome'] == mode and not span.events
    if mode != 'success':
        assert 'llm.first_output_ms' not in span.attributes
    for key in ('completion_tokens', 'prompt_tokens'):
        if usage and type(usage[key]) is int and usage[key] >= 0:
            assert span.attributes['llm.' + key] == usage[key]
            assert span.attributes['llm.' + key + '.unavailable_reason'] == ''
        else:
            assert 'llm.' + key not in span.attributes
            assert span.attributes['llm.' + key + '.unavailable_reason']
