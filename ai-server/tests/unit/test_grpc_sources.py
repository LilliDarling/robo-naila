"""Deterministic PCM and scheduling fixtures for multiplexed gRPC streams."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rpc.generated import naila_pb2 as pb
from rpc.service import NailaAIServicer

pytestmark = pytest.mark.asyncio


def frame(device, event, pcm=b"", rate=16000, conversation="c", preference=None):
    msg = pb.AudioInput(device_id=device, conversation_id=conversation,
                        event=event, audio_pcm=pcm, sample_rate=rate)
    if preference is not None:
        msg.session_config.preferred_output_sample_rate = preference
    return msg


async def inputs(*messages):
    for message in messages:
        yield message


def context():
    ctx = MagicMock()
    ctx.cancelled.return_value = False
    ctx.send_initial_metadata = AsyncMock()
    return ctx


async def drain(stream):
    return [out async for out in stream]


async def blocked(*args):
    await asyncio.Event().wait()


async def collect(service, messages):
    return await drain(service.StreamConversation(messages, context()))


async def test_interleaved_pcm_identity_rates_preferences_and_source_local_cancellation():
    service, calls = NailaAIServicer(), {}
    started, release = asyncio.Event(), asyncio.Event()

    async def process(pcm, rate, device, conversation, ctx, target_output_sample_rate=None):
        calls[device] = (pcm, rate, conversation, target_output_sample_rate)
        if device == "A":
            started.set()
            await release.wait()
        yield pb.AudioOutput(device_id=device, conversation_id=conversation,
                             audio_pcm=pcm[:2], final_stt=device, is_final=True)

    async def messages():
        yield frame("A", pb.SPEECH_EVENT_START, b"aa" * 4000, preference=24000)
        yield frame("B", pb.SPEECH_EVENT_START, b"discarded" * 2000)
        yield frame("B", pb.SPEECH_EVENT_INTERRUPT)  # Preserve A's pending PCM.
        yield frame("B", pb.SPEECH_EVENT_START, b"bb" * 16000, rate=32000)
        yield frame("A", pb.SPEECH_EVENT_CONTINUE, b"cc" * 4000)
        yield frame("B", pb.SPEECH_EVENT_END, rate=32000)
        yield frame("A", pb.SPEECH_EVENT_END)
        await started.wait()
        yield frame("A", pb.SPEECH_EVENT_END, b"short")
        yield frame("B", pb.SPEECH_EVENT_END, b"short", rate=32000)
        yield frame("B", pb.SPEECH_EVENT_END, b"bb" * 16000, rate=32000)
        release.set()  # Neither short END nor B's valid END may cancel A.

    service._process_utterance = process
    result = await collect(service, messages())
    assert calls == {"A": (b"aa" * 4000 + b"cc" * 4000, 16000, "c", 24000),
                     "B": (b"bb" * 16000, 32000, "c", None)}
    assert {(out.device_id, out.conversation_id, out.final_stt, out.is_final)
            for out in result} == {("A", "c", "A", True), ("B", "c", "B", True)}


@pytest.mark.parametrize("action", ["interrupt", "replace", "next"])
@pytest.mark.parametrize("kind", ["running", "completed", "overflow"])
async def test_queued_output_is_retired_only_for_its_source(monkeypatch, action, kind):
    monkeypatch.setattr("rpc.service.MAX_AUDIO_BUFFER_BYTES", 16000)
    service = NailaAIServicer()
    resume, queued, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()
    a_stopped, b_release, fresh_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def process(pcm, rate, device, conversation, ctx, **kwargs):
        if device == "B":
            yield pb.AudioOutput(device_id=device, conversation_id=conversation, audio_pcm=b"gate")
            await b_release.wait()
            yield pb.AudioOutput(device_id=device, conversation_id=conversation, is_final=True)
        elif pcm.startswith(b"a"):
            try:
                yield pb.AudioOutput(device_id=device, conversation_id=conversation, audio_pcm=b"stale")
                queued.set()  # The adapter's put has completed before generator resumes.
                if kind == "running":
                    await blocked()
            finally:
                a_stopped.set()
        else:
            assert pcm == b"n" * 16000  # Replacement must drop the old pending PCM.
            fresh_started.set()
            await b_release.wait()
            yield pb.AudioOutput(device_id=device, conversation_id=conversation,
                                 audio_pcm=b"fresh", is_final=True)

    service._process_utterance = process
    async def messages():
        yield frame("B", pb.SPEECH_EVENT_END, b"b" * 16000)
        await resume.wait()  # Writer is suspended at its first yield.
        if kind == "overflow":
            yield frame("A", pb.SPEECH_EVENT_CONTINUE, b"a" * 16001)
        else:
            yield frame("A", pb.SPEECH_EVENT_END, b"a" * 16000)
            await queued.wait()
        if action == "interrupt":
            yield frame("A", pb.SPEECH_EVENT_INTERRUPT)
        elif action == "replace":
            yield frame("A", pb.SPEECH_EVENT_START, b"old" * 4000)
            yield frame("A", pb.SPEECH_EVENT_START, b"n" * 8000, conversation="new")
            for in_flight in (False, True):
                if in_flight:
                    yield frame("A", pb.SPEECH_EVENT_END, b"n" * 8000, conversation="new")
                    await fresh_started.wait()
                for event in (pb.SPEECH_EVENT_CONTINUE, pb.SPEECH_EVENT_END, pb.SPEECH_EVENT_INTERRUPT):
                    yield frame("A", event, b"stale" * 4000)
        else:
            yield frame("A", pb.SPEECH_EVENT_END, b"n" * 16000)
        retired.set()
        b_release.set()

    stream = service.StreamConversation(messages(), context())
    assert (await anext(stream)).device_id == "B"
    resume.set()
    await asyncio.wait_for(retired.wait(), 2)
    result = await drain(stream)
    assert a_stopped.is_set() == (kind != "overflow")
    assert not any(out.audio_pcm == b"stale" or out.error_code for out in result)
    assert any(out.device_id == "B" and out.is_final for out in result)
    fresh = [out for out in result if out.audio_pcm == b"fresh"]
    assert len(fresh) == (action != "interrupt")
    if action == "replace":
        assert fresh[0].conversation_id == "new"


@pytest.mark.parametrize("event", [pb.SPEECH_EVENT_START, pb.SPEECH_EVENT_CONTINUE, pb.SPEECH_EVENT_END])
async def test_overflow_and_processing_error_are_source_local(monkeypatch, event):
    monkeypatch.setattr("rpc.service.MAX_AUDIO_BUFFER_BYTES", 16000)
    service, seen = NailaAIServicer(), []

    async def process(pcm, rate, device, conversation, ctx, **kwargs):
        seen.append((device, pcm))
        if device == "C":
            raise RuntimeError("fixture failure")
        yield pb.AudioOutput(device_id=device, conversation_id=conversation, is_final=True)

    service._process_utterance = process
    result = await collect(service, inputs(
        frame("A", pb.SPEECH_EVENT_START, b"a" * 8000),
        frame("B", event, b"b" * 16001),
        frame("A", pb.SPEECH_EVENT_END, b"a" * 8000),
        frame("C", pb.SPEECH_EVENT_END, b"c" * 16000),
    ))
    assert sorted(seen) == [("A", b"a" * 16000), ("C", b"c" * 16000)]
    assert {(out.device_id, out.conversation_id) for out in result if out.error_code} == {("B", "c"), ("C", "c")}
    assert any(out.device_id == "A" and out.is_final for out in result)
    assert next(out for out in result if out.device_id == "C").error_message == "Utterance processing failed"


async def test_preferences_first_start_defaults_and_replacement():
    service, calls, processed = NailaAIServicer(), [], asyncio.Queue()

    async def process(pcm, rate, device, conversation, ctx, target_output_sample_rate=None):
        calls.append((device, conversation, rate, target_output_sample_rate))
        yield pb.AudioOutput(device_id=device, conversation_id=conversation, is_final=True)
        await processed.put(None)

    service._process_utterance = process
    async def messages():
        settings = [
            ("A", "c", 24000), ("B", "c", None), ("B", "c", 0),
            ("B", "c", 8000), ("B", "c", 24000), ("A", "new", None),
        ]
        device, conversation, preference = settings[0]
        yield frame(device, pb.SPEECH_EVENT_START, b"p" * 16000,
                    conversation=conversation, preference=preference)
        for index, (device, conversation, _) in enumerate(settings):
            yield frame(device, pb.SPEECH_EVENT_END, rate=48000, conversation=conversation)
            if index + 1 < len(settings):
                device, conversation, preference = settings[index + 1]
                # Pending PCM keeps this context active while its prior output drains.
                yield frame(device, pb.SPEECH_EVENT_START, b"p" * 16000,
                            conversation=conversation, preference=preference)
            await processed.get()
    await collect(service, messages())
    assert calls == [("A", "c", 16000, 24000), ("B", "c", 16000, None),
                     ("B", "c", 16000, None), ("B", "c", 16000, 8000),
                     ("B", "c", 16000, 8000), ("A", "new", 16000, None)]


@pytest.fixture
def queues(monkeypatch):
    """Observe actual queue saturation without patching global asyncio."""
    state = SimpleNamespace(full=asyncio.Event(), terminal=asyncio.Event(),
                            waiting=asyncio.Event(), queue=None)

    class ObservedQueue(asyncio.Queue):
        async def put(self, item):
            state.queue = self
            if self.full():
                state.full.set()
                if not isinstance(item, tuple):
                    state.terminal.set()
            await super().put(item)

        async def get(self):
            if self.empty():
                state.waiting.set()
            return await super().get()

    async def fast_pacing(_):
        await asyncio.sleep(0)

    monkeypatch.setattr("rpc.service.asyncio", SimpleNamespace(
        **(vars(asyncio) | {"Queue": ObservedQueue, "sleep": fast_pacing})))
    return state


@pytest.mark.parametrize("mode", ["close", "cancel", "input_error", "eof_terminal", "error_terminal"])
async def test_full_queue_shutdown_awaits_owned_tasks(monkeypatch, queues, mode):
    service, log = NailaAIServicer(), MagicMock()
    monkeypatch.setattr("rpc.service.logger", log)
    terminal = mode.endswith("terminal")
    devices = "A" if terminal else "AB"
    started = {device: asyncio.Event() for device in devices}
    stopped = set()
    resume, fail, emitted = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def process(pcm, rate, device, conversation, ctx, **kwargs):
        try:
            started[device].set()
            yield pb.AudioOutput(device_id=device, audio_pcm=b"gate")
            if terminal:
                await resume.wait()
                for _ in range(100):
                    yield pb.AudioOutput(device_id=device, audio_pcm=b"queued")
                emitted.set()
            else:
                while True:
                    yield pb.AudioOutput(device_id=device, audio_pcm=b"queued")
        finally:
            stopped.add(device)

    async def messages():
        for device in devices:
            yield frame(device, pb.SPEECH_EVENT_END, b"a" * 16000)
        if mode == "error_terminal":
            await emitted.wait()
            raise RuntimeError("input fixture failure")
        if not terminal:
            await fail.wait()
            raise RuntimeError("input fixture failure")

    service._process_utterance = process
    stream = service.StreamConversation(messages(), context())
    await anext(stream)
    queues.full.clear()  # Require saturation after the writer's first dequeue.
    resume.set()
    await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 2)
    await asyncio.wait_for((queues.terminal if terminal else queues.full).wait(), 2)
    assert queues.queue.full()
    if mode == "cancel":
        queues.waiting.clear()
        task = asyncio.create_task(drain(stream))
        await asyncio.wait_for(queues.waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert any(call.args == ("stream_cancelled",) for call in log.info.call_args_list)
    elif mode == "input_error":
        fail.set()
        with pytest.raises(RuntimeError, match="input fixture failure"):
            await drain(stream)
        assert log.error.call_args.args == ("stream_error",)
    else:
        await asyncio.wait_for(stream.aclose(), 2)
    assert stopped == set(devices)
    closed = log.info.call_args.kwargs
    assert (closed["device_id"], closed["conversation_id"]) == (devices[-1], "c")
    assert 0 <= closed["source_count"] <= len(devices)


async def test_blocked_enqueue_awaits_nested_orchestration(queues):
    service, stopped = NailaAIServicer(), asyncio.Event()
    service.stt_service = MagicMock(is_ready=True)
    service.stt_service.transcribe_audio = AsyncMock(return_value=MagicMock(
        text="fixture", confidence=1.0, transcription_time_ms=1))

    async def orchestrate(data, audio_delivery, transport):
        try:
            await audio_delivery(MagicMock(audio_bytes=b"\x01\x00" * 48000,
                                          sample_rate=48000), "fixture")
        finally:
            await asyncio.sleep(0)
            stopped.set()

    service.orchestrator = MagicMock(process_task=orchestrate)
    task = asyncio.create_task(service._enqueue_utterance(
        SimpleNamespace(put=blocked), b"\x01\x00" * 8000, 16000, "A", "c", context()))
    await asyncio.wait_for(queues.full.wait(), 2)
    assert queues.queue.full()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert stopped.is_set()


@pytest.mark.parametrize("with_output", [False, True])
async def test_idle_eviction_resets_preferences_and_eof_drains_without_count_cap(monkeypatch, with_output):
    service, log = NailaAIServicer(), MagicMock()
    monkeypatch.setattr("rpc.service.logger", log)
    resume, calls = asyncio.Event(), []

    async def process(pcm, rate, device, conversation, ctx, target_output_sample_rate=None):
        calls.append((device, target_output_sample_rate))
        first = len(calls) == 1
        if with_output:
            for sequence in range(1 if first or device not in ("0", "1") else 150):
                yield pb.AudioOutput(device_id=device, conversation_id=conversation,
                                     sequence_num=sequence, is_final=first or sequence == 149
                                     or device not in ("0", "1"))
        elif first:
            resume.set()  # No awaits before processor completion and idle eviction.

    async def messages():
        yield frame("unknown", pb.SPEECH_EVENT_INTERRUPT)
        yield frame("0", pb.SPEECH_EVENT_START, b"a" * 16000, preference=24000)
        yield frame("0", pb.SPEECH_EVENT_END)
        await resume.wait()
        for index in range(150):
            yield frame(str(index), pb.SPEECH_EVENT_END, b"a" * 16000)

    service._process_utterance = process
    stream = service.StreamConversation(messages(), context())
    if with_output:
        assert (await anext(stream)).is_final
        resume.set()  # First source is idle only after its output has drained.
    result = await asyncio.wait_for(drain(stream), 2)
    assert calls[:2] == [("0", 24000), ("0", None)]
    assert len(calls) == 151 and len({device for device, _ in calls}) == 150
    assert not any(out.error_code for out in result)
    for device in ("0", "1"):
        outputs = [out for out in result if out.device_id == device]
        assert [out.sequence_num for out in outputs] == (list(range(150)) if with_output else [])
        if with_output:
            assert outputs[-1].is_final
    assert len(result) == (448 if with_output else 0)
    assert log.info.call_args.kwargs["source_count"] == 0
