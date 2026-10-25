"""Context ownership and opt-in exporter contracts, without model loading."""
import asyncio
import threading
from unittest.mock import patch

import pytest
from opentelemetry import trace

import telemetry as otel

PARENT = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
OTHER = "00-fedcba9876543210fedcba9876543210-fedcba9876543210-01"


@pytest.mark.parametrize("env_enabled", [False, True])
def test_disabled_creates_no_exporter_or_thread(monkeypatch, env_enabled):
    monkeypatch.setenv("NAILA_OTEL_ENABLED", "1" if env_enabled else "0")
    before = threading.enumerate()
    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as exporter:
        assert otel.init(enabled=False, endpoint=None) is None
        exporter.assert_not_called()
    assert threading.enumerate() == before
    monkeypatch.setenv("NAILA_OTEL_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="explicit"):
        otel.init(enabled=True, endpoint=None)





def test_missing_and_invalid_context():
    for value in ("", "garbage", "00-" + "0" * 32 + "-" + "0" * 16 + "-01"):
        assert not trace.get_current_span(otel.extract(value)).get_span_context().is_valid


@pytest.mark.asyncio
async def test_start_parent_frozen_before_later_input():
    from rpc.generated import naila_pb2 as pb
    from rpc.service import NailaAIServicer
    from unittest.mock import AsyncMock, MagicMock
    seen = []
    release = asyncio.Event()
    servicer = NailaAIServicer()
    async def process(*args, **kwargs):
        await release.wait()
        seen.append(trace.get_current_span().get_span_context().trace_id)
    servicer._enqueue_utterance = process
    async def inputs():
        yield pb.AudioInput(event=pb.SPEECH_EVENT_START, traceparent=PARENT, audio_pcm=b"\0" * 48000, sample_rate=48000)
        yield pb.AudioInput(event=pb.SPEECH_EVENT_END, traceparent=OTHER, sample_rate=48000)
        await asyncio.sleep(0)  # let processing start, then replace reader's START parent
        yield pb.AudioInput(event=pb.SPEECH_EVENT_START, traceparent=OTHER)
        release.set()
    rpc = MagicMock()
    rpc.send_initial_metadata = AsyncMock()
    assert [item async for item in servicer.StreamConversation(inputs(), rpc)] == []
    assert seen == [int(PARENT.split("-")[1], 16)]
