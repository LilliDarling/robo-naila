"""Regressions for the four full-diff audio review findings."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.__main__ import main
from benchmarks.peer import Fixture, ReplayTrack
from benchmarks.traces import EvidenceError, validate
from benchmarks.tests.test_harness import Clock, evidence, healthy, record, wav


@pytest.mark.asyncio
@pytest.mark.parametrize('shutdown_ok', [False, True])
async def test_offline_preserves_execution_result_and_allows_late_trace(monkeypatch, tmp_path, wav, evidence, capsys, shutdown_ok):
    import benchmarks.voice as voice
    sample, spans = evidence
    capture = tmp_path / 'traces'
    capture.write_bytes(record(spans[:-1] if shutdown_ok else spans))
    deployment = tmp_path / 'deployment.json'
    deployment.write_text(json.dumps({k: dict(host=k, model='test', os='test', cpu='test') for k in ('pi', 'dedicated')}))
    args = SimpleNamespace(wav=wav, deployment=deployment, run_dir=tmp_path / 'run', traces=capture,
                           hub_url='http://unused', warmup=0, iterations=1, timeout_seconds=1)

    async def health(*args):
        return healthy()

    async def trial(*args):
        return sample

    monkeypatch.setattr(voice.otel, 'init', lambda *args: object())
    monkeypatch.setattr(voice.otel, 'shutdown', lambda provider: shutdown_ok)
    monkeypatch.setattr(voice, 'health', health)
    monkeypatch.setattr(voice, 'trial', trial)
    assert await voice.run(args) == 1
    capsys.readouterr()
    if shutdown_ok:
        with capture.open('ab') as out:
            out.write(record(spans[-1:]))
    assert main(['summarize', '--run-dir', str(args.run_dir), '--traces', str(capture)]) == (0 if shutdown_ok else 1)
    summary = json.loads(capsys.readouterr().out)
    assert summary['complete'] is shutdown_ok
    if not shutdown_ok:
        assert 'shutdown incomplete' in summary['reason']


def test_offline_requires_final_publication(tmp_path, evidence, capsys):
    sample, spans = evidence
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(run_id='run', warmup=0, iterations=1)))
    (tmp_path / 'samples.jsonl').write_text(json.dumps(sample) + '\n')
    capture = tmp_path / 'traces'
    capture.write_bytes(record(spans))
    assert main(['summarize', '--run-dir', str(tmp_path), '--traces', str(capture)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert not summary['complete']
    assert 'finalization' in summary['reason']


@pytest.mark.asyncio
async def test_replay_deadlines_do_not_accumulate_scheduler_overshoot(wav):
    clock = Clock()

    async def late_sleep(delay):
        clock.now += delay + .001

    track = ReplayTrack(Fixture.load(wav), clock, late_sleep)
    track.connected()
    handoffs = []
    while not track.finished.is_set():
        await track.recv()
        if track.first is not None:
            handoffs.append(clock.now)
    assert handoffs[-1] - handoffs[0] == pytest.approx((len(handoffs) - 1) * .02)
    assert track.next_at == pytest.approx(track.pts / 48000)


@pytest.mark.parametrize('end_ns', ['1000', '500'])
def test_completed_spans_allow_equal_or_backward_wall_time(evidence, end_ns):
    sample, spans = evidence
    stt = next(s for s in spans if s['name'] == 'stt')
    stt['endTimeUnixNano'] = end_ns
    validate(spans, sample)


@pytest.mark.parametrize('end', [None, '0'])
def test_missing_span_end_is_still_incomplete(evidence, end):
    sample, spans = evidence
    stt = next(s for s in spans if s['name'] == 'stt')
    stt.pop('endTimeUnixNano')
    if end is not None:
        stt['endTimeUnixNano'] = end
    with pytest.raises(EvidenceError, match='unfinished'):
        validate(spans, sample)


@pytest.mark.parametrize('contents', [b'', b'not a RIFF container'])
def test_malformed_wav_is_handled_setup_error(tmp_path, capsys, contents):
    wav = tmp_path / 'bad.wav'
    wav.write_bytes(contents)
    assert main(['voice', '--hub-url', 'http://unused', '--wav', str(wav),
                 '--run-dir', str(tmp_path / 'run'), '--traces', str(tmp_path / 'traces')]) == 2
    assert 'setup error:' in capsys.readouterr().err
    assert not (tmp_path / 'run').exists()
