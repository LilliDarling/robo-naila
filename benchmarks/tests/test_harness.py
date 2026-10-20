import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from benchmarks.__main__ import main
from benchmarks.artifacts import distribution, summarize, write_json
from benchmarks.peer import Fixture, ReplayTrack
from benchmarks.traces import EvidenceError, REQUIRED, TraceReader, validate
from benchmarks.voice import check_health, wait_for


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / 'fixture.wav'
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 48000, 0, 'NONE', ''))
        out.writeframes(b'\x01\x02' * 24001)
    return path


def test_immutable_fixture_and_padding(wav):
    original = wav.read_bytes()
    fixture = Fixture.load(wav)
    wav.write_bytes(b'replacement')
    assert fixture.sha256 == hashlib.sha256(original).hexdigest()
    assert fixture.padding_samples == 959
    assert fixture.pcm == b'\x01\x02' * 24001 + bytes(959 * 2)


@pytest.mark.parametrize('params', [(2, 2, 48000), (1, 1, 48000), (1, 2, 16000)])
def test_wav_format_rejected(tmp_path, params):
    path = tmp_path / 'bad.wav'
    with wave.open(str(path), 'wb') as out:
        out.setparams((*params, 0, 'NONE', ''))
        out.writeframes(bytes(96000))
    with pytest.raises(ValueError, match='mono PCM16'):
        Fixture.load(path)


class Clock:
    now = 0

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.now += delay


@pytest.mark.asyncio
async def test_delayed_connection_leading_silence_and_exact_replay(wav):
    fixture, clock = Fixture.load(wav), Clock()
    track = ReplayTrack(fixture, clock, clock.sleep)
    for _ in range(35):
        assert bytes((await track.recv()).planes[0]) == bytes(1920)
    assert track.first is None
    track.connected()
    frames = []
    for _ in range(60):
        frame = await track.recv()
        if track.cursor and len(frames) < len(fixture.pcm) // 1920:
            frames.append(bytes(frame.planes[0]))
    assert b''.join(frames) == fixture.pcm
    assert track.first >= .5 + .68
    assert track.end == pytest.approx(track.first + len(frames) * .02)
    assert track.finished.is_set()


@pytest.mark.asyncio
async def test_canceled_recv_does_not_advance(wav):
    clock = Clock()
    entered = asyncio.Event()

    async def blocked(delay):
        entered.set()
        await asyncio.Event().wait()

    track = ReplayTrack(Fixture.load(wav), clock, blocked)
    track.connected()
    clock.now = 1
    task = asyncio.create_task(track.recv())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert track.cursor == track.pts == 0
    track.sleep = clock.sleep
    frames = [bytes((await track.recv()).planes[0]) for _ in range(len(track.fixture.pcm) // 1920)]
    assert b''.join(frames) == track.fixture.pcm


@pytest.mark.asyncio
async def test_delayed_initial_pull_starts_media_clock_without_catchup(wav):
    clock = Clock()
    track = ReplayTrack(Fixture.load(wav), clock, clock.sleep)
    clock.now = 5  # Signaling must not create five seconds of overdue media.
    track.connected()
    handoffs = []
    for _ in range(30):
        await track.recv()
        handoffs.append(clock.now)
    assert handoffs == pytest.approx([5 + i * .02 for i in range(30)])
    assert track.first >= 5.5
    assert track.cursor > 0


def record(spans):
    return json.dumps({'resourceSpans': [{'resource': {'attributes': []}, 'scopeSpans': [{'spans': spans}]}]}).encode() + b'\n'


def test_actual_collector_fixture():
    reader = TraceReader(Path(__file__).parent / 'fixtures/collector-0.137.0-traces.jsonl')
    spans = reader.read(final=True)
    assert len(spans) == 3
    assert len({s['traceId'] for s in spans}) == 1
    assert {s['resource']['service.name'] for s in spans} == {'naila.hub', 'naila.ai'}


def test_partial_duplicate_conflicting_truncated_capture(tmp_path):
    path = tmp_path / 'traces'
    span = dict(traceId='a' * 32, spanId='b' * 16, name='stt')
    data = record([span])
    path.write_bytes(data[:50])
    reader = TraceReader(path)
    assert reader.read() == ()
    with path.open('ab') as out:
        out.write(data[50:] + data)
    snapshot = reader.read()
    assert len(snapshot) == 1
    snapshot[0]['name'] = 'mutated snapshot'
    assert reader.read()[0]['name'] == 'stt'
    with path.open('ab') as out:
        out.write(record([dict(span, name='conflict')]))
    with pytest.raises(EvidenceError, match='conflicting'):
        reader.read()
    path.write_bytes(data[:-1])
    with pytest.raises(EvidenceError, match='trailing'):
        TraceReader(path).read(final=True)


def attr(**values):
    return [{'key': k, 'value': {'boolValue' if isinstance(v, bool) else 'doubleValue' if isinstance(v, (float, int)) else 'stringValue': v}} for k, v in values.items()]


@pytest.fixture
def evidence():
    sample = dict(trial_id='trial', trace_id='a' * 32, root_span_id='0000000000000001', warmup=False,
                  outcome='success', received_frames=10, peer_first_audio_ms=100, peer_input_to_first_audio_ms=600,
                  **{'peer.connect_ms': 20})
    spans = []
    for i, name in enumerate(REQUIRED, 1):
        values = {'outcome': 'success', 'duration_ms': 10}
        if name in ('hub.session', 'hub.response'):
            values = {'outcome': 'ok'}
        if name == 'hub.session':
            values.update(utterance_count=1, multiple_utterances=False, **{'outbound.send_errors': 0, 'outbound.opus_forward_count': 1})
            for field in ('vad_ms', 'outbound_send_ms'):
                values.update({field + '.count': 10, field + '.sum': 5})
        if name == 'hub.webrtc.receive':
            values.update(decode_errors=0, dispatch_errors=0)
            for field in ('opus_decode_ms', 'receive_dispatch_ms'):
                values.update({field + '.count': 10, field + '.sum': 5})
        if name == 'ai.turn':
            values['audio_nonempty'] = True
        if name == 'tts':
            values['cache_hit'] = True
        if name == 'hub.response':
            values.update(hub_response_first_send_ms=50, hub_response_final_send_ms=100)
        spans.append(dict(traceId=sample['trace_id'], spanId=f'{i:016x}', parentSpanId='' if i == 1 else sample['root_span_id'],
                          name=name, startTimeUnixNano='1000', endTimeUnixNano='2000', attributes=attr(**values)))
    return sample, tuple(spans)


def test_completion_requires_turn_and_cleanup(evidence):
    sample, spans = evidence
    with pytest.raises(EvidenceError, match='ai.turn'):
        validate(tuple(s for s in spans if s['name'] != 'ai.turn'), sample, cleanup=False)
    with pytest.raises(EvidenceError, match='hub.session'):
        validate(tuple(s for s in spans if s['name'] != 'hub.session'), sample)
    validate(spans, sample)
    with pytest.raises(EvidenceError, match='missing spans'):
        validate(spans, dict(sample, trace_id='late-old-trace'))


@pytest.mark.parametrize('stage,change', [('llm', {'outcome': 'fallback'}), ('ai.turn', {'outcome': 'cancelled'}),
                                        ('hub.session', {'utterance_count': 2}), ('hub.webrtc.receive', {'decode_errors': 1})])
def test_failed_or_multiple_turn_cannot_pass(evidence, stage, change):
    sample, spans = evidence
    spans = copy.deepcopy(spans)
    span = next(s for s in spans if s['name'] == stage)
    span['attributes'] += attr(**change)
    with pytest.raises(EvidenceError):
        validate(spans, sample)


def test_statistics_warmup_cache_and_failure_counts(evidence):
    sample, spans = evidence
    manifest = dict(run_id='run', warmup=1, iterations=1)
    summary = summarize(manifest, [dict(sample, warmup=True), sample], spans)
    assert summary['complete']
    assert summary['stages']['peer_first_audio_ms']['n'] == 1
    assert summary['stages']['vad_ms_per_trial_mean']['mean'] == .5
    assert summary['tts_cache'] == dict(hit=1, miss=0, unknown=0)
    assert summary['stages']['tts_cache_hit_ms']['n'] == 1
    assert summary['stages']['tts_cache_miss_ms']['n'] == 0
    summary = summarize(manifest, [dict(sample, outcome='failed')], spans)
    assert (summary['attempted'], summary['success'], summary['failed']) == (1, 0, 1)
    assert not summary['complete']
    assert distribution([10, 30])['p95'] == 30
    assert distribution([])['mean'] is None
    with pytest.raises(EvidenceError):
        distribution([float('nan')])


@pytest.mark.parametrize('usage', [False, True])
def test_token_populations_and_producer_throughput(evidence, usage):
    sample, spans = evidence
    llm = next(s for s in spans if s['name'] == 'llm')
    if usage:
        llm['attributes'] += attr(**{'llm.completion_tokens': 20, 'llm.prompt_tokens': 100,
                                    'duration_ms': 2000, 'llm.first_output_ms': 500})
    summary = summarize(dict(run_id='run', warmup=1, iterations=1),
                        [dict(sample, warmup=True), sample], spans)
    assert summary['headline'] == ['peer_first_audio_ms', 'llm.first_output_ms', 'llm.tokens_per_second']
    assert summary['diagnostic_stages'] == ['llm']
    for key, expected in [('completion_tokens', 20), ('prompt_tokens', 100), ('tokens_per_second', 10)]:
        population = summary['stages']['llm.' + key]
        assert population['n'] == int(usage)
        assert population['mean'] == (expected if usage else None)
        if not usage:
            assert 'backend token usage' in population['reason']


def healthy(**overrides):
    return dict(status='ok', grpc_connected=True, active_devices=0, ai_server={'health': 'healthy'},
                tts_frames_dropped=0, grpc_reconnects=0, vad_onsets=0, vad_ends=0, **overrides)


@pytest.mark.parametrize('key,value', [('active_devices', 1), ('grpc_connected', False), ('tts_frames_dropped', 1),
                                     ('grpc_reconnects', 1), ('vad_onsets', 2)])
def test_readiness_is_advisory_and_every_counter_is_checked(key, value):
    before = healthy()
    after = dict(before, **{key: value})
    with pytest.raises(EvidenceError):
        check_health(after, before)


@pytest.mark.asyncio
async def test_deadline_and_completion_gate(monkeypatch):
    clock = Clock()
    monkeypatch.setattr('benchmarks.voice.time.monotonic', clock)
    monkeypatch.setattr('benchmarks.voice.asyncio.sleep', clock.sleep)
    calls = []

    async def pending():
        calls.append(clock.now)
        return False

    with pytest.raises(EvidenceError, match='deadline'):
        await wait_for(pending, .3)
    assert clock.now == .3
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_two_directory_contenders_leave_existing_contents(wav, tmp_path):
    from benchmarks.voice import run
    directory = tmp_path / 'run'
    deployment = tmp_path / 'deployment.json'
    deployment.write_text(json.dumps({k: dict(host=k, model='test', os='test', cpu='test') for k in ('pi', 'dedicated')}))
    args = SimpleNamespace(wav=wav, deployment=deployment, run_dir=directory, warmup=0, iterations=1,
                           timeout_seconds=1, hub_url='http://127.0.0.1:1', traces=tmp_path / 'traces')
    first, second = await asyncio.gather(run(args), run(args), return_exceptions=True)
    assert first == 1
    assert isinstance(second, FileExistsError)
    original = (directory / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        await run(args)
    assert (directory / 'manifest.json').read_bytes() == original


def test_offline_summary_is_read_only_and_truncation_fails(evidence, tmp_path, capsys):
    sample, spans = evidence
    write_json(tmp_path / 'manifest.json', dict(run_id='run', warmup=0, iterations=1))
    (tmp_path / 'samples.jsonl').write_text(json.dumps(sample) + '\n')
    capture = tmp_path / 'traces'
    capture.write_bytes(record(spans))
    write_json(tmp_path / 'summary.json', {'execution_finalized': True, 'execution_error': None})
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(['summarize', '--run-dir', str(tmp_path), '--traces', str(capture)]) == 0
    assert json.loads(capsys.readouterr().out)['complete']
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
    with capture.open('ab') as out:
        out.write(b'{')
    assert main(['summarize', '--run-dir', str(tmp_path), '--traces', str(capture)]) == 1
    assert not json.loads(capsys.readouterr().out)['complete']


@pytest.mark.asyncio
@pytest.mark.parametrize('disconnect,late_cleanup,cleanup_failure', [
    (False, False, None), (True, False, None), (False, True, None),
    (True, True, None), (True, False, 'close'), (True, False, 'flush'),
])
async def test_trial_waits_for_connected_finality_and_task_cleanup(monkeypatch, wav, evidence, disconnect, late_cleanup, cleanup_failure):
    import benchmarks.voice as voice
    from opentelemetry.sdk.trace import TracerProvider
    sample, spans = evidence
    provider = TracerProvider(shutdown_on_exit=False)
    original_tracer = provider.get_tracer('test')
    state = {'active': 0, 'cleanup_reads': 0, 'root': None}
    clock = Clock()
    monkeypatch.setattr(voice.time, 'monotonic', clock)
    monkeypatch.setattr(voice.trace, 'get_tracer', lambda name: SimpleNamespace(start_span=start_span))

    def start_span(*args, **kwargs):
        state['root'] = original_tracer.start_span(*args, **kwargs)
        return state['root']

    class Peer:
        connectionState = 'new'
        localDescription = SimpleNamespace(sdp='offer')

        def __init__(self, config):
            self.handlers = {}
            state['peer'] = self

        def on(self, name):
            return lambda callback: self.handlers.__setitem__(name, callback)

        def addTrack(self, track):
            state['track'] = track

        async def createOffer(self):
            return self.localDescription

        async def setLocalDescription(self, offer):
            pass

        async def setRemoteDescription(self, answer):
            state['active'] = 1  # SDP answer alone must not start scored input.

        async def close(self):
            state['active'] = 0
            if cleanup_failure == 'close':
                raise RuntimeError('close failed')

    class Reader:
        def read(self):
            root = state['root'].get_span_context()
            result = copy.deepcopy(spans)
            for span in result:
                span['traceId'] = f'{root.trace_id:032x}'
                if span['name'] == 'benchmark.voice':
                    span['spanId'] = f'{root.span_id:016x}'
                else:
                    span['parentSpanId'] = f'{root.span_id:016x}'
            if state['active'] == 0:
                state['cleanup_reads'] += 1
                if late_cleanup or state['cleanup_reads'] == 1:
                    return tuple(s for s in result if s['name'] != 'hub.session')
            return result

    async def health(session, url):
        current = healthy()
        current.update(active_devices=state['active'], vad_onsets=int(state['track'].end is not None), vad_ends=int(state['track'].end is not None))
        return current

    async def exchange(*args, **kwargs):
        assert kwargs['trace_header'].startswith('00-')
        return 'answer'

    async def sleep(delay):
        clock.now += delay
        peer = state['peer']
        if peer.connectionState == 'new':
            assert state['track'].first is None and state['track'].ready_at is None
            peer.connectionState = 'connected'
            await peer.handlers['connectionstatechange']()
        elif state['active']:
            if disconnect:
                peer.connectionState = 'disconnected'
                await peer.handlers['connectionstatechange']()
            else:
                state['track'].first, state['track'].end = 0, clock.now - .02
                state['sink'].first, state['sink'].count = clock.now, 1

    real_sink = voice.Sink

    def sink():
        state['sink'] = real_sink()
        return state['sink']

    monkeypatch.setattr(voice, 'RTCPeerConnection', Peer)
    monkeypatch.setattr(voice, 'Sink', sink)
    monkeypatch.setattr(voice, 'health', health)
    monkeypatch.setattr(voice, 'exchange_sdp', exchange)
    monkeypatch.setattr(voice.asyncio, 'sleep', sleep)
    if cleanup_failure == 'flush':
        monkeypatch.setattr(provider, 'force_flush', lambda **kwargs: False)
    result = await voice.trial(SimpleNamespace(hub_url='http://test', timeout_seconds=1), Fixture.load(wav), Reader(), None, provider, False)
    assert result['outcome'] == ('failed' if disconnect or late_cleanup else 'success'), result
    if not disconnect:
        assert state['cleanup_reads'] >= 2
    if disconnect:
        assert result['reason'] == 'EvidenceError: peer disconnected'
    elif late_cleanup:
        assert 'deadline' in result['reason']
    provider.shutdown()


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
