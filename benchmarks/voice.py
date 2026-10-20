"""One active WebRTC peer, one accepted utterance, completion-driven cleanup."""
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import subprocess
import time
from uuid import uuid4

import aiohttp
from audio_client.webrtc import WebRTCClient
from audio_client.profile import get_profile

from .artifacts import summarize, write_json
from .peer import Fixture, ReplayTrack, Sink
from .traces import EvidenceError, TraceReader, validate


def check_health(current, baseline=None, active=0, finished=False):
    if current.get('status') != 'ok' or not current.get('grpc_connected') or current.get('ai_server', {}).get('health') != 'healthy':
        raise EvidenceError('hub/AI unavailable')
    if current['active_devices'] != active:
        raise EvidenceError('unexpected active connections')
    if baseline:
        for key in ('tts_frames_dropped', 'grpc_reconnects'):
            if current[key] != baseline[key]:
                raise EvidenceError('counter changed: ' + key)
        for key in ('vad_onsets', 'vad_ends'):
            delta = current[key] - baseline[key]
            if delta < 0 or delta > 1 or (finished and delta != 1):
                raise EvidenceError('unexpected utterance count: ' + key)


async def health(session, url):
    async with session.get(url + '/health', timeout=aiohttp.ClientTimeout(total=3)) as response:
        response.raise_for_status()
        return await response.json()


async def wait_for(predicate, deadline):
    while True:
        limit = deadline() if callable(deadline) else deadline
        if time.monotonic() >= limit:
            raise EvidenceError('deadline exceeded')
        async with asyncio.timeout(limit - time.monotonic()):
            result = await predicate()
        limit = deadline() if callable(deadline) else deadline
        if time.monotonic() >= limit:
            raise EvidenceError('deadline exceeded')
        if result:
            return result
        await asyncio.sleep(min(.1, max(0, limit - time.monotonic())))


async def trial(args, fixture, reader, session, warmup, previous=None):
    sample = dict(trial_id=str(uuid4()), device_id='benchmark-' + str(uuid4()), warmup=warmup, outcome='failed')
    replay, sink = ReplayTrack(fixture), Sink()
    client = WebRTCClient(args.hub_url, sample['device_id'], None, session,
                          profile=get_profile('mac'), source=replay, on_audio=sink)
    # W3C trace-context v1: sampled external parent; no peer span needs exporting.
    # https://www.w3.org/TR/trace-context-1/#traceparent-header
    sample.update(trace_id=uuid4().hex, root_span_id=uuid4().hex[:16])
    header = f"00-{sample['trace_id']}-{sample['root_span_id']}-01"
    baseline = None
    try:
        baseline = await health(session, args.hub_url)
        check_health(baseline)
        sample['health_before'] = baseline
        if previous and any(baseline[k] != previous[k] for k in ('vad_onsets', 'vad_ends', 'grpc_reconnects', 'tts_frames_dropped')):
            raise EvidenceError('unexpected activity between trials')
        start = time.monotonic()
        async with asyncio.timeout(30):
            await client.connect(trace_header=header)
            async def ready():
                if client.broken.is_set():
                    raise EvidenceError('connection failed during setup')
                return client.connected.is_set()
            await wait_for(ready, start + 30)
        sample['peer.connect_ms'] = (time.monotonic() - start) * 1000
        replay.connected()
        # Bounded replay plus post-input deadline; failures remain active after readiness.
        async def complete():
            if client.broken.is_set():
                raise EvidenceError('peer disconnected')
            if len(client.receive_tasks) > 1:
                raise EvidenceError('unexpected return audio tracks')
            if any(t.done() for t in client.receive_tasks):
                raise EvidenceError('receive sink ended')
            current = await health(session, args.hub_url)
            check_health(current, baseline, active=1)
            snapshot = reader.read()
            if replay.end is None or time.monotonic() < replay.end:
                if sink.first is not None:
                    raise EvidenceError('return before fixture end')
                return False
            if sink.first is not None and sink.first < replay.end:
                raise EvidenceError('return before fixture end')
            try:
                validate(snapshot, sample, cleanup=False)
            except EvidenceError as exc:
                if str(exc).startswith('missing spans:'):
                    return False
                raise
            return sink.count > 0
        replay_deadline = time.monotonic() + 30 + len(fixture.pcm) / 96000
        await wait_for(complete, lambda: replay.end + args.timeout_seconds if replay.end else replay_deadline)
        sample.update(peer_first_audio_ms=(sink.first - replay.end) * 1000,
                      peer_input_to_first_audio_ms=(sink.first - replay.first) * 1000)
    except (Exception, asyncio.CancelledError) as exc:
        sample['reason'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            async with asyncio.timeout(5):
                await client.close()
        except Exception as exc:
            sample.setdefault('reason', 'peer cleanup failed: ' + type(exc).__name__)
        sample.update(received_frames=sink.count, first_fixture_handoff_seconds=replay.first,
                      last_fixture_handoff_seconds=replay.end - .02 if replay.end else None,
                      fixture_end_seconds=replay.end)
    async def cleaned():
        current = await health(session, args.hub_url)
        sample['health_after'] = current
        # Active=1 is permitted while awaiting teardown, never for the next trial.
        check_health(current, baseline, active=current['active_devices'], finished=True)
        if current['active_devices'] > 1:
            raise EvidenceError('unexpected sessions during cleanup')
        try:
            validate(reader.read(), sample)
        except EvidenceError as exc:
            if str(exc).startswith('missing spans:'):
                return False
            raise
        return current['active_devices'] == 0
    try:
        if baseline is None:
            raise EvidenceError('initial health unavailable')
        await wait_for(cleaned, time.monotonic() + 10)
        if 'reason' not in sample:
            sample['outcome'] = 'success'
    except Exception as exc:
        sample.setdefault('reason', str(exc))
    return sample


async def run(args):
    fixture = Fixture.load(args.wav)
    # Path.mkdir(exist_ok=False) arbitrates contenders; never check then create.
    deployment = json.loads(args.deployment.read_text()) if args.deployment else {}
    for machine in ('pi', 'dedicated'):
        if not isinstance(deployment.get(machine), dict) or not all(deployment[machine].get(k) for k in ('host', 'model', 'os', 'cpu')):
            raise ValueError('--deployment needs pi/dedicated host, model, os and cpu from config/status')
    args.run_dir.mkdir()
    manifest = dict(schema_version=1, run_id=str(uuid4()), mode='voice', warmup=args.warmup,
                    iterations=args.iterations, timeout_seconds=args.timeout_seconds, setup_seconds=30, cleanup_seconds=10,
                    hub_url=args.hub_url, wav=str(args.wav.resolve()), traces=str(args.traces.resolve()),
                    fixture_sha256=fixture.sha256, padding_samples=fixture.padding_samples,
                    utc=datetime.now(timezone.utc).isoformat(),
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                    peer=dict(host=platform.node(), os=platform.platform(), cpu=platform.processor() or 'unknown',
                              aiortc=version('aiortc'), av=version('av'), codec='Opus/48000', frame_samples=960,
                              collector_colocation='dedicated machine by default; see deployment'),
                    deployment=deployment,
                    effective_settings=deployment.get('settings', dict(vad=None, stt=None, llm=None, tts=None,
                                                                      unknown_reason='not supplied by config/status')))
    write_json(args.run_dir / 'manifest.json', manifest)
    (args.run_dir / 'samples.jsonl').touch(exist_ok=False)
    samples, reader, error = [], TraceReader(args.traces), None
    try:
        async with aiohttp.ClientSession() as session:
            manifest['initial_health'] = await health(session, args.hub_url)
            check_health(manifest['initial_health'])
            write_json(args.run_dir / 'manifest.json', manifest)
            with (args.run_dir / 'samples.jsonl').open('a') as output:
                previous = manifest['initial_health']
                for index in range(args.warmup + args.iterations):
                    result = await trial(args, fixture, reader, session, index < args.warmup, previous)
                    previous = result.get('health_after')
                    samples.append(result)
                    output.write(json.dumps(result, allow_nan=False) + '\n')
                    output.flush()
                    if result['outcome'] != 'success':
                        break
    except (Exception, asyncio.CancelledError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    capture_error = None
    try:
        snapshot = reader.read(final=True)
    except EvidenceError as exc:
        snapshot, capture_error = reader.snapshot(), str(exc)
    summary = summarize(manifest, samples, snapshot, capture_error, execution_error=error)
    write_json(args.run_dir / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('complete', 'attempted', 'success', 'failed', 'reason')}))
    return 0 if summary['complete'] else 1
