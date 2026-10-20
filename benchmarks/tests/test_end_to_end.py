"""Opt-in actual WebRTC/gRPC/OTLP integration; no downloaded models or devices."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

import pytest

from benchmarks.traces import TraceReader, attributes, validate

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = os.environ.get('NAILA_TEST_COLLECTOR')
HUB = os.environ.get('NAILA_TEST_HUB')
pytestmark = pytest.mark.skipif(not (COLLECTOR and HUB), reason='set NAILA_TEST_COLLECTOR and NAILA_TEST_HUB for real transport integration')


def eventually(read, processes, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert all(p.poll() is None for p in processes), 'integration service exited; inspect retained logs'
        try:
            value = read()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(.1)
    pytest.fail('integration readiness deadline; inspect retained logs')


def test_real_webrtc_grpc_graph_collector(tmp_path):
    import socket
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        collector_port = reservation.getsockname()[1]
    # The collector owns its bind; a contender exits rather than sharing capture.
    environment = dict(os.environ, NAILA_OTEL_ENABLED='1', NAILA_MEMORY_DB_PATH=':memory:',
                       OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=f'http://127.0.0.1:{collector_port}/v1/traces',
                       NAILA_COLLECTOR_ENDPOINT=f'127.0.0.1:{collector_port}', NAILA_CAPTURE_DIR=str(tmp_path),
                       PYTHONPATH=os.pathsep.join(map(str, (ROOT, ROOT / 'ai-server', ROOT / 'devices/audio-client'))),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    processes, logs = [], []

    def start(name, command, env=environment):
        output = (tmp_path / (name + '.log')).open('wb')
        logs.append(output)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    try:
        version = subprocess.check_output([COLLECTOR, '--version'], text=True)
        assert '0.137.0' in version
        start('collector', [COLLECTOR, '--config', str(ROOT / 'benchmarks/collector.yaml')])
        ready = tmp_path / 'ai-ready.json'
        start('ai', [str(ROOT / 'ai-server/.venv/bin/python'), str(Path(__file__).with_name('stub_ai.py')), str(ready)])
        port = eventually(lambda: json.loads(ready.read_text())['port'], processes)
        start('hub', [HUB], dict(environment, NAILA_AI_GRPC_URL=f'http://127.0.0.1:{port}'))

        def healthy():
            with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=1) as response:
                state = json.load(response)
            return state if state.get('grpc_connected') and state.get('ai_server', {}).get('health') == 'healthy' else None

        before = eventually(healthy, processes)
        assert before['active_devices'] == 0
        wav = tmp_path / 'input.wav'
        with wave.open(str(wav), 'wb') as output:
            output.setparams((1, 2, 48000, 0, 'NONE', ''))
            output.writeframes(b''.join(struct.pack('<h', 16000 if (i // 24) % 2 == 0 else -16000) for i in range(48000)))
        deployment = tmp_path / 'deployment.json'
        deployment.write_text(json.dumps({key: dict(host='loopback', model='test fixture', os=sys.platform, cpu='test CPU')
                                          for key in ('pi', 'dedicated')}))
        run_dir = tmp_path / 'run'
        command = [sys.executable, '-m', 'benchmarks', 'voice', '--hub-url', 'http://127.0.0.1:8080',
                   '--wav', str(wav), '--run-dir', str(run_dir), '--traces', str(tmp_path / 'traces.jsonl'),
                   '--deployment', str(deployment), '--warmup', '1', '--iterations', '1', '--timeout-seconds', '30']
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=100)
        (tmp_path / 'peer.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr + f'\nArtifacts: {tmp_path}'
        summary = json.loads((run_dir / 'summary.json').read_text())
        assert (summary['attempted'], summary['success'], summary['failed']) == (2, 2, 0)
        assert summary['complete'] and summary['measured_success'] == 1
        assert summary['stages']['stt']['n'] == summary['stages']['llm']['n'] == 1
        assert summary['stages']['tts']['n'] == 2
        for key in ('llm.completion_tokens', 'llm.prompt_tokens', 'llm.tokens_per_second'):
            assert summary['stages'][key]['n'] == 0  # Pinned streaming API supplies no usage.
            assert summary['stages'][key]['mean'] is None
            assert summary['stages'][key]['reason']
        samples = [json.loads(line) for line in (run_dir / 'samples.jsonl').read_text().splitlines()]
        assert len({sample['device_id'] for sample in samples}) == len({sample['trace_id'] for sample in samples}) == 2
        for sample in samples:
            groups = validate(TraceReader(tmp_path / 'traces.jsonl').read(final=True), sample)
            session = attributes(groups['hub.session'][0]['attributes'])
            assert int(session['outbound.opus_forward_count']) > 0
            assert int(session['outbound.pcm_encode_count']) == 0
            assert sample['received_frames'] > 0 and sample['health_after']['active_devices'] == 0
            assert {s['resource']['service.name'] for group in groups.values() for s in group} == {'naila.hub', 'naila.ai', 'naila.benchmark'}
        after = healthy()
        assert after['vad_onsets'] - before['vad_onsets'] == after['vad_ends'] - before['vad_ends'] == 2
        assert after['grpc_reconnects'] == before['grpc_reconnects']
        assert after['tts_frames_dropped'] == before['tts_frames_dropped']
        recomputed = subprocess.run([sys.executable, '-m', 'benchmarks', 'summarize', '--run-dir', str(run_dir),
                                    '--traces', str(tmp_path / 'traces.jsonl')], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10)
        assert recomputed.returncode == 0, recomputed.stderr
        assert json.loads(recomputed.stdout) == summary
        print('2 joined trials; 3 services; real WebRTC/Opus + gRPC + graph + collector 0.137.0; zero drops/reconnects; offline summary identical')
    finally:
        for process in reversed(processes):
            process.send_signal(signal.SIGINT) if process.poll() is None else None
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for output in logs:
            output.close()
