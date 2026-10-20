"""Owned artifacts and immutable trace joins; no cross-host clock subtraction."""
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from .traces import EvidenceError, attributes, validate


def write_json(path, value):
    path = Path(path)
    # tempfile owns the unique file; replace publishes absent-or-complete JSON.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as out:
        temporary = out.name
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')
    os.replace(temporary, path)


def distribution(values):
    if not values:
        return {'n': 0, 'mean': None, 'min': None, 'max': None, 'p50': None, 'p95': None, 'reason': 'no valid samples'}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise EvidenceError('invalid milliseconds')
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), min=values[0], max=values[-1],
                p50=values[math.ceil(.5 * len(values))-1], p95=values[math.ceil(.95 * len(values))-1])


def summarize(manifest, samples, snapshot, capture_error=None, *, execution_error=None, execution_finalized=True):
    if not execution_finalized:
        execution_error = execution_error or 'run finalization incomplete'
    populations = {name: [] for name in ('peer.connect_ms', 'peer_first_audio_ms', 'peer_input_to_first_audio_ms',
                   'stt', 'llm', 'tts', 'tts_cache_hit_ms', 'tts_cache_miss_ms', 'ai.turn', 'llm.first_output_ms', 'hub_connect_ms',
                   'llm.completion_tokens', 'llm.prompt_tokens', 'llm.tokens_per_second',
                   'hub_response_first_send_ms', 'hub_response_final_send_ms',
                   'vad_ms_per_trial_mean', 'opus_decode_ms_per_trial_mean',
                   'receive_dispatch_ms_per_trial_mean', 'send_ms_per_trial_mean')}
    cache = {'hit': 0, 'miss': 0, 'unknown': 0}
    known_traces = {s['trace_id'] for s in samples}
    if any(s['name'] in ('hub.session', 'ai.turn') and s['traceId'] not in known_traces for s in snapshot):
        capture_error = 'unexpected session/turn trace in dedicated capture'
    results = []
    provenance = {}
    for sample in samples:
        error = capture_error
        try:
            groups = validate(snapshot, sample)
            if sample['outcome'] != 'success':
                raise EvidenceError(sample.get('reason', 'execution failed'))
            if sample.get('received_frames', 0) < 1:
                raise EvidenceError('no decoded return')
            for spans in groups.values():
                for span in spans:
                    resource = span.get('resource', {})
                    # Per-batch resources repeat: https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.137.0/exporter/fileexporter/README.md
                    resources = provenance.setdefault(resource.get('service.name', 'unknown'), [])
                    if resource not in resources:
                        resources.append(resource)
            if not sample['warmup'] and not error:
                for key in ('peer.connect_ms', 'peer_first_audio_ms', 'peer_input_to_first_audio_ms'):
                    populations[key].append(sample[key])
                for name in ('stt', 'llm', 'tts', 'ai.turn'):
                    for span in groups[name]:
                        attrs = attributes(span.get('attributes', []))
                        # OTel timestamps are wall based; only explicitly monotonic attributes score.
                        value = attrs.get('duration_ms')
                        if value is not None:
                            populations[name].append(float(value))
                        if name == 'llm':
                            for key in ('llm.completion_tokens', 'llm.prompt_tokens'):
                                if key in attrs:
                                    populations[key].append(attrs[key])
                            generated = attrs.get('llm.completion_tokens')
                            if generated is not None and value is not None and value > 0:
                                populations['llm.tokens_per_second'].append(generated * 1000 / value)
                        if name == 'tts':
                            hit = attrs.get('cache_hit')
                            cache['hit' if hit is True else 'miss' if hit is False else 'unknown'] += 1
                            if hit is not None and value is not None:
                                populations['tts_cache_hit_ms' if hit else 'tts_cache_miss_ms'].append(float(value))
                for spans in groups.values():
                    for span in spans:
                        attrs = attributes(span.get('attributes', []))
                        for key in ('llm.first_output_ms', 'hub_connect_ms', 'hub_response_first_send_ms', 'hub_response_final_send_ms'):
                            if key in attrs:
                                populations[key].append(float(attrs[key]))
                        for prefix, field in (('vad', 'vad_ms'), ('opus_decode', 'opus_decode_ms'),
                                              ('receive_dispatch', 'receive_dispatch_ms'), ('send', 'outbound_send_ms')):
                            count = int(attrs.get(field + '.count', 0))
                            if count:
                                populations[prefix + '_ms_per_trial_mean'].append(float(attrs[field + '.sum']) / count)
        except (EvidenceError, KeyError, TypeError, ValueError) as exc:
            error = str(exc)
        results.append(dict(trial_id=sample['trial_id'], warmup=sample['warmup'], complete=error is None, reason=error))
    success = sum(r['complete'] for r in results)
    expected = manifest['warmup'] + manifest['iterations']
    stages = {k: distribution(v) for k, v in populations.items()}
    for key in ('llm.completion_tokens', 'llm.prompt_tokens', 'llm.tokens_per_second'):
        if not stages[key]['n']:
            stages[key]['reason'] = 'exact backend token usage or positive LLM duration unavailable'
    return dict(schema_version=1, run_id=manifest['run_id'], complete=success == expected and not capture_error and not execution_error,
                execution_finalized=execution_finalized, execution_error=execution_error,
                attempted=len(samples), success=success, failed=len(samples)-success,
                requested=expected, measured_success=sum(r['complete'] and not r['warmup'] for r in results),
                required_trace_coverage=results, provenance=dict(resources=provenance),
                headline=['peer_first_audio_ms', 'llm.first_output_ms', 'llm.tokens_per_second'],
                diagnostic_stages=['llm'],
                stages=stages, tts_cache=cache,
                reason=execution_error or capture_error, timing_scope='software peer; includes synthesized silence; not audible or physical-device latency')
