"""Incremental decoder for the pinned collector's NDJSON, shared live/offline."""
import copy
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    pass


def attributes(items):
    return {item['key']: next(iter(item['value'].values())) for item in items}


class TraceReader:
    # fileexporter v0.137.0 writes batches, not a joined/deduplicated trace store:
    # https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.137.0/exporter/fileexporter/README.md
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.tail = b''
        self.spans = {}
        self.failure = None

    def snapshot(self):
        return copy.deepcopy(tuple(self.spans.values()))

    def read(self, final=False):
        if self.failure:
            raise EvidenceError(self.failure)
        try:
            with self.path.open('rb') as source:
                if source.seek(0, 2) < self.offset:
                    raise EvidenceError('capture truncated')
                source.seek(self.offset)
                data = source.read()
                self.offset += len(data)
        except FileNotFoundError:
            data = b''
        lines = (self.tail + data).split(b'\n')
        self.tail = lines.pop()
        for line in lines:
            try:
                record = json.loads(line)
                for resource in record['resourceSpans']:
                    for scope in resource.get('scopeSpans', []):
                        for raw in scope.get('spans', []):
                            span = dict(raw, resource=attributes(resource.get('resource', {}).get('attributes', [])))
                            key = (span['traceId'], span['spanId'])
                            if key in self.spans and self.spans[key] != span:
                                raise EvidenceError('conflicting span retransmission')
                            self.spans[key] = span
            except (KeyError, TypeError, ValueError) as exc:
                self.failure = 'invalid/conflicting collector record'
                raise EvidenceError(self.failure) from exc
        if final and self.tail:
            raise EvidenceError('incomplete trailing collector record')
        return self.snapshot()


def grouped(snapshot, trace_id):
    groups = {}
    for span in snapshot:
        if span['traceId'] == trace_id:
            groups.setdefault(span['name'], []).append(span)
    return groups


REQUIRED = ('benchmark.voice', 'hub.connect', 'hub.session', 'hub.webrtc.receive',
            'hub.utterance', 'hub.response', 'ai.turn', 'stt', 'llm', 'tts')


def validate(snapshot, sample, cleanup=True):
    groups = grouped(snapshot, sample['trace_id'])
    required = REQUIRED if cleanup else ('hub.response', 'ai.turn')
    missing = [name for name in required if not groups.get(name)]
    if missing:
        raise EvidenceError('missing spans: ' + ', '.join(missing))
    for name, spans in groups.items():
        if name in REQUIRED and name != 'tts' and len(spans) != 1:
            raise EvidenceError('unexpected repeated span: ' + name)
        for span in spans:
            if int(span.get('endTimeUnixNano', 0)) <= int(span.get('startTimeUnixNano', 0)):
                raise EvidenceError('unfinished span: ' + name)
            attrs = attributes(span.get('attributes', []))
            if span.get('status', {}).get('code') == 2 or attrs.get('outcome') not in (None, 'ok', 'success'):
                raise EvidenceError('unsuccessful stage: ' + name)
    def milliseconds(attrs, key):
        value = attrs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise EvidenceError('missing/invalid milliseconds: ' + key)

    for name in ('ai.turn', 'stt', 'llm', 'tts'):
        for span in groups.get(name, []):
            attrs = attributes(span.get('attributes', []))
            if attrs.get('outcome') != 'success' or attrs.get('caller_cancelled', False):
                raise EvidenceError('stage did not succeed: ' + name)
            milliseconds(attrs, 'duration_ms')
            if name == 'ai.turn' and attrs.get('audio_nonempty') is not True:
                raise EvidenceError('turn produced no audio')
    response = attributes(groups['hub.response'][0].get('attributes', []))
    if response.get('outcome') != 'ok':
        raise EvidenceError('response not final')
    for key in ('hub_response_first_send_ms', 'hub_response_final_send_ms'):
        milliseconds(response, key)
    if response['hub_response_final_send_ms'] < response['hub_response_first_send_ms']:
        raise EvidenceError('invalid response boundaries')
    if cleanup:
        session = attributes(groups['hub.session'][0].get('attributes', []))
        receive = attributes(groups['hub.webrtc.receive'][0].get('attributes', []))
        if session.get('utterance_count') != '1' and session.get('utterance_count') != 1:
            raise EvidenceError('expected exactly one utterance')
        if session.get('multiple_utterances') or session.get('outcome') != 'ok':
            raise EvidenceError('invalid hub session')
        for key in ('decode_errors', 'dispatch_errors'):
            if int(receive.get(key, -1)) != 0:
                raise EvidenceError('receive errors or absent counters')
        if int(session.get('outbound.send_errors', -1)) != 0:
            raise EvidenceError('send errors or absent counters')
        for attrs, prefix in ((receive, 'opus_decode_ms'), (receive, 'receive_dispatch_ms'),
                              (session, 'vad_ms'), (session, 'outbound_send_ms')):
            if int(attrs.get(prefix + '.count', 0)) < 1:
                raise EvidenceError('missing media coverage: ' + prefix)
            milliseconds(attrs, prefix + '.sum')
        if int(session.get('outbound.opus_forward_count', 0)) + int(session.get('outbound.pcm_encode_count', 0)) < 1:
            raise EvidenceError('missing actual outbound codec')
        root = groups['benchmark.voice'][0]
        if root['spanId'] != sample['root_span_id'] or root.get('parentSpanId'):
            raise EvidenceError('wrong trace root')
        ids = {s['spanId'] for spans in groups.values() for s in spans}
        for spans in groups.values():
            for span in spans:
                if span is not root and span.get('parentSpanId') not in ids:
                    raise EvidenceError('missing ancestry')
    return groups
