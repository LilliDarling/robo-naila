# Trace collection foundation

Use a preinstalled **otelcol-contrib 0.137.0** on the dedicated machine.
Start it manually once per fresh capture (no rotation or retention). From the
repository root, select that machine's LAN address and create the directory
atomically:

```sh
export NAILA_CAPTURE_DIR="$(mktemp -d /tmp/naila-capture.XXXXXX)"
export NAILA_COLLECTOR_ENDPOINT=192.168.1.10:4318
otelcol-contrib --config benchmarks/collector.yaml
```

Keep the collector running while the services and fixture run. The collector is
the only writer of `$NAILA_CAPTURE_DIR/traces.jsonl`; do not reuse the directory.
Stop it manually after the exporters have shut down. The fixture runs on a host
that can read this local capture file (normally the dedicated machine).

For each instrumented process, set both:

```sh
export NAILA_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://192.168.1.10:4318/v1/traces
```

The endpoint is the full OTLP/HTTP traces URL, including `/v1/traces`. Omitting the
enable flag creates no exporter worker or connection. Enabling without an
explicit endpoint is a startup error. Export uses SDK bounded queues (2048 spans,
256 per batch, 200ms schedule, 2s exporter timeout). Shutdown waits at most 5s.
An outage or lost spans means incomplete benchmark coverage; audio does not wait
for export. The later harness must enforce its trace collection deadline.

The lightweight peer environment installs these same SDK/exporter pins with
`cd devices/audio-client && uv sync --extra benchmark`; no model environment is
needed for trace export.

Manual APIs: Rust `hub::otel::{extract, inject, accepted_utterance}` and
`opentelemetry::global::tracer`; Python `telemetry.{init,shutdown,extract,span,
in_context,executor_call}`. Python `init(service_name)` defaults to `naila.ai`;
fixture and deferred vision use `naila.benchmark` and `naila.vision-benchmark`.
For the lightweight fixture, put `$PWD/ai-server` on `PYTHONPATH` and
`import telemetry as otel` directly, avoiding the AI package initializer; call
`otel.init("naila.benchmark")`. The helper itself imports only stdlib and OTel.
All processes use UUID `service.instance.id`. No automatic logging export or
exception-text recording is installed. Mark errors/outcomes explicitly before
handling fallbacks; executor producers end their own spans even if their
consumer is cancelled. Use `record_exception=False` when using SDK span context
managers directly.

The parser fixture `tests/fixtures/collector-0.137.0-traces.jsonl` is actual
uncompressed file-exporter output from that version, with generated test IDs
and no media/text payload. Consume complete newline-delimited OTLP JSON records;
retain trailing fragments and deduplicate by `(traceId, spanId)` in the harness.
See the pinned [file exporter configuration](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.137.0/exporter/fileexporter/README.md).

## Voice fixture and offline summary

YOLO/vision is deferred to #46. Only `voice` and `summarize` are implemented.
From repository root, install the existing audio-client environment and expose
both the repository and standalone telemetry helper (no third environment):

```sh
uv sync --locked --extra benchmark --project devices/audio-client
export PYTHONPATH="$PWD:$PWD/devices/audio-client:$PWD/ai-server"
devices/audio-client/.venv/bin/python -m benchmarks voice \
  --hub-url http://PI:8080 --wav /path/question.wav \
  --run-dir /path/new-run --traces "$NAILA_CAPTURE_DIR/traces.jsonl" \
  --deployment /path/deployment.json --warmup 2 --iterations 20 \
  --timeout-seconds 120
devices/audio-client/.venv/bin/python -m benchmarks summarize \
  --run-dir /path/new-run --traces "$NAILA_CAPTURE_DIR/traces.jsonl"
```

Use a fresh capture for each invocation. The peer exclusively creates run-dir;
existing directories exit 2 and are never resumed or overwritten. The collector
owns capture-dir and traces.jsonl. Run artifacts are manifest.json, samples.jsonl
and atomically published summary.json. Offline summary prints recomputed JSON to
stdout without changing any input. Exit 0 means complete execution and coverage;
1 means failed execution/incomplete evidence; 2 means invalid setup/arguments.

On the intended Pi, run the normal release hub with `NAILA_AI_GRPC_URL` set to
the dedicated machine's gRPC address. On that machine run the real orchestration
server with `NAILA_MEMORY_DB_PATH=:memory:`. Enable tracing in both services and
the peer as above. The peer/collector host must be off the Pi, normally the
same dedicated machine; their CPU work can affect results. Only this peer may
use this isolated stack. Health is readiness evidence, not a reservation lock.
After a timeout/cancellation, restart the isolated AI process before rerunning:
native model work can outlive coroutine cancellation. There are no retries.

`--deployment` preserves operator-collected configuration/status facts. For each
of `pi` and `dedicated`, provide `host`, `model`, `os`, and `cpu`, with actual
values from that machine's configuration/status (for example hostname, uname,
Pi device-tree model and lscpu; macOS system_profiler/sysctl). Also record the
AI gRPC endpoint, peer and collector host identities and co-location, and the
effective model settings. Example shape (replace placeholders with actual facts):

```json
{
  "pi": {"host": "PI", "model": "MODEL", "os": "OS", "cpu": "CPU"},
  "dedicated": {"host": "AI", "model": "MODEL", "os": "OS", "cpu": "CPU"},
  "ai_grpc_url": "http://AI:50051",
  "peer_host": "AI", "collector_host": "AI", "colocated_with_ai": true,
  "settings": {
    "vad": {"onset_frames": 3, "hangover_frames": 15},
    "stt": {"model": null, "compute_type": null},
    "llm": {"model": null, "context": null, "max_tokens": null, "temperature": null},
    "tts": {"voice": null, "rate": null},
    "unknown_reason": "Replace nulls from deployment config/status where available"
  }
}
```

The manifest includes source revision/dirty state, immutable WAV SHA-256,
padding, peer host/libraries/Opus configuration and initial health/model/backend
status. Summary retains trace resources and actual outbound codec counters.
Unknown configuration must be explicit; GPU installation never implies every
model runs there. No model-file hashing, prompts, recordings or recognized text
are saved. Required deployment identities must be present for acceptance.

Input must be mono PCM16 48kHz WAV, 0.5–30 seconds, containing exactly one spoken
general question that takes the real LLM path. Read/hash/replay share immutable
bytes. No microphone, speaker, PortAudio stream or AEC opens. Each fresh peer
waits for actual WebRTC connection, sends 500ms leading silence, replays 960-sample
frames with monotonic pacing and pads only the final partial frame. The sink
timestamps/discards decoded frames. Silence, action/fallback/empty recognition,
a second utterance or a response before input end fails; amplitude and quiet
gaps never determine completion. Root/turn/final response evidence and decoded
receipt are required. After closing, completed hub task/track spans and zero
active devices gate the next unique peer; delayed cleanup fails the run.

`peer_first_audio_ms` measures fixture end through first decoded return;
`peer_input_to_first_audio_ms` begins at first fixture-frame handoff instead.
Connection setup is separate. These intervals include both network legs,
codecs, VAD/server work and synthesized leading silence: **they are not audible
response or physical-device latency**. No host clocks are subtracted, and
collection/export delays do not score. Warmups are excluded. Summaries report
n/mean/min/max/nearest-rank p50/p95 (including n=1), null plus a reason for absent
populations, TTS cache hits/misses and per-trial mean VAD/decode/send populations.
Overlapping model stages are never added into a total. FPS is null for voice.

Focused checks, with no model downloads or hardware-speed assertions:

```sh
PYTHONPATH="$PWD:$PWD/devices/audio-client:$PWD/ai-server" \
  devices/audio-client/.venv/bin/python -m pytest -q \
  benchmarks/tests devices/audio-client/tests/test_http.py
```

For the full local stub-model transport gate, first build the normal release hub
with `cargo build -p hub --release`. With the preinstalled pinned collector,
run the same focused pytest command with these additional environment variables:
`NAILA_TEST_HUB="$PWD/target/release/hub"` and
`NAILA_TEST_COLLECTOR=/path/to/otelcol-contrib` (version 0.137.0).
The opt-in integration test uses localhost port 8080, starts its own isolated
AI/collector/hub test processes, and retains diagnostic artifacts under pytest's
temporary directory. It exercises the real graph with native model doubles,
then checks two sequential joined trials and identical offline summarization.
Without those variables, only that transport test is skipped.
