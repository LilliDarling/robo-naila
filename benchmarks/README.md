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
