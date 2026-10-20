# Voice pipeline benchmark

Run one software audio-client peer against an isolated Pi hub and AI server.
It replays the committed question through Opus/WebRTC and discards reply audio.
No microphone, speaker or model environment is needed on the peer.

## Collect traces and prepare services

Use **otelcol-contrib 0.137.0** on a machine that can read the peer's capture file.
Start a fresh capture from the repository root:

```sh
export NAILA_CAPTURE_DIR="$(mktemp -d /tmp/naila-capture.XXXXXX)"
export NAILA_COLLECTOR_ENDPOINT=COLLECTOR_LAN_IP:4318
otelcol-contrib --config benchmarks/collector.yaml
```

On the [Pi hub](../deploy/pi/README.md) and [AI server](../deploy/mac/README.md),
set `NAILA_OTEL_ENABLED=1` and
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://COLLECTOR_LAN_IP:4318/v1/traces`.
On the AI server also set `NAILA_BENCHMARK=1` for tracing and in-memory history;
an explicit `NAILA_MEMORY_DB_PATH` overrides that history default. Restart both
services after changing their environment. Only this peer may use the stack.
The peer generates a sampled traceparent without an SDK or exporter.

## Run and summarize

Create deployment.json with actual host/model/OS/CPU facts for `pi` and
`dedicated` (the AI host), plus peer/collector placement and effective settings:

```json
{"pi": {"host": "PI", "model": "MODEL", "os": "OS", "cpu": "CPU"},
 "dedicated": {"host": "AI", "model": "MODEL", "os": "OS", "cpu": "CPU"},
 "peer_host": "PEER", "collector_host": "COLLECTOR", "colocated_with_ai": false,
 "settings": {"vad": {}, "stt": {}, "llm": {}, "tts": {}}}
```

Record actual model, device, compute, sampling, voice and VAD settings; unknown
values must be explicit. Keep historical manifests unchanged. From repo root:

```sh
uv sync --locked --project devices/audio-client
export PYTHONPATH="$PWD:$PWD/devices/audio-client"
devices/audio-client/.venv/bin/python -m benchmarks voice \
  --hub-url http://PI:8080 --wav benchmarks/fixtures/question.wav \
  --run-dir /path/new-run --traces "$NAILA_CAPTURE_DIR/traces.jsonl" \
  --deployment deployment.json --warmup 2 --iterations 20 --timeout-seconds 120
devices/audio-client/.venv/bin/python -m benchmarks summarize \
  --run-dir /path/new-run --traces "$NAILA_CAPTURE_DIR/traces.jsonl"
```

Use a new run directory and capture each time. Stop the collector after the run.
Runs publish manifest.json, samples.jsonl and an atomic summary.json finalization
record. Offline summary prints JSON without changing inputs: exit 0 is complete,
1 failed/incomplete, 2 invalid setup. Late traces may resolve coverage gaps but
cannot erase execution errors or replace a missing finalization record. After a
timeout/cancellation restart the isolated AI process: native work can outlive it.

Input is mono PCM16 48kHz WAV, 0.5–30s, with one general question.
Success requires completed turn/response traces, decoded receipt and hub cleanup;
silence, fallback, multiple utterances and premature replies fail.
The headline is fixture-end to first decoded reply, model time to first text,
then exact backend tokens/s (null when unavailable). Input-to-reply is separate.
Warmups are excluded; these software timings include codecs/network/VAD and are
not audible or physical-device latency. Host clocks are never subtracted.

## Checks

Run `devices/audio-client/.venv/bin/python -m pytest -q benchmarks/tests` with the
PYTHONPATH above. `cargo test -p hub` also exercises this fixture through Opus/VAD.
For the full graph/transport test, build `cargo build -p hub --release`, then set
`NAILA_TEST_HUB="$PWD/target/release/hub"` and `NAILA_TEST_COLLECTOR=/path/to/otelcol-contrib`
when running pytest; only its stub AI process needs the installed ai-server env.
The [archived run](results/2026-10-20-pi4-hub-m2max-ai/) retains its manifest/table;
removed samples.jsonl and traces.jsonl remain in git history at `3e54959`.
