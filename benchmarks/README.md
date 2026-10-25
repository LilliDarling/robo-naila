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
export NAILA_COLLECTOR_GRPC_ENDPOINT=COLLECTOR_LAN_IP:4317
otelcol-contrib --config benchmarks/collector.yaml
```

On the [Pi hub](../deploy/pi/README.md) and [AI server](../deploy/mac/README.md),
set `NAILA_OTEL_ENABLED=1`. Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to
`http://COLLECTOR_LAN_IP:4317` on the hub (OTLP/gRPC, no `/v1/traces`) and
`http://COLLECTOR_LAN_IP:4318/v1/traces` on the AI server (OTLP/HTTP). Allow both
collector TCP ports through the firewall. On the AI server also set `NAILA_BENCHMARK=1` for tracing and in-memory history;
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

## Results

Both voice runs on 2026-10-20 completed with full required trace coverage:
2 warmups + 20 measured trials, 22/22 successful each. YOLO was deferred.

Shared architecture/settings (deployment facts recorded in the manifests):

- Peer and file OTel collector: MacBook Air M4, macOS 26.6.2, separate from AI.
  Software aiortc/PyAV peer replays `fixtures/question.wav` over Opus/WebRTC;
  fresh connection and utterance per trial, sequential, no microphone/speaker.
- Hub: Pi 4B, Debian 13, release binary; Opus decode → VAD → gRPC to AI,
  reply audio over Opus/RTP. VAD onset 3 frames, hangover 25 frames (500 ms).
- AI: Mac Studio M2 Max, 96 GiB, macOS 27.0.1; benchmark mode, in-memory history.
  STT: faster-whisper small.en, CPU/int8, VAD filter off.
  LLM: Meta-Llama-3.1-8B-Instruct Q4_K_M, llama.cpp Metal, all layers on GPU,
  9 threads, context 4096, max tokens 256, temperature 0.7, top_p 0.9,
  top_k 40, repeat penalty 1.1. TTS: Kokoro v1.0 ONNX, bf_emma,
  24 kHz, speed 1.0, phrase cache on. All machines share a home LAN.

Values below are **p50 / p95 / mean**, in milliseconds, excluding warmups.

| Measure | cc8a2bb (dirty) | 52cde2b (clean) |
| --- | ---: | ---: |
| Hub: VAD end → first reply sent | 1655 / 2025 / 1590 | 1605 / 2015 / 1582 |
| Hub: VAD end → final reply sent | 3854 / 5842 / 4215 | 3950 / 5831 / 4207 |
| Peer: fixture start → first reply | 3338 / 3720 / 3278 | 3288 / 3691 / 3263 |
| Peer: fixture end → first reply | 758 / 1140 / 698 | 708 / 1110 / 684 |
| STT | 844 / 904 / 848 | 825 / 851 / 843 |
| LLM first text | 63 / 78 / 66 | 61 / 73 / 64 |
| LLM total (diagnostic) | 277 / 372 / 290 | 278 / 369 / 288 |
| TTS cache miss (n=14) | 524 / 873 / 588 | 522 / 874 / 587 |
| TTS cache hit (n=6) | 0.4 / 0.5 / 0.4 | 0.4 / 0.4 / 0.4 |
| AI turn (includes paced output) | 3875 / 5863 / 4234 | 3970 / 5853 / 4227 |
| Peer connection setup | 125 / 137 / 128 | 118 / 139 / 121 |
| Hub signaling/negotiation | 6 / 6 / 6 | 6 / 6 / 6 |
| Opus decode (per-trial frame mean) | 0.2 / 0.2 / 0.2 | 0.2 / 0.2 / 0.2 |
| VAD (per-trial frame mean) | 0.1 / 0.1 / 0.1 | 0.1 / 0.1 / 0.1 |
| Outbound send (per-trial frame mean) | 0.1 / 0.2 / 0.1 | 0.1 / 0.1 / 0.1 |

Run notes:

- Previous: revision `cc8a2bb`, dirty tree from untracked local tooling;
  both services exported OTLP/HTTP.
- New: revision `52cde2b`, clean tree. Hub OTLP now goes over gRPC :4317
  directly to the collector; AI exports OTLP/HTTP via the loopback relay.
  Peer and collector ran on the MacBook Air as before. `deployment.json` was
  copied from the previous run's manifest; hardware/OS facts were not re-verified.
- These small sequential runs do not establish that the transport change caused
  the timing differences. The repeated question hit the TTS cache in 6 trials
  and missed in 14 in each run, reducing some first-audio waits.

Caveats: fixture-end timing includes trailing silence in the WAV and understates
user-perceived wait; hub timing starts after the VAD hangover. Peer timings
include both network legs, codecs and synthesized leading silence; they measure
software receipt, not physical-device or audible-response latency. Exact token
counts and tokens/second remain unavailable on the streaming path (null in both
summaries). The health response's generic STT device label is not proof of the
recorded CPU/int8 settings; copied deployment settings were not re-verified.

Files: [previous manifest](results/2026-10-20-pi4-hub-m2max-ai/manifest.json),
[new manifest](results/2026-10-20-pi4-hub-m2max-ai-52cde2b/manifest.json).
Host addresses use role names (`pi-hub`, `ai-server`). Only manifests are archived;
summary.json, samples.jsonl and traces.jsonl are omitted. Previous raw data and
summary remain in git history at `3e54959`. The CLI cannot recertify these reduced
archives without raw inputs and the execution-finalization record.
