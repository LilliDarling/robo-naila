# Voice pipeline benchmark — Pi 4 hub, M2 Max AI server (2026-10-20)

First complete real-deployment run for #45 (sub-issue of #38): 2 warmups +
20 measured trials, 22/22 successful, full trace coverage. Audio pipeline
only; YOLO is deferred (#46).

## Architecture

```
 benchmark peer + OTel collector          Raspberry Pi hub               AI server
 (MacBook Air M4, macOS 26.6.2)           (Pi 4B, Debian 13)             (Mac Studio M2 Max, macOS 27.0.1)
 ┌───────────────────────────┐  WebRTC   ┌────────────────────────┐ gRPC ┌──────────────────────────┐
 │ benchmarks voice          │──Opus────▶│ hub (release binary)   │─────▶│ ai-server (main.py)      │
 │ replays question.wav      │◀─Opus─────│ Opus decode, VAD,      │◀─────│ STT → orchestration      │
 │ OTel collector (file)     │           │ gRPC relay, RTP send   │      │ graph → LLM → TTS        │
 └───────────────────────────┘           └────────────────────────┘      └──────────────────────────┘
              ▲                                   │ OTLP/HTTP                       │ OTLP/HTTP
              └───────────────────────────────────┴─────────────────────────────────┘
```

- **Peer:** a software WebRTC client (aiortc 1.14.0, PyAV 16.1.0) sends the
  committed fixture `benchmarks/fixtures/question.wav` ("what color is the
  sky", mono PCM16 48 kHz, 2.56 s) as real Opus/RTP to the hub, then waits for
  decoded reply audio. No microphone or speaker. One fresh connection and one
  utterance per trial, sequential.
- **Hub:** Pi 4 Model B Rev 1.5, Cortex-A72 4× 1.8 GHz, 7.6 GiB, Debian 13,
  kernel 6.18.50. Release build of `hub/`. VAD onset 3 frames (60 ms),
  hangover 25 frames (500 ms) per the #38 operator amendment. Both `eth0` and
  `wlan0` are on the LAN; `deploy/pi/` host configuration pins each address to
  its own interface. Media used whichever interface ICE selected.
- **AI server:** Mac Studio (Mac14,13), Apple M2 Max, 96 GiB, macOS 27.0.1,
  in benchmark mode (`NAILA_BENCHMARK=1`, in-memory conversation history).
  - STT: faster-whisper `small.en`, CPU, int8, VAD filter off.
  - LLM: Meta-Llama-3.1-8B-Instruct Q4_K_M via llama.cpp, Metal (all layers
    on GPU), 9 threads, context 4096, max tokens 256, temperature 0.7,
    top_p 0.9, top_k 40, repeat penalty 1.1. Streaming.
  - TTS: Kokoro v1.0 (onnx), voice `bf_emma`, 24 kHz, speed 1.0; phrase cache on.
- **Network:** all three machines on one home LAN. The peer/collector machine
  is separate from the AI server (`colocated_with_ai: false`).
- **Source:** revision `cc8a2bb` on `feat/pi-hub-deploy`. The manifest's
  `dirty: true` reflects untracked local tooling files, not tracked changes.

## Results (20 measured trials)

| Measure | p50 | p95 | mean |
| --- | ---: | ---: | ---: |
| Hub: VAD end → first reply audio sent (`hub_response_first_send_ms`) | 1655 | 2025 | 1590 |
| Hub: VAD end → final reply audio sent | 3854 | 5843 | 4215 |
| Peer: first fixture frame → first reply audio received | 3338 | 3720 | 3278 |
| Peer: fixture end → first reply audio received (`peer_first_audio_ms`) | 758 | 1140 | 698 |
| STT | 844 | 904 | 848 |
| LLM time to first text | 63 | 78 | 66 |
| LLM total (diagnostic) | 277 | 372 | 290 |
| TTS, cache miss (n=14) | 524 | 873 | 588 |
| TTS, cache hit (n=6) | 0.4 | 0.5 | 0.4 |
| AI turn (includes paced audio output) | 3875 | 5863 | 4234 |
| Peer connection setup | 125 | 137 | 128 |
| Hub signaling/negotiation | 6 | 6 | 6 |

All values in milliseconds. Hub per-frame costs (per-trial means): Opus decode
~0.2 ms, VAD ~0.1 ms, outbound send ~0.1 ms.

Reading these numbers:

- `peer_first_audio_ms` is anchored at the end of the WAV, which includes about
  1.3 s of trailing silence, so it understates user-perceived wait. The hub
  figure is anchored at VAD end (500 ms after speech stops); end of speech to
  first reply audio is therefore roughly 2.1 s at p50.
- Six trials hit the TTS phrase cache on the first sentence (the same question
  repeats), which produces the 200–340 ms peer outliers.
- LLM token counts and tokens/second are null: llama-cpp-python 0.3.16 does not
  report usage on the streaming path (see #38 operator amendment, 2026-10-20).
- Peer timings include both network legs, Opus encode/decode and synthesized
  leading silence; they are not physical-device or audible-response latency.

## Files

- `manifest.json` — run configuration, provenance and deployment facts.

Host addresses were replaced with role names (`pi-hub`, `ai-server`). The derived
`summary.json` is not committed. The table above preserves the published results;
removed `samples.jsonl` and `traces.jsonl` remain in git history at `3e54959`.
The CLI `summarize` command requires those inputs and the original execution-finalization
record in `summary.json`, so it cannot certify this reduced archive.
