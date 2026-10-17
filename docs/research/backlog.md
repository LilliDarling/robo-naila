# NAILA: ordered product-improvement backlog

**Prerequisites — complete before starting this backlog:**

1. Benchmarking and a reproducible baseline are in place.
2. Working back-and-forth voice conversation is restored on the intended deployment.

This document assumes those prerequisites; it does not certify their completion. Remove any item already fulfilled during that work. The order below is my recommended implementation priority, with explicit dependencies rather than a requirement to finish every preceding item.

Detailed findings, primary sources and caveats remain in the saved [rigor report](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md) and [SOTA lens report](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md). The entries below are **proposed work**, not established capabilities or promises of SOTA performance. Repo citations identify the implementation motivating each proposal; runtime severity remains unverified where noted in those reports.

## 1. Give each conversation its own state and timeline

**What changes:** Own buffering, output format, processing tasks and response generations per device/session. Carry session and response identity through output routing; discard obsolete generations after reconnects. Specify media-clock units and wraparound separately from capture time. Initially admit one device if concurrent sessions are outside scope. Completion means a second connection or an old response cannot replace the active conversation's state or audio.

**Why:** The server currently keeps one buffer/task per RPC while the hub tags multiple devices; output routing uses device identity, and the RTP timestamp is copied into a milliseconds field. These are source-backed risks, not reproduced incidents. [service.py:212](/Users/j/code/robo-naila/ai-server/rpc/service.py:212), [grpc.rs:286](/Users/j/code/robo-naila/hub/src/grpc.rs:286), [grpc.rs:397](/Users/j/code/robo-naila/hub/src/grpc.rs:397), [webrtc.rs:247](/Users/j/code/robo-naila/hub/src/webrtc.rs:247). Details: [rigor report §B](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#b-fix-stream-ownership-and-clock-contracts-before-optimizing-models).

**Depends-on:** None within this backlog.

**Blocking operator questions:** None for isolation and clock correctness. Concurrent-device support is a scope decision; single-device admission lets this item proceed without that decision.

## 2. Hear short requests and let thoughts finish

**What changes:** Improve AEC alignment/configuration and expose whether it is actually active. Adjust VAD, short-utterance handling and endpointing together so short “yes,” “no” and “stop” survive, playback echo does not become a request, and a hesitation does not prematurely end the user's thought. Add semantic/acoustic endpointing only if needed for that behavior.

**Why:** Defaults use aggressive VAD, 300 ms hangover and a 450 ms buffered-utterance cutoff; AEC has a fixed tail and passes through audio when unavailable. [vad.rs:18](/Users/j/code/robo-naila/hub/src/vad.rs:18), [service.py:42](/Users/j/code/robo-naila/ai-server/rpc/service.py:42), [aec.py:9](/Users/j/code/robo-naila/devices/audio-client/audio_client/aec.py:9). Details and oracle sources: [rigor report §E](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#e-tune-the-acoustic-and-turn-taking-operating-point-from-evidence).

**Depends-on:** 1.

**Blocking operator questions:** Which initial users/languages and acoustic setting should be supported? Should the conversational style favor waiting through hesitation or rapid command responses? These choices block the final tuning target, not AEC availability reporting or removal of demonstrated short-request loss.

## 3. Interrupt, correct and resume coherently

**What changes:** Add a conversational controller that distinguishes acknowledgment, continuation, correction and takeover. Wire accepted interruption through hub, inference and playback; clear obsolete queued audio and retain enough pending context to resume or reformulate. Completion means “no, I meant the other one” changes the answer being formed, while “mm-hmm” need not stop it.

**Why:** The hub event enum lacks Interrupt despite server/proto support; server cancellation is not equivalent to flushing playback. [audio.rs:35](/Users/j/code/robo-naila/hub/src/audio.rs:35), [service.py:345](/Users/j/code/robo-naila/ai-server/rpc/service.py:345). The proposed distinctions come from the [SOTA report's continuous-interaction design](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#1-make-speech-an-ongoing-interaction-with-acoustic-context-intact); delivery mechanics are in [rigor report §C](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#c-treat-interruption-as-a-transaction-extending-to-the-loudspeaker).

**Depends-on:** 1, 2.

**Blocking operator questions:** What counts as addressing NAILA? Which acknowledgments should preserve its turn? Should it resume automatically after an interruption, and how should background speech be treated?

## 4. Speak the first useful phrase promptly

**What changes:** Consume lazy STT inference inside the worker boundary and make warmup perform the intended decode, if the suspected execution behavior is confirmed. Decouple response completion from first-phrase delivery so the first sentence does not wait for a second one. Preserve continuous audio and correct finalization; avoid solving delay with unrelated filler speech.

**Why:** STT returns a segment generator from the executor and consumes it in the async caller; response delivery holds a pending sentence until another arrives or generation ends. [stt.py:393](/Users/j/code/robo-naila/ai-server/services/stt.py:393), [stt.py:424](/Users/j/code/robo-naila/ai-server/services/stt.py:424), [response_generator.py:250](/Users/j/code/robo-naila/ai-server/agents/response_generator.py:250). The STT stall is an inference requiring confirmation against the installed library. Details: [rigor report §D](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#d-investigate-avoidable-scheduling-delay-before-a-model-replacement).

**Depends-on:** 1; coordinate completion/cancellation semantics with 3.

**Blocking operator questions:** None. These are implementation changes preserving the chosen conversational behavior.

## 5. Remember what was delivered

**What changes:** Record generated content separately from audio submitted for playback and playback progress acknowledged by the device. Give interrupted turns an explicit partial-delivery state. Construct future context from that distinction, so NAILA does not assume an unheard explanation was communicated. Device playback acknowledgment is evidence of rendering, not proof of human comprehension.

**Why:** Memory currently commits the complete response after graph return without a playout acknowledgment in that method. [orchestrator.py:106](/Users/j/code/robo-naila/ai-server/agents/orchestrator.py:106). Details: [rigor report §C](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#c-treat-interruption-as-a-transaction-extending-to-the-loudspeaker).

**Depends-on:** 1, 3.

**Blocking operator questions:** Should interrupted content remain searchable? What should NAILA repeat or summarize next time? Those choices block recall behavior; separating delivery states can proceed first.

## 6. Build revisable episodic memory

**What changes:** Extend the existing SQLite store with source-linked episodes, a compact working situation, explicit preferences and outstanding agreements. Distinguish user statements from observations and inferences; allow corrections to supersede old assertions and support forgetting. Retrieve relevant episodes beyond the most recent exchanges and consolidate outside the live speech path. Begin with one person and conversational episodes.

**Why:** Memory is currently append-only, keyed by device and recalled by recency. [conversation.py:21](/Users/j/code/robo-naila/ai-server/memory/conversation.py:21), [conversation.py:48](/Users/j/code/robo-naila/ai-server/memory/conversation.py:48). The proposal adapts the MemGPT/Generative Agents patterns described in the [SOTA report's continuity section](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#3-make-personality-a-way-of-participating-with-continuity-across-episodes).

**Depends-on:** 5. Visual episodes can be added after 7.

**Blocking operator questions:** Who is remembered on a shared device? What may be retained or inferred, for how long, and how does the person review/delete/correct it? Which agreements should persist? These block persistent personal-memory behavior, not a transient working situation.

## 7. Share attention to one visible object

**What changes:** Preserve frame identity/time, boxes and selected crops alongside descriptions. Track a focused object across observations; distinguish current observation, last-seen information, user assertion and inference. Let “that one,” “before” and “the other one” update the shared referent; clarify ambiguity. Use a local image-language model only for questions beyond detector evidence, and make unsupported visual claims explicit.

**Why:** The graph clears image data after analysis; the LLM receives description/counts and visual answers use presence/count rules. [orchestration.py:83](/Users/j/code/robo-naila/ai-server/graphs/orchestration.py:83), [response_generator.py:184](/Users/j/code/robo-naila/ai-server/agents/response_generator.py:184), [vision.py:591](/Users/j/code/robo-naila/ai-server/services/vision.py:591). Details: [rigor report §F](/Users/j/code/robo-naila/docs/research/2026-10-17-rigor-report.md#f-preserve-evidence-and-uncertainty-at-the-vision-boundary), [SOTA report shared situation](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#2-give-it-a-shared-situation-beginning-with-attention-to-one-object).

**Depends-on:** 1, 3. Persistent recall of observations also depends on 6.

**Blocking operator questions:** Which one-room/camera experience and visual questions matter first? When is capture permitted, and may frames/crops persist? Is the existing host's memory a firm ceiling? These block camera policy and model selection; evidence provenance can be designed first.

## 8. Keep listening while slower work proceeds

**What changes:** Separate immediate participation from reasoning, memory retrieval, image interpretation and existing actions. Both use one versioned session state; results derived from an outdated referent or request cannot become a current spoken answer. Continue accepting corrections while deliberating, and use silence as a valid immediate response. Retain the current service boundaries rather than adding independent agents for each responsibility.

**Why:** The current graph is sequential. The SOTA proposal combines immediate listening/speaking with deliberation around shared state, informed by Qwen's Thinker/Talker separation. [orchestration.py:48](/Users/j/code/robo-naila/ai-server/graphs/orchestration.py:48). Details, diagram and integration caveats: [SOTA report architecture](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#4-separate-immediate-participation-from-slower-thought-around-one-shared-state).

**Depends-on:** 1, 3, 5. Integrate 6 and 7 as those capabilities land.

**Blocking operator questions:** None for stale-result cancellation and concurrent listening. Whether NAILA may issue anticipatory acknowledgments or speak before an answer is settled is a separate product decision.

## 9. Explore continuous speech and contextual personality

**What changes:** Time-box a speech-native backend exploration behind the existing transport/session controller; preserve bounded recent user/assistant audio and silence timing. Moshi MLX is the initial local reference, PersonaPlex the role/voice-control reference. Choose an integration path only if it supports the intended interaction and hardware envelope. For a retained cascade, explore context-conditioned speech generation and a speech plan carrying intent, certainty and emphasis. Keep one recognizable personality across corrections and pauses.

**Why:** The current task payload carries transcript/confidence, the prompt defines personality textually, and the TTS interface exposes text/voice/speed. [service.py:665](/Users/j/code/robo-naila/ai-server/rpc/service.py:665), [system.txt:1](/Users/j/code/robo-naila/ai-server/prompts/system.txt:1), [tts.py:162](/Users/j/code/robo-naila/ai-server/services/tts.py:162). Native speech and contextual TTS are alternative mechanisms, not guaranteed interchangeable components. References and limits: [SOTA report speech substrate](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#1-make-speech-an-ongoing-interaction-with-acoustic-context-intact), [personality section](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#3-make-personality-a-way-of-participating-with-continuity-across-episodes).

**Depends-on:** 1, 2, 3. Production adoption must preserve 5 and integrate with 8; it does not block 6 or 7. Choose the native versus cascade path before investing in backend-specific voice adaptation.

**Blocking operator questions:** Is the M4 Air/16 GB a firm limit? May reasoning breadth trade off against speech naturalness? What voice, personality boundaries and uses of emotional interpretation are wanted? Hardware/behavior targets block adoption; retention decisions block persistent acoustic history.

## 10. Offer selective, invited initiative

**What changes:** Add a narrow “watch this with me” mode for the chosen shared activity. Scene changes update state silently by default; NAILA speaks when an agreed trigger or relevant unresolved commitment warrants it, and can yield immediately. Maintain task progress across pauses. Begin with one activity and explicit activation.

**Why:** The SOTA report derives speak/silence decisions and progress summaries from ProAssist, using shared attention and memory rather than continuous scene narration. [SOTA report architecture and initiative](/Users/j/code/robo-naila/docs/research/2026-10-17-sota-lens-report.md#4-separate-immediate-participation-from-slower-thought-around-one-shared-state). Existing turn state has input/response/history fields but no explicit shared-activity progress or initiative policy. [states.py:6](/Users/j/code/robo-naila/ai-server/graphs/states.py:6).

**Depends-on:** 3, 6, 7, 8. Does not require adopting the experimental backend in 9.

**Blocking operator questions:** Which activity, activation/exit behavior and intervention triggers? How assertive may it be, and when must it remain silent? Those are necessary product choices before implementing initiative.

This backlog intentionally carries forward product changes rather than instrumentation projects, benchmark integrations or a model-shopping list. No implementation or project checks were performed for this handoff; only the requested report/status artifacts are updated.
