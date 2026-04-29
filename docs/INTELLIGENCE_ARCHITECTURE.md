# NAILA Intelligence Architecture

## 1. Thesis

NAILA is a personal voice assistant being built iteratively. The big-picture vision — multi-user, multi-room, vision-aware, coaching, automation, security — is real and named, but it is not what this document designs.

This document describes the **smallest version of NAILA that is genuinely useful**, and the architecture that lets it grow into the bigger vision without refactor. The bet:

> Ship a system that holds a conversation, answers basic commands like weather and time, and remembers what was just said across restarts. Build every later capability as an additive layer on top of that, with the additive path named in advance.

---

## 2. What NAILA does in v1

Three end-to-end behaviors. Each maps to a specific code path; everything in §3 and §4 exists because one of these requires it.

### 2.1 Hold a conversation that survives restart

> User: "I'm thinking about getting a dog."
> NAILA: "Oh nice — what kind are you considering?"
> *[server restart]*
> User: "Actually, what was that breed I mentioned earlier?"
> NAILA: "You hadn't named one yet — just said you were thinking about getting a dog."

The exchange before restart is persisted. On the next turn, recent exchanges are loaded into the LLM's context. Continuity holds.

### 2.2 Tell me the weather (async action)

> User: "What's the weather?"
> NAILA: "It's 62°F and partly cloudy in [configured location]."

Intent classifier identifies `weather_query`. The action handler calls a weather API, awaits the response, formats a templated reply. **No LLM call.** TTS plays the templated reply.

### 2.3 Tell me the time (sync action)

> User: "What time is it?"
> NAILA: "It's 3:47 PM."

Intent classifier identifies `time_query`. The action handler reads the system clock and returns a templated reply. **No network, no LLM.**

These three behaviors are the entire functional scope of v1.

> **Note on existing code.** Today, `time_query` and `weather_query` produce templated strings inside `_generate_base_response` ([response_generator.py:395-406](../ai-server/agents/response_generator.py#L395-L406)) — but only as the *fallback* after the LLM path is tried first. v1 reorganizes this so action handlers run *before* the LLM path and short-circuit it, matching the substrate boundaries described in §3.

---

## 3. Architecture

Three substrates the harness routes between.

```
┌──────────────────────────────────────────────────────────────┐
│                       HARNESS (LangGraph)                    │
│  Per-turn control flow. Routes to substrate based on intent. │
└──────────────────────────────────────────────────────────────┘
         │
   ┌─────▼──────────┐  ┌──────────────────┐  ┌────────────────┐
   │   HEURISTIC    │  │      LLM         │  │     ACTION     │
   │   Intent       │  │   Conversation   │  │   Sync + async │
   │   classifier,  │  │   response gen   │  │   handlers     │
   │   recall       │  │   (existing)     │  │                │
   └────────────────┘  └──────────────────┘  └────────────────┘
            │                  │                    │
            └──────────┬───────┴────────────────────┘
                       │
              ┌────────▼─────────┐
              │  MEMORY HARNESS  │
              │  Append + recall │
              └──────────────────┘
                       │
              ┌────────▼─────────┐
              │  SQLite (one     │
              │  file, FTS5)     │
              └──────────────────┘
```

### 3.1 The harness

The existing LangGraph orchestrator. The v1 graph topology:

```
process_input → process_vision → retrieve_context → dispatch_action → generate_response → execute_actions → END
```

Per-turn flow:

1. **`process_input`** — STT result + intent classification (existing). Sets `state["intent"]`.
2. **`process_vision`** — YOLOv8 if image data present (existing).
3. **`retrieve_context`** — calls `memory_harness.recall_recent(device_id, n)` to load recent exchanges into `state["context"]["recent_exchanges"]`. Today this node is a near-no-op ([orchestration.py:85-114](../ai-server/graphs/orchestration.py#L85-L114)); v1 fills it in. The existing short-circuit ("skip recall for `time_query` + high-confidence `greeting`") is preserved — recall is wasted work for action intents.
4. **`dispatch_action`** *(new node)* — looks up `state["intent"]` in the action registry. If registered, awaits the handler, sets `state["response_text"]` and `state["action_handled"] = True`. If not registered, passes through unchanged.
5. **`generate_response`** — if `state.get("response_text")` is already set (action handler populated it), skip LLM and streaming; run only the existing non-streaming TTS path on the action text. Otherwise, the conversational path: streaming TTS via `_stream_response_to_audio` (existing).
6. **`execute_actions`** — post-response logging hook (existing). Stays available for side-effects that happen after the response is delivered.
7. After the graph returns, the response_generator's existing memory write is replaced with a call to `memory_harness.commit_exchange(...)`.

**Why dispatch in its own node, not as a conditional edge:** TTS lives inside `generate_response` ([response_generator.py:143-156](../ai-server/agents/response_generator.py#L143-L156) and the streaming path at [response_generator.py:221-231](../ai-server/agents/response_generator.py#L221-L231)). Routing actions around `generate_response` would require duplicating the TTS path. Routing through it with a short-circuit keeps TTS in one place.

**⚠ Implementation risk — streaming TTS short-circuit interaction.** The "skip LLM/streaming when `response_text` is set" check has to be the **very first** thing `generate_response.process()` does, before any of the streaming setup at [response_generator.py:74-84](../ai-server/agents/response_generator.py#L74-L84) runs. The streaming path is deeply entangled with `audio_delivery` callbacks, `is_final` lookahead, and per-sentence delivery timing — if any streaming-state setup runs unconditionally before the short-circuit check, action responses will trigger streaming-path side effects on text that has no LLM stream behind it. Symptoms would be subtle: missing audio, `is_final` flag set wrong, callback fired with the wrong arguments, or stream lookahead getting confused by a complete (non-streaming) response. The short-circuit is a single early-return at the top of `process()` — keep it that way; resist the temptation to slot it deeper in the function.

LangGraph is in place from prior work and is not re-evaluated for v1. If it becomes a constraint later, swapping the orchestrator is a focused refactor; the substrate boundaries stay intact.

### 3.2 Heuristic substrate

Pure-function logic. Two roles in v1:

- **Intent classification.** Existing `input_processor.py` — embedding similarity against pre-computed intent embeddings. Extends to new intents (`get_weather`, `get_time`) by adding entries to `intent_embeddings`.
- **Memory recall.** SQL queries with `LIMIT N ORDER BY ts DESC` for recent exchanges, FTS5 for keyword match.

### 3.3 LLM substrate

Used only for conversational reply generation. Streaming TTS path (already shipped) is unchanged.

The LLM is invoked **once per conversational turn**. Action handlers do not call the LLM.

### 3.4 Action substrate

A registry of handlers, each keyed by intent name.

**Handler signature (uniform sync + async):**

```python
ActionHandler = Callable[[str, dict], Awaitable[str]]
# All handlers are async def. Sync work just doesn't await.
# (utterance, context) -> templated response string

async def handler(utterance: str, context: dict) -> str:
    ...
```

- `utterance` is the raw user text. Handlers do their own param extraction (e.g., parsing "in Boston" out of "what's the weather in Boston"). Default values when extraction fails.
- `context` carries `device_id` and any other per-turn metadata the handler needs.
- Return value is the response string sent to TTS.

**Registration:** action modules call `register(intent_name, handler)` at import time. v1 has two: `weather_query`, `time_query`. Intent names match the existing classifier output ([input_processor.py:118-126](../ai-server/agents/input_processor.py#L118-L126)). Adding a third is one new module + one register call + one intent embedding seed.

**Error handling:** handlers catch their own errors and return a templated error string ("I couldn't reach the weather service right now."). The exchange is still committed; metadata records the failure for later debugging.

**Why uniform async:** every handler is `async def` so the orchestrator awaits uniformly. Sync handlers (like `get_time`) just don't `await` anything inside. This means adding a new async action later (e.g., calendar lookup) doesn't require restructuring the dispatch path.

**What v1's action substrate does NOT include but does NOT preclude:**

- Fire-and-forget actions (e.g., timers that fire later) — added by introducing a second handler type that returns immediately and schedules its real work. Doesn't change the synchronous-result pattern.
- Per-action authorization or per-user permissions — additive, not restructuring.
- Tool registration via decorator or external manifest — additive, current `register()` call is the foundation.

### 3.5 Memory harness

Two operations in v1:

| Operation | Purpose | Substrate |
| --- | --- | --- |
| `commit_exchange(device_id, user_msg, assistant_msg, intent, metadata)` | Append a new turn | Heuristic |
| `recall_recent(device_id, n=10)` | Get last N exchanges for this device, ordered by ts desc | Heuristic |

Both are synchronous SQL queries. No background workers, no LLM use, no session boundaries.

**Implementation note — replacing `ConversationMemory`:** The existing `ai-server/memory/conversation.py:ConversationMemory` is in-RAM. v1 replaces it with a SQLite-backed implementation under the same class name. The new API is **clean and singular** — just `commit_exchange` and `recall_recent`. The old methods (`add_exchange`, `get_history`, `get_context`) are removed. Tests are rewritten to the new API as part of v1 work; we explicitly accept this short-term test churn rather than preserve a dual-API surface as long-term debt. The constructor takes a `db_path` parameter; tests pass `:memory:` (in-memory SQLite, fast and isolated), production passes a configured file path.

**Drop the module-level singleton.** Today `memory_manager = ConversationMemory()` lives at module scope in `conversation.py:224`. This pattern forces tests to monkeypatch `_start_background_cleanup` ([conftest.py:117](../ai-server/conftest.py#L117)) and conflicts with TDD ethos. v1 removes the global and constructs `ConversationMemory(db_path=...)` in `main.py`, then injects it into `NAILAOrchestrationGraph` and any agent that needs it via constructor — matching the existing wiring pattern for `llm_service`, `tts_service`, `vision_service`.

**Why no sessions in v1:** the recall query is bounded by `LIMIT N`, not by session windowing. Adding session boundaries solves a problem that doesn't exist at this scope (grouping multi-day conversations) and introduces a sweep race that has to be handled. Sessions become useful when summaries, multi-device continuity, or coaching require the grouping. Until then, "last 10 exchanges, regardless of when" is sufficient and simpler.

**Why `device_id` and not `user_id`:** v1 is single-user. `device_id` is the natural scoping key today. When voice biometrics arrives, `user_id` is added via additive migration (§5.2), and the recall function gains a parameter.

### 3.6 Code being removed in v1

This section catalogs existing code that becomes dead with v1 and should be removed alongside the v1 work — not left as drift. We are doing TDD from v1 forward; tests are rewritten to the new API as part of the cutover, not preserved as ballast.

**`ai-server/memory/conversation.py` — most of the file.** The class shell stays; nearly every method body and instance attribute is replaced or removed:

- `add_exchange`, `get_history`, `get_context` — replaced by `commit_exchange`, `recall_recent`.
- `_start_background_cleanup`, `_background_cleanup`, `cleanup_old_conversations` — TTL/cleanup is deferred; retention policy is not a v1 concern (§6).
- `_active_devices`, `_cleanup_task`, `last_cleanup`, `cleanup_interval`, `device_metadata` — supporting state for the removed cleanup machinery.
- `clear_device`, `cleared`, `_update_device_metadata` — usage drops with the methods above.
- `get_memory_stats`, `shutdown` — no longer applicable.
- `self._lock` (threading.RLock) — single-process SQLite handles concurrency at the DB layer; not needed.
- `memory_manager = ConversationMemory()` module-level singleton — removed in favor of constructor injection (see §3.5).

**`ai-server/agents/orchestrator.py`** — call sites updated:

- Import of `memory_manager` removed; replace with constructor injection.
- `self.memory.get_context(device_id)` at line 79 — removed (the `retrieve_context` graph node now does this work centrally).
- `self.memory.add_exchange(...)` at line 109 — replaced with `self.memory.commit_exchange(...)`.

**`ai-server/agents/response_generator.py`** — two specific dead branches:

- Lines 399-400: `time_query` and `weather_query` entries in the `responses` dict inside `_generate_base_response`. These intents are now action-handled before `generate_response` runs; the dict entries are unreachable and removed.
- `_generate_followup_response` (lines 365-372) **stays** — it still handles the case where a `question` follows a `time_query` or `weather_query` ("what about tomorrow?" after weather). Action handlers don't preempt this path because the followup intent is `question`, not the original action intent.

**`ai-server/graphs/orchestration.py`** — `_retrieve_context` body (lines 96-108): the placeholder "create new context dict, stamp `context_retrieved=True`" is replaced with the `recall_recent` call. The short-circuit at lines 92-94 (skip recall for high-confidence simple intents) stays.

**Test fixtures and call sites — rewritten to new API:**

- [conftest.py:113-182](../ai-server/conftest.py#L113-L182) — fixtures `clean_memory`, `populated_memory`, `memory_with_history`. Drop the `_start_background_cleanup` monkeypatch (no longer needed; cleanup machinery is gone).
- [tests/fixtures/memory_fixtures.py](../ai-server/tests/fixtures/memory_fixtures.py) — all `add_exchange` calls become `commit_exchange`.
- [tests/e2e/test_end_to_end.py](../ai-server/tests/e2e/test_end_to_end.py) — ~14 call sites; rewrite as part of v1 to use the new API. This is where most of the test churn lands.
- [tests/integration/test_mqtt_integration.py:400-413](../ai-server/tests/integration/test_mqtt_integration.py#L400-L413) — same.
- [tests/performance/test_performance.py:36](../ai-server/tests/performance/test_performance.py#L36) — same.

**What stays:**

- `_response_cache` (TTLCache) in `response_generator.py:16` — internal LLM-substrate optimization, unaffected.
- `_stream_response_to_audio` and the streaming TTS path — preserved for genuinely conversational LLM responses.
- The greeting / question / gratitude / goodbye / general branches in `_generate_base_response` — still used as LLM fallback for those intents.
- `intent_embeddings` setup in `input_processor.py` — still used; gains new entries via action registration.
- `_execute_actions` node in `orchestration.py:134-139` — stays as a post-response logging hook.

---

## 4. Database

### 4.1 Storage

SQLite, one file at a configurable path. WAL mode set on first open. Foreign keys enabled per connection.

The file lives wherever the AI server runs. v1 deployment is the developer's machine; production deployment lives with the AI server (not the Pi).

### 4.2 Schema

```sql
PRAGMA user_version = 1;

CREATE TABLE exchanges (
  exchange_id    INTEGER PRIMARY KEY,
  device_id      TEXT NOT NULL,
  ts             INTEGER NOT NULL,
  user_msg       TEXT NOT NULL,
  assistant_msg  TEXT NOT NULL,
  intent         TEXT,
  metadata       TEXT
);
CREATE INDEX idx_exchanges_device_ts ON exchanges(device_id, ts);

CREATE VIRTUAL TABLE exchanges_fts USING fts5(
  user_msg, assistant_msg,
  content='exchanges', content_rowid='exchange_id'
);

CREATE TRIGGER exchanges_ai AFTER INSERT ON exchanges BEGIN
  INSERT INTO exchanges_fts(rowid, user_msg, assistant_msg)
  VALUES (new.exchange_id, new.user_msg, new.assistant_msg);
END;
CREATE TRIGGER exchanges_ad AFTER DELETE ON exchanges BEGIN
  INSERT INTO exchanges_fts(exchanges_fts, rowid, user_msg, assistant_msg)
  VALUES ('delete', old.exchange_id, old.user_msg, old.assistant_msg);
END;
CREATE TRIGGER exchanges_au AFTER UPDATE ON exchanges BEGIN
  INSERT INTO exchanges_fts(exchanges_fts, rowid, user_msg, assistant_msg)
  VALUES ('delete', old.exchange_id, old.user_msg, old.assistant_msg);
  INSERT INTO exchanges_fts(rowid, user_msg, assistant_msg)
  VALUES (new.exchange_id, new.user_msg, new.assistant_msg);
END;
```

One table, one FTS index, three triggers. That is the entire v1 schema.

**Why FTS5 in v1:** "remember basic info" includes the case of "what did I tell you about X." Keyword search is the cheapest way to support that and ships with SQLite. Vector recall is an additive upgrade (§5.4).

**The `metadata` column** is a JSON dumping ground for per-exchange ad-hoc fields. No formal promotion discipline at this scale; if a key becomes load-bearing, promote it to a column when it does. Don't pre-engineer the audit cadence.

### 4.3 Connection setup

One helper function. Single source of truth for connection-level PRAGMAs:

```python
# ai-server/memory/db.py
def open_connection(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn
```

All AI-server code goes through this helper. No CI grep rule, no enforced-invariants table — just a single helper used by convention. If a stray `sqlite3.connect()` shows up later and causes a problem, fix it then.

### 4.4 Schema versioning and migrations

Schema version tracked in `PRAGMA user_version`. v1 sets it to 1. Future migrations are numbered SQL files in `ai-server/memory/migrations/`, applied in order on startup if `user_version` is below the latest.

No migration framework yet — just an `apply_migrations()` function that compares versions and runs pending SQL files in a transaction. Add framework features (rollback, dry-run, integrity checks) when a migration breaks something and you wish you'd had them.

---

## 5. How this grows

Each future capability is paired with the additive path that lands it. None of these requires restructuring v1 data.

### 5.1 Explicit fact storage ("remember that I prefer tea")

**What it adds:** storage of structured facts the user explicitly wants remembered, surfaced into the LLM's system prompt every conversational turn.

**What lands:**
- New table `facts (fact_id, device_id, fact_text, created_at, updated_at, source_exchange_id, metadata)`. FK to `exchanges.exchange_id` for source attribution.
- New intent `remember_fact` in the intent classifier.
- New action handler `remember_fact_handler` that extracts the fact (rule-based parsing of "remember that X" or LLM extraction; choose when shipping) and writes the row.
- Modification to the conversational LLM path to load active facts into the system prompt.

**No v1 schema changes.** Only additive.

### 5.2 Multi-user identity (voice biometrics)

**What it adds:** different users in the household get different memory contexts; recall scopes by user, not by device.

**What lands:**
- New table `users (user_id, display_name, voice_embedding, created_at)`.
- `ALTER TABLE exchanges ADD COLUMN user_id TEXT REFERENCES users(user_id)`. Backfill existing rows to a default user.
- Voice biometric component (separate ML system) populates `user_id` on new exchanges.
- `recall_recent` gains a `user_id` parameter; existing call sites pass `None` until biometrics ships.

**Not a refactor.** A column addition, a backfill, and a parameter addition. ~2 hours of schema work.

### 5.3 Scheduled / fire-and-forget actions (timers, reminders)

**What it adds:** "set a timer for 10 minutes" — action handler returns immediately, real work fires later.

**What lands:**
- New table `pending_actions (action_id, intent, params_json, fire_at, status, created_at, completed_at)`.
- New handler type that returns a confirmation string immediately and inserts a `pending_actions` row.
- A scheduled worker (cron-style) that wakes on `fire_at` and executes the deferred work.

**Doesn't change v1 substrate signature.** The current `(utterance, context) -> string` shape still applies; the deferred handler returns its string before the real work runs.

### 5.4 Vector recall (semantic search)

**What it adds:** "remind me what we discussed about gardening last week" works even if the keyword doesn't match.

**What lands:**
- `ALTER TABLE exchanges ADD COLUMN embedding BLOB`.
- `sqlite-vec` extension loaded.
- Embedding generation on `commit_exchange` (synchronous; uses the sentence-transformer already loaded for intent classification, or a dedicated memory model — open).
- Backfill job for existing rows (runs in background after migration).
- `recall_similar(device_id, query, k)` operation added to the memory harness.

**During backfill window:** exchanges with `NULL` embedding fall back to FTS5 + recency. The recall function checks for the column being populated before relying on it.

**Sentence-transformer ownership:** today the model is owned by the `InputProcessor` instance ([input_processor.py:108-132](../ai-server/agents/input_processor.py#L108-L132)). When vector recall ships, the model becomes a shared resource: either lifted to a service injected into both `InputProcessor` and the memory harness, or duplicated (cheap on a multi-GB GPU, less cheap on tighter hardware). Decision deferred — it's a v2 concern, not a v1 blocker.

### 5.5 Scaling LLM concurrency

When concurrent users on the hot path become routine, or background LLM workers ship:

- **Model-call broker.** Priority queuing (hot path > background), preemption, fair queuing across concurrent hot-path requests. Intercepts between callers and the LLM service.
- **Continuous batching via server-mode inference.** llama.cpp server mode, vLLM, or equivalent. Multiple concurrent requests served in parallel through batched GPU calls. Real software lift but well-trodden.

These are answers to "can the architecture handle X" questions. Not v1 work. Listed so the answer is "yes, here's how" rather than "we'll figure it out."

### 5.6 Vision, coaching, automation, security, message delivery

Each is its own design doc when its time comes. The v1 architecture supports all of them via the same pattern: new tables that FK to existing v1 data, new substrates that the harness routes to, new intents that extend the classifier. None requires v1 refactor.

The single architectural bet that *would* break if violated: that the unit of conversation history is per-(device, user, time) and that exchanges are append-only. Anything requiring mutable exchanges or fundamentally different history grouping would be a real refactor.

---

## 6. Out of v1 (named, not designed)

| Capability | Why deferred | Reactivation trigger |
| --- | --- | --- |
| Sessions / session lifecycle | Recall is bounded by LIMIT, not window | Summaries needed; multi-device continuity tracked explicitly; coaching requires grouped units |
| Multi-user accommodation | Single user today | Voice biometrics ship |
| Encryption at rest | OS-level disk encryption assumed for v1 | Multi-user OR deployment to less-trusted host OR threat model defined |
| Backup automation | Manual `cp` for now | Daily use proves the data is irreplaceable |
| Retention policy | Volume not yet a concern | Disk pressure measured OR retention surfaces as a privacy question |
| Discipline scaffolding (CI rules, quarterly reviews) | Single-developer scale | Second engineer joins OR feature drift becomes audible |
| Background LLM workers | No distillation use cases yet | Coaching, summarization, or fact extraction ships |
| Knowledge graph | No patterns to detect yet | Coaching arrives |
| Concurrent multi-user inference | Single-user load | Two people genuinely converse simultaneously |

Each row is a deliberate cut, not an oversight. When the trigger fires, that capability gets its own design pass — not a retroactive expansion of this doc.

---

## 7. Failure modes

### 7.1 SQLite locked
WAL mode plus `busy_timeout = 5000` handles serialized writes. v1 has no concurrent writers; this is rare. If the timeout exhausts, the write logs an error and is dropped — the user-facing path is not blocked.

### 7.2 Action handler failure
Handlers catch their own errors and return a templated error string. The exchange is committed with the error reflected in `metadata`. The user gets a coherent response; debugging info lives in the exchange record.

### 7.3 LLM unavailable / timeout
Existing fallback chain (streaming TTS → non-streaming TTS) handles most TTS failures. For LLM unavailability, response generation falls back to a templated apology. Should be rare; should be measured.

### 7.4 FTS5 desync
If a write to `exchanges` succeeds but the FTS5 trigger fails, keyword recall silently returns wrong results. Mitigation: startup check compares row counts between `exchanges` and `exchanges_fts`. Mismatch logs a warning and rebuilds the FTS index automatically (`INSERT INTO exchanges_fts(exchanges_fts) VALUES('rebuild')`).

### 7.5 Disk space exhaustion
Writes will fail. v1 exchange volume is small; this is unlikely soon. Long-term, monitoring + a degraded mode (warn the user, refuse new writes below threshold) addresses this when retention policy is designed.

### 7.6 Session sweep race
**Not applicable in v1** — there are no sessions, so no sweep, so no race. When sessions are added (§5.1 or beyond), the sweep-and-close must use atomic SQL with the staleness condition in the WHERE clause, not a Python-layer check-then-close.

---

## 8. Measurements

Once v1 ships, the following are measured to inform later decisions:

| Metric | What it informs |
| --- | --- |
| `commit_exchange` p50/p99 latency | Whether persistence is on the critical path |
| `recall_recent` p50/p99 latency at growing exchange volume | When indexing or session windowing becomes worth adding |
| Exchanges per day | When retention policy stops being deferrable |
| Database file size growth | When backup/sync becomes a cost concern |
| Action handler latency (per intent) | Whether weather/time meet "feels instant" |
| Number of LLM calls per turn | Whether the routing discipline is holding |
| End-to-end turn latency before/after v1 | That v1 doesn't regress conversation feel |

No specific thresholds set in advance. Initial measurements establish baselines; later passes set targets.

---

## 9. What this doc is not

- **Not** an implementation spec. Function signatures in §3.4 and §3.5 are the API surface; structure of the modules is for the implementer to choose.
- **Not** a roadmap with dates. §5 lists what's possible and what triggers each step. Order and timing are decided when the trigger fires.
- **Not** a privacy or threat-model document. v1 assumes OS-level disk encryption and trusted host. A real threat model is a separate document, written when single-developer scale is no longer the deployment context.
- **Not** complete. §7 lists known failure modes; unknowns remain unknowns. The goal is to be honest about what's been thought through and what hasn't, not to anticipate everything.

The doc's job is to make the bet from §1 explicit, define the v1 surface concretely enough to build, and name the additive paths for the bigger vision. Anything beyond that is over-engineering at this stage.
