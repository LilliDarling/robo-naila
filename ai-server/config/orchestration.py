"""Orchestrator graph routing thresholds.

Tuning dials for the LangGraph orchestrator's routing decisions — when to
skip memory recall, how many exchanges to load, and the minimum classifier
confidence before an action handler is allowed to fire. Each is overridable
via env var for A/B testing without code changes; defaults match the v1
design doc (docs/INTELLIGENCE_ARCHITECTURE.md §8 lists these as measurement
targets).
"""

import os


# Above this classifier confidence on a simple-intent turn (greeting,
# time_query), skip the SQLite recall — history adds no signal and the
# roundtrip is wasted work.
RECALL_SKIP_CONFIDENCE = float(os.getenv("NAILA_RECALL_SKIP_CONFIDENCE", "0.8"))

# Max exchanges loaded from memory per turn. The LLM service trims further
# (see llm_config.CONTEXT_HISTORY_LIMIT) — this is the upper bound at the
# memory layer, not the prompt-context layer.
HISTORY_LIMIT = int(os.getenv("NAILA_HISTORY_LIMIT", "10"))

# Below this STT confidence the transcript is too uncertain to act on;
# action handlers won't fire and the LLM/clarification path takes over.
ACTION_CONFIDENCE_FLOOR = float(os.getenv("NAILA_ACTION_CONFIDENCE_FLOOR", "0.3"))

# Below this intent-classifier similarity, action handlers won't fire and the
# LLM answers. Calibrated on all-MiniLM-L6-v2: real weather/time requests score
# >= 0.49 ("What's the temperature?"), near misses up to 0.44 ("What color is
# the sky?" 0.415 used to fetch the weather). Keyword-fallback intents carry no
# score and are not gated.
ACTION_INTENT_FLOOR = float(os.getenv("NAILA_ACTION_INTENT_FLOOR", "0.47"))
