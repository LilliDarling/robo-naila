"""Unit tests for NAILAOrchestrator's memory wiring.

Locks in the DI contract: orchestrator receives a ConversationMemory via
constructor and calls ``commit_exchange`` after the graph completes a turn.
Recall now happens inside the graph's ``retrieve_context`` node — see
``test_orchestration_graph.py``.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from agents.orchestrator import NAILAOrchestrator


@pytest.fixture
def memory():
    mem = Mock()
    mem.recall_recent.return_value = []
    return mem


@pytest.fixture
def orchestrator(memory, monkeypatch):
    orch = NAILAOrchestrator(memory=memory)

    # Replace the LangGraph run with a stub that simulates a complete turn.
    async def fake_run(state, config=None):
        state["processed_text"] = state.get("raw_input", "")
        state["response_text"] = "stubbed reply"
        state["intent"] = "general"
        state["response_metadata"] = {"intent": "general", "streamed": False}
        return state

    monkeypatch.setattr(orch.graph, "run", fake_run)
    return orch


class TestOrchestratorMemoryWiring:
    def test_constructor_stores_injected_memory(self, memory):
        orch = NAILAOrchestrator(memory=memory)
        assert orch.memory is memory

    def test_constructor_requires_memory(self):
        # Memory is a hard dependency — don't silently fall back to a global.
        with pytest.raises(TypeError):
            NAILAOrchestrator()

    @pytest.mark.asyncio
    async def test_orchestrator_does_not_recall_before_graph(self, orchestrator, memory):
        # Recall is the graph's job now (see retrieve_context node). The
        # orchestrator stays out of the way so we don't double-fetch.
        await orchestrator.process_task(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "hello"}
        )
        memory.recall_recent.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_called_after_graph_with_intent_split_out(
        self, orchestrator, memory
    ):
        await orchestrator.process_task(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "hello"}
        )
        memory.commit_exchange.assert_called_once()
        kwargs = memory.commit_exchange.call_args.kwargs
        args = memory.commit_exchange.call_args.args
        # Accept either positional or keyword device_id; require intent to be a kwarg
        # and metadata to be a dict (not None).
        assert ("dev-x",) == args[:1] or kwargs.get("device_id") == "dev-x"
        assert kwargs.get("intent") == "general"
        assert isinstance(kwargs.get("metadata"), dict)

    @pytest.mark.asyncio
    async def test_no_commit_when_response_is_empty(self, memory, monkeypatch):
        orch = NAILAOrchestrator(memory=memory)

        async def fake_run(state, config=None):
            # Graph errored out before producing a response; nothing to persist.
            return state

        monkeypatch.setattr(orch.graph, "run", fake_run)

        await orch.process_task(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "hello"}
        )
        memory.commit_exchange.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_metadata_round_trips_to_committed_exchange(
        self, memory
    ):
        """LangGraph's typed state must declare ``response_metadata`` or the field
        gets dropped before the orchestrator can read it. If that happens, every
        committed exchange ends up with empty metadata — losing the per-turn
        debugging fields the response_generator carefully assembles."""
        # Use the real graph (no monkeypatched run) so this test exercises
        # LangGraph's actual schema filtering. Stub out the agents inside.
        from unittest.mock import AsyncMock, patch

        async def fake_input(state):
            state["processed_text"] = state.get("raw_input", "")
            state["intent"] = "general"
            return state

        async def fake_response(state, config=None):
            state["response_text"] = "ok"
            state["response_metadata"] = {
                "intent": "general",
                "generation_time_ms": 42,
                "context_used": True,
                "streamed": False,
                "marker": "should_survive_graph",
            }
            return state

        orch = NAILAOrchestrator(memory=memory)
        with patch.object(orch.graph.input_processor, "process",
                          side_effect=fake_input), \
             patch.object(orch.graph.response_generator, "process",
                          side_effect=fake_response):
            await orch.process_task(
                {"task_id": "t1", "device_id": "dev-meta", "transcription": "hi"}
            )

        # The orchestrator commits ``metadata=result.get("response_metadata", {})``.
        # If LangGraph dropped the field, the call args would have ``metadata={}``.
        memory.commit_exchange.assert_called_once()
        committed = memory.commit_exchange.call_args.kwargs["metadata"]
        assert committed.get("marker") == "should_survive_graph", (
            f"response_metadata was dropped by LangGraph schema; got {committed!r}"
        )

    @pytest.mark.asyncio
    async def test_initial_state_seeds_empty_history_for_graph(
        self, memory, monkeypatch
    ):
        # The orchestrator passes an empty conversation_history into the graph;
        # retrieve_context fills it from memory. Asserting the seed protects
        # the contract — if someone ever puts data here pre-graph again, the
        # node would clobber it (or worse, double-fetch).
        orch = NAILAOrchestrator(memory=memory)

        captured = {}

        async def fake_run(state, config=None):
            captured["initial_state"] = dict(state)
            state["processed_text"] = state.get("raw_input", "")
            state["response_text"] = "ok"
            state["intent"] = "general"
            return state

        monkeypatch.setattr(orch.graph, "run", fake_run)

        await orch.process_task(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "now"}
        )

        assert captured["initial_state"]["conversation_history"] == []
        assert captured["initial_state"]["context"] == {}
