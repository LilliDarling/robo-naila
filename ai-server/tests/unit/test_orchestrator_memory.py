"""Unit tests for NAILAOrchestrator's memory wiring.

Locks in the DI contract: orchestrator receives a ConversationMemory via
constructor, calls ``recall_recent`` before the graph runs, and
``commit_exchange`` after.
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
    async def test_recall_called_before_graph_runs(self, orchestrator, memory):
        await orchestrator.process_task_with_callback(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "hello"}
        )
        memory.recall_recent.assert_called_once_with("dev-x", n=10)

    @pytest.mark.asyncio
    async def test_commit_called_after_graph_with_intent_split_out(
        self, orchestrator, memory
    ):
        await orchestrator.process_task_with_callback(
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

        await orch.process_task_with_callback(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "hello"}
        )
        memory.commit_exchange.assert_not_called()

    @pytest.mark.asyncio
    async def test_recalled_history_seeds_initial_state(
        self, memory, monkeypatch
    ):
        memory.recall_recent.return_value = [
            {"user": "earlier", "assistant": "yes", "intent": None, "ts": 1, "metadata": {}}
        ]
        orch = NAILAOrchestrator(memory=memory)

        captured = {}

        async def fake_run(state, config=None):
            captured["initial_state"] = dict(state)
            state["processed_text"] = state.get("raw_input", "")
            state["response_text"] = "ok"
            state["intent"] = "general"
            return state

        monkeypatch.setattr(orch.graph, "run", fake_run)

        await orch.process_task_with_callback(
            {"task_id": "t1", "device_id": "dev-x", "transcription": "now"}
        )

        # Recall result is exposed in the initial state so the graph nodes
        # (and downstream LLM message builder) can use it.
        history = captured["initial_state"]["context"]["recent_exchanges"]
        assert len(history) == 1
        assert history[0]["user"] == "earlier"
