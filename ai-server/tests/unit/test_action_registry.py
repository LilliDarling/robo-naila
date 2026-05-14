"""Unit tests for the action registry.

The registry is process-wide module state: handler modules call ``register``
at import time, the graph's ``dispatch_action`` node looks up handlers by
intent name. v1 has two intents in scope (``time_query``, ``weather_query``);
this test file pins the contract independent of any specific handler.
"""

import pytest

from agents.actions import registry


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test gets a clean registry — module state shouldn't leak across tests."""
    monkeypatch.setattr(registry, "_HANDLERS", {})


class TestRegister:
    def test_register_stores_handler_for_lookup(self):
        async def handler(utterance, context):
            return "ok"

        registry.register("time_query", handler)

        assert registry.get("time_query") is handler

    def test_register_replaces_previous_handler(self):
        async def first(utterance, context):
            return "first"

        async def second(utterance, context):
            return "second"

        registry.register("time_query", first)
        registry.register("time_query", second)

        assert registry.get("time_query") is second

    def test_handlers_for_different_intents_are_independent(self):
        async def time_handler(utterance, context):
            return "time"

        async def weather_handler(utterance, context):
            return "weather"

        registry.register("time_query", time_handler)
        registry.register("weather_query", weather_handler)

        assert registry.get("time_query") is time_handler
        assert registry.get("weather_query") is weather_handler


class TestGet:
    def test_unregistered_intent_returns_none(self):
        # ``dispatch_action`` relies on a None return to know the intent isn't
        # action-handled and should fall through to the LLM path.
        assert registry.get("definitely_not_registered") is None

    def test_get_with_empty_string_returns_none(self):
        assert registry.get("") is None

    def test_get_with_none_returns_none(self):
        # The graph passes ``state.get("intent")`` straight in; that can be None
        # before input_processor sets it. Don't crash, just say "no handler".
        assert registry.get(None) is None
