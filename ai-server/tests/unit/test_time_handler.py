"""Unit tests for the time_query action handler.

Reads the system clock and returns a templated response. No network, no LLM,
no async I/O. The handler is async only to match the substrate's uniform
``Awaitable[str]`` signature (doc §3.4).
"""

import inspect

import pytest
from freezegun import freeze_time

from agents.actions import time_handler


class TestTimeHandlerSignature:
    def test_handler_is_a_coroutine_function(self):
        # Substrate contract: every handler is ``async def``. Sync handlers like
        # this one just don't await anything inside.
        assert inspect.iscoroutinefunction(time_handler.handle)

    def test_handler_accepts_utterance_and_context(self):
        sig = inspect.signature(time_handler.handle)
        params = list(sig.parameters)
        assert params[:2] == ["utterance", "context"]


class TestTimeHandlerBehavior:
    @pytest.mark.asyncio
    @freeze_time("2025-01-15 14:30:00")
    async def test_returns_current_time_in_response(self):
        result = await time_handler.handle("what time is it?", {})
        # Don't pin the exact format — pin the user-visible signal: the time
        # string contains the hour and minute the clock shows.
        assert "2:30" in result

    @pytest.mark.asyncio
    @freeze_time("2025-01-15 14:30:00")
    async def test_returns_pm_indicator_for_afternoon(self):
        result = await time_handler.handle("what time is it?", {})
        assert "PM" in result.upper()

    @pytest.mark.asyncio
    @freeze_time("2025-01-15 09:05:00")
    async def test_returns_am_indicator_for_morning(self):
        result = await time_handler.handle("what time is it?", {})
        assert "AM" in result.upper()

    @pytest.mark.asyncio
    @freeze_time("2025-01-15 14:30:00")
    async def test_works_with_empty_context(self):
        # Handler must not depend on context entries — context is the carrier
        # for per-turn metadata, but time_query needs nothing from it.
        result = await time_handler.handle("what time is it?", {})
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    @freeze_time("2025-01-15 14:30:00")
    async def test_ignores_utterance_content(self):
        # No NLU on the utterance; the time is the time regardless of phrasing.
        a = await time_handler.handle("what time is it?", {})
        b = await time_handler.handle("tell me the time please", {})
        c = await time_handler.handle("", {})
        assert a == b == c


class TestRegistration:
    """The handler module registers itself at import time so the graph just has
    to import it during startup wiring."""

    def test_module_registers_on_import(self):
        from agents.actions import registry

        # Importing the handler module should have triggered registration.
        # We re-import to be explicit even though Python caches.
        import importlib
        from agents.actions import time_handler as th
        importlib.reload(th)

        assert registry.get("time_query") is not None
