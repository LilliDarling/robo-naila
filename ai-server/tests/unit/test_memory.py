"""Unit tests for SQLite-backed ConversationMemory.

The v1 API is intentionally narrow: ``commit_exchange`` and ``recall_recent``.
Older tests covered the in-RAM deque/cleanup machinery — those are gone.
"""

import json
import threading

import pytest

from memory.conversation import ConversationMemory


@pytest.fixture
def memory() -> ConversationMemory:
    """Fresh in-memory database — fast, isolated, and the conftest default."""
    return ConversationMemory(db_path=":memory:")


class TestConstruction:
    def test_in_memory_db_is_usable_immediately(self, memory):
        # Migrations should have run; commit must succeed without further setup.
        memory.commit_exchange("dev1", "hi", "hello", intent="greeting", metadata={})
        assert len(memory.recall_recent("dev1")) == 1

    def test_file_path_persists_across_instances(self, tmp_path):
        db_file = tmp_path / "naila.db"

        first = ConversationMemory(db_path=str(db_file))
        first.commit_exchange("dev1", "remember me", "ok", intent=None, metadata={})

        second = ConversationMemory(db_path=str(db_file))
        history = second.recall_recent("dev1")

        assert len(history) == 1
        assert history[0]["user"] == "remember me"
        assert history[0]["assistant"] == "ok"


class TestCommitExchange:
    def test_persists_user_and_assistant_messages(self, memory):
        memory.commit_exchange(
            "dev1", "what time is it?", "It's 3pm.", intent="time_query", metadata={}
        )
        rows = memory.recall_recent("dev1")
        assert len(rows) == 1
        assert rows[0]["user"] == "what time is it?"
        assert rows[0]["assistant"] == "It's 3pm."

    def test_stores_intent_at_top_level(self, memory):
        memory.commit_exchange(
            "dev1", "hi", "hello", intent="greeting", metadata={}
        )
        assert memory.recall_recent("dev1")[0]["intent"] == "greeting"

    def test_intent_can_be_none(self, memory):
        memory.commit_exchange("dev1", "x", "y", intent=None, metadata={})
        assert memory.recall_recent("dev1")[0]["intent"] is None

    def test_metadata_roundtrip(self, memory):
        meta = {"streamed": True, "generation_time_ms": 42, "nested": {"k": "v"}}
        memory.commit_exchange(
            "dev1", "x", "y", intent="general", metadata=meta
        )
        assert memory.recall_recent("dev1")[0]["metadata"] == meta

    def test_metadata_can_be_none(self, memory):
        memory.commit_exchange("dev1", "x", "y", intent=None, metadata=None)
        # ``None`` and ``{}`` should both surface as an empty dict on read so
        # callers don't have to guard.
        assert memory.recall_recent("dev1")[0]["metadata"] == {}

    def test_assigns_monotonically_increasing_timestamp(self, memory):
        memory.commit_exchange("dev1", "a", "1", intent=None, metadata={})
        memory.commit_exchange("dev1", "b", "2", intent=None, metadata={})
        memory.commit_exchange("dev1", "c", "3", intent=None, metadata={})
        rows = memory.recall_recent("dev1")
        # newest first — timestamps decrease (or are equal, but never lower-than-older)
        timestamps = [r["ts"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)


class TestRecallRecent:
    def test_empty_for_unknown_device(self, memory):
        assert memory.recall_recent("nobody") == []

    def test_returns_newest_first(self, memory):
        memory.commit_exchange("dev1", "first", "1", intent=None, metadata={})
        memory.commit_exchange("dev1", "second", "2", intent=None, metadata={})
        memory.commit_exchange("dev1", "third", "3", intent=None, metadata={})

        rows = memory.recall_recent("dev1")
        assert [r["user"] for r in rows] == ["third", "second", "first"]

    def test_default_limit_is_ten(self, memory):
        for i in range(15):
            memory.commit_exchange("dev1", f"u{i}", f"a{i}", intent=None, metadata={})
        assert len(memory.recall_recent("dev1")) == 10

    def test_respects_custom_limit(self, memory):
        for i in range(5):
            memory.commit_exchange("dev1", f"u{i}", f"a{i}", intent=None, metadata={})
        assert len(memory.recall_recent("dev1", n=2)) == 2

    def test_scopes_by_device_id(self, memory):
        memory.commit_exchange("dev1", "alpha", "1", intent=None, metadata={})
        memory.commit_exchange("dev2", "beta", "2", intent=None, metadata={})
        memory.commit_exchange("dev1", "gamma", "3", intent=None, metadata={})

        dev1 = memory.recall_recent("dev1")
        dev2 = memory.recall_recent("dev2")

        assert {r["user"] for r in dev1} == {"alpha", "gamma"}
        assert {r["user"] for r in dev2} == {"beta"}

    def test_returned_shape_has_required_keys(self, memory):
        memory.commit_exchange("dev1", "x", "y", intent="general", metadata={"k": 1})
        row = memory.recall_recent("dev1")[0]
        assert set(row.keys()) >= {"user", "assistant", "intent", "ts", "metadata"}


class TestFTSIntegration:
    """The FTS5 trigger fires on insert; v1 doesn't expose recall_similar yet,
    but we verify the index is wired so §5.4 can land additively."""

    def test_fts_index_populated_on_commit(self, memory):
        memory.commit_exchange(
            "dev1",
            "tell me about gardening",
            "Sure — what do you want to grow?",
            intent="question",
            metadata={},
        )
        # Reach into the connection for verification — production code never
        # queries FTS directly until §5.4.
        rows = memory._conn.execute(
            "SELECT user_msg, assistant_msg FROM exchanges_fts "
            "WHERE exchanges_fts MATCH 'gardening'"
        ).fetchall()
        assert len(rows) == 1
        assert "gardening" in rows[0][0]


class TestConcurrentWrites:
    """SQLite WAL + busy_timeout serializes writes; commit_exchange must be
    safe to call from threads without manual locking on our side."""

    def test_threads_committing_dont_corrupt(self, tmp_path):
        # File-backed DB so all threads share a real filesystem journal.
        db_file = tmp_path / "naila.db"
        memory = ConversationMemory(db_path=str(db_file))

        errors: list[Exception] = []

        def writer(device_id: str, n: int) -> None:
            try:
                local = ConversationMemory(db_path=str(db_file))
                for i in range(n):
                    local.commit_exchange(
                        device_id, f"u{i}", f"a{i}", intent=None, metadata={}
                    )
            except Exception as e:  # pragma: no cover — surfaces below
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"dev{i}", 20)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"errors during concurrent writes: {errors}"

        # Each device should see exactly its own 20 rows.
        for i in range(4):
            assert len(memory.recall_recent(f"dev{i}", n=100)) == 20
