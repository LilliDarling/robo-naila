"""SQLite-backed conversation memory.

The v1 surface is intentionally narrow: ``commit_exchange`` to append a turn
and ``recall_recent`` to load the last N for a device. Retention, sessions,
metrics, and cleanup are deferred — see docs/INTELLIGENCE_ARCHITECTURE.md §6.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from memory.db import MIGRATIONS_DIR, apply_migrations, open_connection
from utils import get_logger


logger = get_logger(__name__)


class ConversationMemory:
    """Append-only conversation log keyed by device_id."""

    def __init__(self, db_path: str):
        self._conn = open_connection(db_path)
        apply_migrations(self._conn, MIGRATIONS_DIR)
        logger.info("memory_initialized", db_path=db_path)

    def commit_exchange(
        self,
        device_id: str,
        user_msg: str,
        assistant_msg: str,
        intent: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Append a single user/assistant exchange."""
        ts = time.time_ns() // 1_000_000  # millisecond precision
        meta_json = json.dumps(metadata) if metadata is not None else None
        with self._conn:
            self._conn.execute(
                "INSERT INTO exchanges "
                "(device_id, ts, user_msg, assistant_msg, intent, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, ts, user_msg, assistant_msg, intent, meta_json),
            )

    def recall_recent(self, device_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """Return up to ``n`` most-recent exchanges for ``device_id``, newest first."""
        rows = self._conn.execute(
            "SELECT user_msg, assistant_msg, intent, ts, metadata "
            "FROM exchanges WHERE device_id = ? "
            "ORDER BY ts DESC, exchange_id DESC LIMIT ?",
            (device_id, n),
        ).fetchall()
        return [
            {
                "user": user_msg,
                "assistant": assistant_msg,
                "intent": intent,
                "ts": ts,
                "metadata": json.loads(meta_str) if meta_str is not None else {},
            }
            for user_msg, assistant_msg, intent, ts, meta_str in rows
        ]

    def close(self) -> None:
        self._conn.close()
