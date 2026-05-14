"""SQLite connection helper and migration runner for the memory store.

Single source of truth for connection-level PRAGMAs. All AI-server code that
opens the memory database goes through ``open_connection``; ``apply_migrations``
brings a fresh connection up to the latest schema version on startup.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_VERSION_RE = re.compile(r"^(\d+)_")


def open_connection(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with the project's PRAGMAs applied.

    WAL is set unconditionally; on ``:memory:`` databases SQLite ignores it.
    Foreign keys, busy timeout, and synchronous=NORMAL match the v1 design.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _discover_migrations(migrations_dir: Path) -> Iterable[Tuple[int, Path]]:
    """Yield (version, path) pairs for ``NNN_*.sql`` files, sorted numerically."""
    found: list[Tuple[int, Path]] = []
    for entry in migrations_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".sql":
            continue
        match = _VERSION_RE.match(entry.name)
        if not match:
            continue
        found.append((int(match.group(1)), entry))
    found.sort(key=lambda pair: pair[0])
    return found


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply pending migrations up to the latest version found on disk.

    Each migration runs in its own transaction; a failure rolls the migration
    back and leaves ``PRAGMA user_version`` at the previous value. Already-applied
    migrations (version <= current ``user_version``) are skipped.
    """
    (current,) = conn.execute("PRAGMA user_version").fetchone()

    for version, path in _discover_migrations(migrations_dir):
        if version <= current:
            continue
        sql = path.read_text()
        _apply_one(conn, version, sql)
        current = version


def _apply_one(conn: sqlite3.Connection, version: int, sql: str) -> None:
    # ``executescript`` implicitly commits any pending transaction, which would
    # break atomic application. Split the script into individual statements
    # (respecting trigger BEGIN…END blocks via ``sqlite3.complete_statement``)
    # and run them inside an explicit transaction instead.
    original_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")
        try:
            for stmt in _split_statements(sql):
                conn.execute(stmt)
            # PRAGMA user_version takes a literal — interpolation is safe because
            # ``version`` was parsed from a regex-matched filename digit run.
            conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.isolation_level = original_isolation


def _split_statements(sql: str) -> Iterable[str]:
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            stmt = buffer.strip()
            if stmt:
                yield stmt
            buffer = ""
    tail = buffer.strip()
    if tail:
        yield tail
