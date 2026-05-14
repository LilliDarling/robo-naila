"""Unit tests for SQLite connection helper and migration runner."""

import sqlite3
from pathlib import Path

import pytest

from memory.db import open_connection, apply_migrations


class TestOpenConnection:
    """The single source of truth for connection-level PRAGMAs."""

    def test_returns_sqlite_connection(self):
        conn = open_connection(":memory:")
        try:
            assert isinstance(conn, sqlite3.Connection)
        finally:
            conn.close()

    def test_sets_foreign_keys_on(self):
        conn = open_connection(":memory:")
        try:
            (value,) = conn.execute("PRAGMA foreign_keys").fetchone()
            assert value == 1
        finally:
            conn.close()

    def test_sets_busy_timeout(self):
        conn = open_connection(":memory:")
        try:
            (value,) = conn.execute("PRAGMA busy_timeout").fetchone()
            assert value == 5000
        finally:
            conn.close()

    def test_sets_synchronous_normal(self):
        conn = open_connection(":memory:")
        try:
            (value,) = conn.execute("PRAGMA synchronous").fetchone()
            # 1 == NORMAL
            assert value == 1
        finally:
            conn.close()

    def test_file_path_uses_wal(self, tmp_path):
        # WAL is a no-op for :memory: databases (returns "memory"), so verify it
        # against a real file. WAL is what lets readers + a writer coexist.
        db_file = tmp_path / "naila.db"
        conn = open_connection(str(db_file))
        try:
            (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
            assert mode.lower() == "wal"
        finally:
            conn.close()


class TestApplyMigrations:
    """Migration runner: numbered .sql files under memory/migrations/."""

    def _write_migration(self, dir_path: Path, name: str, sql: str) -> None:
        (dir_path / name).write_text(sql)

    def test_fresh_db_starts_at_user_version_zero(self):
        conn = open_connection(":memory:")
        try:
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 0
        finally:
            conn.close()

    def test_applies_single_migration_and_bumps_user_version(self, tmp_path):
        self._write_migration(
            tmp_path,
            "001_initial.sql",
            "CREATE TABLE t (id INTEGER PRIMARY KEY);",
        )
        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
            # Table should exist
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='t'"
            ).fetchone()
            assert row is not None
        finally:
            conn.close()

    def test_applies_migrations_in_numeric_order(self, tmp_path):
        # Out-of-order filename creation; runner should still apply 001 before 002.
        self._write_migration(
            tmp_path,
            "002_add_col.sql",
            "ALTER TABLE t ADD COLUMN extra TEXT;",
        )
        self._write_migration(
            tmp_path,
            "001_initial.sql",
            "CREATE TABLE t (id INTEGER PRIMARY KEY);",
        )
        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 2
            cols = [r[1] for r in conn.execute("PRAGMA table_info(t)").fetchall()]
            assert "extra" in cols
        finally:
            conn.close()

    def test_skips_already_applied_migrations(self, tmp_path):
        self._write_migration(
            tmp_path,
            "001_initial.sql",
            "CREATE TABLE t (id INTEGER PRIMARY KEY);",
        )
        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, tmp_path)
            # Add a second migration and re-run; the first must not re-execute
            # (would error: table already exists).
            self._write_migration(
                tmp_path,
                "002_extra.sql",
                "CREATE TABLE u (id INTEGER PRIMARY KEY);",
            )
            apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 2
        finally:
            conn.close()

    def test_idempotent_when_no_pending_migrations(self, tmp_path):
        self._write_migration(
            tmp_path,
            "001_initial.sql",
            "CREATE TABLE t (id INTEGER PRIMARY KEY);",
        )
        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, tmp_path)
            apply_migrations(conn, tmp_path)
            apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
        finally:
            conn.close()

    def test_failed_migration_rolls_back(self, tmp_path):
        # Migration that creates one table then errors — table must not persist.
        self._write_migration(
            tmp_path,
            "001_initial.sql",
            "CREATE TABLE t (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE t (id INTEGER PRIMARY KEY);",  # duplicate => error
        )
        conn = open_connection(":memory:")
        try:
            with pytest.raises(sqlite3.DatabaseError):
                apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 0
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='t'"
            ).fetchone()
            assert row is None
        finally:
            conn.close()

    def test_ignores_non_sql_files(self, tmp_path):
        self._write_migration(tmp_path, "001_initial.sql", "CREATE TABLE t (id INTEGER);")
        (tmp_path / "README.md").write_text("notes")
        (tmp_path / "001_initial.sql.bak").write_text("nonsense")
        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, tmp_path)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
        finally:
            conn.close()

    def test_applies_bundled_v1_schema(self):
        # Smoke test: the real migrations directory shipped with the project
        # produces the v1 schema (exchanges + FTS5 + triggers, user_version=1).
        from memory import db as db_mod

        conn = open_connection(":memory:")
        try:
            apply_migrations(conn, db_mod.MIGRATIONS_DIR)
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            assert version >= 1

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table')"
                ).fetchall()
            }
            assert "exchanges" in tables
            assert "exchanges_fts" in tables

            triggers = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            }
            assert {"exchanges_ai", "exchanges_ad", "exchanges_au"}.issubset(triggers)
        finally:
            conn.close()
