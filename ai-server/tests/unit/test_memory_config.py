"""Tests for memory configuration defaults.

The default ``DB_PATH`` must be the same regardless of cwd at launch time —
otherwise running ``main.py`` from different directories silently lands the
conversation log in different files.
"""

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture
def reload_memory_config(monkeypatch):
    """Reload ``config.memory`` after env mutation so module-level constants re-evaluate."""

    def _reload(env: dict | None = None):
        monkeypatch.delenv("NAILA_MEMORY_DB_PATH", raising=False)
        for key, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        import config.memory as mod

        return importlib.reload(mod)

    return _reload


class TestMemoryConfigDefault:
    def test_default_path_is_absolute(self, reload_memory_config):
        mod = reload_memory_config()
        assert os.path.isabs(mod.DB_PATH), (
            "DB_PATH default must be absolute so it doesn't shift with cwd"
        )

    def test_default_path_lives_under_ai_server_package(self, reload_memory_config):
        mod = reload_memory_config()
        package_dir = Path(__file__).resolve().parents[2]  # ai-server/
        assert Path(mod.DB_PATH).resolve().is_relative_to(package_dir), (
            f"DB_PATH {mod.DB_PATH} must live under {package_dir}"
        )

    def test_default_path_is_stable_across_cwd(self, reload_memory_config, tmp_path, monkeypatch):
        first = reload_memory_config().DB_PATH
        monkeypatch.chdir(tmp_path)
        second = reload_memory_config().DB_PATH
        assert first == second

    def test_env_override_wins(self, reload_memory_config):
        mod = reload_memory_config({"NAILA_MEMORY_DB_PATH": "/tmp/custom.db"})
        assert mod.DB_PATH == "/tmp/custom.db"

    def test_in_memory_override_passes_through(self, reload_memory_config):
        mod = reload_memory_config({"NAILA_MEMORY_DB_PATH": ":memory:"})
        assert mod.DB_PATH == ":memory:"
