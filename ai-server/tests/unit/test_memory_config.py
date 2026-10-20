"""Memory defaults are package-relative; explicit paths override benchmark mode."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def reload_memory_config(monkeypatch):
    def reload(env=None):
        for key in ('NAILA_MEMORY_DB_PATH', 'NAILA_BENCHMARK'):
            monkeypatch.delenv(key, raising=False)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        import config.benchmark as benchmark
        import config.memory as memory
        importlib.reload(benchmark)
        return importlib.reload(memory)
    return reload


def test_default_path_is_absolute_package_relative_and_cwd_stable(reload_memory_config, tmp_path, monkeypatch):
    first = reload_memory_config().DB_PATH
    assert Path(first).is_absolute()
    assert Path(first).resolve().is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.chdir(tmp_path)
    assert reload_memory_config().DB_PATH == first


@pytest.mark.parametrize('env,expected', [
    ({'NAILA_MEMORY_DB_PATH': '/tmp/custom.db'}, '/tmp/custom.db'),
    ({'NAILA_MEMORY_DB_PATH': ':memory:'}, ':memory:'),
    ({'NAILA_BENCHMARK': '1'}, ':memory:'),
    ({'NAILA_BENCHMARK': '1', 'NAILA_MEMORY_DB_PATH': '/tmp/custom.db'}, '/tmp/custom.db'),
])
def test_memory_path_overrides(reload_memory_config, env, expected):
    assert reload_memory_config(env).DB_PATH == expected
