"""Conversation memory configuration."""

import os
from pathlib import Path

from config import benchmark


# Anchor the default to the ai-server package directory so the path is stable
# regardless of cwd at launch time. Override with ``NAILA_MEMORY_DB_PATH`` for
# production deployments or ``:memory:`` for ephemeral runs. Benchmark mode
# defaults to ``:memory:``.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = ":memory:" if benchmark.ENABLED else str(_PACKAGE_DIR / "data" / "naila.db")

DB_PATH = os.getenv("NAILA_MEMORY_DB_PATH", _DEFAULT_DB_PATH)
