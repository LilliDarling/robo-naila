"""Benchmark run profile.

``NAILA_BENCHMARK=1`` switches the server into benchmark mode: conversation
memory defaults to ``:memory:`` so benchmark turns never reach stored history,
and tracing is enabled. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` must still be
set explicitly. Individual settings (``NAILA_MEMORY_DB_PATH``) override the
profile's defaults.
"""

import os


ENABLED = os.getenv("NAILA_BENCHMARK") == "1"
