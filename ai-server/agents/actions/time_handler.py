"""Sync action handler for ``time_query`` — reads the system clock and returns a
templated reply. Async only to match the substrate's uniform handler signature
(see agents/actions/registry.py).
"""

from datetime import datetime

from agents.actions import registry


async def handle(utterance: str, context: dict) -> str:
    """Return a friendly current-time string. Ignores utterance and context."""
    now = datetime.now()
    return f"It's {now.strftime('%I:%M %p')}."


registry.register("time_query", handle)
