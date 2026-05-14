"""Process-wide registry of action handlers, keyed by intent name.

Handler modules call ``register(intent, handler)`` at import time. The graph's
``dispatch_action`` node calls ``get(intent)`` per turn to look up the handler
(or ``None`` if the intent should fall through to the LLM path).

Re-registering the same intent silently replaces the previous handler — this
is friendly to test reload and to swapping implementations during development.
"""

from typing import Awaitable, Callable, Dict, Optional


ActionHandler = Callable[[str, dict], Awaitable[str]]

_HANDLERS: Dict[str, ActionHandler] = {}


def register(intent: str, handler: ActionHandler) -> None:
    """Register ``handler`` to run when ``intent`` is dispatched."""
    _HANDLERS[intent] = handler


def get(intent: Optional[str]) -> Optional[ActionHandler]:
    """Return the handler for ``intent``, or ``None`` if no handler is registered."""
    if not intent:
        return None
    return _HANDLERS.get(intent)
