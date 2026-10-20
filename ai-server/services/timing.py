"""Monotonic stage evidence, without changing public model result schemas."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time
import threading

from opentelemetry import context, trace
import telemetry as otel

_current = ContextVar("ai_timing_stage", default=None)
_turn = ContextVar("ai_timing_turn", default=None)


@contextmanager
def turn_evidence(turn):
    token = _turn.set(turn)
    try:
        yield
    finally:
        _turn.reset(token)


def current_turn():
    return _turn.get()


def turn_outcome(outcome):
    turn = _turn.get()
    if turn is not None:
        turn.outcome = outcome


class Stage:
    """One owner ends each span; executor producers can outlive their callers."""
    def __init__(self, name, parent=None, **attributes):
        # SDK 1.38 uses wall-clock time_ns for span timestamps; scored waits
        # need a monotonic interval instead of subtracting those timestamps.
        # https://github.com/open-telemetry/opentelemetry-python/blob/v1.38.0/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py
        self.started = time.perf_counter()
        self.span = trace.get_tracer("naila").start_span(name, context=parent, attributes=attributes)
        if name == "llm":
            self.set(**{f"llm.{key}.unavailable_reason": "backend did not report exact usage"
                        for key in ("completion_tokens", "prompt_tokens")})
        self.outcome = "empty"
        self.cancelled = threading.Event()
        # No-op tracing must preserve an extracted parent for downstream tasks.
        self.context = (trace.set_span_in_context(self.span, parent)
                        if self.span.get_span_context().is_valid
                        else parent if parent is not None else context.get_current())

    def set(self, **attributes):
        if not self.span.is_recording():
            return
        for key, value in attributes.items():
            self.span.set_attribute(key, value)

    def cancel(self):
        # Event supplies cross-thread synchronization for the producer owner.
        # https://docs.python.org/3.12/library/threading.html#event-objects
        self.cancelled.set()
        self.set(caller_cancelled=True)

    def end(self):
        if self.cancelled.is_set():
            self.outcome = "cancelled"
        self.set(outcome=self.outcome, duration_ms=(time.perf_counter() - self.started) * 1000)
        self.span.end()


def label(outcome=None, **attributes):
    stage = _current.get()
    if stage is not None:
        if outcome is not None:
            stage.outcome = outcome
        stage.set(**attributes)


def measured(name, value_field=None, native_field=None):
    """Time a full async service call, including resource waits and encoding."""
    def decorate(function):
        @wraps(function)
        async def call(*args, **kwargs):
            stage = Stage(name, **({"streaming": False} if name == "llm" else {}))
            token = _current.set(stage)
            try:
                with otel.attached(stage.context):
                    result = await function(*args, **kwargs)
                value = getattr(result, value_field) if value_field else result
                if stage.outcome == "empty" and value and (not isinstance(value, str) or value.strip()):
                    stage.outcome = "success"
                if native_field:
                    stage.set(**{native_field: getattr(result, native_field)})
                return result
            except asyncio.CancelledError:
                stage.outcome = "cancelled"
                raise
            except Exception:
                stage.outcome = "error"
                raise
            finally:
                _current.reset(token)
                if stage.outcome != "success":
                    turn_outcome("stage_failed")
                stage.end()
        return call
    return decorate
