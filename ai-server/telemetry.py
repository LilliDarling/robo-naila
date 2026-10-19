"""Opt-in manual tracing; callers own spans, executor producers own their end.

No logging instrumentation or payload/exception recording is installed.
"""
from contextlib import contextmanager
import os
import threading
import time
from uuid import uuid4

from opentelemetry import context, trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def init(service_name="naila.ai", *, enabled=None, endpoint=None):
    """The AI server passes its resolved ``config.telemetry`` settings; standalone
    callers (benchmark peer, stub server) fall back to the environment."""
    if enabled is None:
        enabled = os.environ.get("NAILA_OTEL_ENABLED") == "1"
    if not enabled:
        return None
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        raise ValueError("explicit OTEL_EXPORTER_OTLP_TRACES_ENDPOINT required")
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, ALWAYS_ON

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.instance.id": str(uuid4())}),
        sampler=ParentBased(ALWAYS_ON), shutdown_on_exit=False,
    )
    # SDK handles queue synchronization/batching; exporter enforces request timeout.
    # https://opentelemetry-python.readthedocs.io/en/stable/sdk/trace.export.html
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, timeout=2), max_queue_size=2048,
        max_export_batch_size=256, schedule_delay_millis=200, export_timeout_millis=2000,
    ))
    trace.set_tracer_provider(provider)
    return provider


def shutdown(provider):
    """Flush/shutdown off the audio loop, with one five-second caller deadline.

    Python SDK 1.38 shutdown has no timeout and joins its worker:
    https://github.com/open-telemetry/opentelemetry-python/blob/v1.38.0/opentelemetry-sdk/src/opentelemetry/sdk/_shared_internal/__init__.py
    A daemon owner bounds our wait without claiming it can stop a native thread.
    """
    if provider is None:
        return True
    deadline = time.monotonic() + 5
    completed = threading.Event()
    def finish():
        try:
            flushed = provider.force_flush(timeout_millis=max(0, int((deadline - time.monotonic()) * 1000)))
            provider.shutdown()
            if flushed:
                completed.set()
        except Exception:
            pass  # Caller reports incomplete shutdown without capturing exception text.
    worker = threading.Thread(target=finish, daemon=True)
    worker.start()
    worker.join(max(0, deadline - time.monotonic()))
    return completed.is_set()


def extract(traceparent):
    return TraceContextTextMapPropagator().extract({"traceparent": traceparent})


@contextmanager
def attached(parent):
    token = context.attach(parent)
    try:
        yield
    finally:
        context.detach(token)


async def in_context(parent, awaitable):
    """Attach a frozen START parent in its own asyncio task, never in the reader."""
    with attached(parent):
        return await awaitable


def executor_call(function, *args):
    """Capture on the asyncio thread; executor attaches/detaches explicitly.

    The producer function must end its own span in finally. Cancelling its
    asyncio consumer cannot end the producer span or stop native model work.
    """
    parent = context.get_current()
    def call():
        with attached(parent):
            return function(*args)
    return call


def span(name, parent=None, **attributes):
    return trace.get_tracer("naila").start_as_current_span(
        name, context=parent, attributes=attributes,
        record_exception=False, set_status_on_exception=False,
    )
