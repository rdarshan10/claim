"""Telemetry (§18): OpenTelemetry traces and Prometheus metrics.

Both libraries are optional. Without them every helper here is a no-op, so the
app, the tests and template mode keep working on a bare install — telemetry is
decoration on top of the audit log, never a dependency of a customer reply.

* **Traces.** One span per graph node and per LLM call, parented by the turn's
  ``trace_id`` so a slow reply can be pulled apart node by node. Exported over
  OTLP/HTTP when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set; the most recent spans
  are also kept in memory for ``/admin/ops/traces`` and the AIOps agent.
* **Metrics.** Served at ``/metrics`` in Prometheus text format.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
try:
    from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                                   Gauge, Histogram, generate_latest)

    REGISTRY = CollectorRegistry(auto_describe=True)
    _PROM = True
except ImportError:  # pragma: no cover - exercised only on a bare install
    REGISTRY = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    _PROM = False


class _Noop:
    def labels(self, *_: Any, **__: Any) -> "_Noop":
        return self

    def inc(self, *_: Any, **__: Any) -> None: ...
    def observe(self, *_: Any, **__: Any) -> None: ...
    def set(self, *_: Any, **__: Any) -> None: ...


def _counter(name: str, doc: str, labels: list[str]) -> Any:
    return Counter(name, doc, labels, registry=REGISTRY) if _PROM else _Noop()


def _histogram(name: str, doc: str, labels: list[str],
               buckets: tuple[float, ...] | None = None) -> Any:
    if not _PROM:
        return _Noop()
    kwargs: dict[str, Any] = {"registry": REGISTRY}
    if buckets:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, labels, **kwargs)


def _gauge(name: str, doc: str, labels: list[str]) -> Any:
    return Gauge(name, doc, labels, registry=REGISTRY) if _PROM else _Noop()


_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 60)

GRAPH_NODE_SECONDS = _histogram(
    "cc_graph_node_seconds", "Wall time per LangGraph node.", ["graph", "node"],
    _LATENCY_BUCKETS)
GRAPH_NODE_ERRORS = _counter(
    "cc_graph_node_errors_total", "Exceptions raised inside a graph node.",
    ["graph", "node"])
GRAPH_RUNS = _counter(
    "cc_graph_runs_total", "Completed graph runs by outcome.", ["graph", "outcome"])
TURNS = _counter(
    "cc_conversation_turns_total", "Conversation turns by routed intent and outcome.",
    ["intent", "outcome"])
LLM_CALLS = _counter(
    "cc_llm_calls_total", "LLM calls by prompt, model and outcome "
    "(ok, error, cached, fallback, disabled).", ["prompt", "model", "outcome"])
LLM_LATENCY = _histogram(
    "cc_llm_latency_seconds", "Provider latency per LLM call.", ["prompt", "model"],
    _LATENCY_BUCKETS)
LLM_TOKENS = _counter(
    "cc_llm_tokens_total", "Tokens metered from provider usage.",
    ["prompt", "model", "direction"])
LLM_COST = _counter(
    "cc_llm_cost_usd_total", "Spend at published per-token rates.", ["model"])
GUARDRAIL_EVENTS = _counter(
    "cc_guardrail_events_total", "Input blocks, output failures, regenerations and "
    "template fallbacks.", ["kind"])
DOCUMENT_VERDICTS = _counter(
    "cc_document_verdicts_total", "Document pipeline outcomes.", ["status", "reason"])
AIOPS_INCIDENTS = _counter(
    "cc_aiops_incidents_total", "Incidents opened by the AIOps agent.",
    ["severity"])
AIOPS_ANOMALIES = _counter(
    "cc_aiops_anomalies_total", "Anomalies detected per detector.", ["detector"])
AIOPS_REMEDIATIONS = _counter(
    "cc_aiops_remediations_total", "Playbook actions by outcome.", ["action", "outcome"])
AIOPS_SIGNAL = _gauge(
    "cc_aiops_signal", "Latest value of each operational signal.", ["signal"])


def render_metrics() -> tuple[bytes, str]:
    if not _PROM:
        return b"# prometheus_client not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------
_RECENT: "deque[dict[str, Any]]" = deque(maxlen=int(os.getenv("OPS_RECENT_SPANS", "2000")))
_tracer: Any = None
_init_lock = threading.Lock()

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (BatchSpanProcessor, SimpleSpanProcessor,
                                                SpanExporter, SpanExportResult)

    class _RecentSpans(SpanExporter):
        """Keeps finished spans in memory for the ops console and AIOps."""

        def export(self, spans: "list[ReadableSpan]") -> "SpanExportResult":
            for span in spans:
                _RECENT.append({
                    "name": span.name,
                    "trace_id": dict(span.attributes or {}).get("cc.trace_id")
                    or format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "start": span.start_time,
                    "duration_ms": round(((span.end_time or span.start_time)
                                          - span.start_time) / 1e6, 2),
                    "status": span.status.status_code.name,
                    "attributes": {k: v for k, v in (span.attributes or {}).items()},
                })
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:  # noqa: D401 - interface method
            return None

    _OTEL = True
except ImportError:  # pragma: no cover
    _OTEL = False


def init_tracing(service_name: str = "claimcompanion") -> None:
    """Configure the tracer once. Safe to call repeatedly."""
    global _tracer
    if _tracer is not None or not _OTEL:
        return
    with _init_lock:
        if _tracer is not None:
            return
        provider = TracerProvider(resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", service_name)}))
        provider.add_span_processor(SimpleSpanProcessor(_RecentSpans()))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
                    OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except ImportError:
                pass
        _tracer = provider.get_tracer("claimcompanion")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A span with attributes; yields an object with ``set_attribute``."""
    init_tracing()
    if _tracer is None:
        yield _Noop()
        return
    clean = {k: v for k, v in attributes.items()
             if isinstance(v, (str, bool, int, float)) and v is not None}
    with _tracer.start_as_current_span(name, attributes=clean) as current:
        yield current


def mark_error(current: Any, exc: BaseException) -> None:
    if _OTEL and hasattr(current, "record_exception"):
        current.record_exception(exc)
        current.set_status(_otel_trace.Status(_otel_trace.StatusCode.ERROR, str(exc)[:200]))


def recent_spans(limit: int = 200, trace_id: str | None = None,
                 name_prefix: str | None = None) -> list[dict[str, Any]]:
    spans = list(_RECENT)
    if trace_id:
        spans = [s for s in spans if s["trace_id"] == trace_id
                 or s["attributes"].get("cc.trace_id") == trace_id]
    if name_prefix:
        spans = [s for s in spans if s["name"].startswith(name_prefix)]
    return spans[-limit:]


# --------------------------------------------------------------------------
# Graph node instrumentation
# --------------------------------------------------------------------------
def instrument(graph: str, node: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a LangGraph node with a span, a latency histogram and error count."""

    def wrapped(state: Any) -> Any:
        trace_id = state.get("trace_id", "") if isinstance(state, dict) else ""
        started = time.perf_counter()
        with span(f"{graph}.{node}", **{"cc.graph": graph, "cc.node": node,
                                        "cc.trace_id": trace_id}) as current:
            try:
                return fn(state)
            except Exception as exc:
                # GraphInterrupt is control flow, not a failure.
                if type(exc).__name__ not in ("GraphInterrupt", "ParentCommand"):
                    GRAPH_NODE_ERRORS.labels(graph, node).inc()
                    mark_error(current, exc)
                raise
            finally:
                GRAPH_NODE_SECONDS.labels(graph, node).observe(time.perf_counter() - started)

    wrapped.__name__ = f"{graph}_{node}"
    return wrapped
