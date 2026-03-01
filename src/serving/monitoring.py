"""
Monitoring & Observability — CSAO Production
==============================================
Request tracing, latency histograms, alert rule engine, and a
structured logging helper.  Designed to integrate with Prometheus /
Grafana / Datadog in production while working standalone locally.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  REQUEST TRACER
# ═════════════════════════════════════════════════════════════════════════════

class RequestTracer:
    """
    Distributed-tracing-style request tracker.

    Creates a trace_id per request, records span durations for each
    pipeline stage, and emits a structured log line at the end.
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._start = time.perf_counter()
        self._spans: list[dict[str, Any]] = []
        self._current_span: dict | None = None
        self.metadata: dict[str, Any] = {}

    def start_span(self, name: str) -> "RequestTracer":
        self._current_span = {
            "name": name,
            "start_ms": (time.perf_counter() - self._start) * 1000,
        }
        return self

    def end_span(self, **extra: Any) -> float:
        if self._current_span is None:
            return 0.0
        elapsed = (time.perf_counter() - self._start) * 1000
        self._current_span["end_ms"] = elapsed
        self._current_span["duration_ms"] = round(
            elapsed - self._current_span["start_ms"], 2
        )
        self._current_span.update(extra)
        self._spans.append(self._current_span)
        dur = self._current_span["duration_ms"]
        self._current_span = None
        return dur

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "spans": self._spans,
            "metadata": self.metadata,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  LATENCY HISTOGRAM
# ═════════════════════════════════════════════════════════════════════════════

class LatencyHistogram:
    """
    In-memory histogram for tracking P50/P95/P99 latencies.
    In production, this would be a Prometheus histogram.
    """

    def __init__(self, name: str, max_samples: int = 10_000):
        self.name = name
        self._samples: list[float] = []
        self._max_samples = max_samples

    def observe(self, value_ms: float) -> None:
        self._samples.append(value_ms)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        sorted_s = sorted(self._samples)
        idx = int(len(sorted_s) * p / 100)
        return sorted_s[min(idx, len(sorted_s) - 1)]

    def summary(self) -> dict:
        if not self._samples:
            return {"name": self.name, "count": 0}
        return {
            "name": self.name,
            "count": len(self._samples),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "mean_ms": round(sum(self._samples) / len(self._samples), 2),
            "max_ms": round(max(self._samples), 2),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  ERROR RATE TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class ErrorRateTracker:
    """Sliding-window error rate tracker."""

    def __init__(self, window_s: float = 300):
        self._window = window_s
        self._events: list[tuple[float, bool]] = []  # (timestamp, is_error)

    def record(self, is_error: bool) -> None:
        self._events.append((time.monotonic(), is_error))
        self._prune()

    def error_rate(self) -> float:
        self._prune()
        if not self._events:
            return 0.0
        errors = sum(1 for _, e in self._events if e)
        return errors / len(self._events)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        self._events = [(t, e) for t, e in self._events if t > cutoff]

    def summary(self) -> dict:
        self._prune()
        total = len(self._events)
        errors = sum(1 for _, e in self._events if e)
        return {
            "window_s": self._window,
            "total_requests": total,
            "errors": errors,
            "error_rate": round(self.error_rate(), 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  ALERT RULE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AlertRule:
    """Single alert rule definition."""
    name: str
    metric: str           # "latency_p95", "error_rate", "cache_hit_rate"
    operator: str         # "gt", "lt", "gte", "lte"
    threshold: float
    severity: str         # "critical", "warning", "info"
    description: str = ""

    def evaluate(self, value: float) -> bool:
        """Returns True if alert should fire."""
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
        }
        return ops.get(self.operator, lambda v, t: False)(value, self.threshold)


class AlertEngine:
    """Evaluates alert rules against current metrics."""

    DEFAULT_RULES = [
        AlertRule("high_latency",     "latency_p95",     "gt",  400,  "critical",
                  "P95 latency exceeding 400ms for 5+ minutes"),
        AlertRule("extreme_latency",  "latency_p99",     "gt",  800,  "critical",
                  "P99 latency exceeding 800ms"),
        AlertRule("high_error_rate",  "error_rate",      "gt",  0.05, "critical",
                  "Error rate exceeding 5%"),
        AlertRule("low_cache_hit",    "cache_hit_rate",  "lt",  0.30, "warning",
                  "Cache hit rate below 30% — possible cold cache"),
        AlertRule("llm_circuit_open", "llm_fallback_rate","gt", 0.20, "warning",
                  "LLM fallback rate exceeding 20%"),
    ]

    def __init__(self, rules: list[AlertRule] | None = None):
        self.rules = rules or self.DEFAULT_RULES
        self._fired: list[dict] = []

    def evaluate(self, metrics: dict[str, float]) -> list[dict]:
        """Check all rules against current metric values."""
        fired = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is not None and rule.evaluate(value):
                alert = {
                    "rule": rule.name,
                    "severity": rule.severity,
                    "metric": rule.metric,
                    "value": round(value, 4),
                    "threshold": rule.threshold,
                    "description": rule.description,
                    "timestamp": time.time(),
                }
                fired.append(alert)
                self._fired.append(alert)
                logger.warning(
                    f"[ALERT:{rule.severity.upper()}] {rule.name}: "
                    f"{rule.metric}={value:.4f} {rule.operator} {rule.threshold}"
                )
        return fired

    def get_history(self, limit: int = 50) -> list[dict]:
        return self._fired[-limit:]


# ═════════════════════════════════════════════════════════════════════════════
#  MONITORING DASHBOARD (aggregated)
# ═════════════════════════════════════════════════════════════════════════════

class MonitoringDashboard:
    """
    Central monitoring hub.

    Collects latency histograms, error rates, and cache stats,
    then evaluates alert rules against the aggregated metrics.
    """

    def __init__(self) -> None:
        self.latency = LatencyHistogram("e2e_latency")
        self.stage_latencies: dict[str, LatencyHistogram] = {
            "feature_retrieval": LatencyHistogram("feature_retrieval"),
            "candidate_generation": LatencyHistogram("candidate_generation"),
            "llm_reranking": LatencyHistogram("llm_reranking"),
            "post_processing": LatencyHistogram("post_processing"),
        }
        self.errors = ErrorRateTracker(window_s=300)
        self.alerts = AlertEngine()

        # Counters
        self._total_requests = 0
        self._strategy_counts: dict[str, int] = defaultdict(int)

    def record_request(
        self,
        tracer: RequestTracer,
        strategy: str = "full",
        is_error: bool = False,
    ) -> None:
        """Record a completed request."""
        self._total_requests += 1
        self._strategy_counts[strategy] += 1
        self.latency.observe(tracer.total_ms)
        self.errors.record(is_error)

        # Record per-stage latencies
        for span in tracer.to_dict().get("spans", []):
            name = span.get("name", "")
            if name in self.stage_latencies:
                self.stage_latencies[name].observe(span["duration_ms"])

    def check_alerts(self, cache_stats: dict | None = None) -> list[dict]:
        """Evaluate all alert rules against current metrics."""
        metrics = {
            "latency_p95": self.latency.percentile(95),
            "latency_p99": self.latency.percentile(99),
            "error_rate": self.errors.error_rate(),
        }
        if cache_stats:
            l1_hit = cache_stats.get("l1", {}).get("hit_rate", 0)
            metrics["cache_hit_rate"] = l1_hit

        return self.alerts.evaluate(metrics)

    def snapshot(self) -> dict:
        """Full dashboard snapshot for /metrics endpoint."""
        return {
            "total_requests": self._total_requests,
            "e2e_latency": self.latency.summary(),
            "stage_latencies": {
                k: v.summary() for k, v in self.stage_latencies.items()
            },
            "error_rate": self.errors.summary(),
            "strategy_distribution": dict(self._strategy_counts),
            "recent_alerts": self.alerts.get_history(10),
        }
