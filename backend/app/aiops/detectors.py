"""Anomaly detection over operational signals.

Two tests per detector, either of which fires it:

* **Threshold** — a level that is wrong whatever the history says (any stuck
  document, an LLM failing a quarter of its calls).
* **Baseline** — a z-score against this signal's own recent sweeps, for things
  with no absolute "bad" level (cost, latency, injection volume). Needs
  ``MIN_BASELINE`` prior sweeps, so a fresh install only alarms on thresholds.

Deterministic on purpose: whether something is an incident must not depend on
a model being up. The model's job starts after detection — explaining it.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.aiops.signals import Signal
from app.db import connect, query

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
MIN_BASELINE = 12
BASELINE_SWEEPS = 48
Z_THRESHOLD = 3.0


@dataclass(frozen=True)
class Detector:
    id: str
    signal: str
    title: str
    severity: str
    threshold: float | None = None
    # Rates on a handful of events are noise.
    min_samples: int = 1
    # Fire on a baseline z-score as well as the threshold.
    use_baseline: bool = False
    # Baseline anomalies below this value are ignored (a jump from 0 to 1).
    baseline_floor: float = 0.0
    escalate_at: float | None = None   # value at which severity becomes critical


DETECTORS: list[Detector] = [
    Detector("llm_errors", "llm_error_rate", "LLM calls are failing", "high",
             threshold=0.25, min_samples=4, use_baseline=True, baseline_floor=0.1,
             escalate_at=0.5),
    Detector("llm_latency", "llm_p95_latency_ms", "LLM responses are slow", "medium",
             threshold=15000, min_samples=5, use_baseline=True, baseline_floor=4000),
    Detector("template_fallbacks", "template_fallback_rate",
             "Replies are falling back to templates", "high",
             threshold=0.2, min_samples=5, use_baseline=True, baseline_floor=0.05),
    Detector("grounding_failures", "output_guard_fail_rate",
             "Replies are failing the grounding check", "medium",
             threshold=0.35, min_samples=5, use_baseline=True, baseline_floor=0.1),
    Detector("injection_surge", "injection_attempts",
             "Surge in prompt-injection attempts", "medium",
             threshold=10, use_baseline=True, baseline_floor=3),
    Detector("documents_stuck", "documents_stuck",
             "Documents are stuck in the pipeline", "high", threshold=1),
    Detector("review_backlog", "document_review_backlog",
             "Document review backlog is building", "medium",
             threshold=50, use_baseline=True, baseline_floor=10),
    Detector("rejection_spike", "document_rejection_rate",
             "Unusually many documents are being rejected", "medium",
             threshold=0.6, min_samples=5, use_baseline=True, baseline_floor=0.3),
    Detector("escalation_sla", "escalations_sla_breached",
             "Customer cases are past their pick-up SLA", "high", threshold=1),
    Detector("registration_failures", "registration_failures",
             "Claim registrations are failing", "high", threshold=1),
    Detector("registration_hung", "registration_hung",
             "A registration run has hung", "high", threshold=1),
    Detector("claims_at_risk", "claims_at_risk",
             "Claims are going stale without a chase", "medium", threshold=1),
    Detector("negative_feedback", "negative_feedback_rate",
             "Customers are rating replies unhelpful", "medium",
             threshold=0.4, min_samples=5),
    Detector("upset_customers", "upset_customer_rate",
             "A high share of customers are upset", "medium",
             threshold=0.5, min_samples=10, use_baseline=True, baseline_floor=0.25),
    Detector("template_mode", "llm_template_mode",
             "Template mode is still on after an LLM incident", "medium", threshold=1),
    Detector("cost_spike", "llm_cost_usd", "LLM spend is unusually high", "low",
             use_baseline=True, baseline_floor=0.01),
]

BY_ID = {d.id: d for d in DETECTORS}


@dataclass
class Anomaly:
    detector: str
    signal: str
    title: str
    severity: str
    value: float
    reason: str
    threshold: float | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None
    z_score: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _baseline(signal: str, exclude_run: str | None) -> list[float]:
    rows = query(
        """SELECT value FROM aiops_snapshot WHERE signal = ? AND run_id != ?
           ORDER BY at DESC LIMIT ?""", (signal, exclude_run or "", BASELINE_SWEEPS))
    return [float(r["value"]) for r in rows]


def evaluate(detector: Detector, signal: Signal, run_id: str | None = None) -> Anomaly | None:
    value = signal.value
    if signal.sample_size < detector.min_samples:
        return None
    # An LLM that is deliberately off is not an outage.
    if detector.id in ("llm_errors", "llm_latency") and \
            signal.evidence.get("llm_enabled") is False:
        return None
    # Only an incident when this agent switched it on, not a deliberate demo.
    if detector.id == "template_mode" and not signal.evidence.get("set_by_aiops"):
        return None

    reason = ""
    if detector.threshold is not None and value >= detector.threshold:
        reason = f"{signal.name} = {value:g} ≥ threshold {detector.threshold:g}"

    mean = std = z = None
    if detector.use_baseline:
        history = _baseline(signal.name, run_id)
        if len(history) >= MIN_BASELINE:
            mean = sum(history) / len(history)
            std = math.sqrt(sum((h - mean) ** 2 for h in history) / len(history))
            if std > 0:
                z = (value - mean) / std
            elif value > mean:
                z = math.inf
            if (not reason and z is not None and z >= Z_THRESHOLD
                    and value >= detector.baseline_floor):
                reason = (f"{signal.name} = {value:g} is {z:.1f}σ above its baseline "
                          f"{mean:.3g} ± {std:.3g}")
    if not reason:
        return None

    severity = detector.severity
    if detector.escalate_at is not None and value >= detector.escalate_at:
        severity = "critical"
    return Anomaly(detector.id, signal.name, detector.title, severity, value, reason,
                   detector.threshold, mean, std,
                   None if z is None or math.isinf(z) else round(z, 2),
                   signal.evidence)


def detect(signals: dict[str, Signal], run_id: str | None = None) -> list[Anomaly]:
    found = []
    for detector in DETECTORS:
        signal = signals.get(detector.signal)
        if signal is not None and (anomaly := evaluate(detector, signal, run_id)):
            found.append(anomaly)
    found.sort(key=lambda a: SEVERITY_ORDER[a.severity], reverse=True)
    return found


def snapshot(run_id: str, signals: dict[str, Signal]) -> None:
    """Record this sweep's values as the next sweep's baseline."""
    now = datetime.now(timezone.utc).isoformat()
    conn = connect()
    try:
        conn.executemany(
            "INSERT INTO aiops_snapshot (id, run_id, signal, value, at) VALUES (?,?,?,?,?)",
            [(str(uuid.uuid4()), run_id, s.name, float(s.value), now)
             for s in signals.values()])
        # Keep the table bounded: baselines only ever read the latest sweeps.
        conn.execute("""DELETE FROM aiops_snapshot WHERE julianday(at) <
                        julianday('now', '-14 days')""")
        conn.commit()
    finally:
        conn.close()


def worst(anomalies: list[Anomaly]) -> str:
    return max((a.severity for a in anomalies), key=SEVERITY_ORDER.get, default="low")
