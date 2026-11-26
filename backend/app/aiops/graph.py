"""The AIOps agent, as a LangGraph graph.

    collect ─► detect ─► dedupe ─┬─► END                     (nothing new)
                                 └─► diagnose ─► open_incident ─► auto_remediate
                                                                    │
                          ┌─────────────── needs approval? ─────────┤
                          ▼                                         ▼
                      approval  ── interrupt() ── manager ──►    verify ─► close ─► END
                          └──────────────────────────────────────┘

* **Detection is deterministic** (``detectors``): whether something is an
  incident never depends on a model being up.
* **Diagnosis is the model's job**: correlate the anomalies into one root
  cause, explain impact, and pick which *high-risk* playbooks are warranted.
  Without a model, a rule-based diagnosis takes over.
* **Low-risk playbooks run automatically** (re-queue a stuck document, raise a
  breached case's priority, put a stale claim on today's diary).
* **High-risk playbooks pause the graph** with ``interrupt()``. The paused run
  lives in the checkpointer under ``aiops:<run_id>``; a manager's decision
  resumes it with ``Command(resume=...)`` — even after a restart, when the
  checkpointer is SQLite.
* **Verify** re-measures the signals that fired, so an incident is only
  RESOLVED when the numbers say so.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents import runtime
from app.aiops import detectors, playbooks, signals
from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query, query_one
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.ops import telemetry

GRAPH = "aiops"
_sweep_lock = threading.Lock()


class AIOpsState(TypedDict, total=False):
    run_id: str
    thread_id: str
    trigger: str
    actor: str
    signals: dict[str, dict[str, Any]]
    anomalies: list[dict[str, Any]]
    suppressed: list[str]
    incident_id: str | None
    diagnosis: dict[str, Any]
    plan: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    decision: dict[str, Any] | None
    verification: dict[str, Any]
    status: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merged_evidence(anomalies: list[dict[str, Any]], playbook: playbooks.Playbook
                     ) -> dict[str, Any]:
    """The evidence of every anomaly this playbook answers, merged."""
    merged: dict[str, Any] = {}
    for anomaly in anomalies:
        if anomaly["detector"] not in playbook.detectors:
            continue
        for key, value in (anomaly.get("evidence") or {}).items():
            if isinstance(value, list):
                merged[key] = list(dict.fromkeys([*merged.get(key, []), *value]))
            else:
                merged.setdefault(key, value)
    return merged


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
def collect(state: AIOpsState) -> dict[str, Any]:
    current = signals.collect()
    for name, signal in current.items():
        telemetry.AIOPS_SIGNAL.labels(name).set(signal.value)
    return {"signals": {k: v.as_dict() for k, v in current.items()}}


def detect(state: AIOpsState) -> dict[str, Any]:
    current = {k: signals.Signal(**v) for k, v in state["signals"].items()}
    found = detectors.detect(current, state["run_id"])
    detectors.snapshot(state["run_id"], current)
    for anomaly in found:
        telemetry.AIOPS_ANOMALIES.labels(anomaly.detector).inc()
    return {"anomalies": [a.as_dict() for a in found]}


def dedupe(state: AIOpsState) -> dict[str, Any]:
    """Don't reopen what is already open, waiting, or was just handled."""
    cooldown = f"-{get_settings().aiops_cooldown_minutes} minutes"
    rows = query(
        """SELECT detectors FROM aiops_incident
           WHERE status IN ('OPEN','AWAITING_APPROVAL')
              OR (status != 'DISMISSED' AND julianday(created_at) >= julianday('now', ?))""",
        (cooldown,))
    busy: set[str] = set()
    for row in rows:
        busy.update(json.loads(row["detectors"] or "[]"))
    fresh = [a for a in state["anomalies"] if a["detector"] not in busy]
    suppressed = [a["detector"] for a in state["anomalies"] if a["detector"] in busy]
    return {"anomalies": fresh, "suppressed": suppressed,
            "status": "quiet" if not fresh else "detected"}


def _heuristic_diagnosis(anomalies: list[dict[str, Any]],
                         allowed: list[playbooks.Playbook]) -> dict[str, Any]:
    """Rule-based root cause, used when no model is available."""
    fired = {a["detector"]: a for a in anomalies}
    allowed_ids = {p.id for p in allowed}
    recommend: list[str] = []
    causes: list[str] = []

    if "llm_errors" in fired:
        ev = fired["llm_errors"]["evidence"]
        sample = "; ".join(ev.get("error_samples", [])[:2]) or "no error text captured"
        causes.append(f"The {ev.get('provider', 'LLM')} provider is failing "
                      f"{fired['llm_errors']['value']:.0%} of calls ({sample}).")
        if "failover_llm_provider" in allowed_ids:
            recommend.append("failover_llm_provider")
        elif fired["llm_errors"]["severity"] == "critical":
            recommend.append("enable_template_mode")
        if "documents_stuck" in fired:
            causes.append("Documents are stuck because classification and extraction "
                          "wait on the same failing model.")
    elif "documents_stuck" in fired:
        causes.append("Documents lost their worker mid-pipeline (most often a restart "
                      "while a background thread was running).")
    if "llm_latency" in fired and "llm_errors" not in fired:
        causes.append("The LLM provider is slow but answering.")
    if "template_fallbacks" in fired and "llm_errors" not in fired:
        causes.append("Replies are failing grounding: the model is adding facts the "
                      "tools did not return (prompt or model drift).")
    if "template_mode" in fired and "llm_errors" not in fired:
        causes.append("Template mode was switched on during an earlier LLM incident "
                      "and the provider now looks healthy.")
        recommend.append("restore_llm")
    if "escalation_sla" in fired:
        causes.append("Reviewer capacity: customer cases are waiting unassigned.")
    if "registration_failures" in fired:
        errors = fired["registration_failures"]["evidence"].get("errors", [])
        causes.append("The registration bot is failing against the core system"
                      + (f" ({errors[0]})." if errors else "."))
        recommend.append("retry_failed_registrations")
    if "registration_hung" in fired:
        causes.append("A registration bot run stopped making progress.")
    if "claims_at_risk" in fired:
        causes.append("The handler diary is not being worked: open claims have had "
                      "no movement for a month or more.")
    if "injection_surge" in fired:
        causes.append("A burst of prompt-injection attempts, all blocked pre-model.")

    titles = [a["title"] for a in anomalies]
    return {
        "title": titles[0] if len(titles) == 1 else f"{titles[0]} (+{len(titles) - 1} more)",
        "summary": " ".join(f"{a['title']}: {a['reason']}." for a in anomalies[:4]),
        "root_cause": " ".join(causes) or "No single cause stands out; see the anomalies.",
        "confidence": 0.5,
        "recommended_playbooks": [r for r in recommend if r in allowed_ids],
    }


def diagnose(state: AIOpsState) -> dict[str, Any]:
    anomalies = state["anomalies"]
    fired = [a["detector"] for a in anomalies]
    evidence = {k: v for a in anomalies for k, v in (a.get("evidence") or {}).items()}
    allowed = playbooks.applicable(fired, evidence)
    fallback = _heuristic_diagnosis(anomalies, allowed)

    result = gateway.complete(
        "aiops_diagnose",
        {
            "anomalies": wrap_untrusted("anomalies", json.dumps(
                [{k: a[k] for k in ("detector", "title", "severity", "value", "reason",
                                    "evidence")} for a in anomalies], default=str)[:6000]),
            "signals": wrap_untrusted("signals", json.dumps(
                {k: {"value": v["value"], "unit": v["unit"], "n": v["sample_size"]}
                 for k, v in state["signals"].items()})),
            "playbooks": json.dumps([p.public() for p in allowed]),
        },
        tier="primary",
        trace_id=state["run_id"],
        fallback=json.dumps(fallback),
    )
    parsed = result.json(default=None) or {}
    source = "heuristic" if result.degraded or not parsed else result.model
    diagnosis = {**fallback, **{k: v for k, v in parsed.items()
                                if k in fallback and v not in (None, "")}}

    # The model may only pick from what is allowed for what actually fired.
    allowed_ids = {p.id for p in allowed}
    picked = [p for p in diagnosis.get("recommended_playbooks") or []
              if isinstance(p, str) and p in allowed_ids]
    dropped = [p for p in diagnosis.get("recommended_playbooks") or [] if p not in picked]
    try:
        confidence = max(0.0, min(1.0, float(diagnosis.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    diagnosis.update({"recommended_playbooks": picked, "rejected_suggestions": dropped,
                      "source": source, "confidence": confidence})

    # Plan: the routine low-risk runbook always applies; high-risk steps only
    # when the diagnosis asked for them.
    plan = []
    for playbook in allowed:
        if playbook.risk == "low" or playbook.id in picked:
            plan.append({"playbook": playbook.id, "title": playbook.title,
                         "risk": playbook.risk, "recommended": playbook.id in picked,
                         "status": "planned"})
    return {"diagnosis": diagnosis, "plan": plan, "actions": []}


def _save(state: AIOpsState, **changes: Any) -> None:
    values = {**state, **changes}
    execute(
        """UPDATE aiops_incident SET status = ?, title = ?, summary = ?, root_cause = ?,
               diagnosis_source = ?, confidence = ?, plan = ?, actions = ?,
               verification = ?, decided_by = ?, updated_at = ? WHERE id = ?""",
        (values.get("status", "OPEN"), values["diagnosis"].get("title", "Incident"),
         values["diagnosis"].get("summary"), values["diagnosis"].get("root_cause"),
         values["diagnosis"].get("source"), values["diagnosis"].get("confidence"),
         json.dumps(values.get("plan", [])),
         json.dumps(values.get("actions", []), default=str),
         json.dumps(values.get("verification", {}), default=str),
         (values.get("decision") or {}).get("by"), _now(), values["incident_id"]))


def open_incident(state: AIOpsState) -> dict[str, Any]:
    incident_id = str(uuid.uuid4())
    anomalies = state["anomalies"]
    severity = detectors.worst([detectors.Anomaly(**a) for a in anomalies])
    execute(
        """INSERT INTO aiops_incident (id, run_id, thread_id, status, severity, detectors,
               title, anomalies, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (incident_id, state["run_id"], state["thread_id"], "OPEN", severity,
         json.dumps(sorted({a["detector"] for a in anomalies})),
         state["diagnosis"].get("title", "Incident"),
         json.dumps(anomalies, default=str), _now(), _now()))
    telemetry.AIOPS_INCIDENTS.labels(severity).inc()
    audit.record("aiops_incident_opened", actor_type="agent", actor_id="aiops",
                 entity_type="aiops_incident", entity_id=incident_id,
                 payload={"severity": severity,
                          "detectors": [a["detector"] for a in anomalies],
                          "diagnosis_source": state["diagnosis"].get("source"),
                          "plan": [p["playbook"] for p in state["plan"]]},
                 trace_id=state["run_id"])
    values = {**state, "incident_id": incident_id, "status": "OPEN"}
    _save(values)  # type: ignore[arg-type]
    return {"incident_id": incident_id, "status": "OPEN"}


def _execute(state: AIOpsState, plan: list[dict[str, Any]], ids: set[str],
             actor: str) -> list[dict[str, Any]]:
    actions = []
    for step in plan:
        if step["playbook"] not in ids or step["status"] != "planned":
            continue
        playbook = playbooks.PLAYBOOKS[step["playbook"]]
        evidence = _merged_evidence(state["anomalies"], playbook)
        outcome = playbooks.run(playbook.id, evidence,
                                incident_id=state["incident_id"] or "", actor=actor)
        step["status"] = "executed" if outcome["outcome"] == "ok" else outcome["outcome"]
        actions.append(outcome)
    return actions


def auto_remediate(state: AIOpsState) -> dict[str, Any]:
    plan = [dict(p) for p in state["plan"]]
    auto = get_settings().aiops_auto_remediate
    low = {p["playbook"] for p in plan if p["risk"] == "low"} if auto else set()
    actions = _execute(state, plan, low, "aiops")
    for step in plan:
        if step["status"] == "planned":
            step["status"] = "awaiting_approval"
    return {"plan": plan, "actions": [*state.get("actions", []), *actions]}


def request_approval(state: AIOpsState) -> dict[str, Any]:
    """Mark the incident as waiting. Separate from ``approval`` because a node
    that calls interrupt() re-runs from the top on resume."""
    waiting = [p for p in state["plan"] if p["status"] == "awaiting_approval"]
    _save(state, status="AWAITING_APPROVAL")
    audit.record("aiops_approval_requested", actor_type="agent", actor_id="aiops",
                 entity_type="aiops_incident", entity_id=state["incident_id"],
                 payload={"playbooks": [p["playbook"] for p in waiting]},
                 trace_id=state["run_id"])
    return {"status": "AWAITING_APPROVAL"}


def approval(state: AIOpsState) -> dict[str, Any]:
    """Pause for a manager. Resumed with {"approve": [...], "by": ...}."""
    waiting = [p for p in state["plan"] if p["status"] == "awaiting_approval"]
    decision = interrupt({"incident_id": state["incident_id"],
                          "title": state["diagnosis"].get("title"),
                          "root_cause": state["diagnosis"].get("root_cause"),
                          "awaiting": waiting})

    decision = decision if isinstance(decision, dict) else {}
    approved = set(decision.get("approve") or []) & {p["playbook"] for p in waiting}
    by = str(decision.get("by") or "manager")
    plan = [dict(p) for p in state["plan"]]
    for step in plan:
        if step["status"] == "awaiting_approval":
            step["status"] = "planned" if step["playbook"] in approved else "rejected"
    actions = _execute({**state, "plan": plan}, plan, approved, by)  # type: ignore[arg-type]
    audit.record("aiops_decision", actor_type="staff", actor_id=by,
                 entity_type="aiops_incident", entity_id=state["incident_id"],
                 payload={"approved": sorted(approved),
                          "rejected": [p["playbook"] for p in plan
                                       if p["status"] == "rejected"]},
                 trace_id=state["run_id"])
    return {"plan": plan, "actions": [*state.get("actions", []), *actions],
            "decision": {**decision, "by": by}, "status": "OPEN"}


def verify(state: AIOpsState) -> dict[str, Any]:
    """Re-measure what fired. Resolved means the numbers say so."""
    current = signals.collect()
    still, cleared = [], []
    for anomaly in state["anomalies"]:
        detector = detectors.BY_ID[anomaly["detector"]]
        signal = current.get(detector.signal)
        again = detectors.evaluate(detector, signal, state["run_id"]) if signal else None
        (still if again else cleared).append({
            "detector": anomaly["detector"], "before": anomaly["value"],
            "after": signal.value if signal else None})
    return {"verification": {"at": _now(), "cleared": cleared, "still_firing": still}}


def close(state: AIOpsState) -> dict[str, Any]:
    verification = state.get("verification") or {}
    executed = any(a["outcome"] == "ok" for a in state.get("actions", []))
    if not verification.get("still_firing"):
        status = "RESOLVED"
    elif executed:
        status = "MITIGATED"
    else:
        status = "OPEN"
    _save(state, status=status)
    audit.record("aiops_incident_updated", actor_type="agent", actor_id="aiops",
                 entity_type="aiops_incident", entity_id=state["incident_id"],
                 payload={"status": status,
                          "cleared": [c["detector"] for c in verification.get("cleared", [])],
                          "actions": [a["playbook"] for a in state.get("actions", [])]},
                 trace_id=state["run_id"])
    return {"status": status}


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------
def _after_dedupe(state: AIOpsState) -> str:
    return "diagnose" if state.get("anomalies") else END


def _after_auto(state: AIOpsState) -> str:
    waiting = any(p["status"] == "awaiting_approval" for p in state["plan"])
    return "request_approval" if waiting else "verify"


def build_graph(checkpointer: Any = None) -> Any:
    g = StateGraph(AIOpsState)
    for name, fn in [("collect", collect), ("detect", detect), ("dedupe", dedupe),
                     ("diagnose", diagnose), ("open_incident", open_incident),
                     ("auto_remediate", auto_remediate),
                     ("request_approval", request_approval), ("approval", approval),
                     ("verify", verify), ("close", close)]:
        g.add_node(name, telemetry.instrument(GRAPH, name, fn))
    g.add_edge(START, "collect")
    g.add_edge("collect", "detect")
    g.add_edge("detect", "dedupe")
    g.add_conditional_edges("dedupe", _after_dedupe, ["diagnose", END])
    g.add_edge("diagnose", "open_incident")
    g.add_edge("open_incident", "auto_remediate")
    g.add_conditional_edges("auto_remediate", _after_auto, ["request_approval", "verify"])
    g.add_edge("request_approval", "approval")
    g.add_edge("approval", "verify")
    g.add_edge("verify", "close")
    g.add_edge("close", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def compiled() -> Any:
    # An approval pause needs somewhere to wait, so never run without a saver.
    return build_graph(runtime.checkpointer() or runtime.checkpointer("memory"))


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "run_name": "aiops_sweep"}


def _outcome(thread_id: str, values: dict[str, Any]) -> dict[str, Any]:
    snapshot = compiled().get_state(_config(thread_id))
    paused = bool(snapshot.next)
    return {
        "run_id": values.get("run_id"), "thread_id": thread_id,
        "status": "AWAITING_APPROVAL" if paused else values.get("status"),
        "incident_id": values.get("incident_id"),
        "anomalies": [{k: a[k] for k in ("detector", "title", "severity", "value", "reason")}
                      for a in values.get("anomalies", [])],
        "suppressed": values.get("suppressed", []),
        "diagnosis": values.get("diagnosis"),
        "plan": values.get("plan", []),
        "actions": values.get("actions", []),
        "verification": values.get("verification"),
    }


def sweep(trigger: str = "manual", actor: str = "aiops") -> dict[str, Any]:
    """Run one sweep. Returns the outcome, including a pause for approval."""
    if not _sweep_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "A sweep is already running."}
    try:
        run_id = str(uuid.uuid4())
        thread_id = f"aiops:{run_id}"
        with telemetry.span("aiops.sweep", **{"cc.trace_id": run_id, "cc.trigger": trigger}):
            values = compiled().invoke(
                {"run_id": run_id, "thread_id": thread_id, "trigger": trigger,
                 "actor": actor, "actions": [], "status": "started"},
                _config(thread_id))
        telemetry.GRAPH_RUNS.labels(GRAPH, values.get("status") or "unknown").inc()
        return _outcome(thread_id, values)
    finally:
        _sweep_lock.release()


def decide(incident_id: str, *, approve: list[str], by: str) -> dict[str, Any]:
    """Resume a paused incident with a manager's decision."""
    row = query_one("SELECT thread_id, status FROM aiops_incident WHERE id = ?",
                    (incident_id,))
    if row is None:
        raise KeyError(incident_id)
    thread_id = row["thread_id"]
    if not compiled().get_state(_config(thread_id)).next:
        raise ValueError("This incident is not waiting for a decision.")
    values = compiled().invoke(Command(resume={"approve": approve, "by": by}),
                               _config(thread_id))
    return _outcome(thread_id, values)


def set_status(incident_id: str, status: str, by: str) -> None:
    """Manual close (RESOLVED) or dismissal (DISMISSED) by a manager."""
    if status not in ("RESOLVED", "DISMISSED"):
        raise ValueError(status)
    execute("UPDATE aiops_incident SET status = ?, decided_by = ?, updated_at = ? "
            "WHERE id = ?", (status, by, _now(), incident_id))
    audit.record("aiops_incident_updated", actor_type="staff", actor_id=by,
                 entity_type="aiops_incident", entity_id=incident_id,
                 payload={"status": status, "manual": True})


def incidents(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT * FROM aiops_incident"
    params: tuple = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    rows = query(sql + " ORDER BY created_at DESC LIMIT ?", (*params, limit))
    out = []
    for row in rows:
        item = dict(row)
        for key in ("detectors", "anomalies", "plan", "actions", "verification"):
            item[key] = json.loads(item[key] or "null")
        out.append(item)
    return out


# --------------------------------------------------------------------------
# scheduler
# --------------------------------------------------------------------------
_stop = threading.Event()
_thread: threading.Thread | None = None


def start_scheduler() -> None:
    global _thread
    settings = get_settings()
    if not settings.aiops_enabled or (_thread and _thread.is_alive()):
        return
    _stop.clear()

    def loop() -> None:
        # First sweep shortly after start, so the console isn't empty for 5 minutes.
        if _stop.wait(10):
            return
        while not _stop.is_set():
            try:
                sweep(trigger="schedule")
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                audit.record("aiops_sweep_failed", payload={"error": str(exc)[:300]})
            _stop.wait(max(30, settings.aiops_interval_seconds))

    _thread = threading.Thread(target=loop, name="aiops-scheduler", daemon=True)
    _thread.start()


def stop_scheduler() -> None:
    _stop.set()
