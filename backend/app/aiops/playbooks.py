"""Remediation playbooks: the complete list of things the AIOps agent can do.

The model never writes SQL, never calls an arbitrary function and never invents
an action. It may only *recommend* playbook ids from this catalogue, and only
ones registered against a detector that actually fired — anything else is
dropped before execution.

Risk tiers:

* ``low``  — idempotent, internal, reversible, no customer-visible change
  (clear a cache, re-queue a stuck job, raise a ticket's priority, put a claim
  on today's diary). Runs automatically when ``AIOPS_AUTO_REMEDIATE`` is on.
* ``high`` — changes what customers get or touches the core system (switch the
  LLM provider, drop to template mode, re-run a claim registration). Always
  pauses the graph for a manager's decision.

Every execution is audited with the incident id.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query_one

MAX_BATCH = 20

# Whether template mode is on because this agent switched it on. A manager who
# starts in template mode on purpose (the degradation demo) is not an incident.
STATE: dict[str, bool] = {"template_mode_by_aiops": False}


@dataclass(frozen=True)
class Playbook:
    id: str
    title: str
    risk: str
    detectors: tuple[str, ...]
    description: str
    run: Callable[[dict[str, Any]], dict[str, Any]] = field(compare=False)
    # (applicable, reason) — checked before it is offered at all.
    precondition: Callable[[dict[str, Any]], tuple[bool, str]] | None = field(
        default=None, compare=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "risk": self.risk,
                "detectors": list(self.detectors), "description": self.description}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# actions — each takes the merged evidence of the anomalies it answers
# --------------------------------------------------------------------------
def _clear_llm_cache(evidence: dict[str, Any]) -> dict[str, Any]:
    from app.llm import gateway

    before = len(gateway._cache)
    gateway.clear_cache()
    gateway.reset_clients()
    return {"cleared_entries": before}


def _other_provider() -> str | None:
    from app.llm import providers

    current = providers.current().provider
    return next((key for key in providers.PROVIDERS
                 if key != current and providers.configured(key)), None)


def _failover_pre(evidence: dict[str, Any]) -> tuple[bool, str]:
    target = _other_provider()
    return (True, f"fail over to {target}") if target else \
        (False, "no second provider is configured")


def _failover(evidence: dict[str, Any]) -> dict[str, Any]:
    from app.llm import gateway, providers

    target = _other_provider()
    if not target:
        return {"skipped": "no second provider is configured"}
    before = providers.current().as_dict()
    after = providers.select(target).as_dict()
    gateway.clear_cache()
    gateway.reset_clients()
    return {"from": before, "to": after}


def _template_mode_pre(evidence: dict[str, Any]) -> tuple[bool, str]:
    return (True, "LLM is on") if get_settings().llm_enabled else \
        (False, "template mode is already on")


def _enable_template_mode(evidence: dict[str, Any]) -> dict[str, Any]:
    """Stop calling the model; every reply comes from the deterministic templates.

    Status, checklists, rejections and predictions keep working (UC-N7); only
    the prose gets plainer. Reversed by ``restore_llm``.
    """
    from app.llm import gateway

    get_settings().llm_enabled = False
    STATE["template_mode_by_aiops"] = True
    gateway.reset_clients()
    return {"llm_enabled": False}


def _restore_pre(evidence: dict[str, Any]) -> tuple[bool, str]:
    return (True, "template mode is on") if not get_settings().llm_enabled else \
        (False, "LLM is already on")


def _restore_llm(evidence: dict[str, Any]) -> dict[str, Any]:
    from app.llm import gateway, providers

    get_settings().llm_enabled = True
    STATE["template_mode_by_aiops"] = False
    gateway.reset_clients()
    return {"llm_enabled": True, "selection": providers.current().as_dict()}


def _requeue_documents(evidence: dict[str, Any]) -> dict[str, Any]:
    from app.documents import pipeline

    ids = list(dict.fromkeys(evidence.get("document_ids", [])))[:MAX_BATCH]
    for doc_id in ids:
        execute("UPDATE document SET status = 'UPLOADED' WHERE id = ?", (doc_id,))
        threading.Thread(target=pipeline.process, args=(doc_id,), daemon=True).start()
    return {"requeued": ids}


def _prioritise_escalations(evidence: dict[str, Any]) -> dict[str, Any]:
    ids = list(dict.fromkeys(evidence.get("ticket_ids", [])))[:MAX_BATCH * 5]
    for ticket_id in ids:
        execute("UPDATE escalation_ticket SET priority = 'HIGH' WHERE id = ? "
                "AND status = 'OPEN'", (ticket_id,))
    return {"raised_to_high": ids}


def _schedule_chases(evidence: dict[str, Any]) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date().isoformat()
    ids = list(dict.fromkeys(evidence.get("claim_ids", [])))[:MAX_BATCH * 5]
    for claim_id in ids:
        # Bring the date forward to today, but never push an already-overdue
        # date later — overdue claims must keep their place at the top.
        execute("UPDATE claim SET next_review_date = MIN(COALESCE(next_review_date, ?), ?), "
                "diary_note = ? WHERE id = ?",
                (today, today, "AIOps: untouched for 30+ days — chase today before the "
                               "customer has to", claim_id))
    return {"put_on_todays_diary": ids}


def _fail_hung_registrations(evidence: dict[str, Any]) -> dict[str, Any]:
    """Mirror what the bot does on failure: run FAILED, FNOL back to the queue."""
    from app.fnol import intake

    done = []
    for run_id in evidence.get("run_ids", [])[:MAX_BATCH]:
        row = query_one("SELECT fnol_id, status FROM rpa_run WHERE id = ?", (run_id,))
        if not row or row["status"] != "RUNNING":
            continue
        execute("UPDATE rpa_run SET status = 'FAILED', error = ?, finished_at = ? "
                "WHERE id = ?", ("Marked failed by AIOps: no progress", _now(), run_id))
        fnol = intake.get(row["fnol_id"])
        if fnol and fnol["status"] == "REGISTERING" and not fnol.get("claim_id"):
            intake.set_status(row["fnol_id"], "READY_TO_REGISTER")
        done.append(run_id)
    return {"failed_runs": done}


def _retry_registrations(evidence: dict[str, Any]) -> dict[str, Any]:
    from app.fnol import registration

    started, refused = [], []
    for fnol_id in evidence.get("fnol_ids", [])[:MAX_BATCH]:
        try:
            started.append(registration.start(fnol_id, "aiops"))
        except (KeyError, ValueError) as exc:
            refused.append({"fnol_id": fnol_id, "reason": str(exc)[:160]})
    return {"runs_started": started, "refused": refused}


PLAYBOOKS: dict[str, Playbook] = {p.id: p for p in [
    Playbook("clear_llm_cache", "Clear the LLM response cache and clients", "low",
             ("template_fallbacks", "grounding_failures", "negative_feedback",
              "llm_errors"),
             "Drops cached answers and pooled connections so the next call is fresh.",
             _clear_llm_cache),
    Playbook("failover_llm_provider", "Fail over to the other LLM provider", "high",
             ("llm_errors", "llm_latency"),
             "Switches every model call to the second configured provider.",
             _failover, _failover_pre),
    Playbook("enable_template_mode", "Switch to template mode", "high",
             ("llm_errors", "template_fallbacks"),
             "Stops calling the model; replies come from deterministic templates.",
             _enable_template_mode, _template_mode_pre),
    Playbook("restore_llm", "Turn the LLM back on", "high", ("template_mode",),
             "Re-enables model calls after template mode.", _restore_llm, _restore_pre),
    Playbook("requeue_stuck_documents", "Re-run stuck documents", "low",
             ("documents_stuck",),
             "Restarts the verification pipeline for documents that lost their worker.",
             _requeue_documents),
    Playbook("prioritise_breached_escalations", "Raise breached cases to HIGH", "low",
             ("escalation_sla",),
             "Moves unassigned cases past their SLA to the top of the reviewer queue.",
             _prioritise_escalations),
    Playbook("schedule_claim_chases", "Put stale claims on today's diary", "low",
             ("claims_at_risk",),
             "Brings untouched claims forward so a handler chases them today.",
             _schedule_chases),
    Playbook("fail_hung_registrations", "Release hung registration runs", "low",
             ("registration_hung",),
             "Marks hung bot runs failed and returns the FNOL to the queue.",
             _fail_hung_registrations),
    Playbook("retry_failed_registrations", "Retry failed claim registrations", "high",
             ("registration_failures",),
             "Re-runs the registration bot against the core system.",
             _retry_registrations),
]}


def applicable(detector_ids: list[str], evidence: dict[str, Any]) -> list[Playbook]:
    """Playbooks registered against any fired detector whose precondition holds."""
    out = []
    for playbook in PLAYBOOKS.values():
        if not set(playbook.detectors) & set(detector_ids):
            continue
        if playbook.precondition and not playbook.precondition(evidence)[0]:
            continue
        out.append(playbook)
    return out


def run(playbook_id: str, evidence: dict[str, Any], *, incident_id: str,
        actor: str) -> dict[str, Any]:
    """Execute one playbook, audited. Never raises."""
    from app.ops import telemetry

    playbook = PLAYBOOKS[playbook_id]
    try:
        result = playbook.run(evidence)
        outcome = "skipped" if result.get("skipped") else "ok"
    except Exception as exc:  # noqa: BLE001 - one bad action must not sink the incident
        result, outcome = {"error": str(exc)[:300]}, "error"
    telemetry.AIOPS_REMEDIATIONS.labels(playbook_id, outcome).inc()
    audit.record("aiops_remediation", actor_type="agent", actor_id=actor,
                 entity_type="aiops_incident", entity_id=incident_id,
                 payload={"playbook": playbook_id, "risk": playbook.risk,
                          "outcome": outcome, "result": result})
    return {"playbook": playbook_id, "title": playbook.title, "risk": playbook.risk,
            "outcome": outcome, "result": result, "by": actor, "at": _now()}
