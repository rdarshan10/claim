"""Operational signals: what the system is doing right now.

Read-only SQL over tables the app already writes — llm_call, audit_event,
document, escalation_ticket, rpa_run, claim. Nothing here calls a model, so a
sweep works in template mode and during an LLM outage, which is exactly when it
matters most.

Every signal carries ``evidence``: the ids a playbook will act on, so the agent
never has to re-derive *which* documents are stuck from a number.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import get_settings
from app.db import query, query_one

# A document that has sat in a working state this long has lost its worker.
STUCK_DOCUMENT_MINUTES = 15
# An escalation nobody has picked up within this many hours has broken its SLA.
ESCALATION_SLA_HOURS = 4
# A registration bot run still RUNNING after this long has hung.
STUCK_RPA_MINUTES = 10

WORKING_DOC_STATES = ("UPLOADED", "SCANNING", "OCR", "CLASSIFYING", "EXTRACTING",
                      "VALIDATING")


@dataclass
class Signal:
    name: str
    value: float
    unit: str
    description: str
    # Below this many events the rate is noise; detectors ignore it.
    sample_size: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _since(minutes: int) -> str:
    return f"-{int(minutes)} minutes"


def _n(sql: str, params: tuple = ()) -> int:
    row = query_one(sql, params)
    return int((row[0] if row else 0) or 0)


def _events(event_type: str, window: str) -> int:
    return _n("SELECT COUNT(*) FROM audit_event WHERE event_type = ? "
              "AND julianday(at) >= julianday('now', ?)", (event_type, window))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# --------------------------------------------------------------------------
# collectors, one per area
# --------------------------------------------------------------------------
def _llm(window: str) -> list[Signal]:
    rows = query(
        """SELECT ok, cached, latency_ms, model, tokens_in, tokens_out FROM llm_call
           WHERE julianday(at) >= julianday('now', ?)""", (window,))
    live = [r for r in rows if not r["cached"]]
    errors = [r for r in live if not r["ok"]]
    latencies = sorted(int(r["latency_ms"]) for r in live if r["ok"])
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0

    by_model: dict[str, int] = {}
    for r in errors:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1

    samples = [json.loads(r["payload"]).get("error", "")[:200] for r in query(
        """SELECT payload FROM audit_event WHERE event_type = 'llm_unavailable'
           AND julianday(at) >= julianday('now', ?) ORDER BY id DESC LIMIT 5""", (window,))]

    from app.aiops import playbooks
    from app.llm import gateway, providers

    settings = get_settings()
    selection = providers.current()
    cost = 0.0
    for r in live:
        rate_in, rate_out = providers.price_for(r["model"])
        cost += (int(r["tokens_in"]) * rate_in + int(r["tokens_out"]) * rate_out) / 1e6

    return [
        Signal("llm_error_rate", _rate(len(errors), len(live)), "ratio",
               "Share of live LLM calls that failed.", len(live),
               {"errors_by_model": by_model, "error_samples": [s for s in samples if s],
                "provider": selection.provider, "llm_enabled": settings.llm_enabled}),
        Signal("llm_p95_latency_ms", float(p95), "ms",
               "95th percentile latency of successful LLM calls.", len(latencies)),
        Signal("llm_cache_hit_rate", _rate(len(rows) - len(live), len(rows)), "ratio",
               "Share of LLM calls served from the response cache.", len(rows),
               {"cache_entries": len(gateway._cache)}),
        Signal("llm_cost_usd", round(cost, 6), "usd",
               "Spend in the window at published rates.", len(live)),
        Signal("llm_template_mode", 0.0 if settings.llm_enabled else 1.0, "flag",
               "1 when the LLM is switched off and replies are templated.", 1,
               {"set_by_aiops": playbooks.STATE["template_mode_by_aiops"]}),
    ]


def _guardrails(window: str) -> list[Signal]:
    turns = _events("conversation_turn", window)
    fallbacks = _events("guardrail_template_fallback", window)
    output_fails = _events("guardrail_output_fail", window)
    blocks = query(
        """SELECT payload FROM audit_event WHERE event_type = 'guardrail_block'
           AND julianday(at) >= julianday('now', ?)""", (window,))
    injections = sum(1 for b in blocks
                     if "injection" in str(json.loads(b["payload"]).get("category")))
    actors = query(
        """SELECT actor_id, COUNT(*) AS n FROM audit_event
           WHERE event_type = 'guardrail_block' AND julianday(at) >= julianday('now', ?)
           GROUP BY actor_id ORDER BY n DESC LIMIT 5""", (window,))
    return [
        Signal("template_fallback_rate", _rate(fallbacks, turns), "ratio",
               "Turns whose reply failed grounding twice and fell back to a template.",
               turns),
        Signal("output_guard_fail_rate", _rate(output_fails, turns), "ratio",
               "Turns whose first reply failed the output guardrail.", turns),
        Signal("injection_attempts", float(injections), "count",
               "Prompt-injection attempts blocked in the window.", injections,
               {"top_actors": [dict(a) for a in actors]}),
    ]


def _documents(window: str) -> list[Signal]:
    placeholders = ",".join("?" for _ in WORKING_DOC_STATES)
    stuck = query(
        f"""SELECT id, status, uploaded_at FROM document WHERE status IN ({placeholders})
            AND julianday(uploaded_at) < julianday('now', ?)""",
        (*WORKING_DOC_STATES, _since(STUCK_DOCUMENT_MINUTES)))
    backlog = _n("SELECT COUNT(*) FROM document WHERE status = 'NEEDS_REVIEW'")
    verdicts = query(
        """SELECT payload FROM audit_event WHERE event_type = 'document_verdict'
           AND julianday(at) >= julianday('now', ?)""", (window,))
    rejected = sum(1 for v in verdicts
                   if json.loads(v["payload"]).get("status", "").startswith("REJECTED"))
    return [
        Signal("documents_stuck", float(len(stuck)), "count",
               f"Documents in a working state for over {STUCK_DOCUMENT_MINUTES} minutes.",
               len(stuck), {"document_ids": [r["id"] for r in stuck][:50]}),
        Signal("document_review_backlog", float(backlog), "count",
               "Documents waiting for a handler's decision.", backlog),
        Signal("document_rejection_rate", _rate(rejected, len(verdicts)), "ratio",
               "Share of pipeline verdicts that rejected the document.", len(verdicts)),
    ]


def _escalations() -> list[Signal]:
    breached = query(
        """SELECT id, priority, created_at FROM escalation_ticket
           WHERE status = 'OPEN' AND assigned_to IS NULL
             AND julianday(created_at) < julianday('now', ?)
           ORDER BY created_at""", (f"-{ESCALATION_SLA_HOURS} hours",))
    open_count = _n("SELECT COUNT(*) FROM escalation_ticket "
                    "WHERE status IN ('OPEN','ASSIGNED')")
    return [
        Signal("escalations_sla_breached", float(len(breached)), "count",
               f"Unassigned customer cases older than {ESCALATION_SLA_HOURS}h.",
               len(breached),
               {"ticket_ids": [r["id"] for r in breached if r["priority"] != "HIGH"][:50],
                "already_high": sum(1 for r in breached if r["priority"] == "HIGH")}),
        Signal("escalations_open", float(open_count), "count",
               "Customer cases waiting on or with a reviewer.", open_count),
    ]


def _registration(window: str) -> list[Signal]:
    failed = query(
        """SELECT r.id, r.fnol_id, r.error FROM rpa_run r
           JOIN fnol_request f ON f.id = r.fnol_id
           WHERE r.status = 'FAILED' AND julianday(r.started_at) >= julianday('now', ?)
             AND f.status = 'READY_TO_REGISTER' AND f.claim_id IS NULL""", (window,))
    hung = query(
        """SELECT id, fnol_id FROM rpa_run WHERE status = 'RUNNING'
           AND julianday(started_at) < julianday('now', ?)""",
        (_since(STUCK_RPA_MINUTES),))
    return [
        Signal("registration_failures", float(len(failed)), "count",
               "Registration bot runs that failed and left the FNOL unregistered.",
               len(failed),
               {"fnol_ids": sorted({r["fnol_id"] for r in failed}),
                "errors": [str(r["error"] or "")[:160] for r in failed][:5]}),
        Signal("registration_hung", float(len(hung)), "count",
               f"Registration runs still RUNNING after {STUCK_RPA_MINUTES} minutes.",
               len(hung), {"run_ids": [r["id"] for r in hung]}),
    ]


def _claims() -> list[Signal]:
    from app.services import diary

    try:
        # Claims that predate the diary have no review date until backfilled;
        # without this a stalled claim is invisible to the signal.
        diary.backfill()
        items = diary.due()
    except Exception:  # noqa: BLE001 - an older schema without diary columns
        items = []
    # A claim this agent already put on today's diary is in hand until a
    # handler works it; counting it again would make every sweep re-fire.
    at_risk = [i for i in items if i["at_risk"] and not i["chases_exhausted"]
               and not (i.get("diary_note") or "").startswith("AIOps:")]
    return [
        Signal("claims_at_risk", float(len(at_risk)), "count",
               f"Open claims untouched for {diary.AT_RISK_DAYS}+ days that nobody "
               f"has chased to the limit.", len(at_risk),
               {"claim_ids": [i["id"] for i in at_risk][:100],
                "needs_a_call": sum(1 for i in items if i["chases_exhausted"])}),
        Signal("claims_overdue_review", float(sum(1 for i in items
                                                  if (i["days_overdue"] or 0) > 0)),
               "count", "Open claims past their diary review date.", len(items)),
    ]


def _experience(window: str) -> list[Signal]:
    feedback = query(
        """SELECT payload FROM audit_event WHERE event_type = 'chat_feedback'
           AND julianday(at) >= julianday('now', ?)""", (window,))
    negative = sum(1 for f in feedback if not json.loads(f["payload"]).get("helpful", True))
    turns = query(
        """SELECT payload FROM audit_event WHERE event_type = 'conversation_turn'
           AND julianday(at) >= julianday('now', ?)""", (window,))
    upset = sum(1 for t in turns if json.loads(t["payload"]).get("sentiment")
                in ("frustrated", "distressed"))
    return [
        Signal("negative_feedback_rate", _rate(negative, len(feedback)), "ratio",
               "Share of thumbs-down on assistant replies.", len(feedback)),
        Signal("upset_customer_rate", _rate(upset, len(turns)), "ratio",
               "Share of turns read as frustrated or distressed.", len(turns)),
    ]


def collect(window_minutes: int | None = None) -> dict[str, Signal]:
    """Every signal, keyed by name."""
    window = _since(window_minutes or get_settings().aiops_window_minutes)
    signals: list[Signal] = [
        *_llm(window), *_guardrails(window), *_documents(window), *_escalations(),
        *_registration(window), *_claims(), *_experience(window),
    ]
    return {s.name: s for s in signals}
