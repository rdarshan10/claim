"""Claim Status Agent (§9).

Tools only: it has no capability to modify a claim or read documents. Every
fact it emits comes from a repository call, so the reply can be fact-checked
token-by-token by the output guardrail.
"""
from __future__ import annotations

import re

from app.agents.state import GraphState
from app.agents.tools import claim_tools as tools
from app.services import timeline_prediction

CLAIM_REF = re.compile(r"\bCLM-\d+\b", re.IGNORECASE)


def run(state: GraphState) -> GraphState:
    claims = tools.get_claims(state.customer_id)

    if not claims:
        state.facts = {"claims": [], "note": "This customer has no claims on file."}
        return state

    # Resolve which claim the customer means.
    claim = None
    if match := CLAIM_REF.search(state.message):
        claim = tools.find_claim_by_number(state.customer_id, match.group(0))
        if claim is None:
            # Never confirm or deny another customer's claim (UC-N1).
            state.facts = {
                "claims": [_summary(c) for c in claims],
                "lookup_failed": True,
                "note": ("No claim with that reference exists on this customer's account. "
                         "Do not speculate about who it might belong to. Offer to look up "
                         "the claims listed above instead."),
            }
            return state
    elif state.active_claim_id:
        claim = tools.get_claim_detail(state.customer_id, state.active_claim_id)

    if claim is None:
        open_claims = [c for c in claims
                       if c["status"] not in ("SETTLED", "REJECTED", "WITHDRAWN")]
        claim = (open_claims or claims)[0]

    state.active_claim_id = claim["id"]

    history = tools.get_status_history(state.customer_id, claim["id"])
    prediction = tools.predict_timeline(state.customer_id, claim["id"])
    checklist = tools.get_required_documents(state.customer_id, claim["id"])
    overdue = timeline_prediction.is_overdue(claim, history)

    state.facts = {
        "claim": _summary(claim),
        "current_stage_meaning": timeline_prediction.STAGE_MEANING.get(claim["status"], ""),
        "status_history": [
            {"to_status": h["to_status"], "changed_at": h["changed_at"][:10],
             "reason": h["reason"]}
            for h in history
        ],
        "prediction": prediction,
        "outstanding_documents": checklist["outstanding_mandatory"],
        "other_claims_count": len(claims) - 1,
    }

    if overdue:
        state.facts["running_late"] = overdue
        # Acknowledge the wait before informing (§16 hard rules).
        if state.tone_profile == "neutral-warm":
            state.tone_profile = "apologetic-accountable"  # type: ignore[assignment]

    if claim["status"] == "APPROVED" and state.sentiment == "calm":
        state.tone_profile = "celebratory"  # type: ignore[assignment]

    state.cards.append({
        "card_type": "claim_timeline",
        "payload": {
            "claim_number": claim["claim_number"],
            "claim_type": claim["claim_type"],
            "status": claim["status"],
            "status_meaning": timeline_prediction.STAGE_MEANING.get(claim["status"], ""),
            "claimed_amount": claim.get("claimed_amount"),
            "approved_amount": claim.get("approved_amount"),
            "incident_date": claim.get("incident_date"),
            "history": [
                {"status": h["to_status"], "date": h["changed_at"][:10]} for h in history
            ],
            "prediction": prediction,
            "outstanding_documents": checklist["outstanding_mandatory"],
        },
    })
    return state


def _summary(claim: dict) -> dict:
    return {
        "claim_number": claim["claim_number"],
        "claim_type": claim["claim_type"],
        "subtype": claim.get("subtype"),
        "status": claim["status"],
        "claimed_amount": claim.get("claimed_amount"),
        "approved_amount": claim.get("approved_amount"),
        "incident_date": claim.get("incident_date"),
        "filed_at": (claim.get("filed_at") or "")[:10],
        "policy_number": claim.get("policy_number"),
    }
