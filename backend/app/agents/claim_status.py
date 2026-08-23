"""Claim Status Agent (§9).

Tools only: it has no capability to modify a claim or read documents. Every
fact it emits comes from a repository call, so the reply can be fact-checked
token-by-token by the output guardrail.
"""
from __future__ import annotations

import re

from app.agents import document
from app.agents.state import GraphState
from app.agents.tools import claim_tools as tools
from app.services import timeline_prediction

CLAIM_REF = re.compile(r"\bCLM-\d+\b", re.IGNORECASE)


def _timeline_card(claim: dict, history: list, prediction: dict,
                   checklist: dict) -> dict:
    """The where-is-it card for one claim."""
    return {
        "card_type": "claim_timeline",
        "payload": {
            "claim_number": claim["claim_number"],
            "claim_type": claim["claim_type"],
            "status": claim["status"],
            "status_meaning": timeline_prediction.stage_meaning(
                claim["status"], awaiting_customer=bool(checklist["awaiting_customer"])),
            "claimed_amount": claim.get("claimed_amount"),
            "approved_amount": claim.get("approved_amount"),
            "incident_date": claim.get("incident_date"),
            "history": [
                {"status": h["to_status"], "date": h["changed_at"][:10]} for h in history
            ],
            "prediction": prediction,
            "outstanding_documents": checklist["awaiting_customer_labels"],
        },
    }


# A customer with a long history should not get a wall of cards; the reply
# still names every claim, and they can ask about any one by reference.
MAX_OVERVIEW_CARDS = 3


def _overview(state: GraphState, claims: list[dict]) -> GraphState:
    """Answer about every open claim when the customer named none.

    Each gets its own timeline card, and the facts carry all of them so the
    reply speaks to the set rather than to one picked arbitrarily. Documents are
    deliberately left out here: a to-do list per claim buries the answer to the
    question actually asked. Naming a claim drops back into the single-claim
    path, which does show what is outstanding.
    """
    state.facts = {
        "claims": [_summary(c) for c in claims],
        "open_claim_count": len(claims),
        "note": ("This customer has more than one open claim. Give a one-line "
                 "update on each, naming its reference, and ask which they want "
                 "to go into. Do not answer as though there is only one."),
    }
    for claim in claims[:MAX_OVERVIEW_CARDS]:
        history = tools.get_status_history(state.customer_id, claim["id"])
        prediction = tools.predict_timeline(state.customer_id, claim["id"])
        checklist = tools.get_required_documents(state.customer_id, claim["id"])
        state.cards.append(_timeline_card(claim, history, prediction, checklist))
    return state


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
        candidates = open_claims or claims
        # More than one and they named none: answer about all of them. Silently
        # taking the first meant "what's happening with my claim" was answered
        # about whichever sorted first, while the card below showed another —
        # the reply and the card contradicting each other on screen.
        if len(candidates) > 1:
            return _overview(state, candidates)
        claim = candidates[0]

    state.active_claim_id = claim["id"]

    history = tools.get_status_history(state.customer_id, claim["id"])
    prediction = tools.predict_timeline(state.customer_id, claim["id"])
    checklist = tools.get_required_documents(state.customer_id, claim["id"])
    overdue = timeline_prediction.is_overdue(claim, history)

    # The stage sentence is the single source of truth for "where is it". The
    # summary's own status label was a second, terser version of the same thing
    # and the two disagreed once DOCS_PENDING started meaning two things — the
    # reply would quote "waiting for documents" and then say the documents were
    # already in.
    summary = _summary(claim)
    summary.pop("status", None)

    state.facts = {
        "claim": summary,
        "current_stage_meaning": timeline_prediction.stage_meaning(
            claim["status"], awaiting_customer=bool(checklist["awaiting_customer"])),
        "status_history": [
            {"to_status": h["to_status"], "changed_at": h["changed_at"][:10],
             "reason": h["reason"]}
            for h in history
        ],
        # Only the headline estimate. The per-stage breakdown stays out of the
        # facts: handed all five, the responder quoted internal stage dates
        # that contradicted the single date shown on the card beside it.
        "prediction": {
            "expected_completion": (prediction or {}).get("predicted_settlement_date"),
            "give_or_take_days": (prediction or {}).get("band_days"),
            "confidence": (prediction or {}).get("confidence"),
        } if prediction and not (prediction or {}).get("terminal") else None,
        "documents_we_need_from_you": checklist["awaiting_customer_labels"],
        "documents_already_with_us": checklist["with_us_labels"],
        # Every claim on the account, so "how many claims do I have?" and "what
        # is the other one?" can actually be answered. A bare count could only
        # produce "one other claim", with nothing to name when asked which.
        "all_claims_on_this_account": [
            {"claim_number": c["claim_number"],
             "type": c["claim_type"],
             "stage": timeline_prediction.stage_meaning(c["status"]),
             "incident_date": c.get("incident_date"),
             "is_the_one_discussed_above": c["id"] == claim["id"]}
            for c in claims
        ],
        "claims_on_this_account_count": len(claims),
    }

    if overdue:
        state.facts["running_late"] = overdue
        # Acknowledge the wait before informing (§16 hard rules).
        if state.tone_profile == "neutral-warm":
            state.tone_profile = "apologetic-accountable"  # type: ignore[assignment]

    if claim["status"] == "APPROVED" and state.sentiment == "calm":
        state.tone_profile = "celebratory"  # type: ignore[assignment]

    # Order matters: they asked where their claim is, so answer that first. The
    # timeline leads, and what we still need from them follows as the next step
    # — opening with a to-do list reads as a demand rather than an answer.
    state.cards.append(_timeline_card(claim, history, prediction, checklist))

    if checklist["outstanding_mandatory"]:
        state.cards.append({
            "card_type": "action_needed",
            "payload": {
                "claim_id": claim["id"],
                "claim_number": claim["claim_number"],
                "items": [
                    {"doc_type": item["doc_type"],
                     "label": item["doc_type"].replace("_", " "),
                     "state": item["state"],
                     "guidance": document.GUIDANCE.get(item["doc_type"], "")}
                    for item in checklist["items"]
                    if item["mandatory"] and item["state"] in ("MISSING", "REJECTED")
                ],
                "with_us": [
                    {"doc_type": item["doc_type"],
                     "label": item["doc_type"].replace("_", " "),
                     "state": item["state"]}
                    for item in checklist["items"]
                    if item["mandatory"] and item["state"] in ("IN_REVIEW", "UPLOADED")
                ],
            },
        })
    return state


# What each stage is called in front of a customer. The enum is a database
# value, not a phrase anyone should read.
STATUS_LABEL = {
    "FILED": "filed",
    "DOCS_PENDING": "waiting for documents",
    "IN_ASSESSMENT": "being assessed",
    "ADDITIONAL_INFO": "waiting for more information",
    "APPROVED": "approved",
    "PAYMENT_IN_PROGRESS": "being paid",
    "SETTLED": "settled",
    "REJECTED": "not approved",
    "WITHDRAWN": "withdrawn",
}


def _summary(claim: dict) -> dict:
    return {
        "claim_number": claim["claim_number"],
        "claim_type": claim["claim_type"],
        "subtype": claim.get("subtype"),
        "status": STATUS_LABEL.get(claim["status"], claim["status"].lower()),
        "claimed_amount": claim.get("claimed_amount"),
        "approved_amount": claim.get("approved_amount"),
        "incident_date": claim.get("incident_date"),
        "filed_at": (claim.get("filed_at") or "")[:10],
        "policy_number": claim.get("policy_number"),
    }
