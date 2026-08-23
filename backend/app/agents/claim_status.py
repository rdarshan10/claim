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

# How customers name a claim when they don't have its reference to hand.
CLAIM_TYPE_WORDS: dict[str, tuple[str, ...]] = {
    "motor": ("motor", "car", "vehicle", "driving"),
    "home": ("home", "house", "property", "flat", "buildings", "contents"),
    "health": ("health", "medical", "hospital", "treatment"),
}


def _match_by_type(message: str, claims: list[dict]) -> dict | None:
    """Resolve "the motor one" or "my home claim" to a specific claim.

    Only when exactly one claim has that type. With two motor claims the phrase
    is genuinely ambiguous, and picking one would reintroduce the bug this
    solves — answering confidently about a claim the customer didn't mean.
    """
    lowered = (message or "").lower()
    for claim_type, words in CLAIM_TYPE_WORDS.items():
        if not any(re.search(rf"\b{word}\b", lowered) for word in words):
            continue
        hits = [c for c in claims if (c.get("claim_type") or "").lower() == claim_type]
        if len(hits) == 1:
            return hits[0]
    return None


# A customer with a long history should not get a wall of cards; the reply
# still names every claim, and they can ask about any one by reference.
MAX_OVERVIEW_CARDS = 3

# What a customer calls the stage before a claim exists. Staff say FNOL.
# Verification language rather than "with the claims team": before a claim
# exists there is nobody assigned to it, and naming a team implies someone is
# already working the case. What is actually happening is that the details are
# being checked before a claim can be opened.
NOTIFICATION_STAGE = {
    "COLLECTING": "still being filled in",
    "SUBMITTED": "sent for verification",
    "UNDER_REVIEW": "being verified",
    "READY_TO_REGISTER": "verified and about to be set up as a claim",
    "REGISTERING": "being set up as a claim now",
}


# "how many FNOLs", "what about my notification" — asking after something
# already reported, which is answered from notifications rather than claims.
NOTIFICATION_QUESTION = re.compile(
    r"\bfnols?\b|\bnotifications? of loss\b|\bnotifications?\b", re.IGNORECASE)


def _notifications(customer_id: str) -> list[dict]:
    """Notification-of-loss records that have not yet become claims.

    A customer who has reported something counts it as raised, but until it is
    registered there is no claim row for it — so answering only from the claims
    table told someone with an in-flight notification they had none. Registered
    ones are deliberately excluded: those *are* claims and are already listed,
    and counting both would double up.
    """
    from app.fnol import intake

    try:
        records = intake.list_for_customer(customer_id)
    except Exception:  # noqa: BLE001 - never break a status answer over this
        return []
    return [r for r in records
            if not r.get("claim_id") and r["status"] != "REGISTERED"]


def _summarise(records: list[dict]) -> list[dict]:
    """Notification records as facts, in words a customer would recognise."""
    return [
        {"reference": r["reference"],
         "claim_type": r.get("claim_type") or "not yet identified",
         "stage": NOTIFICATION_STAGE.get(r["status"], r["status"].lower()),
         "reported_on": (r.get("created_at") or "")[:10]}
        for r in records
    ]


def _notification_answer(state: GraphState, pending: list[dict],
                         claims: list[dict]) -> GraphState:
    """Answer about notifications of loss, with notification cards.

    They asked about notifications, so the cards below the reply have to be
    notifications. Falling through to the claim timeline put an unrelated claim
    on screen under an answer about FNOLs — the reply and the card disagreeing,
    which is the thing this whole area keeps getting wrong.
    """
    from app.fnol import intake

    state.facts = {
        "notifications_of_loss": _summarise(pending),
        "notifications_of_loss_count": len(pending),
        "claims_on_this_account_count": len(claims),
        "note": ("These are notifications of loss, not claims: they have a "
                 "reference but no claim number yet. Never call them claims and "
                 "never invent a claim number for one. Answer only about these; "
                 "the customer's existing claims are a separate question."),
    }
    for record in pending[:MAX_OVERVIEW_CARDS]:
        state.cards.append(intake.status_card(record))
    return state


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


def _overview(state: GraphState, claims: list[dict]) -> GraphState:
    """Answer about every open claim when the customer named none.

    Each gets its own timeline card, and the facts carry all of them so the
    reply speaks to the set rather than to one picked arbitrarily. Documents are
    deliberately left out here: a to-do list per claim buries the answer to the
    question actually asked. Naming a claim drops back into the single-claim
    path, which does show what is outstanding.
    """
    pending = _notifications(state.customer_id)
    state.facts = {
        "claims": [_summary(c) for c in claims],
        "open_claim_count": len(claims),
        "notifications_of_loss_not_yet_claims": _summarise(pending),
        "notifications_of_loss_count": len(pending),
        "note": ("This customer has more than one open claim. Give a one-line "
                 "update on each, naming its reference, and ask which they want "
                 "to go into. Do not answer as though there is only one."
                 + (" They also have notifications of loss that are not claims "
                    "yet — mention those separately, and never call them claims "
                    "or invent a claim number for them." if pending else "")),
    }
    for claim in claims[:MAX_OVERVIEW_CARDS]:
        history = tools.get_status_history(state.customer_id, claim["id"])
        prediction = tools.predict_timeline(state.customer_id, claim["id"])
        checklist = tools.get_required_documents(state.customer_id, claim["id"])
        state.cards.append(_timeline_card(claim, history, prediction, checklist))
    return state


def run(state: GraphState) -> GraphState:
    claims = tools.get_claims(state.customer_id)
    pending = _notifications(state.customer_id)

    # Asked about notifications specifically — answer from those, with their
    # own cards, rather than falling through to a claim they didn't ask about.
    if pending and NOTIFICATION_QUESTION.search(state.message or ""):
        return _notification_answer(state, pending, claims)

    if not claims:
        state.facts = {
            "claims": [],
            "notifications_of_loss_not_yet_claims": _summarise(pending),
            "note": ("This customer has no claims on file."
                     + (" They do have a notification of loss in progress — say so "
                        "rather than telling them they have nothing." if pending else "")),
        }
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
    else:
        # "the motor one" is as specific as a reference number when only one
        # claim has that type. This has to be checked BEFORE the claim carried
        # over from the previous turn: otherwise a follow-up naming a different
        # claim keeps answering about the old one, and the reply names the claim
        # the customer asked for while the cards below show the sticky one.
        claim = _match_by_type(state.message, claims)
        if claim is None and state.active_claim_id:
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
        # Reported but not yet registered as claims. Counted separately so
        # "how many have I raised?" can be answered honestly: these are real
        # things the customer has told us about, but they have no claim number
        # yet and must not be presented as claims.
        "notifications_of_loss_not_yet_claims": _summarise(pending),
        "notifications_of_loss_count": len(pending),
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
