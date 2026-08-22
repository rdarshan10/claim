"""Escalation Agent (§9): builds the AI context packet and opens a ticket."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone  # noqa: F401 - used by _reassure

from app.agents.state import GraphState
from app.audit import logger as audit
from app.db import execute
from app.repositories import claims as repo

def run(state: GraphState, reason: str = "Customer requested a human") -> GraphState:

    # Asking twice must not open a second ticket — it should reassure and keep
    # the reviewer's context current. Scoped to the customer, not the
    # conversation: a case belongs to the person, so starting a fresh chat and
    # asking again is still the same case.
    from app.repositories import conversations as conv_repo

    if existing := conv_repo.open_ticket_for_customer(state.customer_id):
        # The ticket keeps pointing at the conversation that raised it — that's
        # where the reviewer finds the context. Replies are delivered to
        # whichever thread the customer is using now, resolved at send time.
        return _reassure(state, existing)

    claims = repo.get_claims(state.customer_id)
    claim = None
    if state.active_claim_id:
        claim = repo.get_claim(state.active_claim_id, state.customer_id)
    if claim is None and claims:
        claim = claims[0]

    checklist = repo.checklist(claim["id"], state.customer_id) if claim else None

    context_packet = {
        "conversation_summary": _summarise(state),
        "customer_sentiment": state.sentiment,
        "intent": state.intent,
        "claim_snapshot": {
            "claim_number": claim["claim_number"],
            "status": claim["status"],
            "claim_type": claim["claim_type"],
            "claimed_amount": claim.get("claimed_amount"),
            "incident_date": claim.get("incident_date"),
        } if claim else None,
        "documents_outstanding": checklist["outstanding_mandatory"] if checklist else [],
        "what_the_ai_did": state.guardrail_flags + [f"routed to {state.intent}"],
        "recent_messages": state.history[-6:] + [{"role": "user", "content": state.message}],
    }

    ticket_id = str(uuid.uuid4())
    execute(
        """INSERT INTO escalation_ticket
           (id, conversation_id, claim_id, customer_id, priority, reason,
            context_packet, status, created_at)
           VALUES (?,?,?,?,?,?,?, 'OPEN', ?)""",
        (ticket_id, state.conversation_id or None, claim["id"] if claim else None,
         state.customer_id, "NORMAL", reason, json.dumps(context_packet, default=str),
         datetime.now(timezone.utc).isoformat()),
    )

    audit.record("escalation_created", actor_type="agent", actor_id="escalation",
                 entity_type="ticket", entity_id=ticket_id,
                 payload={"reason": reason, "sentiment": state.sentiment},
                 trace_id=state.trace_id)

    # Pause the assistant: the conversation now waits on a person (§9 interrupt).
    if state.conversation_id:
        from app.repositories import conversations as conv_repo

        conv_repo.set_mode(state.conversation_id, conv_repo.MODE_AWAITING,
                           ticket_id=ticket_id)

    state.escalation_ticket_id = ticket_id
    eta = "within 1 working day"
    state.facts = {
        "colleague_asked": {
            "reference_the_customer_can_quote": ticket_id[:8].upper(),
            "when_they_will_be_in_touch": eta,
            "note": ("Say a colleague will pick this up and give the reference. "
                     "Do not call it an escalation, a ticket or a case — to the "
                     "customer this is simply someone getting back to them."),
        },
        "claim": context_packet["claim_snapshot"],
    }
    state.cards.append({
        "card_type": "handoff",
        "payload": {"ticket_id": ticket_id[:8].upper(), "eta": eta},
    })
    return state


def load_ticket(ticket_id: str) -> dict | None:
    from app.db import query_one

    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    return dict(row) if row else None


def touch_case(state: GraphState, ticket: dict, *, chased: bool = False) -> dict:
    """Keep an open case current with whatever the customer just said.

    Called for *every* turn while a case is open, not only when they ask for a
    human again. Someone venting "why is this taking so long" routes to
    claim_status, but the reviewer still needs to know they're now angry — the
    packet used to be written once at creation and never moved.
    """
    try:
        packet = json.loads(ticket.get("context_packet") or "{}")
    except json.JSONDecodeError:
        packet = {}

    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "sentiment": state.sentiment,
        "intent": state.intent,
        "message": state.message[:300],
        "asked_for_human": chased,
    }
    follow_ups = packet.get("follow_ups", [])
    follow_ups.append(entry)
    packet["follow_ups"] = follow_ups[-10:]

    # Capture how they started before overwriting with how they feel now — the
    # shift from calm to frustrated is the signal a reviewer needs.
    packet.setdefault("initial_sentiment", packet.get("customer_sentiment"))
    packet["customer_sentiment"] = state.sentiment
    packet["chased_count"] = sum(1 for f in packet["follow_ups"]
                                 if f.get("asked_for_human"))
    packet["messages_since_raised"] = len(packet["follow_ups"])
    packet["conversation_summary"] = _summarise(state)
    packet["recent_messages"] = (
        state.history[-6:] + [{"role": "user", "content": state.message}]
    )

    execute("UPDATE escalation_ticket SET context_packet = ? WHERE id = ?",
            (json.dumps(packet, default=str), ticket["id"]))
    audit.record("escalation_context_updated", actor_type="customer",
                 actor_id=state.customer_id, entity_type="ticket",
                 entity_id=ticket["id"],
                 payload={"sentiment": state.sentiment, "intent": state.intent,
                          "asked_for_human": chased},
                 trace_id=state.trace_id)
    return packet


def _reassure(state: GraphState, ticket: dict) -> GraphState:
    """A case is already open: chase it rather than duplicating it."""
    ticket_id = ticket["id"]
    touch_case(state, ticket, chased=True)

    assigned = ticket.get("assigned_to")
    state.escalation_ticket_id = ticket_id
    state.facts = {
        "colleague_asked": {
            "reference_the_customer_can_quote": ticket_id[:8].upper(),
            "already_being_looked_at": True,
            "who_has_it": assigned,
            "asked_on": (ticket.get("created_at") or "")[:10],
            "note": ("A colleague is already on this. Reassure them, confirm it is "
                     "being chased, and do not promise a new time. Do not call it "
                     "an escalation, a ticket or a case."),
        }
    }
    state.cards.append({
        "card_type": "handoff_status",
        "payload": {"ticket_id": ticket_id[:8].upper(),
                    "assigned_to": assigned, "status": ticket.get("status")},
    })
    return state


def _summarise(state: GraphState) -> str:
    """Deterministic summary — no LLM call on the escalation path."""
    turns = len(state.history) // 2
    topics = {turn.get("intent") for turn in state.history if turn.get("intent")}
    topics.discard(None)
    return (
        f"Customer messaged {turns + 1} time(s) this session. "
        f"Topics: {', '.join(sorted(t for t in topics if t)) or state.intent or 'general'}. "
        f"Detected sentiment: {state.sentiment}. "
        f"Latest message: \"{state.message[:200]}\""
    )
