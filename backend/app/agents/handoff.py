"""Human-in-the-loop resume (§9 ``HumanInterrupt -> resume``).

The escalation node pauses the graph; this module is the other half. When a
reviewer acts, their answer re-enters the same conversation and the customer is
pinged back — one continuous thread, no dead end.

**The assistant is always the middleman.** The customer talks only to
ClaimCompanion, which acts for them: it carries their case to a reviewer with
full context, keeps them posted while they wait, and brings the answer back in
its own voice. The reviewer sits behind the assistant and never addresses the
customer directly.

That makes one rule load-bearing: the assistant may **re-voice** a reviewer's
answer, never **re-decide** it. Anything the model produces that isn't supported
by the reviewer's note is discarded and the note is quoted verbatim instead.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.audit import logger as audit
from app.db import execute
from app.guardrails import output_guards
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.repositories import conversations as conv_repo

TONE_BY_SENTIMENT = {
    "distressed": "gentle-supportive",
    "frustrated": "apologetic-accountable",
    "confused": "reassuring",
    "calm": "neutral-warm",
}


def deliver_reviewer_response(
    conversation_id: str,
    *,
    note: str,
    agent_name: str,
    actions: list[str] | None = None,
    sentiment: str = "calm",
    force_verbatim: bool = False,
) -> dict[str, Any]:
    """Bring a reviewer's answer back to the customer, in the assistant's voice.

    Falls back to quoting the reviewer word-for-word whenever the rendering
    can't be verified against their note — a safe answer beats a smooth one.
    """
    trace_id = str(uuid.uuid4())
    actions = actions or []
    note = note.strip()

    verbatim = f"I asked {agent_name} from our claims team, and here's their answer:\n\n“{note}”"
    rendered, source = verbatim, "verbatim"

    if not force_verbatim:
        result = gateway.complete(
            "human_relay",
            {
                "agent_name": agent_name,
                "human_note": wrap_untrusted("reviewer_note", note),
                "actions": "\n".join(f"- {a}" for a in actions) or "(none)",
                "tone_profile": TONE_BY_SENTIMENT.get(sentiment, "neutral-warm"),
            },
            tier="primary",
            trace_id=trace_id,
            fallback="",
        )
        candidate = result.text.strip()

        # The reviewer's note is the only permitted source of fact. If the
        # re-voicing introduced anything else, we quote them instead.
        check = output_guards.check_output(candidate, {"reviewer_note": note,
                                                       "actions": actions,
                                                       "agent_name": agent_name})
        if candidate and check.passed:
            rendered = candidate
            source = "template" if result.degraded else result.model
        else:
            audit.record(
                "relay_rendering_rejected", actor_type="agent", actor_id=agent_name,
                entity_type="conversation", entity_id=conversation_id,
                payload={"failures": check.failures, "ungrounded": check.ungrounded},
                trace_id=trace_id,
            )

    # Posted as the assistant: the customer only ever hears one voice. The
    # reviewer is recorded in author_name for attribution, and their original
    # wording alongside it so the relay can be checked for drift.
    message_id = conv_repo.add_message(
        conversation_id, "assistant", rendered, author_name=agent_name,
        source_note=note, relay_source=source,
    )

    audit.record(
        "reviewer_response_delivered", actor_type="agent", actor_id=agent_name,
        entity_type="conversation", entity_id=conversation_id,
        payload={"source": source, "verbatim_fallback": source == "verbatim",
                 "actions": actions, "note_preview": note[:200]},
        trace_id=trace_id,
    )

    return {"message_id": message_id, "content": rendered, "relayed_from": agent_name,
            "source": source, "verbatim": source == "verbatim"}


def request_information(
    conversation_id: str,
    *,
    request: str,
    agent_name: str,
    claim_context: dict[str, Any] | None = None,
    sentiment: str = "calm",
) -> dict[str, Any]:
    """Ask the customer for something the reviewer or the claim needs.

    The reviewer states what's needed in internal shorthand; the assistant turns
    it into a question the customer can actually act on — what, why, how to get
    it, and one clear next step.
    """
    import json

    trace_id = str(uuid.uuid4())
    request = request.strip()

    fallback = (
        f"To move your claim forward we need: {request}\n\n"
        f"Send it here whenever you're ready and I'll check it straight away. "
        f"If you're not sure how to get hold of it, tell me and I'll find out for you."
    )

    result = gateway.complete(
        "information_request",
        {
            "request": wrap_untrusted("reviewer_request", request),
            "agent_name": agent_name,
            "claim_context": json.dumps(claim_context or {}, default=str),
            "tone_profile": TONE_BY_SENTIMENT.get(sentiment, "neutral-warm"),
        },
        tier="primary",
        trace_id=trace_id,
        fallback=fallback,
    )

    rendered = result.text.strip() or fallback

    # The request is the only source of fact; anything else gets the plain version.
    check = output_guards.check_output(rendered, {"request": request,
                                                  "claim": claim_context or {}})
    if not check.passed:
        audit.record(
            "info_request_rendering_rejected", actor_type="agent", actor_id=agent_name,
            entity_type="conversation", entity_id=conversation_id,
            payload={"failures": check.failures, "ungrounded": check.ungrounded},
            trace_id=trace_id,
        )
        rendered = fallback

    message_id = conv_repo.add_message(
        conversation_id, "assistant", rendered, author_name=agent_name,
        source_note=request, relay_source="information_request",
    )
    audit.record("information_requested", actor_type="agent", actor_id=agent_name,
                 entity_type="conversation", entity_id=conversation_id,
                 payload={"request": request[:300]}, trace_id=trace_id)

    return {"message_id": message_id, "content": rendered, "request": request}


def announce_case_taken(conversation_id: str, agent_name: str,
                        carried: list[str]) -> str:
    """Tell the customer their case has been taken forward, and what was passed on.

    Transparency is the point: the customer sees exactly what was said on their
    behalf (§15.5, "AI summary visible to both sides").
    """
    lines = "\n".join(f"• {item}" for item in carried if item)
    body = (
        f"{agent_name} from our claims team has picked this up. Here's what I've "
        f"passed on for you:\n\n{lines}\n\n"
        f"I'll bring their answer straight back to you here — you don't need to "
        f"chase anyone."
    )
    message_id = conv_repo.add_message(conversation_id, "assistant", body)
    audit.record("case_taken_announced", actor_type="agent", actor_id=agent_name,
                 entity_type="conversation", entity_id=conversation_id,
                 payload={"carried": carried})
    return message_id


def join(conversation_id: str, agent_name: str, ticket_id: str | None = None,
         carried: list[str] | None = None) -> None:
    """A reviewer picks the case up. The assistant stays the customer's contact."""
    conv_repo.set_mode(conversation_id, conv_repo.MODE_HUMAN, agent=agent_name,
                       ticket_id=ticket_id)
    announce_case_taken(conversation_id, agent_name, carried or [])


def resolve(conversation_id: str, agent_name: str, ticket_id: str | None,
            closing_note: str = "", sentiment: str = "calm") -> dict[str, Any]:
    """Reviewer closes the case; the assistant carries on as normal."""
    if ticket_id:
        execute(
            "UPDATE escalation_ticket SET status = 'RESOLVED', resolved_at = ? "
            "WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), ticket_id),
        )

    delivered = None
    if closing_note.strip():
        delivered = deliver_reviewer_response(
            conversation_id, note=closing_note, agent_name=agent_name,
            sentiment=sentiment,
        )

    conv_repo.set_mode(conversation_id, conv_repo.MODE_AI, agent=None)
    conv_repo.add_message(
        conversation_id, "assistant",
        "That's this one closed off. I'm still here — ask me anything about your "
        "claim whenever you need to.",
    )
    audit.record("escalation_resolved", actor_type="agent", actor_id=agent_name,
                 entity_type="ticket", entity_id=ticket_id,
                 payload={"conversation_id": conversation_id})
    return {"resolved": True, "closing_message": delivered}


def notify_document_decision(customer_id: str, doc_id: str, verdict: str,
                             agent_name: str, note: str = "") -> str | None:
    """Ping the customer's conversation when a reviewer rules on a document.

    This closes the dispute loop: the customer asked for a human to look, and the
    answer arrives in the same chat rather than nowhere.
    """
    conversation = conv_repo.latest_for_customer(customer_id)
    if conversation is None:
        return None

    if verdict == "VERIFIED":
        body = (f"Good news — I asked {agent_name} in our claims team to look at "
                f"that document personally, and they've accepted it as valid evidence.")
    elif verdict == "NEEDS_REVIEW":
        body = (f"{agent_name} is still going through that document. I'll come "
                f"back to you here as soon as they've decided.")
    else:
        body = (f"I took that document back to {agent_name} in our claims team. "
                f"They've reviewed it and it still can't be accepted.")
    if note.strip():
        body += f" They said: “{note.strip()}”"

    message_id = conv_repo.add_message(conversation["id"], "assistant", body,
                                       author_name=agent_name)
    audit.record("document_decision_relayed", actor_type="agent", actor_id=agent_name,
                 entity_type="document", entity_id=doc_id,
                 payload={"verdict": verdict, "conversation_id": conversation["id"]})
    return message_id
