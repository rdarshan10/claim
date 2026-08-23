"""Conversation graph (§9 topology).

InputGuardrails -> Sentiment/Router -> {StatusAgent | DocumentAgent |
KnowledgeAgent | Escalation} -> EmpathyResponder -> OutputGuardrails
                                                     -> Regenerate (once)
                                                     -> TemplateFallback

Implemented as an explicit state machine over ``GraphState`` rather than
LangGraph (not installable in this environment). Node signatures and the state
object match LangGraph's shape, so migration is wiring, not redesign — see
TO_BE_DONE.md.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from app.agents import (claim_status, document, empathy, escalation, fnol_agent,
                        knowledge, supervisor)
from app.agents.state import GraphState
from app.audit import logger as audit
from app.db import execute, query_one
from app.guardrails import input_guards, output_guards

AGENTS: dict[str, Callable[[GraphState], GraphState]] = {
    "new_claim": fnol_agent.run,
    "claim_status": claim_status.run,
    "documents": document.run,
    "knowledge": knowledge.run,
}


def run_turn(
    *,
    customer_id: str,
    customer_name: str,
    message: str,
    history: list[dict[str, Any]] | None = None,
    conversation_id: str = "",
    active_claim_id: str | None = None,
    handoff: dict[str, Any] | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> GraphState:
    """Execute one conversational turn end to end.

    ``handoff`` describes any live human review. The assistant keeps answering
    throughout — it is always the customer's contact — but it knows a colleague
    is on the case and says so rather than pretending otherwise.
    """
    state = GraphState(
        customer_id=customer_id,
        customer_name=customer_name,
        conversation_id=conversation_id,
        trace_id=str(uuid.uuid4()),
        message=message,
        history=history or [],
        active_claim_id=active_claim_id,
    )

    def send(frame: dict[str, Any]) -> None:
        if emit:
            emit(frame)

    # --- node: input guardrails ----------------------------------------
    verdict = input_guards.check_input(message)
    if not verdict.allowed:
        state.blocked = True
        state.guardrail_flags.extend(verdict.flags or [verdict.category or "blocked"])
        state.reply = verdict.safe_response or "I can't help with that, I'm afraid."

        audit.record(
            "guardrail_block", actor_type="customer", actor_id=customer_id,
            entity_type="conversation", entity_id=conversation_id,
            payload={"category": verdict.category, "reason": verdict.reason,
                     "message_preview": message[:200]},
            trace_id=state.trace_id,
        )

        # Distress is a block *and* an escalation (§17.2, UC-N10).
        if verdict.category == "distress":
            state.sentiment = "distressed"  # type: ignore[assignment]
            state.tone_profile = "gentle-supportive"  # type: ignore[assignment]
            escalation.run(state, reason="Distress signals detected in conversation")
            send({"type": "handoff", **state.cards[-1]["payload"]})

        send({"type": "token", "content": state.reply})
        send({"type": "done", "blocked": True, "category": verdict.category})
        return state

    state.guardrail_flags.extend(verdict.flags)

    # --- node: small talk -------------------------------------------------
    # Answered before routing: a classifier with no "greeting" label files "hi"
    # under out_of_scope and refuses it. Costs nothing and can't be misrouted.
    if supervisor.is_small_talk(message):
        state.intent = "greeting"  # type: ignore[assignment]
        state.reply = _greeting_reply(state, customer_id)
        send({"type": "token", "content": state.reply})
        send({"type": "done", "intent": "greeting"})
        _audit_turn(state)
        return state

    # --- node: router + sentiment ---------------------------------------
    supervisor.route(state)
    send({"type": "status", "stage": "routing", "intent": state.intent,
          "sentiment": state.sentiment})

    audit.record(
        "agent_routing", actor_type="agent", actor_id="supervisor",
        entity_type="conversation", entity_id=conversation_id,
        payload={"intent": state.intent, "confidence": state.intent_confidence,
                 "sentiment": state.sentiment},
        trace_id=state.trace_id,
    )

    # --- node: notification of loss -------------------------------------
    # Three deterministic overrides the classifier must not undo:
    #   * a bare "yes" to an offer of a colleague is accepting that offer,
    #   * an unambiguous "I want to make a claim" is new_claim regardless of
    #     what the model said, and
    #   * once an intake is open, the customer's next message is almost always
    #     the answer to the question just asked. "Yesterday" or "£400" carries
    #     no intent signal at all and would otherwise route to knowledge.
    # "Yes" after we offered a colleague is an answer to that question, not a
    # new topic. Checked first: an open intake must not swallow it, or the
    # customer accepts the offer and gets asked for their incident date.
    if _accepted_offer(state):
        state.intent = "human_request"  # type: ignore[assignment]
        state.intent_confidence = 1.0
    elif fnol_agent.wants_to_start(message):
        state.intent = "new_claim"  # type: ignore[assignment]
        state.intent_confidence = max(state.intent_confidence, 0.9)
    elif state.intent not in ("human_request", "out_of_scope"):
        from app.fnol import intake as _intake

        if _intake.open_for_customer(customer_id):
            state.intent = "new_claim"  # type: ignore[assignment]

    # --- node: keep an open case current --------------------------------
    # Whatever they asked about, if a colleague is working their case the
    # reviewer needs the customer's latest words and mood. "Why is this taking
    # so long" routes to claim_status, but it is exactly what they should see.
    if handoff and handoff.get("ticket_id"):
        try:
            ticket = escalation.load_ticket(handoff["ticket_id"])
            if ticket:
                escalation.touch_case(state, ticket,
                                      chased=state.intent == "human_request")
        except Exception as exc:  # noqa: BLE001 - never fail a reply over this
            audit.record("case_touch_failed", entity_type="conversation",
                         entity_id=conversation_id,
                         payload={"error": str(exc)[:200]}, trace_id=state.trace_id)

    # --- node: specialised agent ----------------------------------------
    if state.intent == "greeting":
        state.reply = _greeting_reply(state, customer_id)
        send({"type": "token", "content": state.reply})
        send({"type": "done", "intent": "greeting"})
        _audit_turn(state)
        return state
    if state.intent == "human_request":
        escalation.run(state)
    elif state.intent == "out_of_scope":
        state.reply = input_guards.SAFE_RESPONSES["out_of_scope"]
        send({"type": "token", "content": state.reply})
        send({"type": "done", "intent": "out_of_scope"})
        _audit_turn(state)
        return state
    else:
        agent = AGENTS.get(state.intent or "knowledge", knowledge.run)
        if not state.hop(state.intent or "knowledge"):
            state.guardrail_flags.append("hop_budget_exceeded")
        else:
            agent(state)

    # --- node: offer a person (never decide for them) --------------------
    # Frustration is a reason to *ask* whether they want a colleague, not to
    # open a case on their behalf. Only when nothing is already open, and only
    # when they haven't just asked for one — that path escalates for real.
    #
    # Offered at most once per conversation, and only on frustration. Repeating
    # it every turn undercut the assistant: a customer who is merely confused
    # wants an answer, and one who is frustrated has already been asked and
    # said no. Anyone who wants a person can still say so at any point, which
    # routes to human_request and escalates directly.
    if (state.sentiment == "frustrated"
            and state.intent not in ("human_request", "out_of_scope", "new_claim")
            and not handoff
            and not state.escalation_ticket_id
            and not _already_offered(state)):
        state.cards.append({
            "card_type": "offer_human",
            "payload": {
                "reason": "sounds frustrating" if state.sentiment == "frustrated"
                          else "not straightforward",
            },
        })
        # Remember we asked, so their answer means something next turn.
        _mark_offer(state)

    # --- node: empathy responder ----------------------------------------
    # If a colleague is already on the case, that is a fact the customer is
    # entitled to, so it joins the verified set rather than being hidden.
    if handoff:
        state.facts["human_review_in_progress"] = handoff
        if state.intent != "human_request":
            state.cards.append({"card_type": "handoff_status", "payload": handoff})

    send({"type": "status", "stage": "composing"})
    empathy.run(state)

    # --- node: output guardrails ----------------------------------------
    grounding_source = {"facts": state.facts, "citations": state.citations,
                        "customer_name": state.customer_name}
    # Intake turns ask questions rather than asserting anything about a claim,
    # so there are no facts to ground them against. The text is ours, not the
    # model's, which is what the guardrail exists to police.
    check = (output_guards.OutputVerdict()
             if state.intent == "new_claim"
             else output_guards.check_output(state.reply, grounding_source))

    if not check.passed:
        state.guardrail_flags.append("output_guard_fail")
        audit.record(
            "guardrail_output_fail", actor_type="agent", actor_id="output_guards",
            entity_type="conversation", entity_id=conversation_id,
            payload={"failures": check.failures, "ungrounded": check.ungrounded,
                     "attempt": 1},
            trace_id=state.trace_id,
        )

        # Regenerate once, then fall back to the deterministic template.
        state.regenerated = True
        empathy.run(state)
        recheck = output_guards.check_output(state.reply, grounding_source)
        if not recheck.passed:
            state.reply = empathy._template(state)
            state.degraded = True
            state.guardrail_flags.append("template_fallback")
            audit.record(
                "guardrail_template_fallback", actor_type="agent",
                actor_id="output_guards", entity_type="conversation",
                entity_id=conversation_id,
                payload={"failures": recheck.failures}, trace_id=state.trace_id,
            )

    for warning in check.warnings:
        state.guardrail_flags.append(warning)

    # --- emit ------------------------------------------------------------
    send({"type": "token", "content": state.reply})
    for card in state.cards:
        send({"type": "card", "card_type": card["card_type"], "payload": card["payload"]})
    if state.citations:
        send({"type": "citations", "items": state.citations})
    send({"type": "done", "intent": state.intent, "degraded": state.degraded})

    _audit_turn(state)
    return state


def _greeting_reply(state: GraphState, customer_id: str) -> str:
    """Greet, then say what's actually useful — grounded in their real claims."""
    first_name = state.customer_name.split(" ")[0] if state.customer_name else ""
    hello = f"Hello {first_name}" if first_name else "Hello"

    lowered = state.message.lower().strip()
    if re.match(r"^\s*(?:thanks|thank you|ta|cheers)", lowered):
        return ("You're very welcome. Anything else you'd like me to check on your "
                "claim?")
    if re.match(r"^\s*(?:bye|goodbye|see you|later)", lowered):
        return ("Take care. I'll keep an eye on your claim and let you know the "
                "moment anything changes.")

    try:
        from app.agents.tools import claim_tools as tools

        claims = tools.get_claims(customer_id)
    except Exception:  # noqa: BLE001 - a greeting must never fail
        claims = []

    if not claims:
        return (f"{hello} — I'm ClaimCompanion. I can answer questions about your "
                f"claims, check documents you upload, explain anything that's "
                f"unclear, or put you through to a colleague. What can I help with?")

    open_claims = [c for c in claims
                   if c["status"] not in ("SETTLED", "REJECTED", "WITHDRAWN")]
    if open_claims:
        claim = open_claims[0]
        state.active_claim_id = claim["id"]
        state.facts = {"claim": {"claim_number": claim["claim_number"],
                                 "status": claim["status"]}}
        return (f"{hello} — good to hear from you. I can see your claim "
                f"{claim['claim_number']} is open. I can tell you where it's up to, "
                f"what documents we still need, explain anything that's unclear, or "
                f"put you through to a colleague. What would be most useful?")

    return (f"{hello} — I'm ClaimCompanion. Your claims are all closed at the "
            f"moment, but I can talk you through any of them, or help if something "
            f"new has happened. What can I do for you?")


def _audit_turn(state: GraphState) -> None:
    audit.record(
        "conversation_turn", actor_type="customer", actor_id=state.customer_id,
        entity_type="conversation", entity_id=state.conversation_id,
        payload=state.to_audit(), trace_id=state.trace_id,
    )


# Short, unambiguous acceptances. Anything longer is treated as a real message:
# "yes, and my policy number is 123" carries content that must still be routed.
_AFFIRMATIVES = {
    "yes", "y", "yeah", "yep", "yes please", "please", "please do", "ok", "okay",
    "sure", "go ahead", "do that", "we can do that", "that would be fine",
    "that would be good", "that works", "fine", "alright", "sounds good",
    "if you could", "if you would", "i would", "id like that", "i'd like that",
}


def _already_offered(state: GraphState) -> bool:
    """Have we offered a colleague at any point in this conversation?

    Separate from ``offered_human_at``, which is deliberately short-lived: it
    is consumed on the next turn so a stray "yes" much later never re-triggers
    an escalation. This one is never cleared, so the offer is made once and the
    assistant then gets on with helping.
    """
    if not state.conversation_id:
        return False
    row = query_one("SELECT human_offered_ever_at FROM conversation WHERE id = ?",
                    (state.conversation_id,))
    return bool(row and row["human_offered_ever_at"])


def _mark_offer(state: GraphState) -> None:
    """Record that we offered to fetch a colleague on this turn."""
    if not state.conversation_id:
        return
    now = datetime.now(timezone.utc).isoformat()
    # offered_human_at drives "yes" on the very next turn; human_offered_ever_at
    # stops the offer being made a second time.
    execute("UPDATE conversation SET offered_human_at = ?, "
            "human_offered_ever_at = COALESCE(human_offered_ever_at, ?) WHERE id = ?",
            (now, now, state.conversation_id))


def _accepted_offer(state: GraphState) -> bool:
    """True when this turn is a bare 'yes' to an offer made on the last one.

    The offer is cleared whether or not they accepted: it stands for exactly one
    turn, so a 'yes' to something else later never re-triggers it.
    """
    if not state.conversation_id:
        return False
    row = query_one("SELECT offered_human_at FROM conversation WHERE id = ?",
                    (state.conversation_id,))
    if not row or not row["offered_human_at"]:
        return False

    execute("UPDATE conversation SET offered_human_at = NULL WHERE id = ?",
            (state.conversation_id,))

    cleaned = state.message.strip().lower().rstrip(".!").strip()
    return cleaned in _AFFIRMATIVES
