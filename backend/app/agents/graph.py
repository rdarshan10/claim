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
from typing import Any, Callable

from app.agents import claim_status, document, empathy, escalation, knowledge, supervisor
from app.agents.state import GraphState
from app.audit import logger as audit
from app.guardrails import input_guards, output_guards

AGENTS: dict[str, Callable[[GraphState], GraphState]] = {
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

    # --- node: keep an open case current --------------------------------
    # Whatever they asked about, if a colleague is working their case the
    # reviewer needs the customer's latest words and mood. "Why is this taking
    # so long" routes to claim_status, but it is exactly what they should see.
    if handoff and handoff.get("ticket_id"):
        try:
            ticket = escalation.load_ticket(handoff["ticket_id"])
            if ticket:
                ticket = escalation.escalate_priority_if_needed(state, ticket)
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
    check = output_guards.check_output(state.reply, grounding_source)

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
