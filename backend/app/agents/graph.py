"""Conversation graph (§9 topology), built on LangGraph.

    input_guard ─┬─► blocked ─────────────────────────────────────────► END
                 ├─► small_talk ─► finalize ──────────────────────────► END
                 └─► route ─► overrides ─► touch_case ─┬─► greeting / out_of_scope ─► finalize
                                                       ├─► escalation ─┐
                                                       └─► agent:{new_claim, claim_status,
                                                            documents, knowledge} ─┤
                     offer_human ◄────────────────────────────────────────────────┘
                          │
                       compose ─► output_guard ─┬─► emit ─► END
                                    ▲           ├─► regenerate ─┘ (once)
                                    └───────────┴─► template_fallback ─► emit

Each node is instrumented (OTel span + Prometheus histogram) and every
super-step is checkpointed under the thread ``conv:<conversation_id>``, so any
turn can be replayed node by node. Streaming frames go out through LangGraph's
custom stream channel (``get_stream_writer``) rather than a callback threaded
through every function.

The domain agents (``supervisor``, ``claim_status`` …) are unchanged: they take
and mutate a ``GraphState``. ``_node`` adapts them to LangGraph's
dict-in / partial-update-out contract.
"""
from __future__ import annotations

import copy
import re
import uuid
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents import (claim_status, document, empathy, escalation, fnol_agent,
                        knowledge, supervisor)
from app.agents import runtime
from app.agents.state import GraphState
from app.audit import logger as audit
from app.db import execute, query_one
from app.guardrails import input_guards, output_guards
from app.ops import telemetry

GRAPH = "conversation"

AGENTS: dict[str, Callable[[GraphState], GraphState]] = {
    "new_claim": fnol_agent.run,
    "claim_status": claim_status.run,
    "documents": document.run,
    "knowledge": knowledge.run,
}

_STATE_FIELDS = [f.name for f in dc_fields(GraphState)]


class TurnState(TypedDict, total=False):
    # --- GraphState fields (see state.py) ---------------------------------
    customer_id: str
    customer_name: str
    conversation_id: str
    trace_id: str
    message: str
    history: list[dict[str, Any]]
    active_claim_id: str | None
    intent: str | None
    intent_confidence: float
    sentiment: str
    tone_profile: str
    facts: dict[str, Any]
    cards: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    draft: str
    reply: str
    guardrail_flags: list[str]
    blocked: bool
    escalation_ticket_id: str | None
    turn_hop_count: int
    degraded: bool
    regenerated: bool
    # --- graph control ------------------------------------------------------
    handoff: dict[str, Any] | None
    route: str
    block_category: str | None
    output_attempt: int
    output_passed: bool
    output_failures: list[str]


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------
def _to_state(values: dict[str, Any]) -> GraphState:
    # Deep-copied so an agent mutating a list in place can't alter the
    # checkpointed value behind LangGraph's back.
    return GraphState(**{k: copy.deepcopy(values[k]) for k in _STATE_FIELDS if k in values})


def _from_state(state: GraphState) -> dict[str, Any]:
    return {k: getattr(state, k) for k in _STATE_FIELDS}


def _send(frame: dict[str, Any]) -> None:
    try:
        get_stream_writer()(frame)
    except Exception:  # noqa: BLE001 - outside a streaming run there is no writer
        pass


def _node(name: str, fn: Callable[[GraphState, dict[str, Any]], dict[str, Any] | None]):
    """Adapt ``fn(GraphState, raw_values) -> extra updates`` into a graph node."""

    def run(values: dict[str, Any]) -> dict[str, Any]:
        state = _to_state(values)
        extra = fn(state, values) or {}
        return {**_from_state(state), **extra}

    return telemetry.instrument(GRAPH, name, run)


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
def _input_guard(state: GraphState, values: dict[str, Any]) -> dict[str, Any]:
    verdict = input_guards.check_input(state.message)
    if verdict.allowed:
        state.guardrail_flags.extend(verdict.flags)
        return {"route": "small_talk" if supervisor.is_small_talk(state.message) else "route",
                "block_category": None}

    state.blocked = True
    state.guardrail_flags.extend(verdict.flags or [verdict.category or "blocked"])
    state.reply = verdict.safe_response or "I can't help with that, I'm afraid."
    telemetry.GUARDRAIL_EVENTS.labels(f"input_block_{verdict.category or 'other'}").inc()
    audit.record(
        "guardrail_block", actor_type="customer", actor_id=state.customer_id,
        entity_type="conversation", entity_id=state.conversation_id,
        payload={"category": verdict.category, "reason": verdict.reason,
                 "message_preview": state.message[:200]},
        trace_id=state.trace_id,
    )
    return {"route": "blocked", "block_category": verdict.category}


def _blocked(state: GraphState, values: dict[str, Any]) -> None:
    # Distress is a block *and* an escalation (§17.2, UC-N10).
    if values.get("block_category") == "distress":
        state.sentiment = "distressed"  # type: ignore[assignment]
        state.tone_profile = "gentle-supportive"  # type: ignore[assignment]
        escalation.run(state, reason="Distress signals detected in conversation")
        _send({"type": "handoff", **state.cards[-1]["payload"]})
    _send({"type": "token", "content": state.reply})
    _send({"type": "done", "blocked": True, "category": values.get("block_category")})
    telemetry.TURNS.labels("blocked", "blocked").inc()


def _small_talk(state: GraphState, values: dict[str, Any]) -> None:
    # Answered before routing: a classifier with no "greeting" label files "hi"
    # under out_of_scope and refuses it.
    state.intent = "greeting"  # type: ignore[assignment]
    state.reply = _greeting_reply(state, state.customer_id)


def _route(state: GraphState, values: dict[str, Any]) -> None:
    supervisor.route(state)
    _send({"type": "status", "stage": "routing", "intent": state.intent,
           "sentiment": state.sentiment})
    audit.record(
        "agent_routing", actor_type="agent", actor_id="supervisor",
        entity_type="conversation", entity_id=state.conversation_id,
        payload={"intent": state.intent, "confidence": state.intent_confidence,
                 "sentiment": state.sentiment},
        trace_id=state.trace_id,
    )


def _overrides(state: GraphState, values: dict[str, Any]) -> None:
    """Three deterministic overrides the classifier must not undo.

    * a bare "yes" to an offer of a colleague is accepting that offer (checked
      first: an open intake must not swallow it),
    * an unambiguous "I want to make a claim" is new_claim, and
    * an open intake captures the next message — "yesterday" or "£400" is an
      answer to the question just asked — unless it is plainly a question
      about existing claims.
    """
    if _accepted_offer(state):
        state.intent = "human_request"  # type: ignore[assignment]
        state.intent_confidence = 1.0
    elif fnol_agent.wants_to_start(state.message):
        state.intent = "new_claim"  # type: ignore[assignment]
        state.intent_confidence = max(state.intent_confidence, 0.9)
    elif state.intent not in ("human_request", "out_of_scope"):
        from app.fnol import intake as _intake

        if (_intake.open_for_customer(state.customer_id)
                and not _asks_about_existing(state.message)):
            state.intent = "new_claim"  # type: ignore[assignment]


def _touch_case(state: GraphState, values: dict[str, Any]) -> None:
    """Keep an open case current: the reviewer needs the latest words and mood."""
    handoff = values.get("handoff")
    if not (handoff and handoff.get("ticket_id")):
        return
    try:
        ticket = escalation.load_ticket(handoff["ticket_id"])
        if ticket:
            escalation.touch_case(state, ticket, chased=state.intent == "human_request")
    except Exception as exc:  # noqa: BLE001 - never fail a reply over this
        audit.record("case_touch_failed", entity_type="conversation",
                     entity_id=state.conversation_id,
                     payload={"error": str(exc)[:200]}, trace_id=state.trace_id)


def _greeting(state: GraphState, values: dict[str, Any]) -> None:
    state.reply = _greeting_reply(state, state.customer_id)


def _out_of_scope(state: GraphState, values: dict[str, Any]) -> None:
    state.reply = input_guards.SAFE_RESPONSES["out_of_scope"]


def _escalate(state: GraphState, values: dict[str, Any]) -> None:
    escalation.run(state)


def _agent(intent: str):
    def run(state: GraphState, values: dict[str, Any]) -> None:
        if not state.hop(intent):
            state.guardrail_flags.append("hop_budget_exceeded")
            return
        AGENTS.get(intent, knowledge.run)(state)
    return run


def _offer_human(state: GraphState, values: dict[str, Any]) -> None:
    """Frustration is a reason to *ask* whether they want a colleague — once."""
    if (state.sentiment == "frustrated"
            and state.intent not in ("human_request", "out_of_scope", "new_claim")
            and not values.get("handoff")
            and not state.escalation_ticket_id
            and not _already_offered(state)):
        state.cards.append({"card_type": "offer_human",
                            "payload": {"reason": "sounds frustrating"}})
        _mark_offer(state)


def _compose(state: GraphState, values: dict[str, Any]) -> dict[str, Any]:
    handoff = values.get("handoff")
    if handoff:
        state.facts["human_review_in_progress"] = handoff
        if state.intent != "human_request":
            state.cards.append({"card_type": "handoff_status", "payload": handoff})
    _send({"type": "status", "stage": "composing"})
    empathy.run(state)
    return {"output_attempt": 0}


def _grounding(state: GraphState) -> dict[str, Any]:
    return {"facts": state.facts, "citations": state.citations,
            "customer_name": state.customer_name}


def _output_guard(state: GraphState, values: dict[str, Any]) -> dict[str, Any]:
    attempt = int(values.get("output_attempt") or 0) + 1
    # Intake turns ask questions rather than asserting facts; the text is ours.
    if attempt == 1 and state.intent == "new_claim":
        check = output_guards.OutputVerdict()
    else:
        check = output_guards.check_output(state.reply, _grounding(state))

    if attempt == 1:
        if not check.passed:
            state.guardrail_flags.append("output_guard_fail")
            telemetry.GUARDRAIL_EVENTS.labels("output_fail").inc()
            audit.record(
                "guardrail_output_fail", actor_type="agent", actor_id="output_guards",
                entity_type="conversation", entity_id=state.conversation_id,
                payload={"failures": check.failures, "ungrounded": check.ungrounded,
                         "attempt": 1},
                trace_id=state.trace_id,
            )
        state.guardrail_flags.extend(check.warnings)
    return {"output_attempt": attempt, "output_passed": check.passed,
            "output_failures": list(check.failures)}


def _regenerate(state: GraphState, values: dict[str, Any]) -> None:
    state.regenerated = True
    telemetry.GUARDRAIL_EVENTS.labels("regenerate").inc()
    empathy.run(state)


def _template_fallback(state: GraphState, values: dict[str, Any]) -> None:
    state.reply = empathy._template(state)
    state.degraded = True
    state.guardrail_flags.append("template_fallback")
    telemetry.GUARDRAIL_EVENTS.labels("template_fallback").inc()
    audit.record(
        "guardrail_template_fallback", actor_type="agent", actor_id="output_guards",
        entity_type="conversation", entity_id=state.conversation_id,
        payload={"failures": values.get("output_failures", [])}, trace_id=state.trace_id,
    )


def _finalize(state: GraphState, values: dict[str, Any]) -> None:
    """Short answers (greeting, out of scope): reply + done, then audit."""
    _send({"type": "token", "content": state.reply})
    _send({"type": "done", "intent": state.intent})
    _audit_turn(state)
    telemetry.TURNS.labels(state.intent or "none", "direct").inc()


def _emit(state: GraphState, values: dict[str, Any]) -> None:
    _send({"type": "token", "content": state.reply})
    for card in state.cards:
        _send({"type": "card", "card_type": card["card_type"], "payload": card["payload"]})
    if state.citations:
        _send({"type": "citations", "items": state.citations})
    _send({"type": "done", "intent": state.intent, "degraded": state.degraded})
    _audit_turn(state)
    outcome = ("template_fallback" if "template_fallback" in state.guardrail_flags
               else "degraded" if state.degraded else "ok")
    telemetry.TURNS.labels(state.intent or "none", outcome).inc()


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------
def _after_guard(values: dict[str, Any]) -> str:
    return values.get("route") or "route"


def _dispatch(values: dict[str, Any]) -> str:
    intent = values.get("intent") or "knowledge"
    if intent in ("greeting", "out_of_scope"):
        return intent
    if intent == "human_request":
        return "escalation"
    return f"agent_{intent}" if intent in AGENTS else "agent_knowledge"


def _after_output_guard(values: dict[str, Any]) -> str:
    if values.get("output_passed"):
        return "emit"
    return "regenerate" if int(values.get("output_attempt") or 0) < 2 else "template_fallback"


def build_graph(checkpointer: Any = None) -> Any:
    g = StateGraph(TurnState)
    nodes: dict[str, Callable] = {
        "input_guard": _input_guard, "blocked": _blocked, "small_talk": _small_talk,
        "route": _route, "overrides": _overrides, "touch_case": _touch_case,
        "greeting": _greeting, "out_of_scope": _out_of_scope, "escalation": _escalate,
        "offer_human": _offer_human, "compose": _compose,
        "output_guard": _output_guard, "regenerate": _regenerate,
        "template_fallback": _template_fallback, "finalize": _finalize, "emit": _emit,
    }
    nodes.update({f"agent_{intent}": _agent(intent) for intent in AGENTS})
    for name, fn in nodes.items():
        g.add_node(name, _node(name, fn))

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", _after_guard,
                            ["blocked", "small_talk", "route"])
    g.add_edge("blocked", END)
    g.add_edge("small_talk", "finalize")
    g.add_edge("route", "overrides")
    g.add_edge("overrides", "touch_case")
    g.add_conditional_edges("touch_case", _dispatch,
                            ["greeting", "out_of_scope", "escalation",
                             *[f"agent_{i}" for i in AGENTS]])
    g.add_edge("greeting", "finalize")
    g.add_edge("out_of_scope", "finalize")
    g.add_edge("escalation", "offer_human")
    for intent in AGENTS:
        g.add_edge(f"agent_{intent}", "offer_human")
    g.add_edge("offer_human", "compose")
    g.add_edge("compose", "output_guard")
    g.add_conditional_edges("output_guard", _after_output_guard,
                            ["emit", "regenerate", "template_fallback"])
    g.add_edge("regenerate", "output_guard")
    g.add_edge("template_fallback", "emit")
    g.add_edge("finalize", END)
    g.add_edge("emit", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def compiled() -> Any:
    return build_graph(runtime.checkpointer())


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
    """Execute one conversational turn end to end through the graph.

    ``handoff`` describes any live human review: the assistant keeps answering,
    but it knows a colleague is on the case and says so.
    """
    initial = GraphState(
        customer_id=customer_id, customer_name=customer_name,
        conversation_id=conversation_id, trace_id=str(uuid.uuid4()),
        message=message, history=history or [], active_claim_id=active_claim_id,
    )
    # Every key is supplied, so nothing from the thread's previous turn leaks in.
    inputs: dict[str, Any] = {
        **_from_state(initial), "handoff": handoff, "route": "",
        "block_category": None, "output_attempt": 0, "output_passed": False,
        "output_failures": [],
    }
    graph = compiled()
    config = {"configurable": {"thread_id": f"conv:{conversation_id or initial.trace_id}"},
              "run_name": "conversation_turn",
              "metadata": {"trace_id": initial.trace_id, "customer_id": customer_id}}

    final: dict[str, Any] = inputs
    with telemetry.span("conversation.turn", **{"cc.trace_id": initial.trace_id,
                                                "cc.conversation_id": conversation_id}):
        for mode, chunk in graph.stream(inputs, config, stream_mode=["custom", "values"]):
            if mode == "custom":
                if emit:
                    emit(chunk)
            else:
                final = chunk
    telemetry.GRAPH_RUNS.labels(GRAPH, "ok").inc()
    return _to_state(final)


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


# Questions an open intake must not swallow. claim_status was the original set;
# documents and knowledge were added because the assistant's own suggested
# questions live there — "How do I photograph my licence?", "Where do I find the
# claim form?" — and a customer part-way through an intake who tapped one got
# "Sorry, I didn't quite follow. Could you pick one of these?" with no way out
# except abandoning the notification.
_ESCAPES_INTAKE = ("claim_status", "documents", "knowledge")


def _asks_about_existing(message: str) -> bool:
    """Is this a question the intake should step aside for?

    Deliberately narrow: it has to read as a question *and* carry a strong
    signal for one of the intents above. An intake answer is typically a bare
    value — a date, an amount, a place — so it matches neither, and the intake
    keeps capturing as designed.
    """
    lowered = (message or "").strip().lower()

    # A claim reference settles it on its own, whatever the sentence looks like.
    # No answer to "when did it happen" contains "CLM-88429", and the switch
    # button on a document card sends exactly this — "Show me what's outstanding
    # on CLM-88429." — which an open intake was reading as a date.
    from app.agents.tools import claim_tools

    if claim_tools.CLAIM_REF.search(lowered):
        return True

    written_as_question = "?" in lowered
    # "show me", "tell me" and "list" open a request without being questions.
    if not written_as_question and not re.match(
            r"^(how|what|when|which|where|do|does|is|are|show|tell|list|give)\b",
            lowered):
        return False

    # An answer to an intake question is a value — "yesterday", "£400",
    # "Kingsway". Several words ending in a question mark is not one, whatever
    # the router makes of the topic, so it steps aside rather than being fed to
    # the card as an unparseable answer.
    if written_as_question and len(lowered.split()) >= 3:
        return True

    # Without a question mark, the confidence floor does the work. Answer-shaped
    # text that happens to open with a question word — "what happened was I hit
    # a wall" — carries no strong signal and scores well below it, so it stays
    # with the intake as an answer.
    intent, confidence = supervisor._heuristic_intent(lowered)
    return intent in _ESCAPES_INTAKE and confidence >= 0.7


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
