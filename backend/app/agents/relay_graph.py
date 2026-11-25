"""The reviewer → customer relay as a LangGraph graph.

    render ─► ground ─► deliver ─► END

Two flows share it:

* ``relay``  — a reviewer's answer re-voiced in the assistant's voice
  (``handoff.deliver_reviewer_response``).
* ``request`` — reviewer shorthand turned into a question the customer can act
  on (``handoff.request_information``).

The load-bearing rule lives in ``ground``: the reviewer's words are the only
permitted source of fact. If the model's rendering adds anything the source
doesn't support, the safe text (the reviewer verbatim, or a plain template) is
sent instead — a safe answer beats a smooth one.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.audit import logger as audit
from app.guardrails import output_guards
from app.llm import gateway
from app.ops import telemetry
from app.repositories import conversations as conv_repo

GRAPH = "relay"


class RelayState(TypedDict, total=False):
    kind: str                       # "relay" | "request"
    conversation_id: str
    agent_name: str
    trace_id: str
    prompt_key: str
    variables: dict[str, Any]
    model_fallback: str             # what the gateway returns with no model
    safe_text: str                  # what is sent when rendering can't be trusted
    safe_source: str
    grounding_source: dict[str, Any]
    source_note: str
    skip_model: bool                # reviewer ticked "send my exact words"
    actions: list[str]
    candidate: str | None
    model: str
    degraded: bool
    rendered: str
    source: str
    message_id: str


def _render(state: RelayState) -> dict[str, Any]:
    if state.get("skip_model"):
        return {"candidate": None}
    result = gateway.complete(state["prompt_key"], state["variables"], tier="primary",
                              trace_id=state["trace_id"], fallback=state["model_fallback"])
    return {"candidate": result.text.strip(), "model": result.model,
            "degraded": result.degraded}


def _ground(state: RelayState) -> dict[str, Any]:
    candidate = state.get("candidate")
    if candidate is None:
        return {"rendered": state["safe_text"], "source": state["safe_source"]}

    if state["kind"] == "request":
        candidate = candidate or state["safe_text"]
    check = output_guards.check_output(candidate, state["grounding_source"])
    if candidate and check.passed:
        if state["kind"] == "request":
            source = "information_request"
        else:
            source = "template" if state.get("degraded") else state.get("model", "")
        return {"rendered": candidate, "source": source}

    event = ("relay_rendering_rejected" if state["kind"] == "relay"
             else "info_request_rendering_rejected")
    telemetry.GUARDRAIL_EVENTS.labels(f"{state['kind']}_rendering_rejected").inc()
    audit.record(event, actor_type="agent", actor_id=state["agent_name"],
                 entity_type="conversation", entity_id=state["conversation_id"],
                 payload={"failures": check.failures, "ungrounded": check.ungrounded},
                 trace_id=state["trace_id"])
    return {"rendered": state["safe_text"], "source": state["safe_source"]}


def _deliver(state: RelayState) -> dict[str, Any]:
    # Posted as the assistant: the customer only ever hears one voice. The
    # reviewer is recorded in author_name, and their original words alongside,
    # so any relay can be checked for drift.
    message_id = conv_repo.add_message(
        state["conversation_id"], "assistant", state["rendered"],
        author_name=state["agent_name"], source_note=state["source_note"],
        relay_source=state["source"])
    if state["kind"] == "relay":
        audit.record("reviewer_response_delivered", actor_type="agent",
                     actor_id=state["agent_name"], entity_type="conversation",
                     entity_id=state["conversation_id"],
                     payload={"source": state["source"],
                              "verbatim_fallback": state["source"] == "verbatim",
                              "actions": state.get("actions", []),
                              "note_preview": state["source_note"][:200]},
                     trace_id=state["trace_id"])
    else:
        audit.record("information_requested", actor_type="agent",
                     actor_id=state["agent_name"], entity_type="conversation",
                     entity_id=state["conversation_id"],
                     payload={"request": state["source_note"][:300]},
                     trace_id=state["trace_id"])
    return {"message_id": message_id}


def build_graph() -> Any:
    g = StateGraph(RelayState)
    for name, fn in [("render", _render), ("ground", _ground), ("deliver", _deliver)]:
        g.add_node(name, telemetry.instrument(GRAPH, name, fn))
    g.add_edge(START, "render")
    g.add_edge("render", "ground")
    g.add_edge("ground", "deliver")
    g.add_edge("deliver", END)
    return g.compile()


@lru_cache(maxsize=1)
def compiled() -> Any:
    return build_graph()


def run(**inputs: Any) -> dict[str, Any]:
    with telemetry.span(f"relay.{inputs['kind']}", **{"cc.trace_id": inputs["trace_id"]}):
        final = compiled().invoke(inputs, {"run_name": f"relay_{inputs['kind']}"})
    telemetry.GRAPH_RUNS.labels(GRAPH, inputs["kind"]).inc()
    return final
