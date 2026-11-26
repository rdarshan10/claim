"""A registration bot run as a LangGraph graph.

    drive ─► materialise ─► record_success ─► notify ─► END
      │           │               │              │
      └───────────┴───────────────┴──────────────┴──► record_failure ─► END

``drive`` puts the claim on the core system (scripted selectors, or the agentic
bot that reads the form itself); ``materialise`` turns the core system's claim
number into a real claim here; the customer is told in their own thread. Any
step that raises sends the run to ``record_failure``, which returns the
notification to the reviewer's queue rather than leaving it stuck in
REGISTERING.

The scripted and agentic bots differ only in how they drive and in what they
write to the run log, so both compile this one graph with a ``Flavor``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.audit import logger as audit
from app.db import execute
from app.fnol import intake
from app.ops import telemetry

GRAPH = "registration"


@dataclass(frozen=True)
class Flavor:
    name: str
    drive: Callable[[Any, str], dict[str, Any]]
    success_event: str
    failure_event: str
    failure_label: str
    failure_detail_chars: int
    # Scripted runs keep the error in its own column; agentic runs keep it in
    # the result blob. Both shapes are read by the staff console.
    error_in_result: bool
    claim_created_step: bool
    success_payload: Callable[[str, dict[str, Any]], dict[str, Any]]


class RunState(TypedDict, total=False):
    run_id: str
    fnol_id: str
    recorder: Any
    result: dict[str, Any]
    claim_id: str
    error: str | None
    status: str


def _guard(fn: Callable[[RunState], dict[str, Any]]) -> Callable[[RunState], dict[str, Any]]:
    """A bot failure must never crash the app: capture it and route to failure."""

    def run(state: RunState) -> dict[str, Any]:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc) or type(exc).__name__}

    run.__name__ = fn.__name__
    return run


def build_graph(flavor: Flavor) -> Any:
    from app.fnol import registration

    def drive(state: RunState) -> dict[str, Any]:
        return {"result": flavor.drive(state["recorder"], state["fnol_id"])}

    def materialise(state: RunState) -> dict[str, Any]:
        result = dict(state["result"])
        claim_id = registration._materialise(state["fnol_id"], result)
        result["claim_id"] = claim_id
        return {"result": result, "claim_id": claim_id}

    def record_success(state: RunState) -> dict[str, Any]:
        result = state["result"]
        if flavor.claim_created_step:
            state["recorder"].step("Claim created in ClaimCompanion", result["claim_number"])
        execute("UPDATE rpa_run SET status = 'SUCCEEDED', result = ?, finished_at = ? "
                "WHERE id = ?", (json.dumps(result), registration._now(), state["run_id"]))
        # The notification is finished and carries its claim. Leaving the status
        # behind was what let one notification be registered twice.
        intake.set_status(state["fnol_id"], "REGISTERED", claim_id=state["claim_id"])
        audit.record(flavor.success_event, actor_type="agent", actor_id="registration_bot",
                     entity_type="fnol", entity_id=state["fnol_id"],
                     payload=flavor.success_payload(state["run_id"], result))
        return {"status": "SUCCEEDED"}

    def notify(state: RunState) -> dict[str, Any]:
        registration._notify_customer(state["fnol_id"], state["result"])
        return {}

    def record_failure(state: RunState) -> dict[str, Any]:
        error = state.get("error") or "unknown error"
        state["recorder"].step(flavor.failure_label,
                               error[:flavor.failure_detail_chars], state="error")
        if flavor.error_in_result:
            execute("UPDATE rpa_run SET status = 'FAILED', result = ?, finished_at = ? "
                    "WHERE id = ?", (json.dumps({"error": error[:500]}),
                                     registration._now(), state["run_id"]))
        else:
            execute("UPDATE rpa_run SET status = 'FAILED', error = ?, finished_at = ? "
                    "WHERE id = ?", (error[:500], registration._now(), state["run_id"]))
        intake.set_status(state["fnol_id"], "READY_TO_REGISTER")
        audit.record(flavor.failure_event, actor_type="agent", actor_id="registration_bot",
                     entity_type="fnol", entity_id=state["fnol_id"],
                     payload={"run_id": state["run_id"], "error": error[:300]})
        return {"status": "FAILED"}

    def ok_or_fail(next_node: str):
        return lambda state: "record_failure" if state.get("error") else next_node

    g = StateGraph(RunState)
    for name, fn in [("drive", drive), ("materialise", materialise),
                     ("record_success", record_success), ("notify", notify)]:
        g.add_node(name, telemetry.instrument(GRAPH, f"{flavor.name}.{name}", _guard(fn)))
    g.add_node("record_failure",
               telemetry.instrument(GRAPH, f"{flavor.name}.record_failure", record_failure))
    g.add_edge(START, "drive")
    g.add_conditional_edges("drive", ok_or_fail("materialise"), ["materialise", "record_failure"])
    g.add_conditional_edges("materialise", ok_or_fail("record_success"),
                            ["record_success", "record_failure"])
    g.add_conditional_edges("record_success", ok_or_fail("notify"),
                            ["notify", "record_failure"])
    g.add_conditional_edges("notify", ok_or_fail(END), [END, "record_failure"])
    g.add_edge("record_failure", END)
    return g.compile()


def run(graph: Any, flavor: Flavor, run_id: str, fnol_id: str, recorder: Any) -> str:
    """Execute one run to completion; returns SUCCEEDED or FAILED."""
    with telemetry.span(f"registration.{flavor.name}", **{"cc.trace_id": run_id,
                                                          "cc.fnol_id": fnol_id}):
        final = graph.invoke({"run_id": run_id, "fnol_id": fnol_id, "recorder": recorder,
                              "error": None, "status": "RUNNING"},
                             {"run_name": f"registration_{flavor.name}",
                              "metadata": {"run_id": run_id, "fnol_id": fnol_id}})
    telemetry.GRAPH_RUNS.labels(GRAPH, final.get("status", "unknown")).inc()
    return final.get("status", "FAILED")
