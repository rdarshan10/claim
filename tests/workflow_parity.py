"""Deterministic runs of the document pipeline, the registration bot and the
reviewer relay, used to diff the LangGraph ports against the original code.

Run with LLM_ENABLED=false against a freshly generated dataset:

    python -m tests.workflow_parity out.json
"""
from __future__ import annotations

import json
import sys
import threading
import uuid
from datetime import datetime, timezone

from app.db import execute, query, query_one


def documents() -> list[dict]:
    from app.documents import pipeline

    out = []
    for row in query("SELECT id FROM document ORDER BY id"):
        result = pipeline.process(row["id"])
        payload = result.get("payload") or {}
        doc = dict(query_one(
            "SELECT status, doc_type, classification_conf, extraction_conf, "
            "extracted_fields, rejection_code, ocr_quality FROM document WHERE id = ?",
            (row["id"],)))
        rules = [dict(r) for r in query(
            "SELECT rule_id, passed FROM document_validation WHERE document_id = ? "
            "ORDER BY rule_id, passed", (row["id"],))]
        out.append({
            "doc_id": row["id"], "status": result.get("status"),
            "confidence": result.get("confidence"),
            "reason_code": payload.get("reason_code"),
            "recommendation": payload.get("recommendation"),
            "failed_rules": payload.get("failed_rules"),
            "headline": payload.get("headline"),
            "annotations": len(payload.get("annotations") or []),
            "row": doc, "rules": rules,
        })
    out.append({"claims": [dict(r) for r in query(
        "SELECT claim_number, status FROM claim ORDER BY claim_number")]})
    return out


def _wait_for_threads() -> None:
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon \
                and thread.name.startswith("Thread-"):
            thread.join(30)


def registrations() -> list[dict]:
    from app.fnol import intake, registration

    out = []
    customers = query("SELECT id FROM customer ORDER BY id LIMIT 3")
    for i, customer in enumerate(customers):
        record = intake.create(customer["id"])
        claim_type = ("motor", "home", "health")[i % 3]
        execute("UPDATE fnol_request SET claim_type = ? WHERE id = ?",
                (claim_type, record["id"]))
        for key, value in {"incident_date": "2026-11-20",
                           "description": "Reversed into a bollard in a car park.",
                           "location": "Leeds", "estimated_amount": "1200"}.items():
            intake.save_answer(record["id"], key, value)
        intake.set_status(record["id"], "READY_TO_REGISTER")
        run_id = registration.start(record["id"], "parity")
        _wait_for_threads()
        run = dict(query_one("SELECT status, error, result FROM rpa_run WHERE id = ?",
                             (run_id,)))
        result = json.loads(run["result"] or "{}")
        fnol = intake.get(record["id"])
        claim = query_one("SELECT claim_number, status, claim_type FROM claim WHERE id = ?",
                          (fnol.get("claim_id"),)) if fnol.get("claim_id") else None
        out.append({
            "claim_type": claim_type, "run_status": run["status"], "error": run["error"],
            "result_keys": sorted(result), "mode": result.get("mode"),
            "claim_number": result.get("claim_number"),
            "fnol_status": fnol["status"], "claim": dict(claim) if claim else None,
            "audit": [r["event_type"] for r in query(
                "SELECT event_type FROM audit_event WHERE entity_id = ? ORDER BY id",
                (record["id"],))],
        })
    # A bot failure: the run is FAILED and the notification goes back to the queue.
    record = intake.create(customers[1]["id"])
    execute("UPDATE fnol_request SET claim_type = 'motor' WHERE id = ?", (record["id"],))
    intake.set_status(record["id"], "READY_TO_REGISTER")
    original = registration._simulate

    def broken(recorder, values):
        raise RuntimeError("core system down")

    registration._simulate = broken
    try:
        run_id = registration.start(record["id"], "parity")
        _wait_for_threads()
    finally:
        registration._simulate = original
    run = query_one("SELECT status, error FROM rpa_run WHERE id = ?", (run_id,))
    steps = json.loads(query_one("SELECT steps FROM rpa_run WHERE id = ?",
                                 (run_id,))["steps"])
    out.append({"failed_run": dict(run), "fnol_status": intake.get(record["id"])["status"],
                "last_step": {k: steps[-1].get(k) for k in ("title", "detail", "state")},
                "audit": [r["event_type"] for r in query(
                    "SELECT event_type FROM audit_event WHERE entity_id = ? ORDER BY id",
                    (record["id"],))]})

    # A run against a notification that was never approved must be refused.
    record = intake.create(customers[0]["id"])
    try:
        registration.start(record["id"], "parity")
        out.append({"refused": False})
    except ValueError as exc:
        out.append({"refused": str(exc).split(" is ")[1]})
    return out


def relays() -> list[dict]:
    from app.agents import handoff

    out = []
    for verbatim in (False, True):
        conv = str(uuid.uuid4())
        customer = query_one("SELECT id FROM customer ORDER BY id LIMIT 1")["id"]
        execute("INSERT INTO conversation (id, customer_id, started_at) VALUES (?,?,?)",
                (conv, customer, datetime.now(timezone.utc).isoformat()))
        result = handoff.deliver_reviewer_response(
            conv, note="Your invoice is fine, we just need the police report.",
            agent_name="Marcus", actions=["Upload the police report"],
            sentiment="frustrated", force_verbatim=verbatim)
        stored = query_one("SELECT content, author_name, source_note, relay_source "
                           "FROM message WHERE id = ?", (result["message_id"],))
        out.append({k: v for k, v in result.items() if k != "message_id"}
                   | {"stored": dict(stored)})

    conv = str(uuid.uuid4())
    customer = query_one("SELECT id FROM customer ORDER BY id LIMIT 1")["id"]
    execute("INSERT INTO conversation (id, customer_id, started_at) VALUES (?,?,?)",
            (conv, customer, datetime.now(timezone.utc).isoformat()))
    result = handoff.request_information(
        conv, request="need police report + corrected invoice from the garage",
        agent_name="Marcus", claim_context={"claim_number": "CLM-1"}, sentiment="calm")
    stored = query_one("SELECT content, relay_source, source_note FROM message WHERE id = ?",
                       (result["message_id"],))
    out.append({k: v for k, v in result.items() if k != "message_id"}
               | {"stored": dict(stored),
                  "audit": [r["event_type"] for r in query(
                      "SELECT event_type FROM audit_event WHERE entity_id = ? ORDER BY id",
                      (conv,))]})
    return out


if __name__ == "__main__":
    data = {"documents": documents(), "registrations": registrations(), "relays": relays()}
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    print({k: len(v) for k, v in data.items()})
