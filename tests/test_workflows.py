"""The document pipeline, registration bot and reviewer relay as LangGraph graphs."""
from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.db import execute, query, query_one


def _wait() -> None:
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon \
                and thread.name.startswith("Thread-"):
            thread.join(30)


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------
def test_document_graph_topology():
    from app.documents import pipeline

    nodes = set(pipeline.compiled().get_graph().nodes)
    assert {"load", "scan", "ocr", "classify", "extract", "validate", "gate",
            "verdict", "finish"} <= nodes


def test_every_seeded_document_reaches_a_verdict():
    from app.documents import pipeline

    outcomes = Counter()
    for row in query("SELECT id FROM document"):
        result = pipeline.process(row["id"])
        assert result["status"] in ("NEEDS_REVIEW", "REJECTED_QUALITY", "REJECTED_TYPE",
                                    "REJECTED_CORRUPT")
        outcomes[(result["payload"] or {}).get("recommendation")] += 1
    # Nothing is auto-accepted: clean documents are *recommended* to a handler.
    assert outcomes["ACCEPT"] > 0 and outcomes["REJECT"] > 0
    assert query_one("SELECT COUNT(*) FROM document WHERE status = 'VERIFIED'")[0] == 0


def test_unreadable_file_short_circuits_to_finish(tmp_path: Path):
    from app.documents import pipeline

    doc_id = query_one("SELECT id FROM document LIMIT 1")["id"]
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    execute("UPDATE document SET storage_key = ? WHERE id = ?", (str(empty), doc_id))
    count = "SELECT COUNT(*) FROM document_validation WHERE document_id = ?"
    before = query_one(count, (doc_id,))[0]
    result = pipeline.process(doc_id)
    assert result["status"] == "REJECTED_CORRUPT"
    assert result["payload"]["failed_rules"] == ["SCAN"]
    # Stopped at scan: no later stage ran, so no rules were recorded.
    assert query_one(count, (doc_id,))[0] == before
    assert query_one("SELECT status FROM document WHERE id = ?",
                     (doc_id,))["status"] == "REJECTED_CORRUPT"


def test_missing_document_returns_error():
    from app.documents import pipeline

    assert pipeline.process("no-such-doc") == {"status": "ERROR",
                                               "error": "document not found"}


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def _ready_fnol(customer_id: str, claim_type: str = "motor") -> str:
    from app.fnol import intake

    record = intake.create(customer_id)
    execute("UPDATE fnol_request SET claim_type = ? WHERE id = ?", (claim_type, record["id"]))
    intake.save_answer(record["id"], "incident_date", "2026-11-20")
    intake.set_status(record["id"], "READY_TO_REGISTER")
    return record["id"]


def test_registration_graph_registers_a_claim(customer):
    from app.fnol import intake, registration

    fnol_id = _ready_fnol(customer["id"])
    run_id = registration.start(fnol_id, "test")
    _wait()
    run = registration.get_run(run_id)
    assert run["status"] == "SUCCEEDED"
    fnol = intake.get(fnol_id)
    assert fnol["status"] == "REGISTERED" and fnol["claim_id"]
    assert query_one("SELECT status FROM claim WHERE id = ?",
                     (fnol["claim_id"],))["status"] == "DOCS_PENDING"
    events = [r[0] for r in query("SELECT event_type FROM audit_event WHERE entity_id = ? "
                                  "ORDER BY id", (fnol_id,))]
    assert events == ["rpa_run_started", "rpa_run_succeeded", "customer_notified"]


def test_registration_failure_returns_fnol_to_queue(customer, monkeypatch):
    from app.fnol import intake, registration

    def broken(recorder, values):
        raise RuntimeError("core system down")

    monkeypatch.setattr(registration, "_simulate", broken)
    fnol_id = _ready_fnol(customer["id"])
    run_id = registration.start(fnol_id, "test")
    _wait()
    run = registration.get_run(run_id)
    assert run["status"] == "FAILED" and run["error"] == "core system down"
    assert run["steps"][-1]["state"] == "error"
    assert intake.get(fnol_id)["status"] == "READY_TO_REGISTER"


def test_agentic_flavor_records_failure_in_result(customer, monkeypatch):
    from app.fnol import agentic_registration, intake, registration

    def no_browser(recorder, fnol_id):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(agentic_registration, "_drive", no_browser)
    fnol_id = _ready_fnol(customer["id"], "home")
    run_id = agentic_registration.start(fnol_id, "test")
    _wait()
    run = registration.get_run(run_id)
    assert run["status"] == "FAILED"
    assert run["result"] == {"error": "browser unavailable"}
    assert run["steps"][-1]["label"] == "Run failed"
    assert intake.get(fnol_id)["status"] == "READY_TO_REGISTER"


# --------------------------------------------------------------------------
# reviewer relay
# --------------------------------------------------------------------------
NOTE = "Your invoice is fine. We just need the police report."


def _fake_model(monkeypatch, text: str) -> None:
    from app.llm import gateway

    def complete(prompt_key, variables=None, **kw):
        return gateway.LLMResult(text, "fake-model", prompt_key, "1", 3)

    monkeypatch.setattr(gateway, "complete", complete)


def test_relay_quotes_reviewer_when_no_model(conversation):
    from app.agents import handoff

    out = handoff.deliver_reviewer_response(conversation, note=NOTE, agent_name="Marcus")
    assert out["verbatim"] and NOTE in out["content"]
    stored = query_one("SELECT source_note, relay_source, author_name FROM message "
                       "WHERE id = ?", (out["message_id"],))
    assert dict(stored) == {"source_note": NOTE, "relay_source": "verbatim",
                            "author_name": "Marcus"}


def test_relay_uses_grounded_rendering(conversation, monkeypatch):
    from app.agents import handoff

    _fake_model(monkeypatch, "Good news: your invoice is fine. We just need the "
                             "police report from you now.")
    out = handoff.deliver_reviewer_response(conversation, note=NOTE, agent_name="Marcus")
    assert out["source"] == "fake-model" and not out["verbatim"]


def test_relay_rejects_rendering_that_adds_facts(conversation, monkeypatch):
    from app.agents import handoff

    _fake_model(monkeypatch, "Your claim CLM-99999 will be paid £4,200 on 3 December.")
    out = handoff.deliver_reviewer_response(conversation, note=NOTE, agent_name="Marcus")
    assert out["verbatim"] and NOTE in out["content"]
    assert query_one("SELECT COUNT(*) FROM audit_event WHERE event_type = "
                     "'relay_rendering_rejected' AND entity_id = ?", (conversation,))[0] == 1


def test_send_exact_words_skips_the_model(conversation, monkeypatch):
    from app.agents import handoff
    from app.llm import gateway

    def boom(*a, **kw):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(gateway, "complete", boom)
    out = handoff.deliver_reviewer_response(conversation, note=NOTE, agent_name="Marcus",
                                            force_verbatim=True)
    assert out["verbatim"]


def test_information_request_falls_back_to_plain_text(conversation):
    from app.agents import handoff

    out = handoff.request_information(conversation, request="police report",
                                      agent_name="Marcus")
    assert out["content"].startswith("To move your claim forward we need: police report")
    stored = query_one("SELECT relay_source FROM message WHERE id = ?", (out["message_id"],))
    assert stored["relay_source"] == "information_request"


def test_workflow_parity_script_runs():
    """The script the ports were diffed against (0 differences at port time)."""
    from tests import workflow_parity

    docs = workflow_parity.documents()
    assert len([d for d in docs if "doc_id" in d]) == query_one(
        "SELECT COUNT(*) FROM document")[0]
    regs = workflow_parity.registrations()
    assert [r.get("run_status") for r in regs[:3]] == ["SUCCEEDED"] * 3
    assert regs[3]["fnol_status"] == "READY_TO_REGISTER"
    assert len(workflow_parity.relays()) == 3


def test_new_graphs_emit_node_metrics(client, customer, conversation):
    from app.agents import handoff
    from app.documents import pipeline

    pipeline.process(query_one("SELECT id FROM document LIMIT 1")["id"])
    handoff.deliver_reviewer_response(conversation, note=NOTE, agent_name="Marcus")
    body = client.get("/metrics").text
    assert 'graph="documents"' in body and 'graph="relay"' in body
    assert 'cc_document_verdicts_total' in body
