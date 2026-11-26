"""The AIOps agent: detection, diagnosis guardrails, remediation and approval."""
from __future__ import annotations

import json
import uuid

import pytest

from app.aiops import detectors, playbooks, signals
from app.aiops import graph as aiops
from app.db import execute, query, query_one
from tests.conftest import token


# --------------------------------------------------------------------------
# helpers that create operational trouble
# --------------------------------------------------------------------------
def _stuck_document() -> str:
    doc = query_one("SELECT id FROM document ORDER BY uploaded_at DESC LIMIT 1")["id"]
    execute("UPDATE document SET status = 'OCR', uploaded_at = datetime('now','-1 hour') "
            "WHERE id = ?", (doc,))
    return doc


def _breached_case(customer_id: str) -> str:
    ticket = str(uuid.uuid4())
    execute("INSERT INTO escalation_ticket (id, customer_id, priority, reason, status, "
            "created_at) VALUES (?,?, 'NORMAL', 'test', 'OPEN', datetime('now','-6 hours'))",
            (ticket, customer_id))
    return ticket


def _failed_registration(customer_id: str) -> str:
    from app.fnol import intake

    record = intake.create(customer_id)
    intake.set_status(record["id"], "READY_TO_REGISTER")
    execute("INSERT INTO rpa_run (id, fnol_id, status, started_at, error) "
            "VALUES (?,?, 'FAILED', datetime('now','-5 minutes'), 'core system timeout')",
            (str(uuid.uuid4()), record["id"]))
    return record["id"]


def _llm_failures(n_ok: int, n_err: int) -> None:
    for i in range(n_ok + n_err):
        execute("INSERT INTO llm_call (id, at, prompt_key, prompt_version, model, "
                "latency_ms, ok) VALUES (?, datetime('now'), 'classify_turn', '1', "
                "'groq/openai/gpt-oss-20b', 900, ?)", (str(uuid.uuid4()), int(i < n_ok)))


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------
def test_quiet_system_raises_nothing():
    # Seeded stale claims are the only thing a fresh dataset has wrong.
    found = {a.detector for a in detectors.detect(signals.collect())}
    assert found <= {"claims_at_risk"}


def test_threshold_needs_enough_samples():
    rule = detectors.BY_ID["llm_errors"]
    few = signals.Signal("llm_error_rate", 1.0, "ratio", "", sample_size=2,
                         evidence={"llm_enabled": True})
    many = signals.Signal("llm_error_rate", 0.6, "ratio", "", sample_size=10,
                          evidence={"llm_enabled": True})
    assert detectors.evaluate(rule, few) is None
    hit = detectors.evaluate(rule, many)
    assert hit and hit.severity == "critical"          # 0.6 ≥ escalate_at 0.5


def test_llm_errors_ignored_in_template_mode():
    rule = detectors.BY_ID["llm_errors"]
    sig = signals.Signal("llm_error_rate", 1.0, "ratio", "", sample_size=50,
                         evidence={"llm_enabled": False})
    assert detectors.evaluate(rule, sig) is None


def test_baseline_zscore_fires_without_a_threshold():
    rule = detectors.BY_ID["cost_spike"]
    run = "baseline-run"
    for i in range(detectors.MIN_BASELINE):
        detectors.snapshot(f"{run}-{i}", {"llm_cost_usd": signals.Signal(
            "llm_cost_usd", 0.010 + 0.001 * (i % 3), "usd", "")})
    normal = signals.Signal("llm_cost_usd", 0.011, "usd", "", sample_size=5)
    spike = signals.Signal("llm_cost_usd", 0.5, "usd", "", sample_size=5)
    assert detectors.evaluate(rule, normal, "now") is None
    hit = detectors.evaluate(rule, spike, "now")
    assert hit and hit.z_score and hit.z_score > detectors.Z_THRESHOLD


def test_template_mode_only_an_incident_when_aiops_switched_it_on():
    rule = detectors.BY_ID["template_mode"]
    assert detectors.evaluate(rule, signals.collect()["llm_template_mode"]) is None
    playbooks.STATE["template_mode_by_aiops"] = True
    assert detectors.evaluate(rule, signals.collect()["llm_template_mode"]) is not None


def test_signals_see_injected_trouble(customer):
    _stuck_document()
    _breached_case(customer["id"])
    _failed_registration(customer["id"])
    _llm_failures(n_ok=2, n_err=8)
    current = signals.collect()
    assert current["documents_stuck"].value == 1
    assert current["escalations_sla_breached"].value == 1
    assert current["registration_failures"].value == 1
    assert current["llm_error_rate"].value == pytest.approx(0.8)


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------
def test_sweep_auto_fixes_low_risk_and_pauses_for_high_risk(customer):
    doc = _stuck_document()
    ticket = _breached_case(customer["id"])
    fnol_id = _failed_registration(customer["id"])

    result = aiops.sweep()
    assert result["status"] == "AWAITING_APPROVAL"
    detected = {a["detector"] for a in result["anomalies"]}
    assert {"documents_stuck", "escalation_sla", "registration_failures"} <= detected

    plan = {p["playbook"]: p for p in result["plan"]}
    assert plan["requeue_stuck_documents"]["status"] == "executed"
    assert plan["prioritise_breached_escalations"]["status"] == "executed"
    assert plan["retry_failed_registrations"]["status"] == "awaiting_approval"
    assert query_one("SELECT priority FROM escalation_ticket WHERE id = ?",
                     (ticket,))["priority"] == "HIGH"
    # The high-risk action has not run.
    assert query_one("SELECT COUNT(*) FROM rpa_run WHERE fnol_id = ?", (fnol_id,))[0] == 1

    row = query_one("SELECT status, diagnosis_source FROM aiops_incident WHERE id = ?",
                    (result["incident_id"],))
    assert row["status"] == "AWAITING_APPROVAL"
    assert row["diagnosis_source"] == "heuristic"     # LLM is off in tests

    decided = aiops.decide(result["incident_id"],
                           approve=["retry_failed_registrations"], by="manager.elena")
    assert decided["status"] in ("MITIGATED", "RESOLVED")
    retry = [a for a in decided["actions"] if a["playbook"] == "retry_failed_registrations"]
    assert retry and retry[0]["by"] == "manager.elena" and retry[0]["outcome"] == "ok"
    assert query_one("SELECT COUNT(*) FROM rpa_run WHERE fnol_id = ?", (fnol_id,))[0] == 2
    assert decided["verification"]["at"]

    with pytest.raises(ValueError):
        aiops.decide(result["incident_id"], approve=[], by="manager.elena")
    assert doc


def test_scheduling_chases_clears_the_stale_claim_signal():
    assert signals.collect()["claims_at_risk"].value > 0
    result = aiops.sweep()
    assert "claims_at_risk" in {a["detector"] for a in result["anomalies"]}
    cleared = {c["detector"] for c in result["verification"]["cleared"]}
    assert "claims_at_risk" in cleared
    assert result["status"] == "RESOLVED"


def test_rejecting_high_risk_runs_nothing(customer):
    fnol_id = _failed_registration(customer["id"])
    result = aiops.sweep()
    decided = aiops.decide(result["incident_id"], approve=[], by="manager.elena")
    plan = {p["playbook"]: p["status"] for p in decided["plan"]}
    assert plan["retry_failed_registrations"] == "rejected"
    assert query_one("SELECT COUNT(*) FROM rpa_run WHERE fnol_id = ?", (fnol_id,))[0] == 1


def test_open_incident_is_not_reopened(customer):
    _breached_case(customer["id"])
    first = aiops.sweep()
    assert first["incident_id"]
    second = aiops.sweep()
    assert second["status"] == "quiet"
    assert "escalation_sla" in second["suppressed"]


def test_diagnosis_cannot_invent_or_widen_actions(customer, monkeypatch):
    """The model may only pick playbooks registered for what actually fired."""
    from app.llm import gateway

    _breached_case(customer["id"])

    def fake_complete(prompt_key, variables=None, **kw):
        assert prompt_key == "aiops_diagnose"
        text = json.dumps({"title": "Reviewers swamped", "summary": "s",
                           "root_cause": "capacity", "confidence": 0.9,
                           "recommended_playbooks": ["drop_all_tables",
                                                     "enable_template_mode",
                                                     "prioritise_breached_escalations"]})
        return gateway.LLMResult(text, "fake-model", prompt_key, "1.0.0", 5)

    monkeypatch.setattr(gateway, "complete", fake_complete)
    result = aiops.sweep()
    diagnosis = result["diagnosis"]
    assert diagnosis["source"] == "fake-model"
    assert diagnosis["recommended_playbooks"] == ["prioritise_breached_escalations"]
    assert set(diagnosis["rejected_suggestions"]) == {"drop_all_tables",
                                                      "enable_template_mode"}
    assert {p["playbook"] for p in result["plan"]} <= set(playbooks.PLAYBOOKS)
    assert result["status"] != "AWAITING_APPROVAL"


def test_low_risk_waits_for_approval_when_auto_remediation_is_off(customer):
    from app.config import get_settings

    _breached_case(customer["id"])
    get_settings().aiops_auto_remediate = False
    try:
        result = aiops.sweep()
    finally:
        get_settings().aiops_auto_remediate = True
    assert result["status"] == "AWAITING_APPROVAL"
    assert all(p["status"] == "awaiting_approval" for p in result["plan"])


def test_llm_outage_recommends_failover_or_template_mode():
    _llm_failures(n_ok=1, n_err=9)
    from app.config import get_settings

    # An outage, not template mode. No key is configured, so the diagnosis
    # call itself fails over to the rule-based diagnosis.
    get_settings().llm_enabled = True
    result = aiops.sweep()
    assert "llm_errors" in {a["detector"] for a in result["anomalies"]}
    waiting = {p["playbook"] for p in result["plan"] if p["status"] == "awaiting_approval"}
    # No second provider is configured in tests, so failover is not offered.
    assert waiting == {"enable_template_mode"}
    decided = aiops.decide(result["incident_id"], approve=["enable_template_mode"],
                           by="manager.elena")
    assert get_settings().llm_enabled is False
    assert playbooks.STATE["template_mode_by_aiops"] is True
    assert decided["status"] in ("MITIGATED", "RESOLVED")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def test_aiops_api_is_manager_only(client):
    agent = {"Authorization": f"Bearer {token('agent', 'agent.marcus')}"}
    assert client.get("/api/v1/admin/aiops/incidents", headers=agent).status_code == 403


def test_aiops_api_round_trip(client, customer):
    _failed_registration(customer["id"])
    mgr = {"Authorization": f"Bearer {token('manager')}"}

    sig = client.get("/api/v1/admin/aiops/signals", headers=mgr).json()
    assert any(s["name"] == "registration_failures" and s["value"] == 1
               for s in sig["signals"])

    sweep = client.post("/api/v1/admin/aiops/sweep", headers=mgr).json()
    assert sweep["status"] == "AWAITING_APPROVAL"
    incident_id = sweep["incident_id"]

    listed = client.get("/api/v1/admin/aiops/incidents", headers=mgr).json()["incidents"]
    assert listed[0]["id"] == incident_id

    detail = client.get(f"/api/v1/admin/aiops/incidents/{incident_id}", headers=mgr).json()
    assert detail["graph_history"][0]["next"] == ["approval"]

    bad = client.post(f"/api/v1/admin/aiops/incidents/{incident_id}/decision",
                      headers=mgr, json={"approve": ["rm_rf"]})
    assert bad.status_code == 422

    ok = client.post(f"/api/v1/admin/aiops/incidents/{incident_id}/decision",
                     headers=mgr, json={"approve": []})
    assert ok.status_code == 200

    again = client.post(f"/api/v1/admin/aiops/incidents/{incident_id}/decision",
                        headers=mgr, json={"approve": []})
    assert again.status_code == 409

    closed = client.post(f"/api/v1/admin/aiops/incidents/{incident_id}/status",
                         headers=mgr, json={"status": "DISMISSED"})
    assert closed.json() == {"status": "DISMISSED"}


def test_metrics_endpoint_exposes_graph_and_aiops_series(client, customer, conversation):
    from app.agents import graph

    graph.run_turn(customer_id=customer["id"], customer_name=customer["full_name"],
                   message="hi", conversation_id=conversation)
    aiops.sweep()
    body = client.get("/metrics").text
    assert "cc_graph_node_seconds_bucket" in body
    assert 'graph="conversation"' in body and 'graph="aiops"' in body
    assert "cc_conversation_turns_total" in body


def test_conversation_history_endpoint(client, customer, conversation):
    from app.agents import graph

    graph.run_turn(customer_id=customer["id"], customer_name=customer["full_name"],
                   message="where is my claim?", conversation_id=conversation)
    mgr = {"Authorization": f"Bearer {token('manager')}"}
    steps = client.get(f"/api/v1/admin/graph/conversations/{conversation}/history",
                       headers=mgr).json()["steps"]
    ran = {node for s in steps for node in s["next"]}
    assert {"input_guard", "route", "agent_claim_status", "compose", "emit"} <= ran


def test_traces_endpoint_returns_node_spans(client, customer, conversation):
    from app.agents import graph

    state = graph.run_turn(customer_id=customer["id"], customer_name=customer["full_name"],
                           message="what does excess mean?", conversation_id=conversation)
    mgr = {"Authorization": f"Bearer {token('manager')}"}
    spans = client.get("/api/v1/admin/ops/traces", headers=mgr,
                       params={"trace_id": state.trace_id}).json()["spans"]
    names = {s["name"] for s in spans}
    assert {"conversation.input_guard", "conversation.agent_knowledge",
            "conversation.turn"} <= names
    assert query("SELECT 1")  # db still reachable after the client closed
