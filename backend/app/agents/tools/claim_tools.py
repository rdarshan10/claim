"""Typed tools — the only sanctioned path from an agent to data (§10).

Every tool takes ``customer_id`` as its first argument and that value always
originates from the JWT. No tool accepts a customer id supplied in message text,
and no tool can mutate a claim decision — the capability simply does not exist
(§17.1 "jailbreak -> decision manipulation").
"""
from __future__ import annotations

from typing import Any

from app.audit import logger as audit
from app.repositories import claims as repo
from app.services import timeline_prediction


def _audited(name: str, customer_id: str, args: dict[str, Any], result_size: int) -> None:
    audit.record("tool_call", actor_type="agent", actor_id=name,
                 entity_type="customer", entity_id=customer_id,
                 payload={"tool": name, "args": args, "results": result_size})


def get_claims(customer_id: str) -> list[dict[str, Any]]:
    claims = repo.get_claims(customer_id)
    _audited("get_claims", customer_id, {}, len(claims))
    return claims


def get_claim_detail(customer_id: str, claim_id: str) -> dict[str, Any] | None:
    claim = repo.get_claim(claim_id, customer_id)
    _audited("get_claim_detail", customer_id, {"claim_id": claim_id}, 1 if claim else 0)
    return claim


def find_claim_by_number(customer_id: str, claim_number: str) -> dict[str, Any] | None:
    claim = repo.find_claim_by_number(claim_number, customer_id)
    _audited("find_claim_by_number", customer_id, {"claim_number": claim_number},
             1 if claim else 0)
    return claim


def get_status_history(customer_id: str, claim_id: str) -> list[dict[str, Any]]:
    history = repo.get_status_history(claim_id, customer_id)
    _audited("get_status_history", customer_id, {"claim_id": claim_id}, len(history))
    return history


def predict_timeline(customer_id: str, claim_id: str) -> dict[str, Any] | None:
    claim = repo.get_claim(claim_id, customer_id)
    if claim is None:
        return None
    history = repo.get_status_history(claim_id, customer_id)
    # Whether the customer still owes us paperwork changes the DOCS_PENDING
    # estimate materially, and the tool already knows enough to find out.
    checklist = repo.checklist(claim_id, customer_id)
    prediction = timeline_prediction.predict(
        claim, history, awaiting_customer=bool(checklist["awaiting_customer"]))
    _audited("predict_timeline", customer_id, {"claim_id": claim_id}, 1)
    return prediction


def get_required_documents(customer_id: str, claim_id: str) -> dict[str, Any]:
    checklist = repo.checklist(claim_id, customer_id)
    _audited("get_required_documents", customer_id, {"claim_id": claim_id},
             len(checklist["items"]))
    return checklist


def get_document_status(customer_id: str, claim_id: str) -> list[dict[str, Any]]:
    documents = repo.get_documents(claim_id, customer_id)
    _audited("get_document_status", customer_id, {"claim_id": claim_id}, len(documents))
    return documents


def get_rejection_details(customer_id: str, document_id: str) -> dict[str, Any] | None:
    document = repo.get_document(document_id, customer_id)
    _audited("get_rejection_details", customer_id, {"document_id": document_id},
             1 if document else 0)
    if document is None:
        return None
    return document.get("rejection_payload")
