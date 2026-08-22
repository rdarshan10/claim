"""Customer-facing claims endpoints (§14.1). Thin controllers only."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import Principal, current_principal
from app.repositories import claims as repo
from app.services import timeline_prediction

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get("")
async def list_claims(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    claims = repo.get_claims(principal.customer_id)
    for claim in claims:
        checklist = repo.checklist(claim["id"], principal.customer_id)
        claim["checklist"] = checklist
        # DOCS_PENDING reads differently depending on who is holding it.
        claim["status_meaning"] = timeline_prediction.stage_meaning(
            claim["status"], awaiting_customer=bool(checklist["awaiting_customer"]))
    return {"claims": claims}


@router.get("/{claim_id}")
async def get_claim(claim_id: str,
                    principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    claim = repo.get_claim(claim_id, principal.customer_id)
    if claim is None:
        # 404 rather than 403: never reveal that a claim exists for someone else.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")

    history = repo.get_status_history(claim_id, principal.customer_id)
    checklist = repo.checklist(claim_id, principal.customer_id)
    # Whether the customer still owes us paperwork changes both the wording and
    # the estimate, so it is resolved once and used for both.
    awaiting = bool(checklist["awaiting_customer"])
    return {
        **claim,
        "status_meaning": timeline_prediction.stage_meaning(
            claim["status"], awaiting_customer=awaiting),
        "history": history,
        "prediction": timeline_prediction.predict(claim, history, awaiting_customer=awaiting),
        "checklist": checklist,
        "documents": repo.get_documents(claim_id, principal.customer_id),
    }


@router.get("/{claim_id}/timeline")
async def get_timeline(claim_id: str,
                       principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    claim = repo.get_claim(claim_id, principal.customer_id)
    if claim is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    history = repo.get_status_history(claim_id, principal.customer_id)
    return {
        "claim_number": claim["claim_number"],
        "history": [
            {**h, "meaning": timeline_prediction.STAGE_MEANING.get(h["to_status"], "")}
            for h in history
        ],
        "prediction": timeline_prediction.predict(claim, history),
        "running_late": timeline_prediction.is_overdue(claim, history),
    }


@router.get("/{claim_id}/checklist")
async def get_checklist(claim_id: str,
                        principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    from app.agents.document import GUIDANCE

    if repo.get_claim(claim_id, principal.customer_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")

    checklist = repo.checklist(claim_id, principal.customer_id)
    for item in checklist["items"]:
        item["guidance"] = GUIDANCE.get(item["doc_type"], "")
    return checklist
