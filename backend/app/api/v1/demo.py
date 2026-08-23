"""Demo controls — the model switch and the conversation reset.

These exist so a live demo can be steered without restarting the process. Both
are deliberately blunt: the model switch is global, and the reset genuinely
deletes data. Neither belongs in a real deployment, which is why they sit in
their own module behind a flag.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps import Principal, current_principal, require_role
from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query, query_one
from app.llm import gateway, providers

router = APIRouter(tags=["demo"])


# --------------------------------------------------------------------------
# Model switch
# --------------------------------------------------------------------------
class SelectRequest(BaseModel):
    provider: str
    primary: str | None = None
    mini: str | None = None


@router.get("/models")
async def list_models(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Providers, their models, and which is live."""
    return providers.describe()


@router.post("/models/select")
async def select_model(body: SelectRequest,
                       principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Switch provider and models for every subsequent request.

    Open to any signed-in user: this is a demo control, and the switch being
    global is the point — anyone driving the demo can change it.
    """
    if not providers.configured(body.provider):
        spec = providers.PROVIDERS.get(body.provider)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"No API key configured for {spec.label if spec else body.provider}. "
            f"Set {spec.env_key} in .env." if spec else "Unknown provider")

    try:
        selection = providers.select(body.provider, body.primary, body.mini)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    # Cached clients point at the previous endpoint, and cached answers came
    # from the previous model — both would mask the switch.
    gateway.reset_clients()
    gateway.clear_cache()

    audit.record("model_switched", actor_type=principal.role or "customer",
                 actor_id=principal.name, entity_type="config", entity_id="llm",
                 payload=selection.as_dict())
    return {"active": selection.as_dict(), **providers.describe()}


@router.post("/models/reset")
async def reset_model(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Go back to whatever the environment configures."""
    selection = providers.reset()
    gateway.reset_clients()
    gateway.clear_cache()
    audit.record("model_reset", actor_type=principal.role or "customer",
                 actor_id=principal.name, entity_type="config", entity_id="llm",
                 payload=selection.as_dict())
    return {"active": selection.as_dict(), **providers.describe()}


@router.post("/models/test")
async def test_model(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """One real call against the live selection, so a switch can be trusted
    before a demo rather than discovered mid-conversation."""
    selection = providers.current()
    started = datetime.now(timezone.utc)
    try:
        result = gateway.complete(
            "sentiment",
            {"message": "Thanks, that's really helpful."},
            tier="mini",
        )
        latency = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        return {"ok": True, "provider": selection.provider, "model": result.model,
                "latency_ms": latency, "sample": result.text[:200]}
    except Exception as exc:  # noqa: BLE001 - the failure is the answer here
        return {"ok": False, "provider": selection.provider,
                "error": str(exc)[:300]}


# --------------------------------------------------------------------------
# Demo reset
# --------------------------------------------------------------------------
class ResetRequest(BaseModel):
    scope: str = "conversation"   # conversation | customer | all | claims


@router.post("/demo/reset")
async def reset_demo(body: ResetRequest,
                     principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Clear demo state so a run can be repeated from a known point.

    ``conversation`` wipes this customer's chat history only — the usual case
    between takes. ``customer`` also drops their in-flight notifications and any
    open case. ``all`` is staff-only and clears the same for everyone.
    """
    if body.scope not in ("conversation", "customer", "all", "claims"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown scope")
    # ``all`` stays staff-only: it wipes every customer's chat history, which is
    # not something one customer should be able to do to the others. ``claims``
    # is open to whoever is driving the demo — it restores the shipped sample
    # set rather than destroying anything real.
    if body.scope == "all" and principal.role not in ("agent", "manager"):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Only staff can reset everyone's demo state")

    if body.scope == "claims":
        return _reset_claims(principal)

    cleared: dict[str, int] = {}

    if body.scope == "all":
        convs = [r["id"] for r in query("SELECT id FROM conversation")]
    else:
        convs = [r["id"] for r in query(
            "SELECT id FROM conversation WHERE customer_id = ?",
            (principal.customer_id,))]

    if convs:
        marks = ",".join("?" * len(convs))
        cleared["messages"] = query_one(
            f"SELECT COUNT(*) AS n FROM message WHERE conversation_id IN ({marks})",
            convs)["n"]
        execute(f"DELETE FROM message WHERE conversation_id IN ({marks})", convs)
        # Conversations themselves are kept: a live client holds the id, and
        # deleting the row would 404 its next poll mid-demo.
        cleared["conversations_cleared"] = len(convs)

    if body.scope in ("customer", "all"):
        where, params = ("", ()) if body.scope == "all" else \
            ("WHERE customer_id = ?", (principal.customer_id,))

        cleared["notifications"] = query_one(
            f"SELECT COUNT(*) AS n FROM notification {where}", params)["n"]
        execute(f"DELETE FROM notification {where}", params)

        # Only intake that never became a claim — a registered FNOL is the audit
        # record of a real claim and must not be deleted.
        fnol_where = ("WHERE claim_id IS NULL" if body.scope == "all"
                      else "WHERE customer_id = ? AND claim_id IS NULL")
        fnols = [r["id"] for r in query(f"SELECT id FROM fnol_request {fnol_where}", params)]
        if fnols:
            marks = ",".join("?" * len(fnols))
            execute(f"DELETE FROM fnol_document WHERE fnol_id IN ({marks})", fnols)
            execute(f"DELETE FROM rpa_run WHERE fnol_id IN ({marks})", fnols)
            execute(f"DELETE FROM fnol_request WHERE id IN ({marks})", fnols)
        cleared["notifications_of_loss"] = len(fnols)

        tickets = [r["id"] for r in query(
            f"SELECT id FROM escalation_ticket {where}", params)]
        if tickets:
            marks = ",".join("?" * len(tickets))
            execute(f"DELETE FROM escalation_ticket WHERE id IN ({marks})", tickets)
        cleared["cases"] = len(tickets)

    audit.record("demo_reset", actor_type=principal.role or "customer",
                 actor_id=principal.name, entity_type="demo",
                 entity_id=principal.customer_id or "all",
                 payload={"scope": body.scope, "cleared": cleared})

    return {"scope": body.scope, "cleared": cleared}


def _reset_claims(principal: Principal) -> dict[str, Any]:
    """Put the claims book back to its shipped state.

    ``generate()`` already truncates every claim-side table and rebuilds the
    sample data deterministically from seed 42, so the demo personas return
    exactly as they were. What it does not know about is FNOL: those tables
    postdate it and would be left pointing at claims that no longer exist, so
    they are cleared here first.
    """
    counts = {
        "claims_before": query_one("SELECT COUNT(*) AS n FROM claim")["n"],
        "notifications_of_loss": query_one("SELECT COUNT(*) AS n FROM fnol_request")["n"],
    }

    # Order matters: fnol_document and rpa_run both reference fnol_request.
    execute("DELETE FROM fnol_document")
    execute("DELETE FROM rpa_run")
    execute("DELETE FROM fnol_request")

    from datagen.generate import generate

    generate(seed=42, customers=20, claims_per=2)
    # Counted from the table rather than the generator's summary, which reports
    # documents and customers but not claims.
    counts["sample_claims_restored"] = query_one("SELECT COUNT(*) AS n FROM claim")["n"]

    # Recorded *after* the regenerate: generate() truncates audit_event, so an
    # entry written first would be wiped by the very call it describes.
    audit.record("demo_claims_reset", actor_type=principal.role,
                 actor_id=principal.name, entity_type="demo", entity_id="claims",
                 payload=counts)
    return {"scope": "claims", "cleared": counts}


@router.get("/demo/state")
async def demo_state(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """What a reset would clear, so the button can say so before it's pressed."""
    if principal.customer_id:
        convs = [r["id"] for r in query(
            "SELECT id FROM conversation WHERE customer_id = ?", (principal.customer_id,))]
        messages = 0
        if convs:
            marks = ",".join("?" * len(convs))
            messages = query_one(
                f"SELECT COUNT(*) AS n FROM message WHERE conversation_id IN ({marks})",
                convs)["n"]
        return {
            "messages": messages,
            "notifications_of_loss": query_one(
                "SELECT COUNT(*) AS n FROM fnol_request WHERE customer_id = ? AND claim_id IS NULL",
                (principal.customer_id,))["n"],
            "cases": query_one(
                "SELECT COUNT(*) AS n FROM escalation_ticket WHERE customer_id = ?",
                (principal.customer_id,))["n"],
        }
    return {"messages": query_one("SELECT COUNT(*) AS n FROM message")["n"]}
