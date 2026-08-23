"""Agentic registration endpoints — the experiment, kept off the main path.

Separate router so nothing here can affect the scripted bot that staff actually
use. The comparison view is the point: same notification, same core system, one
run driven by hardcoded selectors and one by reading the form.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import Principal, require_role
from app.db import query
from app.fnol import agentic_registration, intake, registration

router = APIRouter(tags=["agentic"])


@router.post("/staff/fnol/{fnol_id}/register-agentic",
             status_code=status.HTTP_202_ACCEPTED)
async def register_agentic(
    fnol_id: str,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Register by reading the form. Returns immediately; poll the run."""
    try:
        run_id = agentic_registration.start(fnol_id, principal.name)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"run_id": run_id, "status": "RUNNING", "mode": "agentic"}


@router.get("/staff/fnol/{fnol_id}/compare")
async def compare(
    fnol_id: str,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Both approaches for one notification, for the side-by-side view."""
    record = intake.get(fnol_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")

    # Newest run of each kind. The two are told apart by who started them —
    # the agentic path tags its actor, so no schema change was needed to keep
    # the experiment out of the scripted bot's way.
    rows = query("SELECT id, started_by FROM rpa_run WHERE fnol_id = ? "
                 "ORDER BY started_at DESC", (fnol_id,))
    runs = [registration.get_run(r["id"]) for r in rows]
    scripted = [r for r in runs if r and "(agentic)" not in (r.get("started_by") or "")]
    agentic = [r for r in runs if r and "(agentic)" in (r.get("started_by") or "")]
    return {
        "reference": record["reference"],
        "status": record["status"],
        "scripted": scripted[0] if scripted else None,
        "agentic": agentic[0] if agentic else None,
    }
