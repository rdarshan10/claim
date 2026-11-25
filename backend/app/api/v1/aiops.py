"""AIOps and graph-ops endpoints. Manager only."""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.aiops import detectors, playbooks, signals
from app.aiops import graph as aiops
from app.api.deps import Principal, require_role
from app.config import get_settings
from app.db import query_one
from app.ops import telemetry

router = APIRouter(tags=["aiops"])
manager = require_role("manager")


@router.get("/admin/aiops/signals")
async def live_signals(principal: Principal = Depends(manager)) -> dict[str, Any]:
    current = await asyncio.to_thread(signals.collect)
    return {"window_minutes": get_settings().aiops_window_minutes,
            "signals": [s.as_dict() for s in current.values()]}


@router.get("/admin/aiops/catalogue")
async def catalogue(principal: Principal = Depends(manager)) -> dict[str, Any]:
    settings = get_settings()
    return {
        "detectors": [d.__dict__ for d in detectors.DETECTORS],
        "playbooks": [p.public() for p in playbooks.PLAYBOOKS.values()],
        "settings": {"enabled": settings.aiops_enabled,
                     "interval_seconds": settings.aiops_interval_seconds,
                     "auto_remediate": settings.aiops_auto_remediate,
                     "cooldown_minutes": settings.aiops_cooldown_minutes},
    }


@router.get("/admin/aiops/incidents")
async def list_incidents(status_filter: str | None = None, limit: int = 50,
                         principal: Principal = Depends(manager)) -> dict[str, Any]:
    items = await asyncio.to_thread(aiops.incidents, status_filter, min(limit, 200))
    return {"incidents": items}


@router.get("/admin/aiops/incidents/{incident_id}")
async def get_incident(incident_id: str,
                       principal: Principal = Depends(manager)) -> dict[str, Any]:
    found = [i for i in aiops.incidents(limit=500) if i["id"] == incident_id]
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    incident = found[0]
    from app.agents import runtime

    incident["graph_history"] = runtime.thread_history(incident["thread_id"],
                                                       aiops.compiled(), limit=20)
    return incident


@router.post("/admin/aiops/sweep")
async def run_sweep(principal: Principal = Depends(manager)) -> dict[str, Any]:
    return await asyncio.to_thread(aiops.sweep, "manual", principal.name)


class Decision(BaseModel):
    approve: list[str] = []


@router.post("/admin/aiops/incidents/{incident_id}/decision")
async def decide(incident_id: str, body: Decision,
                 principal: Principal = Depends(manager)) -> dict[str, Any]:
    unknown = [p for p in body.approve if p not in playbooks.PLAYBOOKS]
    if unknown:
        raise HTTPException(422,
                            f"Unknown playbook(s): {', '.join(unknown)}")
    try:
        return await asyncio.to_thread(aiops.decide, incident_id,
                                       approve=body.approve, by=principal.name)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


class StatusChange(BaseModel):
    status: Literal["RESOLVED", "DISMISSED"]


@router.post("/admin/aiops/incidents/{incident_id}/status")
async def change_status(incident_id: str, body: StatusChange,
                        principal: Principal = Depends(manager)) -> dict[str, str]:
    if query_one("SELECT id FROM aiops_incident WHERE id = ?", (incident_id,)) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    aiops.set_status(incident_id, body.status, principal.name)
    return {"status": body.status}


@router.get("/admin/graph/conversations/{conversation_id}/history")
async def conversation_history(conversation_id: str, limit: int = 60,
                               principal: Principal = Depends(manager)) -> dict[str, Any]:
    """Every checkpointed super-step of a conversation, newest first."""
    from app.agents import graph, runtime

    if runtime.checkpointer() is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Checkpointing is switched off.")
    steps = await asyncio.to_thread(runtime.thread_history, f"conv:{conversation_id}",
                                    graph.compiled(), min(limit, 300))
    return {"conversation_id": conversation_id, "steps": steps}


@router.get("/admin/ops/traces")
async def traces(trace_id: str | None = None, prefix: str | None = None, limit: int = 200,
                 principal: Principal = Depends(manager)) -> dict[str, Any]:
    return {"spans": telemetry.recent_spans(min(limit, 2000), trace_id, prefix)}
