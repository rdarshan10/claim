"""FNOL endpoints — customer intake, staff triage, and the registration bot."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.agents import fnol_agent
from app.api.deps import Principal, current_principal, rate_limit, require_role
from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query_one
from app.fnol import intake, registration

router = APIRouter(tags=["fnol"])


# --------------------------------------------------------------------------
# Customer
# --------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    field: str = Field(min_length=1, max_length=64)
    value: Any = None


@router.get("/fnol")
async def list_notifications(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    records = intake.list_for_customer(principal.customer_id)
    return {"notifications": [
        {**intake.status_card(r)["payload"],
         "items": intake.summary(r),
         "document_count": len(intake.documents(r["id"])),
         "created_at": r["created_at"]}
        for r in records
    ]}


@router.get("/fnol/{fnol_id}")
async def get_notification(fnol_id: str,
                           principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    record = _owned(fnol_id, principal)
    # `card` is the live control for this notification right now: the next
    # question, or the review card when nothing is outstanding. The client uses
    # it to re-sync a card replayed from thread history, which would otherwise
    # show a question the customer answered several turns ago.
    nxt = (intake.next_field(record.get("claim_type"), record["answers"])
           if record["status"] == "COLLECTING" else None)
    card = (intake.question_card(record, nxt) if nxt
            else intake.review_card(record) if record["status"] == "COLLECTING"
            else intake.status_card(record))

    return {
        **record,
        "items": intake.summary(record),
        "documents": [{"id": d["id"], "filename": d["filename"], "doc_type": d["doc_type"]}
                      for d in intake.documents(fnol_id)],
        "missing": intake.missing_mandatory(record.get("claim_type"), record["answers"]),
        "card": card,
    }


@router.post("/fnol/{fnol_id}/answer")
async def answer(fnol_id: str, body: AnswerRequest,
                 principal: Principal = Depends(rate_limit)) -> dict[str, Any]:
    """Record one answer from a question card and return the next question.

    The card path exists alongside free text in chat: tapping an option is the
    same operation, just without the parsing ambiguity.
    """
    record = _owned(fnol_id, principal)
    if record["status"] != "COLLECTING":
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{record['reference']} has already been submitted")

    spec = intake.field_by_key(record.get("claim_type"), body.field)
    if spec is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown field: {body.field}")

    value = intake.normalise(spec, body.value)
    if value is None and not spec.optional:
        # Re-ask rather than fail: the customer gave us something we couldn't
        # parse, which is a conversation to continue, not an error to raise.
        return {
            "fnol_id": fnol_id,
            "reference": record["reference"],
            "complete": False,
            "retry": True,
            "message": _retry_message(spec),
            "card": intake.question_card(record, spec),
        }

    record = intake.save_answer(fnol_id, spec.key, value)
    nxt = intake.next_field(record.get("claim_type"), record["answers"])

    return {
        "fnol_id": fnol_id,
        "reference": record["reference"],
        "complete": nxt is None,
        "card": (intake.question_card(record, nxt) if nxt
                 else intake.review_card(record)),
    }


def _retry_message(spec: Any) -> str:
    if spec.kind == "date":
        return "I couldn't read that as a date — could you pick one instead?"
    if spec.kind == "money":
        return "I didn't catch a figure there — a rough number is fine."
    if spec.kind == "choice":
        return "Could you pick one of these?"
    return "Sorry, I didn't catch that — could you try again?"


@router.post("/fnol/{fnol_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload(fnol_id: str, file: UploadFile = File(...),
                 principal: Principal = Depends(rate_limit)) -> dict[str, Any]:
    """Attach a document during intake — a first incident report, photos, a quote.

    These are not run through the claim document pipeline: there is no claim to
    validate them against yet. They are carried onto the claim at registration
    and processed then.
    """
    record = _owned(fnol_id, principal)
    if record["status"] not in ("COLLECTING", "INFO_REQUIRED"):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{record['reference']} isn't accepting uploads")

    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    doc_id = str(uuid.uuid4())
    safe_name = Path(file.filename or "upload.bin").name
    storage_key = str(Path(settings.blob_dir) / f"fnol_{doc_id}_{safe_name}")
    Path(storage_key).write_bytes(raw)

    # Text formats are readable now and give the reviewer something to see
    # without opening the blob; binaries wait for the claim pipeline.
    content = None
    if Path(safe_name).suffix.lower() in (".txt", ".md", ".csv", ".json"):
        try:
            content = raw.decode("utf-8", errors="replace")[:20000]
        except Exception:  # noqa: BLE001 - unreadable content is not an error here
            content = None

    intake.add_document(fnol_id, safe_name, storage_key, content)
    audit.record("fnol_document_uploaded", actor_type="customer",
                 actor_id=principal.customer_id, entity_type="fnol", entity_id=fnol_id,
                 payload={"filename": safe_name, "bytes": len(raw)})

    return {"id": doc_id, "filename": safe_name,
            "documents": [{"filename": d["filename"]} for d in intake.documents(fnol_id)]}


@router.post("/fnol/{fnol_id}/submit")
async def submit(fnol_id: str,
                 principal: Principal = Depends(rate_limit)) -> dict[str, Any]:
    """Hand the notification to the claims team. Still not a claim."""
    record = _owned(fnol_id, principal)
    if record["status"] not in ("COLLECTING", "INFO_REQUIRED"):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{record['reference']} has already been submitted")

    missing = intake.missing_mandatory(record.get("claim_type"), record["answers"])
    if missing:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Still needed: {', '.join(missing)}")

    intake.set_status(fnol_id, "SUBMITTED")
    record = intake.get(fnol_id)
    audit.record("fnol_submitted", actor_type="customer", actor_id=principal.customer_id,
                 entity_type="fnol", entity_id=fnol_id,
                 payload={"reference": record["reference"],
                          "claim_type": record.get("claim_type")})

    # The confirmation belongs in the thread, so the customer's history reads as
    # one continuous conversation rather than an action that happened elsewhere.
    if record.get("conversation_id"):
        body = (
            f"Thanks — that's submitted. Your reference is **{record['reference']}**.\n\n"
            f"Our claims team will check the details and register the claim on our "
            f"system. I'll let you know here as soon as it has a claim number — "
            f"you don't need to do anything in the meantime."
        )
        execute(
            """INSERT INTO message (id, conversation_id, role, content, created_at)
               VALUES (?,?, 'assistant', ?, ?)""",
            (str(uuid.uuid4()), record["conversation_id"], body,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )

    return {"reference": record["reference"], "status": record["status"],
            "card": intake.receipt_card(record)}


def _owned(fnol_id: str, principal: Principal) -> dict[str, Any]:
    record = intake.get(fnol_id)
    if record is None or record["customer_id"] != principal.customer_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    return record


# --------------------------------------------------------------------------
# Staff triage
# --------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    status: str
    note: str = ""


@router.get("/staff/fnol")
async def staff_queue(include_closed: bool = False,
                      principal: Principal = Depends(require_role("agent", "manager"))) -> dict[str, Any]:
    return {"requests": intake.for_staff(include_closed)}


@router.get("/staff/fnol/{fnol_id}")
async def staff_detail(fnol_id: str,
                       principal: Principal = Depends(require_role("agent", "manager"))) -> dict[str, Any]:
    record = intake.get(fnol_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")

    customer = query_one("SELECT full_name FROM customer WHERE id = ?",
                         (record["customer_id"],))
    policy = query_one(
        "SELECT * FROM policy WHERE customer_id = ? ORDER BY start_date DESC LIMIT 1",
        (record["customer_id"],),
    )
    run = registration.latest_run(fnol_id)

    return {
        "request": record,
        "customer_name": customer["full_name"] if customer else "",
        "policy": dict(policy) if policy else None,
        "items": intake.summary(record),
        "documents": intake.documents(fnol_id),
        "missing": intake.missing_mandatory(record.get("claim_type"), record["answers"]),
        "run": run,
    }


@router.post("/staff/fnol/{fnol_id}/review")
async def review(fnol_id: str, body: ReviewRequest,
                 principal: Principal = Depends(require_role("agent", "manager"))) -> dict[str, Any]:
    """Move a notification through triage.

    INFO_REQUIRED posts the reviewer's question into the customer's thread —
    the assistant carries it, as it does every other message from the team.
    """
    allowed = {"UNDER_REVIEW", "INFO_REQUIRED", "READY_TO_REGISTER", "REJECTED"}
    if body.status not in allowed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Status must be one of {sorted(allowed)}")

    record = intake.get(fnol_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    if record["status"] == "REGISTERED":
        raise HTTPException(status.HTTP_409_CONFLICT, "Already registered")

    intake.set_status(fnol_id, body.status, reviewer=principal.name, note=body.note or None)
    audit.record("fnol_reviewed", actor_type="staff", actor_id=principal.name,
                 entity_type="fnol", entity_id=fnol_id,
                 payload={"status": body.status, "note": body.note[:300]})

    if body.status in ("INFO_REQUIRED", "REJECTED") and record.get("conversation_id"):
        if body.status == "INFO_REQUIRED":
            body_text = (
                f"Our claims team have looked at **{record['reference']}** and need "
                f"a little more before they can register it:\n\n{body.note}\n\n"
                f"Just reply here and I'll pass it on."
            )
        else:
            body_text = (
                f"I'm sorry — our claims team weren't able to take "
                f"**{record['reference']}** forward.\n\n{body.note}\n\n"
                f"If you think that's wrong, say so here and I'll get someone to "
                f"look again."
            )
        execute(
            """INSERT INTO message (id, conversation_id, role, content, created_at)
               VALUES (?,?, 'assistant', ?, ?)""",
            (str(uuid.uuid4()), record["conversation_id"], body_text,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )

    return {"status": body.status, "request": intake.get(fnol_id)}


# --------------------------------------------------------------------------
# Registration bot
# --------------------------------------------------------------------------
@router.post("/staff/fnol/{fnol_id}/register", status_code=status.HTTP_202_ACCEPTED)
async def register(fnol_id: str,
                   principal: Principal = Depends(require_role("agent", "manager"))) -> dict[str, Any]:
    """Start the registration bot. Returns immediately; poll the run for progress."""
    try:
        run_id = registration.start(fnol_id, principal.name)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"run_id": run_id, "status": "RUNNING"}


@router.get("/staff/rpa/{run_id}")
async def run_status(run_id: str,
                     principal: Principal = Depends(require_role("agent", "manager"))) -> dict[str, Any]:
    run = registration.get_run(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run
