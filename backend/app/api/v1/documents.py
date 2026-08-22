"""Document upload and verification endpoints (§14.1)."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import Principal, current_principal, rate_limit
from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query_one
from app.documents import pipeline
from app.repositories import claims as repo

router = APIRouter(tags=["documents"])

# Idempotency keys already seen, so a retried upload doesn't create a second doc.
_idempotency: dict[str, str] = {}


@router.post("/claims/{claim_id}/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    claim_id: str,
    file: UploadFile = File(...),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
    principal: Principal = Depends(rate_limit),
) -> dict[str, Any]:
    settings = get_settings()

    if repo.get_claim(claim_id, principal.customer_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")

    if idempotency_key and idempotency_key in _idempotency:
        return {"doc_id": _idempotency[idempotency_key], "status": "ACCEPTED",
                "idempotent_replay": True}

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    doc_id = str(uuid.uuid4())
    safe_name = Path(file.filename or "upload.txt").name
    storage_key = str(Path(settings.blob_dir) / f"{doc_id}_{safe_name}")
    Path(storage_key).write_bytes(raw)

    execute(
        """INSERT INTO document (id, claim_id, filename, status, storage_key,
                                 extracted_fields, uploaded_at)
           VALUES (?,?,?, 'UPLOADED', ?, '{}', ?)""",
        (doc_id, claim_id, safe_name, storage_key,
         datetime.now(timezone.utc).isoformat()),
    )
    audit.record("document_uploaded", actor_type="customer", actor_id=principal.customer_id,
                 entity_type="document", entity_id=doc_id,
                 payload={"claim_id": claim_id, "filename": safe_name, "bytes": len(raw)})

    if idempotency_key:
        _idempotency[idempotency_key] = doc_id

    # Async in the MVP == a worker thread. Swap for a Celery ``.delay()`` later.
    threading.Thread(target=pipeline.process, args=(doc_id,), daemon=True).start()

    return {"doc_id": doc_id, "status": "ACCEPTED", "job_id": doc_id}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str,
                       principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    document = repo.get_document(doc_id, principal.customer_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return {
        "id": document["id"],
        "filename": document["filename"],
        "doc_type": document["doc_type"],
        "status": document["status"],
        "uploaded_at": document["uploaded_at"],
        "ocr_quality": document["ocr_quality"],
        "classification_conf": document["classification_conf"],
        "extraction_conf": document["extraction_conf"],
        "extracted_fields": document["extracted_fields"],
        "rejection": document.get("rejection_payload"),
        "superseded": document.get("superseded", False),
    }


@router.get("/documents/{doc_id}/annotated")
async def annotated_image(doc_id: str,
                          principal: Principal = Depends(current_principal)) -> FileResponse:
    """The annotated page image with the problem boxed (§11.8)."""
    from app.documents import annotator

    # Staff review other people's documents by definition, so the customer
    # ownership filter cannot apply to them. Customers stay scoped as before.
    staff = principal.role in ("agent", "manager")
    document = repo.get_document(doc_id, None if staff else principal.customer_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    payload = document.get("rejection_payload") or {}
    annotations = payload.get("annotations") or []
    path = annotator.ensure_rendered(doc_id, document["storage_key"], annotations)
    if path is None or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "No annotated image for this document")
    return FileResponse(str(path), media_type="image/png")


@router.post("/documents/{doc_id}/dispute")
async def dispute_document(doc_id: str,
                           principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """One-click dispute -> human review queue (§11.8)."""
    document = repo.get_document(doc_id, principal.customer_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    execute("UPDATE document SET status = 'NEEDS_REVIEW' WHERE id = ?", (doc_id,))
    execute(
        """INSERT INTO fraud_signal (id, claim_id, document_id, signal_type,
                                     explanation, severity, review_status, raised_at)
           VALUES (?,?,?, 'CUSTOMER_DISPUTE', ?, 0.0, 'PENDING', ?)""",
        (str(uuid.uuid4()), document["claim_id"], doc_id,
         "Customer disputed the automated verdict.",
         datetime.now(timezone.utc).isoformat()),
    )
    audit.record("document_disputed", actor_type="customer", actor_id=principal.customer_id,
                 entity_type="document", entity_id=doc_id,
                 payload={"previous_status": document["status"],
                          "previous_reason": document.get("rejection_code")})
    return {
        "doc_id": doc_id,
        "status": "NEEDS_REVIEW",
        "message": ("Thanks for flagging this — a colleague will review it personally "
                    "within 1-2 working days. I've kept your document on file."),
    }
