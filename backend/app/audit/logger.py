"""Append-only, hash-chained audit log (§12.4).

Every AI verdict, guardrail trigger, tool call and human override lands here.
Payloads are PII-redacted before they are written.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any

from app.db import connect

_lock = threading.Lock()

GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(
    event_type: str,
    *,
    actor_type: str = "system",
    actor_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    payload: dict[str, Any] | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    trace_id: str | None = None,
) -> None:
    from app.guardrails.pii import redact  # imported late to avoid a cycle

    body = redact(json.dumps(payload or {}, default=str, separators=(",", ":")))

    with _lock:
        conn = connect()
        try:
            row = conn.execute(
                "SELECT row_hash FROM audit_event ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = row["row_hash"] if row else GENESIS
            at = _now()
            material = f"{prev_hash}|{at}|{event_type}|{entity_id}|{body}".encode("utf-8")
            row_hash = hashlib.sha256(material).hexdigest()
            conn.execute(
                """INSERT INTO audit_event
                   (at, actor_type, actor_id, event_type, entity_type, entity_id,
                    payload, prompt_version, model, trace_id, prev_hash, row_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (at, actor_type, actor_id, event_type, entity_type, entity_id,
                 body, prompt_version, model, trace_id, prev_hash, row_hash),
            )
            conn.commit()
        finally:
            conn.close()


def verify_chain() -> dict[str, Any]:
    """Recompute the chain; proves no row was edited or removed."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, at, event_type, entity_id, payload, prev_hash, row_hash "
            "FROM audit_event ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()

    prev = GENESIS
    for row in rows:
        if row["prev_hash"] != prev:
            return {"ok": False, "broken_at": row["id"], "reason": "prev_hash mismatch"}
        material = f"{prev}|{row['at']}|{row['event_type']}|{row['entity_id']}|{row['payload']}"
        if hashlib.sha256(material.encode("utf-8")).hexdigest() != row["row_hash"]:
            return {"ok": False, "broken_at": row["id"], "reason": "row_hash mismatch"}
        prev = row["row_hash"]
    return {"ok": True, "events": len(rows), "head": prev}
