"""Claim data access. Every method is scoped by ``customer_id``.

There is deliberately no method that fetches a claim without an owning customer,
so cross-customer access is impossible by construction (§17.1).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db import execute, query, query_one


def _row(row: Any) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_claims(customer_id: str) -> list[dict[str, Any]]:
    rows = query(
        """SELECT c.*, p.policy_number, p.product_type, p.coverage_limit
           FROM claim c JOIN policy p ON p.id = c.policy_id
           WHERE p.customer_id = ?
           ORDER BY c.filed_at DESC""",
        (customer_id,),
    )
    return [dict(row) for row in rows]


def get_claim(claim_id: str, customer_id: str) -> dict[str, Any] | None:
    return _row(
        query_one(
            """SELECT c.*, p.policy_number, p.product_type, p.coverage_limit, p.customer_id
               FROM claim c JOIN policy p ON p.id = c.policy_id
               WHERE c.id = ? AND p.customer_id = ?""",
            (claim_id, customer_id),
        )
    )


def find_claim_by_number(claim_number: str, customer_id: str) -> dict[str, Any] | None:
    return _row(
        query_one(
            """SELECT c.*, p.policy_number, p.product_type, p.coverage_limit, p.customer_id
               FROM claim c JOIN policy p ON p.id = c.policy_id
               WHERE UPPER(c.claim_number) = UPPER(?) AND p.customer_id = ?""",
            (claim_number, customer_id),
        )
    )


def get_status_history(claim_id: str, customer_id: str) -> list[dict[str, Any]]:
    rows = query(
        """SELECT h.* FROM claim_status_history h
           JOIN claim c ON c.id = h.claim_id
           JOIN policy p ON p.id = c.policy_id
           WHERE h.claim_id = ? AND p.customer_id = ?
           ORDER BY h.changed_at ASC""",
        (claim_id, customer_id),
    )
    return [dict(row) for row in rows]


def get_required_documents(claim_id: str, customer_id: str) -> list[dict[str, Any]]:
    rows = query(
        """SELECT r.* FROM required_document r
           JOIN claim c ON c.id = r.claim_id
           JOIN policy p ON p.id = c.policy_id
           WHERE r.claim_id = ? AND p.customer_id = ?
           ORDER BY r.mandatory DESC, r.doc_type ASC""",
        (claim_id, customer_id),
    )
    return [dict(row) for row in rows]


def get_documents(claim_id: str, customer_id: str) -> list[dict[str, Any]]:
    rows = query(
        """SELECT d.* FROM document d
           JOIN claim c ON c.id = d.claim_id
           JOIN policy p ON p.id = c.policy_id
           WHERE d.claim_id = ? AND p.customer_id = ?
           ORDER BY d.uploaded_at DESC""",
        (claim_id, customer_id),
    )
    out = []
    for row in rows:
        item = dict(row)
        item["extracted_fields"] = json.loads(item.get("extracted_fields") or "{}")
        if item.get("rejection_payload"):
            item["rejection_payload"] = json.loads(item["rejection_payload"])
        out.append(item)
    return _mark_superseded(out)


def _mark_superseded(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag rejections that no longer need the customer to do anything.

    If someone sends a valid driving licence and later an expired one, the
    requirement is still satisfied. Showing a bare rejection next to a complete
    checklist tells them two contradictory things about the same claim.
    """
    satisfied = {
        d.get("doc_type") for d in documents if d.get("status") == "VERIFIED"
    }
    for document in documents:
        document["superseded"] = bool(
            str(document.get("status", "")).startswith("REJECTED")
            and document.get("doc_type") in satisfied
        )
    return documents


def get_document(document_id: str, customer_id: str | None) -> dict[str, Any] | None:
    """One document with its claim context.

    ``customer_id`` of None skips the ownership filter and is for staff only:
    reviewers work other people's documents, so the filter that protects
    customers from each other would lock them out entirely.
    """
    scope = "" if customer_id is None else " AND p.customer_id = ?"
    params = (document_id,) if customer_id is None else (document_id, customer_id)
    row = query_one(
        f"""SELECT d.*, c.claim_number, c.incident_date, c.claim_type,
                   c.claimed_amount, p.coverage_limit,
                   cu.full_name AS policyholder_name
            FROM document d
            JOIN claim c ON c.id = d.claim_id
            JOIN policy p ON p.id = c.policy_id
            JOIN customer cu ON cu.id = p.customer_id
            WHERE d.id = ?{scope}""",
        params,
    )
    if row is None:
        return None
    item = dict(row)
    item["extracted_fields"] = json.loads(item.get("extracted_fields") or "{}")
    if item.get("rejection_payload"):
        item["rejection_payload"] = json.loads(item["rejection_payload"])

    # Is this rejection already covered by an accepted document of the same type?
    item["superseded"] = False
    if str(item.get("status", "")).startswith("REJECTED") and item.get("doc_type"):
        sibling = query_one(
            "SELECT id FROM document WHERE claim_id = ? AND doc_type = ? "
            "AND status = 'VERIFIED' AND id != ? LIMIT 1",
            (item["claim_id"], item["doc_type"], item["id"]),
        )
        item["superseded"] = sibling is not None
    return item


def set_status(claim_id: str, to_status: str, reason: str, actor_type: str = "system") -> None:
    current = query_one("SELECT status FROM claim WHERE id = ?", (claim_id,))
    from_status = current["status"] if current else None
    execute("UPDATE claim SET status = ? WHERE id = ?", (to_status, claim_id))
    execute(
        """INSERT INTO claim_status_history
           (id, claim_id, from_status, to_status, reason, actor_type, changed_at)
           VALUES (?,?,?,?,?,?,?)""",
        (str(uuid.uuid4()), claim_id, from_status, to_status, reason, actor_type,
         datetime.now(timezone.utc).isoformat()),
    )


def needs_replacing(document: dict[str, Any]) -> bool:
    """Is this a document the customer has to send again?

    Two shapes mean the same thing and both have to count. The seeder writes a
    ``REJECTED_*`` status directly. The live pipeline never does: a failed
    document is stored as ``NEEDS_REVIEW`` carrying ``recommendation: REJECT``,
    because the pipeline recommends and a handler decides (§11.6).

    Testing the status alone put every live rejection into IN_REVIEW, which told
    the customer "nothing for you to do" underneath a card explaining that their
    document was unusable.
    """
    if str(document.get("status", "")).startswith("REJECTED"):
        return True
    return (document.get("status") == "NEEDS_REVIEW"
            and (document.get("rejection_payload") or {}).get("recommendation") == "REJECT")


def checklist(claim_id: str, customer_id: str) -> dict[str, Any]:
    """Required-set minus verified-set, with per-item state (§11.3)."""
    required = get_required_documents(claim_id, customer_id)
    documents = get_documents(claim_id, customer_id)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for doc in documents:
        by_type.setdefault(doc.get("doc_type") or "unknown", []).append(doc)

    items = []
    for req in required:
        uploads = by_type.get(req["doc_type"], [])
        state = "MISSING"
        # Order matters: an acceptable upload outranks an earlier bad one, so a
        # customer who has already sent a good replacement is never asked again.
        if any(d["status"] == "VERIFIED" for d in uploads):
            state = "VERIFIED"
        elif any(d["status"] == "NEEDS_REVIEW" and not needs_replacing(d) for d in uploads):
            state = "IN_REVIEW"
        elif any(needs_replacing(d) for d in uploads):
            state = "REJECTED"
        elif uploads:
            state = "UPLOADED"
        items.append({
            "doc_type": req["doc_type"],
            "mandatory": bool(req["mandatory"]),
            "state": state,
            "document_id": uploads[0]["id"] if uploads else None,
        })

    # Anything not yet accepted still blocks the claim, so this drives the
    # claim's own progress...
    outstanding = [i["doc_type"] for i in items if i["mandatory"] and i["state"] != "VERIFIED"]
    # ...but only these are the customer's to act on. Something already with a
    # handler is our job, not theirs, and asking for it again is wrong.
    awaiting_customer = [i["doc_type"] for i in items
                         if i["mandatory"] and i["state"] in ("MISSING", "REJECTED")]
    with_us = [i["doc_type"] for i in items
               if i["mandatory"] and i["state"] in ("IN_REVIEW", "UPLOADED")]

    return {
        "claim_id": claim_id,
        "items": items,
        "outstanding_mandatory": outstanding,
        "awaiting_customer": awaiting_customer,
        "awaiting_customer_labels": [d.replace("_", " ") for d in awaiting_customer],
        "with_us": with_us,
        "with_us_labels": [d.replace("_", " ") for d in with_us],
        # Readable form for anything that puts these in front of a customer.
        # The model repeats what it is given, so handing it "damage_photos"
        # produced replies with database column names in them.
        "outstanding_labels": [d.replace("_", " ") for d in outstanding],
        "complete": not outstanding,
    }
