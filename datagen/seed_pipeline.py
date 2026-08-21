"""Verify seeded documents offline, with no model calls.

The generator writes documents to disk, but a document sitting at ``UPLOADED``
with no ``doc_type`` is invisible to the checklist — so a freshly seeded demo
looked like nobody had ever uploaded anything.

Running the real pipeline over the corpus would mean two model calls per
document. Instead this runs the parts that are local and free:

    OCR (local) -> known doc_type (ground truth) -> regex extraction ->
    VR-01..VR-08 -> verdict -> template rejection payload with real annotations

The verdicts are therefore real rule outcomes on real OCR output, and the
annotations point at real coordinates. The one difference from a live upload is
that extraction uses the regex fallback rather than the model, so seeded verdicts
are slightly more conservative. ``evals/run_evals.py`` re-runs the full pipeline
when you want to measure the real thing.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db import connect
from app.documents import rejection as rejection_builder
from app.documents import rules as R
from app.documents.pipeline import _regex_extract
from app.documents.ocr import get_adapter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(fields: dict[str, Any]) -> dict[str, Any]:
    """Same deterministic re-parse the live pipeline applies to extracted values."""
    out: dict[str, Any] = {}
    for key in ("document_number", "issuer", "name_on_document"):
        if fields.get(key):
            out[key] = str(fields[key]).strip()
    for key in ("document_date", "expiry_date"):
        if parsed := R._parse_date(fields.get(key)):
            out[key] = parsed.isoformat()
    if fields.get("amount") is not None:
        try:
            out["amount"] = float(str(fields["amount"]).replace(",", "").strip())
        except (TypeError, ValueError):
            pass
    return out


def verify_seeded_documents(verbose: bool = True) -> dict[str, int]:
    """Run every unprocessed seeded document to a verdict."""
    settings = get_settings()
    adapter = get_adapter()
    conn = connect()
    cur = conn.cursor()

    rows = cur.execute(
        """SELECT d.id, d.storage_key, d.filename, d.claim_id,
                  c.incident_date, c.claim_number, p.coverage_limit,
                  cu.full_name AS policyholder_name
           FROM document d
           JOIN claim c ON c.id = d.claim_id
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id
           WHERE d.status = 'UPLOADED'"""
    ).fetchall()

    tally = {"verified": 0, "rejected": 0, "needs_review": 0, "unreadable": 0}
    seen_hashes: dict[str, tuple[str, str]] = {}

    for row in rows:
        doc_id = row["id"]
        path = Path(row["storage_key"])
        # The filename carries the ground-truth type, so no classifier call.
        doc_type = path.stem.split("_", 1)[-1].replace(".txt", "")
        if doc_type not in R.MANDATORY_FIELDS and doc_type != "damage_photo":
            doc_type = Path(row["filename"]).stem

        raw = path.read_bytes() if path.exists() else b""
        sha256 = hashlib.sha256(raw).hexdigest() if raw else ""
        ocr = adapter.read(path)

        cur.execute(
            "UPDATE document SET doc_type = ?, sha256 = ?, ocr_quality = ?, "
            "classification_conf = ? WHERE id = ?",
            (doc_type, sha256, ocr.quality, 0.95 if doc_type else 0.3, doc_id),
        )

        # --- unreadable ------------------------------------------------
        if ocr.error or ocr.quality < settings.ocr_quality_floor:
            failure = R.RuleResult(
                "OCR-01", False, R.REASON_ILLEGIBLE,
                ocr.error or f"Only about {int(ocr.quality * 100)}% of the text "
                             f"came out clearly.",
                {"quality": ocr.quality},
            )
            payload = rejection_builder.build_offline({"id": doc_id}, [failure], ocr)
            cur.execute(
                "UPDATE document SET status = 'REJECTED_QUALITY', rejection_code = ?, "
                "rejection_payload = ?, extraction_conf = 0 WHERE id = ?",
                (payload["reason_code"], json.dumps(payload), doc_id),
            )
            tally["unreadable"] += 1
            continue

        # --- extract + validate ----------------------------------------
        fields = _normalise(_regex_extract(ocr.text))
        # "Reporting Officer:" is a header field, not a signature.
        fields["has_signature_or_stamp"] = any(
            token in ocr.text.lower()
            for token in ("signed", "signature", "stamp", "authorised by",
                          "certified by")
        )

        duplicate = None
        if sha256 in seen_hashes:
            other_doc, other_claim = seen_hashes[sha256]
            duplicate = {"id": other_doc, "claim_number": other_claim,
                         "same_claim": other_claim == row["claim_number"]}
        else:
            seen_hashes[sha256] = (doc_id, row["claim_number"])

        ctx = {
            "doc_type": doc_type,
            "incident_date": row["incident_date"],
            "coverage_limit": row["coverage_limit"],
            "policyholder_name": row["policyholder_name"],
            "page_count": ocr.page_count,
            "duplicate_of": duplicate,
        }
        results = R.run_all(fields, ctx)
        for result in results:
            cur.execute(
                "INSERT INTO document_validation (id, document_id, rule_id, passed, "
                "details, run_at) VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), doc_id, result.rule_id, 1 if result.passed else 0,
                 json.dumps({"message": result.message, "details": result.details},
                            default=str),
                 _now()),
            )

        failures = [r for r in results if not r.passed]
        cur.execute("UPDATE document SET extracted_fields = ?, extraction_conf = ? "
                    "WHERE id = ?", (json.dumps(fields), 0.7, doc_id))

        if not failures:
            cur.execute("UPDATE document SET status = 'VERIFIED' WHERE id = ?", (doc_id,))
            cur.execute("UPDATE required_document SET state = 'VERIFIED' "
                        "WHERE claim_id = ? AND doc_type = ?", (row["claim_id"], doc_type))
            tally["verified"] += 1
            continue

        payload = rejection_builder.build_offline({"id": doc_id, "doc_type": doc_type},
                                                  failures, ocr)

        # Duplicates across claims are a signal for a human, never a rejection.
        if payload["reason_code"] == R.REASON_DUPLICATE and duplicate \
                and not duplicate["same_claim"]:
            cur.execute(
                "UPDATE document SET status = 'NEEDS_REVIEW', rejection_code = ?, "
                "rejection_payload = ? WHERE id = ?",
                (payload["reason_code"], json.dumps(payload), doc_id),
            )
            cur.execute(
                "INSERT INTO fraud_signal (id, claim_id, document_id, signal_type, "
                "explanation, severity, review_status, raised_at) "
                "VALUES (?,?,?, 'DUPLICATE_ACROSS_CLAIMS', ?, 0.8, 'PENDING', ?)",
                (str(uuid.uuid4()), row["claim_id"], doc_id,
                 f"Identical document also filed on claim {duplicate['claim_number']}.",
                 _now()),
            )
            tally["needs_review"] += 1
            continue

        cur.execute(
            "UPDATE document SET status = 'REJECTED_RULES', rejection_code = ?, "
            "rejection_payload = ? WHERE id = ?",
            (payload["reason_code"], json.dumps(payload), doc_id),
        )
        tally["rejected"] += 1

    # --- advance claims whose mandatory set is now complete ---------------
    claims = cur.execute("SELECT id, status FROM claim").fetchall()
    advanced = 0
    for claim in claims:
        outstanding = cur.execute(
            """SELECT COUNT(*) AS n FROM required_document r
               WHERE r.claim_id = ? AND r.mandatory = 1 AND r.state != 'VERIFIED'""",
            (claim["id"],),
        ).fetchone()["n"]
        if not outstanding and claim["status"] == "FILED":
            cur.execute("UPDATE claim SET status = 'IN_ASSESSMENT' WHERE id = ?",
                        (claim["id"],))
            advanced += 1
        elif outstanding and claim["status"] == "FILED":
            cur.execute("UPDATE claim SET status = 'DOCS_PENDING' WHERE id = ?",
                        (claim["id"],))

    conn.commit()
    conn.close()

    if verbose:
        print(f"  verified {tally['verified']} · rejected {tally['rejected']} · "
              f"needs review {tally['needs_review']} · unreadable {tally['unreadable']}")
    return tally
