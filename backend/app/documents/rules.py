"""Deterministic validation rules VR-01..VR-08 (§11.4).

The LLM never decides pass/fail. It only supplies extracted values; these
functions are the sole authority on whether a document is acceptable, which is
what makes every verdict reproducible and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

# Rejection reasons are a closed enum (§11.5).
REASON_ILLEGIBLE = "ILLEGIBLE"
REASON_WRONG_TYPE = "WRONG_DOCUMENT_TYPE"
REASON_NAME_MISMATCH = "NAME_MISMATCH"
REASON_DATE_RANGE = "DATE_OUT_OF_RANGE"
REASON_MISSING_FIELD = "MISSING_FIELD"
REASON_EXPIRED = "EXPIRED_DOCUMENT"
REASON_INCOMPLETE_PAGES = "INCOMPLETE_PAGES"
REASON_MISSING_SIGNATURE = "MISSING_SIGNATURE"
REASON_AMOUNT_INVALID = "AMOUNT_INVALID"
REASON_DUPLICATE = "DUPLICATE_DOCUMENT"

MANDATORY_FIELDS: dict[str, list[str]] = {
    "repair_invoice": ["document_number", "document_date", "amount", "issuer"],
    "pharmacy_bill": ["document_date", "amount", "issuer"],
    "police_report": ["document_number", "document_date"],
    "medical_report": ["document_date", "issuer"],
    "discharge_summary": ["document_date", "issuer"],
    "driving_licence": ["document_number", "expiry_date", "name_on_document"],
    "id_proof": ["document_number", "name_on_document"],
    "bank_statement": ["document_date", "name_on_document"],
    "claim_form": ["document_date", "name_on_document"],
}

SIGNATURE_REQUIRED = {"police_report", "medical_report", "discharge_summary", "claim_form"}
EXPIRY_CHECKED = {"driving_licence", "id_proof"}

# VR-02 only applies where the document's date should relate to the incident.
# A driving licence issued in 2019 is not evidence of anything being wrong — it
# is simply an older document, and rejecting it would reject every valid licence.
DATE_RANGE_CHECKED = {
    "repair_invoice", "police_report", "medical_report", "discharge_summary",
    "pharmacy_bill", "claim_form", "damage_photo",
}
MIN_PAGES = {"claim_form": 1, "police_report": 1, "medical_report": 1}


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    reason_code: str | None = None
    message: str = ""
    details: dict[str, Any] | None = None
    offending_value: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z]+", (name or "").lower()) if len(t) > 1}


def token_set_ratio(a: str, b: str) -> float:
    """Fuzzy name match without rapidfuzz — Jaccard over name tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return round(100.0 * len(ta & tb) / len(ta | tb), 1)


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
def vr01_name_match(fields: dict, ctx: dict) -> RuleResult:
    """Extracted name fuzzy-matches the policyholder (ratio >= 85)."""
    on_doc = fields.get("name_on_document")
    holder = ctx.get("policyholder_name", "")
    if not on_doc:
        return RuleResult("VR-01", True, message="No name on document; rule not applicable.")

    ratio = token_set_ratio(str(on_doc), holder)
    if ratio >= 85:
        return RuleResult("VR-01", True, message=f"Name matches policyholder ({ratio}%).")
    return RuleResult(
        "VR-01", False, REASON_NAME_MISMATCH,
        f"The name on the document ('{on_doc}') doesn't match the policyholder.",
        {"ratio": ratio, "expected": holder, "found": on_doc}, str(on_doc),
    )


def vr02_date_in_range(fields: dict, ctx: dict) -> RuleResult:
    """Document date within [incident_date, incident_date + 90d].

    Only for documents whose date should relate to the incident — an identity
    document's issue date legitimately predates it.
    """
    doc_type = ctx.get("doc_type", "")
    if doc_type and doc_type not in DATE_RANGE_CHECKED:
        return RuleResult("VR-02", True,
                          message=f"Date range not applicable to a {doc_type.replace('_', ' ')}.")

    doc_date = _parse_date(fields.get("document_date"))
    incident = _parse_date(ctx.get("incident_date"))
    if doc_date is None:
        return RuleResult("VR-02", True, message="No date extracted; covered by VR-04.")
    if incident is None:
        return RuleResult("VR-02", True, message="No incident date on claim; rule skipped.")

    if doc_date < incident:
        return RuleResult(
            "VR-02", False, REASON_DATE_RANGE,
            f"The document is dated {doc_date.isoformat()}, which is before the "
            f"incident on {incident.isoformat()}.",
            {"document_date": doc_date.isoformat(), "incident_date": incident.isoformat(),
             "direction": "before_incident"},
            str(fields.get("document_date")),
        )
    if doc_date > incident + timedelta(days=90):
        return RuleResult(
            "VR-02", False, REASON_DATE_RANGE,
            f"The document is dated {doc_date.isoformat()}, more than 90 days after "
            f"the incident on {incident.isoformat()}.",
            {"document_date": doc_date.isoformat(), "incident_date": incident.isoformat(),
             "direction": "too_late"},
            str(fields.get("document_date")),
        )
    return RuleResult("VR-02", True, message="Document date is within the expected window.")


def vr03_amount_valid(fields: dict, ctx: dict) -> RuleResult:
    """Amount parses and is <= coverage limit x 1.2."""
    raw = fields.get("amount")
    if raw is None:
        return RuleResult("VR-03", True, message="No amount extracted; covered by VR-04.")
    try:
        amount = float(str(raw).replace(",", "").replace("£", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return RuleResult("VR-03", False, REASON_AMOUNT_INVALID,
                          f"The amount '{raw}' couldn't be read as a number.",
                          {"raw": str(raw)}, str(raw))

    if amount <= 0:
        return RuleResult("VR-03", False, REASON_AMOUNT_INVALID,
                          f"The amount on the document is {amount}, which isn't valid.",
                          {"amount": amount}, str(raw))

    limit = ctx.get("coverage_limit")
    if limit and amount > float(limit) * 1.2:
        return RuleResult(
            "VR-03", False, REASON_AMOUNT_INVALID,
            f"The amount {amount:.2f} is far above the policy's cover limit of {float(limit):.2f}.",
            {"amount": amount, "coverage_limit": float(limit)}, str(raw),
        )
    return RuleResult("VR-03", True, message=f"Amount {amount:.2f} is within cover.")


def vr04_mandatory_fields(fields: dict, ctx: dict) -> RuleResult:
    """Mandatory fields present for this document type."""
    doc_type = ctx.get("doc_type", "")
    required = MANDATORY_FIELDS.get(doc_type, [])
    missing = [f for f in required if not fields.get(f)]
    if not missing:
        return RuleResult("VR-04", True, message="All required fields were found.")

    pretty = {
        "document_number": "reference number", "document_date": "date",
        "amount": "total amount", "issuer": "issuing organisation",
        "name_on_document": "name", "expiry_date": "expiry date",
    }
    labels = [pretty.get(f, f) for f in missing]
    return RuleResult(
        "VR-04", False, REASON_MISSING_FIELD,
        f"We couldn't find the {', '.join(labels)} on this document.",
        {"missing": missing, "doc_type": doc_type}, ", ".join(labels),
    )


def vr05_not_expired(fields: dict, ctx: dict) -> RuleResult:
    """Document not expired (licence, ID)."""
    doc_type = ctx.get("doc_type", "")
    if doc_type not in EXPIRY_CHECKED:
        return RuleResult("VR-05", True, message="Expiry not applicable to this document type.")

    expiry = _parse_date(fields.get("expiry_date"))
    if expiry is None:
        return RuleResult("VR-05", True, message="No expiry date extracted; covered by VR-04.")
    if expiry < date.today():
        return RuleResult("VR-05", False, REASON_EXPIRED,
                          f"This document expired on {expiry.isoformat()}.",
                          {"expiry_date": expiry.isoformat()}, str(fields.get("expiry_date")))
    return RuleResult("VR-05", True, message=f"Valid until {expiry.isoformat()}.")


def vr06_not_duplicate(fields: dict, ctx: dict) -> RuleResult:
    """Duplicate detection across claims — raises a fraud signal, never an accusation."""
    duplicate_of = ctx.get("duplicate_of")
    if not duplicate_of:
        return RuleResult("VR-06", True, message="No duplicate found.")
    return RuleResult(
        "VR-06", False, REASON_DUPLICATE,
        "This document is identical to one already on file.",
        {"duplicate_document_id": duplicate_of.get("id"),
         "duplicate_claim_number": duplicate_of.get("claim_number"),
         "same_claim": duplicate_of.get("same_claim", False)},
        duplicate_of.get("claim_number"),
    )


def vr07_signature_present(fields: dict, ctx: dict) -> RuleResult:
    """Signature/stamp region present where required."""
    doc_type = ctx.get("doc_type", "")
    if doc_type not in SIGNATURE_REQUIRED:
        return RuleResult("VR-07", True, message="Signature not required for this type.")
    if fields.get("has_signature_or_stamp"):
        return RuleResult("VR-07", True, message="Signature or stamp found.")
    return RuleResult("VR-07", False, REASON_MISSING_SIGNATURE,
                      "This document needs an official signature or stamp, and we "
                      "couldn't find one.",
                      {"doc_type": doc_type}, None)


def vr08_page_count(fields: dict, ctx: dict) -> RuleResult:
    """Page count meets the minimum for this document type."""
    doc_type = ctx.get("doc_type", "")
    minimum = MIN_PAGES.get(doc_type, 1)
    pages = int(ctx.get("page_count", 1) or 1)
    if pages >= minimum:
        return RuleResult("VR-08", True, message=f"{pages} page(s) received.")
    return RuleResult("VR-08", False, REASON_INCOMPLETE_PAGES,
                      f"We only received {pages} page(s); this document needs at least {minimum}.",
                      {"pages": pages, "minimum": minimum}, str(pages))


ALL_RULES = [
    vr01_name_match, vr02_date_in_range, vr03_amount_valid, vr04_mandatory_fields,
    vr05_not_expired, vr06_not_duplicate, vr07_signature_present, vr08_page_count,
]


def run_all(fields: dict, ctx: dict) -> list[RuleResult]:
    return [rule(fields, ctx) for rule in ALL_RULES]
