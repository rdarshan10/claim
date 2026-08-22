"""Cross-document consistency — does this claim's paperwork agree with itself?

Every document is validated on its own: is it readable, is it the right type,
is it signed. Nothing compared them *to each other*, so a police report in
another person's name, an invoice dated before the incident, or a repair bill
for triple the claimed amount all passed individually and reached a handler as
"4 documents verified".

The checks here are deliberately arithmetic and rule-based. A name that does
not match the policyholder is a fact, not an opinion, and a handler challenging
a customer about it needs to be able to say exactly what disagreed with what.
The model's job comes after: turning a set of findings into a short account of
what it likely means, which is judgement rather than fact.

Findings are advisory. Nothing here rejects a document or declines a claim.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from app.db import query, query_one


@dataclass
class Finding:
    code: str
    severity: str            # info | check | serious
    summary: str             # one line a handler can act on
    detail: str              # what disagreed with what, specifically
    document_ids: list[str]


# A name is "different" only beyond these: initials and short forms are normal
# on real paperwork and flagging them would bury the genuine mismatches.
def _name_matches(a: str, b: str) -> bool:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return True                      # nothing to compare is not a mismatch
    if a == b:
        return True
    pa, pb = a.replace(".", " ").split(), b.replace(".", " ").split()
    if not pa or not pb:
        return True
    # Surnames must agree; a first name may be an initial ("P Sharma" vs
    # "Priya Sharma" is the same person, "P Sharma" vs "Marcus Bennett" is not).
    if pa[-1] != pb[-1]:
        return False
    fa, fb = pa[0], pb[0]
    return fa == fb or (len(fa) == 1 and fb.startswith(fa)) or \
           (len(fb) == 1 and fa.startswith(fb))


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _as_money(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).replace("£", "").replace(",", "").strip()
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def check(claim_id: str) -> list[dict[str, Any]]:
    """Every disagreement between this claim's documents and its own record."""
    claim = query_one(
        """SELECT c.*, cu.full_name AS holder
           FROM claim c JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if claim is None:
        return []
    claim = dict(claim)

    docs = [dict(d) for d in query(
        "SELECT id, doc_type, filename, extracted_fields FROM document "
        "WHERE claim_id = ? AND extracted_fields != '{}'", (claim_id,))]

    incident = _as_date(claim.get("incident_date"))
    claimed = _as_money(claim.get("claimed_amount"))
    findings: list[Finding] = []

    for doc in docs:
        try:
            fields = json.loads(doc["extracted_fields"] or "{}")
        except json.JSONDecodeError:
            continue
        label = (doc["doc_type"] or "document").replace("_", " ")

        # 1. Whose name is on it?
        name = fields.get("name_on_document")
        if name and not _name_matches(name, claim["holder"]):
            findings.append(Finding(
                "NAME_MISMATCH", "serious",
                f"The {label} is in a different name",
                f"It names {name}, but the policyholder is {claim['holder']}. "
                f"Either it belongs to another claim, or a third party's "
                f"paperwork has been filed here.",
                [doc["id"]]))

        # 2. Dated before the thing it documents.
        doc_date = _as_date(fields.get("document_date"))
        if doc_date and incident:
            if doc_date < incident:
                findings.append(Finding(
                    "PREDATES_INCIDENT", "serious",
                    f"The {label} is dated before the incident",
                    f"Dated {doc_date.isoformat()}, but the incident was "
                    f"{incident.isoformat()} — {(incident - doc_date).days} days "
                    f"earlier. A document cannot describe something that had not "
                    f"happened.",
                    [doc["id"]]))
            elif (doc_date - incident).days > 365:
                findings.append(Finding(
                    "LONG_AFTER_INCIDENT", "check",
                    f"The {label} is dated over a year after the incident",
                    f"Dated {doc_date.isoformat()} against an incident on "
                    f"{incident.isoformat()}. Worth confirming it relates to "
                    f"this claim.",
                    [doc["id"]]))

        # 3. Does the money agree with what was claimed?
        amount = _as_money(fields.get("total_amount") or fields.get("amount"))
        if amount and claimed and claimed > 0:
            ratio = amount / claimed
            if ratio > 1.5 or ratio < 0.5:
                findings.append(Finding(
                    "AMOUNT_MISMATCH", "check",
                    f"The {label} total does not match the amount claimed",
                    f"The document shows £{amount:,.2f} against £{claimed:,.2f} "
                    f"claimed. One of the two needs correcting before assessment.",
                    [doc["id"]]))

    # 4. Do the documents agree with each other about the date of loss?
    dated = [(d, _as_date(json.loads(d["extracted_fields"] or "{}").get("document_date")))
             for d in docs]
    dated = [(d, dt) for d, dt in dated if dt]
    if len(dated) >= 2:
        spread = (max(dt for _, dt in dated) - min(dt for _, dt in dated)).days
        if spread > 180:
            findings.append(Finding(
                "DATES_INCONSISTENT", "check",
                "The documents span an implausible period",
                f"{spread} days between the earliest and latest document. For a "
                f"single incident that usually means paperwork from more than one "
                f"event has been filed together.",
                [d["id"] for d, _ in dated]))

    return [asdict(f) for f in findings]


def summarise(claim_id: str) -> dict[str, Any]:
    findings = check(claim_id)
    return {
        "claim_id": claim_id,
        "findings": findings,
        "serious": sum(1 for f in findings if f["severity"] == "serious"),
        "checks": sum(1 for f in findings if f["severity"] == "check"),
        "clean": not findings,
    }
