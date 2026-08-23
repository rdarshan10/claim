"""First Notice of Loss — intake field spec and the state machine over it.

A real insurer does not create a claim the moment someone says "I crashed my
car". They take a *notification of loss*, check it, and only then register it on
the core system. This module owns the first half of that: what has to be
collected, what has already been answered, and what to ask next.

The assistant never writes to ``claim``. It fills a ``fnol_request`` and hands
back a reference; registration is a separate, human-gated step (§9, UC-N12).
"""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from app.db import execute, query, query_one

# --------------------------------------------------------------------------
# Field specification
# --------------------------------------------------------------------------
# `kind` drives how the client renders the question card, so a date is a date
# picker rather than a free-text box the customer can get wrong.


@dataclass
class Field:
    key: str
    question: str
    kind: str                       # choice | date | money | text | upload | confirm
    hint: str = ""
    options: list[dict[str, str]] = field(default_factory=list)
    claim_types: tuple[str, ...] = ()   # empty = applies to every claim type
    optional: bool = False
    quick: list[str] = field(default_factory=list)


COMMON: list[Field] = [
    Field(
        key="claim_type",
        question="What kind of claim would you like to open?",
        kind="choice",
        hint="Pick the one that fits best — we can always adjust it later.",
        options=[
            {"value": "motor", "label": "Motor", "icon": "🚗",
             "detail": "Car, van or motorbike"},
            {"value": "home", "label": "Home", "icon": "🏠",
             "detail": "Buildings or contents"},
            {"value": "health", "label": "Health", "icon": "🏥",
             "detail": "Treatment or medical costs"},
        ],
    ),
    Field(
        key="incident_date",
        question="When did it happen?",
        kind="date",
        hint="If you're not sure of the exact day, your best estimate is fine.",
        quick=["Today", "Yesterday"],
    ),
    Field(
        key="description",
        question="In your own words, what happened?",
        kind="text",
        hint="A couple of sentences is plenty — what happened, and where.",
    ),
]

BY_TYPE: list[Field] = [
    # --- motor ----------------------------------------------------------
    Field(
        key="vehicle_registration",
        question="What's the vehicle registration?",
        kind="text",
        hint="The number plate, e.g. AB12 CDE.",
        claim_types=("motor",),
    ),
    Field(
        key="incident_kind",
        question="What sort of incident was it?",
        kind="choice",
        claim_types=("motor",),
        options=[
            {"value": "collision", "label": "Collision", "icon": "💥"},
            {"value": "theft", "label": "Theft", "icon": "🔓"},
            {"value": "vandalism", "label": "Vandalism", "icon": "🔨"},
            {"value": "weather", "label": "Weather damage", "icon": "⛈️"},
            {"value": "other", "label": "Something else", "icon": "❓"},
        ],
    ),
    Field(
        key="third_party",
        question="Was anyone else involved?",
        kind="choice",
        hint="Another driver, a pedestrian, or someone else's property.",
        claim_types=("motor",),
        options=[
            {"value": "no", "label": "No, just me", "icon": "🙋"},
            {"value": "yes", "label": "Yes, someone else was involved", "icon": "👥"},
        ],
    ),
    Field(
        key="police_reference",
        question="Do you have a police reference number?",
        kind="text",
        hint="Only if the police attended or you reported it. Skip if not.",
        claim_types=("motor",),
        optional=True,
        quick=["I don't have one"],
    ),
    # --- home -----------------------------------------------------------
    Field(
        key="incident_kind",
        question="What caused the damage?",
        kind="choice",
        claim_types=("home",),
        options=[
            {"value": "escape_of_water", "label": "Escape of water", "icon": "💧"},
            {"value": "fire", "label": "Fire", "icon": "🔥"},
            {"value": "storm", "label": "Storm", "icon": "🌪️"},
            {"value": "theft", "label": "Theft or break-in", "icon": "🔓"},
            {"value": "other", "label": "Something else", "icon": "❓"},
        ],
    ),
    Field(
        key="property_address",
        question="Which address was affected?",
        kind="text",
        hint="The property covered by the policy.",
        claim_types=("home",),
    ),
    # --- health ---------------------------------------------------------
    Field(
        key="treatment_type",
        question="What kind of treatment is this for?",
        kind="choice",
        claim_types=("health",),
        options=[
            {"value": "consultation", "label": "Consultation", "icon": "🩺"},
            {"value": "procedure", "label": "Procedure or surgery", "icon": "🏥"},
            {"value": "diagnostic", "label": "Tests or scans", "icon": "🔬"},
            {"value": "emergency", "label": "Emergency care", "icon": "🚑"},
        ],
    ),
    Field(
        key="provider_name",
        question="Which hospital or clinic?",
        kind="text",
        claim_types=("health",),
    ),
]

CLOSING: list[Field] = [
    # Deliberately not asked: the customer is not asked to value their own loss
    # at first notification. Registration opens a nominal reserve from
    # DEFAULT_RESERVE, and the case handler sets the real figure when they
    # assess the claim — that is their decision to make, on the evidence, not
    # something to anchor on a number given before anyone has looked.
    Field(
        key="incident_report",
        question="Please upload anything you already have about the incident.",
        kind="upload",
        hint="A first incident report, photos of the damage, a crash report, "
             "a receipt or an invoice. You can add more later.",
        optional=True,
        quick=["I'll add these later"],
    ),
]

# The reserve the bot enters when the customer genuinely doesn't know. Real
# insurers open a nominal reserve rather than blocking registration on a figure
# the customer cannot yet give.
DEFAULT_RESERVE = {"motor": 1500.0, "home": 2500.0, "health": 800.0}

# Phrases that mean "I have no answer", not an answer of "no". A bare "no" is
# excluded deliberately: on a yes/no choice field it is a real answer, and
# treating it as a skip silently discarded it.
UNKNOWN_ANSWERS = {"i don't know yet", "i dont know yet", "i don't know", "i dont know",
                   "not sure", "unknown", "no idea",
                   "i don't have one", "i dont have one", "none", "n/a",
                   "i'll add these later", "ill add these later", "skip"}


def fields_for(claim_type: str | None) -> list[Field]:
    """Every field that applies, in the order they should be asked."""
    ordered = list(COMMON)
    if claim_type:
        ordered += [f for f in BY_TYPE if claim_type in f.claim_types]
        ordered += list(CLOSING)
    return ordered


def next_field(claim_type: str | None, answers: dict[str, Any]) -> Field | None:
    """The next thing to ask, or None when there is nothing mandatory left."""
    for spec in fields_for(claim_type):
        if spec.key in answers:
            continue
        return spec
    return None


def missing_mandatory(claim_type: str | None, answers: dict[str, Any]) -> list[str]:
    return [f.key for f in fields_for(claim_type)
            if not f.optional and f.key not in answers]


def field_by_key(claim_type: str | None, key: str) -> Field | None:
    for spec in fields_for(claim_type):
        if spec.key == key:
            return spec
    return None


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def normalise(spec: Field, raw: Any) -> Any:
    """Coerce a raw answer into the shape the core system expects.

    Returns ``None`` for "the customer declined to answer", which is recorded as
    an explicit skip rather than silently re-asked forever.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in UNKNOWN_ANSWERS and spec.kind != "choice":
        return None

    if spec.kind == "date":
        return _parse_date(text)
    if spec.kind == "money":
        cleaned = re.sub(r"[^\d.]", "", text)
        try:
            return round(float(cleaned), 2) if cleaned else None
        except ValueError:
            return None
    if spec.kind == "choice":
        lowered = text.lower()
        for option in spec.options:
            if lowered in (option["value"], option["label"].lower()):
                return option["value"]
        # Free text against a closed set: accept a clear substring match only.
        for option in spec.options:
            if option["value"] in lowered or option["label"].lower() in lowered:
                return option["value"]
        return None
    return text[:2000]


def _parse_date(text: str) -> str | None:
    lowered = text.lower().strip()
    today = date.today()
    if lowered in ("today", "just now", "this morning"):
        return today.isoformat()
    if lowered == "yesterday":
        return (today.fromordinal(today.toordinal() - 1)).isoformat()

    for pattern, order in (
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", "ymd"),
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "dmy"),
        (r"(\d{1,2})-(\d{1,2})-(\d{4})", "dmy"),
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        a, b, c = match.groups()
        y, m, d = (a, b, c) if order == "ymd" else (c, b, a)
        try:
            parsed = date(int(y), int(m), int(d))
        except ValueError:
            return None
        # A loss cannot be reported before it happens; a future date is almost
        # always a typo (2027 for 2026) and would fail core-system validation.
        return parsed.isoformat() if parsed <= today else None
    return None


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reference() -> str:
    return "FNOL-" + uuid.uuid4().hex[:6].upper()


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["answers"] = json.loads(data.get("answers") or "{}")
    data["asked"] = json.loads(data.get("asked") or "[]")
    return data


def create(customer_id: str, conversation_id: str = "") -> dict[str, Any]:
    fnol_id = str(uuid.uuid4())
    reference = _reference()
    now = _now()
    execute(
        """INSERT INTO fnol_request (id, reference, customer_id, conversation_id,
                                     status, answers, asked, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'COLLECTING', '{}', '[]', ?, ?)""",
        (fnol_id, reference, customer_id, conversation_id, now, now),
    )
    return get(fnol_id)  # type: ignore[return-value]


def get(fnol_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM fnol_request WHERE id = ?", (fnol_id,))
    return _row_to_dict(row) if row else None


def get_by_reference(reference: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM fnol_request WHERE reference = ?", (reference.upper(),))
    return _row_to_dict(row) if row else None


def open_for_customer(customer_id: str) -> dict[str, Any] | None:
    """The intake still being filled in, if there is one.

    Only COLLECTING counts: once submitted the customer is waiting on us, and
    mentioning a new incident should start a fresh notification.
    """
    row = query_one(
        """SELECT * FROM fnol_request WHERE customer_id = ? AND status = 'COLLECTING'
           ORDER BY created_at DESC LIMIT 1""",
        (customer_id,),
    )
    return _row_to_dict(row) if row else None


def list_for_customer(customer_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT * FROM fnol_request WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    )
    return [_row_to_dict(r) for r in rows]


def save_answer(fnol_id: str, key: str, value: Any) -> dict[str, Any]:
    record = get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)

    answers = record["answers"]
    asked = record["asked"]
    answers[key] = value
    if key not in asked:
        asked.append(key)

    claim_type = answers.get("claim_type") if key == "claim_type" else record["claim_type"]
    execute(
        """UPDATE fnol_request SET answers = ?, asked = ?, claim_type = ?, updated_at = ?
           WHERE id = ?""",
        (json.dumps(answers), json.dumps(asked), claim_type, _now(), fnol_id),
    )
    return get(fnol_id)  # type: ignore[return-value]


def set_status(fnol_id: str, status: str, *, reviewer: str | None = None,
               note: str | None = None, claim_id: str | None = None) -> None:
    execute(
        """UPDATE fnol_request
           SET status = ?,
               reviewer = COALESCE(?, reviewer),
               review_note = COALESCE(?, review_note),
               claim_id = COALESCE(?, claim_id),
               updated_at = ?
           WHERE id = ?""",
        (status, reviewer, note, claim_id, _now(), fnol_id),
    )


def attach_policy(fnol_id: str, policy_id: str) -> None:
    execute("UPDATE fnol_request SET policy_id = ?, updated_at = ? WHERE id = ?",
            (policy_id, _now(), fnol_id))


def documents(fnol_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT * FROM fnol_document WHERE fnol_id = ? ORDER BY uploaded_at",
        (fnol_id,),
    )
    return [dict(r) for r in rows]


def add_document(fnol_id: str, filename: str, storage_key: str,
                 content: str | None, doc_type: str | None = None) -> str:
    doc_id = str(uuid.uuid4())
    execute(
        """INSERT INTO fnol_document (id, fnol_id, doc_type, filename, storage_key,
                                      content, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, fnol_id, doc_type, filename, storage_key, content, _now()),
    )
    execute("UPDATE fnol_request SET updated_at = ? WHERE id = ?", (_now(), fnol_id))
    return doc_id


def for_staff(include_closed: bool = False) -> list[dict[str, Any]]:
    """The triage queue: everything a reviewer might need to act on."""
    clause = "" if include_closed else \
        "WHERE f.status IN ('SUBMITTED','UNDER_REVIEW','INFO_REQUIRED','READY_TO_REGISTER','REGISTERING')"
    rows = query(
        f"""SELECT f.*, c.full_name AS customer_name,
                   (SELECT COUNT(*) FROM fnol_document d WHERE d.fnol_id = f.id) AS doc_count
            FROM fnol_request f
            JOIN customer c ON c.id = f.customer_id
            {clause}
            ORDER BY CASE f.status
                       WHEN 'READY_TO_REGISTER' THEN 0
                       WHEN 'SUBMITTED' THEN 1
                       WHEN 'UNDER_REVIEW' THEN 2
                       ELSE 3 END,
                     f.created_at"""
    )
    return [_row_to_dict(r) for r in rows]


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
def summary(record: dict[str, Any]) -> list[dict[str, str]]:
    """Answers as ordered label/value pairs, for the review and confirm cards."""
    answers = record["answers"]
    out: list[dict[str, str]] = []
    for spec in fields_for(record.get("claim_type")):
        if spec.key not in answers:
            continue
        value = answers[spec.key]
        if value is None:
            display = "Not provided"
        elif spec.kind == "money":
            display = f"£{float(value):,.2f}"
        elif spec.kind == "choice":
            display = next((o["label"] for o in spec.options if o["value"] == value),
                           str(value))
        else:
            display = str(value)
        out.append({"key": spec.key, "label": spec.question.rstrip("?"), "value": display})
    return out


def question_card(record: dict[str, Any], spec: Field) -> dict[str, Any]:
    """The interactive card the client renders for one question."""
    total = len(fields_for(record.get("claim_type"))) or 1
    answered = len([k for k in record["answers"] if k in
                    {f.key for f in fields_for(record.get("claim_type"))}])
    return {
        "card_type": "fnol_question",
        "payload": {
            "fnol_id": record["id"],
            "reference": record["reference"],
            "field": spec.key,
            "kind": spec.kind,
            "question": spec.question,
            "hint": spec.hint,
            "options": spec.options,
            "quick": spec.quick,
            "optional": spec.optional,
            "progress": {"answered": answered, "total": total},
        },
    }


def review_card(record: dict[str, Any]) -> dict[str, Any]:
    """Everything collected, for the customer to check before submitting."""
    return {
        "card_type": "fnol_review",
        "payload": {
            "fnol_id": record["id"],
            "reference": record["reference"],
            "claim_type": record.get("claim_type"),
            "items": summary(record),
            "documents": [{"filename": d["filename"], "doc_type": d["doc_type"]}
                          for d in documents(record["id"])],
        },
    }


def receipt_card(record: dict[str, Any]) -> dict[str, Any]:
    """Confirmation after submitting: the reference and what happens next."""
    return {
        "card_type": "fnol_receipt",
        "payload": {
            "fnol_id": record["id"],
            "reference": record["reference"],
            "status": record["status"],
            "claim_type": record.get("claim_type"),
            "items": summary(record),
            "document_count": len(documents(record["id"])),
        },
    }


def status_card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_type": "fnol_status",
        "payload": {
            "fnol_id": record["id"],
            "reference": record["reference"],
            "status": record["status"],
            "claim_type": record.get("claim_type"),
            "review_note": record.get("review_note"),
            "claim_id": record.get("claim_id"),
            # The claim this became, so the customer can see the two references
            # are the same thing rather than guessing.
            "claim_number": _claim_number(record.get("claim_id")),
        },
    }


def _claim_number(claim_id: str | None) -> str | None:
    if not claim_id:
        return None
    row = query_one("SELECT claim_number FROM claim WHERE id = ?", (claim_id,))
    return row["claim_number"] if row else None
