"""Settlement timeline prediction (§21 feature #3).

Deterministic and explainable: per-stage dwell times are sampled from the same
lognormal parameters the synthetic history was generated with, so the prediction
is reproducible and auditable. No LLM is involved in producing a date.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

# Stage order and (median_days, sigma) dwell distributions per claim type.
STAGE_ORDER = [
    "FILED", "DOCS_PENDING", "IN_ASSESSMENT", "ADDITIONAL_INFO",
    "APPROVED", "PAYMENT_IN_PROGRESS", "SETTLED",
]

DWELL: dict[str, dict[str, tuple[float, float]]] = {
    "motor": {
        "FILED": (1.0, 0.4), "DOCS_PENDING": (4.0, 0.6), "IN_ASSESSMENT": (7.0, 0.5),
        "ADDITIONAL_INFO": (3.0, 0.7), "APPROVED": (1.0, 0.3),
        "PAYMENT_IN_PROGRESS": (4.0, 0.35),
    },
    "health": {
        "FILED": (1.0, 0.4), "DOCS_PENDING": (5.0, 0.6), "IN_ASSESSMENT": (9.0, 0.55),
        "ADDITIONAL_INFO": (4.0, 0.7), "APPROVED": (1.0, 0.3),
        "PAYMENT_IN_PROGRESS": (5.0, 0.4),
    },
    "home": {
        "FILED": (1.0, 0.4), "DOCS_PENDING": (5.0, 0.6), "IN_ASSESSMENT": (12.0, 0.6),
        "ADDITIONAL_INFO": (4.0, 0.7), "APPROVED": (2.0, 0.3),
        "PAYMENT_IN_PROGRESS": (5.0, 0.4),
    },
}

STAGE_MEANING = {
    "FILED": "We've received your claim and opened a file.",
    # Overridden by stage_meaning() when everything has actually been sent —
    # this wording only applies while the customer still owes us something.
    "DOCS_PENDING": "We're waiting for your documents before we can assess the claim.",
    "IN_ASSESSMENT": "Our team is reviewing your claim and the evidence you sent.",
    "ADDITIONAL_INFO": "We've asked for something extra before we can decide.",
    "APPROVED": "Your claim has been approved and is moving to payment.",
    "PAYMENT_IN_PROGRESS": "The payment has been raised and is on its way to your bank.",
    "SETTLED": "The claim is closed and paid.",
    "REJECTED": "The claim was not approved under your policy terms.",
    "WITHDRAWN": "The claim was withdrawn.",
}

TERMINAL = {"SETTLED", "REJECTED", "WITHDRAWN"}

# What DOCS_PENDING means once the customer has sent everything and it is
# sitting with a handler. Same database status, different person holding it.
DOCS_WITH_US = "You've sent everything we asked for — we're checking it now."


def stage_meaning(status: str, *, awaiting_customer: bool = True) -> str:
    """The customer-facing sentence for a stage.

    ``awaiting_customer`` distinguishes the two halves of DOCS_PENDING: False
    means the paperwork is in and the wait is on us, not them.
    """
    if status == "DOCS_PENDING" and not awaiting_customer:
        return DOCS_WITH_US
    return STAGE_MEANING.get(status, "")


# Stages every claim actually passes through. ADDITIONAL_INFO is an exception
# branch — most claims never enter it — so forecasting it for everyone
# overstated every timeline and implied we were about to ask for more.
HAPPY_PATH = [s for s in STAGE_ORDER if s != "ADDITIONAL_INFO"]


def _remaining_stages(status: str) -> list[str]:
    if status in TERMINAL:
        return []
    # A claim genuinely sitting in ADDITIONAL_INFO still has to clear it.
    order = STAGE_ORDER if status == "ADDITIONAL_INFO" else HAPPY_PATH
    if status not in order:
        return order[1:]
    return order[order.index(status):]


def predict(claim: dict[str, Any], history: list[dict[str, Any]] | None = None,
            *, awaiting_customer: bool = True) -> dict[str, Any]:
    """Return predicted stages with date ranges and a confidence score.

    ``awaiting_customer`` is False once every required document has been sent.
    Most of the DOCS_PENDING dwell is the customer gathering paperwork, so with
    it already in, that stage collapses to the handler check.
    """
    status = claim.get("status", "FILED")
    claim_type = (claim.get("claim_type") or "motor").lower()
    params = DWELL.get(claim_type, DWELL["motor"])

    if status in TERMINAL:
        return {
            "claim_id": claim.get("id"),
            "terminal": True,
            "status": status,
            "predicted_settlement_date": claim.get("settled_at"),
            "confidence": 1.0,
            "stages": [],
            "basis": "Claim has reached a final status; no prediction needed.",
        }

    cursor = date.today()
    stages: list[dict[str, Any]] = []
    variance_total = 0.0

    for stage in _remaining_stages(status):
        if stage == "SETTLED":
            continue
        median, sigma = params.get(stage, (3.0, 0.5))
        if stage == "DOCS_PENDING" and not awaiting_customer:
            # Everything is in; what remains is a handler signing it off, not
            # the customer chasing documents.
            median, sigma = 1.0, 0.35
        # Lognormal: median is exp(mu); p10/p90 bound the confidence band.
        low = median * math.exp(-1.2816 * sigma)
        high = median * math.exp(1.2816 * sigma)
        variance_total += sigma**2

        start = cursor
        cursor = cursor + timedelta(days=round(median))
        stages.append({
            "stage": stage,
            "meaning": STAGE_MEANING.get(stage, ""),
            "expected_start": start.isoformat(),
            "expected_end": cursor.isoformat(),
            "earliest": (start + timedelta(days=round(low))).isoformat(),
            "latest": (start + timedelta(days=round(high))).isoformat(),
        })

    # Confidence decays with the number of remaining stages and their spread.
    confidence = round(max(0.35, min(0.92, 0.95 - 0.10 * len(stages) - 0.15 * variance_total)), 2)
    band_days = max(2, round(2.0 * math.sqrt(variance_total) * 3))

    return {
        "claim_id": claim.get("id"),
        "terminal": False,
        "status": status,
        "predicted_settlement_date": cursor.isoformat(),
        "band_days": band_days,
        "confidence": confidence,
        "stages": stages,
        "basis": (
            f"Based on typical stage durations for {claim_type} claims "
            f"({len(stages)} stage(s) remaining)."
            + ("" if awaiting_customer
               else " Your documents are already in, so this is shorter than usual.")
        ),
    }


def is_overdue(claim: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Detect a claim sitting in one stage longer than typical (§16 modifier)."""
    if claim.get("status") in TERMINAL or not history:
        return None
    last = history[-1]
    try:
        changed = datetime.fromisoformat(last["changed_at"]).date()
    except (ValueError, TypeError, KeyError):
        return None

    days_in_stage = (date.today() - changed).days
    claim_type = (claim.get("claim_type") or "motor").lower()
    median, _ = DWELL.get(claim_type, DWELL["motor"]).get(claim.get("status", ""), (7.0, 0.5))

    if days_in_stage > median * 1.5:
        return {
            "days_in_stage": days_in_stage,
            "typical_days": round(median),
            "stage": claim.get("status"),
        }
    return None
