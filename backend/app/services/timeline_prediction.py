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


def _remaining_stages(status: str) -> list[str]:
    if status in TERMINAL:
        return []
    if status not in STAGE_ORDER:
        return STAGE_ORDER[1:]
    return STAGE_ORDER[STAGE_ORDER.index(status) :]


def predict(claim: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return predicted stages with date ranges and a confidence score."""
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
