"""Smart Rejection Explanation builder (§11.8) — the flagship feature.

The verdict comes from deterministic rules; the LLM only renders the *language*
around facts that are already fixed. If the LLM is unavailable, the templated
explanation below is used and the feature still works.
"""
from __future__ import annotations

import json
from typing import Any

from app.documents import rules as R
from app.documents.ocr import OCRResult, region_quality_map
from app.llm import gateway

# Fallback copy per reason code, used when the LLM is unavailable (UC-N7).
TEMPLATES: dict[str, dict[str, Any]] = {
    R.REASON_DATE_RANGE: {
        "headline": "The date on this document needs checking",
        "explanation": "The date on your document doesn't line up with your claim. "
                       "This is usually just a typo by whoever issued it.",
        "steps": ["Check this is the right document for this claim.",
                  "If the date is wrong, ask the issuer for a corrected copy.",
                  "Upload the corrected version here and I'll check it straight away."],
    },
    R.REASON_NAME_MISMATCH: {
        "headline": "The name on this document doesn't match your policy",
        "explanation": "The name we read on this document is different from the name on "
                       "your policy. Documents need to be in the policyholder's name.",
        "steps": ["Check you've uploaded the right document.",
                  "If the name is misspelled, ask the issuer to correct it.",
                  "If the document is genuinely in someone else's name, let me know and "
                  "I'll get a colleague to help."],
    },
    R.REASON_MISSING_FIELD: {
        "headline": "Some details are missing from this document",
        "explanation": "We couldn't find everything we need on this document.",
        "steps": ["Check the whole document is in the photo or file.",
                  "If a page is missing, add it and upload again.",
                  "Ask the issuer for a complete copy if the details really aren't there."],
    },
    R.REASON_ILLEGIBLE: {
        "headline": "This document is too hard to read",
        "explanation": "Parts of this document came out too unclear for us to read "
                       "reliably. I've highlighted the areas that gave us trouble.",
        "steps": ["Retake the photo in good, even light.",
                  "Hold the camera directly above the page so it isn't at an angle.",
                  "Make sure the whole page fits in the frame, then upload again."],
    },
    R.REASON_EXPIRED: {
        "headline": "This document has expired",
        "explanation": "The document you uploaded is past its expiry date, so we can't "
                       "accept it as valid.",
        "steps": ["Find the current, in-date version of this document.",
                  "Upload the new one here.",
                  "If you've applied for a renewal, let me know and I'll note it."],
    },
    R.REASON_MISSING_SIGNATURE: {
        "headline": "This document needs an official signature or stamp",
        "explanation": "This type of document has to be signed or stamped by the issuing "
                       "organisation before we can accept it.",
        "steps": ["Check whether the signature or stamp is on another page.",
                  "If not, ask the issuer for a signed copy.",
                  "Upload the signed version here."],
    },
    R.REASON_AMOUNT_INVALID: {
        "headline": "The amount on this document needs checking",
        "explanation": "We couldn't make sense of the amount on this document.",
        "steps": ["Check the total is clearly visible in the image.",
                  "Retake the photo if the figure is blurred or cut off.",
                  "Upload it again and I'll re-check it."],
    },
    R.REASON_INCOMPLETE_PAGES: {
        "headline": "Some pages are missing",
        "explanation": "This document should have more pages than we received.",
        "steps": ["Gather all the pages of the document.",
                  "Photograph or scan each page.",
                  "Upload them together here."],
    },
    R.REASON_WRONG_TYPE: {
        "headline": "This looks like a different kind of document",
        "explanation": "What you uploaded doesn't look like the document we're expecting "
                       "for this claim.",
        "steps": ["Check which file you picked.",
                  "Upload the document from your checklist instead.",
                  "If you think this is the right document, tap 'This looks wrong to me'."],
    },
    R.REASON_DUPLICATE: {
        "headline": "We've seen this document before",
        "explanation": "This document is identical to one already on file. A specialist "
                       "needs to take a closer look before we can continue.",
        "steps": ["No action needed from you right now.",
                  "A colleague will review this within 1-2 working days.",
                  "If you meant to upload a different document, you can add it here."],
    },
}


def primary_reason(failures: list[R.RuleResult]) -> str:
    """Pick the reason code that should drive the verdict.

    A duplicate outranks everything else: it routes to a human and may be a
    fraud signal, so it must not be masked by a name or field failure that
    happens to be checked earlier in rule order.
    """
    codes = [f.reason_code for f in failures if f.reason_code]
    if R.REASON_DUPLICATE in codes:
        return R.REASON_DUPLICATE
    return codes[0] if codes else R.REASON_MISSING_FIELD


def build_annotations(
    failures: list[R.RuleResult], ocr: OCRResult, reason_code: str
) -> list[dict[str, Any]]:
    """Locate the offending values on the page so the UI can draw boxes."""
    annotations: list[dict[str, Any]] = []

    if reason_code == R.REASON_ILLEGIBLE:
        for cell in region_quality_map(ocr):
            if cell["quality"] < 0.55:
                annotations.append({
                    "page": 1, "bbox": cell["bbox"],
                    "label": f"Hard to read (clarity {int(cell['quality'] * 100)}%)",
                    "severity": "error",
                })
        return annotations

    for failure in failures:
        if not failure.offending_value:
            continue
        boxes = ocr.find_word_boxes(failure.offending_value)
        if not boxes and failure.details:
            for value in failure.details.values():
                boxes = ocr.find_word_boxes(str(value))
                if boxes:
                    break
        for box in boxes[:3]:
            annotations.append({
                "page": 1,
                "bbox": list(box),
                "label": f"{failure.rule_id}: {failure.offending_value}"[:60],
                "severity": "error",
                "rule_id": failure.rule_id,
            })
    return annotations


def build_offline(
    document: dict[str, Any],
    failures: list[R.RuleResult],
    ocr: OCRResult,
) -> dict[str, Any]:
    """Template-only rejection payload — no LLM call.

    Used when seeding the demo dataset, where 65 documents would otherwise mean
    65 model calls for language we already have written down.
    """
    reason_code = primary_reason(failures)
    template = TEMPLATES.get(reason_code, TEMPLATES[R.REASON_MISSING_FIELD])

    return {
        "doc_id": document.get("id"),
        "reason_code": reason_code,
        "headline": template["headline"],
        "plain_explanation": template["explanation"],
        "technical_detail": [f.message for f in failures if not f.passed and f.message],
        "annotations": build_annotations(failures, ocr, reason_code),
        "fix_steps": template["steps"],
        "failed_rules": [f.rule_id for f in failures if not f.passed],
        "can_dispute": True,
        "explanation_source": "template",
        "prompt_version": "n/a",
    }


def build(
    document: dict[str, Any],
    failures: list[R.RuleResult],
    ocr: OCRResult,
    claim: dict[str, Any],
    *,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the SmartRejectionExplanation payload (§11.8)."""
    reason_code = primary_reason(failures)
    template = TEMPLATES.get(reason_code, TEMPLATES[R.REASON_MISSING_FIELD])

    rejection_facts = {
        "reason_code": reason_code,
        "failed_rules": [
            {"rule_id": f.rule_id, "message": f.message, "details": f.details}
            for f in failures if not f.passed
        ],
        "document_type": document.get("doc_type"),
    }

    fallback = json.dumps({
        "headline": template["headline"],
        "plain_explanation": template["explanation"],
        "fix_steps": template["steps"],
    })

    result = gateway.complete(
        "rejection_explainer",
        {
            "rejection_facts": json.dumps(rejection_facts, indent=2, default=str),
            "doc_type": document.get("doc_type") or "unknown",
            "claim_context": json.dumps({
                "claim_number": claim.get("claim_number"),
                "claim_type": claim.get("claim_type"),
                "incident_date": claim.get("incident_date"),
            }),
        },
        tier="primary",
        trace_id=trace_id,
        fallback=fallback,
    )

    explanation = result.json(default={}) or {}
    headline = explanation.get("headline") or template["headline"]
    plain = explanation.get("plain_explanation") or template["explanation"]
    steps = explanation.get("fix_steps") or template["steps"]
    if not isinstance(steps, list):
        steps = template["steps"]

    # The precise, deterministic reason always accompanies the friendly copy.
    detail_lines = [f.message for f in failures if not f.passed and f.message]

    return {
        "doc_id": document.get("id"),
        "reason_code": reason_code,
        "headline": headline,
        "plain_explanation": plain,
        "technical_detail": detail_lines,
        "annotations": build_annotations(failures, ocr, reason_code),
        "fix_steps": [str(s) for s in steps][:5],
        "failed_rules": [f.rule_id for f in failures if not f.passed],
        "can_dispute": True,
        "explanation_source": "template" if result.degraded else result.model,
        "prompt_version": result.prompt_version,
    }
