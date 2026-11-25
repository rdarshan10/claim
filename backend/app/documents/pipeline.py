"""Document pipeline (§11).

UPLOADED -> SCANNING -> OCR -> CLASSIFYING -> EXTRACTING -> VALIDATING ->
{VERIFIED | REJECTED_* | NEEDS_REVIEW}

The stages are nodes of a LangGraph graph (see "orchestration" below). It runs
in a background thread for the MVP; ``process`` is a drop-in Celery task body
(takes an id, writes its own state).
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query, query_one
from app.documents import rejection as rejection_builder
from app.documents import rules as R
from app.documents.ocr import OCRResult, get_adapter
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.ops import telemetry

GRAPH = "documents"

VALID_DOC_TYPES = [
    "police_report", "repair_invoice", "damage_photo", "driving_licence",
    "medical_report", "discharge_summary", "pharmacy_bill", "id_proof",
    "bank_statement", "claim_form", "other",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(doc_id: str, status: str, **fields: Any) -> None:
    assignments = ["status = ?"]
    params: list[Any] = [status]
    for key, value in fields.items():
        assignments.append(f"{key} = ?")
        params.append(value)
    params.append(doc_id)
    execute(f"UPDATE document SET {', '.join(assignments)} WHERE id = ?", params)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def scan(path: Path, raw: bytes) -> tuple[bool, str]:
    """Virus/format scan stand-in: magic bytes, size, emptiness (§17.1)."""
    settings = get_settings()
    if not raw:
        return False, "That file appears to be empty and I can't open it."
    if len(raw) > settings.max_upload_bytes:
        return False, f"That file is larger than the {settings.max_upload_bytes // 1048576}MB limit."

    # Reject active content / executables outright.
    if raw[:2] == b"MZ" or raw[:4] == b"\x7fELF":
        return False, "That file type can't be accepted for security reasons."
    if b"/JavaScript" in raw[:4096] or b"/Launch" in raw[:4096]:
        return False, "That file contains active content and can't be accepted."
    return True, ""


def classify(ocr: OCRResult, expected_types: list[str], trace_id: str) -> dict[str, Any]:
    """LLM classification with a deterministic keyword fallback."""
    fallback = json.dumps(_keyword_classify(ocr.text, expected_types))
    result = gateway.complete(
        "doc_classifier",
        {
            "expected_types": ", ".join(expected_types) or "any",
            "document_text": wrap_untrusted("document", ocr.text[:6000]),
        },
        tier="mini",
        trace_id=trace_id,
        fallback=fallback,
    )
    parsed = result.json(default={}) or {}
    doc_type = str(parsed.get("doc_type", "other")).strip().lower()
    if doc_type not in VALID_DOC_TYPES:
        doc_type = "other"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "doc_type": doc_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": str(parsed.get("rationale", ""))[:200],
        "source": "template" if result.degraded else result.model,
    }


def _keyword_classify(text: str, expected: list[str]) -> dict[str, Any]:
    """Deterministic fallback so classification works with no LLM."""
    lowered = (text or "").lower()
    signals = {
        "repair_invoice": ["invoice", "garage", "repair", "labour", "parts", "vat"],
        "police_report": ["police", "incident report", "officer", "constab", "crime ref"],
        "driving_licence": ["driving licence", "driver licence", "dvla", "licence no"],
        "medical_report": ["patient", "diagnosis", "clinician", "consultant", "hospital"],
        "discharge_summary": ["discharge", "admitted", "ward"],
        "pharmacy_bill": ["pharmacy", "prescription", "dispensed", "chemist"],
        "bank_statement": ["sort code", "account number", "statement", "balance"],
        "id_proof": ["passport", "identity card", "national id"],
        "claim_form": ["claim form", "declaration", "signature of claimant"],
    }
    scores = {
        doc_type: sum(1 for term in terms if term in lowered)
        for doc_type, terms in signals.items()
    }
    best = max(scores, key=lambda k: scores[k]) if scores else "other"
    hits = scores.get(best, 0)
    if hits == 0:
        return {"doc_type": "other", "confidence": 0.2, "rationale": "No strong keyword signal."}
    confidence = min(0.88, 0.5 + 0.12 * hits)
    return {"doc_type": best, "confidence": confidence,
            "rationale": f"Matched {hits} keyword signal(s) for {best}."}


def extract(ocr: OCRResult, doc_type: str, trace_id: str) -> dict[str, Any]:
    """LLM extraction, then deterministic re-parse of every value (§10)."""
    fallback = json.dumps(_regex_extract(ocr.text))
    result = gateway.complete(
        "doc_extractor",
        {"doc_type": doc_type,
         "document_text": wrap_untrusted("document", ocr.text[:6000])},
        tier="primary",
        trace_id=trace_id,
        fallback=fallback,
    )
    fields = result.json(default={}) or {}
    if not isinstance(fields, dict):
        fields = {}

    # Deterministic re-verification: a value the LLM invented that doesn't
    # actually appear in the document text is dropped (§10 field extraction).
    text_lower = (ocr.text or "").lower()
    verified: dict[str, Any] = {}
    for key in ("document_number", "issuer", "name_on_document"):
        value = fields.get(key)
        if value and str(value).strip().lower() in text_lower:
            verified[key] = str(value).strip()

    for key in ("document_date", "expiry_date"):
        parsed = R._parse_date(fields.get(key))
        if parsed:
            verified[key] = parsed.isoformat()

    amount = fields.get("amount")
    if amount is not None:
        try:
            verified["amount"] = float(str(amount).replace(",", "").replace("£", "").strip())
        except (TypeError, ValueError):
            pass

    # "Reporting Officer: PC 4471" is a header field, not a signature — matching
    # it meant an unsigned report passed VR-07.
    verified["has_signature_or_stamp"] = bool(fields.get("has_signature_or_stamp")) or bool(
        re.search(r"\bsigned\b|\bsignature\b|\bstamp(?:ed)?\b|\bauthorised by\b|"
                  r"\bcertified by\b", text_lower)
    )

    try:
        confidence = float(fields.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    dropped = [k for k in ("document_number", "issuer", "name_on_document")
               if fields.get(k) and k not in verified]
    if dropped:
        confidence *= 0.7  # the model produced values not present in the document

    verified["_confidence"] = round(max(0.0, min(1.0, confidence)), 3)
    verified["_source"] = "regex" if result.degraded else result.model
    verified["_dropped_unverifiable"] = dropped
    return verified


def _regex_extract(text: str) -> dict[str, Any]:
    """Deterministic fallback extractor."""
    out: dict[str, Any] = {}
    if m := re.search(r"(?:invoice|report|licence|ref(?:erence)?|no)\.?\s*[#:]?\s*"
                      r"([A-Z]{2,}[-/]?\d{3,}|\d{6,})", text or "", re.IGNORECASE):
        out["document_number"] = m.group(1)
    if m := re.search(r"(?:date|dated|issued)\s*[:\-]?\s*"
                      r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                      text or "", re.IGNORECASE):
        out["document_date"] = m.group(1)
    if m := re.search(r"(?:expiry|expires|valid until)\s*[:\-]?\s*"
                      r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
                      text or "", re.IGNORECASE):
        out["expiry_date"] = m.group(1)
    if m := re.search(r"(?:total|amount due|grand total)\s*[:\-]?\s*[£$€]?\s*"
                      r"([\d,]+\.\d{2})", text or "", re.IGNORECASE):
        out["amount"] = m.group(1)
    # The label is case-insensitive; the captured name is not, so we still only
    # take a properly capitalised name rather than the rest of the sentence.
    # Horizontal whitespace only — \s would run past the end of the line and
    # swallow the first word of the next one.
    if m := re.search(r"(?i:name|customer|patient|holder|billed to|"
                      r"signature of claimant)[ \t]*[:\-][ \t]*"
                      r"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z'\-]+){0,3})", text or ""):
        out["name_on_document"] = m.group(1).strip()

    if m := re.search(r"(?:from|issued by|garage|clinic|pharmacy|hospital)\s*[:\-]\s*"
                      r"([A-Z][\w&'\- ]{2,40})", text or "", re.IGNORECASE):
        out["issuer"] = m.group(1).strip()
    else:
        # Invoices, bills and reports put the issuing organisation on the first
        # line with no label at all.
        out.setdefault("issuer", _leading_organisation(text))
        if not out["issuer"]:
            out.pop("issuer")

    out["confidence"] = 0.55
    return out


DOCUMENT_TITLES = ("invoice", "claim form", "receipt", "statement", "report",
                   "summary", "licence", "license", "certificate")


def _leading_organisation(text: str) -> str:
    """First non-blank line, when it names an organisation rather than the form."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(title in line.lower() for title in DOCUMENT_TITLES):
            return ""  # it's the document's title, not who issued it
        return line[:60] if 2 < len(line) <= 60 else ""
    return ""


def find_duplicate(sha256: str, doc_id: str) -> dict[str, Any] | None:
    """VR-06 support: identical document already on file (§11.4)."""
    rows = query(
        """SELECT d.id, d.claim_id, c.claim_number
           FROM document d JOIN claim c ON c.id = d.claim_id
           WHERE d.sha256 = ? AND d.id != ?""",
        (sha256, doc_id),
    )
    if not rows:
        return None
    row = rows[0]
    current = query_one("SELECT claim_id FROM document WHERE id = ?", (doc_id,))
    return {
        "id": row["id"],
        "claim_number": row["claim_number"],
        "same_claim": bool(current and current["claim_id"] == row["claim_id"]),
    }


def raise_fraud_signal(claim_id: str, doc_id: str, signal_type: str,
                       explanation: str, severity: float) -> None:
    """Signals only, never decisions — always routed to a human (§5, UC-N2)."""
    execute(
        """INSERT INTO fraud_signal
           (id, claim_id, document_id, signal_type, explanation, severity, review_status, raised_at)
           VALUES (?,?,?,?,?,?, 'PENDING', ?)""",
        (str(uuid.uuid4()), claim_id, doc_id, signal_type, explanation, severity, _now()),
    )
    audit.record("fraud_signal_raised", entity_type="document", entity_id=doc_id,
                 payload={"signal_type": signal_type, "severity": severity,
                          "explanation": explanation})


# --------------------------------------------------------------------------
# orchestration — a LangGraph graph
# --------------------------------------------------------------------------
#
#   load ─► scan ─► ocr ─► classify ─► extract ─► validate ─► gate ─► verdict
#             │       │        │                                │        │
#             └───────┴────────┴────────────► finish ◄──────────┴────────┘
#
# Every stage may reach a verdict early (unreadable, illegible, unsure, wrong
# type, fraud signal, low confidence); ``finish`` is the single place a verdict
# is written, audited and the claim advanced. Not checkpointed: the OCR result
# carries word geometry, and a run is short and idempotent — re-running a
# document is the recovery path (the AIOps ``requeue_stuck_documents`` playbook).
class DocState(TypedDict, total=False):
    doc_id: str
    trace_id: str
    document: dict[str, Any]
    claim_id: str
    raw: bytes
    sha256: str
    ocr: Any
    expected: list[str]
    classification: dict[str, Any]
    doc_type: str
    fields: dict[str, Any]
    extraction_conf: float
    duplicate: dict[str, Any] | None
    results: list[Any]
    failures: list[Any]
    overall: float | None
    verdict: dict[str, Any] | None
    result: dict[str, Any]


def _verdict(status: str, payload: dict[str, Any] | None,
             overall: float | None = None) -> dict[str, Any]:
    return {"verdict": {"status": status, "payload": payload, "overall": overall}}


def _load(state: DocState) -> dict[str, Any]:
    doc_id = state["doc_id"]
    row = query_one(
        """SELECT d.*, c.claim_number, c.incident_date, c.claim_type, c.claimed_amount,
                  c.id AS cid, p.coverage_limit, cu.full_name AS policyholder_name
           FROM document d
           JOIN claim c ON c.id = d.claim_id
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id
           WHERE d.id = ?""",
        (doc_id,),
    )
    if row is None:
        return {"document": {}, "result": {"status": "ERROR", "error": "document not found"}}
    document = dict(row)
    audit.record("document_pipeline_started", entity_type="document", entity_id=doc_id,
                 payload={"claim_id": document["claim_id"]}, trace_id=state["trace_id"])
    return {"document": document, "claim_id": document["claim_id"]}


def _scan_node(state: DocState) -> dict[str, Any]:
    doc_id = state["doc_id"]
    _set_status(doc_id, "SCANNING")
    path = Path(state["document"]["storage_key"])
    raw = path.read_bytes() if path.exists() else b""
    ok, message = scan(path, raw)
    if ok:
        return {"raw": raw}
    return {"raw": raw, **_verdict("REJECTED_CORRUPT", {
        "doc_id": doc_id, "reason_code": "CORRUPT_FILE",
        "headline": "I couldn't open that file",
        "plain_explanation": message,
        "technical_detail": [message],
        "annotations": [], "can_dispute": True,
        "fix_steps": ["Try re-saving the file, or take a fresh photo.",
                      "Make sure the upload finished before closing the app.",
                      "Upload it again here."],
        "failed_rules": ["SCAN"],
    })}


def _ocr_node(state: DocState) -> dict[str, Any]:
    doc_id, document = state["doc_id"], state["document"]
    _set_status(doc_id, "OCR")
    ocr = get_adapter().read(Path(document["storage_key"]))
    sha256 = hashlib.sha256(state["raw"]).hexdigest()
    execute("UPDATE document SET sha256 = ?, ocr_quality = ? WHERE id = ?",
            (sha256, ocr.quality, doc_id))
    update: dict[str, Any] = {"ocr": ocr, "sha256": sha256, "raw": b""}

    if ocr.error or ocr.quality < get_settings().ocr_quality_floor:
        detail = ocr.error or (
            f"Only about {int(ocr.quality * 100)}% of the text came out clearly."
        )
        payload = rejection_builder.build(
            {**document, "doc_type": document.get("doc_type")},
            [R.RuleResult("OCR-01", False, R.REASON_ILLEGIBLE, detail,
                          {"quality": ocr.quality})],
            ocr, document, trace_id=state["trace_id"],
        )
        update.update(_verdict("REJECTED_QUALITY", payload))
    return update


def _classify_node(state: DocState) -> dict[str, Any]:
    doc_id, document, ocr = state["doc_id"], state["document"], state["ocr"]
    settings = get_settings()
    _set_status(doc_id, "CLASSIFYING")
    expected = [r["doc_type"] for r in query(
        "SELECT doc_type FROM required_document WHERE claim_id = ?", (state["claim_id"],))]
    classification = classify(ocr, expected, state["trace_id"])
    doc_type = classification["doc_type"]
    execute("UPDATE document SET doc_type = ?, classification_conf = ? WHERE id = ?",
            (doc_type, classification["confidence"], doc_id))
    update: dict[str, Any] = {"expected": expected, "classification": classification,
                              "doc_type": doc_type}

    if classification["confidence"] < settings.classification_floor:
        update.update(_verdict("NEEDS_REVIEW", {
            "doc_id": doc_id, "reason_code": "LOW_CONFIDENCE",
            "headline": "A specialist is checking this document",
            "plain_explanation": "I wasn't confident enough about this document to decide "
                                 "on my own, so I've passed it to a colleague. This usually "
                                 "takes 1-2 working days.",
            "technical_detail": [f"Classification confidence "
                                 f"{classification['confidence']:.2f} below "
                                 f"{settings.classification_floor}."],
            "annotations": [], "fix_steps": [], "can_dispute": False, "failed_rules": [],
        }))
    elif expected and doc_type not in expected and doc_type != "other":
        payload = rejection_builder.build(
            {**document, "doc_type": doc_type},
            [R.RuleResult("VR-00", False, R.REASON_WRONG_TYPE,
                          f"This looks like a {doc_type.replace('_', ' ')}, but this claim "
                          f"needs: {', '.join(t.replace('_', ' ') for t in expected)}.",
                          {"found": doc_type, "expected": expected})],
            ocr, document, trace_id=state["trace_id"],
        )
        update.update(_verdict("REJECTED_TYPE", payload))
    return update


def _extract_node(state: DocState) -> dict[str, Any]:
    doc_id = state["doc_id"]
    _set_status(doc_id, "EXTRACTING")
    fields = extract(state["ocr"], state["doc_type"], state["trace_id"])
    extraction_conf = float(fields.pop("_confidence", 0.5))
    fields.pop("_source", None)
    fields.pop("_dropped_unverifiable", [])
    execute("UPDATE document SET extracted_fields = ?, extraction_conf = ? WHERE id = ?",
            (json.dumps(fields), extraction_conf, doc_id))
    return {"fields": fields, "extraction_conf": extraction_conf}


def _validate_node(state: DocState) -> dict[str, Any]:
    """Deterministic rules — the model never decides whether a rule passed."""
    doc_id, document = state["doc_id"], state["document"]
    _set_status(doc_id, "VALIDATING")
    duplicate = find_duplicate(state["sha256"], doc_id)
    ctx = {
        "doc_type": state["doc_type"],
        "incident_date": document["incident_date"],
        "coverage_limit": document["coverage_limit"],
        "policyholder_name": document["policyholder_name"],
        "page_count": state["ocr"].page_count,
        "duplicate_of": duplicate,
    }
    results = R.run_all(state["fields"], ctx)
    for result in results:
        execute(
            """INSERT INTO document_validation (id, document_id, rule_id, passed, details, run_at)
               VALUES (?,?,?,?,?,?)""",
            (str(uuid.uuid4()), doc_id, result.rule_id, 1 if result.passed else 0,
             json.dumps({"message": result.message, "details": result.details}, default=str),
             _now()),
        )
    return {"duplicate": duplicate, "results": results,
            "failures": [r for r in results if not r.passed]}


def _gate_node(state: DocState) -> dict[str, Any]:
    """Confidence gate (§11.6) and fraud signals — to a human, never an accusation."""
    doc_id, claim_id = state["doc_id"], state["claim_id"]
    overall = round(0.3 * state["ocr"].quality + 0.3 * state["classification"]["confidence"]
                    + 0.4 * state["extraction_conf"], 3)
    duplicate = state.get("duplicate")
    if duplicate and not duplicate["same_claim"]:
        raise_fraud_signal(claim_id, doc_id, "DUPLICATE_ACROSS_CLAIMS",
                           f"Identical document also filed on claim "
                           f"{duplicate['claim_number']}.", 0.8)
        return {"overall": overall,
                **_verdict("NEEDS_REVIEW", {**_specialist_payload(doc_id),
                                            "recommendation": "REVIEW"}, overall)}

    date_failure = next((f for f in state["failures"] if f.details
                         and f.details.get("direction") == "before_incident"), None)
    if date_failure:
        raise_fraud_signal(claim_id, doc_id, "PRE_INCIDENT_DATE", date_failure.message, 0.6)

    if overall < get_settings().auto_verdict_floor:
        return {"overall": overall,
                **_verdict("NEEDS_REVIEW", _specialist_payload(doc_id), overall)}
    return {"overall": overall}


def _verdict_node(state: DocState) -> dict[str, Any]:
    """The pipeline recommends; a case handler decides (§11.6, §17.3).

    Nothing is accepted onto a claim without a person having looked — the AI
    does the reading and puts a recommendation in front of them.
    """
    doc_id, document, doc_type = state["doc_id"], state["document"], state["doc_type"]
    overall = state["overall"]
    if state["failures"]:
        payload = rejection_builder.build({**document, "doc_type": doc_type},
                                          state["failures"], state["ocr"], document,
                                          trace_id=state["trace_id"])
        payload["recommendation"] = "REJECT"
        return _verdict("NEEDS_REVIEW", payload, overall)

    execute("UPDATE required_document SET state = 'IN_REVIEW' "
            "WHERE claim_id = ? AND doc_type = ?", (state["claim_id"], doc_type))
    return _verdict("NEEDS_REVIEW", _recommend_accept(doc_id, doc_type), overall)


def _finish_node(state: DocState) -> dict[str, Any]:
    verdict = state["verdict"] or {}
    result = _finish(state["doc_id"], verdict["status"], state["trace_id"],
                     state["claim_id"], verdict.get("payload"), overall=verdict.get("overall"))
    telemetry.DOCUMENT_VERDICTS.labels(
        verdict["status"], (verdict.get("payload") or {}).get("reason_code") or "-").inc()
    return {"result": result}


def _next(default: str):
    def route(state: DocState) -> str:
        return "finish" if state.get("verdict") else default
    return route


def build_graph() -> Any:
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(DocState)
    stages = [("load", _load), ("scan", _scan_node), ("ocr", _ocr_node),
              ("classify", _classify_node), ("extract", _extract_node),
              ("validate", _validate_node), ("gate", _gate_node),
              ("verdict", _verdict_node), ("finish", _finish_node)]
    for name, fn in stages:
        g.add_node(name, telemetry.instrument(GRAPH, name, fn))

    g.add_edge(START, "load")
    g.add_conditional_edges("load", lambda s: END if s.get("result") else "scan",
                            ["scan", END])
    g.add_conditional_edges("scan", _next("ocr"), ["ocr", "finish"])
    g.add_conditional_edges("ocr", _next("classify"), ["classify", "finish"])
    g.add_conditional_edges("classify", _next("extract"), ["extract", "finish"])
    g.add_edge("extract", "validate")
    g.add_edge("validate", "gate")
    g.add_conditional_edges("gate", _next("verdict"), ["verdict", "finish"])
    g.add_edge("verdict", "finish")
    g.add_edge("finish", END)
    return g.compile()


@lru_cache(maxsize=1)
def compiled() -> Any:
    return build_graph()


def process(doc_id: str) -> dict[str, Any]:
    """Full pipeline for one document. Safe to call from a worker thread, and a
    drop-in Celery task body: it takes an id and writes its own state."""
    trace_id = str(uuid.uuid4())
    with telemetry.span("documents.process", **{"cc.trace_id": trace_id,
                                                "cc.document_id": doc_id}):
        final = compiled().invoke({"doc_id": doc_id, "trace_id": trace_id, "verdict": None},
                                  {"run_name": "document_pipeline",
                                   "metadata": {"doc_id": doc_id, "trace_id": trace_id}})
    telemetry.GRAPH_RUNS.labels(GRAPH, "ok" if final.get("result", {}).get("status") != "ERROR"
                                else "error").inc()
    return final["result"]


def _recommend_accept(doc_id: str, doc_type: str) -> dict[str, Any]:
    """What the customer sees while a handler checks a clean document."""
    label = (doc_type or "document").replace("_", " ")
    return {
        "doc_id": doc_id,
        "reason_code": "AWAITING_REVIEW",
        "recommendation": "ACCEPT",
        "headline": "Received — with our claims team",
        "plain_explanation": (
            f"Thanks — we've read your {label} and everything checks out. "
            f"One of our claims handlers is confirming it's what we need, and I'll "
            f"let you know here as soon as it's accepted. That's a check on the "
            f"document itself — the claim is assessed separately once we have "
            f"everything. You don't need to do anything."
        ),
        "fix_steps": [],
        "technical_detail": ["All automated checks passed."],
        "failed_rules": [],
        "can_dispute": False,
    }


def _specialist_payload(doc_id: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id, "reason_code": "NEEDS_REVIEW",
        "headline": "A specialist is taking a closer look",
        "plain_explanation": "We need a colleague to check this document before we can "
                             "continue. This usually takes 1-2 working days, and you don't "
                             "need to do anything right now.",
        "technical_detail": [], "annotations": [], "fix_steps": [],
        "can_dispute": False, "failed_rules": [],
    }


def _finish(doc_id: str, status: str, trace_id: str, claim_id: str,
            payload: dict[str, Any] | None, overall: float | None = None) -> dict[str, Any]:
    _set_status(
        doc_id, status,
        rejection_code=(payload or {}).get("reason_code"),
        rejection_payload=json.dumps(payload) if payload else None,
    )
    audit.record("document_verdict", entity_type="document", entity_id=doc_id,
                 payload={"status": status, "reason_code": (payload or {}).get("reason_code"),
                          "overall_confidence": overall,
                          "failed_rules": (payload or {}).get("failed_rules", [])},
                 trace_id=trace_id)
    _advance_claim(claim_id)
    return {"doc_id": doc_id, "status": status, "confidence": overall, "payload": payload}


def _advance_claim(claim_id: str) -> None:
    """A claim moves to assessment only when every mandatory doc is VERIFIED (§11.3)."""
    rows = query(
        """SELECT r.doc_type, r.mandatory,
                  (SELECT COUNT(*) FROM document d
                   WHERE d.claim_id = r.claim_id AND d.doc_type = r.doc_type
                     AND d.status = 'VERIFIED') AS verified
           FROM required_document r WHERE r.claim_id = ?""",
        (claim_id,),
    )
    outstanding = [r["doc_type"] for r in rows if r["mandatory"] and not r["verified"]]
    claim = query_one("SELECT status FROM claim WHERE id = ?", (claim_id,))
    if not claim:
        return

    from app.repositories import claims as claim_repo

    if not outstanding and claim["status"] in ("FILED", "DOCS_PENDING"):
        claim_repo.set_status(claim_id, "IN_ASSESSMENT",
                              "All mandatory documents verified", actor_type="system")
    elif outstanding and claim["status"] == "FILED":
        claim_repo.set_status(claim_id, "DOCS_PENDING",
                              f"Awaiting: {', '.join(outstanding)}", actor_type="system")
