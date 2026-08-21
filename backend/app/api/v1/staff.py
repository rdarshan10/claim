"""Staff console APIs (§14.3). RBAC enforced on every route."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.agents import handoff
from app.api.deps import Principal, require_role
from app.audit import logger as audit
from app.db import execute, query, query_one
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.services import timeline_prediction

router = APIRouter(tags=["staff"])


@router.get("/staff/review-queue")
async def review_queue(
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    documents = query(
        """SELECT d.id, d.filename, d.doc_type, d.status, d.rejection_code,
                  d.ocr_quality, d.classification_conf, d.extraction_conf,
                  d.extracted_fields, d.rejection_payload, d.uploaded_at,
                  c.claim_number, c.claim_type, cu.full_name AS customer_name
           FROM document d
           JOIN claim c ON c.id = d.claim_id
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id
           WHERE d.status = 'NEEDS_REVIEW'
           ORDER BY d.uploaded_at ASC LIMIT 50"""
    )
    signals = query(
        """SELECT f.*, c.claim_number FROM fraud_signal f
           JOIN claim c ON c.id = f.claim_id
           WHERE f.review_status = 'PENDING'
           ORDER BY f.severity DESC, f.raised_at ASC LIMIT 50"""
    )

    def hydrate(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["extracted_fields"] = json.loads(item.get("extracted_fields") or "{}")
        if item.get("rejection_payload"):
            item["rejection_payload"] = json.loads(item["rejection_payload"])
        return item

    return {
        "documents": [hydrate(row) for row in documents],
        "fraud_signals": [dict(row) for row in signals],
    }


class DecisionRequest(BaseModel):
    verdict: str  # VERIFIED | REJECTED_RULES
    note: str = ""


@router.post("/staff/documents/{doc_id}/decision")
async def decide(doc_id: str, body: DecisionRequest,
                 principal: Principal = Depends(require_role("agent", "manager"))
                 ) -> dict[str, Any]:
    """Human override of an AI verdict — always recorded (§17.3)."""
    if body.verdict not in ("VERIFIED", "REJECTED_RULES", "NEEDS_REVIEW"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown verdict")

    document = query_one("SELECT * FROM document WHERE id = ?", (doc_id,))
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    previous = document["status"]
    execute("UPDATE document SET status = ? WHERE id = ?", (body.verdict, doc_id))
    if body.verdict == "VERIFIED":
        execute("UPDATE required_document SET state = 'VERIFIED' "
                "WHERE claim_id = ? AND doc_type = ?",
                (document["claim_id"], document["doc_type"]))

    execute("UPDATE fraud_signal SET review_status = 'REVIEWED' WHERE document_id = ?",
            (doc_id,))

    audit.record("human_override", actor_type=principal.role, actor_id=principal.name,
                 entity_type="document", entity_id=doc_id,
                 payload={"from": previous, "to": body.verdict, "note": body.note[:500],
                          "ai_verdict": previous,
                          "overturned": previous != body.verdict})

    from app.documents.pipeline import _advance_claim
    _advance_claim(document["claim_id"])

    # Close the loop in the chat: the customer asked a person to look, so the
    # answer arrives where they asked, not nowhere.
    owner = query_one(
        """SELECT p.customer_id FROM document d
           JOIN claim c ON c.id = d.claim_id
           JOIN policy p ON p.id = c.policy_id WHERE d.id = ?""",
        (doc_id,),
    )
    relayed = None
    if owner:
        relayed = handoff.notify_document_decision(
            owner["customer_id"], doc_id, body.verdict, principal.name, body.note
        )

    return {"doc_id": doc_id, "status": body.verdict, "previous_status": previous,
            "customer_notified": bool(relayed)}


@router.get("/staff/escalations")
async def escalations(
    include_resolved: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    where = "" if include_resolved else "WHERE e.status != 'RESOLVED'"
    rows = query(
        f"""SELECT e.*, cu.full_name AS customer_name, c.claim_number
            FROM escalation_ticket e
            JOIN customer cu ON cu.id = e.customer_id
            LEFT JOIN claim c ON c.id = e.claim_id
            {where}
            ORDER BY CASE e.priority WHEN 'URGENT' THEN 0 WHEN 'HIGH' THEN 1 ELSE 2 END,
                     e.created_at ASC"""
    )
    tickets = []
    for row in rows:
        item = dict(row)
        item["context_packet"] = json.loads(item.get("context_packet") or "{}")
        tickets.append(item)
    return {"tickets": tickets}


@router.post("/staff/escalations/{ticket_id}/claim")
async def claim_ticket(ticket_id: str,
                       principal: Principal = Depends(require_role("agent", "manager"))
                       ) -> dict[str, Any]:
    """Take the case. The assistant tells the customer, and shows them what it
    passed on — the customer keeps talking to the assistant throughout."""
    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    ticket = dict(row)

    execute("UPDATE escalation_ticket SET status = 'ASSIGNED', assigned_to = ? "
            "WHERE id = ?", (principal.name, ticket_id))
    audit.record("escalation_assigned", actor_type=principal.role,
                 actor_id=principal.name, entity_type="ticket", entity_id=ticket_id)

    # Phrased for the customer, not lifted from the internal packet.
    packet = json.loads(ticket.get("context_packet") or "{}")
    snapshot = packet.get("claim_snapshot") or {}
    carried: list[str] = ["Why you got in touch, and what you've already told me"]
    if snapshot.get("claim_number"):
        carried.append(
            f"Your claim {snapshot['claim_number']} and where it's got to "
            f"({str(snapshot.get('status', '')).replace('_', ' ').lower()})"
        )
    if outstanding := packet.get("documents_outstanding"):
        carried.append("The documents still outstanding: "
                       + ", ".join(d.replace("_", " ") for d in outstanding))
    carried.append("Everything I've already checked, so you don't repeat yourself")

    if ticket.get("conversation_id"):
        handoff.join(ticket["conversation_id"], principal.name, ticket_id,
                     carried=[c for c in carried if c])

    return {"ticket_id": ticket_id, "status": "ASSIGNED",
            "assigned_to": principal.name}


@router.get("/staff/escalations/{ticket_id}/case")
async def case_file(ticket_id: str,
                    principal: Principal = Depends(require_role("agent", "manager"))
                    ) -> dict[str, Any]:
    """Everything a reviewer needs in one payload: thread, claim, documents."""
    from app.repositories import claims as claim_repo
    from app.repositories import conversations as conv_repo

    ticket = query_one(
        """SELECT e.*, cu.full_name AS customer_name
           FROM escalation_ticket e JOIN customer cu ON cu.id = e.customer_id
           WHERE e.id = ?""",
        (ticket_id,),
    )
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")

    ticket_data = dict(ticket)
    ticket_data["context_packet"] = json.loads(ticket_data.get("context_packet") or "{}")

    # The conversation that raised the case, plus every other one this customer
    # has had — the problem is often described in an earlier chat.
    thread = (conv_repo.thread(ticket["conversation_id"])
              if ticket["conversation_id"] else [])
    all_conversations = conv_repo.all_threads_for_customer(ticket["customer_id"])
    for conversation in all_conversations:
        conversation["is_origin"] = conversation["id"] == ticket["conversation_id"]
        conversation["is_active"] = (
            conversation["id"] == conv_repo.delivery_target(
                ticket["customer_id"], ticket["conversation_id"])
        )

    claim = None
    documents: list[dict[str, Any]] = []
    checklist = None
    if ticket["claim_id"]:
        claim = claim_repo.get_claim(ticket["claim_id"], ticket["customer_id"])
        if claim:
            documents = claim_repo.get_documents(ticket["claim_id"],
                                                 ticket["customer_id"])
            checklist = claim_repo.checklist(ticket["claim_id"], ticket["customer_id"])
            claim["history"] = claim_repo.get_status_history(ticket["claim_id"],
                                                             ticket["customer_id"])
            claim["prediction"] = timeline_prediction.predict(claim, claim["history"])

    # Attach the verification evidence: every rule that ran, and the document
    # text the verdict was actually made from. A reviewer overruling the system
    # needs to see what it saw, not a summary of it.
    for document in documents:
        rules = query(
            "SELECT rule_id, passed, details, run_at FROM document_validation "
            "WHERE document_id = ? ORDER BY rule_id",
            (document["id"],),
        )
        document["validations"] = [
            {"rule_id": r["rule_id"], "passed": bool(r["passed"]),
             **json.loads(r["details"] or "{}")}
            for r in rules
        ]
        document["content"] = _document_text(document.get("storage_key"))

    signals = query(
        "SELECT * FROM fraud_signal WHERE claim_id = ? ORDER BY raised_at DESC",
        (ticket["claim_id"],),
    ) if ticket["claim_id"] else []

    return {
        "ticket": ticket_data,
        "thread": thread,
        "conversations": all_conversations,
        "claim": claim,
        "checklist": checklist,
        "documents": documents,
        "fraud_signals": [dict(row) for row in signals],
    }


def _document_text(storage_key: str | None, limit: int = 8000) -> str:
    """The document as the pipeline read it. Empty when it isn't text-readable."""
    if not storage_key:
        return ""
    from pathlib import Path

    path = Path(storage_key)
    if not path.exists() or path.suffix.lower() not in {".txt", ".md", ".csv", ".json"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


class ReplyRequest(BaseModel):
    note: str = Field(min_length=1, max_length=4000)
    force_verbatim: bool = False


@router.post("/staff/escalations/{ticket_id}/reply")
async def reply_to_customer(
    ticket_id: str, body: ReplyRequest,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """The reviewer answers. The assistant carries it back to the customer.

    The reviewer's note is the only source of fact; anything the assistant adds
    that isn't in it is discarded and the note is quoted verbatim instead.
    """
    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    ticket = dict(row)
    if not ticket["conversation_id"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "This ticket has no conversation to reply into")

    from app.repositories import conversations as conv_repo

    # Deliver to the thread the customer is using now, not necessarily the one
    # that raised the case — they may have started a new chat while waiting.
    target = conv_repo.delivery_target(ticket["customer_id"], ticket["conversation_id"])

    packet = json.loads(ticket.get("context_packet") or "{}")
    delivered = handoff.deliver_reviewer_response(
        target,
        note=body.note,
        agent_name=principal.name,
        sentiment=packet.get("customer_sentiment", "calm"),
        force_verbatim=body.force_verbatim,
    )
    execute("UPDATE escalation_ticket SET status = 'ANSWERED', assigned_to = ? "
            "WHERE id = ?", (principal.name, ticket_id))
    return delivered


@router.get("/staff/conversations/{conversation_id}/relay-log")
async def relay_log(conversation_id: str,
                    principal: Principal = Depends(require_role("agent", "manager"))
                    ) -> dict[str, Any]:
    """Side-by-side view of every relay: what the reviewer wrote vs what was sent.

    This is the accountability surface for the middleman design — if the
    assistant ever drifts from a reviewer's words, it is visible here and in the
    audit log, not buried.
    """
    from app.repositories import conversations as conv_repo

    relays = []
    for message in conv_repo.thread(conversation_id):
        if not message.get("source_note"):
            continue
        relays.append({
            "message_id": message["id"],
            "at": message["created_at"],
            "reviewer": message.get("author_name"),
            "reviewer_wrote": message["source_note"],
            "customer_received": message["content"],
            "rendered_by": message.get("relay_source"),
            "verbatim": message.get("relay_source") == "verbatim",
            "identical": message["source_note"].strip() in message["content"],
        })

    return {
        "conversation_id": conversation_id,
        "relays": relays,
        "count": len(relays),
        "verbatim_count": sum(1 for r in relays if r["verbatim"]),
    }


class AssistRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


SUGGESTED_QUESTIONS = [
    "What has this customer already been told?",
    "What's actually blocking this claim?",
    "Why was the document rejected, and is the rule right?",
    "Draft a reply I can edit.",
]


@router.post("/staff/escalations/{ticket_id}/assist")
async def reviewer_copilot(
    ticket_id: str, body: AssistRequest,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """The assistant working for the reviewer.

    Same grounding discipline as the customer side: answers come from the case
    file only, and it never decides anything — it surfaces evidence so the
    reviewer can.
    """
    case = await case_file(ticket_id, principal)

    # Trim to what a reviewer actually needs, and keep the prompt affordable.
    claim = case.get("claim") or {}
    compact = {
        "ticket": {
            "priority": case["ticket"]["priority"],
            "reason": case["ticket"]["reason"],
            "status": case["ticket"]["status"],
            "customer_sentiment":
                case["ticket"]["context_packet"].get("customer_sentiment"),
        },
        "claim": {
            "claim_number": claim.get("claim_number"),
            "claim_type": claim.get("claim_type"),
            "status": claim.get("status"),
            "claimed_amount": claim.get("claimed_amount"),
            "approved_amount": claim.get("approved_amount"),
            "incident_date": claim.get("incident_date"),
            "coverage_limit": claim.get("coverage_limit"),
            "history": [{"status": h["to_status"], "at": h["changed_at"][:10]}
                        for h in (claim.get("history") or [])],
            "prediction": claim.get("prediction"),
        },
        "checklist": case.get("checklist"),
        "documents": [
            {"doc_type": d.get("doc_type"), "status": d.get("status"),
             "rejection_code": d.get("rejection_code"),
             "ocr_quality": d.get("ocr_quality"),
             "classification_conf": d.get("classification_conf"),
             "extraction_conf": d.get("extraction_conf"),
             "extracted_fields": d.get("extracted_fields"),
             "failed_rules": (d.get("rejection_payload") or {}).get("failed_rules"),
             "technical_detail": (d.get("rejection_payload") or {}).get(
                 "technical_detail")}
            for d in case.get("documents", [])
        ],
        "fraud_signals": case.get("fraud_signals"),
        "conversation": [
            {"role": m["role"],
             "author": m.get("author_name"),
             "content": m["content"][:500]}
            for m in case.get("thread", [])
        ],
    }

    result = gateway.complete(
        "reviewer_copilot",
        {"case_file": json.dumps(compact, indent=2, default=str)[:14000],
         "question": wrap_untrusted("reviewer_question", body.question)},
        tier="primary",
        fallback=("I can't reach the model right now. The case file is on screen: "
                  "check the checklist for blockers and the failed rule IDs on any "
                  "rejected document."),
    )

    audit.record("reviewer_copilot_query", actor_type=principal.role,
                 actor_id=principal.name, entity_type="ticket", entity_id=ticket_id,
                 payload={"question": body.question[:300],
                          "degraded": result.degraded},
                 prompt_version=result.prompt_version, model=result.model)

    return {"answer": result.text.strip(), "model": result.model,
            "degraded": result.degraded, "suggestions": SUGGESTED_QUESTIONS}


class InfoRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)


@router.post("/staff/escalations/{ticket_id}/request-info")
async def request_info(
    ticket_id: str, body: InfoRequest,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Ask the customer for what's needed, phrased so they can act on it."""
    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    ticket = dict(row)
    if not ticket["conversation_id"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No conversation on this ticket")

    from app.repositories import conversations as conv_repo

    packet = json.loads(ticket.get("context_packet") or "{}")
    snapshot = packet.get("claim_snapshot") or {}
    target = conv_repo.delivery_target(ticket["customer_id"], ticket["conversation_id"])

    return handoff.request_information(
        target,
        request=body.request,
        agent_name=principal.name,
        claim_context={"claim_number": snapshot.get("claim_number"),
                       "claim_type": snapshot.get("claim_type"),
                       "status": snapshot.get("status"),
                       "incident_date": snapshot.get("incident_date")},
        sentiment=packet.get("customer_sentiment", "calm"),
    )


@router.post("/staff/escalations/{ticket_id}/close-duplicate")
async def close_duplicate(
    ticket_id: str,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Close a case that duplicates another open one for the same customer.

    Closes it silently — the customer is not told, because from their side
    nothing happened: their real case is still open and being worked.
    """
    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    ticket = dict(row)

    others = query(
        "SELECT id FROM escalation_ticket WHERE customer_id = ? AND id != ? "
        "AND status != 'RESOLVED' ORDER BY created_at ASC",
        (ticket["customer_id"], ticket_id),
    )
    if not others:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This is the customer's only open case — resolve it properly instead.",
        )

    execute("UPDATE escalation_ticket SET status = 'RESOLVED', resolved_at = ?, "
            "reason = reason || ' | closed as duplicate' WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), ticket_id))
    audit.record("escalation_closed_duplicate", actor_type=principal.role,
                 actor_id=principal.name, entity_type="ticket", entity_id=ticket_id,
                 payload={"kept_open": others[0]["id"], "silent": True})
    return {"ticket_id": ticket_id, "status": "RESOLVED",
            "kept_open": others[0]["id"], "customer_notified": False}


class ResolveRequest(BaseModel):
    closing_note: str = ""


@router.post("/staff/escalations/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str, body: ResolveRequest,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    row = query_one("SELECT * FROM escalation_ticket WHERE id = ?", (ticket_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    ticket = dict(row)

    from app.repositories import conversations as conv_repo

    packet = json.loads(ticket.get("context_packet") or "{}")
    target = conv_repo.delivery_target(ticket["customer_id"], ticket["conversation_id"])
    return handoff.resolve(
        target, principal.name, ticket_id,
        closing_note=body.closing_note,
        sentiment=packet.get("customer_sentiment", "calm"),
    )


@router.get("/staff/audit/events")
async def audit_events(entity_id: str = "", limit: int = 100,
                       principal: Principal = Depends(require_role("manager"))
                       ) -> dict[str, Any]:
    if entity_id:
        rows = query("SELECT * FROM audit_event WHERE entity_id = ? "
                     "ORDER BY id DESC LIMIT ?", (entity_id, min(limit, 500)))
    else:
        rows = query("SELECT * FROM audit_event ORDER BY id DESC LIMIT ?",
                     (min(limit, 500),))
    return {
        "events": [dict(row) for row in rows],
        "chain": audit.verify_chain(),
    }


@router.get("/admin/metrics/costs")
async def costs(principal: Principal = Depends(require_role("manager"))) -> dict[str, Any]:
    totals = query_one(
        "SELECT COUNT(*) AS calls, SUM(ok) AS ok_calls, AVG(latency_ms) AS avg_latency "
        "FROM llm_call"
    )
    return {"by_prompt": gateway.cost_summary(), "totals": dict(totals) if totals else {}}


@router.get("/admin/metrics/quality")
async def quality(principal: Principal = Depends(require_role("manager"))) -> dict[str, Any]:
    """Human-override rate is the key AI quality metric (§18)."""
    overrides = query(
        "SELECT payload FROM audit_event WHERE event_type = 'human_override'"
    )
    overturned = 0
    for row in overrides:
        try:
            if json.loads(row["payload"]).get("overturned"):
                overturned += 1
        except json.JSONDecodeError:
            continue

    verdicts = query_one(
        "SELECT COUNT(*) AS n FROM audit_event WHERE event_type = 'document_verdict'"
    )
    blocks = query_one(
        "SELECT COUNT(*) AS n FROM audit_event WHERE event_type = 'guardrail_block'"
    )
    fallbacks = query_one(
        "SELECT COUNT(*) AS n FROM audit_event "
        "WHERE event_type = 'guardrail_template_fallback'"
    )
    return {
        "document_verdicts": verdicts["n"] if verdicts else 0,
        "human_reviews": len(overrides),
        "ai_verdicts_overturned": overturned,
        "override_rate": round(overturned / len(overrides), 3) if overrides else None,
        "guardrail_blocks": blocks["n"] if blocks else 0,
        "template_fallbacks": fallbacks["n"] if fallbacks else 0,
    }
