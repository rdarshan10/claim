"""Staff console APIs (§14.3). RBAC enforced on every route."""
from __future__ import annotations

import json
import uuid
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
from app.repositories import claims as claim_repo
from app.repositories import conversations
from app.services import consistency, diary, timeline_prediction

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
    elif body.verdict == "REJECTED_RULES":
        # Back to outstanding: the customer has to send a replacement, and the
        # checklist is what tells them so.
        execute("UPDATE required_document SET state = 'MISSING' "
                "WHERE claim_id = ? AND doc_type = ? AND state = 'IN_REVIEW'",
                (document["claim_id"], document["doc_type"]))

    execute("UPDATE fraud_signal SET review_status = 'REVIEWED' WHERE document_id = ?",
            (doc_id,))

    # What the AI actually concluded is its recommendation, not the status it
    # parked the document at. The pipeline never signs a document off itself —
    # a clean one is left at NEEDS_REVIEW carrying "ACCEPT" for a handler to
    # confirm (§11.6) — so comparing statuses counted every single decision as
    # an override and pinned the rate at 100%, whichever way the handler ruled.
    stored = document["rejection_payload"]
    recommendation = (json.loads(stored).get("recommendation") if stored else None)
    ai_verdict = {"ACCEPT": "VERIFIED", "REJECT": "REJECTED_RULES"}.get(
        recommendation, previous)

    audit.record("human_override", actor_type=principal.role, actor_id=principal.name,
                 entity_type="document", entity_id=doc_id,
                 payload={"from": previous, "to": body.verdict, "note": body.note[:500],
                          "ai_verdict": ai_verdict,
                          "ai_recommendation": recommendation,
                          "overturned": body.verdict != ai_verdict})

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
        label = (document["doc_type"] or "document").replace("_", " ")
        headline = ("Your " + label + " has been accepted"
                    if body.verdict == "VERIFIED"
                    else "Your " + label + " needs another look")
        execute(
            """INSERT INTO notification (id, customer_id, claim_id, kind, channel,
                                         body, sent_at)
               VALUES (?,?,?,?, 'in_app', ?, ?)""",
            (str(uuid.uuid4()), owner["customer_id"], document["claim_id"],
             "document_accepted" if body.verdict == "VERIFIED" else "document_rejected",
             headline, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )

    return {"doc_id": doc_id, "status": body.verdict, "previous_status": previous,
            "customer_notified": bool(relayed)}


class AssessmentRequest(BaseModel):
    decision: str                                   # APPROVE | DECLINE | MORE_INFO
    amount: float | None = None
    note: str = Field(default="", max_length=2000)


# What one role may settle without a second signature. A handler can clear
# routine motor repairs; anything larger goes to a manager.
AUTHORITY_LIMIT = {"agent": 5000.0, "manager": 100000.0}


@router.get("/staff/assessments")
async def assessment_queue(
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Claims whose paperwork is complete and which now need a decision."""
    rows = query(
        """SELECT c.*, cu.full_name AS customer_name, p.policy_number,
                  p.coverage_limit
           FROM claim c
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id
           WHERE c.status IN ('IN_ASSESSMENT', 'ADDITIONAL_INFO')
           ORDER BY c.filed_at ASC LIMIT 50"""
    )
    claims = []
    for row in rows:
        item = dict(row)
        docs = query(
            """SELECT doc_type, status FROM document
               WHERE claim_id = ? AND status = 'VERIFIED'""",
            (item["id"],),
        )
        item["verified_documents"] = [d["doc_type"] for d in docs]
        signals = query(
            "SELECT signal_type, severity, explanation FROM fraud_signal WHERE claim_id = ?",
            (item["id"],),
        )
        item["fraud_signals"] = [dict(sig) for sig in signals]
        item["your_limit"] = AUTHORITY_LIMIT.get(principal.role, 0.0)
        claims.append(item)
    return {"claims": claims, "authority_limit": AUTHORITY_LIMIT.get(principal.role, 0.0)}


@router.post("/staff/claims/{claim_id}/assess")
async def assess(claim_id: str, body: AssessmentRequest,
                 principal: Principal = Depends(require_role("agent", "manager"))
                 ) -> dict[str, Any]:
    """Approve, decline, or ask for more on a claim.

    This is the decision the whole pipeline exists to support, and the one place
    money is committed — so it is gated on authority and always audited.
    """
    if body.decision not in ("APPROVE", "DECLINE", "MORE_INFO"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown decision")

    claim = query_one(
        """SELECT c.*, p.coverage_limit, p.customer_id
           FROM claim c JOIN policy p ON p.id = c.policy_id WHERE c.id = ?""",
        (claim_id,),
    )
    if claim is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    if claim["status"] not in ("IN_ASSESSMENT", "ADDITIONAL_INFO"):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"Claim is {claim['status']}, not awaiting assessment")
    if not body.note.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "A decision needs a reason — the customer sees it")

    if body.decision == "APPROVE":
        amount = float(body.amount or 0)
        if amount <= 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Approving a claim needs a settlement amount")
        limit = AUTHORITY_LIMIT.get(principal.role, 0.0)
        if amount > limit:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"£{amount:,.2f} is above your authority limit of £{limit:,.2f}. "
                f"A manager needs to approve this one.")
        if claim["coverage_limit"] and amount > claim["coverage_limit"]:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"£{amount:,.2f} exceeds the policy's cover of "
                f"£{claim['coverage_limit']:,.2f}")

        execute("UPDATE claim SET approved_amount = ? WHERE id = ?", (amount, claim_id))
        claim_repo.set_status(claim_id, "APPROVED",
                              f"Approved by {principal.name}: {body.note[:200]}",
                              actor_type=principal.role)
        body_text = (
            f"Good news — your claim **{claim['claim_number']}** has been approved "
            f"for **£{amount:,.2f}**.\n\n{body.note}\n\n"
            f"The payment will be arranged next, and I'll let you know here once "
            f"it's on its way."
        )
    elif body.decision == "DECLINE":
        claim_repo.set_status(claim_id, "REJECTED",
                              f"Declined by {principal.name}: {body.note[:200]}",
                              actor_type=principal.role)
        body_text = (
            f"I'm sorry — our claims team weren't able to approve claim "
            f"**{claim['claim_number']}**.\n\n{body.note}\n\n"
            f"If you think that's wrong you can ask us to look again — just say so "
            f"here and I'll pass it on."
        )
    else:
        claim_repo.set_status(claim_id, "ADDITIONAL_INFO",
                              f"More information requested by {principal.name}",
                              actor_type=principal.role)
        body_text = (
            f"Our claims team need a bit more before they can finish assessing "
            f"claim **{claim['claim_number']}**.\n\n{body.note}\n\n"
            f"Reply here and I'll pass it straight on."
        )

    audit.record("claim_assessed", actor_type=principal.role, actor_id=principal.name,
                 entity_type="claim", entity_id=claim_id,
                 payload={"decision": body.decision, "amount": body.amount,
                          "note": body.note[:500], "from_status": claim["status"]})

    # Told in the thread they already have open, and recorded as a notification
    # so it survives them not being in the chat at the time.
    conversation = query_one(
        """SELECT id FROM conversation WHERE customer_id = ?
           ORDER BY started_at DESC LIMIT 1""",
        (claim["customer_id"],),
    )
    if conversation:
        execute(
            """INSERT INTO message (id, conversation_id, role, content, created_at)
               VALUES (?,?, 'assistant', ?, ?)""",
            (str(uuid.uuid4()), conversation["id"], body_text,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    execute(
        """INSERT INTO notification (id, customer_id, claim_id, kind, channel, body, sent_at)
           VALUES (?,?,?,?, 'in_app', ?, ?)""",
        (str(uuid.uuid4()), claim["customer_id"], claim_id,
         f"claim_{body.decision.lower()}", body_text[:400],
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )

    return {"claim_id": claim_id, "decision": body.decision,
            "status": query_one("SELECT status FROM claim WHERE id = ?",
                                (claim_id,))["status"]}


@router.get("/staff/claims")
async def claims_book(
    mine: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Every live claim, with what it is actually waiting on.

    Cases only ever showed escalations, so a claim registered from a
    notification of loss existed nowhere in the staff console — the customer
    was told who was looking after it while no handler could see it. This is
    the list a handler works from.
    """
    rows = query(
        """SELECT c.id, c.claim_number, c.claim_type, c.status, c.handler,
                  c.claimed_amount, c.reserve_amount, c.approved_amount,
                  c.incident_date, c.filed_at,
                  cu.full_name AS customer_name, cu.id AS customer_id,
                  f.reference AS fnol_reference
           FROM claim c
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id
           LEFT JOIN fnol_request f ON f.claim_id = c.id
           WHERE c.status NOT IN ('SETTLED','REJECTED','WITHDRAWN')
           -- Oldest first: the claim that has been waiting longest is the one
           -- somebody should pick up, regardless of who shouted loudest.
           ORDER BY c.filed_at ASC"""
    )

    out = []
    for row in rows:
        item = dict(row)
        if mine and (item.get("handler") or "") != principal.name:
            continue
        # What the claim is blocked on, computed the same way the customer sees
        # it so the two views cannot disagree.
        checklist = claim_repo.checklist(item["id"], item["customer_id"])
        item["awaiting_customer"] = checklist["awaiting_customer_labels"]
        item["with_us"] = checklist["with_us_labels"]
        item["needs_attention"] = bool(checklist["with_us_labels"])
        item["unread_docs"] = query_one(
            "SELECT COUNT(*) AS n FROM document WHERE claim_id = ? "
            "AND status = 'NEEDS_REVIEW'", (item["id"],))["n"]
        out.append(item)

    return {"claims": out, "count": len(out)}


@router.post("/staff/claims/{claim_id}/take")
async def take_claim(claim_id: str,
                     principal: Principal = Depends(require_role("agent", "manager"))
                     ) -> dict[str, Any]:
    """Become the handler for a claim, or confirm you already are."""
    row = query_one("SELECT * FROM claim WHERE id = ?", (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")

    owner = dict(row).get("handler")
    if owner and owner != principal.name:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{owner} is already handling this claim. Refresh to see the current state.")
    if owner == principal.name:
        return {"claim_id": claim_id, "handler": owner, "already_yours": True}

    execute("UPDATE claim SET handler = ? WHERE id = ?", (principal.name, claim_id))
    audit.record("claim_handler_assigned", actor_type=principal.role,
                 actor_id=principal.name, entity_type="claim", entity_id=claim_id,
                 payload={"handler": principal.name})
    return {"claim_id": claim_id, "handler": principal.name}


@router.get("/staff/claims/{claim_id}/consistency")
async def claim_consistency(
    claim_id: str,
    explain: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Where this claim's documents disagree with each other or with the claim.

    Each document was already validated on its own, but nothing compared them,
    so a police report in someone else's name or an invoice predating the
    incident reached the handler as "verified". The findings are computed
    deterministically — a handler challenging a customer needs to say exactly
    what disagreed with what. ``explain=true`` adds a short read on what they
    likely mean together, which is judgement rather than fact.
    """
    row = query_one(
        """SELECT c.claim_number, c.claim_type, cu.full_name AS holder
           FROM claim c JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)

    result = consistency.summarise(claim_id)
    result["claim_number"] = claim["claim_number"]

    if explain and result["findings"]:
        lines = "\n".join(
            f"- [{f['severity']}] {f['summary']}: {f['detail']}"
            for f in result["findings"])
        completion = gateway.complete(
            "consistency_explainer",
            {"claim_number": claim["claim_number"], "claim_type": claim["claim_type"],
             "holder": claim["holder"], "findings": lines},
            tier="primary",
            # The findings are the value here; losing the interpretation is a
            # degradation, not a failure.
            fallback="",
        )
        result["interpretation"] = completion.text.strip() or None
        result["degraded"] = completion.degraded

    return result


# --------------------------------------------------------------------------
# Handler diary
# --------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    when: str                       # ISO date the claim comes back
    note: str | None = None


class ChaseRequest(BaseModel):
    message: str
    set_next_review: bool = True


@router.get("/staff/diary")
async def diary_view(
    mine: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """What needs attention today, most overdue first.

    This is how a claims handler actually works — a dated list of claims that
    come back to them — rather than a queue of documents. Claims predating the
    diary are backfilled on first view, dated from their last real movement so
    a long-stalled claim shows as long overdue rather than as new.
    """
    diary.backfill()
    handler = principal.name if mine else None
    return {"items": diary.due(handler), "summary": diary.summary(handler),
            "at_risk_days": diary.AT_RISK_DAYS,
            "max_automated_chases": diary.MAX_AUTOMATED_CHASES}


@router.post("/staff/claims/{claim_id}/review-date")
async def set_review_date(claim_id: str, body: ReviewRequest,
                          principal: Principal = Depends(require_role("agent", "manager"))
                          ) -> dict[str, Any]:
    """Push a claim out to a date, with a note saying what it is waiting for."""
    if query_one("SELECT id FROM claim WHERE id = ?", (claim_id,)) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    diary.set_review(claim_id, body.when, body.note)
    audit.record("claim_diarised", actor_type=principal.role, actor_id=principal.name,
                 entity_type="claim", entity_id=claim_id,
                 payload={"next_review_date": body.when, "note": body.note})
    return {"claim_id": claim_id, "next_review_date": body.when, "note": body.note}


@router.post("/staff/claims/{claim_id}/draft-chase")
async def draft_chase(claim_id: str,
                      principal: Principal = Depends(require_role("agent", "manager"))
                      ) -> dict[str, Any]:
    """Draft the chase message. The handler edits and sends it — we only save
    them writing the same message twenty times a day."""
    row = query_one(
        """SELECT c.*, cu.full_name AS customer_name, cu.id AS customer_id
           FROM claim c JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)

    checklist = claim_repo.checklist(claim_id, claim["customer_id"])
    waiting_on_customer = checklist["awaiting_customer_labels"]
    if not waiting_on_customer:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Nothing is outstanding with the customer — there is nothing to chase.")

    days_waiting = query_one(
        "SELECT CAST(julianday('now') - julianday(?) AS INT) AS d",
        (claim["filed_at"],))["d"]

    facts = {
        "customer_first_name": (claim["customer_name"] or "").split(" ")[0],
        "claim_number": claim["claim_number"],
        "claim_type": claim["claim_type"],
        "days_since_they_told_us": days_waiting,
        "waiting_on_customer": waiting_on_customer,
        "already_with_us": checklist["with_us_labels"],
        "times_already_chased": claim.get("chase_count") or 0,
        "handler_name": principal.name,
    }

    result = gateway.complete(
        "chase_drafter", {"facts": json.dumps(facts, indent=2)}, tier="primary",
        # If the model is unavailable the handler still gets a usable draft
        # rather than a dead button.
        fallback=(f"Hello {facts['customer_first_name']}, we're still waiting on a "
                  f"few things before we can move {claim['claim_number']} forward: "
                  f"{', '.join(waiting_on_customer)}. You can upload them in your "
                  f"claim page whenever you're ready, and we'll pick it straight up."),
    )
    return {"claim_id": claim_id, "draft": result.text.strip(),
            "waiting_on_customer": waiting_on_customer,
            "already_with_us": checklist["with_us_labels"],
            "times_already_chased": facts["times_already_chased"],
            "degraded": result.degraded}


@router.post("/staff/claims/{claim_id}/chase")
async def send_chase(claim_id: str, body: ChaseRequest,
                     principal: Principal = Depends(require_role("agent", "manager"))
                     ) -> dict[str, Any]:
    """Send the handler's chase to the customer and re-diarise the claim."""
    row = query_one(
        """SELECT c.*, cu.id AS customer_id FROM claim c
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)
    now = datetime.now(timezone.utc).isoformat()

    # Into the customer's live thread, so it lands where they are already
    # talking to us rather than in a separate inbox they never open.
    target = conversations.delivery_target(claim["customer_id"], None)
    if target:
        # "agent", not "assistant": these are the handler's own words, and the
        # thread must not present them as the AI speaking.
        conversations.add_message(
            target, "agent", body.message,
            author_name=principal.name, relay_source="handler_chase")

    execute("INSERT INTO notification (id, customer_id, claim_id, kind, channel, "
            "body, read, sent_at) VALUES (?,?,?,?,?,?,0,?)",
            (str(uuid.uuid4()), claim["customer_id"], claim_id, "documents_chased",
             "in_app", body.message, now))

    execute("UPDATE claim SET last_chased_at = ?, chase_count = COALESCE(chase_count, 0) + 1 "
            "WHERE id = ?", (now, claim_id))

    next_review = None
    if body.set_next_review:
        next_review = diary.next_date_for(claim["status"])
        if next_review:
            diary.set_review(claim_id, next_review,
                             f"Chased on {now[:10]} — waiting for the customer")

    audit.record("customer_chased", actor_type=principal.role, actor_id=principal.name,
                 entity_type="claim", entity_id=claim_id,
                 payload={"chase_count": (claim.get("chase_count") or 0) + 1,
                          "next_review_date": next_review})
    return {"claim_id": claim_id, "sent": True, "next_review_date": next_review,
            "delivered_to_conversation": target}


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
            -- Oldest first. Ordering by a mood-derived priority let a loud
            -- customer overtake someone who had simply been waiting longer.
            ORDER BY e.created_at ASC"""
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

    # Somebody else already has it. Their tab was stale — tell them who, rather
    # than moving the case and telling the customer twice that someone new is
    # on it. Re-taking your own case is a harmless no-op.
    owner = ticket.get("assigned_to")
    if owner and owner != principal.name:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{owner} already has this case. Refresh to see the current state.")
    if owner == principal.name:
        return {"ticket_id": ticket_id, "assigned_to": owner, "already_yours": True}

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


# --------------------------------------------------------------------------
# Admin overview
# --------------------------------------------------------------------------
# What a handled contact costs when a person does it. Published UK contact-centre
# benchmarks put a live-agent contact around £4-6; the conservative end is used
# so the saving is never overstated, and it is exposed as a parameter rather
# than baked in so anyone reading the number can challenge it.
USD_TO_GBP = 0.79

# The prompts a customer conversation triggers. Everything else in llm_call is
# per-claim work (document extraction, chase drafts, consistency checks) or a
# staff tool, and must not be charged against contact volume.
CONVERSATION_PROMPTS = {
    "classify_turn", "router", "sentiment", "empathy_responder",
    "knowledge_answer", "information_request",
}

# Customer messages in one contact. Used to convert a measured per-turn cost
# into a per-contact one.


# --------------------------------------------------------------------------
# Cost reporting
# --------------------------------------------------------------------------
# The prompts a customer conversation triggers. Everything else in llm_call is
# per-claim work (document extraction, chase drafts, consistency checks) or a
# staff tool, and must not be charged against contact volume.
CONVERSATION_PROMPTS = {
    "classify_turn", "router", "sentiment", "empathy_responder",
    "knowledge_answer", "information_request",
}

# Customer messages in one contact, used to turn a measured per-turn cost into
# a per-contact one. The only estimate on this page — everything else is
# metered — so it is shown alongside the figure it produces.
TURNS_PER_CONTACT = 6


@router.get("/staff/claims/{claim_id}/consistency")
async def claim_consistency(
    claim_id: str,
    explain: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """Where this claim's documents disagree with each other or with the claim.

    Each document was already validated on its own, but nothing compared them,
    so a police report in someone else's name or an invoice predating the
    incident reached the handler as "verified". The findings are computed
    deterministically — a handler challenging a customer needs to say exactly
    what disagreed with what. ``explain=true`` adds a short read on what they
    likely mean together, which is judgement rather than fact.
    """
    row = query_one(
        """SELECT c.claim_number, c.claim_type, cu.full_name AS holder
           FROM claim c JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)

    result = consistency.summarise(claim_id)
    result["claim_number"] = claim["claim_number"]

    if explain and result["findings"]:
        lines = "\n".join(
            f"- [{f['severity']}] {f['summary']}: {f['detail']}"
            for f in result["findings"])
        completion = gateway.complete(
            "consistency_explainer",
            {"claim_number": claim["claim_number"], "claim_type": claim["claim_type"],
             "holder": claim["holder"], "findings": lines},
            tier="primary",
            # The findings are the value here; losing the interpretation is a
            # degradation, not a failure.
            fallback="",
        )
        result["interpretation"] = completion.text.strip() or None
        result["degraded"] = completion.degraded

    return result


# --------------------------------------------------------------------------
# Handler diary
# --------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    when: str                       # ISO date the claim comes back
    note: str | None = None


class ChaseRequest(BaseModel):
    message: str
    set_next_review: bool = True


@router.get("/staff/diary")
async def diary_view(
    mine: bool = False,
    principal: Principal = Depends(require_role("agent", "manager")),
) -> dict[str, Any]:
    """What needs attention today, most overdue first.

    This is how a claims handler actually works — a dated list of claims that
    come back to them — rather than a queue of documents. Claims predating the
    diary are backfilled on first view, dated from their last real movement so
    a long-stalled claim shows as long overdue rather than as new.
    """
    diary.backfill()
    handler = principal.name if mine else None
    return {"items": diary.due(handler), "summary": diary.summary(handler),
            "at_risk_days": diary.AT_RISK_DAYS,
            "max_automated_chases": diary.MAX_AUTOMATED_CHASES}


@router.post("/staff/claims/{claim_id}/review-date")
async def set_review_date(claim_id: str, body: ReviewRequest,
                          principal: Principal = Depends(require_role("agent", "manager"))
                          ) -> dict[str, Any]:
    """Push a claim out to a date, with a note saying what it is waiting for."""
    if query_one("SELECT id FROM claim WHERE id = ?", (claim_id,)) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    diary.set_review(claim_id, body.when, body.note)
    audit.record("claim_diarised", actor_type=principal.role, actor_id=principal.name,
                 entity_type="claim", entity_id=claim_id,
                 payload={"next_review_date": body.when, "note": body.note})
    return {"claim_id": claim_id, "next_review_date": body.when, "note": body.note}


@router.post("/staff/claims/{claim_id}/draft-chase")
async def draft_chase(claim_id: str,
                      principal: Principal = Depends(require_role("agent", "manager"))
                      ) -> dict[str, Any]:
    """Draft the chase message. The handler edits and sends it — we only save
    them writing the same message twenty times a day."""
    row = query_one(
        """SELECT c.*, cu.full_name AS customer_name, cu.id AS customer_id
           FROM claim c JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)

    checklist = claim_repo.checklist(claim_id, claim["customer_id"])
    waiting_on_customer = checklist["awaiting_customer_labels"]
    if not waiting_on_customer:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Nothing is outstanding with the customer — there is nothing to chase.")

    days_waiting = query_one(
        "SELECT CAST(julianday('now') - julianday(?) AS INT) AS d",
        (claim["filed_at"],))["d"]

    facts = {
        "customer_first_name": (claim["customer_name"] or "").split(" ")[0],
        "claim_number": claim["claim_number"],
        "claim_type": claim["claim_type"],
        "days_since_they_told_us": days_waiting,
        "waiting_on_customer": waiting_on_customer,
        "already_with_us": checklist["with_us_labels"],
        "times_already_chased": claim.get("chase_count") or 0,
        "handler_name": principal.name,
    }

    result = gateway.complete(
        "chase_drafter", {"facts": json.dumps(facts, indent=2)}, tier="primary",
        # If the model is unavailable the handler still gets a usable draft
        # rather than a dead button.
        fallback=(f"Hello {facts['customer_first_name']}, we're still waiting on a "
                  f"few things before we can move {claim['claim_number']} forward: "
                  f"{', '.join(waiting_on_customer)}. You can upload them in your "
                  f"claim page whenever you're ready, and we'll pick it straight up."),
    )
    return {"claim_id": claim_id, "draft": result.text.strip(),
            "waiting_on_customer": waiting_on_customer,
            "already_with_us": checklist["with_us_labels"],
            "times_already_chased": facts["times_already_chased"],
            "degraded": result.degraded}


@router.post("/staff/claims/{claim_id}/chase")
async def send_chase(claim_id: str, body: ChaseRequest,
                     principal: Principal = Depends(require_role("agent", "manager"))
                     ) -> dict[str, Any]:
    """Send the handler's chase to the customer and re-diarise the claim."""
    row = query_one(
        """SELECT c.*, cu.id AS customer_id FROM claim c
           JOIN policy p ON p.id = c.policy_id
           JOIN customer cu ON cu.id = p.customer_id WHERE c.id = ?""",
        (claim_id,))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found")
    claim = dict(row)
    now = datetime.now(timezone.utc).isoformat()

    # Into the customer's live thread, so it lands where they are already
    # talking to us rather than in a separate inbox they never open.
    target = conversations.delivery_target(claim["customer_id"], None)
    if target:
        # "agent", not "assistant": these are the handler's own words, and the
        # thread must not present them as the AI speaking.
        conversations.add_message(
            target, "agent", body.message,
            author_name=principal.name, relay_source="handler_chase")

    execute("INSERT INTO notification (id, customer_id, claim_id, kind, channel, "
            "body, read, sent_at) VALUES (?,?,?,?,?,?,0,?)",
            (str(uuid.uuid4()), claim["customer_id"], claim_id, "documents_chased",
             "in_app", body.message, now))

    execute("UPDATE claim SET last_chased_at = ?, chase_count = COALESCE(chase_count, 0) + 1 "
            "WHERE id = ?", (now, claim_id))

    next_review = None
    if body.set_next_review:
        next_review = diary.next_date_for(claim["status"])
        if next_review:
            diary.set_review(claim_id, next_review,
                             f"Chased on {now[:10]} — waiting for the customer")

    audit.record("customer_chased", actor_type=principal.role, actor_id=principal.name,
                 entity_type="claim", entity_id=claim_id,
                 payload={"chase_count": (claim.get("chase_count") or 0) + 1,
                          "next_review_date": next_review})
    return {"claim_id": claim_id, "sent": True, "next_review_date": next_review,
            "delivered_to_conversation": target}


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
            -- Oldest first. Ordering by a mood-derived priority let a loud
            -- customer overtake someone who had simply been waiting longer.
            ORDER BY e.created_at ASC"""
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

    # Somebody else already has it. Their tab was stale — tell them who, rather
    # than moving the case and telling the customer twice that someone new is
    # on it. Re-taking your own case is a harmless no-op.
    owner = ticket.get("assigned_to")
    if owner and owner != principal.name:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{owner} already has this case. Refresh to see the current state.")
    if owner == principal.name:
        return {"ticket_id": ticket_id, "assigned_to": owner, "already_yours": True}

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


# --------------------------------------------------------------------------
# Admin overview
# --------------------------------------------------------------------------
# What a handled contact costs when a person does it. Published UK contact-centre
# benchmarks put a live-agent contact around £4-6; the conservative end is used
# so the saving is never overstated, and it is exposed as a parameter rather
# than baked in so anyone reading the number can challenge it.
USD_TO_GBP = 0.79

# The prompts a customer conversation triggers. Everything else in llm_call is
# per-claim work (document extraction, chase drafts, consistency checks) or a
# staff tool, and must not be charged against contact volume.
CONVERSATION_PROMPTS = {
    "classify_turn", "router", "sentiment", "empathy_responder",
    "knowledge_answer", "information_request",
}

# Customer messages in one contact. Used to convert a measured per-turn cost
# into a per-contact one.


# --------------------------------------------------------------------------
# Savings assumptions
@router.get("/admin/overview")
async def admin_overview(
    annual_contacts: int = 250_000,
    principal: Principal = Depends(require_role("manager")),
) -> dict[str, Any]:
    """What the AI costs to run, measured.

    Every figure here comes from a metered call priced at the provider's
    published rate. There is deliberately no savings case: that would rest on
    industry benchmarks rather than this system's own data, and a projection is
    the first thing a sceptical reader discounts. What the AI did is reported
    as counts, not as money.
    """
    summary = gateway.cost_summary()

    live = [r for r in summary if not r["cached"]]
    hits = [r for r in summary if r["cached"]]
    spend_usd = sum(r["usd"] for r in live)
    avoided_usd = sum(r["usd"] for r in hits)

    calls = sum(int(r["calls"]) for r in live)
    cache_hits = sum(int(r["calls"]) for r in hits)
    failed = sum(int(r["calls"]) - int(r["ok_calls"] or 0) for r in live)

    gbp = lambda usd: round(usd * USD_TO_GBP, 4)  # noqa: E731

    # Cost per contact, from the prompts a conversation actually triggers.
    # Document extraction, chase drafting and consistency checks are per-claim
    # work and would inflate this several times over if included.
    per_conversation = sum(
        r["usd"] for r in live if r["prompt_key"] in CONVERSATION_PROMPTS)
    turns = query_one("SELECT COUNT(*) AS n FROM message WHERE role = 'user'")["n"]
    usd_per_turn = per_conversation / max(turns, 1)
    usd_per_contact = usd_per_turn * TURNS_PER_CONTACT

    # What the system did. Counts, not money — each is a row someone can find.
    convs = query_one("SELECT COUNT(*) AS n FROM conversation")["n"]
    escalated = query_one(
        "SELECT COUNT(DISTINCT customer_id) AS n FROM escalation_ticket")["n"]
    registered = query_one(
        "SELECT COUNT(*) AS n FROM fnol_request WHERE claim_id IS NOT NULL")["n"]
    rpa_done = query_one(
        "SELECT COUNT(*) AS n FROM rpa_run WHERE status = 'SUCCEEDED'")["n"]
    chases = query_one(
        "SELECT COUNT(*) AS n FROM audit_event WHERE event_type = 'customer_chased'")["n"]
    documents = query_one(
        "SELECT COUNT(*) AS n FROM document WHERE status != 'MISSING'")["n"]

    return {
        "spend": {
            "llm_calls": calls,
            "failed_calls": failed,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / (calls + cache_hits), 3)
                              if (calls + cache_hits) else None,
            "tokens_in": sum(int(r["tokens_in"] or 0) for r in live),
            "tokens_out": sum(int(r["tokens_out"] or 0) for r in live),
            "usd": round(spend_usd, 4),
            "gbp": gbp(spend_usd),
            "cache_saved_usd": round(avoided_usd, 4),
            "cache_saved_gbp": gbp(avoided_usd),
            # Everything metered came from a priced model, or it did not. Said
            # plainly so an unpriced model reads as a gap, not as free usage.
            "fully_priced": all(r["priced"] for r in live) if live else True,
        },
        "unit_cost": {
            "usd_per_turn": round(usd_per_turn, 6),
            "gbp_per_turn": gbp(usd_per_turn),
            "turns_measured": turns,
            "turns_per_contact": TURNS_PER_CONTACT,
            "gbp_per_contact": gbp(usd_per_contact),
            "annual_contacts": annual_contacts,
            "gbp_per_year": round(usd_per_contact * annual_contacts * USD_TO_GBP, 2),
        },
        "activity": {
            "conversations": convs,
            "conversations_reaching_a_person": escalated,
            "claims_registered_by_bot": registered,
            "bot_runs_completed": rpa_done,
            "chases_sent": chases,
            "documents_processed": documents,
        },
        "by_prompt": summary,
    }


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
