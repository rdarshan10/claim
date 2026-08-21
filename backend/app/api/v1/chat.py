"""Chat endpoints — WebSocket streaming plus a non-streaming fallback (§14.2)."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.agents import graph
from app.api.deps import Principal, current_principal, rate_limit
from app.db import execute, query, query_one
from app.security import jwt

router = APIRouter(prefix="/chat", tags=["chat"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_history(conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
    rows = query(
        """SELECT role, content, intent FROM message
           WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
        (conversation_id, limit),
    )
    return [dict(row) for row in reversed(rows)]


def _persist(conversation_id: str, role: str, content: str,
             intent: str | None = None, sentiment: str | None = None,
             citations: list | None = None) -> str:
    message_id = str(uuid.uuid4())
    execute(
        """INSERT INTO message (id, conversation_id, role, content, intent, sentiment,
                                citations, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (message_id, conversation_id, role, content, intent, sentiment,
         json.dumps(citations or []), _now()),
    )
    return message_id


@router.post("/conversations", status_code=status.HTTP_200_OK)
async def start_conversation(
    fresh: bool = False,
    principal: Principal = Depends(current_principal),
) -> dict[str, Any]:
    """Resume the customer's conversation, or start their first one.

    Resuming is the default and it matters: the assistant promises to bring a
    reviewer's answer back "here". Handing the customer a blank thread on their
    next visit would drop exactly the message they came back for.
    """
    from app.repositories import conversations as conv_repo

    first_name = principal.name.split(" ")[0] if principal.name else ""
    existing = None if fresh else conv_repo.latest_for_customer(principal.customer_id)

    if existing:
        unseen = conv_repo.unseen_for_customer(existing["id"])
        ticket = conv_repo.open_ticket_for(existing["id"])
        greeting = (
            f"Welcome back{' ' + first_name if first_name else ''} — I've kept "
            f"everything from last time."
        )
        if unseen:
            greeting += (f" There {'is' if unseen == 1 else 'are'} {unseen} new "
                         f"message{'' if unseen == 1 else 's'} for you below.")
        elif ticket:
            greeting += (f" Your case {ticket['id'][:8].upper()} is still with the "
                         f"claims team — I'll bring their answer straight here.")
        return {
            "conversation_id": existing["id"],
            "resumed": True,
            "unseen": unseen,
            "greeting": greeting,
            "suggestions": _suggestions(principal.customer_id),
        }

    conversation_id = str(uuid.uuid4())
    execute(
        "INSERT INTO conversation (id, customer_id, channel, started_at) VALUES (?,?,?,?)",
        (conversation_id, principal.customer_id, "web", _now()),
    )
    return {
        "conversation_id": conversation_id,
        "resumed": False,
        "unseen": 0,
        "greeting": (
            f"Hello{' ' + first_name if first_name else ''} — I'm "
            f"ClaimCompanion. I can check your claim status, tell you which documents we "
            f"still need, explain anything that's unclear, or put you through to a "
            f"colleague. What would you like to do?"
        ),
        "suggestions": _suggestions(principal.customer_id),
    }


# How to phrase the chip for whichever document is actually outstanding.
DOC_CHIP = {
    "police_report": "How do I get a police report?",
    "repair_invoice": "What should the repair invoice show?",
    "driving_licence": "How do I photograph my licence?",
    "claim_form": "Where do I find the claim form?",
    "medical_report": "How do I get a medical report?",
    "discharge_summary": "How do I get a discharge summary?",
    "pharmacy_bill": "How do I get a pharmacy receipt?",
    "id_proof": "What ID can I use?",
    "bank_statement": "Which bank statement do you need?",
    "damage_photo": "How should I photograph the damage?",
}


def _suggestions(customer_id: str) -> list[str]:
    """Chips reflecting what this customer needs to do *right now*.

    Recomputed on every poll: after the police report is accepted, offering
    "How do I get a police report?" is noise.
    """
    from app.repositories import claims as claim_repo

    base = ["Where is my claim?", "What do you still need from me?",
            "What does 'excess' mean?", "Talk to a person"]
    try:
        claims = claim_repo.get_claims(customer_id)
    except Exception:  # noqa: BLE001 - suggestions must never break the greeting
        return base

    open_claims = [c for c in claims
                   if c["status"] not in ("SETTLED", "REJECTED", "WITHDRAWN")]
    if not open_claims:
        return ["Where is my claim?", "What does 'excess' mean?", "Talk to a person"]

    checklist = claim_repo.checklist(open_claims[0]["id"], customer_id)
    outstanding = checklist["outstanding_mandatory"]

    if outstanding:
        chips = ["What do you still need from me?"]
        # Name the document they're actually missing, not a fixed example.
        for doc_type in outstanding[:2]:
            if chip := DOC_CHIP.get(doc_type):
                chips.append(chip)
        chips.append("Talk to a person")
        return chips[:4]

    return ["Where is my claim?", "When will I be paid?",
            "What happens next?", "Talk to a person"]


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    claim_id: str | None = None


def _handoff_context(conversation_id: str, customer_id: str) -> dict[str, Any] | None:
    """Live human-review state, so the assistant can speak to it honestly.

    Looked up by customer: an open case follows the person across chats.
    """
    from app.repositories import conversations as conv_repo

    ticket = conv_repo.open_ticket_for_customer(customer_id)
    if ticket is None:
        return None
    return {
        "ticket_id": ticket["id"],          # internal: used to keep the case current
        "ticket_reference": ticket["id"][:8].upper(),
        "priority": ticket["priority"],
        "status": ticket["status"],
        "assigned_to": ticket.get("assigned_to"),
        "raised_at": (ticket.get("created_at") or "")[:10],
    }


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    since: str = "",
    principal: Principal = Depends(current_principal),
) -> dict[str, Any]:
    """The shared thread. Polled by the client so reviewer answers arrive live."""
    from app.repositories import conversations as conv_repo

    conversation = conv_repo.get_for_customer(conversation_id, principal.customer_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    # Read the unseen count before marking the thread seen, so the client can
    # show "while you were away" for messages that arrived between visits.
    unseen = conv_repo.unseen_for_customer(conversation_id)
    last_seen = conversation.get("last_seen_at")
    messages = conv_repo.thread(conversation_id, since=since or None)
    conv_repo.mark_seen(conversation_id)

    return {
        "messages": messages,
        "mode": conversation.get("mode", "AI"),
        "assigned_agent": conversation.get("assigned_agent"),
        "handoff": _handoff_context(conversation_id, principal.customer_id),
        "unseen": unseen,
        "last_seen_at": last_seen,
        # Recomputed every poll: chips must follow the checklist, not the state
        # it was in when the conversation started.
        "suggestions": _suggestions(principal.customer_id),
    }


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: MessageRequest,
    principal: Principal = Depends(rate_limit),
) -> dict[str, Any]:
    conversation = query_one(
        "SELECT id FROM conversation WHERE id = ? AND customer_id = ?",
        (conversation_id, principal.customer_id),
    )
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    history = _load_history(conversation_id)
    _persist(conversation_id, "user", body.message)

    state = await asyncio.to_thread(
        graph.run_turn,
        customer_id=principal.customer_id,
        customer_name=principal.name,
        message=body.message,
        history=history,
        conversation_id=conversation_id,
        active_claim_id=body.claim_id,
        handoff=_handoff_context(conversation_id, principal.customer_id),
    )

    message_id = _persist(conversation_id, "assistant", state.reply,
                          state.intent, state.sentiment, state.citations)

    return {
        "message_id": message_id,
        "reply": state.reply,
        "intent": state.intent,
        "sentiment": state.sentiment,
        "cards": state.cards,
        "citations": state.citations,
        "blocked": state.blocked,
        "degraded": state.degraded,
        "guardrail_flags": state.guardrail_flags,
        "active_claim_id": state.active_claim_id,
    }


@router.websocket("/conversations/{conversation_id}/stream")
async def stream(websocket: WebSocket, conversation_id: str) -> None:
    """Token/card frames per §14.2. Token is passed as a query param."""
    await websocket.accept()

    token = websocket.query_params.get("token", "")
    try:
        claims = jwt.decode(token)
    except jwt.AuthError:
        await websocket.send_json({"type": "error", "message": "Not authenticated"})
        await websocket.close(code=4401)
        return

    principal = Principal(claims)
    conversation = query_one(
        "SELECT id FROM conversation WHERE id = ? AND customer_id = ?",
        (conversation_id, principal.customer_id),
    )
    if conversation is None:
        await websocket.send_json({"type": "error", "message": "Conversation not found"})
        await websocket.close(code=4404)
        return

    loop = asyncio.get_running_loop()

    try:
        while True:
            payload = await websocket.receive_json()
            message = str(payload.get("message", ""))[:4000]
            claim_id = payload.get("claim_id")
            if not message.strip():
                continue

            history = _load_history(conversation_id)
            _persist(conversation_id, "user", message)

            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            def emit(frame: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, frame)

            def work() -> Any:
                try:
                    return graph.run_turn(
                        customer_id=principal.customer_id,
                        customer_name=principal.name,
                        message=message,
                        history=history,
                        conversation_id=conversation_id,
                        active_claim_id=claim_id,
                        handoff=_handoff_context(conversation_id, principal.customer_id),
                        emit=emit,
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = asyncio.create_task(asyncio.to_thread(work))

            while True:
                frame = await queue.get()
                if frame is None:
                    break
                await websocket.send_json(frame)

            state = await task
            message_id = _persist(conversation_id, "assistant", state.reply,
                                  state.intent, state.sentiment, state.citations)
            await websocket.send_json({"type": "persisted", "message_id": message_id,
                                       "active_claim_id": state.active_claim_id})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
        await websocket.send_json({
            "type": "error",
            "message": "Our assistant hit a problem. Your claim details are still "
                       "available on your dashboard.",
        })
        from app.audit import logger as audit
        audit.record("chat_error", entity_type="conversation", entity_id=conversation_id,
                     payload={"error": str(exc)[:300]})


class FeedbackRequest(BaseModel):
    message_id: str
    helpful: bool
    reason: str = ""


@router.post("/conversations/{conversation_id}/feedback")
async def feedback(conversation_id: str, body: FeedbackRequest,
                   principal: Principal = Depends(current_principal)) -> dict[str, str]:
    from app.audit import logger as audit

    audit.record("chat_feedback", actor_type="customer", actor_id=principal.customer_id,
                 entity_type="message", entity_id=body.message_id,
                 payload={"helpful": body.helpful, "reason": body.reason[:300]})
    return {"status": "recorded"}
