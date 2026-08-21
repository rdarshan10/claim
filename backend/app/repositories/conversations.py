"""Conversation and handoff state.

A conversation has one thread and three modes:

    AI            the assistant answers
    AWAITING_HUMAN a ticket is open, nobody has picked it up yet
    HUMAN_ACTIVE   a named reviewer has joined; the assistant relays only

The thread is shared — customer, assistant and reviewer messages all live in
``message``, so neither side ever sees a different history.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db import execute, query, query_one

MODE_AI = "AI"
MODE_AWAITING = "AWAITING_HUMAN"
MODE_HUMAN = "HUMAN_ACTIVE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(conversation_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM conversation WHERE id = ?", (conversation_id,))
    return dict(row) if row else None


def get_for_customer(conversation_id: str, customer_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM conversation WHERE id = ? AND customer_id = ?",
                    (conversation_id, customer_id))
    return dict(row) if row else None


def latest_for_customer(customer_id: str) -> dict[str, Any] | None:
    row = query_one(
        "SELECT * FROM conversation WHERE customer_id = ? ORDER BY started_at DESC LIMIT 1",
        (customer_id,),
    )
    return dict(row) if row else None


def set_mode(conversation_id: str, mode: str, *, agent: str | None = None,
             ticket_id: str | None = None) -> None:
    execute(
        "UPDATE conversation SET mode = ?, assigned_agent = ?, ticket_id = "
        "COALESCE(?, ticket_id) WHERE id = ?",
        (mode, agent, ticket_id, conversation_id),
    )


def add_message(conversation_id: str, role: str, content: str, *,
                author_name: str | None = None, intent: str | None = None,
                sentiment: str | None = None, citations: list | None = None,
                source_note: str | None = None,
                relay_source: str | None = None) -> str:
    """Append to the shared thread. ``role`` is user | assistant | agent | system.

    ``source_note`` keeps a reviewer's original wording beside what the customer
    was actually sent, so a relay can always be checked for drift.
    """
    message_id = str(uuid.uuid4())
    execute(
        """INSERT INTO message (id, conversation_id, role, content, intent, sentiment,
                                citations, author_name, source_note, relay_source,
                                created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (message_id, conversation_id, role, content, intent, sentiment,
         json.dumps(citations or []), author_name, source_note, relay_source, _now()),
    )
    return message_id


def thread(conversation_id: str, since: str | None = None,
           limit: int = 200) -> list[dict[str, Any]]:
    """The full shared thread, oldest first. ``since`` is an ISO timestamp."""
    if since:
        rows = query(
            """SELECT id, role, content, author_name, intent, sentiment, citations,
                      source_note, relay_source, created_at
               FROM message WHERE conversation_id = ? AND created_at > ?
               ORDER BY created_at ASC LIMIT ?""",
            (conversation_id, since, limit),
        )
    else:
        rows = query(
            """SELECT id, role, content, author_name, intent, sentiment, citations,
                      source_note, relay_source, created_at
               FROM message WHERE conversation_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (conversation_id, limit),
        )
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["citations"] = json.loads(item.get("citations") or "[]")
        except json.JSONDecodeError:
            item["citations"] = []
        out.append(item)
    return out


def history_for_agent(conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
    """Compact history for prompt context."""
    return [
        {"role": m["role"], "content": m["content"][:400], "intent": m.get("intent")}
        for m in thread(conversation_id)[-limit:]
    ]


def unseen_for_customer(conversation_id: str) -> int:
    """Messages that arrived since the customer last read the thread.

    Only counts messages they didn't write — a reviewer's answer relayed by the
    assistant is exactly the thing they need to come back to.
    """
    conversation = get(conversation_id)
    if conversation is None:
        return 0
    last_seen = conversation.get("last_seen_at")
    if not last_seen:
        return 0
    row = query_one(
        "SELECT COUNT(*) AS n FROM message WHERE conversation_id = ? "
        "AND role != 'user' AND created_at > ?",
        (conversation_id, last_seen),
    )
    return row["n"] if row else 0


def mark_seen(conversation_id: str) -> None:
    execute("UPDATE conversation SET last_seen_at = ? WHERE id = ?",
            (_now(), conversation_id))


def open_ticket_for(conversation_id: str) -> dict[str, Any] | None:
    row = query_one(
        "SELECT * FROM escalation_ticket WHERE conversation_id = ? "
        "AND status != 'RESOLVED' ORDER BY created_at DESC LIMIT 1",
        (conversation_id,),
    )
    return dict(row) if row else None


def open_ticket_for_customer(customer_id: str) -> dict[str, Any] | None:
    """The customer's open case, whichever conversation raised it.

    A case belongs to the person, not to a chat window. Scoping this to one
    conversation meant starting a new chat and asking for a human again opened a
    second ticket for the same problem.
    """
    row = query_one(
        "SELECT * FROM escalation_ticket WHERE customer_id = ? "
        "AND status != 'RESOLVED' ORDER BY created_at DESC LIMIT 1",
        (customer_id,),
    )
    return dict(row) if row else None


def delivery_target(customer_id: str, fallback_conversation_id: str | None) -> str | None:
    """Where a reviewer's answer should land: the customer's newest thread.

    Resolved at send time rather than stored on the ticket, so the answer
    arrives in the chat they're actually looking at even if they started a new
    one while waiting.
    """
    latest = latest_for_customer(customer_id)
    return latest["id"] if latest else fallback_conversation_id


def all_threads_for_customer(customer_id: str,
                             limit: int = 10) -> list[dict[str, Any]]:
    """Every conversation this customer has had, newest first.

    A reviewer needs the whole history, not one chat window — the problem is
    often described in an earlier conversation than the one that raised the case.
    """
    rows = query(
        "SELECT id, started_at, mode, assigned_agent FROM conversation "
        "WHERE customer_id = ? ORDER BY started_at DESC LIMIT ?",
        (customer_id, limit),
    )
    out = []
    for row in rows:
        conversation = dict(row)
        conversation["messages"] = thread(row["id"])
        conversation["message_count"] = len(conversation["messages"])
        out.append(conversation)
    return out
