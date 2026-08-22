"""The handler's diary — what needs attention today, and why.

A claims handler does not "work a queue of documents"; they work a diary. Every
open claim carries a date it comes back to them and a note of what they are
waiting for. On that date they make one decision: chase the customer, chase a
third party, revise the reserve, settle, or push the date out with a reason.

Nothing in this system did that, so claims sat untouched — the demo data has
claims stalled 80+ days with nobody chasing. That is not a small inefficiency:
it is the single largest source of both handler workload (the customer
eventually rings) and complaints.

This module decides two things:

* when a claim should next be looked at, from its status; and
* what the handler should do when it surfaces.

Both are deliberately rule-based rather than model-driven. A diary date is an
operational commitment — a handler needs to know why a claim is in front of
them today, and "the model thought so" is not an answer they can give a
customer or an auditor.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.db import execute, query, query_one

# How long a claim may sit in each state before a handler should look at it.
# These are service-level intentions, not SLAs the code enforces elsewhere.
REVIEW_INTERVAL_DAYS: dict[str, int] = {
    "FILED": 2,               # get it moving
    "DOCS_PENDING": 7,        # chase the customer weekly
    "ADDITIONAL_INFO": 7,     # same, but we asked a specific question
    "IN_ASSESSMENT": 5,       # keep assessment honest
    "APPROVED": 3,            # push it to payment
    "PAYMENT_IN_PROGRESS": 5,
}

# What the handler is actually being asked to do when the claim surfaces.
ACTION_BY_STATUS: dict[str, str] = {
    "FILED": "Open it and set the reserve",
    "DOCS_PENDING": "Chase the outstanding documents",
    "ADDITIONAL_INFO": "Chase the information we asked for",
    "IN_ASSESSMENT": "Assess and decide",
    "APPROVED": "Release the payment",
    "PAYMENT_IN_PROGRESS": "Confirm the payment cleared",
}

# A claim nobody has touched for this long is not just due — it is a complaint
# waiting to happen, and handlers should see it before the customer rings.
AT_RISK_DAYS = 30

# Chasing has diminishing returns and becomes harassment. After this many
# attempts the answer is a phone call from a person, not another message.
MAX_AUTOMATED_CHASES = 3


def _today() -> date:
    return datetime.now(timezone.utc).date()


def interval_for(status: str) -> int | None:
    """Days until a claim in this status should be looked at again."""
    return REVIEW_INTERVAL_DAYS.get(status)


def next_date_for(status: str, *, from_day: date | None = None) -> str | None:
    days = interval_for(status)
    if days is None:
        return None
    return ((from_day or _today()) + timedelta(days=days)).isoformat()


def set_review(claim_id: str, when: str, note: str | None = None) -> None:
    execute("UPDATE claim SET next_review_date = ?, diary_note = ? WHERE id = ?",
            (when, note, claim_id))


def backfill() -> int:
    """Give every open claim a diary date it does not already have.

    Existing claims predate the diary, so without this they would never
    surface. Dated from the claim's own last movement rather than from today,
    so a claim that has been stalled for 80 days shows as 80 days overdue —
    which is the truth, and the whole point.
    """
    rows = query(
        """SELECT c.id, c.status, c.filed_at,
                  (SELECT MAX(changed_at) FROM claim_status_history h
                    WHERE h.claim_id = c.id) AS last_moved
           FROM claim c
           WHERE c.next_review_date IS NULL
             AND c.status NOT IN ('SETTLED','REJECTED','WITHDRAWN')"""
    )
    n = 0
    for row in rows:
        item = dict(row)
        anchor = (item["last_moved"] or item["filed_at"] or "")[:10]
        try:
            base = date.fromisoformat(anchor)
        except ValueError:
            base = _today()
        when = next_date_for(item["status"], from_day=base)
        if when:
            set_review(item["id"], when,
                       ACTION_BY_STATUS.get(item["status"], "Review this claim"))
            n += 1
    return n


def due(handler: str | None = None, *, include_future: bool = False) -> list[dict[str, Any]]:
    """Claims needing attention, most overdue first.

    Ordering by how overdue a claim is — rather than by when it was filed — is
    what makes this a diary rather than another list. The claim that has been
    waiting longest past its own review date is the one a handler should pick
    up next.
    """
    today = _today().isoformat()
    where = ["c.status NOT IN ('SETTLED','REJECTED','WITHDRAWN')",
             "c.next_review_date IS NOT NULL"]
    params: list[Any] = []
    if not include_future:
        where.append("c.next_review_date <= ?")
        params.append(today)
    if handler:
        where.append("c.handler = ?")
        params.append(handler)

    rows = query(
        f"""SELECT c.id, c.claim_number, c.claim_type, c.status, c.handler,
                   c.next_review_date, c.diary_note, c.last_chased_at,
                   c.chase_count, c.filed_at,
                   cu.full_name AS customer_name, cu.id AS customer_id,
                   CAST(julianday(?) - julianday(c.next_review_date) AS INT) AS days_overdue,
                   CAST(julianday(?) - julianday(
                        COALESCE((SELECT MAX(changed_at) FROM claim_status_history h
                                   WHERE h.claim_id = c.id), c.filed_at)) AS INT) AS days_since_movement
            FROM claim c
            JOIN policy p ON p.id = c.policy_id
            JOIN customer cu ON cu.id = p.customer_id
            WHERE {' AND '.join(where)}
            ORDER BY days_overdue DESC, c.filed_at ASC""",
        (today, today, *params),
    )

    out = []
    for row in rows:
        item = dict(row)
        item["action"] = ACTION_BY_STATUS.get(item["status"], "Review this claim")
        item["at_risk"] = (item["days_since_movement"] or 0) >= AT_RISK_DAYS
        # Whether a person should now pick up the phone instead of the system
        # sending another message.
        item["chases_exhausted"] = (item["chase_count"] or 0) >= MAX_AUTOMATED_CHASES
        out.append(item)
    return out


def summary(handler: str | None = None) -> dict[str, Any]:
    """Counts for the diary tab, so a handler sees the shape of their day."""
    items = due(handler)
    return {
        "due_today": len(items),
        "overdue": sum(1 for i in items if (i["days_overdue"] or 0) > 0),
        "at_risk": sum(1 for i in items if i["at_risk"]),
        "needs_a_call": sum(1 for i in items if i["chases_exhausted"]),
    }
