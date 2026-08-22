"""FNOL agent — collects a notification of loss through the conversation.

Owns the "I want to make a claim" intent. It never writes to ``claim``: it fills
a ``fnol_request`` one question at a time and hands back a reference. Whether
that becomes a claim is a reviewer's decision, and the registration bot's job.

Each question goes out as an interactive card so the customer taps rather than
types, but free text always works — someone who writes "it was a collision on
the 3rd" should not be made to use a date picker.
"""
from __future__ import annotations

from typing import Any

from app.agents.state import GraphState
from app.audit import logger as audit
from app.fnol import intake

# Phrases that mean "start a claim" rather than "ask about my claim". Checked
# before the LLM router because the two are easy to confuse and the consequence
# of getting it wrong (silently answering about an existing claim) is bad.
START_PHRASES = (
    "make a claim", "file a claim", "start a claim", "open a claim", "new claim",
    "raise a claim", "submit a claim", "register a claim", "claim for",
    "want to claim", "need to claim", "報", "report an incident", "report a loss",
    "had an accident", "been in an accident", "my car was", "i crashed",
    "was stolen", "got stolen", "broke into", "burst pipe", "water damage",
)


def wants_to_start(message: str) -> bool:
    lowered = (message or "").lower()
    return any(phrase in lowered for phrase in START_PHRASES)


def run(state: GraphState) -> GraphState:
    """One turn of intake: record any answer, then ask the next question."""
    record = intake.open_for_customer(state.customer_id)

    if record is None:
        record = intake.create(state.customer_id, state.conversation_id)
        audit.record("fnol_started", actor_type="customer", actor_id=state.customer_id,
                     entity_type="fnol", entity_id=record["id"],
                     payload={"reference": record["reference"]}, trace_id=state.trace_id)
        state.facts["fnol_reference"] = record["reference"]
        state.facts["fnol_started"] = True
        state.reply = (
            f"Of course — I can take the details now. I've opened notification "
            f"{record['reference']} for you.\n\nNothing is submitted until you've "
            f"checked it over, and you can stop at any point and come back."
        )
        _ask_next(state, record)
        return state

    # A message arriving mid-intake is almost always the answer to the question
    # just asked, so try it against that field before treating it as anything else.
    consumed = False
    if state.message.strip():
        pending = intake.next_field(record.get("claim_type"), record["answers"])
        if pending is not None:
            value = intake.normalise(pending, state.message)
            if value is not None or pending.optional:
                record = intake.save_answer(record["id"], pending.key, value)
                consumed = True
            else:
                state.facts["fnol_reference"] = record["reference"]
                state.reply = _retry_message(pending)
                state.cards.append(intake.question_card(record, pending))
                return state

    state.facts["fnol_reference"] = record["reference"]

    # Someone returning to a half-finished notification should be told where
    # they are, not silently re-asked the same question as though nothing had
    # happened. Only when their message wasn't itself an answer.
    if not consumed and record["answers"]:
        answered = len(record["answers"])
        state.reply = (
            f"You've already started notification {record['reference']} — I've kept "
            f"the {answered} thing{'s' if answered != 1 else ''} you told me. "
            f"Let's carry on from where we left off."
        )

    _ask_next(state, record)
    return state


def _ask_next(state: GraphState, record: dict[str, Any]) -> None:
    """Emit the next question card, or the review card when nothing is left."""
    spec = intake.next_field(record.get("claim_type"), record["answers"])

    if spec is None:
        state.facts["fnol_complete"] = True
        state.facts["fnol_summary"] = intake.summary(record)
        state.cards.append(intake.review_card(record))
        if not state.reply:
            state.reply = (
                f"That's everything I need for {record['reference']}. Have a quick "
                f"look over it below — if it's right, send it to our claims team "
                f"and they'll get it registered."
            )
        return

    state.facts["fnol_next_field"] = spec.key
    state.cards.append(intake.question_card(record, spec))
    if not state.reply:
        state.reply = spec.question


def _retry_message(spec: Any) -> str:
    """Said when an answer couldn't be understood. Names the expected shape."""
    if spec.kind == "date":
        return ("Sorry, I couldn't make sense of that date. Could you give it as "
                "a day, month and year — or tap one of the options below?")
    if spec.kind == "money":
        return ("Sorry, I didn't catch a figure there. A rough number is fine — "
                "or tap \"I don't know yet\" and we'll set it later.")
    if spec.kind == "choice":
        return ("Sorry, I didn't quite follow. Could you pick one of these?")
    return "Sorry, I didn't catch that — could you try again?"


# --------------------------------------------------------------------------
# Status lookups
# --------------------------------------------------------------------------
def describe_open(customer_id: str) -> dict[str, Any] | None:
    """Any notification the customer has in flight, for the status agent."""
    records = [r for r in intake.list_for_customer(customer_id)
               if r["status"] in ("SUBMITTED", "UNDER_REVIEW", "INFO_REQUIRED",
                                  "READY_TO_REGISTER", "REGISTERING")]
    return records[0] if records else None


STATUS_WORDING = {
    "SUBMITTED": "with our claims team, waiting to be picked up",
    "UNDER_REVIEW": "being checked by our claims team",
    "INFO_REQUIRED": "waiting on a bit more information from you",
    "READY_TO_REGISTER": "approved and queued for registration",
    "REGISTERING": "being registered on our claims system right now",
}
