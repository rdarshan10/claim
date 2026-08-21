"""Empathy Responder (§16) — post-processor, not routable.

Renders already-verified facts into the customer-facing message using a tone
profile. It cannot fetch data, so it cannot introduce a fact that the output
guardrail hasn't got a source for.
"""
from __future__ import annotations

import json

from app.agents.state import GraphState
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.services import timeline_prediction


def run(state: GraphState) -> GraphState:
    # The Knowledge Agent already produced grounded prose; don't re-render it.
    if state.draft and state.intent == "knowledge":
        state.reply = state.draft
        return state

    fallback = _template(state)
    result = gateway.complete(
        "empathy_responder",
        {
            "message": wrap_untrusted("customer_message", state.message),
            "facts": json.dumps(state.facts, indent=2, default=str),
            "tone_profile": state.tone_profile,
            "context": f"Customer first name: {state.customer_name.split(' ')[0]}"
                       if state.customer_name else "",
        },
        tier="primary",
        trace_id=state.trace_id,
        fallback=fallback,
    )
    state.reply = result.text.strip()
    state.degraded = state.degraded or result.degraded
    return state


def _template(state: GraphState) -> str:
    """Template ladder used when the LLM is unavailable (UC-N7).

    Status and checklist answers are DB-driven, so they keep working fully.
    """
    facts = state.facts
    name = state.customer_name.split(" ")[0] if state.customer_name else ""
    greeting = f"{name}, " if name else ""

    # If a colleague is on the case, say so — in every answer, not just the one
    # that raised it.
    review = facts.get("human_review_in_progress") or {}
    review_line = ""
    if review:
        who = review.get("assigned_to")
        review_line = (
            f" {who} in our claims team is looking at this for you"
            if who else " A colleague is looking at this for you"
        ) + f" (reference {review.get('ticket_reference')}); I'll bring their answer here."

    if state.intent == "human_request" and facts.get("escalation"):
        esc = facts["escalation"]
        if esc.get("already_open"):
            who = esc.get("assigned_to")
            owner = f"{who} is" if who else "A colleague is"
            return (
                f"{owner} already on this for you — I raised it on "
                f"{esc.get('raised_at')} under reference {esc['ticket_reference']}, and "
                f"I've just flagged that you've chased it. I'll bring their answer "
                f"straight back to you here. Anything you'd like me to add for them?"
            )
        return (
            f"Of course — I've taken this to our claims team for you. Your reference is "
            f"{esc['ticket_reference']} and I'll have an answer back to you "
            f"{esc['eta']}. You won't need to chase anyone; I'll bring it here. "
            f"Is there anything you'd like me to add to the note for them?"
        )

    if facts.get("lookup_failed"):
        return (
            "I couldn't find a claim with that reference on your account. Claim numbers "
            "look like CLM-12345 and you'll find yours at the top of any email we've sent "
            "you. Would you like me to list the claims I can see instead?"
        )

    if not facts.get("claim") and not facts.get("checklist") and facts.get("claims") == []:
        return (
            "I can't see any claims on your account at the moment. If you've just started "
            "one it may take a few minutes to appear. Would you like me to get a colleague "
            "to check for you?"
        )

    if state.intent == "claim_status" and facts.get("claim"):
        claim = facts["claim"]
        meaning = facts.get("current_stage_meaning", "")
        parts = [
            f"{greeting}your claim {claim['claim_number']} is currently at "
            f"\"{claim['status'].replace('_', ' ').title()}\". {meaning}".strip()
        ]
        if late := facts.get("running_late"):
            parts.insert(0, (
                f"I want to be straight with you: this claim has been at this stage for "
                f"{late['days_in_stage']} days, longer than the usual "
                f"{late['typical_days']}. I've flagged it."
            ))
        prediction = facts.get("prediction") or {}
        if prediction.get("predicted_settlement_date") and not prediction.get("terminal"):
            parts.append(
                f"Based on similar claims I'd expect this to complete around "
                f"{prediction['predicted_settlement_date']} "
                f"(give or take {prediction.get('band_days', 3)} days)."
            )
        if outstanding := facts.get("outstanding_documents"):
            pretty = ", ".join(d.replace("_", " ") for d in outstanding)
            parts.append(f"To keep things moving I still need: {pretty}.")
        else:
            parts.append("You don't need to do anything right now — I'll update you as "
                         "soon as anything changes.")
        if review_line:
            parts.append(review_line.strip())
        return " ".join(parts)

    if state.intent == "documents" and facts.get("checklist"):
        outstanding = facts.get("outstanding_mandatory", [])
        if not outstanding:
            message = (
                f"{greeting}good news — every document we need for claim "
                f"{facts['claim_number']} has been checked and accepted. "
                f"There's nothing left for you to send."
            ).capitalize()
            # Acknowledge a rejected upload they no longer need to act on,
            # rather than leaving them wondering what happened to it.
            if covered := facts.get("rejected_but_already_covered"):
                doc = covered[0]["document"].replace("_", " ")
                message += (
                    f" I did notice the {doc} you sent most recently couldn't be "
                    f"accepted, but you don't need to do anything — we already have "
                    f"an accepted one on file."
                )
            return message
        pretty = ", ".join(d.replace("_", " ") for d in outstanding)
        lines = [f"{greeting}for claim {facts['claim_number']} I still need: {pretty}."]
        for doc_type in outstanding[:2]:
            if guidance := facts.get("guidance", {}).get(doc_type):
                lines.append(f"{doc_type.replace('_', ' ').capitalize()}: {guidance}")
        lines.append("Upload them here and I'll check each one straight away.")
        return " ".join(lines)

    if state.intent == "documents" and facts.get("recent_rejections"):
        rejection = facts["recent_rejections"][0]
        return (
            f"{rejection['headline']}. I've highlighted the problem on the document below, "
            f"along with the steps to fix it. Upload the corrected version and I'll check "
            f"it right away."
        )

    return (
        "I'm having a little trouble composing a full answer right now, but I can still "
        "look things up for you. Would you like your claim status, your document checklist, "
        "or to speak with a colleague?"
    )


def stage_meaning(status: str) -> str:
    return timeline_prediction.STAGE_MEANING.get(status, "")
