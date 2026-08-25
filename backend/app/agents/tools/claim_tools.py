"""Typed tools — the only sanctioned path from an agent to data (§10).

Every tool takes ``customer_id`` as its first argument and that value always
originates from the JWT. No tool accepts a customer id supplied in message text,
and no tool can mutate a claim decision — the capability simply does not exist
(§17.1 "jailbreak -> decision manipulation").
"""
from __future__ import annotations

import re
from typing import Any

from app.audit import logger as audit
from app.repositories import claims as repo
from app.services import timeline_prediction

# "CLM-88430" written out. Unambiguous: nothing else in a message looks like it,
# so failing to find it is worth saying out loud rather than answering about
# some other claim.
CLAIM_REF = re.compile(r"\bCLM-\d+\b", re.IGNORECASE)

# The same claim, typed the way people actually type it — "what's left on 88430".
# A bare number could be anything (an invoice, a year, an amount), so it is only
# ever accepted when it turns out to name one of this customer's claims.
BARE_REF = re.compile(r"\b\d{4,}\b")


def _audited(name: str, customer_id: str, args: dict[str, Any], result_size: int) -> None:
    audit.record("tool_call", actor_type="agent", actor_id=name,
                 entity_type="customer", entity_id=customer_id,
                 payload={"tool": name, "args": args, "results": result_size})


# "the other claim" names a claim by contrast with the one already in view. It
# is as specific as a reference number when the customer has exactly two, and
# left unhandled it fell through to the claim the thread was already on — so the
# reply talked about the other claim while the cards below showed the current
# one, contradicting each other on screen.
# The word between "the" and "claim"/"one" is captured rather than matched
# literally, so a typed message still resolves. Customers type "the oter one"
# and an exact match sent them the claim they were already looking at while the
# reply talked about the other one.
OTHER_REF = re.compile(
    r"\b(?:the|my|that)\s+([a-z]{3,7})\s+(?:claim|one)\b", re.IGNORECASE)


def _is_other(word: str) -> bool:
    """Is this word "other", allowing for one typo?

    One edit covers the common slips — a dropped letter ("oter"), a doubled one
    ("otherr"), a swapped pair ("ohter"). Swaps count as one edit rather than
    two, because transposing adjacent letters is the typo people actually make.
    Two edits would start matching real words that mean something else, so the
    bar stays at one: "first", "second" and "motor" are choices, not misspellings.
    """
    word = (word or "").lower()
    target = "other"
    if word == target:
        return True
    if abs(len(word) - len(target)) > 1:
        return False

    # Optimal string alignment: Levenshtein plus adjacent transposition.
    rows = [[0] * (len(target) + 1) for _ in range(len(word) + 1)]
    for i in range(len(word) + 1):
        rows[i][0] = i
    for j in range(len(target) + 1):
        rows[0][j] = j
    for i in range(1, len(word) + 1):
        for j in range(1, len(target) + 1):
            cost = word[i - 1] != target[j - 1]
            rows[i][j] = min(rows[i - 1][j] + 1,        # deletion
                             rows[i][j - 1] + 1,        # insertion
                             rows[i - 1][j - 1] + cost)  # substitution
            if (i > 1 and j > 1 and word[i - 1] == target[j - 2]
                    and word[i - 2] == target[j - 1]):
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)  # swap
    return rows[len(word)][len(target)] <= 1


def resolve_other(message: str, claims: list[dict[str, Any]],
                  active_claim_id: str | None) -> dict[str, Any] | None:
    """The claim the customer means by "the other one", if that is unambiguous.

    Only when exactly one claim is not the one in view. With three claims the
    phrase genuinely does not say which, and picking one would reintroduce the
    bug it solves.
    """
    if not active_claim_id:
        return None
    if not any(_is_other(m.group(1)) for m in OTHER_REF.finditer(message or "")):
        return None
    others = [c for c in claims if c["id"] != active_claim_id]
    return others[0] if len(others) == 1 else None


def resolve_reference(customer_id: str, message: str) -> tuple[dict[str, Any] | None, bool]:
    """Which claim a message names, if any.

    Returns ``(claim, was_explicit)``. ``was_explicit`` marks a written-out
    ``CLM-`` reference, which the caller should refuse to guess past when it
    does not resolve — a bare number that matches nothing is far more likely to
    be an invoice total than a claim the customer cannot see.
    """
    text = message or ""
    if match := CLAIM_REF.search(text):
        return find_claim_by_number(customer_id, match.group(0)), True

    for digits in BARE_REF.findall(text):
        claim = find_claim_by_number(customer_id, f"CLM-{digits}")
        if claim is not None:
            return claim, False
    return None, False


def get_claims(customer_id: str) -> list[dict[str, Any]]:
    claims = repo.get_claims(customer_id)
    _audited("get_claims", customer_id, {}, len(claims))
    return claims


def get_claim_detail(customer_id: str, claim_id: str) -> dict[str, Any] | None:
    claim = repo.get_claim(claim_id, customer_id)
    _audited("get_claim_detail", customer_id, {"claim_id": claim_id}, 1 if claim else 0)
    return claim


def find_claim_by_number(customer_id: str, claim_number: str) -> dict[str, Any] | None:
    claim = repo.find_claim_by_number(claim_number, customer_id)
    _audited("find_claim_by_number", customer_id, {"claim_number": claim_number},
             1 if claim else 0)
    return claim


def get_status_history(customer_id: str, claim_id: str) -> list[dict[str, Any]]:
    history = repo.get_status_history(claim_id, customer_id)
    _audited("get_status_history", customer_id, {"claim_id": claim_id}, len(history))
    return history


def predict_timeline(customer_id: str, claim_id: str) -> dict[str, Any] | None:
    claim = repo.get_claim(claim_id, customer_id)
    if claim is None:
        return None
    history = repo.get_status_history(claim_id, customer_id)
    # Whether the customer still owes us paperwork changes the DOCS_PENDING
    # estimate materially, and the tool already knows enough to find out.
    checklist = repo.checklist(claim_id, customer_id)
    prediction = timeline_prediction.predict(
        claim, history, awaiting_customer=bool(checklist["awaiting_customer"]))
    _audited("predict_timeline", customer_id, {"claim_id": claim_id}, 1)
    return prediction


def get_required_documents(customer_id: str, claim_id: str) -> dict[str, Any]:
    checklist = repo.checklist(claim_id, customer_id)
    _audited("get_required_documents", customer_id, {"claim_id": claim_id},
             len(checklist["items"]))
    return checklist


def get_document_status(customer_id: str, claim_id: str) -> list[dict[str, Any]]:
    documents = repo.get_documents(claim_id, customer_id)
    _audited("get_document_status", customer_id, {"claim_id": claim_id}, len(documents))
    return documents


def get_rejection_details(customer_id: str, document_id: str) -> dict[str, Any] | None:
    document = repo.get_document(document_id, customer_id)
    _audited("get_rejection_details", customer_id, {"document_id": document_id},
             1 if document else 0)
    if document is None:
        return None
    return document.get("rejection_payload")
