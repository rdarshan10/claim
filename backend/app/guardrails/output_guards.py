"""Output guardrails: grounding cross-check → PII leak scan → tone lint (§17.2).

The numeric/date/status cross-check is deterministic: every fact-shaped token in
a reply must appear in the tool results that produced it. An LLM cannot argue
its way past this.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.guardrails.pii import contains_pii

MONEY = re.compile(r"[£$€]\s?[\d,]+(?:\.\d{1,2})?")
DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DATE_TEXT = re.compile(
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{0,4}\b",
    re.IGNORECASE,
)
CLAIM_REF = re.compile(r"\bCLM-\d+\b", re.IGNORECASE)
POLICY_REF = re.compile(r"\bPOL-?\w+\b", re.IGNORECASE)
STATUSES = [
    "FILED", "DOCS_PENDING", "IN_ASSESSMENT", "ADDITIONAL_INFO", "APPROVED",
    "PAYMENT_IN_PROGRESS", "SETTLED", "REJECTED", "WITHDRAWN",
]

BLAME_PHRASES = [
    "you failed to", "you should have", "your fault", "you didn't bother",
    "you neglected", "as i already told you",
]


@dataclass
class OutputVerdict:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def _canonical_number(token: str) -> str | None:
    """Canonical form of a numeric token, so '£1,840.00' == '1840'."""
    digits = re.sub(r"[^\d.]", "", token or "")
    if not digits or digits == ".":
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def _grounding_corpus(tool_results: Any) -> str:
    # ensure_ascii=False matters: escaping '£' to '£' injects the digits
    # 0, 0 and 3 into the corpus and corrupts every number extracted from it.
    return json.dumps(tool_results, default=str, ensure_ascii=False).lower()


def check_output(reply: str, tool_results: Any) -> OutputVerdict:
    verdict = OutputVerdict()
    corpus = _grounding_corpus(tool_results)
    text = reply or ""

    # --- deterministic fact cross-check --------------------------------
    corpus_numbers = {
        canonical
        for match in re.findall(r"\d[\d,]*(?:\.\d+)?", corpus)
        if (canonical := _canonical_number(match))
    }
    for token in MONEY.findall(text):
        canonical = _canonical_number(token)
        if canonical and canonical not in corpus_numbers:
            verdict.ungrounded.append(f"amount {token}")

    for token in DATE_ISO.findall(text):
        if token.lower() not in corpus:
            verdict.ungrounded.append(f"date {token}")

    for token in CLAIM_REF.findall(text) + POLICY_REF.findall(text):
        if token.lower() not in corpus:
            verdict.ungrounded.append(f"reference {token}")

    for status in STATUSES:
        pretty = status.replace("_", " ").lower()
        if pretty in text.lower() and status.lower() not in corpus and pretty not in corpus:
            verdict.ungrounded.append(f"status {status}")

    if verdict.ungrounded:
        verdict.passed = False
        verdict.failures.append("grounding: unsupported facts " + ", ".join(verdict.ungrounded))

    # --- PII leak scan --------------------------------------------------
    if leaked := contains_pii(text):
        # Amounts/dates already covered above; only flag identity-shaped PII.
        identity = [label for label in leaked if label in {"EMAIL", "CARD", "NINO", "SORTCODE"}]
        if identity:
            verdict.passed = False
            verdict.failures.append(f"pii_leak: {', '.join(identity)}")

    # --- tone lint ------------------------------------------------------
    lowered = text.lower()
    for phrase in BLAME_PHRASES:
        if phrase in lowered:
            verdict.warnings.append(f"tone: blaming phrase '{phrase}'")

    if text and not re.search(r"[?]|next|upload|you can|i'll|i will|would you", lowered):
        verdict.warnings.append("tone: no clear next action offered")

    return verdict
