"""PII redaction applied before anything leaves the process (LLM calls, logs)."""
from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,12}\b")),
    ("SORTCODE", re.compile(r"\b\d{2}-\d{2}-\d{2}\b")),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE)),
    ("NINO", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.IGNORECASE)),
]


def redact(text: str) -> str:
    """Replace PII with typed placeholders. Idempotent and order-stable."""
    if not text:
        return text
    out = text
    for label, pattern in PATTERNS:
        out = pattern.sub(f"[{label}_REDACTED]", out)
    return out


def contains_pii(text: str) -> list[str]:
    """Return the PII categories found — used by the output leak scan."""
    return [label for label, pattern in PATTERNS if pattern.search(text or "")]
