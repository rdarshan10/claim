"""How a customer gets hold of each document type.

Pure data and pure functions — no repository, no tools, no claim access. That
matters because the knowledge agent reads this, and its whole security property
is that a prompt injection which captures it still cannot reach customer data
(§9). Anything imported here has to keep that true.

This text used to live only in the document agent, which meant the assistant
could not answer its own suggested question: "How do I photograph my licence?"
routes to knowledge, found nothing in the knowledge base, and offered a human —
while the answer sat in a dict one module away.
"""
from __future__ import annotations

import re

GUIDANCE: dict[str, str] = {
    "police_report": "Ask the police station that recorded the incident for a copy — "
                     "it usually takes 2-3 days and there may be a small fee.",
    "repair_invoice": "Your garage can email this to you. It needs the garage's name, "
                      "an invoice number, the date and the total.",
    "damage_photo": "Take photos in daylight showing the whole vehicle, then close-ups "
                    "of each damaged area.",
    "driving_licence": "Photograph both sides of your licence, flat and in good light.",
    "claim_form": "You can download this from your policy documents, or I can email you "
                  "a fresh copy.",
    "medical_report": "Ask the treating clinic or hospital for a copy of the report.",
    "discharge_summary": "The hospital gives this to you when you leave; the ward can "
                         "reprint it if you've lost it.",
    "pharmacy_bill": "Your pharmacy can reprint a receipt if you have the prescription date.",
    "id_proof": "A passport or national ID card photographed flat, with all corners visible.",
    # Home claims — created by FNOL registration, and previously had no guidance
    # at all, so the checklist named them without saying how to get them.
    "damage_photos": "Photograph each damaged area in daylight — one wide shot of the "
                     "room, then close-ups. Include anything ruined that you're claiming for.",
    "repair_quote": "A written quote from a tradesperson or contractor. It needs their "
                    "name, the work described, and the total.",
    "treatment_invoice": "The clinic or hospital can email this. It needs the provider's "
                         "name, the treatment date and the amount.",
    # There were two "bank_statement" entries here; the second silently replaced
    # the first, so the payee wording had never once been shown to a customer.
    "bank_statement": "Download a PDF from your banking app — the last 3 months is enough. "
                      "It needs to show the account you'd like to be paid into.",
}

# Ordered: the first match wins, so specific document names beat the generic
# "photo" wording. "How do I photograph my licence?" is about the licence.
_DOC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("driving_licence", r"\b(?:driving|driver'?s?)\s+licen[cs]e\b|\blicen[cs]e\b"),
    ("police_report", r"\bpolice\s+(?:report|reference)\b|\bcrime\s+(?:report|reference)\b"),
    ("claim_form", r"\bclaim\s+form\b"),
    ("repair_invoice", r"\b(?:repair|garage)\s+(?:invoice|bill)\b|\binvoice\b"),
    ("repair_quote", r"\brepair\s+quote\b|\bquote\b"),
    ("discharge_summary", r"\bdischarge\s+summary\b"),
    ("medical_report", r"\bmedical\s+report\b"),
    ("treatment_invoice", r"\btreatment\s+invoice\b"),
    ("pharmacy_bill", r"\bpharmacy\b|\bprescription\b|\bchemist\b"),
    ("bank_statement", r"\bbank\s+statement\b"),
    ("id_proof", r"\bpassport\b|\bid\s+(?:proof|card)\b|\bidentity\b"),
    ("damage_photo", r"\b(?:damage\s+)?photos?\b|\bphotograph\b|\bpictures?\b"),
)

# Only how-to and where-from questions. "Is my licence covered?" is a cover
# question and belongs to the knowledge base, not to a how-to-send-it answer.
_HOW_TO = re.compile(
    r"\b(?:how\s+(?:do|can|should)\s+i|how\s+to|where\s+(?:do|can)\s+i|"
    r"where\s+(?:do|can)\s+i\s+(?:find|get)|what\s+do\s+i\s+(?:do|send)"
    r")\b", re.IGNORECASE)


def doc_type_for(message: str) -> str | None:
    """Which document a how-to question is asking about, if any."""
    text = message or ""
    if not _HOW_TO.search(text):
        return None
    for doc_type, pattern in _DOC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return doc_type
    return None


def answer_for(message: str) -> str | None:
    """The guidance text for a how-to question, or None if it isn't one."""
    doc_type = doc_type_for(message)
    return GUIDANCE.get(doc_type) if doc_type else None
