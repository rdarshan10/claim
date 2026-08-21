"""Document Agent (§9): checklist gaps, verification results, rejection explanations."""
from __future__ import annotations

from app.agents.state import GraphState
from app.agents.tools import claim_tools as tools

GUIDANCE = {
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
    "bank_statement": "Download a PDF from your banking app — the last 3 months is enough.",
}


def run(state: GraphState) -> GraphState:
    claims = tools.get_claims(state.customer_id)
    if not claims:
        state.facts = {"claims": [], "note": "This customer has no claims on file."}
        return state

    claim = None
    if state.active_claim_id:
        claim = tools.get_claim_detail(state.customer_id, state.active_claim_id)
    if claim is None:
        open_claims = [c for c in claims
                       if c["status"] not in ("SETTLED", "REJECTED", "WITHDRAWN")]
        claim = (open_claims or claims)[0]
    state.active_claim_id = claim["id"]

    checklist = tools.get_required_documents(state.customer_id, claim["id"])
    documents = tools.get_document_status(state.customer_id, claim["id"])

    rejected = [
        d for d in documents
        if str(d["status"]).startswith("REJECTED") and d.get("rejection_payload")
    ]
    # A rejection the customer no longer needs to act on: they already sent an
    # acceptable version of that document. Mentioning it as a problem alongside
    # a complete checklist contradicts itself.
    actionable = [d for d in rejected if not d.get("superseded")]
    superseded = [d for d in rejected if d.get("superseded")]

    state.facts = {
        "claim_number": claim["claim_number"],
        "claim_type": claim["claim_type"],
        "checklist": [
            {"document": item["doc_type"].replace("_", " "),
             "state": item["state"], "required": item["mandatory"]}
            for item in checklist["items"]
        ],
        "outstanding_mandatory": checklist["outstanding_mandatory"],
        "checklist_complete": checklist["complete"],
        "guidance": {
            doc_type: GUIDANCE.get(doc_type, "")
            for doc_type in checklist["outstanding_mandatory"]
        },
        "recent_rejections": [
            {"document": d.get("doc_type"),
             "headline": d["rejection_payload"].get("headline"),
             "reason_code": d["rejection_payload"].get("reason_code")}
            for d in actionable[:2]
        ],
        "rejected_but_already_covered": [
            {"document": d.get("doc_type"),
             "reason_code": d["rejection_payload"].get("reason_code"),
             "note": ("We already hold an accepted version of this document, so "
                      "the customer does not need to do anything about it. "
                      "Reassure them; do not ask for a replacement.")}
            for d in superseded[:2]
        ],
    }

    state.cards.append({
        "card_type": "checklist",
        "payload": {
            "claim_number": claim["claim_number"],
            "items": [
                {**item, "guidance": GUIDANCE.get(item["doc_type"], "")}
                for item in checklist["items"]
            ],
            "complete": checklist["complete"],
        },
    })

    # Surface the most recent Smart Rejection Explanation inline (§11.8), but
    # only where the customer still has something to do about it.
    if actionable:
        state.cards.append({
            "card_type": "doc_rejection",
            "payload": actionable[0]["rejection_payload"],
        })

    return state
