"""Hand-written knowledge base corpus (§13).

Deliberately curated, not scraped: every chunk is something a claims handler
would actually say, written at a B1 reading level so retrieved passages can be
quoted almost verbatim when the LLM is unavailable.
"""
from __future__ import annotations

KB: list[dict[str, object]] = [
    {
        "title": "What 'excess' means",
        "doc_class": "glossary",
        "chunks": [
            "Your excess is the amount you pay towards a claim yourself. If your excess "
            "is £250 and your repair costs £1,500, we pay £1,250 and you pay £250. The "
            "excess is usually taken off your settlement automatically, so you don't need "
            "to pay us separately.",
            "There are two kinds of excess. The compulsory excess is set by us and can't "
            "be changed. The voluntary excess is the extra amount you chose when you took "
            "out the policy, usually in exchange for a lower premium. Your total excess is "
            "the two added together, and it's shown on your policy schedule.",
        ],
    },
    {
        "title": "How a motor claim works, step by step",
        "doc_class": "process_guide",
        "chunks": [
            "A motor claim moves through five stages. First we file the claim and give you "
            "a claim number. Then we collect documents from you. Then the claim goes into "
            "assessment, where our team reviews the evidence and the repair costs. Then a "
            "decision is made. Finally, if the claim is approved, we make the payment.",
            "Most motor claims take between two and four weeks from start to payment. The "
            "stage that varies most is document collection, because it depends on how "
            "quickly you can get things like a police report. Once we have everything, "
            "assessment usually takes about a week.",
            "Your claim can only move into assessment once every required document has "
            "been checked and accepted. If one document is missing or rejected, the claim "
            "waits at the document stage. Nothing is lost while it waits.",
        ],
    },
    {
        "title": "Why we ask for a police report",
        "doc_class": "document_guide",
        "chunks": [
            "We ask for a police report when an accident involved another vehicle, an "
            "injury, or a theft. The report gives us an independent record of what "
            "happened, which protects you if the other driver disagrees with your account.",
            "To get a copy of a police report, contact the police station that recorded "
            "the incident and quote the crime or incident reference number they gave you "
            "at the scene. It usually takes two to three working days, and some forces "
            "charge a small fee.",
        ],
    },
    {
        "title": "Taking a good photo of a document",
        "doc_class": "document_guide",
        "chunks": [
            "Put the document flat on a table in good, even daylight. Hold your phone "
            "directly above it rather than at an angle, and make sure all four corners are "
            "inside the frame. Avoid shadows, especially your own. If the text looks blurry "
            "on your screen, it will be blurry to us too.",
            "If part of a document is too unclear for us to read, we'll tell you straight "
            "away and highlight exactly which part was the problem, so you only need to "
            "retake it once.",
        ],
    },
    {
        "title": "What happens if a document is rejected",
        "doc_class": "process_guide",
        "chunks": [
            "If we can't accept a document, we'll tell you why in plain English, show you "
            "the exact part of the document that caused the problem, and give you the steps "
            "to fix it. A rejected document doesn't harm your claim and doesn't count "
            "against you. You can upload a corrected version as many times as you need.",
            "If you think we've made a mistake about a document, use the 'This looks wrong "
            "to me' button. That sends it straight to a human colleague, who will look at "
            "it personally within one to two working days.",
        ],
    },
    {
        "title": "How settlement payments are made",
        "doc_class": "process_guide",
        "chunks": [
            "Once a claim is approved, payment is raised to the bank account we hold for "
            "you. It usually reaches your account within three to five working days. For "
            "motor repairs we sometimes pay the garage directly instead, in which case you "
            "won't see the money in your account at all.",
            "The amount you receive is the approved amount minus your excess. If you have "
            "any unpaid premium on the policy, that may also be deducted. The breakdown is "
            "always shown on your claim page.",
        ],
    },
    {
        "title": "What 'in assessment' means",
        "doc_class": "glossary",
        "chunks": [
            "'In assessment' means our claims team is actively reviewing your claim. They "
            "check the documents you sent, confirm the incident is covered by your policy, "
            "and review whether the costs are reasonable. You don't need to do anything "
            "while a claim is in assessment unless we ask you for something.",
        ],
    },
    {
        "title": "What 'subrogation' means",
        "doc_class": "glossary",
        "chunks": [
            "Subrogation is when we recover money from the person who caused the accident, "
            "or from their insurer, after we've paid your claim. It happens in the "
            "background and doesn't delay your payment. If we recover your excess as part "
            "of that, we'll refund it to you.",
        ],
    },
    {
        "title": "Making a complaint",
        "doc_class": "process_guide",
        "chunks": [
            "If you're unhappy with how your claim has been handled, you can ask to speak "
            "to a manager at any time and we'll arrange it. We aim to resolve complaints "
            "within eight weeks. If you're still unhappy after that, you can take your "
            "complaint to the Financial Ombudsman Service, which is free and independent.",
        ],
    },
    {
        "title": "Appealing a rejected claim",
        "doc_class": "process_guide",
        "chunks": [
            "If your claim is turned down, we'll tell you exactly which part of your policy "
            "the decision was based on. You have thirty days to appeal. To appeal, send us "
            "anything new that we haven't already seen, and ask for the decision to be "
            "reviewed. A different member of the team will look at it.",
        ],
    },
    {
        "title": "Why we ask for a repair invoice",
        "doc_class": "document_guide",
        "chunks": [
            "The repair invoice tells us what work was done and what it cost. We need it to "
            "show the garage's name, an invoice number, the date of the work, and the total "
            "amount. An estimate or quote isn't the same as an invoice, though we may accept "
            "an estimate earlier in the process.",
            "The invoice must be dated on or after the date of the accident. If the date "
            "looks wrong, it's usually a typing mistake by the garage, and they can issue a "
            "corrected copy.",
        ],
    },
    {
        "title": "Health claim documents",
        "doc_class": "document_guide",
        "chunks": [
            "For a health claim we usually need a medical report from the treating "
            "clinician, a discharge summary if you stayed in hospital, and the bills or "
            "pharmacy receipts for anything you're claiming back. The medical report needs "
            "to be signed or stamped by the clinic.",
        ],
    },
    {
        "title": "How long claims take",
        "doc_class": "process_guide",
        "chunks": [
            "Motor claims typically settle in two to four weeks. Health claims usually take "
            "three to five weeks because medical evidence takes longer to gather. Home "
            "claims vary the most, because some need a surveyor to visit. We give you a "
            "predicted date on your claim page and update it as things move.",
        ],
    },
    {
        "title": "Keeping your claim moving",
        "doc_class": "process_guide",
        "chunks": [
            "The single biggest cause of delay is a missing or unclear document. Uploading "
            "everything on your checklist as early as you can is the best thing you can do. "
            "We check each document the moment it arrives, so you'll know within seconds if "
            "something needs redoing.",
        ],
    },
]
