"""Supervisor / intent router + sentiment (§9).

Classifies with the mini model and falls back to deterministic heuristics, so
routing never depends on LLM availability (UC-N7).
"""
from __future__ import annotations

import json
import re

from app.agents.state import GraphState
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway

INTENT_SIGNALS: dict[str, list[str]] = {
    # Checked before claim_status: "my claim" appears in both "make a claim" and
    # "where is my claim", and dict order decides ties in _heuristic_intent.
    "new_claim": [
        r"\b(?:make|file|start|open|raise|submit|register|lodge)\s+(?:a|an|my|new)?\s*claim\b",
        r"\bnew claim\b", r"\bclaim for\b",
        r"\b(?:want|need|would like) to claim\b",
        r"\breport (?:an? )?(?:incident|loss|accident|theft|break-?in)\b",
        r"\b(?:had|been in|was in) (?:an? )?(?:accident|crash|collision|prang)\b",
        r"\bi (?:crashed|reversed into|bumped|scraped)\b",
        r"\b(?:was|got|been) (?:stolen|burgled|broken into|vandalised|vandalized)\b",
        r"\bburst pipe\b", r"\bwater damage\b", r"\bflood(?:ed|ing)?\b",
        r"\bfire damage\b", r"\bstorm damage\b", r"\bwrite[- ]?off\b",
    ],
    "human_request": [
        r"\b(?:talk|speak|chat) to (?:a |an )?(?:human|person|agent|someone|advisor)\b",
        r"\breal person\b", r"\bcall me\b", r"\bcomplain(?:t|ing)?\b",
        r"\bmanager\b", r"\bescalate\b",
        # "I want someone to actually look at this" — a human request without
        # ever using the word "human".
        r"\b(?:someone|a person|a human)\s+(?:to\s+)?(?:actually\s+)?"
        r"(?:look|check|review|see|go through)\b",
        r"\b(?:want|need)\s+(?:someone|a human|a person)\b",
        r"\bhuman (?:review|being|check)\b",
    ],
    "documents": [
        r"\bdocument", r"\bupload", r"\binvoice", r"\breceipt", r"\bphoto",
        r"\bwhat (?:do|does) (?:you|they) (?:still )?need\b", r"\bmissing\b",
        r"\breject(?:ed|ion)?\b", r"\bchecklist\b", r"\bpolice report\b",
        r"\blicence\b", r"\blicense\b", r"\bsend (?:you|in)\b", r"\bpaperwork\b",
    ],
    "claim_status": [
        r"\bstatus\b", r"\bwhere is my claim\b", r"\bwhen will i (?:be paid|get)\b",
        r"\bhow long\b", r"\bprogress\b", r"\bupdate on\b", r"\bpaid\b",
        r"\bsettle(?:d|ment)?\b", r"\bmy claim\b", r"\bapproved\b", r"\btimeline\b",
        # Details already on the claim record. Without these, "when was the
        # incident?" matched nothing at all and fell through to the knowledge
        # default — which has no article about this customer's own claim, so it
        # correctly refused to guess and offered a human instead. The date was
        # sitting in the claim the whole time.
        r"\bincident\b", r"\bdate of (?:loss|incident|accident)\b",
        r"\bwhen (?:was|did) (?:the|my|it)\b",
        r"\bhow much (?:am|did|is|was|will)\b",
        r"\bclaim (?:number|reference|ref)\b", r"\bclm-\d+\b",
        # Asking after something already reported, not starting a new one. The
        # new_claim patterns above need a verb ("report a loss"), so these bare
        # nouns don't collide with them.
        r"\bfnols?\b", r"\bnotifications? of loss\b", r"\bnotifications?\b",
        # Counting what they already have. Kept to "how many claims" rather than
        # a bare \bclaims?\b, which would pull knowledge questions like "what is
        # the excess on my claim" over to this agent.
        r"\bhow many claims?\b",
    ],
    "knowledge": [
        r"\bwhat is\b", r"\bwhat does .* mean\b", r"\bhow (?:do|does) .* work\b",
        r"\bexplain\b", r"\bexcess\b", r"\bpolicy (?:cover|term)", r"\bam i covered\b",
        r"\bwhy do you need\b",
    ],
}

TONE_BY_SENTIMENT = {
    "calm": "neutral-warm",
    "confused": "reassuring",
    "frustrated": "apologetic-accountable",
    "distressed": "gentle-supportive",
}

SENTIMENT_SIGNALS = {
    "frustrated": [r"\bstill waiting\b", r"\bweeks?\b.*\bnothing\b", r"\bridiculous\b",
                   r"\bunacceptable\b", r"\bfed up\b", r"\bagain\b.*\basked\b",
                   r"\bno one\b.*\b(?:replied|called|answered)\b", r"!!+"],
    "confused": [r"\bconfused\b", r"\bdon'?t understand\b", r"\bwhat does .* mean\b",
                 r"\bnot sure what\b", r"\bhow do i\b"],
    "distressed": [r"\bpassed away\b", r"\bdied\b", r"\bfuneral\b", r"\bbereave",
                   r"\bcan'?t afford\b", r"\bdesperate\b", r"\bin hospital\b",
                   r"\bstruggling\b", r"\bscared\b"],
}


GREETING_PATTERNS = [
    r"^\s*(?:hi|hey|hello|hiya|yo|howdy|good\s+(?:morning|afternoon|evening))\b",
    r"^\s*(?:thanks|thank you|ta|cheers|much appreciated)\b",
    r"^\s*(?:ok|okay|right|sure|got it|understood|fine)\s*[.!]?\s*$",
    r"^\s*(?:bye|goodbye|see you|later)\b",
    r"^\s*(?:how are you|you there|are you there|anyone there)\b",
    r"^\s*(?:what can you do|who are you|what are you|help)\s*[?.!]?\s*$",
]


def is_small_talk(message: str) -> bool:
    """Greetings, thanks and sign-offs.

    Handled deterministically before routing: a classifier with no 'greeting'
    label will file "hi" under out_of_scope and refuse it, which is a terrible
    first impression for a customer who has just opened the chat.
    """
    text = (message or "").strip()
    if len(text.split()) > 6:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in GREETING_PATTERNS)


def _heuristic_intent(message: str) -> tuple[str, float]:
    lowered = (message or "").lower()
    scores: dict[str, int] = {}
    for intent, patterns in INTENT_SIGNALS.items():
        hits = sum(1 for p in patterns if re.search(p, lowered))
        if hits:
            scores[intent] = hits

    if not scores:
        return "knowledge", 0.35

    # A human request always wins; it is never worth "routing around".
    if "human_request" in scores:
        return "human_request", 0.9

    best = max(scores, key=lambda k: scores[k])
    return best, min(0.9, 0.55 + 0.15 * scores[best])


def _heuristic_sentiment(message: str) -> tuple[str, float]:
    lowered = (message or "").lower()
    for sentiment in ("distressed", "frustrated", "confused"):
        for pattern in SENTIMENT_SIGNALS[sentiment]:
            if re.search(pattern, lowered):
                return sentiment, 0.7
    return "calm", 0.6


def route(state: GraphState) -> GraphState:
    """Classify intent and sentiment in one call, then pick a tone profile."""
    heuristic_intent, heuristic_conf = _heuristic_intent(state.message)
    heuristic_sentiment, heuristic_sent_conf = _heuristic_sentiment(state.message)
    history_text = "\n".join(
        f"{turn['role']}: {turn['content'][:200]}" for turn in state.history[-6:]
    )

    result = gateway.complete(
        "classify_turn",
        {"history": history_text or "(new conversation)",
         "message": wrap_untrusted("customer_message", state.message)},
        tier="mini",
        trace_id=state.trace_id,
        fallback=json.dumps({
            "intent": heuristic_intent, "intent_confidence": heuristic_conf,
            "sentiment": heuristic_sentiment,
            "sentiment_confidence": heuristic_sent_conf,
            "injection_suspected": False,
        }),
    )
    parsed = result.json(default={}) or {}
    intent = str(parsed.get("intent", heuristic_intent)).strip()
    if intent not in INTENT_SIGNALS and intent not in ("out_of_scope", "greeting"):
        intent = heuristic_intent

    try:
        confidence = float(parsed.get("intent_confidence", heuristic_conf))
    except (TypeError, ValueError):
        confidence = heuristic_conf

    # A deterministic human request overrides a model that routed elsewhere.
    if heuristic_intent == "human_request":
        intent, confidence = "human_request", 0.95

    if parsed.get("injection_suspected"):
        state.guardrail_flags.append("router_injection_suspected")

    state.intent = intent  # type: ignore[assignment]
    state.intent_confidence = round(confidence, 2)
    state.degraded = state.degraded or result.degraded

    sentiment = str(parsed.get("sentiment", heuristic_sentiment)).strip().lower()
    if sentiment not in TONE_BY_SENTIMENT:
        sentiment = heuristic_sentiment

    # Deterministic distress detection is never overridden downwards — a model
    # that reads "I can't go on" as merely 'frustrated' must not win that call.
    if heuristic_sentiment == "distressed":
        sentiment = "distressed"

    state.sentiment = sentiment  # type: ignore[assignment]
    state.tone_profile = TONE_BY_SENTIMENT[sentiment]  # type: ignore[assignment]
    return state


def _sentiment(state: GraphState) -> None:
    """Sentiment only — used on paths that skip routing (e.g. handoff relay)."""
    heuristic, heuristic_conf = _heuristic_sentiment(state.message)
    result = gateway.complete(
        "sentiment",
        {"message": wrap_untrusted("customer_message", state.message)},
        tier="mini",
        trace_id=state.trace_id,
        fallback=json.dumps({"sentiment": heuristic, "confidence": heuristic_conf}),
    )
    parsed = result.json(default={}) or {}
    sentiment = str(parsed.get("sentiment", heuristic)).strip().lower()
    if sentiment not in TONE_BY_SENTIMENT:
        sentiment = heuristic
    if heuristic == "distressed":
        sentiment = "distressed"

    state.sentiment = sentiment  # type: ignore[assignment]
    state.tone_profile = TONE_BY_SENTIMENT[sentiment]  # type: ignore[assignment]
