"""Typed graph state (§9).

Mirrors the LangGraph ``TypedDict`` state so migrating to LangGraph is a wiring
change, not a redesign.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Intent = Literal["claim_status", "documents", "knowledge", "human_request", "out_of_scope"]
Sentiment = Literal["calm", "confused", "frustrated", "distressed"]
ToneProfile = Literal[
    "neutral-warm", "reassuring", "apologetic-accountable", "celebratory", "gentle-supportive"
]


@dataclass
class GraphState:
    # --- identity (from JWT only, never from message text) --------------
    customer_id: str
    customer_name: str = ""
    conversation_id: str = ""
    trace_id: str = ""

    # --- turn inputs ----------------------------------------------------
    message: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    active_claim_id: str | None = None

    # --- routing / classification --------------------------------------
    intent: Intent | None = None
    intent_confidence: float = 0.0
    sentiment: Sentiment = "calm"
    tone_profile: ToneProfile = "neutral-warm"

    # --- agent outputs --------------------------------------------------
    facts: dict[str, Any] = field(default_factory=dict)
    cards: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    draft: str = ""
    reply: str = ""

    # --- control --------------------------------------------------------
    guardrail_flags: list[str] = field(default_factory=list)
    blocked: bool = False
    escalation_ticket_id: str | None = None
    turn_hop_count: int = 0
    degraded: bool = False
    regenerated: bool = False

    def hop(self, node: str) -> bool:
        """Enforce the per-turn agent hop budget (§9)."""
        from app.config import get_settings

        self.turn_hop_count += 1
        return self.turn_hop_count <= get_settings().max_turn_hops

    def to_audit(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "sentiment": self.sentiment,
            "tone_profile": self.tone_profile,
            "guardrail_flags": self.guardrail_flags,
            "blocked": self.blocked,
            "hops": self.turn_hop_count,
            "degraded": self.degraded,
            "regenerated": self.regenerated,
            "cards": [c.get("card_type") for c in self.cards],
        }
