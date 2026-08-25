"""Knowledge Agent (§9).

Least privilege in action: this module imports no claim repository and holds no
tool that could read a claim record. A prompt injection that captures it still
cannot reach customer data.
"""
from __future__ import annotations

from app.agents.state import GraphState
# Pure data and pure functions, no claim access — see the note in that module.
# The least-privilege property of this agent still holds.
from app.documents import guidance as doc_guidance
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway
from app.rag import retriever

DONT_KNOW = (
    "I don't have that in my guidance notes, so I'd rather not guess. "
    "Would you like me to put you through to a colleague who can answer properly?"
)


def run(state: GraphState) -> GraphState:
    retrieval = retriever.search_with_floor(state.message, top_k=4)

    if not retrieval["grounded"]:
        # "How do I photograph my licence?" is one of the assistant's own
        # suggested questions and the answer is a known constant, not something
        # to retrieve. Offering a colleague for it — which is what happened
        # while this text lived only in the document agent — reads as the
        # assistant not knowing its own instructions.
        how_to = doc_guidance.answer_for(state.message)
        if how_to:
            state.facts = {
                "retrieval": "answered_from_document_guidance",
                "doc_type": doc_guidance.doc_type_for(state.message),
                "guidance": how_to,
            }
            state.draft = how_to
            return state

        state.facts = {
            "retrieval": "no_relevant_passages",
            "best_score": retrieval["best_score"],
            "floor": retrieval["floor"],
        }
        state.draft = DONT_KNOW
        state.guardrail_flags.append("rag_below_floor")
        return state

    hits = retrieval["hits"]
    passages = "\n\n".join(
        f"[{i + 1}] ({hit['title']}) {hit['content']}" for i, hit in enumerate(hits)
    )

    result = gateway.complete(
        "knowledge_answer",
        {"question": wrap_untrusted("customer_message", state.message),
         "passages": wrap_untrusted("retrieved", passages)},
        tier="primary",
        trace_id=state.trace_id,
        fallback=_template_answer(hits),
    )

    state.draft = result.text.strip()
    state.degraded = state.degraded or result.degraded
    state.citations = [
        {"n": i + 1, "title": hit["title"], "chunk_id": hit["chunk_id"],
         "score": hit["score"]}
        for i, hit in enumerate(hits)
    ]
    state.facts = {
        "retrieved_passages": [
            {"n": i + 1, "title": hit["title"], "content": hit["content"],
             "score": hit["score"]}
            for i, hit in enumerate(hits)
        ]
    }
    state.cards.append({
        "card_type": "citations",
        "payload": {"items": state.citations},
    })
    return state


def _template_answer(hits: list[dict]) -> str:
    """No LLM? Quote the best passage verbatim rather than saying nothing."""
    best = hits[0]
    return (
        f"Here's what our guidance says about that:\n\n{best['content']}\n\n"
        f"(From: {best['title']}.) Would you like me to explain any part of that?"
    )
