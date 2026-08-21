"""Hybrid retrieval over the knowledge base.

MVP uses pure-Python BM25 (no embedding provider, no pgvector). The
``search_knowledge`` signature is what the Knowledge Agent depends on, so
swapping in pgvector + embeddings later changes only this module (§10).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.db import query

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "of", "to", "in", "on", "for", "and", "or", "it", "this", "that",
    "my", "me", "i", "you", "your", "what", "how", "when", "why", "can", "will",
    "with", "at", "as", "if", "so", "but", "from", "by", "about", "have", "has",
}

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if t and t not in STOPWORDS and len(t) > 1]


@lru_cache(maxsize=1)
def _index() -> dict[str, Any]:
    rows = query(
        """SELECT c.id, c.content, c.metadata, d.title, d.doc_class
           FROM kb_chunk c JOIN kb_document d ON d.id = c.kb_document_id"""
    )
    chunks = []
    df: Counter[str] = Counter()
    for row in rows:
        tokens = tokenize(row["content"] + " " + (row["title"] or ""))
        chunks.append({
            "id": row["id"], "content": row["content"], "title": row["title"],
            "doc_class": row["doc_class"], "tokens": tokens,
            "tf": Counter(tokens), "length": len(tokens),
            "metadata": json.loads(row["metadata"] or "{}"),
        })
        df.update(set(tokens))

    total = len(chunks) or 1
    avg_len = sum(c["length"] for c in chunks) / total if chunks else 1.0
    idf = {
        term: math.log(1 + (total - count + 0.5) / (count + 0.5))
        for term, count in df.items()
    }
    return {"chunks": chunks, "idf": idf, "avg_len": avg_len or 1.0}


def refresh() -> None:
    _index.cache_clear()


def search(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """BM25 ranking with a normalised relevance score in [0, 1]."""
    index = _index()
    chunks = index["chunks"]
    if not chunks:
        return []

    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    scored = []
    for chunk in chunks:
        score = 0.0
        for term in q_tokens:
            tf = chunk["tf"].get(term, 0)
            if not tf:
                continue
            idf = index["idf"].get(term, 0.0)
            denom = tf + K1 * (1 - B + B * chunk["length"] / index["avg_len"])
            score += idf * (tf * (K1 + 1)) / denom
        if score > 0:
            scored.append((score, chunk))

    if not scored:
        return []

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Normalise against the theoretical max for this query so the floor is stable.
    max_possible = sum(index["idf"].get(term, 0.0) for term in q_tokens) or 1.0

    results = []
    for score, chunk in scored[:top_k]:
        results.append({
            "chunk_id": chunk["id"],
            "title": chunk["title"],
            "content": chunk["content"],
            "doc_class": chunk["doc_class"],
            "score": round(min(1.0, score / max_possible), 3),
        })
    return results


def search_with_floor(question: str, top_k: int = 4) -> dict[str, Any]:
    """Apply the relevance floor: below it we say 'I don't know' (UC-N6)."""
    floor = get_settings().rag_relevance_floor
    hits = search(question, top_k)
    passing = [h for h in hits if h["score"] >= floor]
    return {
        "hits": passing,
        "best_score": hits[0]["score"] if hits else 0.0,
        "floor": floor,
        "grounded": bool(passing),
    }
