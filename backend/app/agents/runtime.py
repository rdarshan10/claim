"""Shared LangGraph runtime: the checkpointer every graph compiles against.

One SQLite file holds the checkpoints of every graph (threads are namespaced
``conv:…``, ``aiops:…``), kept apart from the business database so checkpoint
churn never contends with claim writes.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.config import get_settings

_lock = threading.Lock()
_savers: dict[str, Any] = {}


def checkpointer(kind: str | None = None) -> Any | None:
    """The process-wide checkpointer of the requested kind (default: configured).

    Returns ``None`` when checkpointing is switched off.
    """
    kind = (kind or get_settings().graph_checkpointer).lower()
    if kind == "none":
        return None
    with _lock:
        if kind in _savers:
            return _savers[kind]
        if kind == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver

            path = Path(get_settings().graph_checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Graph turns run on worker threads (asyncio.to_thread); the saver
            # serialises access with its own lock.
            conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            saver = SqliteSaver(conn)
            saver.setup()
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            saver = InMemorySaver()
        _savers[kind] = saver
        return saver


def reset() -> None:
    """Forget cached savers (tests point GRAPH_CHECKPOINT_PATH somewhere new)."""
    with _lock:
        for saver in _savers.values():
            conn = getattr(saver, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        _savers.clear()


def thread_history(thread_id: str, graph: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Every checkpoint of one thread, newest first — node-by-node replay."""
    config = {"configurable": {"thread_id": thread_id}}
    out: list[dict[str, Any]] = []
    for snapshot in graph.get_state_history(config, limit=limit):
        values = snapshot.values or {}
        out.append({
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "created_at": snapshot.created_at,
            "step": (snapshot.metadata or {}).get("step"),
            "source": (snapshot.metadata or {}).get("source"),
            "next": list(snapshot.next),
            "values": {k: values.get(k) for k in (
                "trace_id", "message", "intent", "intent_confidence", "sentiment",
                "route", "reply", "guardrail_flags", "degraded", "regenerated",
                "blocked", "output_attempt", "status", "summary") if k in values},
        })
    return out
