"""Test fixtures.

Settings are read from the environment at import time, so the environment is
fixed here, before anything under ``app`` is imported: a throwaway database,
template mode (no LLM, fully deterministic), its own checkpoint file, and the
AIOps scheduler off so sweeps only happen when a test asks for one.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="cc-tests-"))

os.environ.update({
    "DB_PATH": str(_TMP / "cc.db"),
    "BLOB_DIR": str(_TMP / "blobs"),
    "GRAPH_CHECKPOINT_PATH": str(_TMP / "checkpoints.db"),
    "GRAPH_CHECKPOINTER": "sqlite",
    "LLM_ENABLED": "false",
    "GROQ_API_KEY": "",
    "LLM_API_KEY": "",
    "AIOPS_ENABLED": "false",
    "AIOPS_AUTO_REMEDIATE": "true",
    "RATE_LIMIT_PER_MINUTE": "10000",
})
for path in (ROOT / "backend", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402

_PRISTINE = _TMP / "pristine.db"


@pytest.fixture(scope="session", autouse=True)
def _dataset() -> None:
    """Generate the synthetic dataset once; every test starts from a copy."""
    from datagen import generate

    argv = sys.argv
    sys.argv = ["generate", "--seed", "42", "--customers", "6", "--claims-per", "2"]
    try:
        generate.main()
    finally:
        sys.argv = argv
    shutil.copy(os.environ["DB_PATH"], _PRISTINE)
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


def _settle_background_work(timeout: float = 20.0) -> None:
    """Playbooks start pipeline and registration threads. On Windows a thread
    still holding the database open stops the reset copy, so wait them out."""
    import threading
    import time

    deadline = time.monotonic() + timeout
    for thread in threading.enumerate():
        if thread is threading.current_thread() or not thread.daemon:
            continue
        if thread.name.startswith(("Thread-", "aiops")) and thread.is_alive():
            thread.join(max(0.0, deadline - time.monotonic()))


@pytest.fixture(autouse=True)
def fresh_db(_dataset):
    """Reset the business database and any runtime switches a test flipped."""
    from app.aiops import playbooks
    from app.config import get_settings
    from app.db import init_db
    from app.llm import gateway, providers

    _settle_background_work()
    for suffix in ("-wal", "-shm"):
        Path(os.environ["DB_PATH"] + suffix).unlink(missing_ok=True)
    shutil.copy(_PRISTINE, os.environ["DB_PATH"])
    init_db()
    get_settings().llm_enabled = False
    playbooks.STATE["template_mode_by_aiops"] = False
    providers.reset()
    gateway.clear_cache()
    yield


@pytest.fixture
def customer():
    from app.db import query_one

    row = query_one("SELECT id, full_name FROM customer ORDER BY id LIMIT 1")
    return dict(row)


@pytest.fixture
def conversation(customer):
    import uuid
    from datetime import datetime, timezone

    from app.db import execute

    cid = str(uuid.uuid4())
    execute("INSERT INTO conversation (id, customer_id, started_at) VALUES (?,?,?)",
            (cid, customer["id"], datetime.now(timezone.utc).isoformat()))
    return cid


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def token(role: str, name: str = "manager.elena") -> str:
    from app.security import jwt

    return jwt.encode({"sub": f"staff:{name}", "name": name, "role": role})
