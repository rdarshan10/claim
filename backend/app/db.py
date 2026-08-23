"""SQLite access layer.

MVP stands in for PostgreSQL 16 + pgvector. Everything goes through the
repository classes in ``app.repositories`` so the engine can be swapped without
touching business logic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.config import get_settings

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS customer (
  id TEXT PRIMARY KEY,
  full_name TEXT NOT NULL,
  email_enc TEXT NOT NULL,
  email_hmac TEXT NOT NULL UNIQUE,
  phone_enc TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customer(id),
  policy_number TEXT NOT NULL UNIQUE,
  product_type TEXT NOT NULL,
  coverage_limit REAL NOT NULL,
  start_date TEXT NOT NULL,
  end_date TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim (
  id TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL REFERENCES policy(id),
  claim_number TEXT NOT NULL UNIQUE,
  claim_type TEXT NOT NULL,
  subtype TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'FILED','DOCS_PENDING','IN_ASSESSMENT','ADDITIONAL_INFO','APPROVED',
    'PAYMENT_IN_PROGRESS','SETTLED','REJECTED','WITHDRAWN')),
  claimed_amount REAL,
  approved_amount REAL,
  incident_date TEXT NOT NULL,
  filed_at TEXT NOT NULL,
  settled_at TEXT,
  predicted_settlement_date TEXT,
  prediction_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_claim_policy ON claim(policy_id);

CREATE TABLE IF NOT EXISTS claim_status_history (
  id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL REFERENCES claim(id),
  from_status TEXT,
  to_status TEXT NOT NULL,
  reason TEXT,
  actor_type TEXT NOT NULL,
  changed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_claim ON claim_status_history(claim_id);

CREATE TABLE IF NOT EXISTS required_document (
  id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL REFERENCES claim(id),
  doc_type TEXT NOT NULL,
  mandatory INTEGER NOT NULL DEFAULT 1,
  state TEXT NOT NULL DEFAULT 'MISSING'
);
CREATE INDEX IF NOT EXISTS idx_reqdoc_claim ON required_document(claim_id);

CREATE TABLE IF NOT EXISTS document (
  id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL REFERENCES claim(id),
  filename TEXT NOT NULL,
  doc_type TEXT,
  status TEXT NOT NULL,
  storage_key TEXT,
  sha256 TEXT,
  ocr_quality REAL,
  classification_conf REAL,
  extraction_conf REAL,
  extracted_fields TEXT NOT NULL DEFAULT '{}',
  rejection_code TEXT,
  rejection_payload TEXT,
  uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_claim ON document(claim_id);
CREATE INDEX IF NOT EXISTS idx_document_sha ON document(sha256);

CREATE TABLE IF NOT EXISTS document_validation (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES document(id),
  rule_id TEXT NOT NULL,
  passed INTEGER NOT NULL,
  details TEXT NOT NULL DEFAULT '{}',
  run_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fraud_signal (
  id TEXT PRIMARY KEY,
  claim_id TEXT NOT NULL REFERENCES claim(id),
  document_id TEXT,
  signal_type TEXT NOT NULL,
  explanation TEXT NOT NULL,
  severity REAL NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'PENDING',
  raised_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customer(id),
  channel TEXT NOT NULL DEFAULT 'web',
  started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS message (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversation(id),
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  intent TEXT,
  sentiment TEXT,
  citations TEXT NOT NULL DEFAULT '[]',
  prompt_version TEXT,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_conv ON message(conversation_id);

CREATE TABLE IF NOT EXISTS escalation_ticket (
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
  claim_id TEXT,
  customer_id TEXT NOT NULL,
  priority TEXT NOT NULL,
  reason TEXT NOT NULL,
  context_packet TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'OPEN',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notification (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  claim_id TEXT,
  kind TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'in_app',
  body TEXT NOT NULL,
  read INTEGER NOT NULL DEFAULT 0,
  sent_at TEXT NOT NULL
);

-- First Notice of Loss. The customer's intake sits here until a reviewer is
-- satisfied and the registration bot puts it on the core system; only then does
-- a row appear in `claim`. Keeping FNOL separate means an abandoned or rejected
-- notification never pollutes the claims book, and the record of *how* a claim
-- originated survives after registration.
CREATE TABLE IF NOT EXISTS fnol_request (
  id TEXT PRIMARY KEY,
  reference TEXT NOT NULL UNIQUE,
  customer_id TEXT NOT NULL REFERENCES customer(id),
  policy_id TEXT REFERENCES policy(id),
  conversation_id TEXT,
  claim_id TEXT REFERENCES claim(id),      -- set once registration succeeds
  status TEXT NOT NULL DEFAULT 'COLLECTING' CHECK (status IN (
    'COLLECTING','SUBMITTED','UNDER_REVIEW','INFO_REQUIRED',
    'READY_TO_REGISTER','REGISTERING','REGISTERED','REJECTED')),
  claim_type TEXT,
  answers TEXT NOT NULL DEFAULT '{}',      -- collected field -> value
  asked TEXT NOT NULL DEFAULT '[]',        -- fields already put to the customer
  review_note TEXT,
  reviewer TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fnol_customer ON fnol_request(customer_id);
CREATE INDEX IF NOT EXISTS idx_fnol_status ON fnol_request(status);

-- Documents attached during intake. Separate from `document` because those
-- require a claim_id, which does not exist yet at FNOL time. On registration
-- these are copied across to the new claim.
CREATE TABLE IF NOT EXISTS fnol_document (
  id TEXT PRIMARY KEY,
  fnol_id TEXT NOT NULL REFERENCES fnol_request(id),
  doc_type TEXT,
  filename TEXT NOT NULL,
  storage_key TEXT NOT NULL,
  content TEXT,
  uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fnoldoc_fnol ON fnol_document(fnol_id);

-- One registration bot run. Steps stream to the staff console as they happen,
-- and the row is the audit record of what the bot did on the core system.
CREATE TABLE IF NOT EXISTS rpa_run (
  id TEXT PRIMARY KEY,
  fnol_id TEXT NOT NULL REFERENCES fnol_request(id),
  status TEXT NOT NULL DEFAULT 'RUNNING' CHECK (status IN (
    'RUNNING','SUCCEEDED','FAILED')),
  steps TEXT NOT NULL DEFAULT '[]',
  frames TEXT NOT NULL DEFAULT '[]',       -- base64 screenshots, newest last
  result TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  started_by TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rpa_fnol ON rpa_run(fnol_id);

CREATE TABLE IF NOT EXISTS kb_document (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source TEXT,
  doc_class TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_chunk (
  id TEXT PRIMARY KEY,
  kb_document_id TEXT NOT NULL REFERENCES kb_document(id),
  content TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}'
);

-- Append-only, hash-chained. No UPDATE/DELETE is ever issued against this.
CREATE TABLE IF NOT EXISTS audit_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  actor_type TEXT NOT NULL,
  actor_id TEXT,
  event_type TEXT NOT NULL,
  entity_type TEXT,
  entity_id TEXT,
  payload TEXT NOT NULL DEFAULT '{}',
  prompt_version TEXT,
  model TEXT,
  trace_id TEXT,
  prev_hash TEXT NOT NULL,
  row_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_event(entity_id);

CREATE TABLE IF NOT EXISTS llm_call (
  id TEXT PRIMARY KEY,
  at TEXT NOT NULL,
  prompt_key TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model TEXT NOT NULL,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER NOT NULL DEFAULT 0,
  ok INTEGER NOT NULL DEFAULT 1,
  trace_id TEXT
);
"""


def connect() -> sqlite3.Connection:
    settings = get_settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Additive column migrations, applied idempotently on startup.
MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, DDL)
    ("conversation", "mode", "TEXT NOT NULL DEFAULT 'AI'"),
    ("conversation", "assigned_agent", "TEXT"),
    ("conversation", "ticket_id", "TEXT"),
    # When the customer last read the thread, so we can show them what arrived
    # while they were away.
    ("conversation", "last_seen_at", "TEXT"),
    ("message", "author_name", "TEXT"),
    ("message", "delivered_to_agent", "INTEGER NOT NULL DEFAULT 0"),
    # The reviewer's original words, kept beside what the customer was sent, so
    # any relay can be audited for drift.
    ("message", "source_note", "TEXT"),
    ("message", "relay_source", "TEXT"),
    ("escalation_ticket", "assigned_to", "TEXT"),
    ("escalation_ticket", "resolved_at", "TEXT"),
    # Who owns the claim. The registration bot picks an adjuster and the
    # customer is told their name, so it has to be queryable rather than living
    # only in the chat message and the bot's result blob.
    ("claim", "handler", "TEXT"),
    ("claim", "reserve_amount", "REAL"),
    # Served from the response cache rather than the provider. A real column,
    # not a suffix on the model name: this is the main cost lever and it has to
    # be groupable.
    ("llm_call", "cached", "INTEGER NOT NULL DEFAULT 0"),
    # A handler works a diary: every open claim carries a date it comes back to
    # them, and a note of what they are waiting for. Without this, a claim only
    # resurfaces when the customer chases — which is how claims sit untouched
    # for months and turn into complaints.
    ("claim", "next_review_date", "TEXT"),
    ("claim", "diary_note", "TEXT"),
    ("claim", "last_chased_at", "TEXT"),
    ("claim", "chase_count", "INTEGER NOT NULL DEFAULT 0"),
    # Set when the assistant offers to fetch a person, cleared on the next
    # turn. Without it a plain "yes" has no antecedent and gets classified as
    # a fresh question — the customer accepts the offer and is answered about
    # something else entirely.
    ("conversation", "offered_human_at", "TEXT"),
    # Set the first time the offer is made and never cleared. ``offered_human_at``
    # above is a one-turn signal and so cannot suppress anything: while a
    # customer stayed frustrated the offer was re-made on every single turn,
    # which reads as the assistant trying to hand them off rather than help.
    ("conversation", "human_offered_ever_at", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, ddl in MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    conn = connect()
    try:
        return conn.execute(sql, tuple(params)).fetchall()
    finally:
        conn.close()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> None:
    conn = connect()
    try:
        conn.execute(sql, tuple(params))
        conn.commit()
    finally:
        conn.close()
