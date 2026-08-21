# To be done — build status and increments

Tracks what is built, what is deliberately substituted, and how to finish each
piece. Section numbers refer to SOLUTION_DESIGN (`../instructions.md`).

**Status key:** ✅ done · 🟡 partial · ⬜ not started

---

## 1. Where the MVP stands

Phase 1 of the roadmap (§22) is complete and running. The vertical slice works
end to end: sign in → ask about a claim → see a grounded timeline → upload a bad
document → get an annotated rejection with fix steps → dispute it → review it in
the staff console → verify the audit chain.

Verified working (see the smoke run in the README):

- Cross-customer access returns 404, never 403 or data.
- Injection and out-of-scope input blocked in ~15 ms, pre-model, audited.
- A pre-incident invoice rejected by rule VR-02, explained in plain English, with
  3 annotation boxes and a 28 KB annotated PNG.
- Audit hash chain intact across 79 events.
- Agent role denied the audit trail (403); manager allowed.

---

## 2. Substitutions made, and how to undo them

Each of these is behind an interface, so undoing it is a swap, not a rewrite.

| § | Design calls for | MVP uses | How to swap |
|---|---|---|---|
| 5 | PostgreSQL 16 + pgvector | SQLite (`backend/app/db.py`) | Docker isn't available on this machine. Schema is portable; replace `db.py` with asyncpg/SQLAlchemy and port `SCHEMA` (already written with FK constraints and CHECKs). All access is via `repositories/`. |
| 5 | LangGraph orchestrator | Explicit state machine (`agents/graph.py`) | `pip install langgraph`. `GraphState` already mirrors the LangGraph `TypedDict`; each node is a `f(state) -> state`. Wire nodes into a `StateGraph`, add a Postgres checkpointer, and use `interrupt()` for the escalation pause. |
| 5 | Celery + Redis workers | `threading.Thread` per upload | `pipeline.process(doc_id)` is already a pure task body. Add `@celery.task` and change the call site in `api/v1/documents.py` to `.delay()`. |
| 5 | MinIO / S3 | Local disk (`data/blobs/`) | Replace the two `Path.read_bytes`/`write_bytes` call sites with an S3 client; add signed URL generation for `/documents/{id}/annotated`. |
| 10 | Tesseract 5 OCR | `TextOCRAdapter` over text-rendered documents | `ocr.py` defines the `OCRAdapter` protocol and `set_adapter()`. Write `TesseractAdapter.read()` returning the same `Word(text, bbox, confidence)` list — everything downstream (annotation geometry, quality map, thresholds) already consumes that shape. |
| 10 | pgvector hybrid retrieval | Pure-Python BM25 (`rag/retriever.py`) | The embedding model **is available**: `azure/genailab-maas-text-embedding-3-large` (configured as `llm_model_embedding`). Add an `embed()` to the gateway, store vectors, and fuse cosine + BM25 by reciprocal rank. Keep `search_with_floor()`'s signature. |
| 15 | React 18 + TypeScript | Streamlit (`frontend/app.py`) | Chosen for speed; the UI holds no business logic, so a React rewrite consumes the same REST/WS API unchanged. |
| 17 | AES-GCM field encryption | HMAC-keystream + blind index (`security/crypto.py`) | `encrypt`/`decrypt`/`blind_index` are the whole surface. Swap the bodies for AES-GCM via `cryptography` and pull the key from Key Vault. |
| 17 | ClamAV | Magic-byte + active-content checks (`pipeline.scan`) | Add a ClamAV daemon call inside `scan()`. |

---

## 3. What is genuinely not built yet

### Recently completed

- ✅ **Human-in-the-loop round trip** (the §9 `HumanInterrupt → resume` edge).
  Assistant-as-middleman: it carries the case to a reviewer, shows the customer
  what it passed on, keeps answering while they wait, and re-voices the
  reviewer's answer back into the same thread. Reviewer workspace has the
  thread, claim, checklist, signals and per-document verdict tools. Document
  decisions ping the customer's chat, closing the dispute loop. Re-asking for a
  human chases the open case instead of duplicating it.
- ✅ **Reviewer interface.** Copilot grounded to the case file (§20's "claims-agent
  copilot mode", pulled forward); full verification evidence per document — all
  eight rules with pass/fail and messages, confidence breakdown, extracted
  fields, raw document text, annotated page; and an "ask the customer" action
  that turns reviewer shorthand into an answerable question.
- ✅ **Relay log + demo view.** Every relay stores the reviewer's original words
  beside what was sent (`message.source_note`), exposed at
  `/staff/conversations/{id}/relay-log` and rendered side-by-side in the staff
  console's **Demo: the middleman** tab.
- ✅ **Intent routing fix** — "I want someone to actually look at this" is a human
  request without using the word "human"; the heuristics missed it and routed to
  `documents`. Added phrasing patterns, verified no false positives.
- ✅ **Model routing fixes** — merged intent+sentiment into one mini call,
  removed ladder-climbing for mini-tier calls, fast-fail first attempt, and an
  in-process response cache for classification and knowledge answers.

### Next increment (highest value first)

0. ⬜ **Handoff polish.** Known rough edges: the customer's chat polls on a
   6-second `sleep`+`rerun` loop (fine for a demo, wasteful in general — use
   `st.fragment(run_every=…)` or the WS endpoint); reviewers get no notification
   that a customer replied while they hold the case; there is no SLA timer or
   auto-nudge if a case sits unanswered; and the copilot re-sends the whole case
   file on every question rather than caching it against the thread.
   **Effort: S.**
0b. ⬜ **Proactive questioning.** The assistant asks the customer for what's
   missing when a *reviewer* tells it to, and lists outstanding documents when
   asked — but it never volunteers a question off its own back. A claim sitting
   in `DOCS_PENDING` for 81 days should prompt "shall I help you get the police
   report?" without anyone triggering it. Needs the event/notification loop
   (item 2) to have somewhere to fire from. **Effort: M.**
1. ⬜ **WebSocket streaming in the UI.** The backend endpoint
   `WS /chat/conversations/{id}/stream` is written and emits the full §14.2 frame
   contract (`token`/`card`/`citations`/`handoff`/`done`), but the Streamlit
   client calls the non-streaming POST. Streamlit can't hold a WS easily —
   either poll a generator with `st.write_stream`, or do this when the React UI
   lands. **Effort: S (UI only).**
2. ⬜ **Proactive notifications + event replay** (§13.3, UC-P4). The
   `notification` table exists and is unused. Needs: an event replay file, a
   consumer that advances claim statuses on a timer, and a notification bell.
   This is demo item 4 of the 7-step stage script. **Effort: M.**
3. 🟡 **Timeline prediction from sampled history.** Currently uses hard-coded
   per-claim-type dwell distributions (`services/timeline_prediction.py`). The
   design wants these *fitted* from the generated `claim_status_history`
   lognormals. The generator already emits the history. **Effort: S.**
4. ⬜ **Real embeddings + hybrid fusion** — see §2 table above. **Effort: M.**
5. ⬜ **Vision cross-check on ambiguous documents** (§10). `gpt-4.1` and
   `Llama-3.2-90B-Vision` are both available. Send the page image alongside OCR
   text when classification confidence lands in 0.5–0.7. **Effort: M.**

### Known gaps and defects

- 🟡 **Verdict accuracy is ~0.75 on the labelled set** (doc-type accuracy is
  1.0). Two failure modes seen: (a) a document labelled clean was rejected for
  `DATE_OUT_OF_RANGE` because the generator picks incident dates independently
  of the rendered document date; (b) a corrupted document still verified because
  the corruption didn't affect a field any rule checks. **Both are generator
  bugs, not pipeline bugs** — `datagen/generate.py` should derive document dates
  from the incident date and assert that each corruption maps to a rule that can
  actually catch it. Fix before quoting an accuracy number anywhere.
- ⬜ No CI pipeline. `evals/run_evals.py` returns a non-zero exit on gate
  failure and is ready to wire into GitHub Actions, but no workflow file exists.
- ⬜ No pytest suite. The evals cover guardrails, grounding and the document
  pipeline; unit tests for `rules.py`, `crypto.py`, `jwt.py` and the repository
  scoping are still missing. Design target is ≥ 80% coverage (§25).
- ⬜ No OpenTelemetry traces, Prometheus metrics or Grafana dashboards (§18).
  `trace_id` is threaded through the graph and audit log already, so wiring OTel
  is mostly decoration. `/admin/metrics/quality` and `/admin/metrics/costs`
  serve the numbers a dashboard would show.
- 🟡 **Rate limiting and idempotency keys are in-process dicts**
  (`api/deps.py`, `api/v1/documents.py`). Correct for one process, wrong the
  moment there are two. Move to Redis.
- ⬜ Token accounting is stubbed — `llm_call.tokens_in/out` are always 0 because
  the LangChain response metadata isn't unpacked. Latency and call counts are
  real. Cost figures therefore aren't available yet.
- ⬜ No PII redaction *reversal*. Redaction happens before model calls, which
  means a customer's own name is stripped from prompts where it would be
  harmless; pseudonymise-and-restore would read better (§17.1).
- ⬜ Refresh tokens aren't implemented — access tokens last an hour and that's
  it. OTP is hard-coded to `000000` by design for the demo.

### Deliberately out of scope for the MVP

Multilingual, voice (Whisper is available on the endpoint), WhatsApp/SMS,
Kubernetes, Kafka, core-system adapters, DSAR tooling, model cards, DPIA (§20,
§22 Phase 3).

---

## 4. Build order for the next session

Do them in this order; each is independently demoable.

1. Fix the two generator bugs, re-run `evals`, and get verdict accuracy honest.
2. Add the pytest suite around `rules.py` and repository scoping (fast, no LLM).
3. Event replay + notifications — completes the 7-step stage demo.
4. Fit the timeline distributions from generated history.
5. Embeddings + hybrid retrieval.
6. WS streaming (with the React UI, if that's the direction).
