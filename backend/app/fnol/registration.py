"""Registration bot — puts a verified FNOL onto the core system (POLARIS).

This is deliberately a *browser* automation rather than an API call. The core
system is the kind of legacy application that has no integration surface, which
is exactly why insurers put bots in front of them; driving the real UI is what
makes this an honest simulation rather than an animation.

Playwright drives a headless Chrome against ``/core-system``, typing into the
same form a human would use. Screenshots are captured as it goes so a reviewer
can watch, and every step is recorded on ``rpa_run`` for audit.

If Playwright isn't available the run degrades to a deterministic simulation —
the workflow still completes and the claim is still registered (§10, UC-N7).
"""
from __future__ import annotations

import base64
import json
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query_one
from app.fnol import intake

# The bot's own view of the core system. Same origin as the app, because the
# portal is served by this process.
PORTAL_PATH = "/core-system"

ADJUSTERS = {"motor": "M.BENNETT", "home": "R.OKAFOR", "health": "S.PATEL"}
LOSS_TYPES = {"motor": "MOTOR", "home": "HOME", "health": "HEALTH"}
PERILS = {
    "collision": "MOT-COL-01", "theft": "MOT-THF-02", "vandalism": "MOT-VAN-03",
    "weather": "MOT-WTH-04", "escape_of_water": "HOM-EOW-01", "fire": "HOM-FIR-02",
    "storm": "HOM-STM-03", "consultation": "MED-CON-01", "procedure": "MED-PRC-02",
    "diagnostic": "MED-DIA-03", "emergency": "MED-EMG-04",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Run record
# --------------------------------------------------------------------------
def get_run(run_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM rpa_run WHERE id = ?", (run_id,))
    if row is None:
        return None
    data = dict(row)
    data["steps"] = json.loads(data.get("steps") or "[]")
    data["frames"] = json.loads(data.get("frames") or "[]")
    data["result"] = json.loads(data.get("result") or "{}")
    return data


def latest_run(fnol_id: str) -> dict[str, Any] | None:
    row = query_one(
        "SELECT id FROM rpa_run WHERE fnol_id = ? ORDER BY started_at DESC LIMIT 1",
        (fnol_id,),
    )
    return get_run(row["id"]) if row else None


class _Recorder:
    """Accumulates steps and frames, flushing each one to the DB immediately.

    The staff console polls this row, so a step is only useful if it is visible
    while the run is still going — batching to the end would defeat the point.
    """

    # Frames are base64 PNGs in a JSON column. Keeping every one of a 20-step run
    # would bloat the row into megabytes, so only the most recent are retained.
    MAX_FRAMES = 14

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.steps: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []

    def step(self, label: str, detail: str = "", state: str = "done") -> None:
        self.steps.append({"label": label, "detail": detail, "state": state,
                           "at": _now()})
        self._flush()

    def frame(self, png: bytes, label: str) -> None:
        self.frames.append({
            "label": label,
            "at": _now(),
            "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        })
        del self.frames[:-self.MAX_FRAMES]
        self._flush()

    def _flush(self) -> None:
        execute("UPDATE rpa_run SET steps = ?, frames = ? WHERE id = ?",
                (json.dumps(self.steps), json.dumps(self.frames), self.run_id))


# --------------------------------------------------------------------------
# Field mapping
# --------------------------------------------------------------------------
def _form_values(record: dict[str, Any], customer: dict[str, Any],
                 policy: dict[str, Any] | None) -> dict[str, str]:
    """Translate FNOL answers into what the core system's form expects."""
    answers = record["answers"]
    claim_type = record.get("claim_type") or "motor"

    amount = answers.get("estimated_amount")
    if not amount:
        # The core system will not open a claim with no reserve, so a nominal
        # figure is entered and the adjuster revises it. This mirrors what a
        # human registrar does when the customer cannot yet estimate.
        amount = intake.DEFAULT_RESERVE.get(claim_type, 1000.0)

    return {
        "policy_number": (policy or {}).get("policy_number", ""),
        "claimant_name": customer.get("full_name", ""),
        "loss_type": LOSS_TYPES.get(claim_type, "MOTOR"),
        "peril": PERILS.get(str(answers.get("incident_kind")
                                or answers.get("treatment_type")), "GEN-OTH-99"),
        "date_of_loss": str(answers.get("incident_date") or date.today().isoformat()),
        "date_reported": date.today().isoformat(),
        "description": str(answers.get("description") or "")[:400],
        "reserve": f"{float(amount):.2f}",
        "adjuster": ADJUSTERS.get(claim_type, "M.BENNETT"),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def start(fnol_id: str, started_by: str) -> str:
    """Kick off a registration run in the background; returns the run id."""
    record = intake.get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)
    if record["status"] not in ("READY_TO_REGISTER", "UNDER_REVIEW", "SUBMITTED"):
        raise ValueError(f"FNOL {record['reference']} is {record['status']}, not ready")

    run_id = str(uuid.uuid4())
    execute(
        """INSERT INTO rpa_run (id, fnol_id, status, steps, frames, result,
                                started_by, started_at)
           VALUES (?, ?, 'RUNNING', '[]', '[]', '{}', ?, ?)""",
        (run_id, fnol_id, started_by, _now()),
    )
    intake.set_status(fnol_id, "REGISTERING")

    audit.record("rpa_run_started", actor_type="staff", actor_id=started_by,
                 entity_type="fnol", entity_id=fnol_id,
                 payload={"run_id": run_id, "reference": record["reference"]})

    threading.Thread(target=_run, args=(run_id, fnol_id), daemon=True).start()
    return run_id


def _run(run_id: str, fnol_id: str) -> None:
    recorder = _Recorder(run_id)
    try:
        result = _drive(recorder, fnol_id)
        claim_id = _materialise(fnol_id, result)
        result["claim_id"] = claim_id

        execute(
            """UPDATE rpa_run SET status = 'SUCCEEDED', result = ?, finished_at = ?
               WHERE id = ?""",
            (json.dumps(result), _now(), run_id),
        )
        intake.set_status(fnol_id, "REGISTERED", claim_id=claim_id)
        audit.record("rpa_run_succeeded", actor_type="agent", actor_id="registration_bot",
                     entity_type="fnol", entity_id=fnol_id,
                     payload={"run_id": run_id, "claim_number": result.get("claim_number"),
                              "mode": result.get("mode")})
        _notify_customer(fnol_id, result)
    except Exception as exc:  # noqa: BLE001 — a bot failure must never crash the app
        recorder.step("Registration failed", str(exc)[:200], state="error")
        execute(
            """UPDATE rpa_run SET status = 'FAILED', error = ?, finished_at = ?
               WHERE id = ?""",
            (str(exc)[:500], _now(), run_id),
        )
        # Back to the reviewer's queue rather than stuck in REGISTERING.
        intake.set_status(fnol_id, "READY_TO_REGISTER")
        audit.record("rpa_run_failed", actor_type="agent", actor_id="registration_bot",
                     entity_type="fnol", entity_id=fnol_id,
                     payload={"run_id": run_id, "error": str(exc)[:300]})


# --------------------------------------------------------------------------
# The automation itself
# --------------------------------------------------------------------------
def _drive(recorder: _Recorder, fnol_id: str) -> dict[str, Any]:
    record = intake.get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)

    customer = dict(query_one("SELECT * FROM customer WHERE id = ?",
                              (record["customer_id"],)) or {})
    policy = query_one(
        "SELECT * FROM policy WHERE customer_id = ? ORDER BY start_date DESC LIMIT 1",
        (record["customer_id"],),
    )
    policy = dict(policy) if policy else None
    if policy:
        intake.attach_policy(fnol_id, policy["id"])

    values = _form_values(record, customer, policy)

    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        recorder.step("Browser automation unavailable",
                      "Playwright is not installed — running in simulation mode.",
                      state="warn")
        return _simulate(recorder, values)

    return _drive_browser(recorder, values)


def _last_claim_sequence() -> int:
    """Highest claim number issued so far.

    The core system continues the insurer's single numbering sequence, so it has
    to start from what is already on the books — otherwise it re-issues a number
    that belongs to an existing claim.
    """
    row = query_one(
        "SELECT MAX(CAST(SUBSTR(claim_number, 5) AS INTEGER)) AS n FROM claim"
    )
    return int(row["n"]) if row and row["n"] else 88410


def _launch(pw: Any, recorder: _Recorder) -> Any:
    """Real Chrome where the machine has it, Playwright's own build otherwise.

    ``channel="chrome"`` puts the browser an audience recognises on screen, but
    that distribution only exists where Google Chrome is installed — elsewhere
    the launch raises and the whole registration fails. Playwright's bundled
    Chromium drives the form identically, so falling back keeps this a genuine
    browser run rather than dropping to simulation over a missing binary.
    """
    args = ["--force-device-scale-factor=1"]
    try:
        return pw.chromium.launch(channel="chrome", args=args)
    except Exception:  # noqa: BLE001 - any launch failure means "no Chrome here"
        recorder.step("Using bundled Chromium",
                      "Google Chrome is not installed on this host; Playwright's "
                      "own Chromium build drives the form instead.")
        return pw.chromium.launch(args=args)


def _drive_browser(recorder: _Recorder, values: dict[str, str]) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    settings = get_settings()
    base = settings.rpa_portal_url

    recorder.step("Launching browser", "Headless Chrome, RPA service account")
    with sync_playwright() as pw:
        browser = _launch(pw, recorder)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 900})
            page.goto(f"{base}{PORTAL_PATH}?seq={_last_claim_sequence()}",
                      wait_until="domcontentloaded", timeout=30000)
            recorder.step("Opened POLARIS", "Claims Management System v4.2.1")
            recorder.frame(page.screenshot(), "Core system loaded")

            def fill(selector: str, value: str, label: str) -> None:
                """Type into one field, highlighting it so the run is watchable."""
                if not value:
                    return
                page.eval_on_selector(selector, "el => el.classList.add('bot-active')")
                # Typed rather than set: the core system's own handlers fire on
                # input, and a bot that bypasses them would miss its validation.
                page.fill(selector, "")
                page.type(selector, value, delay=22)
                recorder.step(label, value[:90])
                recorder.frame(page.screenshot(), label)
                page.eval_on_selector(selector, "el => el.classList.remove('bot-active')")

            fill("#policy_number", values["policy_number"], "Entered policy number")
            fill("#claimant_name", values["claimant_name"], "Entered claimant")

            page.click("#verify_btn")
            page.wait_for_timeout(420)
            verified = page.inner_text("#policy_status")
            recorder.step("Verified policy", verified,
                          state="done" if "VERIFIED" in verified else "error")
            recorder.frame(page.screenshot(), "Policy verification")
            if "VERIFIED" not in verified:
                raise RuntimeError(f"Core system rejected the policy: {verified}")

            page.select_option("#loss_type", values["loss_type"])
            recorder.step("Selected loss type", values["loss_type"])
            fill("#peril", values["peril"], "Entered peril code")
            fill("#date_of_loss", values["date_of_loss"], "Entered date of loss")
            fill("#date_reported", values["date_reported"], "Entered date reported")
            fill("#description", values["description"], "Entered loss description")
            fill("#reserve", values["reserve"], "Set initial reserve")

            page.select_option("#adjuster", values["adjuster"])
            recorder.step("Assigned adjuster", values["adjuster"])
            recorder.frame(page.screenshot(), "Reserve and assignment")

            page.click("#register_btn")
            page.wait_for_timeout(650)

            submit_status = page.inner_text("#submit_status")
            if "REGISTERED" not in submit_status:
                recorder.frame(page.screenshot(), "Registration rejected")
                raise RuntimeError(f"Core system rejected the submission: {submit_status}")

            page.wait_for_selector("#issued_claim_number", timeout=10000)
            claim_number = page.inner_text("#issued_claim_number").strip()
            adjuster = page.inner_text("#issued_adjuster").strip()

            recorder.step("Claim registered", claim_number)
            recorder.frame(page.screenshot(), "Registration receipt")

            return {
                "mode": "browser",
                "claim_number": claim_number,
                "adjuster": adjuster,
                "reserve": float(values["reserve"]),
                "loss_type": values["loss_type"],
                "date_of_loss": values["date_of_loss"],
                "policy_number": values["policy_number"],
            }
        finally:
            browser.close()


def _simulate(recorder: _Recorder, values: dict[str, str]) -> dict[str, Any]:
    """Deterministic fallback when no browser is available.

    Produces the same result shape so the rest of the workflow — and the staff
    view — behaves identically; only the frames are missing.
    """
    import time

    for label, detail in (
        ("Opened POLARIS", "Claims Management System v4.2.1"),
        ("Entered policy number", values["policy_number"]),
        ("Entered claimant", values["claimant_name"]),
        ("Verified policy", "VERIFIED · IN FORCE"),
        ("Selected loss type", values["loss_type"]),
        ("Entered peril code", values["peril"]),
        ("Entered date of loss", values["date_of_loss"]),
        ("Entered loss description", values["description"][:90]),
        ("Set initial reserve", values["reserve"]),
        ("Assigned adjuster", values["adjuster"]),
    ):
        recorder.step(label, detail)
        time.sleep(0.45)

    claim_number = f"CLM-{_last_claim_sequence() + 1}"
    recorder.step("Claim registered", claim_number)

    return {
        "mode": "simulated",
        "claim_number": claim_number,
        "adjuster": values["adjuster"],
        "reserve": float(values["reserve"]),
        "loss_type": values["loss_type"],
        "date_of_loss": values["date_of_loss"],
        "policy_number": values["policy_number"],
    }


# --------------------------------------------------------------------------
# Materialise the registered claim
# --------------------------------------------------------------------------
# What the customer must supply once a claim exists. Registration opens the
# claim; these are what move it out of DOCS_PENDING.
REQUIRED_DOCS = {
    "motor": [("claim_form", 1), ("driving_licence", 1), ("police_report", 1),
              ("repair_invoice", 1)],
    "home": [("claim_form", 1), ("damage_photos", 1), ("repair_quote", 1)],
    "health": [("claim_form", 1), ("medical_report", 1), ("treatment_invoice", 1)],
}


def _materialise(fnol_id: str, result: dict[str, Any]) -> str:
    """Create the real claim row the core system just registered.

    Everything downstream — My Claims, the checklist, uploads, the status agent —
    reads ``claim``, so the claim only becomes real to the product here.
    """
    record = intake.get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)

    policy = query_one("SELECT * FROM policy WHERE id = ?", (record["policy_id"],)) \
        if record.get("policy_id") else None
    if policy is None:
        policy = query_one(
            "SELECT * FROM policy WHERE customer_id = ? ORDER BY start_date DESC LIMIT 1",
            (record["customer_id"],),
        )
    if policy is None:
        raise RuntimeError("No policy on file for this customer")

    claim_id = str(uuid.uuid4())
    claim_type = record.get("claim_type") or "motor"
    answers = record["answers"]
    incident_date = str(answers.get("incident_date") or date.today().isoformat())
    now = _now()

    # A newly registered claim is awaiting documentation, exactly as the core
    # system's receipt says — not FILED, which would imply nothing is outstanding.
    # What the customer told us during intake, which may be nothing — the field
    # is optional and "I don't know yet" is a legitimate answer at FNOL.
    estimated = answers.get("estimated_amount")
    claimed = float(estimated) if isinstance(estimated, (int, float)) else None

    execute(
        """INSERT INTO claim (id, policy_id, claim_number, claim_type, subtype,
                              status, claimed_amount, incident_date, filed_at,
                              predicted_settlement_date, prediction_confidence,
                              handler, reserve_amount)
           VALUES (?,?,?,?,?, 'DOCS_PENDING', ?,?,?,?,?,?,?)""",
        (claim_id, policy["id"], result["claim_number"], claim_type,
         answers.get("incident_kind") or answers.get("treatment_type"),
         # The customer's own estimate. The reserve is our internal provision
         # and is stored separately: writing it here made an insurer-set figure
         # look like the amount the customer had asked for.
         claimed, incident_date, now,
         (date.today() + timedelta(days=32)).isoformat(), 0.4,
         result.get("adjuster"), result.get("reserve")),
    )

    for from_status, to_status, reason in (
        (None, "FILED", f"Registered on POLARIS from {record['reference']}"),
        ("FILED", "DOCS_PENDING", "Awaiting supporting documents"),
    ):
        execute(
            """INSERT INTO claim_status_history (id, claim_id, from_status, to_status,
                                                 reason, actor_type, changed_at)
               VALUES (?,?,?,?,?, 'system', ?)""",
            (str(uuid.uuid4()), claim_id, from_status, to_status, reason, now),
        )

    for doc_type, mandatory in REQUIRED_DOCS.get(claim_type, REQUIRED_DOCS["motor"]):
        execute(
            """INSERT INTO required_document (id, claim_id, doc_type, mandatory, state)
               VALUES (?,?,?,?, 'MISSING')""",
            (str(uuid.uuid4()), claim_id, doc_type, mandatory),
        )

    # Carry intake uploads onto the claim so the customer never re-sends what
    # they already gave us. They enter the normal pipeline from UPLOADED.
    for attachment in intake.documents(fnol_id):
        execute(
            """INSERT INTO document (id, claim_id, filename, status, storage_key,
                                     extracted_fields, uploaded_at)
               VALUES (?,?,?, 'UPLOADED', ?, '{}', ?)""",
            (str(uuid.uuid4()), claim_id, attachment["filename"],
             attachment["storage_key"], attachment["uploaded_at"]),
        )

    audit.record("claim_registered", actor_type="agent", actor_id="registration_bot",
                 entity_type="claim", entity_id=claim_id,
                 payload={"fnol": record["reference"], "claim_number": result["claim_number"],
                          "adjuster": result.get("adjuster"), "reserve": result.get("reserve")})
    return claim_id


# --------------------------------------------------------------------------
# Tell the customer, in the thread they already have open
# --------------------------------------------------------------------------
def _notify_customer(fnol_id: str, result: dict[str, Any]) -> None:
    """Post the good news into the customer's existing conversation.

    The whole product promise is one continuous thread — so this lands as an
    assistant message in the conversation the notification came from, not as a
    separate alert the customer has to go and find.
    """
    record = intake.get(fnol_id)
    if record is None:
        return

    body = (
        f"Good news — the claim you reported as **{record['reference']}** is now "
        f"registered. Its claim number is **{result['claim_number']}**, and you'll "
        f"find it under *My claims*.\n\n"
        f"{result.get('adjuster', 'An adjuster')} is looking after it, and we've opened it "
        f"with an initial reserve of £{float(result.get('reserve') or 0):,.2f}. "
        f"The next step is the supporting documents — I've listed what's needed on the "
        f"claim, and you can upload them whenever you're ready.\n\n"
        f"From here on, quote **{result['claim_number']}** when you get in touch."
    )

    conversation_id = record.get("conversation_id")
    if conversation_id:
        row = query_one("SELECT id FROM conversation WHERE id = ?", (conversation_id,))
        if row:
            execute(
                """INSERT INTO message (id, conversation_id, role, content, created_at)
                   VALUES (?,?, 'assistant', ?, ?)""",
                (str(uuid.uuid4()), conversation_id, body, _now()),
            )

    execute(
        """INSERT INTO notification (id, customer_id, claim_id, kind, channel, body, sent_at)
           VALUES (?,?,?, 'claim_registered', 'in_app', ?, ?)""",
        (str(uuid.uuid4()), record["customer_id"], result.get("claim_id"), body, _now()),
    )
    audit.record("customer_notified", actor_type="agent", actor_id="registration_bot",
                 entity_type="fnol", entity_id=fnol_id,
                 payload={"claim_number": result["claim_number"],
                          "conversation_id": conversation_id})
