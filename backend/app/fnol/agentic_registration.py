"""Registration by reading the form, not by knowing it in advance.

The deterministic bot in ``registration.py`` hardcodes POLARIS's selectors and
its enums::

    page.select_option("#loss_type", "MOTOR")
    page.click("#register_btn")

That is fast, reproducible and cheap, and it is the right tool for one core
system that does not change. It is also the whole integration: rename an id,
add a required field, or call the value ``AUTO`` instead of ``MOTOR`` and the
run fails. A second core system needs a second hand-written script.

This module is the other approach, kept deliberately separate so the two can be
compared side by side. It scrapes whatever form is on screen — labels, ids,
types, and crucially the real ``<option>`` values — asks the model which claim
value belongs in which field, then executes that mapping with Playwright.

The split matters: the model decides *what goes where*, once, and deterministic
code does the clicking. So the screenshots, the step log and the audit record
are exactly as they are for the scripted bot, and a run can still be replayed
and checked afterwards.

It finishes the job: on a confirmed registration it calls the scripted path's
``_materialise`` so the claim row, the customer's notification and the audit
entry all appear exactly as they would otherwise. A handler picking "agentic" in
the register switch must get a real claim out of it, not a filled-in form.

It reads the outcome off the page as text — a receipt saying REGISTERED with a
claim reference on it — rather than by element id, since knowing the receipt's
ids would be the same coupling this is trying to avoid.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Any

from app.audit import logger as audit
from app.config import get_settings
from app.db import execute, query_one
from app.fnol import intake, registration
from app.guardrails.input_guards import wrap_untrusted
from app.llm import gateway

# Read-only reuse of the scripted path. Importing these keeps the two runs
# genuinely comparable — same recorder, same run table, same claim numbering —
# without editing anything over there.
_Recorder = registration._Recorder
_now = registration._now
PORTAL_PATH = registration.PORTAL_PATH


# What the page tells us about itself. Everything the mapper reasons about comes
# from here, so a field the form does not expose simply cannot be filled.
_SCRAPE_FIELDS = """
() => [...document.querySelectorAll('input, select, textarea')]
  .filter(el => el.type !== 'hidden' && !el.disabled)
  .map(el => {
    const label = el.labels && el.labels[0] ? el.labels[0].innerText.trim() : null;
    const out = {
      id: el.id || null,
      name: el.name || null,
      label,
      type: el.tagName === 'SELECT' ? 'select'
          : el.tagName === 'TEXTAREA' ? 'textarea' : (el.type || 'text'),
      required: !!el.required,
      value: el.value || null,
    };
    if (el.tagName === 'SELECT') {
      out.options = [...el.options].map(o => o.value).filter(Boolean);
    }
    return out;
  })
  .filter(f => f.id || f.name)
"""

_SCRAPE_BUTTONS = """
() => [...document.querySelectorAll('button, input[type=submit]')]
  .filter(el => !el.disabled)
  .map(el => ({ id: el.id || null,
                text: (el.innerText || el.value || '').trim().slice(0, 60) }))
  .filter(b => b.id)
"""


def start(fnol_id: str, started_by: str) -> str:
    """Kick off an agentic run in the background; returns the run id.

    Same approval gate as the scripted bot: this opens a real claim in the core
    system, so it is not something to run on an untriaged notification.
    """
    record = intake.get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)
    # READY_TO_REGISTER only. This briefly allowed REGISTERED too, so the
    # experiment could be re-run on the same notification — harmless beside the
    # scripted bot, but once this became a mode on the real Register button it
    # meant a notification could be registered over and over, opening a fresh
    # claim each time.
    if record["status"] != "READY_TO_REGISTER":
        raise ValueError(
            f"FNOL {record['reference']} is {record['status']}. It must be "
            f"approved for registration first.")

    # A notification that already produced a claim is finished, whatever its
    # status says. Belt and braces against state drift, since the cost of
    # getting this wrong is a second claim on the books for one loss.
    if record.get("claim_id"):
        raise ValueError(
            f"FNOL {record['reference']} is already registered as a claim.")

    run_id = str(uuid.uuid4())
    execute(
        """INSERT INTO rpa_run (id, fnol_id, status, steps, frames, result,
                                started_by, started_at)
           VALUES (?, ?, 'RUNNING', '[]', '[]', '{}', ?, ?)""",
        (run_id, fnol_id, f"{started_by} (agentic)", _now()),
    )
    # Out of the reviewer's queue for the duration, exactly as the scripted bot
    # does. Without this the Register button stayed live while the run was in
    # flight and after it finished.
    intake.set_status(fnol_id, "REGISTERING")
    audit.record("agentic_run_started", actor_type="staff", actor_id=started_by,
                 entity_type="fnol", entity_id=fnol_id,
                 payload={"run_id": run_id, "reference": record["reference"],
                          "mode": "agentic"})
    threading.Thread(target=_run, args=(run_id, fnol_id), daemon=True).start()
    return run_id


def _run(run_id: str, fnol_id: str) -> None:
    recorder = _Recorder(run_id)
    try:
        result = _drive(recorder, fnol_id)
        # The claim only becomes real to the product here, exactly as it does on
        # the scripted path — otherwise picking "agentic" would fill the core
        # system's form and leave the customer with no claim.
        claim_id = registration._materialise(fnol_id, result)
        result["claim_id"] = claim_id
        recorder.step("Claim created in ClaimCompanion", result["claim_number"])
        execute("UPDATE rpa_run SET status = 'SUCCEEDED', result = ?, "
                "finished_at = ? WHERE id = ?",
                (json.dumps(result), _now(), run_id))
        # The same closing sequence as the scripted bot: the notification is
        # finished and carries its claim, and the customer is told. Leaving the
        # status behind was what let one notification be registered twice.
        intake.set_status(fnol_id, "REGISTERED", claim_id=claim_id)
        audit.record("agentic_run_succeeded", actor_type="agent",
                     actor_id="registration_bot", entity_type="fnol",
                     entity_id=fnol_id,
                     payload={"run_id": run_id, "mode": "agentic",
                              "claim_number": result.get("claim_number")})
        registration._notify_customer(fnol_id, result)
    except Exception as exc:  # noqa: BLE001 - the failure is the finding here
        recorder.step("Run failed", str(exc)[:300], state="error")
        execute("UPDATE rpa_run SET status = 'FAILED', result = ?, "
                "finished_at = ? WHERE id = ?",
                (json.dumps({"error": str(exc)[:500]}), _now(), run_id))
        # Back to the reviewer's queue rather than stuck in REGISTERING.
        intake.set_status(fnol_id, "READY_TO_REGISTER")
        audit.record("agentic_run_failed", actor_type="agent",
                     actor_id="registration_bot", entity_type="fnol",
                     entity_id=fnol_id,
                     payload={"run_id": run_id, "error": str(exc)[:300]})


def _claim_payload(fnol_id: str) -> dict[str, Any]:
    """The claim data, with no idea what the form will ask for.

    Deliberately not shaped to POLARIS: this is what we know, in our own words,
    and the mapper's job is to work out where it goes. The scripted path's
    ``_form_values`` does the opposite — it already knows the target's field
    names, which is exactly the coupling being tested here.
    """
    record = intake.get(fnol_id)
    if record is None:
        raise KeyError(fnol_id)
    customer = query_one(
        """SELECT cu.full_name FROM fnol_request f
           JOIN customer cu ON cu.id = f.customer_id WHERE f.id = ?""",
        (fnol_id,))
    policy = query_one(
        """SELECT p.policy_number FROM policy p
           JOIN fnol_request f ON f.customer_id = p.customer_id
           WHERE f.id = ? ORDER BY p.start_date DESC LIMIT 1""",
        (fnol_id,))
    answers = record["answers"]
    # query_one hands back a sqlite3.Row, which has no .get().
    customer = dict(customer) if customer else {}
    policy = dict(policy) if policy else {}
    return {
        "policy_number": policy.get("policy_number"),
        "claimant_name": customer.get("full_name"),
        "kind_of_claim": record.get("claim_type"),
        "what_happened": answers.get("description"),
        "date_of_incident": answers.get("incident_date"),
        "incident_kind": answers.get("incident_kind") or answers.get("treatment_type"),
        "reserve_to_open": intake.DEFAULT_RESERVE.get(record.get("claim_type"), 1000.0),
    }


def _map_fields(recorder: _Recorder, fields: list, buttons: list,
                claim: dict) -> dict[str, Any]:
    """Ask the model which value belongs in which field."""
    result = gateway.complete(
        "form_mapper",
        {"form": wrap_untrusted("form", json.dumps(fields, indent=1)),
         "buttons": wrap_untrusted("buttons", json.dumps(buttons)),
         "claim": json.dumps(claim, indent=1, default=str)},
        tier="primary",
    )
    plan = result.json(default={}) or {}
    if not plan.get("fields"):
        raise RuntimeError("The model returned no field mapping for this form.")
    recorder.step(
        "Planned the form",
        f"{len(plan['fields'])} field(s) mapped by {result.model}"
        + (f"; no data for {', '.join(plan['unmapped'])}" if plan.get("unmapped") else ""),
    )
    return plan


def _drive(recorder: _Recorder, fnol_id: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    settings = get_settings()
    claim = _claim_payload(fnol_id)

    recorder.step("Launching browser", "Agentic mode — the form is read, not assumed")
    with sync_playwright() as pw:
        browser = registration._launch(pw, recorder)
        try:
            page = browser.new_page(viewport={"width": 1180, "height": 900})
            page.goto(f"{settings.rpa_portal_url}{PORTAL_PATH}"
                      f"?seq={registration._last_claim_sequence()}",
                      wait_until="domcontentloaded", timeout=30000)
            recorder.frame(page.screenshot(), "Page loaded")

            fields = page.evaluate(_SCRAPE_FIELDS)
            buttons = page.evaluate(_SCRAPE_BUTTONS)
            recorder.step("Read the form",
                          f"{len(fields)} field(s) and {len(buttons)} button(s) "
                          f"discovered — no selectors were hardcoded")

            plan = _map_fields(recorder, fields, buttons, claim)
            by_id = {f["id"]: f for f in fields if f.get("id")}

            filled = []
            for position, item in enumerate(plan["fields"]):
                field = by_id.get(item.get("id"))
                if field is None:
                    recorder.step("Skipped a field",
                                  f"{item.get('id')} is not on this page", state="warn")
                    continue
                value = str(item.get("value", ""))
                try:
                    if field["type"] == "select":
                        page.select_option(f"#{field['id']}", value, timeout=8000)
                    else:
                        page.fill(f"#{field['id']}", value, timeout=8000)
                except Exception as exc:  # noqa: BLE001 - report, don't abort
                    recorder.step(f"Could not fill {field.get('label') or field['id']}",
                                  str(exc).splitlines()[0][:120], state="warn")
                    continue
                filled.append({"id": field["id"], "value": value,
                               "why": item.get("why", "")})
                recorder.step(f"Entered {field.get('label') or field['id']}", value)

                # Some forms gate the rest of themselves behind a lookup — on
                # POLARIS the loss-type select stays empty until the policy is
                # verified. The mapper reports that button if it sees one; press
                # it once the opening identity fields are in.
                if plan.get("verify") and position == 1:
                    try:
                        page.click(f"#{plan['verify']}", timeout=5000)
                        page.wait_for_timeout(400)
                        recorder.step("Verified the policy", "Unlocks the rest of the form")
                        recorder.frame(page.screenshot(), "After verification")
                    except Exception:  # noqa: BLE001 - optional, form may not gate
                        pass

            recorder.frame(page.screenshot(), "Form completed")
            page.click(f"#{plan['submit']}", timeout=8000)
            page.wait_for_timeout(900)
            recorder.frame(page.screenshot(), "Submitted")

            # Read the outcome off the page as text rather than by id. Knowing
            # "#issued_claim_number" would be the same coupling this is trying
            # to avoid — a receipt that says REGISTERED and shows a claim
            # reference is something any core system's confirmation screen has.
            body = page.inner_text("body")
            registered = "REGISTERED" in body.upper()
            match = re.search(r"\bCLM-\d+\b", body)
            claim_number = match.group(0) if match else None
            recorder.step("Core system responded",
                          f"{'registered' if registered else 'no confirmation'}"
                          + (f" as {claim_number}" if claim_number else ""),
                          state="done" if registered and claim_number else "warn")
            recorder.frame(page.screenshot(), "Registration receipt")

            if not (registered and claim_number):
                raise RuntimeError(
                    "The core system did not confirm a registration. Nothing was "
                    "written to the claims book.")

            return {"mode": "agentic", "claim_number": claim_number,
                    "fields_filled": filled,
                    "unmapped": plan.get("unmapped", []),
                    "submit_button": plan.get("submit"),
                    "discovered_fields": len(fields)}
        finally:
            browser.close()
