"""Synthetic data generator (§13).

Deterministic with ``--seed`` so demos are reproducible. Generates top-down
(customer -> policy -> claim -> documents) for referential integrity, renders
documents as text, applies a corruption pipeline to ~30% of them, and emits
``expected_labels.json`` which doubles as the eval golden set (§18).

    python -m datagen.generate --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app.config import get_settings
from app.db import connect, init_db
from app.security import crypto
from datagen.kb_corpus import KB
from datagen.seed_pipeline import verify_seeded_documents

FIRST_NAMES = ["Priya", "Marcus", "Elena", "James", "Aisha", "Tom", "Sofia", "Daniel",
               "Grace", "Omar", "Chloe", "Ravi", "Hannah", "Luca", "Nadia", "Ben",
               "Fatima", "Oliver", "Mei", "Samuel"]
LAST_NAMES = ["Sharma", "Okafor", "Rossi", "Whitfield", "Khan", "Bennett", "Alvarez",
              "Murphy", "Osei", "Haddad", "Clarke", "Patel", "Nowak", "Ferrari",
              "Ahmed", "Turner", "Silva", "Wright", "Chen", "Hughes"]
GARAGES = ["Halford Autos", "Kingsway Motors", "Bramley Garage", "Northgate Bodyshop",
           "Riverside Car Care"]
CLINICS = ["St Anne's Clinic", "Meadowbrook Hospital", "Parkview Medical Centre"]

# Stable IDs across reseeds. Random UUIDs meant a reseed invalidated every
# issued JWT (its `sub` pointed at a customer that no longer existed), which
# logged people out mid-conversation for no reason they could see.
ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, "|".join(parts)))


REQUIRED_BY_TYPE = {
    "motor": ["claim_form", "police_report", "repair_invoice", "driving_licence"],
    "health": ["claim_form", "medical_report", "pharmacy_bill"],
    "home": ["claim_form", "repair_invoice"],
}

STATUS_MIX = (
    ["FILED"] * 15 + ["DOCS_PENDING"] * 20 + ["IN_ASSESSMENT"] * 25
    + ["ADDITIONAL_INFO"] * 10 + ["APPROVED"] * 10 + ["PAYMENT_IN_PROGRESS"] * 5
    + ["SETTLED"] * 10 + ["REJECTED"] * 5
)

STAGE_SEQUENCE = ["FILED", "DOCS_PENDING", "IN_ASSESSMENT", "ADDITIONAL_INFO",
                  "APPROVED", "PAYMENT_IN_PROGRESS", "SETTLED"]


# --------------------------------------------------------------------------
# document rendering
# --------------------------------------------------------------------------
def render_repair_invoice(rng, holder, incident, amount, number, garage):
    return f"""{garage}
Unit 4, Bramley Industrial Estate
VAT Registered

REPAIR INVOICE

Invoice No: {number}
Date: {incident.isoformat()}
Billed to: {holder}

Description                     Amount
Parts - front bumper assembly   {amount * 0.42:.2f}
Parts - headlamp unit           {amount * 0.18:.2f}
Labour - 6.5 hrs                {amount * 0.31:.2f}
Paint and materials             {amount * 0.09:.2f}

Total: {amount:.2f}

Payment due within 30 days.
Authorised by: Service Manager
"""


def render_police_report(rng, holder, incident, number):
    return f"""METROPOLITAN POLICE SERVICE
ROAD TRAFFIC INCIDENT REPORT

Incident Reference: {number}
Date: {incident.isoformat()}
Reporting Officer: PC {rng.randint(1000, 9999)}

Name: {holder}
Location: Junction of Kingsway and Elm Road

Summary of incident:
Vehicle A was stationary at traffic lights when Vehicle B failed to stop and
made contact with the rear offside. No injuries were reported at the scene.
Both drivers exchanged details.

Signed: PC {rng.randint(1000, 9999)}
Official stamp applied.
"""


def render_claim_form(rng, holder, incident, number):
    return f"""INSURANCE CLAIM FORM

Reference: {number}
Date: {incident.isoformat()}
Name: {holder}

Section 1 - Incident details
Date of incident: {incident.isoformat()}
Description: Collision at a road junction causing damage to the rear of the vehicle.

Section 2 - Declaration
I declare the information given is true and complete to the best of my knowledge.

Signature of claimant: {holder}
Signed and dated.
"""


def render_driving_licence(rng, holder, expiry, number):
    return f"""DRIVER AND VEHICLE LICENSING AGENCY
DRIVING LICENCE

Licence No: {number}
Name: {holder}
Date of issue: {(expiry - timedelta(days=3650)).isoformat()}
Valid until: {expiry.isoformat()}
Categories: B, BE
Issued by: DVLA Swansea
"""


def render_medical_report(rng, holder, incident, clinic):
    return f"""{clinic}
CLINICAL REPORT

Date: {incident.isoformat()}
Patient: {holder}
Consultant: Dr A. Mahmood

Diagnosis: Soft tissue injury to the cervical spine following a road traffic
collision. Range of movement reduced. No fracture identified on imaging.

Treatment: Analgesia and referral to physiotherapy, six sessions.
Prognosis: Expected full recovery within eight to twelve weeks.

Signed: Dr A. Mahmood
Clinic stamp applied.
"""


def render_pharmacy_bill(rng, holder, incident, amount, number):
    return f"""Meadow Pharmacy
Prescription Receipt

Receipt No: {number}
Date: {incident.isoformat()}
Name: {holder}

Dispensed:
  Naproxen 500mg x28          {amount * 0.4:.2f}
  Co-codamol 30/500 x32       {amount * 0.35:.2f}
  Topical gel                 {amount * 0.25:.2f}

Total: {amount:.2f}
Issued by: Meadow Pharmacy
"""


RENDERERS = {
    "repair_invoice": "repair_invoice", "police_report": "police_report",
    "claim_form": "claim_form", "driving_licence": "driving_licence",
    "medical_report": "medical_report", "pharmacy_bill": "pharmacy_bill",
}


# --------------------------------------------------------------------------
# corruption pipeline (§13) — each defect maps to an expected rejection reason
# --------------------------------------------------------------------------
def corrupt(text: str, kind: str, rng, incident: date, holder: str) -> tuple[str, str]:
    """Apply a realistic defect. Returns (text, expected_reason_code)."""
    if kind == "blur":
        # Simulates low OCR confidence: characters degrade to noise.
        lines = text.splitlines()
        for i in range(len(lines) // 2, len(lines)):
            lines[i] = "".join("~" if rng.random() < 0.6 else ch for ch in lines[i])
        return "\n".join(lines), "ILLEGIBLE"

    if kind == "bad_date":
        wrong = (incident - timedelta(days=rng.randint(5, 40))).isoformat()
        return text.replace(f"Date: {incident.isoformat()}", f"Date: {wrong}", 1), \
            "DATE_OUT_OF_RANGE"

    if kind == "wrong_name":
        other = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        for label in ("Billed to:", "Name:", "Patient:"):
            if label in text:
                return text.replace(f"{label} {holder}", f"{label} {other}", 1), \
                    "NAME_MISMATCH"
        return text, "NAME_MISMATCH"

    if kind == "missing_field":
        for prefix in ("Invoice No:", "Incident Reference:", "Receipt No:", "Licence No:"):
            if prefix in text:
                lines = [ln for ln in text.splitlines() if not ln.startswith(prefix)]
                return "\n".join(lines), "MISSING_FIELD"
        return text, "MISSING_FIELD"

    if kind == "expired":
        past = (date.today() - timedelta(days=rng.randint(30, 900))).isoformat()
        import re
        return re.sub(r"Valid until: \d{4}-\d{2}-\d{2}", f"Valid until: {past}", text), \
            "EXPIRED_DOCUMENT"

    if kind == "no_signature":
        lines = [ln for ln in text.splitlines()
                 if not ln.lower().startswith(("signed", "signature", "official stamp",
                                               "clinic stamp", "authorised by"))]
        return "\n".join(lines), "MISSING_SIGNATURE"

    return text, "NONE"


CORRUPTIONS_FOR = {
    "repair_invoice": ["blur", "bad_date", "wrong_name", "missing_field"],
    "police_report": ["blur", "missing_field", "no_signature"],
    "claim_form": ["blur", "no_signature", "wrong_name"],
    "driving_licence": ["expired", "missing_field", "wrong_name"],
    "medical_report": ["blur", "no_signature"],
    "pharmacy_bill": ["bad_date", "missing_field"],
}


# --------------------------------------------------------------------------
def generate(seed: int, customers: int, claims_per: int) -> dict:
    rng = random.Random(seed)
    settings = get_settings()
    init_db()

    blob_dir = Path(settings.blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)

    conn = connect()
    cur = conn.cursor()

    # Idempotent re-seed.
    for table in ("document_validation", "fraud_signal", "document", "required_document",
                  "claim_status_history", "message", "escalation_ticket", "notification",
                  "conversation", "claim", "policy", "customer", "kb_chunk", "kb_document",
                  "audit_event", "llm_call"):
        cur.execute(f"DELETE FROM {table}")

    labels: list[dict] = []
    now = datetime.now(timezone.utc)

    # --- demo personas first, so they always exist ----------------------
    personas = [
        ("Priya Sharma", "priya@example.com", "motor", "DOCS_PENDING"),
        ("Marcus Bennett", "marcus@example.com", "motor", "IN_ASSESSMENT"),
        ("Elena Rossi", "elena@example.com", "health", "APPROVED"),
        ("James Whitfield", "james@example.com", "home", "SETTLED"),
        ("Aisha Khan", "aisha@example.com", "health", "FILED"),
    ]

    people: list[tuple[str, str, str]] = []
    for name, email, _, _ in personas:
        people.append((stable_id("customer", email), name, email))
    for i in range(customers - len(personas)):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        email = f"user{i}@example.com"
        people.append((stable_id("customer", email), name, email))

    for customer_id, name, email in people:
        cur.execute(
            "INSERT INTO customer (id, full_name, email_enc, email_hmac, phone_enc, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (customer_id, name, crypto.encrypt(email), crypto.blind_index(email),
             crypto.encrypt(f"+4477{rng.randint(10000000, 99999999)}"), now.isoformat()),
        )

    claim_counter = 88400
    policy_counter = 100000

    for index, (customer_id, name, email) in enumerate(people):
        forced = personas[index] if index < len(personas) else None
        n_claims = 1 if forced else rng.randint(1, claims_per)

        for c in range(n_claims):
            product = forced[2] if forced and c == 0 else rng.choice(
                ["motor", "motor", "health", "home"])
            coverage = {"motor": 15000.0, "health": 25000.0, "home": 40000.0}[product]

            policy_counter += 1
            policy_id = stable_id("policy", f"POL{policy_counter}")
            start = date.today() - timedelta(days=rng.randint(60, 700))
            cur.execute(
                "INSERT INTO policy (id, customer_id, policy_number, product_type, "
                "coverage_limit, start_date, end_date, status) VALUES (?,?,?,?,?,?,?,?)",
                (policy_id, customer_id, f"POL{policy_counter}", product, coverage,
                 start.isoformat(), (start + timedelta(days=365)).isoformat(), "ACTIVE"),
            )

            status = forced[3] if forced and c == 0 else rng.choice(STATUS_MIX)
            incident = date.today() - timedelta(days=rng.randint(5, 90))
            filed = incident + timedelta(days=rng.randint(0, 3))
            claimed = round(rng.uniform(400, min(6000, coverage * 0.4)), 2)

            claim_counter += 1
            claim_id = stable_id("claim", f"CLM-{claim_counter}")
            approved = round(claimed * rng.uniform(0.75, 1.0), 2) if status in (
                "APPROVED", "PAYMENT_IN_PROGRESS", "SETTLED") else None
            settled_at = (filed + timedelta(days=rng.randint(14, 35))).isoformat() \
                if status == "SETTLED" else None

            cur.execute(
                "INSERT INTO claim (id, policy_id, claim_number, claim_type, subtype, "
                "status, claimed_amount, approved_amount, incident_date, filed_at, "
                "settled_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (claim_id, policy_id, f"CLM-{claim_counter}", product,
                 "accident" if product == "motor" else "treatment", status, claimed,
                 approved, incident.isoformat(),
                 datetime.combine(filed, datetime.min.time()).isoformat(), settled_at),
            )

            # status history up to the current stage
            cursor_date = filed
            for stage in STAGE_SEQUENCE:
                cur.execute(
                    "INSERT INTO claim_status_history (id, claim_id, from_status, "
                    "to_status, reason, actor_type, changed_at) VALUES (?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()), claim_id, None, stage, "Stage reached", "system",
                     datetime.combine(cursor_date, datetime.min.time()).isoformat()),
                )
                if stage == status:
                    break
                cursor_date += timedelta(days=rng.randint(1, 9))

            required = REQUIRED_BY_TYPE[product]
            for doc_type in required:
                cur.execute(
                    "INSERT INTO required_document (id, claim_id, doc_type, mandatory, "
                    "state) VALUES (?,?,?,1,'MISSING')",
                    (str(uuid.uuid4()), claim_id, doc_type),
                )

            # --- documents ---------------------------------------------
            # Priya (persona 0) is left with documents to upload, on purpose:
            # the stage demo starts from an incomplete checklist.
            if forced and index == 0:
                doc_types = ["claim_form"]
            elif status == "FILED":
                doc_types = []
            elif status == "DOCS_PENDING":
                doc_types = required[: max(1, len(required) - 2)]
            else:
                doc_types = required

            for doc_type in doc_types:
                if doc_type not in RENDERERS:
                    continue
                number = f"{doc_type[:3].upper()}-{rng.randint(10000, 99999)}"
                if doc_type == "repair_invoice":
                    text = render_repair_invoice(rng, name, incident, claimed, number,
                                                 rng.choice(GARAGES))
                elif doc_type == "police_report":
                    text = render_police_report(rng, name, incident, number)
                elif doc_type == "claim_form":
                    text = render_claim_form(rng, name, incident, number)
                elif doc_type == "driving_licence":
                    text = render_driving_licence(
                        rng, name, date.today() + timedelta(days=rng.randint(200, 2000)),
                        number)
                elif doc_type == "medical_report":
                    text = render_medical_report(rng, name, incident, rng.choice(CLINICS))
                else:
                    text = render_pharmacy_bill(rng, name, incident,
                                                round(rng.uniform(20, 250), 2), number)

                expected_reason = "NONE"
                # 30% of documents carry a realistic defect (§13).
                if rng.random() < 0.30 and doc_type in CORRUPTIONS_FOR:
                    kind = rng.choice(CORRUPTIONS_FOR[doc_type])
                    text, expected_reason = corrupt(text, kind, rng, incident, name)

                doc_id = str(uuid.uuid4())
                path = blob_dir / f"{doc_id}_{doc_type}.txt"
                path.write_text(text, encoding="utf-8")

                doc_status = "UPLOADED"
                cur.execute(
                    "INSERT INTO document (id, claim_id, filename, doc_type, status, "
                    "storage_key, extracted_fields, uploaded_at) VALUES (?,?,?,?,?,?,'{}',?)",
                    (doc_id, claim_id, f"{doc_type}.txt", None, doc_status, str(path),
                     (now - timedelta(days=rng.randint(0, 10))).isoformat()),
                )

                labels.append({
                    "doc_id": doc_id,
                    "claim_number": f"CLM-{claim_counter}",
                    "expected_doc_type": doc_type,
                    "expected_verdict": "VERIFIED" if expected_reason == "NONE"
                                        else "REJECTED",
                    "expected_reason_code": expected_reason,
                    "storage_key": str(path),
                })

    # --- fraud seeds: the same invoice on two different claims ----------
    fraud_rows = cur.execute(
        "SELECT id, claim_id, storage_key FROM document WHERE storage_key LIKE "
        "'%repair_invoice%' LIMIT 2"
    ).fetchall()
    if len(fraud_rows) == 2:
        source_text = Path(fraud_rows[0]["storage_key"]).read_text(encoding="utf-8")
        Path(fraud_rows[1]["storage_key"]).write_text(source_text, encoding="utf-8")
        labels.append({
            "doc_id": fraud_rows[1]["id"],
            "expected_doc_type": "repair_invoice",
            "expected_verdict": "NEEDS_REVIEW",
            "expected_reason_code": "DUPLICATE_DOCUMENT",
            "storage_key": fraud_rows[1]["storage_key"],
            "note": "Duplicate invoice seeded across two claims (fraud signal demo).",
        })

    # --- knowledge base -------------------------------------------------
    for entry in KB:
        kb_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO kb_document (id, title, source, doc_class, updated_at) "
            "VALUES (?,?,?,?,?)",
            (kb_id, entry["title"], "curated", entry["doc_class"], now.isoformat()),
        )
        for chunk in entry["chunks"]:  # type: ignore[union-attr]
            cur.execute(
                "INSERT INTO kb_chunk (id, kb_document_id, content, metadata) "
                "VALUES (?,?,?,?)",
                (str(uuid.uuid4()), kb_id, chunk,
                 json.dumps({"doc_class": entry["doc_class"]})),
            )

    conn.commit()
    conn.close()

    labels_path = Path(__file__).parent / "expected_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    # Documents left at UPLOADED with no doc_type are invisible to the checklist,
    # so a freshly seeded demo looks like nobody ever uploaded anything. Run them
    # to a verdict offline (no model calls).
    print("Verifying seeded documents...")
    verdicts = verify_seeded_documents()

    return {
        "verdicts": verdicts,
        "customers": len(people),
        "documents": len(labels),
        "kb_chunks": sum(len(e["chunks"]) for e in KB),  # type: ignore[arg-type]
        "labels_file": str(labels_path),
        "demo_logins": [p[1] for p in personas],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--claims-per", type=int, default=2)
    args = parser.parse_args()

    summary = generate(args.seed, args.customers, args.claims_per)
    print(json.dumps(summary, indent=2))
    print("\nDemo logins (OTP 000000):")
    for email in summary["demo_logins"]:
        print(f"  {email}")


if __name__ == "__main__":
    main()
