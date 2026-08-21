"""ClaimCompanion — Streamlit customer portal and staff console.

Talks to the FastAPI backend over HTTP only; it holds no business logic and no
database access, so the API stays the single enforcement point for auth and
guardrails.

    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API = os.getenv("CLAIMCOMPANION_API", "http://127.0.0.1:8000/api/v1")

STATUS_COLOURS = {
    "FILED": "#64748b", "DOCS_PENDING": "#d97706", "IN_ASSESSMENT": "#2563eb",
    "ADDITIONAL_INFO": "#d97706", "APPROVED": "#16a34a",
    "PAYMENT_IN_PROGRESS": "#16a34a", "SETTLED": "#15803d",
    "REJECTED": "#dc2626", "WITHDRAWN": "#64748b",
}

STATE_ICON = {"VERIFIED": "✅", "MISSING": "⬜", "UPLOADED": "⏳",
              "IN_REVIEW": "🔍", "REJECTED": "⚠️"}

st.set_page_config(page_title="ClaimCompanion", page_icon="🛡️", layout="wide")

# Streamlit renders every tab in one pass, so widget keys must be unique across
# the whole page, not just within the tab you're looking at.
_widget_seq = 0


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api(method: str, path: str, **kwargs: Any) -> Any:
    headers = kwargs.pop("headers", {})
    if token := st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.request(method, f"{API}{path}", headers=headers,
                                    timeout=180, **kwargs)
    except requests.RequestException as exc:
        # A dropped connection is transient — never clear the session for it, or
        # a backend restart silently logs the customer out mid-conversation.
        st.warning(f"Lost contact with the backend for a moment. "
                   f"[{type(exc).__name__}]")
        # Counter, not the path: the same endpoint can fail twice in one render
        # pass and identical widget keys crash the page.
        global _widget_seq
        _widget_seq += 1
        if st.button("Try again", key=f"retry_{_widget_seq}"):
            st.rerun()
        st.stop()

    if response.status_code == 401:
        # Say *why*, and keep it on screen — clearing state then rerunning threw
        # the message away and looked like a random logout.
        reason = "Your sign-in is no longer valid"
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            detail = ""
        if "Unknown principal" in str(detail):
            reason = "The demo data was reset, so your sign-in no longer matches"
        st.session_state.clear()
        st.query_params.clear()
        st.session_state.login_notice = f"{reason} — please sign in again."
        st.rerun()
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"_error": detail, "_status": response.status_code}
    if response.headers.get("content-type", "").startswith("image/"):
        return response.content
    return response.json()


def restore_session() -> None:
    """Bring back a sign-in after a page refresh or a server restart.

    Streamlit keeps session_state in the server process, so refreshing the tab —
    or the app restarting — drops it and dumps the customer at the login screen
    mid-conversation. The token is parked in the URL so the session can be
    rebuilt. (Demo-grade: a token in a URL leaks via history and referrers; a
    real deployment wants an httpOnly cookie. See TO_BE_DONE.md.)
    """
    if st.session_state.get("token"):
        return
    token = st.query_params.get("t")
    if not token:
        return

    try:
        response = requests.get(f"{API}/auth/me",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=30)
    except requests.RequestException:
        return  # backend momentarily unreachable; don't discard the token

    if response.ok:
        data = response.json()
        st.session_state.update(token=token, name=data["name"], role=data["role"],
                                customer_id=data["customer_id"])
    else:
        st.query_params.clear()


def remember_session(token: str) -> None:
    st.query_params["t"] = token


def money(value: Any) -> str:
    try:
        return f"£{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------
# login
# --------------------------------------------------------------------------
def login_view() -> None:
    st.title("🛡️ ClaimCompanion")
    st.caption("Your claim, explained in plain English.")

    if notice := st.session_state.pop("login_notice", None):
        st.warning(notice)

    tab_customer, tab_staff = st.tabs(["Customer sign in", "Staff sign in"])

    with tab_customer:
        st.info("Demo accounts: **priya@example.com**, marcus@example.com, "
                "elena@example.com, james@example.com, aisha@example.com — "
                "code **000000**")
        with st.form("customer_login"):
            email = st.text_input("Email", value="priya@example.com")
            otp = st.text_input("One-time code", value="000000")
            if st.form_submit_button("Sign in", type="primary"):
                result = api("POST", "/auth/login", json={"email": email, "otp": otp})
                if "_error" in result:
                    st.error(result["_error"])
                else:
                    st.session_state.update(token=result["token"], name=result["name"],
                                            role="customer",
                                            customer_id=result["customer_id"])
                    remember_session(result["token"])
                    st.rerun()

    with tab_staff:
        st.info("Any username works. Start it with **manager** for manager rights "
                "(audit + metrics); anything else is an agent. Code **000000**.")
        with st.form("staff_login"):
            username = st.text_input("Username", value="manager.elena")
            otp = st.text_input("One-time code", value="000000", key="staff_otp")
            if st.form_submit_button("Sign in", type="primary"):
                result = api("POST", "/auth/staff/login",
                             json={"username": username, "otp": otp})
                if "_error" in result:
                    st.error(result["_error"])
                else:
                    st.session_state.update(token=result["token"], name=result["name"],
                                            role=result["role"])
                    remember_session(result["token"])
                    st.rerun()


# --------------------------------------------------------------------------
# cards
# --------------------------------------------------------------------------
def render_timeline_card(payload: dict) -> None:
    colour = STATUS_COLOURS.get(payload.get("status", ""), "#64748b")
    st.markdown(
        f"<div style='border-left:4px solid {colour};padding:10px 14px;"
        f"background:rgba(127,127,127,0.08);border-radius:6px;'>"
        f"<b>{payload.get('claim_number','')}</b> · "
        f"<span style='color:{colour};font-weight:600;'>"
        f"{payload.get('status','').replace('_',' ').title()}</span><br>"
        f"<span style='opacity:0.8;font-size:0.9em;'>"
        f"{payload.get('status_meaning','')}</span></div>",
        unsafe_allow_html=True,
    )

    prediction = payload.get("prediction") or {}
    if prediction.get("predicted_settlement_date") and not prediction.get("terminal"):
        cols = st.columns(3)
        cols[0].metric("Expected completion", prediction["predicted_settlement_date"])
        cols[1].metric("Give or take", f"{prediction.get('band_days', 0)} days")
        cols[2].metric("Confidence", f"{int(prediction.get('confidence', 0) * 100)}%")
        st.caption(prediction.get("basis", ""))

    if history := payload.get("history"):
        steps = " → ".join(
            f"**{h['status'].replace('_', ' ').title()}** ({h['date']})" for h in history
        )
        st.markdown(f"<small>{steps}</small>", unsafe_allow_html=True)

    if outstanding := payload.get("outstanding_documents"):
        st.warning("Still needed: " + ", ".join(d.replace("_", " ") for d in outstanding))


def render_checklist_card(payload: dict) -> None:
    st.markdown(f"**Document checklist — {payload.get('claim_number','')}**")
    for item in payload.get("items", []):
        icon = STATE_ICON.get(item["state"], "⬜")
        label = item["doc_type"].replace("_", " ").title()
        required = "" if item["mandatory"] else " _(optional)_"
        st.markdown(f"{icon} **{label}**{required} — `{item['state']}`")
        if item["state"] == "MISSING" and item.get("guidance"):
            with st.expander("How to get this"):
                st.write(item["guidance"])
    if payload.get("complete"):
        st.success("Everything we need has been checked and accepted.")


def render_rejection_card(payload: dict, doc_id: str | None = None,
                          context: str = "main") -> None:
    """The Smart Rejection Explanation — the flagship view.

    ``context`` disambiguates widget keys. Streamlit renders every tab in a
    single pass, so the same document can be drawn by the chat, claims and
    upload tabs at once — identical keys then collide and crash the page.
    """
    st.error(f"**{payload.get('headline','Document needs attention')}**")
    st.write(payload.get("plain_explanation", ""))

    doc_id = doc_id or payload.get("doc_id")
    if doc_id:
        image = api("GET", f"/documents/{doc_id}/annotated")
        if isinstance(image, bytes):
            st.image(image, caption="We've highlighted the problem area.",
                     use_container_width=True)

    if steps := payload.get("fix_steps"):
        st.markdown("**How to fix it**")
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")

    with st.expander("Technical detail (what our rules found)"):
        st.write(f"Reason code: `{payload.get('reason_code')}`")
        st.write(f"Rules that failed: `{', '.join(payload.get('failed_rules', [])) or '—'}`")
        for line in payload.get("technical_detail", []):
            st.write(f"• {line}")
        st.caption(f"Explanation written by: {payload.get('explanation_source', '—')} "
                   f"(prompt v{payload.get('prompt_version', '—')})")

    if payload.get("can_dispute") and doc_id:
        if st.button("This looks wrong to me", key=f"dispute_{context}_{doc_id}"):
            result = api("POST", f"/documents/{doc_id}/dispute")
            st.success(result.get("message", "Sent for human review."))


def render_card(card: dict, context: str = "main") -> None:
    kind = card.get("card_type")
    payload = card.get("payload", {})
    if kind == "claim_timeline":
        render_timeline_card(payload)
    elif kind == "checklist":
        render_checklist_card(payload)
    elif kind == "doc_rejection":
        render_rejection_card(payload, context=context)
    elif kind == "handoff":
        st.info(f"🤝 I've taken this to our claims team — reference "
                f"**{payload.get('ticket_id')}** ({payload.get('priority')}). "
                f"I'll have their answer back to you {payload.get('eta')}.")
    elif kind == "handoff_status":
        who = payload.get("assigned_to")
        st.caption(f"🤝 Case {payload.get('ticket_reference') or payload.get('ticket_id')}"
                   f" · {payload.get('priority','')} · "
                   f"{('with ' + who) if who else 'waiting to be picked up'}")
    elif kind == "citations":
        sources = ", ".join(f"[{i['n']}] {i['title']}" for i in payload.get("items", []))
        st.caption(f"Sources: {sources}")


# --------------------------------------------------------------------------
# customer views
# --------------------------------------------------------------------------
def chat_view() -> None:
    st.subheader("Ask me anything about your claim")

    if "conversation_id" not in st.session_state:
        # Resumes by default — a reviewer's reply may have arrived since the
        # customer last had this page open.
        result = api("POST", "/chat/conversations")
        st.session_state.conversation_id = result["conversation_id"]
        st.session_state.greeting = result["greeting"]
        st.session_state.suggestions = result.get("suggestions", [])
        st.session_state.turn_cards = {}

    cid = st.session_state.conversation_id

    # The server owns the thread, so anything a reviewer sends back appears here
    # without the customer doing anything.
    state = api("GET", f"/chat/conversations/{cid}/messages")
    thread = state.get("messages", [])
    handoff = state.get("handoff")

    # Only mark the "while you were away" divider on the first render of a
    # resumed session; polling shouldn't keep re-announcing it.
    if "away_marker" not in st.session_state:
        unseen = state.get("unseen", 0)
        st.session_state.away_marker = (
            thread[-unseen]["id"] if unseen and len(thread) >= unseen else None
        )

    if handoff:
        who = handoff.get("assigned_to")
        if who:
            st.info(f"🤝 **{who}** from our claims team is on your case "
                    f"(ref **{handoff['ticket_reference']}**). I'll bring their "
                    f"answer straight here — no need to chase.")
        else:
            st.info(f"🤝 I've taken your case to our claims team "
                    f"(ref **{handoff['ticket_reference']}**, "
                    f"{handoff.get('priority','NORMAL').lower()} priority). "
                    f"I'll bring their answer straight here.")

    with st.chat_message("assistant"):
        st.write(st.session_state.greeting)

    for message in thread:
        role = message["role"]

        if message["id"] == st.session_state.get("away_marker"):
            st.markdown(
                "<div style='text-align:center;opacity:0.65;font-size:0.85em;"
                "border-top:1px solid rgba(127,127,127,0.35);margin:14px 0 6px;"
                "padding-top:6px;'>while you were away</div>",
                unsafe_allow_html=True,
            )

        if role == "system":
            st.caption(f"— {message['content']} —")
            continue

        with st.chat_message("user" if role == "user" else "assistant"):
            # Relayed answers stay in the assistant's voice, but the customer is
            # told a real person is behind them.
            if role == "assistant" and message.get("author_name"):
                st.caption(f"🤝 Relayed from {message['author_name']}, claims team")
            st.write(message["content"])
            for card in st.session_state.turn_cards.get(message["id"], []):
                render_card(card, context=f"chat_{message['id']}")

    # Chips come from the poll response so they follow the checklist — once the
    # police report is accepted, stop offering to explain how to get one.
    suggestions = state.get("suggestions") or st.session_state.get("suggestions", [])
    cols = st.columns(len(suggestions) or 1)
    pending = None
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"sugg_{i}_{hash(suggestion) & 0xffff}"):
            pending = suggestion

    typed = st.chat_input("Type your message…")
    prompt = pending or typed

    if not prompt:
        if handoff:
            st.caption("Waiting on my colleague — this refreshes automatically.")
            _autorefresh()
        return

    with st.spinner("Checking your claim…"):
        result = api("POST", f"/chat/conversations/{cid}/messages",
                     json={"message": prompt,
                           "claim_id": st.session_state.get("active_claim_id")})
    if "_error" in result:
        st.error(result["_error"])
        return

    st.session_state.active_claim_id = result.get("active_claim_id")
    flags = result.get("guardrail_flags", [])
    if result.get("blocked"):
        flags = flags + ["blocked"]
    if result.get("degraded"):
        flags = flags + ["template-fallback (LLM unavailable)"]
    if result.get("message_id") and result.get("cards"):
        st.session_state.turn_cards[result["message_id"]] = result["cards"]
    if flags:
        st.session_state.last_flags = flags
    st.rerun()


def _autorefresh(seconds: int = 6) -> None:
    """Poll for a reviewer's answer without the customer clicking anything."""
    try:
        import time
        time.sleep(seconds)
        st.rerun()
    except Exception:
        if st.button("Check for a reply"):
            st.rerun()


def claims_view() -> None:
    data = api("GET", "/claims")
    claims = data.get("claims", [])
    if not claims:
        st.info("You don't have any claims on file yet.")
        return

    for claim in claims:
        colour = STATUS_COLOURS.get(claim["status"], "#64748b")
        outstanding = claim.get("checklist", {}).get("outstanding_mandatory", [])
        badge = f" · ⚠️ {len(outstanding)} document(s) needed" if outstanding else ""

        with st.expander(
            f"{claim['claim_number']} — {claim['claim_type'].title()} — "
            f"{claim['status'].replace('_', ' ').title()}{badge}",
            expanded=claims.index(claim) == 0,
        ):
            cols = st.columns(4)
            cols[0].metric("Claimed", money(claim.get("claimed_amount")))
            cols[1].metric("Approved", money(claim.get("approved_amount")))
            cols[2].metric("Incident", claim.get("incident_date", "—"))
            cols[3].markdown(
                f"<div style='padding-top:8px;'><span style='background:{colour};"
                f"color:white;padding:4px 10px;border-radius:12px;font-size:0.8em;'>"
                f"{claim['status'].replace('_',' ')}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(claim.get("status_meaning", ""))

            detail = api("GET", f"/claims/{claim['id']}")
            render_timeline_card({
                "claim_number": claim["claim_number"],
                "status": claim["status"],
                "status_meaning": claim.get("status_meaning", ""),
                "history": [{"status": h["to_status"], "date": h["changed_at"][:10]}
                            for h in detail.get("history", [])],
                "prediction": detail.get("prediction"),
                "outstanding_documents": outstanding,
            })

            st.divider()
            render_checklist_card({**detail.get("checklist", {}),
                                   "claim_number": claim["claim_number"]})

            for document in detail.get("documents", []):
                if not document.get("rejection_payload"):
                    continue
                st.divider()
                if document.get("superseded"):
                    # Already covered by an accepted document of the same type —
                    # show it for transparency, but don't alarm anyone.
                    label = (document.get("doc_type") or "document").replace("_", " ")
                    with st.expander(f"ℹ️ A {label} you sent couldn't be accepted "
                                     f"— no action needed"):
                        st.info("You don't need to do anything about this one. We "
                                "already have an accepted version on file.")
                        st.write(document["rejection_payload"].get(
                            "plain_explanation", ""))
                else:
                    render_rejection_card(document["rejection_payload"], document["id"],
                                          context="claims")


def upload_view() -> None:
    st.subheader("Upload a document")
    data = api("GET", "/claims")
    claims = data.get("claims", [])
    if not claims:
        st.info("You don't have any claims to upload against.")
        return

    labels = {f"{c['claim_number']} — {c['claim_type'].title()}": c for c in claims}
    chosen = st.selectbox("Which claim is this for?", list(labels))
    claim = labels[chosen]

    outstanding = claim.get("checklist", {}).get("outstanding_mandatory", [])
    if outstanding:
        st.warning("Still needed: " + ", ".join(d.replace("_", " ") for d in outstanding))

    st.caption("Tip: put the document flat in good light, hold the camera directly "
               "above it, and fit all four corners in the frame.")

    uploaded = st.file_uploader("Choose a file",
                                type=["txt", "md", "csv", "json", "png", "jpg", "pdf"])
    if uploaded and st.button("Upload and check", type="primary"):
        with st.spinner("Uploading…"):
            result = api("POST", f"/claims/{claim['id']}/documents",
                         files={"file": (uploaded.name, uploaded.getvalue())})
        if "_error" in result:
            st.error(result["_error"])
            return

        doc_id = result["doc_id"]
        placeholder = st.empty()
        stages = {"UPLOADED": "Uploading…", "SCANNING": "Checking the file…",
                  "OCR": "Reading your document…", "CLASSIFYING": "Working out what it is…",
                  "EXTRACTING": "Pulling out the details…",
                  "VALIDATING": "Checking it against your claim…"}

        import time
        for _ in range(90):
            document = api("GET", f"/documents/{doc_id}")
            status = document.get("status", "UPLOADED")
            if status in stages:
                placeholder.info(stages[status])
                time.sleep(1)
                continue

            placeholder.empty()
            if status == "VERIFIED":
                st.success(f"✅ Your {(document.get('doc_type') or 'document')
                                     .replace('_', ' ')} looks good — we've matched it "
                           f"to claim {claim['claim_number']}.")
                with st.expander("What we read from it"):
                    st.json(document.get("extracted_fields", {}))
            elif status == "NEEDS_REVIEW":
                st.info("🔍 A specialist is taking a closer look at this one. "
                        "This usually takes 1–2 working days.")
                if document.get("rejection"):
                    st.write(document["rejection"].get("plain_explanation", ""))
            elif document.get("superseded"):
                label = (document.get("doc_type") or "document").replace("_", " ")
                st.info(f"We couldn't accept that {label} — but **you don't need "
                        f"to do anything**. We already have an accepted "
                        f"{label} on file for this claim.")
                with st.expander("Why couldn't it be accepted?"):
                    st.write(document["rejection"].get("plain_explanation", ""))
            else:
                if document.get("rejection"):
                    render_rejection_card(document["rejection"], doc_id,
                                          context="upload")
                else:
                    st.error(f"Document status: {status}")

            cols = st.columns(3)
            cols[0].metric("Readability", f"{int((document.get('ocr_quality') or 0)*100)}%")
            cols[1].metric("Type confidence",
                           f"{int((document.get('classification_conf') or 0)*100)}%")
            cols[2].metric("Detail confidence",
                           f"{int((document.get('extraction_conf') or 0)*100)}%")
            break
        else:
            st.warning("Still processing — check your claim page in a moment.")


# --------------------------------------------------------------------------
# staff views
# --------------------------------------------------------------------------
PRIORITY_BADGE = {"URGENT": "🔴", "HIGH": "🟠", "NORMAL": "🔵"}


def case_workspace() -> None:
    """The reviewer's desk: pick a case, see everything, act on it.

    The reviewer never talks to the customer directly — ClaimCompanion carries
    the answer back in its own voice, so the customer keeps one point of contact.
    """
    data = api("GET", "/staff/escalations")
    tickets = data.get("tickets", [])
    if not tickets:
        st.success("No open cases. 🎉")
        return

    labels = {
        f"{PRIORITY_BADGE.get(t['priority'],'⚪')} {t['priority']} · "
        f"{t['customer_name']} · {t.get('claim_number') or 'no claim'} · "
        f"{t['status']}": t
        for t in tickets
    }
    chosen = st.selectbox(f"{len(tickets)} open case(s)", list(labels))
    ticket = labels[chosen]

    # More than one open case for the same person is almost always a duplicate.
    same_customer = [t for t in tickets
                     if t["customer_name"] == ticket["customer_name"]]
    if len(same_customer) > 1:
        st.warning(f"{ticket['customer_name']} has {len(same_customer)} open cases. "
                   f"A case belongs to the customer, so these are probably "
                   f"duplicates — close the extras and work the oldest.")
        if st.button("Close this one as a duplicate", key=f"dup_{ticket['id']}"):
            result = api("POST", f"/staff/escalations/{ticket['id']}/close-duplicate")
            if "_error" in result:
                st.error(result["_error"])
            else:
                st.success("Closed as a duplicate. The customer wasn't notified — "
                           "from their side nothing changed.")
                st.rerun()

    case = api("GET", f"/staff/escalations/{ticket['id']}/case")
    if "_error" in case:
        st.error(case["_error"])
        return

    packet = case["ticket"].get("context_packet", {})
    claim = case.get("claim")

    follow_ups = packet.get("follow_ups", [])
    mood = packet.get("customer_sentiment", "—")
    initial = packet.get("initial_sentiment")

    header = st.columns(4)
    header[0].metric("Priority", ticket["priority"])
    header[1].metric(
        "Customer mood now", mood,
        delta=(f"was {initial}" if initial and initial != mood else None),
        delta_color="off",
    )
    header[2].metric("Times chased", len(follow_ups))
    header[3].metric("Assigned to", case["ticket"].get("assigned_to") or "unassigned")

    if mood in ("frustrated", "distressed") or len(follow_ups) >= 2:
        st.error(f"⚠️ This customer is **{mood}** and has chased "
                 f"{len(follow_ups)} time(s). Read their latest message before "
                 f"replying.")

    st.caption(f"**Why it was raised:** {case['ticket']['reason']}")
    if follow_ups:
        with st.expander(f"They've chased {len(follow_ups)} time(s) — what they said"):
            for f in reversed(follow_ups):
                st.markdown(f"`{f['at'][11:16]}` · _{f['sentiment']}_ — "
                            f"“{f['message']}”")
    st.info(f"**What ClaimCompanion passed on for the customer:** "
            f"{packet.get('conversation_summary','—')}")

    if not case["ticket"].get("assigned_to"):
        if st.button("📥 Take this case", type="primary"):
            api("POST", f"/staff/escalations/{ticket['id']}/claim")
            st.rerun()
        st.caption("Taking the case tells the customer, in the assistant's voice, "
                   "that you're on it — and shows them what was passed to you.")

    left, right = st.columns([3, 2])

    # ---- the conversation the customer is actually having ------------------
    with left:
        conversations = case.get("conversations") or []
        total = sum(c["message_count"] for c in conversations)
        st.markdown(f"#### Conversation history")
        st.caption(f"{len(conversations)} conversation(s), {total} messages — the "
                   f"customer's full history, not just the chat that raised this.")

        with st.container(height=460):
            for conversation in reversed(conversations):  # oldest first
                tags = []
                if conversation.get("is_origin"):
                    tags.append("raised this case")
                if conversation.get("is_active"):
                    tags.append("your reply goes here")
                label = f"{conversation['started_at'][:16].replace('T', ' ')}"
                if tags:
                    label += f" · {' · '.join(tags)}"
                st.markdown(
                    f"<div style='opacity:0.6;font-size:0.8em;border-top:1px solid "
                    f"rgba(127,127,127,0.35);margin-top:10px;padding-top:6px;'>"
                    f"{label}</div>", unsafe_allow_html=True)

                if not conversation["messages"]:
                    st.caption("_(empty)_")
                for message in conversation["messages"]:
                    role = message["role"]
                    if role == "system":
                        st.caption(f"— {message['content']} —")
                        continue
                    speaker = ("Customer" if role == "user"
                               else f"ClaimCompanion (from {message['author_name']})"
                               if message.get("author_name") else "ClaimCompanion")
                    icon = "🧑" if role == "user" else "🛡️"
                    st.markdown(f"{icon} **{speaker}** · "
                                f"`{message['created_at'][11:16]}`")
                    st.markdown(f"> {message['content']}")

        # ---- the assistant, working for the reviewer ----------------------
        st.markdown("#### Ask ClaimCompanion about this case")
        st.caption("Grounded in the case file only — it surfaces evidence, it "
                   "doesn't decide. Answers here are never sent to the customer.")

        quick = st.columns(2)
        asked = None
        prompts = [
            "What has this customer already been told?",
            "What's actually blocking this claim?",
            "Why was the document rejected — is the rule right?",
            "Draft a reply I can edit.",
        ]
        for i, prompt in enumerate(prompts):
            if quick[i % 2].button(prompt, key=f"cop_{ticket['id']}_{i}",
                                   use_container_width=True):
                asked = prompt

        typed = st.text_input("Or ask your own", key=f"copq_{ticket['id']}")
        if st.button("Ask", key=f"copgo_{ticket['id']}") and typed.strip():
            asked = typed

        if asked:
            with st.spinner("Reading the case file…"):
                answer = api("POST", f"/staff/escalations/{ticket['id']}/assist",
                             json={"question": asked})
            if "_error" in answer:
                st.error(answer["_error"])
            else:
                st.session_state[f"cop_ans_{ticket['id']}"] = answer["answer"]

        if cached := st.session_state.get(f"cop_ans_{ticket['id']}"):
            st.info(cached)
            if st.button("Use this as my reply", key=f"copuse_{ticket['id']}"):
                st.session_state[f"note_{ticket['id']}"] = cached
                st.rerun()

        st.divider()
        st.markdown("#### Your answer")
        st.caption("ClaimCompanion delivers this to the customer in its own voice. "
                   "It may only re-word — any fact it adds that isn't in your note "
                   "is discarded and you get quoted verbatim instead.")
        note = st.text_area("Note to the customer", height=120,
                            key=f"note_{ticket['id']}",
                            placeholder="e.g. I've checked the invoice against the "
                                        "garage's records and it's genuine — the date "
                                        "was a typing error. I've accepted it.")
        verbatim = st.checkbox(
            "Send my exact words (recommended for any decision)",
            key=f"verb_{ticket['id']}",
        )

        send_col, resolve_col = st.columns(2)
        if send_col.button("📤 Send to customer", type="primary",
                           key=f"send_{ticket['id']}", disabled=not note.strip()):
            result = api("POST", f"/staff/escalations/{ticket['id']}/reply",
                         json={"note": note, "force_verbatim": verbatim})
            if "_error" in result:
                st.error(result["_error"])
            else:
                if result.get("verbatim") and not verbatim:
                    st.warning("The assistant's wording couldn't be verified against "
                               "your note, so you were quoted verbatim.")
                st.success("Delivered. What the customer sees:")
                st.markdown(f"> {result['content']}")

        if resolve_col.button("✅ Send & close case", key=f"res_{ticket['id']}",
                              disabled=not note.strip()):
            api("POST", f"/staff/escalations/{ticket['id']}/resolve",
                json={"closing_note": note})
            st.success("Case closed and the customer told.")
            st.rerun()

    # ---- the tools ---------------------------------------------------------
    with right:
        st.markdown("#### Claim")
        if claim:
            st.markdown(f"**{claim['claim_number']}** · {claim['claim_type'].title()} · "
                        f"`{claim['status']}`")
            cols = st.columns(2)
            cols[0].metric("Claimed", money(claim.get("claimed_amount")))
            cols[1].metric("Approved", money(claim.get("approved_amount")))
            st.caption(f"Incident {claim.get('incident_date')} · "
                       f"Policy {claim.get('policy_number')}")
            prediction = claim.get("prediction") or {}
            if prediction.get("predicted_settlement_date"):
                st.caption(f"Predicted completion "
                           f"{prediction['predicted_settlement_date']} "
                           f"(±{prediction.get('band_days',0)}d, "
                           f"{int(prediction.get('confidence',0)*100)}% confidence)")
        else:
            st.caption("No claim attached to this case.")

        if checklist := case.get("checklist"):
            st.markdown("#### Checklist")
            for item in checklist.get("items", []):
                st.markdown(f"{STATE_ICON.get(item['state'],'⬜')} "
                            f"{item['doc_type'].replace('_',' ').title()} — "
                            f"`{item['state']}`")

        if signals := case.get("fraud_signals"):
            st.markdown("#### Signals")
            for signal in signals:
                st.warning(f"**{signal['signal_type']}** "
                           f"(severity {signal['severity']}) — {signal['explanation']}")

        st.markdown("#### Documents & verification")
        st.caption("Everything the verdict was made from — you're overruling a "
                   "machine, so you get to see exactly what it saw.")

        for document in case.get("documents", []):
            icon = ("✅" if document["status"] == "VERIFIED"
                    else "🔍" if document["status"] == "NEEDS_REVIEW" else "⚠️")
            with st.expander(f"{icon} {document.get('doc_type') or 'unclassified'} — "
                             f"`{document['status']}`"):
                overall = round(
                    0.3 * (document.get("ocr_quality") or 0)
                    + 0.3 * (document.get("classification_conf") or 0)
                    + 0.4 * (document.get("extraction_conf") or 0), 2)
                cols = st.columns(4)
                cols[0].metric("Read", f"{int((document.get('ocr_quality') or 0)*100)}%")
                cols[1].metric("Type",
                               f"{int((document.get('classification_conf') or 0)*100)}%")
                cols[2].metric("Detail",
                               f"{int((document.get('extraction_conf') or 0)*100)}%")
                cols[3].metric("Overall", f"{int(overall*100)}%")

                doc_tabs = st.tabs(["Rules", "What we read", "The document",
                                    "Annotated"])

                # --- every rule that ran, pass and fail alike ---------------
                with doc_tabs[0]:
                    validations = document.get("validations") or []
                    if not validations:
                        st.caption("No rules have run on this document yet.")
                    for rule in validations:
                        mark = "✅" if rule["passed"] else "❌"
                        st.markdown(f"{mark} **{rule['rule_id']}** — "
                                    f"{rule.get('message','')}")
                        if not rule["passed"] and rule.get("details"):
                            st.json(rule["details"])
                    if payload := document.get("rejection_payload"):
                        st.divider()
                        st.error(f"**Verdict:** {payload.get('reason_code')} — "
                                 f"{payload.get('headline','')}")
                        st.caption("What the customer was told: "
                                   f"{payload.get('plain_explanation','')}")

                with doc_tabs[1]:
                    st.json(document.get("extracted_fields", {}))
                    st.caption("Values the model reported that don't appear in the "
                               "document text are dropped before rules run.")

                with doc_tabs[2]:
                    if content := document.get("content"):
                        st.code(content, language=None)
                    else:
                        st.caption("Not text-readable — see the Annotated tab.")

                with doc_tabs[3]:
                    image = api("GET", f"/documents/{document['id']}/annotated")
                    if isinstance(image, bytes):
                        st.image(image, use_container_width=True)
                    else:
                        st.caption("No annotations — nothing was flagged on the page.")

                doc_note = st.text_input("Reviewer note (the customer sees this)",
                                         key=f"cnote_{document['id']}")
                accept, reject = st.columns(2)
                if accept.button("✅ Accept", key=f"cacc_{document['id']}"):
                    api("POST", f"/staff/documents/{document['id']}/decision",
                        json={"verdict": "VERIFIED", "note": doc_note})
                    st.rerun()
                if reject.button("❌ Reject", key=f"crej_{document['id']}"):
                    api("POST", f"/staff/documents/{document['id']}/decision",
                        json={"verdict": "REJECTED_RULES", "note": doc_note})
                    st.rerun()

        # ---- ask the customer for what's missing --------------------------
        st.divider()
        st.markdown("#### Ask the customer for something")
        st.caption("Say what you need in your own shorthand. ClaimCompanion turns "
                   "it into a question they can act on — what, why, how to get it.")
        need = st.text_area("What do you need?", height=80,
                            key=f"need_{ticket['id']}",
                            placeholder="e.g. police report + confirmation the car "
                                        "was taxed on the incident date")
        if st.button("📨 Ask the customer", key=f"ask_{ticket['id']}",
                     disabled=not need.strip()):
            result = api("POST", f"/staff/escalations/{ticket['id']}/request-info",
                         json={"request": need})
            if "_error" in result:
                st.error(result["_error"])
            else:
                st.success("Asked. What the customer sees:")
                st.markdown(f"> {result['content']}")


def demo_view() -> None:
    """The three-way flow, side by side: reviewer → assistant → customer.

    Every relay is shown with the reviewer's original words next to what the
    customer actually received, so the middleman is inspectable rather than
    something you have to take on trust.
    """
    st.markdown("### The assistant as middleman")
    st.caption("The customer talks only to ClaimCompanion. The reviewer works "
               "behind it. This view shows both sides of every message it carried.")

    tickets = api("GET", "/staff/escalations").get("tickets", [])
    resolved = api("GET", "/staff/escalations", params={"include_resolved": "1"})
    all_tickets = tickets or resolved.get("tickets", [])
    if not all_tickets:
        st.info("No cases yet. As a customer, ask to speak to a person — then come "
                "back here.")
        return

    labels = {f"{t['customer_name']} · {t.get('claim_number') or '—'} · {t['status']}": t
              for t in all_tickets}
    ticket = labels[st.selectbox("Case", list(labels))]

    if not ticket.get("conversation_id"):
        st.warning("This case has no conversation attached.")
        return

    case = api("GET", f"/staff/escalations/{ticket['id']}/case")
    log = api("GET", f"/staff/conversations/{ticket['conversation_id']}/relay-log")

    cols = st.columns(3)
    cols[0].metric("Messages carried", log.get("count", 0))
    cols[1].metric("Sent verbatim", log.get("verbatim_count", 0))
    cols[2].metric("Case status", ticket["status"])

    st.divider()

    for relay in log.get("relays", []):
        left, middle, right = st.columns([5, 2, 5])

        with left:
            st.markdown(f"**🧑‍💼 {relay['reviewer']} wrote**")
            st.markdown(
                f"<div style='background:rgba(127,127,127,0.10);padding:10px 14px;"
                f"border-radius:8px;font-size:0.92em;'>{relay['reviewer_wrote']}</div>",
                unsafe_allow_html=True,
            )

        with middle:
            st.markdown("<div style='text-align:center;padding-top:28px;font-size:1.6em;'>"
                        "🛡️<br>→</div>", unsafe_allow_html=True)
            if relay["verbatim"]:
                st.caption("quoted verbatim")
            elif relay["rendered_by"] == "information_request":
                st.caption("turned into a request")
            else:
                st.caption("re-voiced")

        with right:
            st.markdown("**🧑 Customer received**")
            st.markdown(
                f"<div style='background:rgba(37,99,235,0.10);padding:10px 14px;"
                f"border-radius:8px;font-size:0.92em;'>{relay['customer_received']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"via {relay['rendered_by']} · {relay['at'][11:16]}")

        st.divider()

    st.markdown("#### Full customer thread")
    st.caption("One continuous conversation — the customer never sees a handover.")
    for message in case.get("thread", []):
        role = message["role"]
        if role == "system":
            st.caption(f"— {message['content']} —")
            continue
        icon = "🧑" if role == "user" else "🛡️"
        who = "Customer" if role == "user" else "ClaimCompanion"
        if message.get("author_name") and role == "assistant":
            who += f" (carrying {message['author_name']}'s answer)"
        st.markdown(f"{icon} **{who}**")
        st.markdown(f"> {message['content']}")


def staff_view() -> None:
    tabs = st.tabs(["Review queue", "Cases", "Demo: the middleman",
                    "Audit trail", "AI metrics"])

    with tabs[0]:
        data = api("GET", "/staff/review-queue")
        documents = data.get("documents", [])
        signals = data.get("fraud_signals", [])

        st.markdown(f"**{len(documents)} document(s) awaiting review · "
                    f"{len(signals)} fraud signal(s)**")

        for document in documents:
            with st.expander(f"{document['claim_number']} — "
                             f"{document.get('doc_type') or 'unclassified'} — "
                             f"{document['customer_name']}"):
                cols = st.columns(3)
                cols[0].metric("Readability",
                               f"{int((document.get('ocr_quality') or 0)*100)}%")
                cols[1].metric("Type conf.",
                               f"{int((document.get('classification_conf') or 0)*100)}%")
                cols[2].metric("Detail conf.",
                               f"{int((document.get('extraction_conf') or 0)*100)}%")

                st.write("**AI reading:**")
                st.json(document.get("extracted_fields", {}))

                if payload := document.get("rejection_payload"):
                    st.write(f"**AI said:** {payload.get('headline','')}")
                    image = api("GET", f"/documents/{document['id']}/annotated")
                    if isinstance(image, bytes):
                        st.image(image, use_container_width=True)

                note = st.text_input("Reviewer note", key=f"note_{document['id']}")
                left, right = st.columns(2)
                if left.button("✅ Accept", key=f"acc_{document['id']}"):
                    api("POST", f"/staff/documents/{document['id']}/decision",
                        json={"verdict": "VERIFIED", "note": note})
                    st.rerun()
                if right.button("❌ Reject", key=f"rej_{document['id']}"):
                    api("POST", f"/staff/documents/{document['id']}/decision",
                        json={"verdict": "REJECTED_RULES", "note": note})
                    st.rerun()

        if signals:
            st.divider()
            st.markdown("**Fraud signals** — explainable indicators only, never decisions.")
            for signal in signals:
                st.warning(f"**{signal['signal_type']}** on {signal['claim_number']} "
                           f"(severity {signal['severity']}) — {signal['explanation']}")

    with tabs[1]:
        case_workspace()

    with tabs[2]:
        demo_view()

    with tabs[3]:
        if st.session_state.get("role") != "manager":
            st.info("The audit trail is available to managers.")
        else:
            entity = st.text_input("Filter by entity id (optional)")
            data = api("GET", "/staff/audit/events",
                       params={"entity_id": entity, "limit": 100})
            chain = data.get("chain", {})
            if chain.get("ok"):
                st.success(f"🔗 Hash chain intact across {chain.get('events')} events.")
            else:
                st.error(f"⚠️ Chain broken: {chain}")
            for event in data.get("events", []):
                st.markdown(
                    f"`{event['at'][:19]}` **{event['event_type']}** "
                    f"· {event.get('actor_type')} · {event.get('entity_type')} "
                    f"`{(event.get('entity_id') or '')[:8]}`"
                )
                with st.expander("payload"):
                    st.code(event.get("payload", "{}"), language="json")

    with tabs[4]:
        if st.session_state.get("role") != "manager":
            st.info("Metrics are available to managers.")
        else:
            quality = api("GET", "/admin/metrics/quality")
            cols = st.columns(4)
            cols[0].metric("Document verdicts", quality.get("document_verdicts", 0))
            cols[1].metric("Human reviews", quality.get("human_reviews", 0))
            cols[2].metric("AI verdicts overturned",
                           quality.get("ai_verdicts_overturned", 0))
            cols[3].metric("Guardrail blocks", quality.get("guardrail_blocks", 0))
            if quality.get("override_rate") is not None:
                st.metric("Override rate (key quality metric)",
                          f"{quality['override_rate'] * 100:.1f}%")
            st.metric("Template fallbacks (LLM unavailable)",
                      quality.get("template_fallbacks", 0))

            st.divider()
            costs = api("GET", "/admin/metrics/costs")
            st.markdown("**LLM usage by prompt**")
            if rows := costs.get("by_prompt"):
                st.dataframe(rows, use_container_width=True)
            else:
                st.caption("No LLM calls recorded yet.")


# --------------------------------------------------------------------------
def main() -> None:
    restore_session()

    if "token" not in st.session_state:
        login_view()
        return

    with st.sidebar:
        st.markdown(f"### 🛡️ ClaimCompanion")
        st.markdown(f"Signed in as **{st.session_state.get('name','')}** "
                    f"(`{st.session_state.get('role')}`)")
        if st.button("Sign out"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
        if st.session_state.get("role") == "customer":
            if st.button("Start a new conversation"):
                result = api("POST", "/chat/conversations", params={"fresh": "true"})
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.greeting = result["greeting"]
                st.session_state.pop("away_marker", None)
                st.session_state.turn_cards = {}
                st.rerun()
        st.divider()
        health = requests.get(f"{API.rsplit('/api', 1)[0]}/health", timeout=10).json()
        st.caption(f"LLM: {'🟢' if health.get('llm_configured') else '🔴 template mode'} "
                   f"{health.get('llm_model','')}")

    if st.session_state.get("role") in ("agent", "manager"):
        st.title("Staff console")
        staff_view()
        return

    tab_chat, tab_claims, tab_upload = st.tabs(
        ["💬 Assistant", "📋 My claims", "📤 Upload a document"]
    )
    with tab_chat:
        chat_view()
    with tab_claims:
        claims_view()
    with tab_upload:
        upload_view()


main()
