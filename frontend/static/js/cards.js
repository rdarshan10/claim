/* Card renderers shared by the portal and the staff console.

   These mirror the agent's card contract: the backend decides *what* to show
   (card_type + payload), the client only decides how it looks. */

import { api, annotatedImage, hasAnnotations, esc, md, money, pct, humanise, titleCase, statusBadge,
         STATUS_LABEL, STATE_ICON, alertBox, toast } from "./api.js";
import { renderFnolCard } from "./fnol-cards.js";

export function timelineCard(payload) {
  const prediction = payload.prediction || {};
  const history = payload.history || [];
  const outstanding = payload.outstanding_documents || [];

  let html = `<div class="tl-card">
    <div class="tl-head">
      <div>
        <div class="tl-num">${esc(payload.claim_number || "")}</div>
        <div class="tl-meaning">${esc(payload.status_meaning || "")}</div>
      </div>
      ${statusBadge(payload.status || "")}
    </div>`;

  // A settled claim has no useful "expected completion" — suppress the whole
  // block rather than showing a date that already passed.
  if (prediction.predicted_settlement_date && !prediction.terminal) {
    html += `<div class="metrics" style="margin-top:14px;">
      <div class="metric"><div class="label">Expected</div>
        <div class="value">${esc(prediction.predicted_settlement_date)}</div></div>
      <div class="metric"><div class="label">Give or take</div>
        <div class="value">${esc(prediction.band_days || 0)} days</div></div>
      <div class="metric"><div class="label">Confidence</div>
        <div class="value">${pct(prediction.confidence)}</div></div>
    </div>`;
    if (prediction.basis) html += `<div class="tiny" style="margin-top:7px;">${esc(prediction.basis)}</div>`;
  }

  if (history.length) {
    const last = history.length - 1;
    html += `<div class="steps">` + history.map((h, i) =>
      `<div class="step ${i === last ? "now" : "done"}">
        <span class="step-dot"></span>
        <div><b>${esc(STATUS_LABEL[h.status] || titleCase(h.status))}</b><span class="tiny"> ${esc(h.date)}</span>
          ${i === last ? '<span class="step-now">now</span>' : ""}</div></div>`
    ).join("") + `</div>`;
  }

  // Outstanding documents are the action card's job now — repeating them here
  // put the same list on screen twice.
  return html + `</div>`;
}

export function checklistCard(payload) {
  const items = payload.items || [];
  let html = `<div class="card-head"><h3>Document checklist</h3>
    <span class="tiny">${esc(payload.claim_number || "")}</span></div>`;

  html += items.map(item => {
    const icon = STATE_ICON[item.state] || "⬜";
    const optional = item.mandatory ? "" : ` <span class="tiny">(optional)</span>`;
    const guidance = (item.state === "MISSING" && item.guidance)
      ? `<details class="disclose" style="margin-top:7px;">
           <summary>How to get this</summary>
           <div class="body">${md(item.guidance)}</div></details>`
      : "";
    return `<div class="check-item">
      <div class="row"><span class="check-ico">${icon}</span>
        <span class="grow"><b>${esc(titleCase(item.doc_type))}</b>${optional}</span>
        <span class="badge ${item.state === "VERIFIED" ? "b-green" : item.state === "REJECTED" ? "b-red" : "b-slate"}">${esc(item.state)}</span>
      </div>${guidance}</div>`;
  }).join("");

  if (payload.complete) html += alertBox("ok", "Everything we need has been checked and accepted.");
  return html;
}

/** The Smart Rejection Explanation — the flagship view.
 *  Returns a live element (not a string): it lazy-loads the annotated image and
 *  owns the dispute button's click handler. */
export function rejectionCard(payload, docId, options = {}) {
  const node = document.createElement("div");
  node.className = "card reject-card";
  docId = docId || payload.doc_id;

  const steps = payload.fix_steps || [];
  const failed = (payload.failed_rules || []).join(", ") || "—";

  node.innerHTML = `
    <div class="reject-head">
      <span class="reject-ico">⚠️</span>
      <div class="grow"><h3>${esc(payload.headline || "Document needs attention")}</h3></div>
    </div>
    <p style="margin-top:10px;">${md(payload.plain_explanation || "")}</p>
    <div class="annot"></div>
    ${steps.length ? `<div class="fix"><h4>How to fix it</h4><ol>${
      steps.map(s => `<li>${md(s)}</li>`).join("")}</ol></div>` : ""}
    <details class="disclose" style="margin-top:12px;">
      <summary>Technical detail (what our rules found)</summary>
      <div class="body">
        <div class="tiny">Reason code <code class="mono">${esc(payload.reason_code || "—")}</code></div>
        <div class="tiny">Rules that failed <code class="mono">${esc(failed)}</code></div>
        ${(payload.technical_detail || []).map(line => `<div class="tiny">• ${esc(line)}</div>`).join("")}
        <div class="tiny" style="margin-top:8px;opacity:.75;">
          Explanation written by ${esc(payload.explanation_source || "—")}
          (prompt v${esc(payload.prompt_version || "—")})</div>
      </div>
    </details>
    <div class="dispute-slot"></div>`;

  if (docId && hasAnnotations({ rejection_payload: payload })) {
    // Not every rejection has a page to annotate; absence is normal, so failure
    // here is silent rather than an error the customer has to interpret.
    annotatedImage(docId).then(blob => {
      if (!blob) return;
      const img = document.createElement("img");
      img.src = URL.createObjectURL(blob);
      img.alt = "The problem area, highlighted";
      img.className = "annot-img";
      const caption = document.createElement("div");
      caption.className = "tiny";
      caption.textContent = "We've highlighted the problem area.";
      node.querySelector(".annot").append(img, caption);
    });
  }

  if (payload.can_dispute && docId && !options.hideDispute) {
    const button = document.createElement("button");
    button.className = "btn btn-sm";
    button.style.marginTop = "12px";
    button.textContent = "This looks wrong to me";
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const result = await api("POST", `/documents/${docId}/dispute`);
        button.replaceWith(Object.assign(document.createElement("div"), {
          className: "alert alert-ok",
          textContent: result.message || "Sent for human review.",
        }));
      } catch (error) { toast(error.message, "err"); button.disabled = false; }
    });
    node.querySelector(".dispute-slot").append(button);
  }
  return node;
}

/** Render one agent card into `host`. Card types the client doesn't know are
 *  skipped rather than shown raw — the backend may add types we predate.
 *  `handlers` is passed through to the FNOL cards, which are interactive. */
export function renderCard(host, card, handlers = {}, replay = false) {
  const payload = card.payload || {};

  // Intake cards own their own interaction, so they build their own element.
  if (String(card.card_type || "").startsWith("fnol_")) {
    const node = renderFnolCard(card, handlers, replay);
    if (node) host.append(node);
    return;
  }

  const wrap = document.createElement("div");

  switch (card.card_type) {
    case "claim_timeline":
      wrap.className = "card card-tight";
      wrap.innerHTML = timelineCard(payload);
      break;
    case "checklist":
      wrap.className = "card card-tight";
      wrap.innerHTML = checklistCard(payload);
      break;
    case "doc_rejection":
      host.append(rejectionCard(payload, payload.doc_id));
      return;
    case "handoff":
      wrap.innerHTML = alertBox("info",
        `🤝 I've passed this to our claims team — reference <b>${esc(payload.ticket_id)}</b>.
         I'll have their answer back to you ${esc(payload.eta)}.`);
      break;
    case "handoff_status": {
      const who = payload.assigned_to;
      wrap.className = "tiny";
      wrap.style.marginTop = "6px";
      wrap.textContent = `🤝 Case ${payload.ticket_reference || payload.ticket_id} · ` +
        (who ? `with ${who}` : "waiting to be picked up");
      break;
    }
    case "action_needed": {
      // Two different things, never mixed: what the customer must send, and
      // what they have already sent and we are checking. Counting them together
      // told people to upload something they had just uploaded.
      const items = payload.items || [];
      const withUs = payload.with_us || [];
      if (!items.length && !withUs.length) return;
      wrap.className = "card action-card" + (items.length ? "" : " all-clear");

      const heading = items.length
        ? `${items.length} thing${items.length === 1 ? "" : "s"} left to send`
        : `Nothing left to send`;
      const sub = items.length
        ? `for ${esc(payload.claim_number)}`
        : `We're checking what you've sent for ${esc(payload.claim_number)}`;

      // Where else the customer could be looking. When the claim was a guess the
      // line says so first — silently picking reads as "these are your
      // documents" and could have them attaching evidence to the wrong claim.
      // The switch itself is offered either way, so the correction is reversible.
      const others = payload.other_claims || [];
      const switcher = others.length
        ? `<div class="tiny action-assumed">
             ${payload.assumed ? "Showing your most recent claim. " : ""}
             ${others.map(c =>
               `<button class="link-btn" data-switch="${esc(c.claim_id)}"
                        data-switch-number="${esc(c.claim_number)}"
                 >Switch to ${esc(c.claim_number)}</button>`).join(" ")}
           </div>`
        : "";

      wrap.innerHTML = `
        <div class="action-head">
          <span class="action-ico">${items.length ? "📋" : "✅"}</span>
          <div class="grow">
            <b>${esc(heading)}</b>
            <div class="tiny">${sub}</div>
            ${switcher}
          </div>
        </div>
        ${items.length ? `<div class="action-items">
          ${items.map(item => `
            <div class="action-item">
              <div class="row">
                <span class="action-state">○</span>
                <span class="grow"><b>${esc(titleCase(item.label))}</b>
                  ${item.state === "REJECTED"
                    ? `<span class="tiny"> · needs replacing</span>` : ""}</span>
              </div>
              ${item.guidance
                ? `<details class="disclose action-help">
                     <summary>How to get this</summary>
                     <div class="body">${md(item.guidance)}</div></details>` : ""}
            </div>`).join("")}
        </div>` : ""}
        ${withUs.length ? `<div class="with-us">
          <span class="tiny">With our team — nothing for you to do</span>
          ${withUs.map(item => `<div class="with-us-item">🔍
            ${esc(titleCase(item.label))}</div>`).join("")}
        </div>` : ""}
        ${items.length ? `<div class="btn-row" style="margin-top:12px;">
          <button class="btn btn-primary btn-sm" data-upload="${esc(payload.claim_id)}">
            📎 Attach documents</button>
        </div>` : ""}`;

      wrap.querySelector("[data-upload]")?.addEventListener("click", () => {
        handlers.onUpload?.(payload.claim_id);
      });
      wrap.querySelectorAll("[data-switch]").forEach(btn => {
        btn.addEventListener("click", () => {
          handlers.onSwitchClaim?.(btn.dataset.switch, btn.dataset.switchNumber);
        });
      });
      break;
    }
    case "offer_human": {
      // An offer, not an escalation: the customer decides whether this needs a
      // person. Declining silently is fine — no case is opened either way.
      wrap.className = "card card-tight offer-card";
      wrap.innerHTML = `<div class="row">
          <span class="offer-ico">🤝</span>
          <div class="grow"><b>Would it help to speak to someone?</b>
            <div class="tiny">I can pass this to a colleague in our claims team —
              they'll pick it up and I'll bring their answer back here.</div></div>
        </div>
        <div class="btn-row" style="margin-top:10px;">
          <button class="btn btn-primary btn-sm" data-offer="yes">Yes, please</button>
          <button class="btn btn-sm" data-offer="no">No, carry on</button>
        </div>`;
      wrap.querySelector('[data-offer="yes"]').addEventListener("click", () => {
        handlers.onAskHuman?.();
        wrap.innerHTML = `<div class="tiny">Passing this to our claims team…</div>`;
      });
      wrap.querySelector('[data-offer="no"]').addEventListener("click", () => wrap.remove());
      break;
    }
    case "citations": {
      const items = payload.items || [];
      if (!items.length) return;
      wrap.className = "citations";
      wrap.innerHTML = `<span class="tiny">Sources</span> ` + items.map(item =>
        `<span class="cite">[${esc(item.n)}] ${esc(item.title)}</span>`).join("");
      break;
    }
    default:
      return;
  }
  host.append(wrap);
}

/** Confidence trio used on documents in both surfaces. */
/** Receipt for one uploaded document.
 *
 *  Every file gets its own, so sending three at once reads as three outcomes
 *  rather than one merged sentence — and so the thread still shows what was
 *  sent after the verdict wording has scrolled away. */
export function documentReceiptCard(document_) {
  const status = String(document_.status || "");
  const verified = status === "VERIFIED";
  const rejected = status.startsWith("REJECTED");
  const badge = verified ? "b-green" : rejected ? "b-red" : "b-slate";
  // "With our team" rather than the raw NEEDS_REVIEW: a clean document sitting
  // with a handler is not a problem, and the enum reads like one.
  const wording = verified ? "Accepted" : rejected ? "Not accepted" : "With our team";
  // Deliberately no confidence scores: readability and classification numbers
  // are how the pipeline reasons, not something a customer can act on, and a
  // "70%" beside an accepted document only invites doubt about a settled fact.
  const rejection = document_.rejection_payload || document_.rejection || {};
  const note = verified
    ? "Checked and accepted for this claim."
    : rejected
    ? (rejection.headline || "We couldn't accept this one.")
    : "Read and passed to one of our claims handlers — nothing for you to do.";
  return `<div class="card card-tight doc-receipt">
    <div class="row">
      <span class="check-ico">${verified ? "✅" : rejected ? "⚠️" : "🔍"}</span>
      <span class="grow"><b>${esc(titleCase(document_.doc_type || "document"))}</b>
        <div class="tiny">${esc(document_.filename || "")}</div></span>
      <span class="badge ${badge}">${esc(wording)}</span>
    </div>
    <div class="tiny doc-receipt-note">${esc(note)}</div>
  </div>`;
}

export function confidenceMetrics(document_, withOverall = false) {
  const ocr = document_.ocr_quality || 0;
  const cls = document_.classification_conf || 0;
  const ext = document_.extraction_conf || 0;
  const overall = 0.3 * ocr + 0.3 * cls + 0.4 * ext;
  return `<div class="metrics">
    <div class="metric"><div class="label">Readability</div><div class="value">${pct(ocr)}</div></div>
    <div class="metric"><div class="label">Type conf.</div><div class="value">${pct(cls)}</div></div>
    <div class="metric"><div class="label">Detail conf.</div><div class="value">${pct(ext)}</div></div>
    ${withOverall ? `<div class="metric"><div class="label">Overall</div><div class="value">${pct(overall)}</div></div>` : ""}
  </div>`;
}

export function jsonBlock(value) {
  return `<pre class="code">${esc(JSON.stringify(value || {}, null, 2))}</pre>`;
}

export { money };
