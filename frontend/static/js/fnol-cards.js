/* FNOL intake cards — the interactive question flow.

   Each card is one question. Tapping an option and typing an answer are the
   same operation to the backend, so the customer can switch between them
   mid-conversation without losing their place. */

import { api, esc, md, el, toast, alertBox, humanise, titleCase } from "./api.js";

/** Build the question card. `onAnswered(result)` fires with the API's reply,
 *  which carries the next card — so the flow advances without a page fetch. */
export function questionCard(payload, onAnswered) {
  const node = el("div", { class: "card fnol-card" });
  const progress = payload.progress || {};
  const pct = progress.total ? Math.round((progress.answered / progress.total) * 100) : 0;

  node.innerHTML = `
    <div class="fnol-head">
      <span class="fnol-ref">${esc(payload.reference)}</span>
      <div class="fnol-bar"><div class="fnol-bar-fill" style="width:${pct}%"></div></div>
      <span class="tiny">${progress.answered || 0}/${progress.total || 0}</span>
    </div>
    <h3 class="fnol-q">${esc(payload.question)}</h3>
    ${payload.hint ? `<p class="fnol-hint">${esc(payload.hint)}</p>` : ""}
    <div class="fnol-body"></div>`;

  const body = node.querySelector(".fnol-body");

  const submit = async (value, button) => {
    if (value === undefined || value === null || value === "") return;
    node.querySelectorAll("button, input, textarea, select")
        .forEach(el_ => { el_.disabled = true; });
    if (button) button.classList.add("chosen");
    try {
      const result = await api("POST", `/fnol/${payload.fnol_id}/answer`,
                               { json: { field: payload.field, value } });

      // The server couldn't use that answer and is asking again. Re-enable this
      // card rather than stacking a duplicate question below it.
      if (result.retry) {
        node.querySelectorAll("button, input, textarea, select")
            .forEach(el_ => { el_.disabled = false; });
        if (button) button.classList.remove("chosen");
        let warn = node.querySelector(".fnol-retry");
        if (!warn) {
          warn = el("div", { class: "fnol-retry tiny" });
          node.querySelector(".fnol-body").prepend(warn);
        }
        warn.textContent = result.message || "Sorry, could you try that again?";
        return;
      }

      node.classList.add("answered");
      // Keep the answer visible: the thread is the customer's record of what
      // they told us, so a card that erases itself loses that.
      node.querySelector(".fnol-body").innerHTML =
        `<div class="fnol-given">✓ ${esc(displayValue(payload, value))}</div>`;
      onAnswered?.(result);
    } catch (error) {
      node.querySelectorAll("button, input, textarea, select")
          .forEach(el_ => { el_.disabled = false; });
      if (button) button.classList.remove("chosen");
      toast(error.message, "err");
    }
  };

  switch (payload.kind) {
    case "choice": {
      const grid = el("div", { class: "fnol-options" });
      payload.options.forEach(option => {
        const button = el("button", { class: "fnol-option" });
        button.innerHTML = `<span class="opt-icon">${esc(option.icon || "•")}</span>
          <span class="opt-label">${esc(option.label)}</span>
          ${option.detail ? `<span class="opt-detail">${esc(option.detail)}</span>` : ""}`;
        button.addEventListener("click", () => submit(option.value, button));
        grid.append(button);
      });
      body.append(grid);
      break;
    }

    case "date": {
      const row = el("div", { class: "fnol-inline" });
      const input = el("input", { type: "date", class: "fnol-input" });
      // A loss cannot be reported before it happened, and the core system
      // rejects future dates — so stop them at the picker.
      input.max = new Date().toISOString().slice(0, 10);
      const go = el("button", { class: "btn btn-primary btn-sm" }, "Confirm");
      go.addEventListener("click", () => submit(input.value, go));
      input.addEventListener("keydown", e => { if (e.key === "Enter") go.click(); });
      row.append(input, go);
      body.append(row, quickRow(payload, submit));
      break;
    }

    case "money": {
      const row = el("div", { class: "fnol-inline" });
      const wrap = el("div", { class: "fnol-money" });
      const input = el("input", { type: "text", inputmode: "decimal",
                                  placeholder: "0.00", class: "fnol-input" });
      wrap.append(el("span", { class: "fnol-prefix" }, "£"), input);
      const go = el("button", { class: "btn btn-primary btn-sm" }, "Confirm");
      go.addEventListener("click", () => submit(input.value, go));
      input.addEventListener("keydown", e => { if (e.key === "Enter") go.click(); });
      row.append(wrap, go);
      body.append(row, quickRow(payload, submit));
      break;
    }

    case "upload": {
      body.append(uploadZone(payload, submit));
      body.append(quickRow(payload, submit));
      break;
    }

    default: {  // text
      const row = el("div", { class: "fnol-inline" });
      const input = el("textarea", { class: "fnol-input", rows: "2",
                                     placeholder: "Type your answer…" });
      const go = el("button", { class: "btn btn-primary btn-sm" }, "Confirm");
      go.addEventListener("click", () => submit(input.value.trim(), go));
      input.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go.click(); }
      });
      row.append(input, go);
      body.append(row, quickRow(payload, submit));
    }
  }

  return node;
}

function quickRow(payload, submit) {
  const quick = payload.quick || [];
  const row = el("div", { class: "fnol-quick" });
  quick.forEach(label => {
    const button = el("button", { class: "chip-sugg" }, label);
    button.addEventListener("click", () => submit(label, button));
    row.append(button);
  });
  return row;
}

/** Upload zone for the incident-report step. Files post immediately; the step
 *  itself is only marked answered when the customer says they're done. */
function uploadZone(payload, submit) {
  const wrap = el("div", { class: "fnol-upload" });
  wrap.innerHTML = `
    <div class="fnol-drop">
      <div class="fnol-drop-ico">📎</div>
      <div><b>Add a file</b> or drag it here</div>
      <div class="tiny">A first incident report, photos, a quote or a receipt</div>
      <input type="file" class="hidden" multiple
             accept=".txt,.md,.csv,.json,.png,.jpg,.jpeg,.pdf">
    </div>
    <div class="fnol-files"></div>`;

  const drop = wrap.querySelector(".fnol-drop");
  const input = wrap.querySelector("input[type=file]");
  const files = wrap.querySelector(".fnol-files");
  let uploaded = 0;

  const send = async fileList => {
    for (const file of fileList) {
      const pill = el("div", { class: "fnol-file" },
        el("span", { class: "spinner" }), ` ${file.name}`);
      files.append(pill);
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        await api("POST", `/fnol/${payload.fnol_id}/documents`, { form });
        uploaded += 1;
        pill.innerHTML = `✅ ${esc(file.name)}`;
        done.textContent = `Done — continue with ${uploaded} file${uploaded === 1 ? "" : "s"}`;
        done.classList.remove("hidden");
      } catch (error) {
        pill.innerHTML = `⚠️ ${esc(file.name)} — ${esc(error.message)}`;
        pill.classList.add("failed");
      }
    }
  };

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files.length) send(e.dataTransfer.files);
  });
  input.addEventListener("change", () => { if (input.files.length) send(input.files); });

  const done = el("button", { class: "btn btn-primary btn-sm hidden" }, "Done — continue");
  done.addEventListener("click", () => submit(`${uploaded} file(s) attached`, done));
  wrap.append(done);
  return wrap;
}

function displayValue(payload, value) {
  if (payload.kind === "choice") {
    const option = (payload.options || []).find(o => o.value === value);
    return option ? option.label : value;
  }
  // Only prefix a currency symbol onto something that is actually a figure —
  // "I don't know yet" is a valid answer here and "£I don't know yet" is not.
  if (payload.kind === "money" && /^[\d.,\s]+$/.test(String(value))) {
    return "£" + String(value).trim();
  }
  return value;
}

/** The review card: everything collected, with a submit button. */
export function reviewCard(payload, onSubmitted) {
  const node = el("div", { class: "card fnol-card fnol-review" });
  node.innerHTML = `
    <div class="fnol-head">
      <span class="fnol-ref">${esc(payload.reference)}</span>
      <span class="badge b-blue">Ready to send</span>
    </div>
    <h3 class="fnol-q">Please check these details</h3>
    <p class="fnol-hint">Nothing has been sent yet. If anything's wrong, just tell
      me in the chat and I'll change it.</p>
    <dl class="fnol-summary">
      ${(payload.items || []).map(item =>
        `<dt>${esc(item.label)}</dt><dd>${esc(item.value)}</dd>`).join("")}
    </dl>
    ${(payload.documents || []).length ? `<div class="fnol-attached">
      <span class="tiny">Attached</span>
      ${payload.documents.map(d => `<span class="fnol-file">📎 ${esc(d.filename)}</span>`).join("")}
    </div>` : ""}
    <div class="fnol-actions"></div>`;

  const actions = node.querySelector(".fnol-actions");
  const send = el("button", { class: "btn btn-primary" }, "Send to the claims team");
  send.addEventListener("click", async () => {
    send.disabled = true;
    send.innerHTML = '<span class="spinner"></span>';
    try {
      const result = await api("POST", `/fnol/${payload.fnol_id}/submit`);
      node.classList.add("answered");
      actions.innerHTML = `<span class="badge b-green">✓ Sent</span>`;
      onSubmitted?.(result);
    } catch (error) {
      send.disabled = false;
      send.textContent = "Send to the claims team";
      toast(error.message, "err");
    }
  });
  actions.append(send);
  return node;
}

/** Receipt shown after submitting — the reference and what happens next. */
export function receiptCard(payload) {
  const node = el("div", { class: "card fnol-card fnol-receipt" });
  node.innerHTML = `
    <div class="fnol-receipt-top">
      <span class="fnol-receipt-ico">✓</span>
      <div>
        <div class="tiny">Your reference</div>
        <div class="fnol-refbig">${esc(payload.reference)}</div>
      </div>
    </div>
    <p>Our claims team will check the details and register the claim on our
       system. I'll let you know here as soon as it has a claim number.</p>
    <div class="fnol-steps">
      <div class="fnol-step done"><span></span>Details collected</div>
      <div class="fnol-step done"><span></span>Sent to the claims team</div>
      <div class="fnol-step"><span></span>Checked by a claims handler</div>
      <div class="fnol-step"><span></span>Registered — you get a claim number</div>
    </div>`;
  return node;
}

const STATUS_LABEL = {
  COLLECTING: ["b-slate", "Still being filled in"],
  // Verification wording: a notification has no claim and nobody assigned yet,
  // so "with the claims team" promised more than is happening. It is being
  // checked to see whether a claim can be opened.
  SUBMITTED: ["b-blue", "Sent for verification"],
  UNDER_REVIEW: ["b-blue", "Being verified"],
  INFO_REQUIRED: ["b-amber", "We need a bit more"],
  READY_TO_REGISTER: ["b-blue", "Verified — queued for registration"],
  REGISTERING: ["b-blue", "Being registered now"],
  REGISTERED: ["b-green", "Registered"],
  REJECTED: ["b-red", "Not taken forward"],
};

export function statusCard(payload) {
  const [cls, label] = STATUS_LABEL[payload.status] || ["b-slate", payload.status];
  const node = el("div", { class: "card fnol-card card-tight" });
  node.innerHTML = `
    <div class="row-between">
      <div>
        <b>${esc(payload.reference)}</b>
        ${payload.claim_type ? `<span class="tiny"> · ${esc(titleCase(payload.claim_type))}</span>` : ""}
      </div>
      <span class="badge ${cls}">${esc(label)}</span>
    </div>
    ${payload.claim_number ? `<div class="fnol-became">
        <span class="tiny">Registered as</span>
        <span class="ref-tag">${esc(payload.claim_number)}</span>
      </div>` : ""}
    ${payload.review_note ? `<p class="tiny" style="margin-top:8px;">${esc(payload.review_note)}</p>` : ""}`;
  return node;
}

/** Route an FNOL card by type. Returns null for anything not ours.
 *
 *  `replay` marks a card being re-rendered from thread history rather than
 *  arriving fresh. Question and review cards are live controls, so on replay
 *  they re-sync against the server: showing a question the customer already
 *  answered — or a submit button for something already sent — is worse than
 *  showing nothing while it loads. */
export function renderFnolCard(card, handlers = {}, replay = false) {
  const payload = card.payload || {};
  switch (card.card_type) {
    case "fnol_question":
      return replay ? liveCard(payload, handlers) : questionCard(payload, handlers.onAnswered);
    case "fnol_review":
      return replay ? liveCard(payload, handlers) : reviewCard(payload, handlers.onSubmitted);
    case "fnol_receipt": return receiptCard(payload);
    case "fnol_status": return statusCard(payload);
    default: return null;
  }
}

/** Placeholder that replaces itself with whatever the notification needs now. */
function liveCard(payload, handlers) {
  const host = el("div", { class: "card fnol-card card-tight" },
    el("div", { class: "row" }, el("span", { class: "spinner" }),
       el("span", { class: "tiny" }, ` ${payload.reference}`)));

  api("GET", `/fnol/${payload.fnol_id}`).then(record => {
    // Once it has left the customer's hands there is nothing to interact with;
    // a status line is the honest thing to show.
    if (record.status !== "COLLECTING") {
      host.replaceWith(statusCard({
        fnol_id: record.id, reference: record.reference, status: record.status,
        claim_type: record.claim_type, review_note: record.review_note,
        claim_id: record.claim_id,
      }));
      return;
    }
    // The server hands back whichever control this notification needs now, so
    // the customer resumes on the right question rather than the one that
    // happened to be captured in this message.
    const live = renderFnolCard(record.card, handlers, false);
    host.replaceWith(live || statusCard({
      reference: record.reference, status: record.status, claim_type: record.claim_type,
    }));
  }).catch(() => {
    host.replaceWith(statusCard({ reference: payload.reference, status: "COLLECTING" }));
  });

  return host;
}
