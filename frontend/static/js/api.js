/* Shared API client + small DOM helpers.
   The backend is the single enforcement point for auth and guardrails; this
   layer only carries the token and turns failures into something visible. */

const API = window.CC_API || "/api/v1";

// Session lives in sessionStorage, not the URL. The Streamlit build parked the
// token in a query param because its state died on refresh; a normal page keeps
// state client-side, so the token never needs to enter browser history.
export const session = {
  get token() { return sessionStorage.getItem("cc_token"); },
  get name() { return sessionStorage.getItem("cc_name") || ""; },
  get role() { return sessionStorage.getItem("cc_role") || ""; },
  get customerId() { return sessionStorage.getItem("cc_customer_id") || ""; },
  set(data) {
    sessionStorage.setItem("cc_token", data.token);
    sessionStorage.setItem("cc_name", data.name || "");
    sessionStorage.setItem("cc_role", data.role || "");
    if (data.customer_id) sessionStorage.setItem("cc_customer_id", data.customer_id);
  },
  clear() {
    ["cc_token", "cc_name", "cc_role", "cc_customer_id"].forEach(k => sessionStorage.removeItem(k));
  },
};

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

/** Thrown-on-failure fetch. Callers that want to render an error inline should
 *  catch; anything uncaught surfaces as a toast via the global handler. */
export async function api(method, path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (session.token) headers["Authorization"] = `Bearer ${session.token}`;

  let body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  } else if (opts.form) {
    body = opts.form; // browser sets the multipart boundary
  }

  const qs = opts.params ? "?" + new URLSearchParams(
    Object.entries(opts.params).filter(([, v]) => v !== undefined && v !== null && v !== "")
  ) : "";

  let response;
  try {
    response = await fetch(`${API}${path}${qs}`, { method, headers, body });
  } catch {
    // Transient: a backend restart shouldn't log anyone out.
    throw new ApiError("Lost contact with the backend. Check it's running, then retry.", 0);
  }

  if (response.status === 401) {
    let detail = "";
    try { detail = (await response.json()).detail || ""; } catch { /* non-JSON body */ }
    const reason = String(detail).includes("Unknown principal")
      ? "The demo data was reset, so your sign-in no longer matches"
      : "Your sign-in is no longer valid";
    session.clear();
    sessionStorage.setItem("cc_notice", `${reason} — please sign in again.`);
    location.href = "/";
    throw new ApiError(reason, 401);
  }

  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = await response.text(); }
    if (detail && typeof detail === "object") detail = JSON.stringify(detail);
    throw new ApiError(detail || `Request failed (${response.status})`, response.status);
  }

  const type = response.headers.get("content-type") || "";
  if (type.startsWith("image/")) return await response.blob();
  if (type.includes("json")) return await response.json();
  return await response.text();
}

// ------------------------------------------------------------------ helpers

/** Escape before interpolating into innerHTML. Document text, reviewer notes and
 *  model output all reach the DOM — none of it may be trusted as markup. */
export function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** Inline markdown: **bold**, *italic*, `code`. Escapes first, so the only tags
 *  that can appear are the ones produced here. */
export function md(value) {
  return inline(esc(value)).replace(/\n/g, "<br>");
}

function inline(escaped) {
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, '<code class="mono">$1</code>')
    // Claim, notification and case references are the things people copy out
    // of a reply, so make them findable at a glance.
    .replace(/\b(CLM-\d+|FNOL-[A-Z0-9]+|POL\d+)\b/g, '<span class="ref-tag">$1</span>');
}

/** Block-level markdown for assistant replies: paragraphs, bullet and numbered
 *  lists. The model writes structured answers ("you still need: a, b, c") and
 *  flattening them into one run of text is what made them hard to read. */
export function richText(value) {
  const lines = esc(value ?? "").split(/\r?\n/);
  const out = [];
  let list = null;          // "ul" | "ol" | null
  let para = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
  };
  const flushList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); flushList(); continue; }

    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^(\d+)[.)]\s+(.*)$/);

    if (bullet || numbered) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) { flushList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline((bullet ? bullet[1] : numbered[2]))}</li>`);
      continue;
    }
    flushList();
    para.push(line);
  }
  flushPara();
  flushList();
  return out.join("");
}

export function money(value) {
  const n = parseFloat(value);
  return isNaN(n) ? "—" : "£" + n.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function pct(value) { return Math.round((value || 0) * 100) + "%"; }

export function titleCase(value) {
  return String(value || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export function humanise(value) { return String(value || "").replace(/_/g, " "); }

export const STATUS_CLASS = {
  FILED: "b-slate", DOCS_PENDING: "b-amber", IN_ASSESSMENT: "b-blue",
  ADDITIONAL_INFO: "b-amber", APPROVED: "b-green", PAYMENT_IN_PROGRESS: "b-green",
  SETTLED: "b-green", REJECTED: "b-red", WITHDRAWN: "b-slate",
};

export const STATE_ICON = {
  VERIFIED: "✅", MISSING: "⬜", UPLOADED: "⏳", IN_REVIEW: "🔍", REJECTED: "⚠️",
};

export const PRIORITY_DOT = { URGENT: "🔴", HIGH: "🟠", NORMAL: "🔵" };

// What each stage is called in front of a customer. The enum is a database
// value — "DOCS PENDING" on a badge is a column name, not a status.
export const STATUS_LABEL = {
  FILED: "Filed",
  DOCS_PENDING: "Waiting for documents",
  IN_ASSESSMENT: "Being assessed",
  ADDITIONAL_INFO: "Waiting for information",
  APPROVED: "Approved",
  PAYMENT_IN_PROGRESS: "Payment on the way",
  SETTLED: "Settled",
  REJECTED: "Not approved",
  WITHDRAWN: "Withdrawn",
};

export function statusBadge(status) {
  const cls = STATUS_CLASS[status] || "b-slate";
  const label = STATUS_LABEL[status] || titleCase(status);
  return `<span class="badge ${cls}"><span class="dot"></span>${esc(label)}</span>`;
}

/** `el("div", {class: "card"}, child, "text")` — terser than document.createElement
 *  chains and keeps text as text (no accidental HTML injection). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function toast(message, kind = "") {
  let host = document.getElementById("toasts");
  if (!host) {
    host = el("div", { id: "toasts" });
    document.body.append(host);
  }
  const node = el("div", { class: `toast ${kind}` }, message);
  host.append(node);
  setTimeout(() => {
    node.style.transition = "opacity .3s";
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 300);
  }, kind === "err" ? 6000 : 3400);
}

export function alertBox(kind, html) {
  const ico = { ok: "✅", warn: "⚠️", error: "⛔", info: "ℹ️" }[kind] || "ℹ️";
  return `<div class="alert alert-${kind}"><span class="ico">${ico}</span><div>${html}</div></div>`;
}

/** Wrap a click handler so the button shows progress and can't be double-fired. */
export async function withBusy(button, fn) {
  if (!button) return fn();
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>';
  try { return await fn(); }
  finally { button.disabled = false; button.innerHTML = original; }
}

/** True when a document actually has something boxed on the page.

 *  The rejection payload already lists the annotations, so checking it first
 *  avoids requesting an image that isn't there — most documents have none, and
 *  a request per document would log a 404 in the console for each one. */
export function hasAnnotations(document_) {
  const payload = document_?.rejection_payload || document_?.rejection || {};
  return Array.isArray(payload.annotations) && payload.annotations.length > 0;
}

/** Fetch an annotated page image, or null when there isn't one.
 *  Absence is a value here rather than a throw: a clean document legitimately
 *  has no annotation, and the API says so with a 404. */
export async function annotatedImage(docId) {
  try {
    const headers = session.token ? { Authorization: `Bearer ${session.token}` } : {};
    const response = await fetch(`${API}/documents/${docId}/annotated`, { headers });
    if (!response.ok) return null;
    return await response.blob();
  } catch {
    return null;
  }
}

export function requireAuth(...roles) {
  if (!session.token) { location.href = "/"; return false; }
  if (roles.length && !roles.includes(session.role)) {
    location.href = session.role === "customer" ? "/portal" : "/staff";
    return false;
  }
  return true;
}

// Drawn rather than the 🛡️ emoji, which renders monochrome on some systems.
const BRAND_SHIELD = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"
  style="width:17px;height:17px;display:block;color:var(--indigo-600);">
  <path d="M12 1.6 3.6 5v6.4c0 5.2 3.6 10 8.4 11.2 4.8-1.2 8.4-6 8.4-11.2V5L12 1.6Z"/></svg>`;

export function mountHeader(subtitle) {
  const initials = (session.name || "?").split(/\s+/).map(p => p[0]).slice(0, 2).join("").toUpperCase();
  return `
    <header class="app-header">
      <div class="brand"><span class="shield">${BRAND_SHIELD}</span> ClaimCompanion
        ${subtitle ? `<span class="sub">${esc(subtitle)}</span>` : ""}</div>
      <div class="header-spacer"></div>
      <div class="header-meta">
        <span id="demo-slot"></span>
        <span id="llm-health" class="tiny"></span>
        <div class="who"><b>${esc(session.name)}</b><span>${esc(session.role)}</span></div>
        <div class="avatar">${esc(initials)}</div>
        <button class="btn btn-sm" id="signout">Sign out</button>
      </div>
    </header>`;
}

export function wireHeader() {
  document.getElementById("signout")?.addEventListener("click", () => {
    session.clear();
    location.href = "/";
  });
  // Health is informational; a failure here must never block the page.
  fetch("/health").then(r => r.json()).then(h => {
    const node = document.getElementById("llm-health");
    if (node) {
      node.innerHTML = h.llm_configured
        ? `<span title="${esc(h.llm_model)}">🟢 LLM ready</span>`
        : `🔴 template mode`;
    }
  }).catch(() => {});
}

// Anything that escapes a handler still tells the user something went wrong.
window.addEventListener("unhandledrejection", event => {
  if (event.reason instanceof ApiError) {
    toast(event.reason.message, "err");
    event.preventDefault();
  }
});
