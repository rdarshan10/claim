/* Demo controls: the model switch and the reset.

   Both are for driving a demo, not for running a service — the switch is global
   and the reset genuinely deletes chat history. They live in one bar so it is
   obvious they are the same category of thing. */

import { api, esc, el, toast, withBusy } from "./api.js";

/** Mount the bar into `host`. `onReset` fires after state is cleared so the
 *  page can rebuild whatever it was showing. */
export function mountDemoBar(host, { onReset } = {}) {
  const bar = el("div", { class: "demo-bar" });
  bar.innerHTML = `
    <button class="demo-toggle" id="demo-toggle" title="Demo controls">
      <span class="demo-dot"></span><span id="demo-label">Model</span>
    </button>
    <div class="demo-panel hidden" id="demo-panel">
      <div class="demo-section">
        <div class="demo-head">Model
          <span class="tiny" id="demo-active"></span></div>
        <div id="demo-providers"></div>
        <label class="field" style="margin-top:9px;"><span>Primary — extraction, empathy, explanations</span>
          <select id="demo-primary"></select></label>
        <label class="field"><span>Mini — routing and sentiment, every turn</span>
          <select id="demo-mini"></select></label>
        <div class="btn-row">
          <button class="btn btn-primary btn-sm" id="demo-apply">Apply</button>
          <button class="btn btn-sm" id="demo-test">Test</button>
          <button class="btn btn-ghost btn-sm" id="demo-revert">Reset to .env</button>
        </div>
        <div id="demo-test-result"></div>
      </div>

      <div class="demo-section">
        <div class="demo-head">Start over</div>
        <div class="tiny" id="demo-state">…</div>
        <div class="btn-row" style="margin-top:9px;">
          <button class="btn btn-sm" id="demo-reset-chat">Clear this chat</button>
          <button class="btn btn-sm" id="demo-reset-all">Clear my activity</button>
        </div>
        <div class="tiny" style="margin-top:7px;opacity:.8;">
          Chat history, notifications, and any notification of loss that hasn't
          become a claim. Your claims stay.
        </div>
        <div id="demo-claims-row" class="hidden">
          <div class="demo-divider"></div>
          <button class="btn btn-danger btn-sm btn-block" id="demo-reset-claims">
            Reset claims to the sample set</button>
          <div class="tiny" style="margin-top:7px;opacity:.8;">
            Removes claims created by running the intake flow and puts the
            sample claims back exactly as they shipped. Resets every persona.
          </div>
        </div>
      </div>
    </div>`;
  host.append(bar);

  const $ = id => bar.querySelector("#" + id);
  let catalogue = null;

  const toggle = $("demo-toggle");
  const panel = $("demo-panel");
  toggle.addEventListener("click", () => {
    panel.classList.toggle("hidden");
    if (!panel.classList.contains("hidden")) { loadModels(); loadState(); }
  });
  // Clicking away closes it — a panel that covers the chat and stays open is
  // worse than one extra click.
  document.addEventListener("click", event => {
    if (!bar.contains(event.target)) panel.classList.add("hidden");
  });

  function renderProviders() {
    const active = catalogue.active;
    $("demo-active").textContent = shortName(active.primary);
    $("demo-label").textContent = shortName(active.primary);

    $("demo-providers").innerHTML = catalogue.providers.map(p => `
      <button class="prov ${p.key === active.provider ? "on" : ""}"
              data-prov="${esc(p.key)}" ${p.configured ? "" : "disabled"}>
        <span class="prov-name">${esc(p.label)}</span>
        <span class="tiny">${p.configured ? esc(p.note) : "No API key configured"}</span>
      </button>`).join("");

    bar.querySelectorAll(".prov").forEach(button => {
      button.addEventListener("click", () => {
        const provider = catalogue.providers.find(p => p.key === button.dataset.prov);
        fillModels(provider, provider.default_primary, provider.default_mini);
        bar.querySelectorAll(".prov").forEach(b => b.classList.toggle("on", b === button));
      });
    });

    const provider = catalogue.providers.find(p => p.key === active.provider);
    fillModels(provider, active.primary, active.mini);
  }

  function fillModels(provider, primary, mini) {
    const options = provider.models.map(m =>
      `<option value="${esc(m.id)}">${esc(m.label)} — ${esc(m.note || m.id)}</option>`).join("");
    $("demo-primary").innerHTML = options;
    $("demo-mini").innerHTML = options;
    $("demo-primary").value = primary;
    $("demo-mini").value = mini;
  }

  function selectedProvider() {
    return bar.querySelector(".prov.on")?.dataset.prov || catalogue.active.provider;
  }

  async function loadModels() {
    if (catalogue) return;
    try {
      catalogue = await api("GET", "/models");
      renderProviders();
    } catch (error) {
      $("demo-providers").innerHTML = `<div class="tiny">${esc(error.message)}</div>`;
    }
  }

  async function loadState() {
    try {
      const state = await api("GET", "/demo/state");
      $("demo-state").textContent =
        `${state.messages} message(s)` +
        (state.notifications_of_loss ? `, ${state.notifications_of_loss} notification(s) of loss` : "") +
        (state.cases ? `, ${state.cases} case(s)` : "");
    } catch { $("demo-state").textContent = ""; }
  }

  $("demo-apply").addEventListener("click", async event => {
    await withBusy(event.currentTarget, async () => {
      try {
        const result = await api("POST", "/models/select", {
          json: { provider: selectedProvider(),
                  primary: $("demo-primary").value, mini: $("demo-mini").value },
        });
        catalogue = result;
        renderProviders();
        toast(`Now using ${shortName(result.active.primary)}.`, "ok");
      } catch (error) { toast(error.message, "err"); }
    });
  });

  $("demo-test").addEventListener("click", async event => {
    const out = $("demo-test-result");
    out.innerHTML = `<div class="tiny"><span class="spinner"></span> Calling the model…</div>`;
    await withBusy(event.currentTarget, async () => {
      const result = await api("POST", "/models/test");
      out.innerHTML = result.ok
        ? `<div class="tiny ok-line">✅ ${esc(result.model)} replied in ${result.latency_ms} ms</div>`
        : `<div class="tiny err-line">⛔ ${esc(result.error)}</div>`;
    });
  });

  $("demo-revert").addEventListener("click", async event => {
    await withBusy(event.currentTarget, async () => {
      catalogue = await api("POST", "/models/reset");
      renderProviders();
      toast("Back to the configured default.", "ok");
    });
  });

  const doReset = async (scope, button) => {
    await withBusy(button, async () => {
      try {
        const result = await api("POST", "/demo/reset", { json: { scope } });
        const n = result.cleared.messages || 0;
        toast(`Cleared ${n} message(s). Starting fresh.`, "ok");
        panel.classList.add("hidden");
        await loadState();
        onReset?.(scope);
      } catch (error) { toast(error.message, "err"); }
    });
  };
  $("demo-reset-chat").addEventListener("click", e => doReset("conversation", e.currentTarget));
  $("demo-reset-all").addEventListener("click", e => doReset("customer", e.currentTarget));

  // Regenerating the claims book takes a few seconds and discards anything
  // created during a run, so it is separated from the everyday resets and asks
  // first. Open to whoever is driving the demo, staff or customer — the book is
  // shared, so this resets it for every persona, not just the one signed in.
  $("demo-claims-row").classList.remove("hidden");
  $("demo-reset-claims").addEventListener("click", async event => {
    const ok = confirm("Reset the claims book? Claims created by running the "
      + "intake flow will be deleted, and the sample claims regenerated "
      + "exactly as they shipped. This affects every persona, not just you.");
    if (!ok) return;
    await withBusy(event.currentTarget, async () => {
      try {
        const result = await api("POST", "/demo/reset", { json: { scope: "claims" } });
        const c = result.cleared || {};
        toast(`Claims reset — ${c.sample_claims_restored} sample claim(s) restored.`, "ok");
        panel.classList.add("hidden");
        onReset?.("claims");
        // Every id held in this tab — claims, documents, the open conversation
        // — was just deleted and rebuilt. Clearing caches is not enough: the
        // page must be rebuilt from the new dataset or the next click fetches
        // something that no longer exists.
        setTimeout(() => window.location.reload(), 700);
      } catch (error) { toast(error.message, "err"); }
    });
  });

  // The label shows what is live, so it has to be right before the panel opens.
  loadModels();
  return bar;
}

function shortName(modelId) {
  const tail = String(modelId || "").split("/").pop() || "";
  return tail.replace(/^genailab-maas-/, "").replace(/^gpt-oss-/, "OSS ");
}
