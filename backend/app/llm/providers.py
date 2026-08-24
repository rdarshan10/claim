"""Model providers and the live switch between them.

Two providers are supported and either can serve the whole app:

* **groq**       — Groq's OpenAI-compatible chat completions, via the ``groq`` SDK.
* **genailab**   — the TCS GenAI Lab endpoint, via ``langchain_openai.ChatOpenAI``.
  It needs a different key and does not present a valid certificate, so it keeps
  its own key and its own ``verify`` setting.

The selection lives in memory, not in ``.env``: this exists so a demo can be
switched mid-conversation without a restart. A process restart falls back to
whatever the environment says, which is the behaviour you want in anything
non-interactive.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    note: str = ""
    # USD per million tokens, as published by the provider. Kept beside the
    # model so a rate change is a one-line edit next to the thing it prices,
    # and so anything reporting spend cites a real number rather than a guess.
    # Zero means "not published" — reported as unknown, never as free.
    usd_in_per_m: float = 0.0
    usd_out_per_m: float = 0.0


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    models: list[Model]
    default_primary: str
    default_mini: str
    transport: str                  # "groq" | "openai_compat"
    env_key: str                    # environment variable holding its API key
    verify_ssl: bool = True
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        key="groq",
        label="Groq",
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com"),
        transport="groq",
        env_key="GROQ_API_KEY",
        verify_ssl=True,
        note="OpenAI-compatible. Fast; shared daily request budget.",
        models=[
            Model("openai/gpt-oss-120b", "GPT-OSS 120B", "Strongest here; use for extraction and empathy",
                  usd_in_per_m=0.15, usd_out_per_m=0.75),
            Model("openai/gpt-oss-20b", "GPT-OSS 20B", "Fast and cheap; good for routing and sentiment",
                  usd_in_per_m=0.10, usd_out_per_m=0.50),
            Model("llama-3.3-70b-versatile", "Llama 3.3 70B", "General purpose",
                  usd_in_per_m=0.59, usd_out_per_m=0.79),
            Model("llama-3.1-8b-instant", "Llama 3.1 8B", "Fastest; weakest instruction following",
                  usd_in_per_m=0.05, usd_out_per_m=0.08),
        ],
        default_primary="openai/gpt-oss-120b",
        default_mini="openai/gpt-oss-20b",
    ),
    "genailab": Provider(
        key="genailab",
        label="GenAI Lab",
        base_url=os.getenv("GENAILAB_BASE_URL", "https://genailab.tcs.in"),
        transport="openai_compat",
        env_key="GENAILAB_API_KEY",
        # The lab endpoint's certificate does not validate; this is why the
        # original build passed verify=False. Scoped to this provider only —
        # it must never leak onto Groq or anything production-facing.
        verify_ssl=False,
        note="TCS lab endpoint. Certificate is not validated.",
        # The lab does not publish its own per-token rates. These two are Azure
        # OpenAI deployments, so they are priced at Azure's published list rate
        # for the model behind them — which is what the lab is passing through.
        # Treat them as an upper bound on a subsidised internal endpoint rather
        # than an invoice. Sonnet stays unpriced until someone confirms a rate:
        # a wrong number here is worse than an honest "unknown".
        models=[
            Model("azure/genailab-maas-gpt-4.1", "GPT-4.1", "Strong instruction follower",
                  usd_in_per_m=2.00, usd_out_per_m=8.00),
            Model("azure/genailab-maas-gpt-4.1-mini", "GPT-4.1 mini", "Cheap and fast",
                  usd_in_per_m=0.40, usd_out_per_m=1.60),
            Model("genailab-maas-sonnet-4.6", "Sonnet 4.6", "Different vendor — real failover"),
        ],
        default_primary="azure/genailab-maas-gpt-4.1",
        default_mini="azure/genailab-maas-gpt-4.1-mini",
    ),
}


@dataclass
class Selection:
    provider: str
    primary: str
    mini: str

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "primary": self.primary, "mini": self.mini}


_lock = threading.Lock()
_override: Selection | None = None


def _from_env() -> Selection:
    """What the environment asks for. Inferred from the configured model ids so
    an existing .env keeps working without naming a provider."""
    settings = get_settings()
    named = os.getenv("LLM_PROVIDER", "").strip().lower()
    if named in PROVIDERS:
        provider = named
    else:
        provider = "genailab" if "genailab" in settings.llm_model_primary else "groq"

    spec = PROVIDERS[provider]
    return Selection(
        provider=provider,
        primary=settings.llm_model_primary or spec.default_primary,
        mini=settings.llm_model_mini or spec.default_mini,
    )


def current() -> Selection:
    with _lock:
        return _override or _from_env()


def select(provider: str, primary: str | None = None, mini: str | None = None) -> Selection:
    """Switch provider and/or models for every request from here on."""
    global _override
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    spec = PROVIDERS[provider]

    chosen_primary = primary or spec.default_primary
    chosen_mini = mini or spec.default_mini
    known = {m.id for m in spec.models}
    for model in (chosen_primary, chosen_mini):
        if model not in known:
            raise ValueError(f"{model} is not available on {spec.label}")

    with _lock:
        _override = Selection(provider, chosen_primary, chosen_mini)
        return _override


def reset() -> Selection:
    """Drop any override and go back to what the environment configures."""
    global _override
    with _lock:
        _override = None
    return _from_env()


def api_key(provider: str) -> str:
    """The key for one provider.

    Its own variable wins. The generic ``LLM_API_KEY`` is only accepted for the
    provider the environment actually points at — falling back unconditionally
    would send a Groq key to the lab endpoint and report both as configured.
    """
    spec = PROVIDERS[provider]
    own = os.getenv(spec.env_key)
    if own:
        return own
    if _from_env().provider == provider:
        return os.getenv("LLM_API_KEY") or get_settings().llm_api_key or ""
    return ""


def configured(provider: str) -> bool:
    return bool(api_key(provider))


def describe() -> dict[str, Any]:
    """Everything the switch UI needs to render itself."""
    active = current()
    return {
        "active": active.as_dict(),
        "providers": [
            {
                "key": spec.key,
                "label": spec.label,
                "base_url": spec.base_url,
                "note": spec.note,
                "configured": configured(spec.key),
                "verify_ssl": spec.verify_ssl,
                "models": [{"id": m.id, "label": m.label, "note": m.note}
                           for m in spec.models],
                "default_primary": spec.default_primary,
                "default_mini": spec.default_mini,
            }
            for spec in PROVIDERS.values()
        ],
    }


def price_for(model_ref: str) -> tuple[float, float]:
    """USD per million tokens for a metered model reference.

    Accepts what llm_call stores — "provider/model-id" — and tolerates a bare
    model id. Unknown models price at zero, which callers must present as
    "unknown", not as free.
    """
    ref = (model_ref or "").strip()
    for provider in PROVIDERS.values():
        for model in provider.models:
            # llm_call stores "groq/openai/gpt-oss-120b": the provider key, a
            # slash, then a model id that itself contains slashes. Suffix match
            # is what survives that, and a bare id still matches exactly.
            if ref == model.id or ref.endswith("/" + model.id):
                return model.usd_in_per_m, model.usd_out_per_m
    return 0.0, 0.0
