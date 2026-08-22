"""LLM Gateway — the ONLY module in the app permitted to call an LLM API.

Wraps the provider SDK (Groq's OpenAI-compatible chat-completions API) and adds
the things a production caller needs:
versioned prompts, structured-output coercion, retries, cost metering, audit
trails, and a template fallback so the product still works with no LLM at all
(§10, UC-N7).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.audit import logger as audit
from app.config import get_settings
from app.db import execute
from app.guardrails.pii import redact
from app.llm import providers

PROMPT_DIR = Path(__file__).parent / "prompts"


class LLMUnavailable(Exception):
    """Raised when every model tier failed; callers must degrade gracefully."""


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_key: str
    prompt_version: str
    latency_ms: int
    degraded: bool = False

    def json(self, default: Any = None) -> Any:
        return extract_json(self.text, default)


# --------------------------------------------------------------------------
# Prompt registry (versioned YAML, §10 "prompts live only in prompts/*.yaml")
# --------------------------------------------------------------------------
_prompt_cache: dict[str, dict[str, Any]] = {}


def load_prompt(key: str) -> dict[str, Any]:
    if key in _prompt_cache:
        return _prompt_cache[key]

    path = PROMPT_DIR / f"{key}.yaml"
    if not path.exists():
        raise KeyError(f"Unknown prompt key: {key}")

    import yaml  # available in the venv

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "version" not in data:
        raise ValueError(f"Prompt {key} is missing a version field")
    _prompt_cache[key] = data
    return data


def render(template: str, variables: dict[str, Any]) -> str:
    """Safe ``{{var}}`` substitution — no f-strings, no eval."""
    def replace(match: re.Match[str]) -> str:
        return str(variables.get(match.group(1).strip(), ""))

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)


def extract_json(text: str, default: Any = None) -> Any:
    """Pull the first JSON object/array out of a model reply."""
    if not text:
        return default
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
# Keyed on (provider, timeout): switching provider must not reuse a client
# pointed at the previous endpoint.
_clients: dict[tuple[str, float], Any] = {}

# ("system", "human") is a LangChain-ism the rest of this module speaks; the
# OpenAI-shaped APIs want ("system", "user").
_ROLE_MAP = {"system": "system", "human": "user", "user": "user", "ai": "assistant"}


def reset_clients() -> None:
    """Drop cached clients. Called when the provider selection changes so the
    next request builds a client against the new endpoint."""
    _clients.clear()


def _client(provider_key: str, timeout: float) -> Any:
    """One client per (provider, timeout). Cached: constructing one opens a pool."""
    key = (provider_key, timeout)
    if key in _clients:
        return _clients[key]

    settings = get_settings()
    if not settings.llm_enabled:
        raise LLMUnavailable("LLM disabled")

    spec = providers.PROVIDERS[provider_key]
    api_key = providers.api_key(provider_key)
    if not api_key:
        raise LLMUnavailable(f"No API key configured for {spec.label}")

    if spec.transport == "groq":
        from groq import Groq

        client = Groq(api_key=api_key, base_url=spec.base_url, timeout=timeout,
                      max_retries=0)  # the tier ladder below is our retry policy
    else:
        # GenAI Lab and anything else OpenAI-shaped. verify comes from the
        # provider, not global settings — the lab endpoint's certificate does
        # not validate and that must not weaken any other provider.
        import httpx
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            base_url=spec.base_url,
            model=providers.current().primary,   # replaced per call below
            api_key=api_key,
            http_client=httpx.Client(verify=spec.verify_ssl, timeout=timeout),
            max_retries=0,
        )

    _clients[key] = client
    return client


def _invoke(provider_key: str, model: str, payload: list[tuple[str, str]],
            timeout: float) -> tuple[str, int, int]:
    """Single completion, with the token usage the provider reports.

    Returns ``(text, tokens_in, tokens_out)``. The counts come from the
    response rather than an estimate: llm_call has always had the columns but
    metered zeros, so nothing downstream could report real spend. A provider
    that omits usage yields zeros, which read as "unknown", not "free"."""
    settings = get_settings()
    spec = providers.PROVIDERS[provider_key]
    client = _client(provider_key, timeout)

    if spec.transport == "groq":
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": _ROLE_MAP.get(role, role), "content": text}
                      for role, text in payload],
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
            top_p=1,
            stream=False,
        )
        usage = getattr(completion, "usage", None)
        return (completion.choices[0].message.content or "",
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0))

    # LangChain client: the model is bound at construction, so rebind per call
    # rather than caching a client per model as well as per timeout.
    response = client.bind(model=model).invoke(payload)
    text = (response.content if isinstance(response.content, str)
            else str(response.content))
    # LangChain surfaces the same numbers under a different name.
    meta = getattr(response, "usage_metadata", None) or {}
    return (text, int(meta.get("input_tokens", 0) or 0),
            int(meta.get("output_tokens", 0) or 0))


# --------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------
# Knowledge answers repeat heavily ("what is excess?"), and classification of
# identical short messages does too. Keyed on prompt + version + model so a
# prompt edit or model swap can never serve a stale answer.
_cache: "OrderedDict[str, str]" = OrderedDict()
_CACHE_MAX = 256
CACHEABLE = {"classify_turn", "sentiment", "router", "knowledge_answer"}


def _cache_key(prompt_key: str, version: str, model: str, payload: list) -> str:
    # Provider is part of the key: the same model id on a different endpoint is
    # a different answer, and a stale hit would hide the switch entirely.
    material = f"{prompt_key}|{version}|{model}|{payload}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: str, value: str) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def clear_cache() -> None:
    """Drop cached answers. Called on a provider or model switch — a cached
    reply from the previous model would hide the change entirely."""
    _cache.clear()


def _avoided_tokens(prompt_key: str, provider_key: str, model: str) -> tuple[int, int]:
    """What a cache hit saved, from the last real call to the same prompt+model.

    Uses a measured previous call rather than an average so the figure is
    defensible. No prior call means zeros — better an understated saving than
    an invented one.
    """
    from app.db import query_one

    row = query_one(
        """SELECT tokens_in, tokens_out FROM llm_call
           WHERE prompt_key = ? AND model = ? AND cached = 0 AND tokens_in > 0
           ORDER BY at DESC LIMIT 1""",
        (prompt_key, f"{provider_key}/{model}"),
    )
    return (int(row["tokens_in"]), int(row["tokens_out"])) if row else (0, 0)


def _meter(prompt_key: str, version: str, model: str, latency_ms: int, ok: bool,
           trace_id: str | None, tokens_in: int = 0, tokens_out: int = 0,
           cached: bool = False) -> None:
    """Record one call. `cached` is a real column rather than a suffix on the
    model name — served-from-cache is the single biggest cost lever here, and
    string-matching " (cached)" to find it was never going to survive."""
    execute(
        """INSERT INTO llm_call (id, at, prompt_key, prompt_version, model,
                                 tokens_in, tokens_out, latency_ms, ok, trace_id,
                                 cached)
           VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), prompt_key, version, model, tokens_in, tokens_out,
         latency_ms, 1 if ok else 0, trace_id, 1 if cached else 0),
    )


def complete(
    prompt_key: str,
    variables: dict[str, Any] | None = None,
    *,
    tier: str = "mini",
    trace_id: str | None = None,
    fallback: str | None = None,
) -> LLMResult:
    """Render a versioned prompt, call the model, meter and audit the call.

    Falls back through model tiers, then to ``fallback`` text. Raises
    ``LLMUnavailable`` only if no fallback was supplied.
    """
    settings = get_settings()
    spec = load_prompt(prompt_key)
    version = str(spec["version"])
    variables = variables or {}

    system_prompt = render(spec.get("system", ""), variables)
    user_prompt = render(spec.get("user", ""), variables)

    # PII never leaves the process un-redacted (§17.1).
    payload = [
        ("system", redact(system_prompt)),
        ("human", redact(user_prompt)),
    ]

    # Failover ladder: requested tier -> primary -> cross-provider fallback.
    # Deduplicated so a shared model name doesn't produce a pointless retry.
    #
    # Mini-tier calls do NOT climb the ladder: routing and sentiment have good
    # deterministic fallbacks, so one fast attempt then heuristics beats three
    # slow failures in front of a waiting customer.
    # Models come from the live selection, not the environment, so a switch in
    # the console takes effect on the very next turn.
    selection = providers.current()
    provider_key = selection.provider
    if tier == "mini":
        ladder = [selection.mini]
    else:
        ladder = [selection.primary, selection.mini]
    tiers = list(dict.fromkeys(m for m in ladder if m))

    # Fast-fail the first attempt so a dead endpoint costs seconds, not minutes.
    timeouts = [settings.llm_first_attempt_timeout] + [settings.llm_timeout_seconds] * 3

    cacheable = prompt_key in CACHEABLE
    last_error: Exception | None = None

    for attempt, model in enumerate(tiers):
        cache_key = (_cache_key(prompt_key, version, f"{provider_key}/{model}", payload)
                     if cacheable else "")
        if cache_key and (hit := _cache_get(cache_key)) is not None:
            # Metered with the tokens this call avoided, taken from the last
            # real call to the same prompt+model. That is what makes the cache
            # saving a measured number rather than an assertion.
            avoided_in, avoided_out = _avoided_tokens(prompt_key, provider_key, model)
            _meter(prompt_key, version, f"{provider_key}/{model}", 0, True, trace_id,
                   avoided_in, avoided_out, cached=True)
            return LLMResult(hit, model, prompt_key, version, 0)

        started = time.perf_counter()
        try:
            text, tokens_in, tokens_out = _invoke(
                provider_key, model, payload, timeouts[attempt])
            latency_ms = int((time.perf_counter() - started) * 1000)

            _meter(prompt_key, version, f"{provider_key}/{model}", latency_ms, True,
                   trace_id, tokens_in, tokens_out)
            audit.record(
                "llm_call",
                entity_type="prompt",
                entity_id=prompt_key,
                payload={"tier": tier, "latency_ms": latency_ms, "chars_out": len(text)},
                prompt_version=version,
                model=model,
                trace_id=trace_id,
            )
            if cache_key and text.strip():
                _cache_put(cache_key, text)
            return LLMResult(text, model, prompt_key, version, latency_ms)
        except Exception as exc:  # noqa: BLE001 - any transport error degrades the tier
            last_error = exc
            latency_ms = int((time.perf_counter() - started) * 1000)
            _meter(prompt_key, version, f"{provider_key}/{model}", latency_ms, False, trace_id)

    audit.record(
        "llm_unavailable",
        entity_type="prompt",
        entity_id=prompt_key,
        payload={"error": str(last_error)[:400], "degraded": fallback is not None},
        prompt_version=version,
        trace_id=trace_id,
    )

    if fallback is not None:
        return LLMResult(fallback, "template-fallback", prompt_key, version, 0, degraded=True)
    raise LLMUnavailable(str(last_error))


def cost_summary() -> list[dict[str, Any]]:
    """Spend per prompt and model, priced from the provider's published rates.

    This used to count calls and average latency under the name "cost" while
    metering zero tokens, so it could not report spend at all. Cached rows are
    split out: their tokens are what was *avoided*, and adding those to the
    bill would invert the number.
    """
    from app.db import query

    rows = query(
        """SELECT model, prompt_key, cached,
                  COUNT(*) AS calls, SUM(ok) AS ok_calls,
                  AVG(latency_ms) AS avg_latency_ms,
                  SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out
           FROM llm_call
           GROUP BY model, prompt_key, cached
           ORDER BY calls DESC"""
    )

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        rate_in, rate_out = providers.price_for(item["model"])
        tokens_in = int(item["tokens_in"] or 0)
        tokens_out = int(item["tokens_out"] or 0)
        usd = (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000
        item["cached"] = bool(item["cached"])
        item["priced"] = bool(rate_in or rate_out)
        # For a cached row this is what the hit avoided, not what it cost.
        item["usd"] = round(usd, 6)
        out.append(item)
    return out
