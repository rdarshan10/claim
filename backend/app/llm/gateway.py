"""LLM Gateway — the ONLY module in the app permitted to call an LLM API.

Wraps the invocation pattern from ``model.py`` (langchain_openai.ChatOpenAI
against the GenAI Lab endpoint) and adds the things a production caller needs:
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
_clients: dict[tuple[str, float], Any] = {}


def _client(model: str, timeout: float) -> Any:
    key = (model, timeout)
    if key in _clients:
        return _clients[key]

    settings = get_settings()
    if not settings.llm_enabled or not settings.llm_api_key:
        raise LLMUnavailable("LLM disabled or API key missing")

    import httpx
    from langchain_openai import ChatOpenAI

    http_client = httpx.Client(verify=settings.llm_verify_ssl, timeout=timeout)
    _clients[key] = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=model,
        api_key=settings.llm_api_key,
        http_client=http_client,
        max_retries=0,  # the ladder below is our retry policy
    )
    return _clients[key]


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


def _meter(prompt_key: str, version: str, model: str, latency_ms: int, ok: bool,
           trace_id: str | None) -> None:
    execute(
        """INSERT INTO llm_call (id, at, prompt_key, prompt_version, model,
                                 tokens_in, tokens_out, latency_ms, ok, trace_id)
           VALUES (?, datetime('now'), ?, ?, ?, 0, 0, ?, ?, ?)""",
        (str(uuid.uuid4()), prompt_key, version, model, latency_ms, 1 if ok else 0, trace_id),
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
    if tier == "mini":
        ladder = [settings.llm_model_mini]
    else:
        ladder = [settings.llm_model_primary, settings.llm_model_fallback]
    tiers = list(dict.fromkeys(m for m in ladder if m))

    # Fast-fail the first attempt so a dead endpoint costs seconds, not minutes.
    timeouts = [settings.llm_first_attempt_timeout] + [settings.llm_timeout_seconds] * 3

    cacheable = prompt_key in CACHEABLE
    last_error: Exception | None = None

    for attempt, model in enumerate(tiers):
        cache_key = _cache_key(prompt_key, version, model, payload) if cacheable else ""
        if cache_key and (hit := _cache_get(cache_key)) is not None:
            _meter(prompt_key, version, f"{model} (cached)", 0, True, trace_id)
            return LLMResult(hit, model, prompt_key, version, 0)

        started = time.perf_counter()
        try:
            response = _client(model, timeouts[attempt]).invoke(payload)
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = response.content if isinstance(response.content, str) else str(response.content)

            _meter(prompt_key, version, model, latency_ms, True, trace_id)
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
            _meter(prompt_key, version, model, latency_ms, False, trace_id)

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
    from app.db import query

    rows = query(
        """SELECT model, prompt_key, COUNT(*) AS calls,
                  SUM(ok) AS ok_calls, AVG(latency_ms) AS avg_latency_ms
           FROM llm_call GROUP BY model, prompt_key ORDER BY calls DESC"""
    )
    return [dict(row) for row in rows]
