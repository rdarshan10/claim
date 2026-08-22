"""Central, env-driven configuration. No secrets in code."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # claimcompanion/ (app -> backend -> here)


def _load_dotenv() -> None:
    """Minimal .env loader (python-dotenv is not a dependency)."""
    for candidate in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


class Settings:
    """All runtime settings come from the environment."""

    app_name: str = "ClaimCompanion"
    api_prefix: str = "/api/v1"

    # --- storage -------------------------------------------------------
    db_path: str = os.getenv("DB_PATH", str(BASE_DIR / "data" / "claimcompanion.db"))
    blob_dir: str = os.getenv("BLOB_DIR", str(BASE_DIR / "data" / "blobs"))

    # --- auth ----------------------------------------------------------
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-only-change-me")
    jwt_ttl_seconds: int = int(os.getenv("JWT_TTL_SECONDS", "3600"))

    # --- llm -----------------------------------------------------------
    # Model tiering (§10): cheap/fast for the ~90% of calls that are routing and
    # sentiment; a stronger instruction-follower for extraction, empathy and the
    # Smart Rejection Explanation.
    # Provider is Groq (OpenAI-compatible chat completions). The mini tier is the
    # 20B model — routing and sentiment are easy calls made on every turn, and
    # the daily request budget is shared, so spend the 120B only where it earns
    # its keep. Fallback is the 20B again: same vendor, so this is a retry rather
    # than true cross-provider redundancy — a Groq-wide outage degrades us to
    # template mode, which is the designed behaviour (UC-N7).
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com")
    llm_model_mini: str = os.getenv("LLM_MODEL_MINI", "openai/gpt-oss-20b")
    llm_model_primary: str = os.getenv("LLM_MODEL_PRIMARY", "openai/gpt-oss-120b")
    llm_model_fallback: str = os.getenv("LLM_MODEL_FALLBACK", "openai/gpt-oss-20b")
    # Unused today (RAG is pure-Python BM25); Groq serves no embedding model.
    llm_model_embedding: str = os.getenv("LLM_MODEL_EMBEDDING", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", os.getenv("API_KEY", "")))
    llm_verify_ssl: bool = os.getenv("LLM_VERIFY_SSL", "true").lower() == "true"
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    # Most calls here ask for JSON against a closed enum; sampling at 1.0 only
    # adds format drift for no benefit.
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # Groq caps this tier at 8K tokens/minute, so a runaway reply would eat the
    # whole minute's budget in one call.
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    # Fast-fail the first attempt: a dead endpoint should cost seconds, not the
    # full ladder of 60s timeouts in front of a waiting customer.
    llm_first_attempt_timeout: float = float(
        os.getenv("LLM_FIRST_ATTEMPT_TIMEOUT", "20")
    )
    llm_enabled: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"

    # --- registration bot (RPA) ----------------------------------------
    # Where the bot's browser reaches the core system. Same process serves it,
    # so this is the app's own address as seen from the bot's browser.
    rpa_portal_url: str = os.getenv("RPA_PORTAL_URL", "http://127.0.0.1:8010")

    # --- guardrails / limits -------------------------------------------
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    max_turn_hops: int = int(os.getenv("MAX_TURN_HOPS", "3"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # --- confidence thresholds (§11.6) ---------------------------------
    ocr_quality_floor: float = 0.55
    classification_floor: float = 0.70
    auto_verdict_floor: float = 0.70
    rag_relevance_floor: float = float(os.getenv("RAG_RELEVANCE_FLOOR", "0.12"))

    def __init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.blob_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
