"""Minimal reference: how ClaimCompanion talks to the GenAI Lab endpoint.

This is the bare invoke pattern the whole application is built on. In the app
itself, *only* ``backend/app/llm/gateway.py`` is permitted to call an LLM — it
wraps this same pattern with versioned prompts, structured-output parsing,
model tiering, retries, cost metering, audit logging and a template fallback for
when the endpoint is unavailable.

The key is read from the environment. Never hard-code it: anything committed to
a repository should be assumed public forever, even in a private repo.

    setx API_KEY "sk-..."        # once, then reopen the terminal
    python examples/model_invoke.py
"""
from __future__ import annotations

import os
import sys

import httpx
from langchain_openai import ChatOpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "https://genailab.tcs.in")
MODEL = os.getenv("LLM_MODEL_PRIMARY", "azure/genailab-maas-gpt-4.1")
API_KEY = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")

if not API_KEY:
    sys.exit("Set API_KEY in your environment first (see .env.example).")

# verify=False matches the lab endpoint's certificate setup. Do not carry this
# into anything production-facing — it disables TLS verification entirely.
client = httpx.Client(verify=False, timeout=60.0)

llm = ChatOpenAI(
    base_url=BASE_URL,
    model=MODEL,
    api_key=API_KEY,
    http_client=client,
)

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    response = llm.invoke(prompt)
    print(response.content)
