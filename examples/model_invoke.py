"""Minimal reference: how ClaimCompanion talks to the model provider.

This is the bare invoke pattern the whole application is built on. In the app
itself, *only* ``backend/app/llm/gateway.py`` is permitted to call an LLM — it
wraps this same pattern with versioned prompts, structured-output parsing,
model tiering, retries, cost metering, audit logging and a template fallback for
when the endpoint is unavailable.

The key is read from the environment. Never hard-code it: anything committed to
a repository should be assumed public forever, even in a private repo.

    setx GROQ_API_KEY "gsk_..."   # once, then reopen the terminal
    python examples/model_invoke.py

    # or against the lab endpoint
    setx LLM_PROVIDER genailab
    setx GENAILAB_API_KEY "sk-..."
"""
from __future__ import annotations

import os
import sys

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def groq_invoke(prompt: str) -> None:
    from groq import Groq

    key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY first (see .env.example).")

    client = Groq(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com"))
    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_PRIMARY", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_completion_tokens=2048, top_p=1, stream=True,
    )
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
    print()


def genailab_invoke(prompt: str) -> None:
    """The lab endpoint speaks the OpenAI shape but presents a certificate that
    does not validate, which is why verify=False appears here and nowhere else."""
    import httpx
    from langchain_openai import ChatOpenAI

    key = os.getenv("GENAILAB_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    if not key:
        sys.exit("Set GENAILAB_API_KEY first (see .env.example).")

    llm = ChatOpenAI(
        base_url=os.getenv("GENAILAB_BASE_URL", "https://genailab.tcs.in"),
        model=os.getenv("LLM_MODEL_PRIMARY", "azure/genailab-maas-gpt-4.1"),
        api_key=key,
        http_client=httpx.Client(verify=False, timeout=60.0),
    )
    print(llm.invoke(prompt).content)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    if PROVIDER == "genailab":
        genailab_invoke(text)
    else:
        groq_invoke(text)
