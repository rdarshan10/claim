"""Minimal HS256 JWT (no external dependency).

Identity always originates here: ``customer_id`` is read from the token and
never from message content (§17 "identity from token only").
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.config import get_settings


class AuthError(Exception):
    pass


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(message: bytes) -> str:
    secret = get_settings().jwt_secret.encode("utf-8")
    return _b64e(hmac.new(secret, message, hashlib.sha256).digest())


def encode(claims: dict[str, Any], ttl_seconds: int | None = None) -> str:
    settings = get_settings()
    ttl = ttl_seconds if ttl_seconds is not None else settings.jwt_ttl_seconds
    payload = {**claims, "iat": int(time.time()), "exp": int(time.time()) + ttl}
    header = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode("ascii")
    return f"{header}.{body}.{_sign(signing_input)}"


def decode(token: str) -> dict[str, Any]:
    try:
        header, body, signature = token.split(".")
    except ValueError as exc:
        raise AuthError("Malformed token") from exc

    expected = _sign(f"{header}.{body}".encode("ascii"))
    if not hmac.compare_digest(expected, signature):
        raise AuthError("Bad signature")

    claims: dict[str, Any] = json.loads(_b64d(body))
    if claims.get("exp", 0) < time.time():
        raise AuthError("Token expired")
    return claims
