"""Field-level protection for PII columns.

MVP uses HMAC-derived keystream encryption + a deterministic HMAC index so
lookups by email still work. The interface matches what an AES-GCM/KMS
implementation would expose, so it is a drop-in swap later.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

from app.config import get_settings


def _key() -> bytes:
    return hashlib.sha256(
        (os.getenv("FIELD_ENCRYPTION_KEY") or get_settings().jwt_secret).encode("utf-8")
    ).digest()


def _keystream(nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(_key(), nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def encrypt(plaintext: str) -> str:
    if plaintext is None:
        return ""
    raw = plaintext.encode("utf-8")
    nonce = os.urandom(12)
    cipher = bytes(a ^ b for a, b in zip(raw, _keystream(nonce, len(raw))))
    tag = hmac.new(_key(), nonce + cipher, hashlib.sha256).digest()[:16]
    return base64.b64encode(nonce + tag + cipher).decode("ascii")


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    blob = base64.b64decode(ciphertext)
    nonce, tag, cipher = blob[:12], blob[12:28], blob[28:]
    expected = hmac.new(_key(), nonce + cipher, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Ciphertext failed integrity check")
    return bytes(a ^ b for a, b in zip(cipher, _keystream(nonce, len(cipher)))).decode("utf-8")


def blind_index(value: str) -> str:
    """Deterministic, searchable index over an encrypted field."""
    return hmac.new(_key(), value.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()
