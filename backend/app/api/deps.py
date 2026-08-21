"""Auth, RBAC and rate-limit dependencies."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings
from app.db import query_one
from app.security import jwt

_buckets: dict[str, deque[float]] = defaultdict(deque)


class Principal:
    def __init__(self, claims: dict[str, Any]) -> None:
        self.customer_id: str = claims.get("sub", "")
        self.name: str = claims.get("name", "")
        self.role: str = claims.get("role", "customer")


async def current_principal(authorization: str = Header(default="")) -> Principal:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        claims = jwt.decode(authorization.split(" ", 1)[1].strip())
    except jwt.AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    principal = Principal(claims)
    if principal.role == "customer":
        exists = query_one("SELECT id FROM customer WHERE id = ?", (principal.customer_id,))
        if exists is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown principal")
    return principal


def require_role(*roles: str):
    async def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")
        return principal

    return dependency


async def rate_limit(principal: Principal = Depends(current_principal)) -> Principal:
    limit = get_settings().rate_limit_per_minute
    now = time.time()
    bucket = _buckets[principal.customer_id]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many requests — please slow down.")
    bucket.append(now)
    return principal
