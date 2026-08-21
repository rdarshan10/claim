"""OTP-simulated login (hackathon stand-in for OIDC)."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.audit import logger as audit
from app.db import query_one
from app.security import crypto, jwt

router = APIRouter(prefix="/auth", tags=["auth"])

DEMO_OTP = "000000"


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    otp: str = Field(default=DEMO_OTP, max_length=10)


class TokenResponse(BaseModel):
    token: str
    customer_id: str
    name: str
    role: str


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    if body.otp != DEMO_OTP:
        audit.record("auth_failure", actor_type="anonymous",
                     payload={"reason": "bad_otp"})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect code")

    row = query_one("SELECT id, full_name FROM customer WHERE email_hmac = ?",
                    (crypto.blind_index(body.email),))
    if row is None:
        # Same message whether or not the account exists (no enumeration).
        audit.record("auth_failure", actor_type="anonymous",
                     payload={"reason": "unknown_account"})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "We couldn't sign you in with those details")

    token = jwt.encode({"sub": row["id"], "name": row["full_name"], "role": "customer"})
    audit.record("auth_success", actor_type="customer", actor_id=row["id"],
                 entity_type="customer", entity_id=row["id"])
    return TokenResponse(token=token, customer_id=row["id"], name=row["full_name"],
                         role="customer")


@router.get("/me", response_model=TokenResponse)
async def me(authorization: str = Header(default="")) -> TokenResponse:
    """Identity for a token the client already holds.

    Lets a refreshed page restore its session instead of bouncing the customer
    back to the login screen mid-conversation.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(token)
    except jwt.AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    subject = claims.get("sub", "")
    if not subject.startswith("staff:"):
        if query_one("SELECT id FROM customer WHERE id = ?", (subject,)) is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown principal")

    return TokenResponse(token=token, customer_id=subject,
                         name=claims.get("name", ""),
                         role=claims.get("role", "customer"))


class StaffLoginRequest(BaseModel):
    username: str
    otp: str = DEMO_OTP


@router.post("/staff/login", response_model=TokenResponse)
async def staff_login(body: StaffLoginRequest) -> TokenResponse:
    """Staff SSO stub — role comes from the username prefix for the demo."""
    if body.otp != DEMO_OTP:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect code")
    role = "manager" if body.username.startswith("manager") else "agent"

    # Customers see this name in chat, so turn "agent.marcus" into "Marcus".
    display_name = (body.username.split(".")[-1].replace("_", " ").strip().title()
                    or body.username)

    token = jwt.encode({"sub": f"staff:{body.username}", "name": display_name,
                        "role": role})
    audit.record("auth_success", actor_type=role, actor_id=body.username)
    return TokenResponse(token=token, customer_id=f"staff:{body.username}",
                         name=display_name, role=role)
