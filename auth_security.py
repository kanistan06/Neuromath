"""Security primitives for authentication and account recovery."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

import config


_COMMON_PASSWORDS = {
    "123456789012",
    "admin123456!",
    "letmein12345!",
    "password123!",
    "password1234",
    "qwerty123456!",
    "welcome12345!",
}


@dataclass(frozen=True)
class PasswordPolicyResult:
    valid: bool
    message: str = ""


def validate_password(password: str, *, email: str = "", name: str = "") -> PasswordPolicyResult:
    """Apply one policy to signup and password-reset operations."""
    value = str(password or "")
    if len(value) < config.PASSWORD_MIN_LENGTH:
        return PasswordPolicyResult(
            False,
            f"Password must contain at least {config.PASSWORD_MIN_LENGTH} characters.",
        )
    if len(value) > 128:
        return PasswordPolicyResult(False, "Password must not exceed 128 characters.")
    if not re.search(r"[a-z]", value):
        return PasswordPolicyResult(False, "Password must contain a lowercase letter.")
    if not re.search(r"[A-Z]", value):
        return PasswordPolicyResult(False, "Password must contain an uppercase letter.")
    if not re.search(r"\d", value):
        return PasswordPolicyResult(False, "Password must contain a number.")
    if not re.search(r"[^A-Za-z0-9]", value):
        return PasswordPolicyResult(False, "Password must contain a symbol.")
    if value.lower() in _COMMON_PASSWORDS:
        return PasswordPolicyResult(False, "Choose a less common password.")

    lowered = value.lower()
    local_part = str(email or "").split("@", 1)[0].strip().lower()
    if len(local_part) >= 4 and local_part in lowered:
        return PasswordPolicyResult(False, "Password must not contain your email name.")
    for part in re.findall(r"[a-z0-9]+", str(name or "").lower()):
        if len(part) >= 4 and part in lowered:
            return PasswordPolicyResult(False, "Password must not contain your name.")
    return PasswordPolicyResult(True)


def new_one_time_token() -> tuple[str, str]:
    """Return a URL-safe raw token and the only representation stored in the DB."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_token(raw)


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(str(raw_token or "").encode("utf-8")).hexdigest()


def new_numeric_otp(length: int = 6) -> str:
    """Generate a zero-padded cryptographically secure numeric OTP."""
    safe_length = max(6, min(10, int(length)))
    return f"{secrets.randbelow(10 ** safe_length):0{safe_length}d}"


def hash_otp(raw_otp: str, *, user_id: int, token_id: int) -> str:
    """
    Key the OTP digest with the Flask secret.

    A six-digit OTP has too little entropy for an ordinary database hash. HMAC
    prevents an attacker who only obtains the database from testing every code.
    Binding it to both row and user also prevents reuse across reset requests.
    """
    payload = f"reset_password:{int(user_id)}:{int(token_id)}:{raw_otp}".encode("utf-8")
    secret = str(config.FLASK_SECRET_KEY).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def rate_limit_identity(remote_addr: str, email: str = "") -> str:
    """Avoid storing raw email addresses in rate-limit keys."""
    normalized = f"{remote_addr or 'unknown'}|{str(email or '').strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

