import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from cryptography.fernet import Fernet
from jwt.exceptions import PyJWTError
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import get_settings

settings = get_settings()

password_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))


def get_fernet() -> Fernet:
    if not settings.encryption_key:
        raise ValueError("ENCRYPTION_KEY not configured")
    return Fernet(settings.encryption_key.encode())


def encrypt_session(session_string: str) -> str:
    fernet = get_fernet()
    return fernet.encrypt(session_string.encode()).decode()


def decrypt_session(encrypted_session: str) -> str:
    fernet = get_fernet()
    return fernet.decrypt(encrypted_session.encode()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def hash_api_key(api_key: str) -> str:
    """Hash an API key with SHA-256.

    Argon2 exists to make guessing a human-chosen password expensive. API keys
    are generated random tokens with far more entropy than any attacker can
    search, so the memory-hard KDF buys nothing - and it cost 387ms on every
    single authenticated request, which was the largest component of search
    latency. GitHub and Stripe hash their tokens the same way.
    """
    return "sha256$" + hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify against SHA-256, falling back to the old KDF for existing keys."""
    if hashed_key.startswith("sha256$"):
        return hmac.compare_digest(hashed_key, hash_api_key(api_key))
    # Keys issued before the change still carry an Argon2 or bcrypt hash.
    return password_hash.verify(api_key, hashed_key)


def compute_api_key_prefix(api_key: str) -> str:
    """Compute a SHA256 prefix for O(1) API key lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def get_key_hint(api_key: str) -> str:
    """Extract last 4 chars for display: wai_****abcd."""
    return f"wai_****{api_key[-4:]}"


def generate_api_key() -> str:
    return f"wai_{secrets.token_urlsafe(32)}"


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def create_onboarding_token(user_id: str) -> str:
    """An invitation only permits Telegram login, never archive or API-key access."""
    return jwt.encode(
        {
            "sub": user_id,
            "type": "onboarding",
            "jti": secrets.token_urlsafe(24),
            "exp": datetime.now(UTC) + timedelta(minutes=30),
        },
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"require": ["exp", "type"]},
        )
        return payload
    except PyJWTError:
        return None
