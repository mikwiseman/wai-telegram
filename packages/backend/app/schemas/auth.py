from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    created_at: datetime
    auto_index_since: datetime | None = None


# --- API Key schemas ---


VALID_SCOPES = {"read", "write", "draft"}


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    expires_in_days: int | None = Field(
        default=365,
        ge=1,
        le=3650,
        description="Key expiration in days (1-3650). Null for no expiration.",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["read", "write"],
        description="Permission scopes: 'read', 'draft' and/or 'write'.",
    )

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one scope is required")
        invalid = set(v) - VALID_SCOPES
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Valid scopes: {VALID_SCOPES}")
        return sorted(set(v))


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    key_hint: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    scopes: list[str]


class ApiKeyCreateResponse(BaseModel):
    id: UUID
    name: str
    api_key: str
    key_hint: str
    expires_at: datetime | None
    scopes: list[str]
    message: str = "Store this API key securely. It won't be shown again."


class ApiKeyUpdateRequest(BaseModel):
    is_active: bool
