from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RequestCodeRequest(BaseModel):
    phone_number: str


class RequestCodeResponse(BaseModel):
    phone_code_hash: str
    code_type: str = (
        "unknown"  # app, sms, call, flash_call, missed_call, email, fragment_sms
    )
    message: str = "Verification code sent"


class VerifyCodeRequest(BaseModel):
    phone_number: str
    phone_code_hash: str
    code: str
    password: str | None = None  # For 2FA


class VerifyCodeResponse(BaseModel):
    success: bool
    telegram_user_id: int
    access_token: str | None = None
    refresh_token: str | None = None
    message: str = "Successfully authenticated with Telegram"


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    phone_number: str
    telegram_user_id: int | None
    is_active: bool
    created_at: datetime
