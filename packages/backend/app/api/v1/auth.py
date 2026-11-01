import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, RequireWrite
from app.core.database import get_db
from app.core.limiter import limiter
from app.core.observability import log_event
from app.core.security import (
    compute_api_key_prefix,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    get_key_hint,
    hash_api_key,
    verify_password,
)
from app.models.api_key import ApiKey
from app.models.chat import TelegramChat
from app.models.message import TelegramMessage
from app.models.user import User
from app.schemas.auth import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    ApiKeyUpdateRequest,
    RefreshRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Login with email and password."""
    result = await db.execute(
        select(User).where(
            User.email == form_data.username,
            User.is_active.is_(True),
        )
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.password_hash):
        log_event(
            logger,
            logging.WARNING,
            "Login failed because credentials are invalid",
            event_name="auth.login.failed",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})
    log_event(
        logger,
        logging.INFO,
        "User login succeeded",
        event_name="auth.login.success",
        user_id=user.id,
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Refresh access token."""
    payload = decode_token(request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        log_event(
            logger,
            logging.WARNING,
            "Refresh token rejected",
            event_name="auth.refresh.invalid_token",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    result = await db.execute(
        select(User).where(
            User.id == UUID(user_id),
            User.is_active.is_(True),
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        log_event(
            logger,
            logging.WARNING,
            "Refresh token user not found",
            event_name="auth.refresh.user_not_found",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    access_token = create_access_token({"sub": str(user.id)})
    new_refresh_token = create_refresh_token({"sub": str(user.id)})
    log_event(
        logger,
        logging.INFO,
        "Access token refreshed successfully",
        event_name="auth.refresh.success",
        user_id=user.id,
    )

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: CurrentUser) -> UserResponse:
    """Get current user information."""
    return UserResponse(
        id=user.id,
        email=user.email,
        created_at=user.created_at,
        auto_index_since=user.auto_index_since,
    )


# --- API Key Management ---

MAX_API_KEYS_PER_USER = 25


@router.get("/api-keys", response_model=list[ApiKeyResponse])
@limiter.limit("30/minute")
async def list_api_keys(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ApiKeyResponse]:
    """List all API keys for the current user."""
    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == user.id)
        .order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_hint=k.key_hint,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            scopes=k.scopes.split(","),
        )
        for k in keys
    ]


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
@limiter.limit("10/minute")
async def create_api_key(
    request: Request,
    body: ApiKeyCreateRequest,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiKeyCreateResponse:
    """Create a new named API key."""
    user = ctx.user
    count = (
        await db.execute(
            select(func.count()).select_from(ApiKey).where(ApiKey.user_id == user.id)
        )
    ).scalar()
    if count >= MAX_API_KEYS_PER_USER:
        log_event(
            logger,
            logging.WARNING,
            "API key creation rejected because user reached limit",
            event_name="auth.api_key.create.limit_reached",
            user_id=user.id,
            existing_keys=count,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum of {MAX_API_KEYS_PER_USER} API keys allowed per user",
        )

    raw_key = generate_api_key()

    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now(UTC) + timedelta(days=body.expires_in_days)

    scopes_str = ",".join(body.scopes)

    api_key = ApiKey(
        user_id=user.id,
        name=body.name,
        key_hash=hash_api_key(raw_key),
        key_prefix=compute_api_key_prefix(raw_key),
        key_hint=get_key_hint(raw_key),
        expires_at=expires_at,
        scopes=scopes_str,
    )
    db.add(api_key)
    await db.flush()
    log_event(
        logger,
        logging.INFO,
        "API key created successfully",
        event_name="auth.api_key.create.success",
        user_id=user.id,
        key_id=api_key.id,
        scopes=body.scopes,
        expires_in_days=body.expires_in_days,
    )

    return ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        api_key=raw_key,
        key_hint=api_key.key_hint,
        expires_at=api_key.expires_at,
        scopes=body.scopes,
    )


@router.post("/api-keys/test")
@limiter.limit("10/minute")
async def test_mcp_connection(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Test MCP connection - returns backend health + user data summary."""
    chat_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramChat)
            .where(TelegramChat.user_id == user.id)
        )
    ).scalar()
    message_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramMessage)
            .join(TelegramChat, TelegramMessage.chat_id == TelegramChat.id)
            .where(TelegramChat.user_id == user.id)
        )
    ).scalar()

    return {
        "success": True,
        "message": "Backend is reachable and your data is available.",
        "chat_count": chat_count,
        "message_count": message_count,
    }


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")
async def revoke_api_key(
    request: Request,
    key_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Revoke (delete) an API key."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        log_event(
            logger,
            logging.WARNING,
            "API key revocation requested for unknown key",
            event_name="auth.api_key.revoke.not_found",
            user_id=user.id,
            key_id=key_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )
    await db.delete(api_key)
    await db.flush()
    log_event(
        logger,
        logging.INFO,
        "API key revoked successfully",
        event_name="auth.api_key.revoke.success",
        user_id=user.id,
        key_id=key_id,
    )


@router.patch("/api-keys/{key_id}", response_model=ApiKeyResponse)
@limiter.limit("10/minute")
async def toggle_api_key(
    request: Request,
    key_id: UUID,
    body: ApiKeyUpdateRequest,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiKeyResponse:
    """Toggle an API key active/inactive."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        log_event(
            logger,
            logging.WARNING,
            "API key toggle requested for unknown key",
            event_name="auth.api_key.toggle.not_found",
            user_id=user.id,
            key_id=key_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    api_key.is_active = body.is_active
    await db.flush()
    log_event(
        logger,
        logging.INFO,
        "API key state updated",
        event_name="auth.api_key.toggle.success",
        user_id=user.id,
        key_id=key_id,
        is_active=body.is_active,
    )

    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_hint=api_key.key_hint,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        scopes=api_key.scopes.split(","),
    )
