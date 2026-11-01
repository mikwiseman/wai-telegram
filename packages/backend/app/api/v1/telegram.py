import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telethon import TelegramClient
from telethon.errors import (
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
)

from app.core.auth import AuthContext, RequireWrite, get_auth_context
from app.core.security import create_access_token, create_refresh_token
from app.core.database import get_db
from app.core.limiter import limiter
from app.models.session import TelegramSession
from app.schemas.telegram import (
    RequestCodeRequest,
    RequestCodeResponse,
    SessionResponse,
    VerifyCodeRequest,
    VerifyCodeResponse,
)
from app.services import telegram_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Auth clients with TTL tracking: key -> (client, created_timestamp)
MAX_AUTH_CLIENTS = 50
_auth_clients: dict[str, tuple[TelegramClient, float]] = {}
_AUTH_CLIENT_TTL = 300  # 5 minutes


def _auth_client_key(ctx: AuthContext, phone: str) -> str:
    # Separate pending Telegram authorizations for independently issued invitations.
    identity = f"{ctx.user.id}:{ctx.session_id}" if ctx.session_id else str(ctx.user.id)
    return f"{identity}:{phone}"


async def _disconnect_auth_client(client: TelegramClient) -> None:
    try:
        await client.disconnect()
    except Exception as e:
        logger.debug(f"Client disconnect error (non-critical): {e}")


async def _replace_auth_client(client_key: str, client: TelegramClient) -> None:
    existing = _auth_clients.get(client_key)
    if existing:
        await _disconnect_auth_client(existing[0])
    _auth_clients[client_key] = (client, time.time())


async def _pop_and_disconnect_auth_client(client_key: str) -> None:
    entry = _auth_clients.pop(client_key, None)
    if not entry:
        return
    await _disconnect_auth_client(entry[0])


async def _cleanup_expired_auth_clients() -> None:
    """Remove auth clients older than TTL, with proper async disconnect."""
    now = time.time()
    expired = [k for k, (_, ts) in _auth_clients.items() if now - ts > _AUTH_CLIENT_TTL]
    for key in expired:
        client, _ = _auth_clients.pop(key)
        logger.info("Cleaned up expired Telegram authentication client")
        await _disconnect_auth_client(client)

    # Enforce max limit — evict oldest if over cap
    if len(_auth_clients) > MAX_AUTH_CLIENTS:
        sorted_keys = sorted(_auth_clients, key=lambda k: _auth_clients[k][1])
        for key in sorted_keys[: len(_auth_clients) - MAX_AUTH_CLIENTS]:
            client, _ = _auth_clients.pop(key)
            logger.info("Evicted Telegram authentication client over limit")
            await _disconnect_auth_client(client)


@router.post("/request-code", response_model=RequestCodeResponse)
@limiter.limit("3/minute")
async def request_code(
    request: Request,
    body: RequestCodeRequest,
    ctx: RequireWrite,
) -> RequestCodeResponse:
    """Request verification code for Telegram authentication."""
    await _cleanup_expired_auth_clients()
    try:
        client, phone_code_hash, code_type = await telegram_client.request_code(
            body.phone_number
        )
        await _replace_auth_client(_auth_client_key(ctx, body.phone_number), client)
        return RequestCodeResponse(
            phone_code_hash=phone_code_hash,
            code_type=code_type,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/verify-code", response_model=VerifyCodeResponse)
@limiter.limit("10/minute")
async def verify_code(
    request: Request,
    body: VerifyCodeRequest,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VerifyCodeResponse:
    """Verify code and complete Telegram authentication."""
    user = ctx.user
    await _cleanup_expired_auth_clients()
    client_key = _auth_client_key(ctx, body.phone_number)
    entry = _auth_clients.get(client_key)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending authentication. Request a code first.",
        )
    client, _ = entry

    try:
        session_string, telegram_user_id = await telegram_client.verify_code(
            client,
            body.phone_number,
            body.phone_code_hash,
            body.code,
            body.password,
        )

        if (
            user.expected_telegram_user_id is not None
            and telegram_user_id != user.expected_telegram_user_id
        ):
            await _pop_and_disconnect_auth_client(client_key)
            raise HTTPException(
                403, "Это приглашение предназначено для другого Telegram-аккаунта."
            )

        # Save session
        await telegram_client.save_session(
            db,
            user.id,
            body.phone_number,
            session_string,
            telegram_user_id,
        )
        await _pop_and_disconnect_auth_client(client_key)

        return VerifyCodeResponse(
            success=True,
            telegram_user_id=telegram_user_id,
            access_token=create_access_token({"sub": str(user.id)})
            if ctx.auth_type == "onboarding"
            else None,
            refresh_token=create_refresh_token({"sub": str(user.id)})
            if ctx.auth_type == "onboarding"
            else None,
        )
    except HTTPException:
        raise
    except (PhoneCodeInvalidError, PasswordHashInvalidError):
        raise HTTPException(400, "Неверный код или пароль. Попробуйте ещё раз.")
    except PhoneCodeExpiredError:
        await _pop_and_disconnect_auth_client(client_key)
        raise HTTPException(400, "Код истёк. Запросите новый.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        await _pop_and_disconnect_auth_client(client_key)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/session", response_model=SessionResponse | None)
async def get_session(
    ctx: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionResponse | None:
    """Get current Telegram session info."""
    if ctx.auth_type == "onboarding":
        return None
    user = ctx.user
    result = await db.execute(
        select(TelegramSession).where(
            TelegramSession.user_id == user.id,
            TelegramSession.is_active == True,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return None
    return SessionResponse.model_validate(session)


@router.delete("/session")
async def delete_session(
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete Telegram session (logout)."""
    await telegram_client.delete_session(db, ctx.user.id)
    return {"message": "Session deleted successfully"}
