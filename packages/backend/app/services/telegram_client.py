import json
import logging
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telethon import TelegramClient
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    SessionExpiredError,
    SessionPasswordNeededError,
    SessionRevokedError,
    UnauthorizedError,
    UserDeactivatedBanError,
    UserDeactivatedError,
)
from telethon.sessions import StringSession

from app.core.config import get_settings
from app.core.database import get_db_context
from app.core.security import decrypt_session, encrypt_session
from app.models.session import TelegramSession
from app.models.settings import UserSettings

logger = logging.getLogger(__name__)
settings = get_settings()
CLIENT_SESSION_ATTR = "_wai_session_id"
SESSION_EXPIRED_MESSAGE = "Telegram session expired. Reconnect Telegram and try again."

AUTH_SESSION_ERRORS = (
    UnauthorizedError,
    AuthKeyUnregisteredError,
    SessionRevokedError,
    SessionExpiredError,
    UserDeactivatedError,
    UserDeactivatedBanError,
)


class TelegramSessionUnauthorizedError(ValueError):
    """Raised when a persisted Telethon session is no longer authorized."""


class NoActiveTelegramSessionError(TelegramSessionUnauthorizedError):
    """Raised when there is no active Telegram session row for a user."""


def is_session_authorization_error(exc: BaseException) -> bool:
    return isinstance(exc, AUTH_SESSION_ERRORS)


def get_client_session_id(client: TelegramClient) -> UUID | None:
    session_id = getattr(client, CLIENT_SESSION_ATTR, None)
    return session_id if isinstance(session_id, UUID) else None


async def invalidate_unauthorized_session(
    user_id: UUID,
    reason: str,
    *,
    session_id: UUID | None = None,
) -> bool:
    """Disable the invalid session and remediate listener state safely."""
    restart_listener = False
    invalidated_session_id: UUID | None = None

    async with get_db_context() as db:
        active_sessions = (
            (
                await db.execute(
                    select(TelegramSession).where(
                        TelegramSession.user_id == user_id,
                        TelegramSession.is_active == True,
                    )
                )
            )
            .scalars()
            .all()
        )

        if session_id is not None:
            target_session = next(
                (session for session in active_sessions if session.id == session_id),
                None,
            )
        elif len(active_sessions) == 1:
            target_session = active_sessions[0]
        else:
            target_session = None

        if target_session is None:
            logger.warning(
                "Skipped Telegram session invalidation for user %s: no matching active session (session_id=%s)",
                user_id,
                session_id,
            )
            return False

        target_session.is_active = False
        invalidated_session_id = target_session.id

        remaining_active_sessions = (
            (
                await db.execute(
                    select(TelegramSession.id).where(
                        TelegramSession.user_id == user_id,
                        TelegramSession.is_active == True,
                    )
                )
            )
            .scalars()
            .all()
        )

        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == user_id)
        )
        user_settings = result.scalar_one_or_none()

        if remaining_active_sessions:
            restart_listener = bool(
                user_settings is not None and user_settings.realtime_sync_enabled
            )
        else:
            if user_settings is None:
                user_settings = UserSettings(
                    user_id=user_id,
                    realtime_sync_enabled=False,
                )
                db.add(user_settings)
            else:
                user_settings.realtime_sync_enabled = False

    redis = aioredis.from_url(settings.redis_url)
    try:
        await redis.delete(f"listener:active:{user_id}")
        await redis.publish(
            "listener:cmd:global",
            json.dumps({"command": "stop_user", "user_id": str(user_id)}),
        )
        if restart_listener:
            await redis.publish(
                "listener:cmd:global",
                json.dumps({"command": "start_user", "user_id": str(user_id)}),
            )
    finally:
        await redis.aclose()

    logger.warning(
        "Disabled Telegram session %s for user %s: %s (listener_restart=%s)",
        invalidated_session_id,
        user_id,
        reason,
        restart_listener,
    )
    return True


async def invalidate_client_authorization(
    client: TelegramClient,
    user_id: UUID,
    reason: Exception | str,
) -> None:
    try:
        await client.disconnect()
    except Exception:
        logger.debug("Failed to disconnect unauthorized Telegram client", exc_info=True)

    await invalidate_unauthorized_session(
        user_id,
        str(reason),
        session_id=get_client_session_id(client),
    )


async def _ensure_client_authorized(
    client: TelegramClient,
    user_id: UUID,
    session_id: UUID,
) -> None:
    try:
        me = await client.get_me()
    except AUTH_SESSION_ERRORS as exc:
        await invalidate_client_authorization(client, user_id, exc)
        raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from exc

    if me is None:
        await invalidate_client_authorization(
            client,
            user_id,
            "Telegram session is no longer authorized",
        )
        raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE)


async def get_client(
    user_id: UUID,
    db: AsyncSession,
    *,
    receive_updates: bool = False,
) -> TelegramClient:
    """Create a fresh Telegram client for a user.

    Clients are intentionally not cached globally to avoid cross-event-loop reuse.
    """
    from app.models.user import User

    result = await db.execute(
        select(TelegramSession)
        .join(User, User.id == TelegramSession.user_id)
        .where(
            TelegramSession.user_id == user_id,
            TelegramSession.is_active == True,
            User.is_active.is_(True),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise NoActiveTelegramSessionError("No active Telegram session found")

    session_string = decrypt_session(session.session_string)
    client = TelegramClient(
        StringSession(session_string),
        settings.telegram_api_id,
        settings.telegram_api_hash,
        device_model=settings.telegram_device_model,
        system_version=settings.telegram_system_version,
        app_version=settings.telegram_app_version,
        flood_sleep_threshold=settings.telegram_flood_sleep_threshold,
        receive_updates=receive_updates,
    )
    await client.connect()
    setattr(client, CLIENT_SESSION_ATTR, session.id)
    await _ensure_client_authorized(client, user_id, session.id)
    return client


async def create_auth_client() -> TelegramClient:
    """Create a temporary client for authentication."""
    client = TelegramClient(
        StringSession(),
        settings.telegram_api_id,
        settings.telegram_api_hash,
        device_model=settings.telegram_device_model,
        system_version=settings.telegram_system_version,
        app_version=settings.telegram_app_version,
        flood_sleep_threshold=settings.telegram_flood_sleep_threshold,
        receive_updates=False,
    )
    await client.connect()
    return client


def _get_code_type_name(sent_code) -> str:
    """Extract human-readable code delivery type from SentCode result."""
    code_type = sent_code.type
    type_name = type(code_type).__name__
    type_map = {
        "SentCodeTypeApp": "app",
        "SentCodeTypeSms": "sms",
        "SentCodeTypeCall": "call",
        "SentCodeTypeFlashCall": "flash_call",
        "SentCodeTypeMissedCall": "missed_call",
        "SentCodeTypeEmailCode": "email",
        "SentCodeTypeFragmentSms": "fragment_sms",
    }
    return type_map.get(type_name, "unknown")


async def request_code(phone_number: str) -> tuple[TelegramClient, str, str]:
    """Request verification code for phone number.

    Returns (client, phone_code_hash, code_type).
    """
    client = await create_auth_client()
    try:
        result = await client.send_code_request(phone_number)
        code_type = _get_code_type_name(result)
        logger.info(
            f"Telegram login code requested: type={code_type}, "
            f"timeout={getattr(result, 'timeout', None)}s"
        )
        return client, result.phone_code_hash, code_type
    except FloodWaitError as e:
        wait_time = int(e.seconds * settings.flood_wait_multiplier)
        logger.warning(f"Telegram login rate limited: wait {wait_time}s")
        await client.disconnect()
        raise ValueError(f"Too many attempts. Please wait {wait_time} seconds.")
    except Exception:
        await client.disconnect()
        raise


async def verify_code(
    client: TelegramClient,
    phone_number: str,
    phone_code_hash: str,
    code: str,
    password: str | None = None,
) -> tuple[str, int]:
    """Verify code and complete authentication. Returns (session_string, telegram_user_id)."""
    try:
        await client.sign_in(
            phone=phone_number,
            code=code,
            phone_code_hash=phone_code_hash,
        )
    except SessionPasswordNeededError:
        if not password:
            raise ValueError("Two-factor authentication is enabled. Password required.")
        await client.sign_in(password=password)

    me = await client.get_me()
    session_string = client.session.save()
    return session_string, me.id


async def save_session(
    db: AsyncSession,
    user_id: UUID,
    phone_number: str,
    session_string: str,
    telegram_user_id: int,
) -> TelegramSession:
    """Save encrypted Telegram session to database."""
    # Ensure stale in-memory client for this user is not reused.
    await disconnect_client(user_id)

    # Deactivate any existing sessions for this user
    result = await db.execute(
        select(TelegramSession).where(
            TelegramSession.user_id == user_id, TelegramSession.is_active == True
        )
    )
    existing = result.scalars().all()
    for session in existing:
        session.is_active = False

    # Create new session
    encrypted_session = encrypt_session(session_string)
    new_session = TelegramSession(
        user_id=user_id,
        phone_number=phone_number,
        session_string=encrypted_session,
        telegram_user_id=telegram_user_id,
        is_active=True,
    )
    db.add(new_session)
    await db.flush()
    return new_session


async def start_listener(user_id: UUID) -> None:
    """Best-effort wakeup; the existing health check retries missed notifications."""
    redis = aioredis.from_url(
        settings.redis_url, socket_connect_timeout=1, socket_timeout=1
    )
    try:
        await redis.publish(
            "listener:cmd:global",
            json.dumps({"command": "start_user", "user_id": str(user_id)}),
        )
    except Exception:
        logger.warning("Listener wakeup deferred to health check")
    finally:
        await redis.aclose()


async def disconnect_client(user_id: UUID) -> None:
    """Disconnect and remove a cached client if present.

    No-op because sync clients are now created per operation.
    """
    _ = user_id


async def delete_session(db: AsyncSession, user_id: UUID) -> None:
    """Delete user's Telegram session."""
    await disconnect_client(user_id)
    result = await db.execute(
        select(TelegramSession).where(TelegramSession.user_id == user_id)
    )
    sessions = result.scalars().all()
    for session in sessions:
        await db.delete(session)
