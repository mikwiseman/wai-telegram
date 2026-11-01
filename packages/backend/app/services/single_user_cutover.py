"""Evidence-gated transition from legacy multi-user state to one active owner."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.models.chat import TelegramChat
from app.models.digital_agent import DigitalAgent
from app.models.message import TelegramMessage
from app.models.session import TelegramSession
from app.models.settings import UserSettings
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User


class OwnerEvidenceError(RuntimeError):
    """Raised before mutation when the three owner signals do not agree."""


@dataclass(frozen=True)
class OwnerEvidence:
    owner_user_id: UUID
    active_session_users: tuple[UUID, ...]
    recent_api_key_users: tuple[UUID, ...]
    top_message_volume_users: tuple[UUID, ...]
    owner_chat_count: int
    owner_message_count: int
    total_users: int

    def public_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "owner_user_id",
            "active_session_users",
            "recent_api_key_users",
            "top_message_volume_users",
        ):
            value = payload[key]
            if isinstance(value, tuple):
                payload[key] = [str(item) for item in value]
            else:
                payload[key] = str(value)
        return payload


@dataclass(frozen=True)
class CutoverResult:
    owner_user_id: UUID
    users_deactivated: int
    api_keys_disabled: int
    telegram_sessions_wiped: int
    settings_disabled: int
    sync_jobs_cancelled: int
    agents_paused: int
    archived_chat_count: int
    archived_message_count: int
    cancelled_job_ids: tuple[UUID, ...]
    deactivated_user_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class ArchivedSessionCredential:
    session_id: UUID
    encrypted_session_string: str


async def collect_owner_evidence(
    db: AsyncSession,
    *,
    expected_owner_user_id: UUID,
    recent_key_window: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> OwnerEvidence:
    """Require one identical owner from active session, recent key, and volume."""
    current_time = now or datetime.now(UTC)
    cutoff = current_time - recent_key_window

    session_users = tuple(
        sorted(
            set(
                (
                    await db.execute(
                        select(TelegramSession.user_id).where(
                            TelegramSession.is_active.is_(True)
                        )
                    )
                ).scalars()
            ),
            key=str,
        )
    )
    key_users = tuple(
        sorted(
            set(
                (
                    await db.execute(
                        select(ApiKey.user_id).where(
                            ApiKey.is_active.is_(True),
                            ApiKey.last_used_at.isnot(None),
                            ApiKey.last_used_at >= cutoff,
                        )
                    )
                ).scalars()
            ),
            key=str,
        )
    )

    volume_rows = (
        await db.execute(
            select(
                TelegramChat.user_id,
                func.count(func.distinct(TelegramChat.id)),
                func.count(TelegramMessage.id),
            )
            .outerjoin(TelegramMessage, TelegramMessage.chat_id == TelegramChat.id)
            .group_by(TelegramChat.user_id)
        )
    ).all()
    max_messages = max((int(row[2]) for row in volume_rows), default=0)
    top_volume_users = tuple(
        sorted(
            (row[0] for row in volume_rows if int(row[2]) == max_messages),
            key=str,
        )
    )

    agreed = {session_users, key_users, top_volume_users}
    expected = (expected_owner_user_id,)
    if agreed != {expected} or max_messages <= 0:
        raise OwnerEvidenceError(
            "Owner evidence is ambiguous: active session, recent API key, and "
            "largest message volume must identify only OWNER_USER_ID"
        )

    owner_row = next(row for row in volume_rows if row[0] == expected_owner_user_id)
    total_users = (
        await db.execute(select(func.count()).select_from(User))
    ).scalar_one()
    return OwnerEvidence(
        owner_user_id=expected_owner_user_id,
        active_session_users=session_users,
        recent_api_key_users=key_users,
        top_message_volume_users=top_volume_users,
        owner_chat_count=int(owner_row[1]),
        owner_message_count=int(owner_row[2]),
        total_users=int(total_users),
    )


async def apply_single_user_cutover(
    db: AsyncSession,
    evidence: OwnerEvidence,
    *,
    reason: str = "single-user production cutover",
    now: datetime | None = None,
) -> CutoverResult:
    """Apply all database-side revocations in the caller's transaction."""
    invited = (
        await db.execute(
            select(User.id).where(User.invited_by_id.is_not(None)).limit(1)
        )
    ).scalar_one_or_none()
    if invited is not None:
        raise RuntimeError(
            "Single-owner cutover is disabled after accounts have been invited"
        )
    current_time = now or datetime.now(UTC)
    owner_id = evidence.owner_user_id

    deactivated_user_ids = tuple(
        (await db.execute(select(User.id).where(User.id != owner_id))).scalars()
    )
    cancelled_job_ids = tuple(
        (
            await db.execute(
                select(SyncJob.id).where(
                    SyncJob.user_id.in_(deactivated_user_ids),
                    SyncJob.status.in_((SyncStatus.PENDING, SyncStatus.IN_PROGRESS)),
                )
            )
        ).scalars()
    )

    archived_chat_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramChat)
            .where(TelegramChat.user_id.in_(deactivated_user_ids))
        )
    ).scalar_one()
    archived_message_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramMessage)
            .join(TelegramChat, TelegramMessage.chat_id == TelegramChat.id)
            .where(TelegramChat.user_id.in_(deactivated_user_ids))
        )
    ).scalar_one()

    users_result = await db.execute(
        update(User)
        .where(User.id.in_(deactivated_user_ids))
        .values(
            is_active=False,
            deactivated_at=current_time,
            deactivation_reason=reason,
        )
    )
    keys_result = await db.execute(
        update(ApiKey)
        .where(ApiKey.user_id.in_(deactivated_user_ids), ApiKey.is_active.is_(True))
        .values(is_active=False)
    )
    sessions_result = await db.execute(
        update(TelegramSession)
        .where(TelegramSession.user_id.in_(deactivated_user_ids))
        .values(is_active=False, session_string="")
    )
    settings_result = await db.execute(
        update(UserSettings)
        .where(UserSettings.user_id.in_(deactivated_user_ids))
        .values(
            realtime_sync_enabled=False,
            digest_enabled=False,
            digest_telegram_enabled=False,
        )
    )
    jobs_result = await db.execute(
        update(SyncJob)
        .where(SyncJob.id.in_(cancelled_job_ids))
        .values(
            status=SyncStatus.CANCELLED,
            completed_at=current_time,
            error_message="Cancelled by single-user production cutover",
        )
    )
    agents_result = await db.execute(
        update(DigitalAgent)
        .where(
            DigitalAgent.user_id.in_(deactivated_user_ids),
            DigitalAgent.status == "active",
        )
        .values(status="paused", next_run_at=None)
    )
    await db.execute(
        update(User)
        .where(User.id == owner_id)
        .values(
            is_active=True,
            deactivated_at=None,
            deactivation_reason=None,
        )
    )
    await db.flush()

    return CutoverResult(
        owner_user_id=owner_id,
        users_deactivated=users_result.rowcount,
        api_keys_disabled=keys_result.rowcount,
        telegram_sessions_wiped=sessions_result.rowcount,
        settings_disabled=settings_result.rowcount,
        sync_jobs_cancelled=jobs_result.rowcount,
        agents_paused=agents_result.rowcount,
        archived_chat_count=int(archived_chat_count),
        archived_message_count=int(archived_message_count),
        cancelled_job_ids=cancelled_job_ids,
        deactivated_user_ids=deactivated_user_ids,
    )


async def collect_archived_session_credentials(
    db: AsyncSession,
    owner_user_id: UUID,
) -> tuple[ArchivedSessionCredential, ...]:
    invited = (
        await db.execute(
            select(User.id).where(User.invited_by_id.is_not(None)).limit(1)
        )
    ).scalar_one_or_none()
    if invited is not None:
        raise RuntimeError(
            "Single-owner cutover is disabled after accounts have been invited"
        )
    rows = (
        await db.execute(
            select(TelegramSession.id, TelegramSession.session_string).where(
                TelegramSession.user_id != owner_user_id,
                TelegramSession.session_string != "",
            )
        )
    ).all()
    return tuple(
        ArchivedSessionCredential(
            session_id=session_id,
            encrypted_session_string=session_string,
        )
        for session_id, session_string in rows
    )


async def revoke_archived_telegram_sessions(
    credentials: tuple[ArchivedSessionCredential, ...],
) -> int:
    """Revoke Telegram authorizations before their encrypted strings are wiped."""
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    from app.core.config import get_settings
    from app.core.security import decrypt_session
    from app.services.telegram_client import AUTH_SESSION_ERRORS

    settings = get_settings()
    revoked = 0
    for credential in credentials:
        session_string = decrypt_session(credential.encrypted_session_string)
        client = TelegramClient(
            StringSession(session_string),
            settings.telegram_api_id,
            settings.telegram_api_hash,
            receive_updates=False,
        )
        try:
            await client.connect()
            if await client.is_user_authorized():
                await client.log_out()
                revoked += 1
        except AUTH_SESSION_ERRORS:
            # Telegram already considers this authorization revoked/expired.
            pass
        finally:
            if client.is_connected():
                await client.disconnect()
    return revoked


async def purge_deactivated_runtime_state(
    redis_url: str,
    result: CutoverResult,
) -> int:
    """Remove only per-user locks/caches and cancelled-job progress keys."""
    client = aioredis.from_url(redis_url)
    keys: set[bytes | str] = set()
    try:
        for user_id in result.deactivated_user_ids:
            for pattern in (
                f"listener:active:{user_id}",
                f"listener:cmd:{user_id}",
                f"sync:{user_id}:*",
                f"agent:{user_id}:*",
                f"conversation:{user_id}:*",
            ):
                async for key in client.scan_iter(match=pattern):
                    keys.add(key)
            await client.publish(
                "listener:cmd:global",
                f'{{"command":"stop_user","user_id":"{user_id}"}}',
            )
        for job_id in result.cancelled_job_ids:
            for pattern in (f"sync:{job_id}:*", f"bulk:{job_id}:*"):
                async for key in client.scan_iter(match=pattern):
                    keys.add(key)
        if keys:
            return int(await client.delete(*keys))
        return 0
    finally:
        await client.aclose()
