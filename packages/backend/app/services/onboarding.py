import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.settings import UserSettings
from app.models.user import User


def invitation_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_invitation(
    db: AsyncSession, owner_id: UUID, telegram_user_id: int
) -> tuple[User, str]:
    owner = await db.get(User, owner_id)
    if not owner or not owner.is_active or owner.invited_by_id is not None:
        raise ValueError("Only the active owner can invite an account")
    if telegram_user_id <= 0:
        raise ValueError("A Telegram user ID is required")
    user = (
        await db.execute(
            select(User).where(
                User.expected_telegram_user_id == telegram_user_id,
            )
        )
    ).scalar_one_or_none()
    if user is None:
        user = User(
            email=f"telegram-{telegram_user_id}@users.invalid",
            password_hash=hash_password(secrets.token_urlsafe(48)),
            invited_by_id=owner_id,
            expected_telegram_user_id=telegram_user_id,
            is_active=False,
        )
        db.add(user)
        await db.flush()
        db.add(
            UserSettings(
                user_id=user.id, digest_enabled=False, realtime_sync_enabled=True
            )
        )
    elif user.invited_by_id != owner_id:
        raise ValueError("This account was not invited by this owner")
    token = secrets.token_urlsafe(32)
    user.invite_token_hash = invitation_hash(token)
    user.invite_expires_at = datetime.now(UTC) + timedelta(days=7)
    await db.flush()
    return user, token


async def accept_invitation(db: AsyncSession, token: str) -> UUID | None:
    active_owner_ids = select(User.id).where(
        User.is_active.is_(True), User.invited_by_id.is_(None)
    )
    # Consume atomically: a link cannot create two independently replayable sessions.
    return (
        await db.execute(
            update(User)
            .where(
                User.invite_token_hash == invitation_hash(token),
                User.invite_expires_at > datetime.now(UTC),
                User.invited_by_id.in_(active_owner_ids),
                User.expected_telegram_user_id.is_not(None),
            )
            .values(
                invite_token_hash=None,
                invite_expires_at=None,
                is_active=True,
                deactivated_at=None,
                deactivation_reason=None,
            )
            .returning(User.id)
        )
    ).scalar_one_or_none()
