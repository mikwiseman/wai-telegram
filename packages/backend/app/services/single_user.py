"""Owner and explicitly invited account invariants."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.models.user import User


class OwnerConfigurationError(RuntimeError):
    """Raised when database account state does not match OWNER_USER_ID."""


async def validate_active_owner(db: AsyncSession, owner_user_id: UUID) -> None:
    users = (
        (await db.execute(select(User).where(User.is_active.is_(True)))).scalars().all()
    )
    if not any(user.id == owner_user_id for user in users) or any(
        user.id != owner_user_id
        and (
            user.invited_by_id != owner_user_id
            or user.expected_telegram_user_id is None
        )
        for user in users
    ):
        raise OwnerConfigurationError(
            "OWNER_USER_ID must be active; other active accounts must be explicitly invited by the owner"
        )


async def is_user_active(db: AsyncSession, user_id: UUID) -> bool:
    return (
        await db.execute(
            select(User.id).where(
                User.id == user_id,
                User.is_active.is_(True),
            )
        )
    ).scalar_one_or_none() is not None


async def lock_active_user(db: AsyncSession, user_id: UUID) -> bool:
    """Hold a shared row lock while a short write transaction serves the user.

    The lock prevents the single-user cutover from committing deactivation in
    the middle of a listener transaction. PostgreSQL releases it when the
    surrounding transaction commits or rolls back.
    """
    return (
        await db.execute(
            select(User.id)
            .where(
                User.id == user_id,
                User.is_active.is_(True),
            )
            .with_for_update(read=True)
        )
    ).scalar_one_or_none() is not None


async def is_user_active_in_database(user_id: UUID) -> bool:
    """Check active state in a fresh transaction for task entry points."""
    async with get_db_context() as db:
        return await is_user_active(db, user_id)
