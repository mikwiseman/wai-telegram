from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    invited_by_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    expected_telegram_user_id: Mapped[int | None] = mapped_column(
        BigInteger, unique=True
    )
    invite_token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    invite_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # When set, discover every chat automatically and load history from this date.
    auto_index_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    deactivated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deactivation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    # Relationships
    telegram_sessions: Mapped[list["TelegramSession"]] = relationship(
        "TelegramSession", back_populates="user", cascade="all, delete-orphan"
    )
    chats: Mapped[list["TelegramChat"]] = relationship(
        "TelegramChat", back_populates="user", cascade="all, delete-orphan"
    )
    sync_jobs: Mapped[list["SyncJob"]] = relationship(
        "SyncJob", back_populates="user", cascade="all, delete-orphan"
    )
    digests: Mapped[list["DailyDigest"]] = relationship(
        "DailyDigest", back_populates="user", cascade="all, delete-orphan"
    )
    settings: Mapped["UserSettings | None"] = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey", back_populates="user", cascade="all, delete-orphan"
    )


# Import for type hints
from app.models.api_key import ApiKey  # noqa: E402
from app.models.chat import TelegramChat  # noqa: E402
from app.models.digest import DailyDigest  # noqa: E402
from app.models.session import TelegramSession  # noqa: E402
from app.models.settings import UserSettings  # noqa: E402
from app.models.sync_job import SyncJob  # noqa: E402
