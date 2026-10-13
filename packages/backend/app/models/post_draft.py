from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PostDraft(Base):
    """An immutable post kept in Wai, not in Telegram's text-only draft storage."""

    __tablename__ = "post_drafts"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'publishing', 'published', 'unknown')"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("telegram_chats.id", ondelete="CASCADE")
    )
    caption: Mapped[str] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(String(200))
    mime_type: Mapped[str] = mapped_column(String(32))
    media: Mapped[bytes] = mapped_column(LargeBinary)
    sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="draft")
    telegram_message_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
