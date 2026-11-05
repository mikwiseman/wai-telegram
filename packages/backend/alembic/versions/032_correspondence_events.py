"""Transactional source events for the owner's correspondence workflow."""

import sqlalchemy as sa
from alembic import op

revision = "032_correspondence_events"
down_revision = "031_automatic_indexing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "correspondence_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "chat_id",
            sa.Uuid(),
            sa.ForeignKey("telegram_chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("destination_fingerprint", sa.String(64), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        "ix_correspondence_events_user_id", "correspondence_events", ["user_id"]
    )
    op.create_index(
        "ix_correspondence_events_next_attempt_at",
        "correspondence_events",
        ["next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_table("correspondence_events")
