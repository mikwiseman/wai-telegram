"""Store complete, unpublished posts in Wai.

Revision ID: 028_post_drafts
Revises: 027_rename_find_files_tool
"""

from alembic import op
import sqlalchemy as sa

revision = "028_post_drafts"
down_revision = "027_rename_find_files_tool"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "post_drafts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_id",
            sa.Uuid(),
            sa.ForeignKey("telegram_chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("caption", sa.Text(), nullable=False),
        sa.Column("file_name", sa.String(200), nullable=False),
        sa.Column("mime_type", sa.String(32), nullable=False),
        sa.Column("media", sa.LargeBinary(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("telegram_message_id", sa.BigInteger()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('draft', 'publishing', 'published', 'unknown')"),
    )
    op.create_index("ix_post_drafts_user_id", "post_drafts", ["user_id"])


def downgrade() -> None:
    op.drop_table("post_drafts")
