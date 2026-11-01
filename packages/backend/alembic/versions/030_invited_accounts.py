"""Invited accounts and explicit chat indexing selection.

Revision ID: 030_invited_accounts
Revises: 029_file_draft_mime
"""

from alembic import op
import sqlalchemy as sa

revision = "030_invited_accounts"
down_revision = "029_file_draft_mime"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("invited_by_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_users_invited_by", "users", "users", ["invited_by_id"], ["id"]
    )
    op.add_column(
        "users", sa.Column("expected_telegram_user_id", sa.BigInteger(), nullable=True)
    )
    op.create_unique_constraint(
        "uq_users_expected_telegram", "users", ["expected_telegram_user_id"]
    )
    op.add_column("users", sa.Column("invite_token_hash", sa.String(64), nullable=True))
    op.create_unique_constraint("uq_users_invite_token", "users", ["invite_token_hash"])
    op.add_column(
        "users",
        sa.Column("invite_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_index("uq_users_single_active", table_name="users")
    op.add_column(
        "telegram_chats",
        sa.Column(
            "indexing_enabled", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
    )
    op.add_column(
        "telegram_chats",
        sa.Column("history_synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "telegram_chats", sa.Column("history_cursor", sa.BigInteger(), nullable=True)
    )
    op.add_column(
        "user_settings",
        sa.Column(
            "embedding_backfill_enabled",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    # Preserve the existing archive and listener behaviour; new invited chats opt in.
    op.execute("UPDATE telegram_chats SET indexing_enabled = true")


def downgrade():
    # Never silently deactivate a connected guest to make rollback fit.
    connection = op.get_bind()
    if (
        connection.execute(
            sa.text("SELECT count(*) FROM users WHERE is_active")
        ).scalar_one()
        > 1
    ):
        raise RuntimeError(
            "Deactivate invited accounts before reverting to single-owner mode"
        )
    op.create_index(
        "uq_users_single_active",
        "users",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.drop_column("telegram_chats", "history_synced_at")
    op.drop_column("telegram_chats", "history_cursor")
    op.drop_column("user_settings", "embedding_backfill_enabled")
    op.drop_column("telegram_chats", "indexing_enabled")
    op.drop_constraint("uq_users_invite_token", "users", type_="unique")
    op.drop_constraint("uq_users_expected_telegram", "users", type_="unique")
    op.drop_constraint("fk_users_invited_by", "users", type_="foreignkey")
    for name in (
        "invite_expires_at",
        "invite_token_hash",
        "expected_telegram_user_id",
        "invited_by_id",
    ):
        op.drop_column("users", name)
