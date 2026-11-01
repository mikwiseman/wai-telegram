"""Automatic chat discovery with a bounded history period."""

from alembic import op
import sqlalchemy as sa

revision = "031_automatic_indexing"
down_revision = "030_invited_accounts"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("auto_index_since", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade():
    op.drop_column("users", "auto_index_since")
