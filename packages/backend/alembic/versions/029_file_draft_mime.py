"""Allow document MIME types in the existing owner-scoped draft storage."""

from alembic import op
import sqlalchemy as sa

revision = "029_file_draft_mime"
down_revision = "028_post_drafts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "post_drafts", "mime_type", existing_type=sa.String(32), type_=sa.String(255)
    )


def downgrade() -> None:
    # Preserve attachments added after upgrade; longer types remain valid generic files.
    op.execute(
        "UPDATE post_drafts SET mime_type = 'application/octet-stream' WHERE length(mime_type) > 32"
    )
    op.alter_column(
        "post_drafts", "mime_type", existing_type=sa.String(255), type_=sa.String(32)
    )
