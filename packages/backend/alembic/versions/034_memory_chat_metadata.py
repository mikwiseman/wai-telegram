"""Reconcile memory when chat metadata or its selected scope changes."""

from alembic import op

revision = "034_memory_chat_metadata"
down_revision = "033_memory_batches"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX memory_days_pending")
    op.execute(
        "CREATE INDEX memory_days_pending ON memory_days(changed_at,chat_id,day DESC) WHERE revision<>prepared_revision"
    )
    op.execute("DROP INDEX memory_passages_pending")
    op.execute(
        "CREATE INDEX memory_passages_pending ON memory_passages(day DESC,id) WHERE embedding IS NULL AND batch_id IS NULL AND failures<3"
    )
    op.execute("""
    CREATE FUNCTION mark_memory_chat() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      INSERT INTO memory_days(chat_id,day)
      SELECT NEW.id,(m.sent_at AT TIME ZONE 'UTC')::date
      FROM telegram_messages m WHERE m.chat_id=NEW.id
      GROUP BY (m.sent_at AT TIME ZONE 'UTC')::date
      ON CONFLICT(chat_id,day) DO UPDATE SET
        revision=memory_days.revision+1,changed_at=now();
      RETURN NULL;
    END $$;
    """)
    op.execute("""
    CREATE TRIGGER memory_chat_edit AFTER UPDATE OF title,user_id,indexing_enabled
    ON telegram_chats FOR EACH ROW WHEN (NEW.indexing_enabled AND
      (OLD.title,OLD.user_id,OLD.indexing_enabled) IS DISTINCT FROM
      (NEW.title,NEW.user_id,NEW.indexing_enabled))
    EXECUTE FUNCTION mark_memory_chat();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER memory_chat_edit ON telegram_chats")
    op.execute("DROP FUNCTION mark_memory_chat()")
    op.execute("DROP INDEX memory_days_pending")
    op.execute(
        "CREATE INDEX memory_days_pending ON memory_days(changed_at) WHERE revision<>prepared_revision"
    )
    op.execute("DROP INDEX memory_passages_pending")
    op.execute(
        "CREATE INDEX memory_passages_pending ON memory_passages(id) WHERE embedding IS NULL AND batch_id IS NULL AND failures<3"
    )
