"""Rebuildable source passages and durable OpenAI Batch receipts."""

from alembic import op

revision = "033_memory_batches"
down_revision = "032_correspondence_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    ddl = """
      CREATE TABLE memory_days (
        chat_id uuid NOT NULL REFERENCES telegram_chats(id) ON DELETE CASCADE,
        day date NOT NULL,
        revision bigint NOT NULL DEFAULT 1,
        prepared_revision bigint NOT NULL DEFAULT 0,
        changed_at timestamptz NOT NULL DEFAULT now(),
        prepared_at timestamptz,
        source_messages integer NOT NULL DEFAULT 0,
        PRIMARY KEY(chat_id,day)
      );
      CREATE INDEX memory_days_pending ON memory_days(changed_at)
        WHERE revision <> prepared_revision;
      CREATE TABLE memory_batches (
        id uuid PRIMARY KEY,
        provider_id text UNIQUE,
        input_file_id text,
        output_file_id text,
        status text NOT NULL,
        passage_ids text[] NOT NULL,
        input_tokens bigint NOT NULL,
        actual_tokens bigint NOT NULL DEFAULT 0,
        applied integer NOT NULL DEFAULT 0,
        failed integer NOT NULL DEFAULT 0,
        error_code text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      );
      CREATE TABLE memory_passages (
        id text PRIMARY KEY,
        user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        chat_id uuid NOT NULL,
        day date NOT NULL,
        revision bigint NOT NULL,
        message_ids uuid[] NOT NULL,
        text text NOT NULL,
        tokens integer NOT NULL,
        model text NOT NULL DEFAULT 'text-embedding-3-large',
        dimensions integer NOT NULL DEFAULT 3072,
        embedding halfvec(3072),
        batch_id uuid REFERENCES memory_batches(id),
        failures integer NOT NULL DEFAULT 0,
        error_code text,
        embedded_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        FOREIGN KEY(chat_id,day) REFERENCES memory_days(chat_id,day) ON DELETE CASCADE,
        CHECK(model = 'text-embedding-3-large' AND dimensions = 3072),
        CHECK(tokens > 0 AND tokens <= 1400)
      );
      CREATE INDEX memory_passages_day ON memory_passages(chat_id,day);
      CREATE INDEX memory_passages_pending ON memory_passages(id)
        WHERE embedding IS NULL AND batch_id IS NULL AND failures < 3;
      CREATE INDEX memory_passages_embedding ON memory_passages
        USING hnsw (embedding halfvec_cosine_ops) WITH (m=16,ef_construction=64);
      CREATE FUNCTION mark_memory_day() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN
        IF TG_OP <> 'INSERT' AND EXISTS(SELECT 1 FROM telegram_chats WHERE id=OLD.chat_id) THEN
          INSERT INTO memory_days(chat_id,day) VALUES
            (OLD.chat_id,(OLD.sent_at AT TIME ZONE 'UTC')::date)
          ON CONFLICT(chat_id,day) DO UPDATE SET
            revision=memory_days.revision+1,changed_at=now();
        END IF;
        IF TG_OP <> 'DELETE' THEN
          INSERT INTO memory_days(chat_id,day) VALUES
            (NEW.chat_id,(NEW.sent_at AT TIME ZONE 'UTC')::date)
          ON CONFLICT(chat_id,day) DO UPDATE SET
            revision=memory_days.revision+1,changed_at=now();
        END IF;
        RETURN NULL;
      END $$;
      CREATE TRIGGER memory_message_insert AFTER INSERT ON telegram_messages
        FOR EACH ROW EXECUTE FUNCTION mark_memory_day();
      CREATE TRIGGER memory_message_delete AFTER DELETE ON telegram_messages
        FOR EACH ROW EXECUTE FUNCTION mark_memory_day();
      CREATE TRIGGER memory_message_edit AFTER UPDATE OF
        text,content_text,content_summary,sender_name,media_file_name,thread_id,sent_at,deleted_at
        ON telegram_messages FOR EACH ROW WHEN (
          (OLD.text,OLD.content_text,OLD.content_summary,OLD.sender_name,OLD.media_file_name,
           OLD.thread_id,OLD.sent_at,OLD.deleted_at) IS DISTINCT FROM
          (NEW.text,NEW.content_text,NEW.content_summary,NEW.sender_name,NEW.media_file_name,
           NEW.thread_id,NEW.sent_at,NEW.deleted_at)
        ) EXECUTE FUNCTION mark_memory_day();
    """
    tables, function = ddl.split("CREATE FUNCTION", 1)
    function, triggers = function.split("END $$;", 1)
    for statement in tables.split(";"):
        if statement.strip():
            op.execute(statement)
    op.execute("CREATE FUNCTION" + function + "END $$;")
    for statement in triggers.split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade() -> None:
    ddl = """
      DROP TRIGGER memory_message_insert ON telegram_messages;
      DROP TRIGGER memory_message_delete ON telegram_messages;
      DROP TRIGGER memory_message_edit ON telegram_messages;
      DROP FUNCTION mark_memory_day();
      DROP TABLE memory_passages;
      DROP TABLE memory_batches;
      DROP TABLE memory_days;
    """
    for statement in ddl.split(";"):
        if statement.strip():
            op.execute(statement)
