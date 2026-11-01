"""PostgreSQL-only contract checks for migrations and hybrid retrieval."""

import importlib.util
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.chat import ChatType, TelegramChat
from app.models.media import MediaObject
from app.models.message import MessageContentChunk, MessageRevision, TelegramMessage
from app.models.metadata import MetadataReconciliationCheckpoint
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User
from app.schemas.search import SearchRequest
from app.services.search_service import semantic_search
from app.services.metadata_reconciliation import _save_metadata_batch


def _load_channel_identity_migration():
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "026_canonical_channel_peer_ids.py"
    )
    spec = importlib.util.spec_from_file_location(
        "canonical_channel_peer_ids_migration",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_CONTRACT_DATABASE_URL"),
    reason="POSTGRES_CONTRACT_DATABASE_URL is not configured",
)
@pytest.mark.parametrize("chat_type", [ChatType.SUPERGROUP, ChatType.CHANNEL])
async def test_channel_identity_migration_merges_legacy_and_canonical_rows(
    chat_type: ChatType,
):
    engine = create_async_engine(os.environ["POSTGRES_CONTRACT_DATABASE_URL"])
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    user_id = uuid4()
    raw_channel_id = 7_654_321
    canonical_channel_id = -(1_000_000_000_000 + raw_channel_id)
    unpaired_raw_id = raw_channel_id + 1
    unpaired_canonical_id = -(1_000_000_000_000 + unpaired_raw_id)
    now = datetime.now(UTC)

    try:
        async with session_factory() as db:
            transaction = await db.begin()
            try:
                user = User(
                    id=user_id,
                    email=f"channel-identity-{user_id}@example.com",
                    password_hash="not-used",
                    is_active=False,
                )
                legacy = TelegramChat(
                    user_id=user_id,
                    telegram_chat_id=raw_channel_id,
                    chat_type=chat_type,
                    title="Legacy title",
                    last_activity_at=now - timedelta(days=1),
                    last_message_text="legacy preview",
                    unread_count=99,
                )
                canonical = TelegramChat(
                    user_id=user_id,
                    telegram_chat_id=canonical_channel_id,
                    chat_type=chat_type,
                    title="Current title",
                    last_activity_at=now,
                    last_message_text="current preview",
                    unread_count=3,
                )
                unpaired = TelegramChat(
                    user_id=user_id,
                    telegram_chat_id=unpaired_raw_id,
                    chat_type=chat_type,
                    title="Unpaired title",
                    unread_count=7,
                )
                db.add_all([user, legacy, canonical, unpaired])
                await db.flush()

                db.add_all(
                    [
                        TelegramMessage(
                            chat_id=legacy.id,
                            telegram_message_id=1,
                            text="duplicate legacy message",
                            sent_at=now - timedelta(minutes=2),
                        ),
                        TelegramMessage(
                            chat_id=legacy.id,
                            telegram_message_id=2,
                            text="legacy-only message",
                            sent_at=now - timedelta(minutes=1),
                        ),
                        TelegramMessage(
                            chat_id=canonical.id,
                            telegram_message_id=1,
                            text="canonical message",
                            sent_at=now - timedelta(minutes=2),
                        ),
                        SyncJob(
                            user_id=user_id,
                            chat_id=legacy.id,
                            status=SyncStatus.COMPLETED,
                        ),
                        MetadataReconciliationCheckpoint(
                            user_id=user_id,
                            chat_id=legacy.id,
                            last_telegram_message_id=8,
                            processed_count=8,
                            status="completed",
                            started_at=now - timedelta(hours=2),
                            completed_at=now - timedelta(hours=1),
                        ),
                        MetadataReconciliationCheckpoint(
                            user_id=user_id,
                            chat_id=canonical.id,
                            last_telegram_message_id=4,
                            processed_count=4,
                            status="pending",
                            started_at=now - timedelta(hours=1),
                        ),
                    ]
                )
                await db.flush()

                migration = _load_channel_identity_migration()
                await db.run_sync(
                    lambda sync_session: migration.merge_legacy_channel_peer_ids(
                        sync_session.connection()
                    )
                )
                db.expire_all()

                chats = (
                    (
                        await db.execute(
                            select(TelegramChat).where(TelegramChat.user_id == user_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(chats) == 2
                chats_by_telegram_id = {chat.telegram_chat_id: chat for chat in chats}
                merged = chats_by_telegram_id[canonical_channel_id]
                assert merged.id == canonical.id
                assert merged.telegram_chat_id == canonical_channel_id
                assert merged.title == "Current title"
                assert merged.last_message_text == "current preview"
                assert merged.unread_count == 3
                assert merged.total_messages_synced == 2
                assert merged.last_message_id == 2
                assert chats_by_telegram_id[unpaired_canonical_id].id == unpaired.id
                assert chats_by_telegram_id[unpaired_canonical_id].unread_count == 7

                messages = (
                    (
                        await db.execute(
                            select(TelegramMessage)
                            .where(TelegramMessage.chat_id == canonical.id)
                            .order_by(TelegramMessage.telegram_message_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                assert [message.telegram_message_id for message in messages] == [1, 2]
                assert messages[0].text == "canonical message"

                job = (
                    await db.execute(select(SyncJob).where(SyncJob.user_id == user_id))
                ).scalar_one()
                assert job.chat_id == canonical.id

                checkpoint = (
                    await db.execute(
                        select(MetadataReconciliationCheckpoint).where(
                            MetadataReconciliationCheckpoint.user_id == user_id
                        )
                    )
                ).scalar_one()
                assert checkpoint.chat_id == canonical.id
                assert checkpoint.last_telegram_message_id == 8
                assert checkpoint.processed_count == 8
                assert checkpoint.status == "completed"
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_CONTRACT_DATABASE_URL"),
    reason="POSTGRES_CONTRACT_DATABASE_URL is not configured",
)
async def test_hybrid_search_uses_fts_trigram_and_pgvector_after_migrations():
    engine = create_async_engine(os.environ["POSTGRES_CONTRACT_DATABASE_URL"])
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    user_id = uuid4()
    embedding = [0.0] * 1536
    embedding[0] = 1.0

    try:
        async with engine.connect() as connection:
            index_names = set(
                (
                    await connection.execute(
                        text(
                            "SELECT indexname FROM pg_indexes "
                            "WHERE schemaname = current_schema()"
                        )
                    )
                ).scalars()
            )
        assert "ix_telegram_messages_search_vector_gin" in index_names
        assert "ix_message_content_chunks_search_vector_gin" in index_names
        assert "ix_telegram_messages_media_file_name_trgm" in index_names
        assert "ix_telegram_messages_searchable_metadata_trgm" in index_names

        async with session_factory() as db, db.begin():
            user = User(
                id=user_id,
                email=f"postgres-contract-{user_id}@example.com",
                password_hash="not-used",
                is_active=False,
            )
            chat = TelegramChat(
                user_id=user_id,
                telegram_chat_id=9_001,
                chat_type=ChatType.PRIVATE,
                title="Postgres contract",
                indexing_enabled=True,
            )
            db.add_all([user, chat])
            await db.flush()
            message = TelegramMessage(
                chat_id=chat.id,
                telegram_message_id=9_002,
                text="Quarterly roadmap",
                has_media=True,
                media_type="document",
                media_file_name="exact-roadmap-2026.pdf",
                content_text="многоязычный текст " * 100_000,
                searchable_metadata="https://hidden.example/private-target",
                hidden_urls=["https://hidden.example/private-target"],
                embedding=embedding,
                sent_at=datetime.now(UTC),
            )
            db.add(message)
            await db.flush()
            vector_size = (
                await db.execute(
                    text(
                        "SELECT octet_length(search_vector::text) "
                        "FROM telegram_messages WHERE id = :message_id"
                    ),
                    {"message_id": message.id},
                )
            ).scalar_one()
            assert vector_size > 0
            db.add(
                MessageContentChunk(
                    message_id=message.id,
                    chunk_index=0,
                    text="The launch phrase is violet telescope.",
                    embedding=embedding,
                )
            )
            await db.flush()

            with patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
            ) as exact_embedding:
                exact_response = await semantic_search(
                    db,
                    user_id,
                    SearchRequest(
                        query="violet telescope",
                        mode="exact",
                        chat_types=["private"],
                    ),
                )
                filtered_response = await semantic_search(
                    db,
                    user_id,
                    SearchRequest(
                        query="violet telescope",
                        mode="exact",
                        chat_types=["group", "supergroup"],
                    ),
                )

            exact_embedding.assert_not_awaited()
            assert [item.telegram_message_id for item in exact_response.results] == [
                9_002
            ]
            assert filtered_response.results == []

            with patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
                return_value=embedding,
            ):
                for query in (
                    "exact-roadmap-2026.pdf",
                    "https://hidden.example/private-target",
                    "violet telescope",
                ):
                    response = await semantic_search(
                        db,
                        user_id,
                        SearchRequest(query=query, limit=5),
                    )
                    assert [item.telegram_message_id for item in response.results] == [
                        9_002
                    ]

                pagination_messages = []
                for offset in range(3):
                    pagination_message = TelegramMessage(
                        chat_id=chat.id,
                        telegram_message_id=9_010 + offset,
                        text="pagination-uniform-token",
                        has_media=False,
                        embedding=embedding,
                        sent_at=datetime.now(UTC) - timedelta(minutes=offset),
                    )
                    db.add(pagination_message)
                    pagination_messages.append(pagination_message)
                await db.flush()

                seen: list[int] = []
                cursor = None
                for _page_number in range(10):
                    page = await semantic_search(
                        db,
                        user_id,
                        SearchRequest(
                            query="pagination-uniform-token",
                            limit=1,
                            cursor=cursor,
                        ),
                    )
                    assert len(page.results) == 1
                    seen.append(page.results[0].telegram_message_id)
                    cursor = page.next_cursor
                    if not page.has_more:
                        break
                    assert cursor is not None
                else:
                    pytest.fail("Search pagination did not terminate")
                assert {9_010, 9_011, 9_012}.issubset(seen)
                assert len(set(seen)) == len(seen)

            checkpoint = MetadataReconciliationCheckpoint(
                user_id=user_id,
                chat_id=chat.id,
            )
            db.add(checkpoint)
            await db.flush()

            @asynccontextmanager
            async def same_session():
                yield db

            reconciled_values = {
                "chat_id": chat.id,
                "telegram_message_id": message.telegram_message_id,
                "text": "Quarterly roadmap, edited",
                "has_media": True,
                "media_type": "document",
                "media_file_name": "exact-roadmap-2026.pdf",
                "media_mime_type": "application/pdf",
                "media_file_size": 123,
                "media_duration_seconds": None,
                "sender_id": 42,
                "sender_name": "Owner",
                "is_outgoing": True,
                "sent_at": message.sent_at,
                "entities": [{"type": "MessageEntityBold"}],
                "visible_urls": [],
                "hidden_urls": ["https://hidden.example/private-target"],
                "buttons": [],
                "webpage_preview": None,
                "reply_to_message_id": None,
                "thread_id": None,
                "forward_origin": None,
                "album_id": None,
                "reactions": [],
                "edited_at": datetime.now(UTC),
                "poll": None,
                "contact": None,
                "location": None,
                "service_event": None,
                "searchable_metadata": "exact-roadmap-2026.pdf",
            }
            with (
                patch(
                    "app.services.metadata_reconciliation.get_db_context",
                    side_effect=same_session,
                ),
                patch(
                    "app.services.metadata_reconciliation.is_user_active",
                    new_callable=AsyncMock,
                    return_value=True,
                ),
            ):
                assert (
                    await _save_metadata_batch(
                        user_id,
                        checkpoint.id,
                        [reconciled_values],
                        message.telegram_message_id,
                    )
                    == 1
                )

            revision = (
                await db.execute(
                    select(MessageRevision).where(
                        MessageRevision.message_id == message.id
                    )
                )
            ).scalar_one()
            assert revision.revision == 1
            assert revision.text == "Quarterly roadmap"
            await db.refresh(message)
            assert message.text == "Quarterly roadmap, edited"
            await db.rollback()
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_CONTRACT_DATABASE_URL"),
    reason="POSTGRES_CONTRACT_DATABASE_URL is not configured",
)
async def test_postgres_allows_invited_accounts_and_enforces_media_integrity():
    engine = create_async_engine(os.environ["POSTGRES_CONTRACT_DATABASE_URL"])
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as db:
        transaction = await db.begin()
        try:
            await db.execute(update(User).values(is_active=False))
            owner = User(
                email=f"owner-contract-{uuid4()}@example.com",
                password_hash="not-used",
                is_active=True,
            )
            archived = User(
                email=f"archive-contract-{uuid4()}@example.com",
                password_hash="not-used",
                is_active=False,
            )
            db.add_all([owner, archived])
            await db.flush()

            from app.services.single_user import validate_active_owner

            guest = User(
                email=f"invited-contract-{uuid4()}@example.com",
                password_hash="not-used",
                is_active=True,
                invited_by_id=owner.id,
                expected_telegram_user_id=987654,
            )
            db.add(guest)
            await db.flush()
            await validate_active_owner(db, owner.id)

            chat = TelegramChat(
                user_id=owner.id,
                telegram_chat_id=9_100,
                chat_type=ChatType.PRIVATE,
                title="Owner constraint",
            )
            db.add(chat)
            await db.flush()
            message = TelegramMessage(
                chat_id=chat.id,
                telegram_message_id=9_101,
                has_media=True,
                media_type="document",
                sent_at=datetime.now(UTC),
            )
            db.add(message)
            await db.flush()

            with pytest.raises(DBAPIError, match="does not own message"):
                async with db.begin_nested():
                    db.add(
                        MediaObject(
                            user_id=archived.id,
                            message_id=message.id,
                            cache_key="a" * 64,
                        )
                    )
                    await db.flush()

            with pytest.raises(IntegrityError):
                async with db.begin_nested():
                    db.add(
                        MediaObject(
                            user_id=owner.id,
                            message_id=message.id,
                            cache_key="b" * 64,
                            status="invalid-status",
                            byte_offset=-1,
                        )
                    )
                    await db.flush()
        finally:
            await transaction.rollback()
            await engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_CONTRACT_DATABASE_URL"),
    reason="POSTGRES_CONTRACT_DATABASE_URL is not configured",
)
async def test_selected_embedding_repair_and_search_never_cross_accounts():
    from app.models.settings import UserSettings
    from app.tasks.embedding_tasks import repair_selected_embeddings
    from app.services.indexing_service import get_indexing_status

    engine = create_async_engine(os.environ["POSTGRES_CONTRACT_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    vector = [1.0] + [0.0] * 1535
    try:
        async with sessions() as db:
            transaction = await db.begin()
            try:
                users = [
                    User(
                        email=f"repair-{uuid4()}@example.test",
                        password_hash="unused",
                        is_active=True,
                    )
                    for _ in range(2)
                ]
                db.add_all(users)
                await db.flush()
                db.add_all(
                    [
                        UserSettings(
                            user_id=users[0].id, embedding_backfill_enabled=True
                        ),
                        UserSettings(
                            user_id=users[1].id, embedding_backfill_enabled=False
                        ),
                    ]
                )
                chats = [
                    TelegramChat(
                        user_id=user.id,
                        telegram_chat_id=i + 101,
                        title="Test school",
                        chat_type=ChatType.PRIVATE,
                        indexing_enabled=selected,
                    )
                    for i, (user, selected) in enumerate(
                        [
                            (users[0], True),
                            (users[0], False),
                            (users[1], True),
                        ]
                    )
                ]
                db.add_all(chats)
                await db.flush()
                messages = [
                    TelegramMessage(
                        chat_id=chat.id,
                        telegram_message_id=i + 1,
                        text="school launch violet",
                        sent_at=datetime.now(UTC),
                    )
                    for i, chat in enumerate(chats)
                ]
                db.add_all(messages)
                await db.flush()

                @asynccontextmanager
                async def database():
                    yield db
                    await db.flush()

                with (
                    patch("app.tasks.embedding_tasks.get_db_context", database),
                    patch(
                        "app.tasks.embedding_tasks.indexing_has_capacity",
                        return_value=True,
                    ),
                    patch(
                        "app.services.embedding_service.generate_embeddings",
                        new_callable=AsyncMock,
                    ) as provider,
                ):
                    provider.side_effect = RuntimeError("temporary outage")
                    assert await repair_selected_embeddings() == 0
                    assert all(message.embedding is None for message in messages)
                    provider.side_effect = None
                    provider.return_value = [vector]
                    assert await repair_selected_embeddings() == 1
                    assert messages[0].embedding is not None
                    assert messages[1].embedding is None
                    assert messages[2].embedding is None
                    provider.reset_mock()
                    assert await repair_selected_embeddings() == 0
                    provider.assert_not_awaited()

                # Exact and vector search both enforce account and selection filters.
                for message in messages[1:]:
                    message.embedding = vector
                await db.flush()
                with patch(
                    "app.services.search_service.generate_query_embedding",
                    new_callable=AsyncMock,
                    return_value=vector,
                ):
                    for mode in ["exact", "hybrid"]:
                        result = await semantic_search(
                            db,
                            users[0].id,
                            SearchRequest(query="school launch violet", mode=mode),
                        )
                        assert [
                            item.telegram_message_id for item in result.results
                        ] == [1]
                status = await get_indexing_status(db, users[0].id)
                assert len(status["chats"]) == 1
                assert status["chats"][0]["indexed_texts"] == 1
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()
