from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy import select
from telethon.errors import SessionRevokedError

from app.listener.main import TelegramListener
from app.models.chat import ChatType, TelegramChat
from app.models.media import MediaObject, MediaObjectStatus, MediaStage
from app.models.message import MediaProcessingStatus, TelegramMessage


class TestTelegramListener:
    async def test_handle_message_checks_activity_in_the_write_transaction(
        self, db_session, test_user
    ):
        chat = TelegramChat(
            indexing_enabled=True,
            user_id=test_user.id,
            telegram_chat_id=321,
            chat_type=ChatType.PRIVATE,
            title="Existing chat",
        )
        db_session.add(chat)
        await db_session.flush()

        context_entries = 0

        @asynccontextmanager
        async def db_context():
            nonlocal context_entries
            context_entries += 1
            yield db_session

        event_message = SimpleNamespace(
            id=11,
            text="hello",
            media=None,
            action=None,
            sender_id=42,
            out=False,
            date=datetime.now(UTC),
        )
        event = SimpleNamespace(
            message=event_message,
            chat_id=chat.telegram_chat_id,
            get_sender=AsyncMock(return_value=None),
        )

        with (
            patch("app.listener.main.get_db_context", db_context),
            patch("app.listener.main.embed_messages", new_callable=AsyncMock),
        ):
            await TelegramListener()._handle_message(test_user.id, event)

        # One transaction performs the activity check and write; a separate
        # best-effort transaction handles embeddings after the insert.
        assert context_entries == 2
        stored = await db_session.scalar(
            select(TelegramMessage).where(
                TelegramMessage.chat_id == chat.id,
                TelegramMessage.telegram_message_id == event_message.id,
            )
        )
        assert stored is not None

    async def test_start_user_ignores_inactive_user(self):
        listener = TelegramListener()
        user_id = uuid4()

        db = AsyncMock()
        context = MagicMock()
        context.return_value.__aenter__ = AsyncMock(return_value=db)
        context.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.listener.main.get_db_context", context),
            patch(
                "app.listener.main.is_user_active",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "app.listener.main.get_client",
                new_callable=AsyncMock,
            ) as get_client,
        ):
            await listener._start_user(user_id)

        get_client.assert_not_awaited()
        assert user_id not in listener.clients

    async def test_handle_unauthorized_client_invalidates_and_stops_user(self):
        listener = TelegramListener()
        listener.redis = AsyncMock()

        user_id = uuid4()
        client = AsyncMock()
        listener.clients[user_id] = client

        with patch(
            "app.listener.main.invalidate_client_authorization",
            new_callable=AsyncMock,
        ) as mock_invalidate:
            await listener._handle_unauthorized_client(
                user_id,
                client,
                SessionRevokedError(request=None),
            )

        mock_invalidate.assert_awaited_once()
        assert mock_invalidate.await_args.args[0] is client
        assert mock_invalidate.await_args.args[1] == user_id
        assert user_id not in listener.clients
        client.disconnect.assert_awaited_once()
        listener.redis.delete.assert_awaited_once_with(f"listener:active:{user_id}")

    async def test_private_delete_without_chat_id_tombstones_unique_owner_message(
        self, db_session, test_user
    ):
        chat = TelegramChat(
            indexing_enabled=True,
            user_id=test_user.id,
            telegram_chat_id=123,
            chat_type=ChatType.PRIVATE,
            title="Saved Messages",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=77,
            text="delete me",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()

        @asynccontextmanager
        async def db_context():
            yield db_session

        listener = TelegramListener()
        event = SimpleNamespace(chat_id=None, deleted_ids=[77])
        with (
            patch("app.listener.main.get_db_context", db_context),
            patch(
                "app.listener.main.is_user_active",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await listener._handle_delete(test_user.id, event)

        stored = await db_session.scalar(
            select(TelegramMessage).where(TelegramMessage.id == message.id)
        )
        assert stored is not None
        assert stored.deleted_at is not None

    async def test_private_delete_without_chat_id_skips_ambiguous_message_id(
        self, db_session, test_user
    ):
        chats = [
            TelegramChat(
                indexing_enabled=True,
                user_id=test_user.id,
                telegram_chat_id=telegram_chat_id,
                chat_type=ChatType.PRIVATE,
                title=f"Private {telegram_chat_id}",
            )
            for telegram_chat_id in (123, 456)
        ]
        db_session.add_all(chats)
        await db_session.flush()
        messages = [
            TelegramMessage(
                chat_id=chat.id,
                telegram_message_id=77,
                text="same local Telegram ID",
                sent_at=datetime.now(UTC),
            )
            for chat in chats
        ]
        db_session.add_all(messages)
        await db_session.flush()

        @asynccontextmanager
        async def db_context():
            yield db_session

        listener = TelegramListener()
        event = SimpleNamespace(chat_id=None, deleted_ids=[77])
        with (
            patch("app.listener.main.get_db_context", db_context),
            patch(
                "app.listener.main.is_user_active",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await listener._handle_delete(test_user.id, event)

        stored = (
            await db_session.execute(
                select(TelegramMessage).where(
                    TelegramMessage.id.in_([message.id for message in messages])
                )
            )
        ).scalars()
        assert all(message.deleted_at is None for message in stored)

    async def test_media_edit_invalidates_stale_cached_bytes(
        self, db_session, test_user
    ):
        chat = TelegramChat(
            indexing_enabled=True,
            user_id=test_user.id,
            telegram_chat_id=789,
            chat_type=ChatType.PRIVATE,
            title="Edited media",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=91,
            text="old caption",
            has_media=True,
            media_type="document",
            media_processing_status="ready",
            content_text="old content",
            content_summary="old summary",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()
        cached = MediaObject(
            user_id=test_user.id,
            message_id=message.id,
            telegram_media_id="document:old",
            cache_key="6" * 64,
            relative_path="66/cache/original.pdf",
            sha256="5" * 64,
            status=MediaObjectStatus.READY,
            stage=MediaStage.COMPLETE,
        )
        db_session.add(cached)
        await db_session.flush()

        @asynccontextmanager
        async def db_context():
            yield db_session

        edited_message = SimpleNamespace(
            id=91,
            text="new caption",
            edit_date=datetime.now(UTC),
            media=SimpleNamespace(
                document=SimpleNamespace(
                    id=222,
                    access_hash=333,
                    dc_id=4,
                    file_reference=b"new-reference",
                )
            ),
        )
        event = SimpleNamespace(chat_id=789, message=edited_message)
        with (
            patch("app.listener.main.get_db_context", db_context),
            patch(
                "app.listener.main.is_user_active",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.listener.main.get_media_info",
                return_value=SimpleNamespace(
                    media_type="document",
                    file_name="new.pdf",
                    mime_type="application/pdf",
                    file_size=123,
                    duration_seconds=None,
                ),
            ),
            patch("app.listener.main.extract_message_metadata", return_value={}),
        ):
            await TelegramListener()._handle_edit(test_user.id, event)

        assert cached.relative_path is None
        assert cached.sha256 is None
        assert cached.status == MediaObjectStatus.PENDING
        assert message.content_text is None
        assert message.content_summary is None
        assert message.media_processing_status == MediaProcessingStatus.PENDING
