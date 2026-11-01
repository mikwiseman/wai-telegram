from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from app.models.chat import ChatType, TelegramChat
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User
from app.services.sync_service import (
    TelegramSessionUnauthorizedError,
    _get_chat_title,
    _get_chat_type,
    _get_media_type,
    _get_sender_name,
    sync_chats,
    sync_messages,
)


class TestGetChatType:
    def test_private_user(self):
        dialog = MagicMock()
        from telethon.tl.types import User as TelegramUser

        dialog.entity = MagicMock(spec=TelegramUser)
        assert _get_chat_type(dialog) == ChatType.PRIVATE

    def test_group(self):
        dialog = MagicMock()
        from telethon.tl.types import Chat

        dialog.entity = MagicMock(spec=Chat)
        assert _get_chat_type(dialog) == ChatType.GROUP

    def test_supergroup(self):
        dialog = MagicMock()
        from telethon.tl.types import Channel

        dialog.entity = MagicMock(spec=Channel)
        dialog.entity.megagroup = True
        assert _get_chat_type(dialog) == ChatType.SUPERGROUP

    def test_channel(self):
        dialog = MagicMock()
        from telethon.tl.types import Channel

        dialog.entity = MagicMock(spec=Channel)
        dialog.entity.megagroup = False
        assert _get_chat_type(dialog) == ChatType.CHANNEL


class TestGetChatTitle:
    def test_user_full_name(self):
        dialog = MagicMock()
        from telethon.tl.types import User as TelegramUser

        dialog.entity = MagicMock(spec=TelegramUser)
        dialog.entity.first_name = "John"
        dialog.entity.last_name = "Doe"
        assert _get_chat_title(dialog) == "John Doe"

    def test_user_first_name_only(self):
        dialog = MagicMock()
        from telethon.tl.types import User as TelegramUser

        dialog.entity = MagicMock(spec=TelegramUser)
        dialog.entity.first_name = "Alice"
        dialog.entity.last_name = ""
        assert _get_chat_title(dialog) == "Alice"

    def test_group_title(self):
        dialog = MagicMock()
        from telethon.tl.types import Chat

        dialog.entity = MagicMock(spec=Chat)
        dialog.entity.title = "My Group"
        assert _get_chat_title(dialog) == "My Group"


class TestGetSenderName:
    def test_user_sender(self):
        from telethon.tl.types import User as TelegramUser

        message = MagicMock()
        message.sender = MagicMock(spec=TelegramUser)
        message.sender.first_name = "Bob"
        message.sender.last_name = "Smith"
        assert _get_sender_name(message) == "Bob Smith"

    def test_no_sender(self):
        message = MagicMock()
        message.sender = None
        assert _get_sender_name(message) is None


class TestGetMediaType:
    def test_no_media(self):
        message = MagicMock()
        message.media = None
        assert _get_media_type(message) is None

    def test_photo(self):
        from telethon.tl.types import MessageMediaPhoto

        message = MagicMock()
        message.media = MagicMock(spec=MessageMediaPhoto)
        assert _get_media_type(message) == "photo"

    def test_voice(self):
        from telethon.tl.types import MessageMediaDocument

        message = MagicMock()
        message.media = MagicMock(spec=MessageMediaDocument)
        doc = MagicMock()
        voice_attr = MagicMock()
        voice_attr.voice = True
        doc.attributes = [voice_attr]
        message.media.document = doc
        assert _get_media_type(message) == "voice"


class TestSyncChats:
    @pytest.fixture(autouse=True)
    def _disable_external_rate_limiter(self):
        with patch("app.services.sync_service.record_request"):
            yield

    async def test_zero_dialog_limit_refreshes_all_dialogs(self, db_session, test_user):
        async def iter_dialogs(*_args, **_kwargs):
            if False:
                yield

        mock_client = AsyncMock()
        mock_client.iter_dialogs = MagicMock(return_value=iter_dialogs())

        with (
            patch(
                "app.services.sync_service.get_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch("app.services.sync_service.settings.sync_dialog_limit", 0),
        ):
            await sync_chats(db_session, test_user.id)

        mock_client.iter_dialogs.assert_called_once_with(limit=None)

    async def test_uses_canonical_peer_id_for_group_dialogs(
        self, db_session, test_user
    ):
        from telethon.tl.types import Chat, ChatPhotoEmpty

        entity = Chat(
            id=5461206247,
            title="Wai News",
            photo=ChatPhotoEmpty(),
            participants_count=2,
            date=None,
            version=1,
        )
        dialog = MagicMock()
        dialog.entity = entity
        dialog.message = None
        dialog.date = None
        dialog.unread_count = 0

        async def iter_dialogs(*_args, **_kwargs):
            yield dialog

        mock_client = AsyncMock()
        mock_client.iter_dialogs = MagicMock(return_value=iter_dialogs())

        with patch(
            "app.services.sync_service.get_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            chats = await sync_chats(db_session, test_user.id)

        assert len(chats) == 1
        assert chats[0].telegram_chat_id == -5461206247

    async def test_auth_error_invalidates_client(self, db_session, test_user):
        from telethon.errors import SessionRevokedError

        async def broken_iter_dialogs(*_args, **_kwargs):
            raise SessionRevokedError(request=None)
            yield

        mock_client = AsyncMock()
        mock_client.iter_dialogs = MagicMock(return_value=broken_iter_dialogs())
        mock_client.disconnect = AsyncMock()

        with (
            patch(
                "app.services.sync_service.get_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "app.services.sync_service.invalidate_client_authorization",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            with pytest.raises(
                TelegramSessionUnauthorizedError, match="Reconnect Telegram"
            ):
                await sync_chats(db_session, test_user.id)

        mock_invalidate.assert_awaited_once_with(mock_client, test_user.id, ANY)
        mock_client.disconnect.assert_awaited()


class TestSyncMessagesResolvesEntity:
    """Regression: sync_messages must resolve a proper InputPeer before
    GetHistoryRequest, otherwise channels with missing/stale access_hash
    fail with 'Invalid channel object'."""

    async def test_resolves_chat_entity_before_iter_messages(
        self, db_session, test_user
    ):
        chat = TelegramChat(
            indexing_enabled=True,
            user_id=test_user.id,
            telegram_chat_id=1234567890,
            access_hash=None,  # stale — the exact case that used to fail
            chat_type=ChatType.CHANNEL,
            title="Some Channel",
        )
        db_session.add(chat)
        job = SyncJob(user_id=test_user.id, chat_id=None, status=SyncStatus.PENDING)
        db_session.add(job)
        await db_session.flush()
        job.chat_id = chat.id
        await db_session.flush()

        resolved_peer = object()  # marker

        class EmptyAsyncIter:
            total = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_client = MagicMock()
        mock_client.iter_messages = MagicMock(return_value=EmptyAsyncIter())

        with patch(
            "app.services.sync_service._resolve_chat_entity",
            new_callable=AsyncMock,
            return_value=resolved_peer,
        ) as mock_resolve:
            await sync_messages(
                db_session, test_user.id, chat.id, job.id, client=mock_client
            )

        mock_resolve.assert_awaited_once()
        # First positional arg to iter_messages must be the resolved peer,
        # NOT the bare chat.telegram_chat_id.
        call_args, _ = mock_client.iter_messages.call_args
        assert call_args[0] is resolved_peer

    async def test_rejects_sync_job_owned_by_another_user(self, db_session, test_user):
        chat = TelegramChat(
            indexing_enabled=True,
            user_id=test_user.id,
            telegram_chat_id=987654321,
            chat_type=ChatType.PRIVATE,
            title="Owner chat",
        )
        archived_user = User(
            email="archived@example.com",
            password_hash="not-used",
            is_active=False,
        )
        db_session.add_all([chat, archived_user])
        await db_session.flush()
        foreign_job = SyncJob(
            user_id=archived_user.id,
            chat_id=None,
            status=SyncStatus.PENDING,
        )
        db_session.add(foreign_job)
        await db_session.flush()

        with pytest.raises(ValueError, match="Sync job not found"):
            await sync_messages(
                db_session,
                test_user.id,
                chat.id,
                foreign_job.id,
                client=MagicMock(),
            )
