"""Tests for app.services.messaging_service — pure unit tests (no Telegram calls)."""

import socket
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.messaging_service import (
    _handle_telethon_error,
    _sanitize_file_name,
    _validate_url,
)


# ---------------------------------------------------------------------------
# _validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_allows_http(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))
            ]
            _validate_url("http://example.com/file.pdf")

    def test_allows_https(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))
            ]
            _validate_url("https://example.com/file.pdf")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("ftp://example.com/file.pdf")

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("file:///etc/passwd")

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            _validate_url("http:///path")

    def test_rejects_private_ip(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))
            ]
            with pytest.raises(ValueError, match="private"):
                _validate_url("http://internal.example.com/file")

    def test_rejects_loopback(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
            ]
            with pytest.raises(ValueError, match="private"):
                _validate_url("http://localhost/file")

    def test_rejects_link_local(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.1", 0))
            ]
            with pytest.raises(ValueError, match="private"):
                _validate_url("http://link-local.example.com/file")

    def test_rejects_unresolvable_hostname(self):
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name not found")
            with pytest.raises(ValueError, match="Cannot resolve hostname"):
                _validate_url("http://nonexistent.invalid/file")

    def test_strips_ipv6_zone_id(self):
        """Ensure IPv6 zone IDs like %eth0 are stripped before parsing."""
        with patch("app.services.messaging_service.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fe80::1%eth0", 80, 0, 0))
            ]
            with pytest.raises(ValueError, match="private"):
                _validate_url("http://[fe80::1%25eth0]/file")


# ---------------------------------------------------------------------------
# _sanitize_file_name
# ---------------------------------------------------------------------------


class TestSanitizeFileName:
    def test_normal_filename(self):
        assert _sanitize_file_name("report.pdf") == "report.pdf"

    def test_strips_path_components(self):
        assert _sanitize_file_name("/etc/passwd") == "passwd"

    def test_removes_null_bytes(self):
        assert _sanitize_file_name("file\x00name.txt") == "file_name.txt"

    def test_removes_slashes(self):
        # Path.name strips forward slashes; backslash is replaced by _
        assert _sanitize_file_name("a/b\\c.txt") == "b_c.txt"

    def test_strips_forward_slash_path(self):
        assert _sanitize_file_name("dir/subdir/file.txt") == "file.txt"

    def test_removes_dangerous_chars(self):
        result = _sanitize_file_name('file|<>:"name.txt')
        assert "|" not in result
        assert "<" not in result
        assert ">" not in result

    def test_strips_leading_dashes(self):
        assert _sanitize_file_name("--flag.txt") == "flag.txt"

    def test_strips_leading_dots(self):
        assert _sanitize_file_name(".hidden.txt") == "hidden.txt"

    def test_truncates_long_name(self):
        long_name = "a" * 300 + ".pdf"
        result = _sanitize_file_name(long_name)
        assert len(result) <= 200

    def test_preserves_extension_on_truncation(self):
        long_name = "a" * 300 + ".pdf"
        result = _sanitize_file_name(long_name)
        assert result.endswith(".pdf")

    def test_empty_after_sanitize_returns_file(self):
        assert _sanitize_file_name("...") == "file"

    def test_returns_file_for_empty_string(self):
        assert _sanitize_file_name("") == "file"


# ---------------------------------------------------------------------------
# _handle_telethon_error
# ---------------------------------------------------------------------------


class TestHandleTelethonError:
    def test_flood_wait_error(self):
        from telethon.errors import FloodWaitError

        # FloodWaitError needs a real request; create via RPCError subclass
        try:
            raise FloodWaitError(request=None, capture=42)
        except FloodWaitError as err:
            with pytest.raises(ValueError, match="rate limit"):
                _handle_telethon_error(err)

    def test_chat_write_forbidden(self):
        from telethon.errors import ChatWriteForbiddenError

        try:
            raise ChatWriteForbiddenError(request=None)
        except ChatWriteForbiddenError as err:
            with pytest.raises(ValueError, match="permission"):
                _handle_telethon_error(err)

    def test_user_banned(self):
        from telethon.errors import UserBannedInChannelError

        try:
            raise UserBannedInChannelError(request=None)
        except UserBannedInChannelError as err:
            with pytest.raises(ValueError, match="banned"):
                _handle_telethon_error(err)

    def test_generic_error(self):
        err = RuntimeError("something broke")
        with pytest.raises(ValueError, match="Telegram error"):
            _handle_telethon_error(err)


# ---------------------------------------------------------------------------
# send_message (mock Telethon)
# ---------------------------------------------------------------------------


class TestSendMessage:
    async def test_send_message_success(self, db_session, test_user):
        from app.services.messaging_service import send_message
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = MagicMock()
        mock_result.id = 999

        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_result)
        mock_client.disconnect = AsyncMock()
        resolved_entity = object()

        with (
            patch(
                "app.services.messaging_service.get_client", return_value=mock_client
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
        ):
            result = await send_message(db_session, test_user.id, chat.id, "Hello")

        assert result["telegram_message_id"] == 999
        assert result["text"] == "Hello"
        mock_client.send_message.assert_awaited_once_with(resolved_entity, "Hello")
        mock_client.disconnect.assert_awaited_once()

    async def test_send_message_chat_not_found(self, db_session, test_user):
        from app.services.messaging_service import send_message

        with pytest.raises(ValueError, match="not found"):
            await send_message(db_session, test_user.id, uuid4(), "Hello")

    async def test_send_message_auth_error_invalidates_client(
        self, db_session, test_user
    ):
        from app.services.messaging_service import send_message
        from app.services.telegram_client import TelegramSessionUnauthorizedError
        from telethon.errors import SessionRevokedError
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.send_message = AsyncMock(
            side_effect=SessionRevokedError(request=None)
        )
        resolved_entity = object()

        with (
            patch(
                "app.services.messaging_service.get_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
            patch(
                "app.services.messaging_service.invalidate_client_authorization",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            with pytest.raises(
                TelegramSessionUnauthorizedError,
                match="Reconnect Telegram",
            ):
                await send_message(db_session, test_user.id, chat.id, "Hello")

        mock_invalidate.assert_awaited_once()


# ---------------------------------------------------------------------------
# save_draft (mock Telethon)
# ---------------------------------------------------------------------------


class TestSaveDraft:
    async def test_save_draft_success_does_not_send_message(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft
        from telethon.tl.functions.messages import SaveDraftRequest
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock(return_value=True)
        mock_client.disconnect = AsyncMock()
        mock_client.send_message = AsyncMock()
        resolved_entity = object()
        text = "Черновик с emoji 🌱 и ссылкой https://example.com\nВторая строка"

        with (
            patch(
                "app.services.messaging_service.get_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
        ):
            result = await save_draft(db_session, test_user.id, chat.id, text)

        request = mock_client.await_args.args[0]
        assert isinstance(request, SaveDraftRequest)
        assert request.peer is resolved_entity
        assert request.message == text
        assert result == {
            "chat_id": str(chat.id),
            "text": text,
            "saved": True,
            "sent": False,
            "replaces_existing_draft": True,
        }
        mock_client.send_message.assert_not_awaited()
        mock_client.disconnect.assert_awaited_once()

    async def test_save_draft_rejects_blank_text_before_connecting(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft

        with (
            patch("app.services.messaging_service.get_client") as get_client,
            pytest.raises(ValueError, match="must not be empty"),
        ):
            await save_draft(db_session, test_user.id, uuid4(), " \n\t ")

        get_client.assert_not_called()

    async def test_save_draft_chat_not_found(self, db_session, test_user):
        from app.services.messaging_service import save_draft

        with pytest.raises(ValueError, match="not found"):
            await save_draft(db_session, test_user.id, uuid4(), "Draft")

    async def test_save_draft_cannot_access_another_users_chat(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft
        from tests.factories import TelegramChatFactory, UserFactory

        archived_user = UserFactory.create(is_active=False)
        db_session.add(archived_user)
        await db_session.flush()
        foreign_chat = TelegramChatFactory.create(user_id=archived_user.id)
        db_session.add(foreign_chat)
        await db_session.flush()

        with pytest.raises(ValueError, match="not found"):
            await save_draft(
                db_session,
                test_user.id,
                foreign_chat.id,
                "Must stay isolated",
            )

    async def test_save_draft_auth_error_invalidates_client(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft
        from app.services.telegram_client import TelegramSessionUnauthorizedError
        from telethon.errors import SessionRevokedError
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock(side_effect=SessionRevokedError(request=None))
        mock_client.disconnect = AsyncMock()
        resolved_entity = object()

        with (
            patch(
                "app.services.messaging_service.get_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
            patch(
                "app.services.messaging_service.invalidate_client_authorization",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            with pytest.raises(
                TelegramSessionUnauthorizedError,
                match="Reconnect Telegram",
            ):
                await save_draft(db_session, test_user.id, chat.id, "Draft")

        mock_invalidate.assert_awaited_once()
        mock_client.disconnect.assert_awaited_once()

    async def test_save_draft_deleted_recipient_does_not_invalidate_owner_session(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft
        from telethon.errors import InputUserDeactivatedError
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock(side_effect=InputUserDeactivatedError(request=None))
        mock_client.disconnect = AsyncMock()

        with (
            patch(
                "app.services.messaging_service.get_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=object(),
            ),
            patch(
                "app.services.messaging_service.invalidate_client_authorization",
                new_callable=AsyncMock,
            ) as mock_invalidate,
        ):
            with pytest.raises(ValueError, match="Telegram error"):
                await save_draft(db_session, test_user.id, chat.id, "Draft")

        mock_invalidate.assert_not_awaited()
        mock_client.disconnect.assert_awaited_once()

    async def test_save_draft_false_confirmation_raises_and_disconnects(
        self, db_session, test_user
    ):
        from app.services.messaging_service import save_draft
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock(return_value=False)
        mock_client.disconnect = AsyncMock()

        with (
            patch(
                "app.services.messaging_service.get_client",
                return_value=mock_client,
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=object(),
            ),
            pytest.raises(ValueError, match="did not confirm"),
        ):
            await save_draft(db_session, test_user.id, chat.id, "Draft")

        mock_client.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# draft reconciliation (mock Telethon)
# ---------------------------------------------------------------------------


class TestDraftReconciliation:
    async def test_list_drafts_returns_live_text_for_owned_chats(
        self, db_session, test_user
    ):
        from app.services.messaging_service import list_drafts
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(
            user_id=test_user.id,
            telegram_chat_id=123456,
            title="Candidate",
            username="candidate",
        )
        db_session.add(chat)
        await db_session.flush()

        draft = SimpleNamespace(
            is_empty=False,
            entity=object(),
            text="Current draft",
            date=datetime.now(UTC),
            reply_to_msg_id=42,
            link_preview=False,
        )
        mock_client = AsyncMock()
        mock_client.get_drafts.return_value = [draft]
        mock_client.disconnect = AsyncMock()

        with (
            patch(
                "app.services.messaging_service.get_client", return_value=mock_client
            ),
            patch("app.services.messaging_service.get_peer_id", return_value=123456),
        ):
            result = await list_drafts(db_session, test_user.id)

        assert result["count"] == 1
        assert result["unmatched_count"] == 0
        assert result["drafts"][0]["chat_id"] == str(chat.id)
        assert result["drafts"][0]["text"] == "Current draft"
        assert result["drafts"][0]["reply_to_message_id"] == 42
        mock_client.disconnect.assert_awaited_once()

    async def test_clear_draft_verifies_empty_without_sending(
        self, db_session, test_user
    ):
        from app.services.messaging_service import clear_draft
        from telethon.tl.functions.messages import SaveDraftRequest
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_client = AsyncMock(return_value=True)
        mock_client.get_drafts.return_value = SimpleNamespace(is_empty=True)
        mock_client.disconnect = AsyncMock()
        mock_client.send_message = AsyncMock()
        resolved_entity = object()

        with (
            patch(
                "app.services.messaging_service.get_client", return_value=mock_client
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
        ):
            result = await clear_draft(db_session, test_user.id, chat.id)

        request = mock_client.await_args.args[0]
        assert isinstance(request, SaveDraftRequest)
        assert request.peer is resolved_entity
        assert request.message == ""
        assert result == {
            "chat_id": str(chat.id),
            "cleared": True,
            "saved": True,
            "sent": False,
        }
        mock_client.get_drafts.assert_awaited_once_with(resolved_entity)
        mock_client.send_message.assert_not_awaited()
        mock_client.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_chat_entity
# ---------------------------------------------------------------------------


class TestResolveChatEntity:
    def test_dialog_match_tolerates_marked_chat_ids(self):
        from app.services.messaging_service import _dialog_matches_chat
        from tests.factories import TelegramChatFactory

        entity = SimpleNamespace(id=123)
        dialog = SimpleNamespace(entity=entity)
        chat = TelegramChatFactory.create(telegram_chat_id=-1000000000123)

        with (
            patch(
                "app.services.messaging_service._entity_chat_type",
                return_value=chat.chat_type,
            ),
            patch(
                "app.services.messaging_service.get_peer_id",
                return_value=-1000000000123,
            ),
        ):
            assert _dialog_matches_chat(dialog, chat) is True

    async def test_prefers_stored_input_peer_for_private_chat(self, db_session):
        from app.services.messaging_service import _resolve_chat_entity
        from telethon.tl.types import InputPeerUser
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(telegram_chat_id=123, access_hash=999)
        mock_client = AsyncMock()

        entity = await _resolve_chat_entity(mock_client, db_session, chat)

        assert isinstance(entity, InputPeerUser)
        assert entity.user_id == 123
        assert entity.access_hash == 999
        mock_client.get_input_entity.assert_not_called()

    async def test_persists_access_hash_from_cached_entity(self, db_session):
        from app.services.messaging_service import _resolve_chat_entity
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(access_hash=None)
        cached_entity = SimpleNamespace(access_hash=777)
        mock_client = AsyncMock()
        mock_client.get_input_entity = AsyncMock(return_value=cached_entity)

        entity = await _resolve_chat_entity(mock_client, db_session, chat)

        assert entity is cached_entity
        assert chat.access_hash == 777

    async def test_falls_back_to_dialog_warmup(self, db_session):
        from app.services.messaging_service import _resolve_chat_entity
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(username="alice")
        resolved_entity = object()
        dialog = SimpleNamespace(entity=object(), input_entity=resolved_entity)

        mock_client = AsyncMock()
        mock_client.get_input_entity = AsyncMock(side_effect=ValueError("cache miss"))

        async def fake_iter_dialogs():
            yield dialog

        mock_client.iter_dialogs = fake_iter_dialogs

        with patch(
            "app.services.messaging_service._dialog_matches_chat", return_value=True
        ):
            entity = await _resolve_chat_entity(mock_client, db_session, chat)

        assert entity is resolved_entity

    async def test_falls_back_to_username_lookup_after_dialog_scan(self, db_session):
        from app.services.messaging_service import _resolve_chat_entity
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(username="alice")
        resolved_entity = object()

        mock_client = AsyncMock()
        mock_client.get_input_entity = AsyncMock(
            side_effect=[ValueError("cache miss"), resolved_entity]
        )

        async def fake_iter_dialogs():
            if False:
                yield None

        mock_client.iter_dialogs = fake_iter_dialogs

        entity = await _resolve_chat_entity(mock_client, db_session, chat)

        assert entity is resolved_entity
        assert mock_client.get_input_entity.await_count == 2
        assert mock_client.get_input_entity.await_args_list[1].args == ("alice",)

    async def test_raises_clear_error_when_entity_cannot_be_resolved(self, db_session):
        from app.services.messaging_service import _resolve_chat_entity
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(title="Important Chat")

        mock_client = AsyncMock()
        mock_client.get_input_entity = AsyncMock(side_effect=ValueError("cache miss"))

        async def fake_iter_dialogs():
            if False:
                yield None

        mock_client.iter_dialogs = fake_iter_dialogs

        with pytest.raises(ValueError, match="Could not resolve Telegram entity"):
            await _resolve_chat_entity(mock_client, db_session, chat)


# ---------------------------------------------------------------------------
# send_file (streaming download to temp file)
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], *, content_length: int | None = None):
        self._chunks = chunks
        self.headers = {
            "content-length": str(
                content_length
                if content_length is not None
                else sum(len(chunk) for chunk in chunks)
            )
        }
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int = 64 * 1024):
        for chunk in self._chunks:
            yield chunk


class _FakeHTTPClient:
    def __init__(self, chunks: list[bytes], *, content_length: int | None = None):
        self._chunks = chunks
        self._content_length = content_length

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method: str, url: str):
        assert method == "GET"
        assert url == "https://example.com/doc.pdf"
        return _FakeStreamResponse(
            self._chunks,
            content_length=self._content_length,
        )


class _RedirectHTTPClient(_FakeHTTPClient):
    def stream(self, method: str, url: str):
        response = _FakeStreamResponse([])
        response.status_code = 302
        response.headers = {"location": "http://127.0.0.1/private"}
        return response


class TestSendFile:
    async def test_send_file_has_no_service_size_limit_and_uses_media_volume(
        self,
        db_session,
        test_user,
        tmp_path,
    ):
        from app.services.messaging_service import send_file
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = MagicMock()
        mock_result.id = 456

        observed = {}

        async def fake_telethon_send_file(
            chat_id, file_path, caption=None, file_name=None
        ):
            observed["chat_id"] = chat_id
            observed["caption"] = caption
            observed["file_name"] = file_name
            observed["file_bytes"] = Path(file_path).read_bytes()
            observed["file_path"] = Path(file_path)
            return mock_result

        mock_client = AsyncMock()
        mock_client.send_file.side_effect = fake_telethon_send_file
        mock_client.disconnect = AsyncMock()
        resolved_entity = object()

        fake_http_client = _FakeHTTPClient(
            [b"hello ", b"world"],
            content_length=10_000_000_000,
        )
        http_client_factory = MagicMock(return_value=fake_http_client)
        media_root = tmp_path / "media"
        service_settings = SimpleNamespace(
            environment="production",
            media_root=media_root,
            media_download_stall_timeout_seconds=120.0,
            media_download_chunk_bytes=512 * 1024,
        )

        with (
            patch("app.services.messaging_service._validate_url", return_value=None),
            patch(
                "app.services.messaging_service.httpx.AsyncClient",
                http_client_factory,
            ),
            patch(
                "app.services.messaging_service.get_settings",
                return_value=service_settings,
            ),
            patch(
                "app.services.messaging_service.get_client", return_value=mock_client
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
        ):
            result = await send_file(
                db_session,
                test_user.id,
                chat.id,
                "https://example.com/doc.pdf",
                caption="Report",
            )

        assert result["telegram_message_id"] == 456
        assert result["file_name"] == "doc.pdf"
        assert observed["chat_id"] is resolved_entity
        assert observed["caption"] == "Report"
        assert observed["file_name"] == "doc.pdf"
        assert observed["file_bytes"] == b"hello world"
        assert media_root / "outbound-work" in observed["file_path"].parents
        timeout = http_client_factory.call_args.kwargs["timeout"]
        assert timeout.read == 120.0
        mock_client.disconnect.assert_awaited_once()

    async def test_send_file_revalidates_redirect_targets(
        self,
        db_session,
        test_user,
    ):
        from app.services.messaging_service import send_file
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        validate_url = MagicMock(
            side_effect=[None, ValueError("private redirect is forbidden")]
        )
        with (
            patch("app.services.messaging_service._validate_url", validate_url),
            patch(
                "app.services.messaging_service.httpx.AsyncClient",
                return_value=_RedirectHTTPClient([]),
            ),
            pytest.raises(ValueError, match="private redirect"),
        ):
            await send_file(
                db_session,
                test_user.id,
                chat.id,
                "https://example.com/doc.pdf",
            )

        assert validate_url.call_count == 2


# ---------------------------------------------------------------------------
# reply_to_message (mock Telethon)
# ---------------------------------------------------------------------------


class TestReplyToMessage:
    async def test_reply_success(self, db_session, test_user):
        from app.services.messaging_service import reply_to_message
        from tests.factories import TelegramChatFactory

        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = MagicMock()
        mock_result.id = 1001

        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value=mock_result)
        mock_client.disconnect = AsyncMock()
        resolved_entity = object()

        with (
            patch(
                "app.services.messaging_service.get_client", return_value=mock_client
            ),
            patch(
                "app.services.messaging_service._resolve_chat_entity",
                new_callable=AsyncMock,
                return_value=resolved_entity,
            ),
        ):
            result = await reply_to_message(
                db_session, test_user.id, chat.id, 500, "Reply text"
            )

        assert result["telegram_message_id"] == 1001
        mock_client.send_message.assert_awaited_once()
        # Verify reply_to was passed
        call_kwargs = mock_client.send_message.call_args
        assert call_kwargs.args[0] is resolved_entity
        assert (
            call_kwargs.kwargs.get("reply_to") == 500
            or call_kwargs[1].get("reply_to") == 500
        )
