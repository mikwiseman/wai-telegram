from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.models.chat import ChatType, TelegramChat
from app.models.media import TranscriptSegment
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.services.telegram_client import TelegramSessionUnauthorizedError


class TestListChats:
    async def test_exact_telegram_id_lookup_preserves_owner_scope(
        self, auth_client, db_session, test_user
    ):
        wanted = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=-1001564999252,
            chat_type=ChatType.SUPERGROUP,
            title="Wanted",
        )
        unrelated = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=123,
            chat_type=ChatType.PRIVATE,
            title="Unrelated",
        )
        foreign = TelegramChat(
            user_id=uuid4(),
            telegram_chat_id=-1001564999252,
            chat_type=ChatType.SUPERGROUP,
            title="Foreign",
        )
        db_session.add_all([wanted, unrelated, foreign])
        await db_session.flush()
        response = await auth_client.get(
            "/api/v1/chats", params={"telegram_chat_id": -1001564999252}
        )
        assert response.status_code == 200
        assert [c["id"] for c in response.json()["chats"]] == [str(wanted.id)]
        assert response.json()["total"] == 1
        missing = await auth_client.get(
            "/api/v1/chats", params={"telegram_chat_id": -1001111111111}
        )
        assert missing.json()["chats"] == []
        assert missing.json()["total"] == 0

    async def test_empty_list(self, auth_client):
        response = await auth_client.get("/api/v1/chats")
        assert response.status_code == 200
        data = response.json()
        assert data["chats"] == []
        assert data["total"] == 0

    async def test_with_chats(self, auth_client, db_session, test_user):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=12345,
            chat_type=ChatType.PRIVATE,
            title="Test Chat",
            last_activity_at=datetime.now(UTC),
        )
        db_session.add(chat)
        await db_session.flush()

        response = await auth_client.get("/api/v1/chats")
        assert response.status_code == 200
        data = response.json()
        assert len(data["chats"]) == 1
        assert data["chats"][0]["title"] == "Test Chat"
        assert data["total"] == 1

    async def test_type_filter(self, auth_client, db_session, test_user):
        private_chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=111,
            chat_type=ChatType.PRIVATE,
            title="Private",
        )
        group_chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=222,
            chat_type=ChatType.GROUP,
            title="Group",
        )
        db_session.add_all([private_chat, group_chat])
        await db_session.flush()

        response = await auth_client.get("/api/v1/chats?chat_type=private")
        assert response.status_code == 200
        data = response.json()
        assert len(data["chats"]) == 1
        assert data["chats"][0]["title"] == "Private"

    async def test_unread_only_filter(self, auth_client, db_session, test_user):
        unread_chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=333,
            chat_type=ChatType.PRIVATE,
            title="Unread",
            unread_count=3,
        )
        read_chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=444,
            chat_type=ChatType.PRIVATE,
            title="Read",
            unread_count=0,
        )
        db_session.add_all([unread_chat, read_chat])
        await db_session.flush()

        response = await auth_client.get("/api/v1/chats?unread_only=true")

        assert response.status_code == 200
        data = response.json()
        assert [chat["title"] for chat in data["chats"]] == ["Unread"]
        assert data["chats"][0]["unread_count"] == 3
        assert data["total"] == 1

    async def test_unauthenticated(self, client):
        response = await client.get("/api/v1/chats")
        assert response.status_code == 401


class TestGetChat:
    async def test_get_chat_success(self, auth_client, db_session, test_user):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=12345,
            chat_type=ChatType.PRIVATE,
            title="My Chat",
        )
        db_session.add(chat)
        await db_session.flush()

        response = await auth_client.get(f"/api/v1/chats/{chat.id}")
        assert response.status_code == 200
        assert response.json()["title"] == "My Chat"

    async def test_get_chat_not_found(self, auth_client):
        response = await auth_client.get(f"/api/v1/chats/{uuid4()}")
        assert response.status_code == 404

    async def test_get_other_users_chat(self, auth_client, db_session):
        other_user_id = uuid4()
        chat = TelegramChat(
            user_id=other_user_id,
            telegram_chat_id=99999,
            chat_type=ChatType.PRIVATE,
            title="Other User Chat",
        )
        db_session.add(chat)
        await db_session.flush()

        response = await auth_client.get(f"/api/v1/chats/{chat.id}")
        assert response.status_code == 404


class TestGetChatMessages:
    async def test_messages_success(self, auth_client, db_session, test_user):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=12345,
            chat_type=ChatType.PRIVATE,
            title="Chat with Messages",
        )
        db_session.add(chat)
        await db_session.flush()

        msg = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=1,
            text="Hello world",
            has_media=False,
            sender_id=12345,
            sender_name="Test",
            is_outgoing=False,
            sent_at=datetime.now(UTC),
        )
        db_session.add(msg)
        await db_session.flush()

        response = await auth_client.get(f"/api/v1/chats/{chat.id}/messages")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["text"] == "Hello world"

    async def test_messages_chat_not_found(self, auth_client):
        response = await auth_client.get(f"/api/v1/chats/{uuid4()}/messages")
        assert response.status_code == 404


class TestGetMessageContent:
    async def test_returns_full_media_content(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=54321,
            chat_type=ChatType.PRIVATE,
            title="Media chat",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=42,
            text="Исходная подпись",
            has_media=True,
            media_type="video",
            content_text="Полная расшифровка",
            content_summary="Краткое резюме",
            media_processing_status=MediaProcessingStatus.READY,
            sender_id=1,
            is_outgoing=False,
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()

        response = await auth_client.get(f"/api/v1/chats/{chat.id}/messages/42/content")

        assert response.status_code == 200
        payload = response.json()
        assert payload["text"] == "Исходная подпись"
        assert payload["content_text"] == "Полная расшифровка"
        assert payload["content_summary"] == "Краткое резюме"
        assert payload["media_processing_status"] == "ready"

    async def test_does_not_expose_another_users_content(
        self,
        auth_client,
        db_session,
    ):
        other_chat = TelegramChat(
            user_id=uuid4(),
            telegram_chat_id=98765,
            chat_type=ChatType.PRIVATE,
            title="Private",
        )
        db_session.add(other_chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=other_chat.id,
            telegram_message_id=7,
            text=None,
            has_media=True,
            media_type="document",
            content_text="Secret",
            content_summary="Secret summary",
            media_processing_status=MediaProcessingStatus.READY,
            is_outgoing=False,
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()

        response = await auth_client.get(
            f"/api/v1/chats/{other_chat.id}/messages/7/content"
        )

        assert response.status_code == 404


class TestMessageMediaEndpoints:
    async def test_transcript_paginates_without_truncation(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55001,
            chat_type=ChatType.PRIVATE,
            title="Transcript",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=501,
            has_media=True,
            media_type="voice",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()
        db_session.add_all(
            [
                TranscriptSegment(
                    message_id=message.id,
                    sequence=sequence,
                    start_ms=sequence * 1000,
                    end_ms=(sequence + 1) * 1000,
                    text=f"segment {sequence}",
                )
                for sequence in range(3)
            ]
        )
        await db_session.flush()

        first = await auth_client.get(
            f"/api/v1/chats/{chat.id}/messages/501/transcript",
            params={"limit": 2},
        )
        second = await auth_client.get(
            f"/api/v1/chats/{chat.id}/messages/501/transcript",
            params={"cursor": 2, "limit": 2},
        )

        assert first.status_code == 200
        assert [row["sequence"] for row in first.json()["segments"]] == [0, 1]
        assert first.json()["has_more"] is True
        assert first.json()["next_cursor"] == 2
        assert second.status_code == 200
        assert [row["sequence"] for row in second.json()["segments"]] == [2]
        assert second.json()["has_more"] is False
        assert second.json()["next_cursor"] is None

    async def test_media_get_and_head_return_nginx_internal_headers(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55003,
            chat_type=ChatType.PRIVATE,
            title="Download",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=503,
            has_media=True,
            media_type="document",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()
        cached = SimpleNamespace(
            relative_path="ab/cache/original.pdf",
            sha256="c" * 64,
            file_name="report.pdf",
            mime_type="application/pdf",
        )

        with patch(
            "app.api.v1.chats.get_cached_media_for_download",
            new=AsyncMock(return_value=cached),
        ):
            get_response = await auth_client.get(
                f"/api/v1/chats/{chat.id}/messages/503/media"
            )
            head_response = await auth_client.head(
                f"/api/v1/chats/{chat.id}/messages/503/media"
            )

        for response in (get_response, head_response):
            assert response.status_code == 200
            assert response.headers["x-accel-redirect"].endswith(
                "/ab/cache/original.pdf"
            )
            assert response.headers["etag"] == f'"{"c" * 64}"'
            assert "report.pdf" in response.headers["content-disposition"]
        assert head_response.content == b""

    async def test_media_cache_miss_streams_from_telegram(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        """Nothing staged is no longer a dead end - the original is streamed."""
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55004,
            chat_type=ChatType.PRIVATE,
            title="Cache miss",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=504,
            has_media=True,
            media_type="video",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()

        from app.services.media_stream_service import StreamableMedia

        async def chunks():
            yield b"stream"
            yield b"ed"

        with patch(
            "app.api.v1.chats.open_media_stream",
            new_callable=AsyncMock,
            return_value=(
                StreamableMedia(
                    file_name="clip.mp4", mime_type="video/mp4", size_bytes=8
                ),
                chunks(),
            ),
        ):
            response = await auth_client.get(
                f"/api/v1/chats/{chat.id}/messages/504/media"
            )

        assert response.status_code == 200
        assert response.content == b"streamed"
        assert response.headers["content-type"].startswith("video/mp4")
        assert "clip.mp4" in response.headers["content-disposition"]
        assert response.headers["accept-ranges"] == "bytes"

    async def test_media_stream_reports_a_source_telegram_deleted(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55009,
            chat_type=ChatType.PRIVATE,
            title="Gone",
        )
        db_session.add(chat)
        await db_session.flush()
        db_session.add(
            TelegramMessage(
                chat_id=chat.id,
                telegram_message_id=509,
                has_media=True,
                media_type="photo",
                sent_at=datetime.now(UTC),
            )
        )
        await db_session.flush()

        from app.services.media_stream_service import MediaStreamUnavailable

        with patch(
            "app.api.v1.chats.open_media_stream",
            new_callable=AsyncMock,
            side_effect=MediaStreamUnavailable("Telegram source media was deleted"),
        ):
            response = await auth_client.get(
                f"/api/v1/chats/{chat.id}/messages/509/media"
            )

        assert response.status_code == 404
        assert "deleted" in response.json()["detail"]

    async def test_prepare_is_idempotent_while_dispatch_is_pending(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55002,
            chat_type=ChatType.PRIVATE,
            title="Prepare",
        )
        db_session.add(chat)
        await db_session.flush()
        message = TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=502,
            has_media=True,
            media_type="document",
            media_file_name="large.bin",
            sent_at=datetime.now(UTC),
        )
        db_session.add(message)
        await db_session.flush()

        with patch("app.api.v1.chats.enqueue_media_processing") as enqueue:
            first = await auth_client.post(
                f"/api/v1/chats/{chat.id}/messages/502/prepare"
            )
            second = await auth_client.post(
                f"/api/v1/chats/{chat.id}/messages/502/prepare"
            )

        assert first.status_code == 200
        assert second.status_code == 200
        enqueue.assert_called_once_with([message.id])

    async def test_prepare_reports_deferred_pipeline_without_enqueuing(
        self,
        auth_client,
        db_session,
        test_user,
    ):
        chat = TelegramChat(
            user_id=test_user.id,
            telegram_chat_id=55005,
            chat_type=ChatType.PRIVATE,
            title="Deferred media",
        )
        db_session.add(chat)
        await db_session.flush()
        db_session.add(
            TelegramMessage(
                chat_id=chat.id,
                telegram_message_id=505,
                has_media=True,
                media_type="document",
                sent_at=datetime.now(UTC),
            )
        )
        await db_session.flush()

        with (
            patch(
                "app.api.v1.chats.get_settings",
                return_value=SimpleNamespace(media_pipeline_enabled=False),
            ),
            patch("app.api.v1.chats.enqueue_media_processing") as enqueue,
        ):
            response = await auth_client.post(
                f"/api/v1/chats/{chat.id}/messages/505/prepare"
            )

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "media_pipeline_deferred"
        enqueue.assert_not_called()


class TestRefreshChats:
    async def test_refresh_chats(self, auth_client):
        with patch(
            "app.api.v1.chats.sync_chats",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = await auth_client.post("/api/v1/chats/refresh")
            assert response.status_code == 200
            data = response.json()
            assert data["chats"] == []

    async def test_refresh_chats_expired_session_returns_400(self, auth_client):
        with patch(
            "app.api.v1.chats.sync_chats",
            new_callable=AsyncMock,
            side_effect=TelegramSessionUnauthorizedError(
                "Telegram session expired. Reconnect Telegram and try again."
            ),
        ):
            response = await auth_client.post("/api/v1/chats/refresh")
        assert response.status_code == 400
        assert "Reconnect Telegram" in response.json()["detail"]
