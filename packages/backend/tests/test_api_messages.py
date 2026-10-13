"""Tests for app.api.v1.messages — API endpoint integration tests."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.services.telegram_client import TelegramSessionUnauthorizedError
from tests.factories import TelegramChatFactory


class TestSendMessageEndpoint:
    async def test_send_message_success(self, auth_client, db_session, test_user):
        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = {
            "telegram_message_id": 123,
            "chat_id": str(chat.id),
            "text": "Hello",
        }

        with patch(
            "app.api.v1.messages.send_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{chat.id}/send",
                json={"text": "Hello"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["telegram_message_id"] == 123
        assert data["text"] == "Hello"

    async def test_send_message_chat_not_found(self, auth_client):
        with patch(
            "app.api.v1.messages.send_message",
            new_callable=AsyncMock,
            side_effect=ValueError("Chat not found"),
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/send",
                json={"text": "Hello"},
            )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    async def test_send_message_expired_session_returns_400(self, auth_client):
        with patch(
            "app.api.v1.messages.send_message",
            new_callable=AsyncMock,
            side_effect=TelegramSessionUnauthorizedError(
                "Telegram session expired. Reconnect Telegram and try again."
            ),
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/send",
                json={"text": "Hello"},
            )
        assert response.status_code == 400
        assert "Reconnect Telegram" in response.json()["detail"]

    async def test_send_message_unauthenticated(self, client):
        response = await client.post(
            f"/api/v1/messages/{uuid4()}/send",
            json={"text": "Hello"},
        )
        assert response.status_code == 401


class TestSendFileEndpoint:
    async def test_send_file_success(self, auth_client, db_session, test_user):
        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = {
            "telegram_message_id": 456,
            "chat_id": str(chat.id),
            "file_name": "doc.pdf",
        }

        with patch(
            "app.api.v1.messages.send_file",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{chat.id}/send-file",
                json={"file_url": "https://example.com/doc.pdf"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["telegram_message_id"] == 456
        assert data["file_name"] == "doc.pdf"

    async def test_send_file_invalid_url(self, auth_client):
        with patch(
            "app.api.v1.messages.send_file",
            new_callable=AsyncMock,
            side_effect=ValueError("Unsupported URL scheme"),
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/send-file",
                json={"file_url": "ftp://example.com/file"},
            )
        assert response.status_code == 400

    async def test_send_file_expired_session_returns_400(self, auth_client):
        with patch(
            "app.api.v1.messages.send_file",
            new_callable=AsyncMock,
            side_effect=TelegramSessionUnauthorizedError(
                "Telegram session expired. Reconnect Telegram and try again."
            ),
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/send-file",
                json={"file_url": "https://example.com/file.pdf"},
            )
        assert response.status_code == 400
        assert "Reconnect Telegram" in response.json()["detail"]


class TestDraftFileEndpoint:
    async def test_obsolete_upload_fails_without_connecting_to_telegram(
        self, auth_client
    ):
        with patch(
            "app.services.messaging_service.get_client", new_callable=AsyncMock
        ) as client:
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/draft-file", content=b"image-bytes"
            )
        assert response.status_code == 410
        assert "attachments" in response.json()["detail"]
        client.assert_not_awaited()


class TestReplyMessageEndpoint:
    async def test_reply_success(self, auth_client, db_session, test_user):
        chat = TelegramChatFactory.create(user_id=test_user.id)
        db_session.add(chat)
        await db_session.flush()

        mock_result = {
            "telegram_message_id": 789,
            "chat_id": str(chat.id),
            "text": "Reply text",
        }

        with patch(
            "app.api.v1.messages.reply_to_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{chat.id}/reply",
                json={"telegram_message_id": 100, "text": "Reply text"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["telegram_message_id"] == 789

    async def test_reply_unauthenticated(self, client):
        response = await client.post(
            f"/api/v1/messages/{uuid4()}/reply",
            json={"telegram_message_id": 100, "text": "Reply text"},
        )
        assert response.status_code == 401

    async def test_reply_expired_session_returns_400(self, auth_client):
        with patch(
            "app.api.v1.messages.reply_to_message",
            new_callable=AsyncMock,
            side_effect=TelegramSessionUnauthorizedError(
                "Telegram session expired. Reconnect Telegram and try again."
            ),
        ):
            response = await auth_client.post(
                f"/api/v1/messages/{uuid4()}/reply",
                json={"telegram_message_id": 100, "text": "Reply text"},
            )
        assert response.status_code == 400
        assert "Reconnect Telegram" in response.json()["detail"]
