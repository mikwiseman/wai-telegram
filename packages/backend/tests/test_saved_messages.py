"""Saved Messages must target the authenticated session, even without a synced chat."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from telethon.tl.types import InputPeerSelf

from app.services import messaging_service as messaging


async def test_saved_text_needs_no_chat_row_and_keeps_literal_text():
    db = AsyncMock()
    client = AsyncMock()
    client.send_message.return_value = SimpleNamespace(id=123)
    text = "**Как есть** 🌱 https://example.com"
    with patch.object(messaging, "get_client", return_value=client):
        result = await messaging.send_message(db, uuid4(), "me", text)
    peer, sent_text = client.send_message.await_args.args
    assert isinstance(peer, InputPeerSelf)
    assert sent_text == text
    assert client.send_message.await_args.kwargs["parse_mode"] is None
    db.execute.assert_not_awaited()
    client.disconnect.assert_awaited_once()
    assert result["chat_id"] == "me"
    assert result["telegram_message_id"] == 123


async def test_saved_text_route_accepts_me(auth_client):
    with patch("app.api.v1.messages.send_message", new_callable=AsyncMock) as send:
        send.return_value = {"chat_id": "me", "telegram_message_id": 123, "text": "hi"}
        response = await auth_client.post(
            "/api/v1/messages/me/send", json={"text": "hi"}
        )
    assert response.status_code == 200
    assert send.await_args.args[2] == "me"


async def test_saved_url_route_accepts_me(auth_client):
    with patch("app.api.v1.messages.send_file", new_callable=AsyncMock) as send:
        send.return_value = {
            "chat_id": "me",
            "telegram_message_id": 124,
            "file_name": "a.pdf",
        }
        response = await auth_client.post(
            "/api/v1/messages/me/send-file",
            json={"file_url": "https://example.com/a.pdf", "file_name": "a.pdf"},
        )
    assert response.status_code == 200
    assert send.await_args.args[2] == "me"


async def test_saved_upload_preserves_bytes_name_and_cleans_temp(auth_client):
    data = b"original\x00binary\xffbytes"
    client = AsyncMock()
    observed = {}

    async def send_file(peer, path, **kwargs):
        assert isinstance(peer, InputPeerSelf)
        observed.update(path=Path(path), data=Path(path).read_bytes(), **kwargs)
        return SimpleNamespace(id=125)

    client.send_file.side_effect = send_file
    with patch.object(messaging, "get_client", return_value=client):
        response = await auth_client.post(
            "/api/v1/messages/me/upload",
            content=data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-File-Name": quote("Отчёт.zip"),
                "X-Telegram-Caption": quote("**Без изменения**"),
            },
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["chat_id"] == "me"
    assert result["telegram_message_id"] == 125
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["file_size"] == len(data)
    assert observed["data"] == data
    assert observed["file_name"] == "Отчёт.zip"
    assert observed["caption"] == "**Без изменения**"
    assert observed["force_document"] is True
    assert observed["parse_mode"] is None
    assert not observed["path"].exists()
    client.disconnect.assert_awaited_once()


async def test_saved_upload_empty_rejected_before_telegram(auth_client):
    with patch.object(messaging, "get_client", new_callable=AsyncMock) as get_client:
        response = await auth_client.post(
            "/api/v1/messages/me/upload",
            content=b"",
            headers={"X-File-Name": "empty.txt"},
        )
    assert response.status_code == 400
    get_client.assert_not_awaited()


@pytest.mark.parametrize("scopes", [set(), {"read"}, {"read", "draft"}])
async def test_saved_upload_requires_write_scope(app, client, test_user, scopes):
    from app.core.auth import AuthContext, get_auth_context

    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        test_user, scopes=scopes
    )
    with patch.object(messaging, "get_client", new_callable=AsyncMock) as get_client:
        response = await client.post(
            "/api/v1/messages/me/upload",
            content=b"file",
            headers={"X-File-Name": "a.txt"},
        )
    assert response.status_code == 403
    get_client.assert_not_awaited()


async def test_saved_upload_has_no_arbitrary_recipient(auth_client):
    response = await auth_client.post(
        f"/api/v1/messages/{uuid4()}/upload",
        content=b"file",
        headers={"X-File-Name": "a.txt"},
    )
    assert response.status_code == 404


async def test_saved_url_file_uses_self_peer_without_a_chat_row():
    from tests.test_service_messaging import _FakeHTTPClient

    db = AsyncMock()
    client = AsyncMock()
    client.send_file.return_value = SimpleNamespace(id=126)
    with (
        patch.object(messaging, "get_client", return_value=client),
        patch.object(messaging, "_validate_url"),
        patch.object(
            messaging.httpx, "AsyncClient", return_value=_FakeHTTPClient([b"pdf"])
        ),
    ):
        result = await messaging.send_file(
            db, uuid4(), "me", "https://example.com/doc.pdf"
        )
    assert isinstance(client.send_file.await_args.args[0], InputPeerSelf)
    assert client.send_file.await_args.kwargs["force_document"] is True
    assert result["file_name"] == "doc.pdf"
    db.execute.assert_not_awaited()
    client.disconnect.assert_awaited_once()


async def test_saved_upload_cleans_temp_after_telegram_failure(auth_client):
    client = AsyncMock()
    paths = []

    async def failed_send(peer, path, **kwargs):
        paths.append(Path(path))
        raise ConnectionError("interrupted")

    client.send_file.side_effect = failed_send
    with patch.object(messaging, "get_client", return_value=client):
        response = await auth_client.post(
            "/api/v1/messages/me/upload",
            content=b"data",
            headers={"X-File-Name": "a.txt"},
        )
    assert response.status_code == 400
    assert paths and not paths[0].exists()
    client.disconnect.assert_awaited_once()
