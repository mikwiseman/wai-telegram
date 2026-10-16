"""File drafts retain actual bytes without any Telegram side effect."""

import hashlib
import base64
import re
from pathlib import Path
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from urllib.parse import quote
from uuid import uuid4

import pytest

from app.core.auth import AuthContext, get_auth_context
from tests.factories import TelegramChatFactory, UserFactory

DOCUMENT = b"PK\x03\x04test-docx-content"
NAME = "Договор.docx"
TEXT = "Евгений, привет!\n\nПравки в приложении"


def test_nginx_accepts_encoded_file_limit_and_keeps_download_tokens_private():
    config = (Path(__file__).parents[3] / "nginx/telegram-ai.conf").read_text()
    mcp = config.split("location = /mcp {", 1)[1].split("}", 1)[0]
    limit_mib = int(re.search(r"client_max_body_size (\d+)m;", mcp)[1])
    encoded_limit = 4 * ((10 * 1024 * 1024 + 2) // 3)
    assert limit_mib * 1024 * 1024 > encoded_limit + 2048
    download = config.split("location ~ ^/api/v1/post-drafts/", 1)[1].split("}", 1)[0]
    assert "/download$" in download and "access_log off;" in download


async def chat_for(db, user):
    chat = TelegramChatFactory.create(user_id=user.id)
    db.add(chat)
    await db.flush()
    return chat


async def put_file(client, chat, draft_id, content=DOCUMENT):
    return await client.put(
        f"/api/v1/post-drafts/{draft_id}",
        content=content,
        headers={
            "X-Chat-ID": str(chat.id),
            "X-File-Name": quote(NAME),
            "X-Telegram-Caption": quote(TEXT),
            "X-Draft-Media-Type": "file",
        },
    )


async def test_file_round_trip_via_tools_has_exact_bytes_and_no_telegram(
    auth_client, db_session, test_user
):
    chat = await chat_for(db_session, test_user)
    draft_id = uuid4()
    with patch(
        "app.services.messaging_service.get_client", new_callable=AsyncMock
    ) as telegram:
        saved = await put_file(auth_client, chat, draft_id)
        assert saved.status_code == 200, saved.text
        read = await auth_client.post(
            "/api/v1/tools/get_file_draft",
            json={"arguments": {"draft_id": str(draft_id)}},
        )
        assert read.status_code == 200, read.text
        info = read.json()
        assert info["file_name"] == NAME
        assert info["caption"] == TEXT
        assert info["file_size"] == len(DOCUMENT)
        assert info["sha256"] == hashlib.sha256(DOCUMENT).hexdigest()
        assert (
            info["mime_type"]
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert info["has_media"] is True and info["sent"] is False
        assert info["storage"] == "wai_telegram" and info["synced_to_telegram"] is False
        download = await auth_client.get(info["media_download_url"])
        assert download.status_code == 200
        assert download.content == DOCUMENT
        assert "attachment;" in download.headers["content-disposition"]
    telegram.assert_not_awaited()


async def test_save_file_url_mcp_handler_persists_and_retries_idempotently(
    auth_client, db_session, test_user
):
    chat = await chat_for(db_session, test_user)
    arguments = {
        "draft_id": str(uuid4()),
        "chat_id": str(chat.id),
        "text": TEXT,
        "file_url": "https://example.com/contract.docx",
        "file_name": NAME,
    }

    async def download(url, path, *, max_bytes):
        assert url == arguments["file_url"]
        assert max_bytes == 10 * 1024 * 1024
        path.write_bytes(DOCUMENT)

    with (
        patch(
            "app.services.post_draft_service._download_url_to_path",
            side_effect=download,
        ),
        patch(
            "app.services.messaging_service.get_client", new_callable=AsyncMock
        ) as telegram,
    ):
        for _ in range(2):
            saved = await auth_client.post(
                "/api/v1/tools/save_file_draft", json={"arguments": arguments}
            )
            assert saved.status_code == 200, saved.text
            assert saved.json()["sent"] is False
        changed = await auth_client.post(
            "/api/v1/tools/save_file_draft",
            json={"arguments": {**arguments, "text": "Other text"}},
        )
        assert changed.status_code == 400
        listing = await auth_client.post(
            "/api/v1/tools/list_file_drafts",
            json={"arguments": {"chat_id": str(chat.id)}},
        )
    assert [d["id"] for d in listing.json()["drafts"]] == [arguments["draft_id"]]
    telegram.assert_not_awaited()


async def test_file_draft_download_token_is_bound_to_owner_and_document(
    auth_client, client, db_session, test_user
):
    from app.services.post_draft_service import create_draft_download_token

    chat = await chat_for(db_session, test_user)
    draft_id = uuid4()
    assert (await put_file(auth_client, chat, draft_id)).status_code == 200
    token = create_draft_download_token(test_user.id, draft_id)
    response = await client.get(
        f"/api/v1/post-drafts/{draft_id}/download", params={"token": token}
    )
    assert response.content == DOCUMENT
    assert (
        await client.get(
            f"/api/v1/post-drafts/{uuid4()}/download", params={"token": token}
        )
    ).status_code == 401
    assert (
        await client.get(
            f"/api/v1/post-drafts/{draft_id}/download", params={"token": "invalid"}
        )
    ).status_code == 401
    expired = create_draft_download_token(
        test_user.id, draft_id, expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    assert (
        await client.get(
            f"/api/v1/post-drafts/{draft_id}/download", params={"token": expired}
        )
    ).status_code == 401
    other = UserFactory.create(is_active=False)
    db_session.add(other)
    await db_session.flush()
    other_token = create_draft_download_token(other.id, draft_id)
    assert (
        await client.get(
            f"/api/v1/post-drafts/{draft_id}/download", params={"token": other_token}
        )
    ).status_code == 404


async def test_inline_file_mcp_round_trip_and_invalid_sources(
    auth_client, db_session, test_user
):
    chat = await chat_for(db_session, test_user)
    args = {
        "draft_id": str(uuid4()),
        "chat_id": str(chat.id),
        "text": TEXT,
        "file_name": NAME,
        "file_base64": base64.b64encode(DOCUMENT).decode(),
    }
    with patch(
        "app.services.messaging_service.get_client", new_callable=AsyncMock
    ) as telegram:
        result = await auth_client.post(
            "/api/v1/tools/save_file_draft", json={"arguments": args}
        )
        assert result.status_code == 200, result.text
        assert result.json()["sha256"] == hashlib.sha256(DOCUMENT).hexdigest()
        assert (
            await auth_client.get(result.json()["media_download_url"])
        ).content == DOCUMENT
        for wrong in (
            {**args, "file_base64": "%%%"},
            {**args, "file_url": "https://example.com/docx"},
        ):
            bad = await auth_client.post(
                "/api/v1/tools/save_file_draft", json={"arguments": wrong}
            )
            assert bad.status_code == 400
    telegram.assert_not_awaited()


async def test_file_drafts_are_owner_scoped_and_draft_access_is_required(
    app, client, auth_client, db_session, test_user
):
    chat = await chat_for(db_session, test_user)
    draft_id = uuid4()
    assert (await put_file(auth_client, chat, draft_id)).status_code == 200
    test_user.is_active = False
    await db_session.flush()
    other = UserFactory.create(is_active=True)
    db_session.add(other)
    await db_session.flush()
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        other, scopes={"read"}
    )
    read = await client.post(
        "/api/v1/tools/get_file_draft", json={"arguments": {"draft_id": str(draft_id)}}
    )
    assert read.status_code == 400
    listing = await client.post(
        "/api/v1/tools/list_file_drafts", json={"arguments": {}}
    )
    assert listing.json()["drafts"] == []
    assert (await put_file(client, chat, uuid4())).status_code == 403
    denied = await client.post(
        "/api/v1/tools/save_file_draft",
        json={
            "arguments": {
                "draft_id": str(uuid4()),
                "chat_id": str(chat.id),
                "text": TEXT,
                "file_url": "https://example.com/a.docx",
                "file_name": NAME,
            }
        },
    )
    assert denied.status_code == 403


@pytest.mark.parametrize(
    "content", [b"", b"x" * (10 * 1024 * 1024 + 1)], ids=["empty", "oversize"]
)
async def test_empty_or_oversize_file_rejected(
    auth_client, db_session, test_user, content
):
    chat = await chat_for(db_session, test_user)
    response = await put_file(auth_client, chat, uuid4(), content)
    assert response.status_code in {400, 413}


async def test_mcp_send_requires_write_and_explicit_approval_and_sends_once(
    app, client, auth_client, db_session, test_user
):
    chat = await chat_for(db_session, test_user)
    draft_id = uuid4()
    assert (await put_file(auth_client, chat, draft_id)).status_code == 200
    arguments = {"draft_id": str(draft_id)}
    with patch(
        "app.services.post_draft_service._send_file_path", new_callable=AsyncMock
    ) as sender:
        for approved in (None, False, "true"):
            response = await auth_client.post(
                "/api/v1/tools/send_file_draft",
                json={"arguments": {**arguments, "approve": approved}},
            )
            assert response.status_code == 400
        app.dependency_overrides[get_auth_context] = lambda: AuthContext(
            test_user, scopes={"read", "draft"}
        )
        denied = await client.post(
            "/api/v1/tools/send_file_draft",
            json={"arguments": {**arguments, "approve": True}},
        )
        assert denied.status_code == 403
        sender.assert_not_awaited()

        app.dependency_overrides[get_auth_context] = lambda: AuthContext(
            test_user, scopes={"read", "write"}
        )

        async def send(db, user_id, chat_id, path, caption, target, *, force_document):
            assert force_document is True
            assert (
                chat_id == chat.id and path.read_bytes() == DOCUMENT and caption == TEXT
            )
            return {"telegram_message_id": 123}

        sender.side_effect = send
        for _ in range(2):
            response = await client.post(
                "/api/v1/tools/send_file_draft",
                json={"arguments": {**arguments, "approve": True}},
            )
            assert response.status_code == 200, response.text
            assert response.json()["sent"] is True
            assert response.json()["telegram_message_id"] == 123
        sender.assert_awaited_once()
