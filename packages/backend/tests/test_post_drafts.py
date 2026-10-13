"""A Wai post draft preserves the complete post without contacting Telegram."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import quote
from uuid import uuid4

import pytest

from app.core.auth import AuthContext, get_auth_context
from tests.factories import TelegramChatFactory, UserFactory

IMAGE = b"\xff\xd8\xff\xe0draft-photo\xff\xd9"
CAPTION = "[Раньше писал](https://example.com/post)\n\nНовый текст"


async def make_chat(db, user):
    chat = TelegramChatFactory.create(user_id=user.id)
    db.add(chat)
    await db.flush()
    return chat


async def save(client, chat, *, draft_id=None, caption=CAPTION, image=IMAGE):
    draft_id = draft_id or uuid4()
    response = await client.put(
        f"/api/v1/post-drafts/{draft_id}",
        content=image,
        headers={
            "X-Chat-ID": str(chat.id),
            "X-File-Name": quote("График.jpg"),
            "X-Telegram-Caption": quote(caption),
            "Content-Type": "image/jpeg",
        },
    )
    return draft_id, response


async def test_stores_and_reads_exact_post_without_telegram(
    auth_client, db_session, test_user
):
    chat = await make_chat(db_session, test_user)
    with patch(
        "app.services.messaging_service.get_client", new_callable=AsyncMock
    ) as telegram:
        draft_id, response = await save(auth_client, chat)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["storage"] == "wai_telegram"
        assert result["status"] == "draft"
        assert result["sent"] is False
        assert result["synced_to_telegram"] is False
        assert result["caption"] == CAPTION
        assert result["chat_id"] == str(chat.id)
        assert result["sha256"] == hashlib.sha256(IMAGE).hexdigest()
        read = await auth_client.get(f"/api/v1/post-drafts/{draft_id}")
        media = await auth_client.get(f"/api/v1/post-drafts/{draft_id}/media")
    assert read.json() == result
    assert media.content == IMAGE
    assert media.headers["content-type"] == "image/jpeg"
    assert media.headers["cache-control"] == "private, no-store"
    telegram.assert_not_awaited()


async def test_retry_is_idempotent_and_cannot_change_reviewed_post(
    auth_client, db_session, test_user
):
    chat = await make_chat(db_session, test_user)
    draft_id, first = await save(auth_client, chat)
    _, retry = await save(auth_client, chat, draft_id=draft_id)
    assert retry.status_code == 200
    assert retry.json() == first.json()
    _, changed = await save(auth_client, chat, draft_id=draft_id, caption="Different")
    assert changed.status_code == 409
    assert (await auth_client.get(f"/api/v1/post-drafts/{draft_id}")).json()[
        "caption"
    ] == CAPTION


async def test_owner_isolation(auth_client, db_session, test_user):
    other = UserFactory.create(is_active=False)
    db_session.add(other)
    await db_session.flush()
    chat = await make_chat(db_session, other)
    _, forbidden = await save(auth_client, chat)
    assert forbidden.status_code == 404
    mine = await make_chat(db_session, test_user)
    draft_id, saved = await save(auth_client, mine)
    assert saved.status_code == 200
    from app.models.post_draft import PostDraft

    draft = await db_session.get(PostDraft, draft_id)
    draft.user_id = other.id
    await db_session.flush()
    for suffix in ("", "/media"):
        assert (
            await auth_client.get(f"/api/v1/post-drafts/{draft_id}{suffix}")
        ).status_code == 404
    assert (
        await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
    ).status_code == 404


@pytest.mark.parametrize(
    "caption,image",
    [("", IMAGE), ("x" * 1025, IMAGE), ("text", b""), ("text", b"not an image")],
)
async def test_invalid_content_rejected(
    auth_client, db_session, test_user, caption, image
):
    chat = await make_chat(db_session, test_user)
    with patch(
        "app.services.messaging_service.get_client", new_callable=AsyncMock
    ) as telegram:
        _, response = await save(auth_client, chat, caption=caption, image=image)
    assert response.status_code == 400
    telegram.assert_not_awaited()


async def test_scopes_separate_drafting_from_publishing(
    app, client, db_session, test_user
):
    chat = await make_chat(db_session, test_user)
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        test_user, scopes={"read"}
    )
    _, response = await save(client, chat)
    assert response.status_code == 403
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        test_user, scopes={"read", "draft"}
    )
    draft_id, response = await save(client, chat)
    assert response.status_code == 200
    result = await client.post(
        f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
    )
    assert result.status_code == 403


async def test_publish_requires_explicit_approve(auth_client, db_session, test_user):
    chat = await make_chat(db_session, test_user)
    draft_id, _ = await save(auth_client, chat)
    for body in ({}, {"approve": False}):
        result = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json=body
        )
        assert result.status_code == 422


async def test_publish_sends_stored_bytes_once(auth_client, db_session, test_user):
    chat = await make_chat(db_session, test_user)
    draft_id, _ = await save(auth_client, chat)
    seen = {}

    async def send(db, user_id, chat_id, path, caption, target):
        seen.update(
            path=path, image=Path(path).read_bytes(), caption=caption, chat_id=chat_id
        )
        return {
            "telegram_message_id": 456,
            "chat_id": str(chat_id),
            "file_name": path.name,
        }

    with patch(
        "app.services.post_draft_service._send_file_path", side_effect=send
    ) as sender:
        sent = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
        retry = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
    assert sent.status_code == retry.status_code == 200
    assert sent.json()["status"] == "published"
    assert sent.json()["sent"] is True
    assert sent.json()["telegram_message_id"] == 456
    assert retry.json() == sent.json()
    assert (
        seen["image"] == IMAGE
        and seen["caption"] == CAPTION
        and seen["chat_id"] == chat.id
    )
    assert not seen["path"].exists()
    sender.assert_awaited_once()


async def test_uncertain_send_cannot_be_retried(auth_client, db_session, test_user):
    chat = await make_chat(db_session, test_user)
    draft_id, _ = await save(auth_client, chat)
    with patch(
        "app.services.post_draft_service._send_file_path",
        side_effect=ConnectionError("secret body"),
    ) as sender:
        failed = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
        retry = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
    assert failed.status_code == 503
    assert "secret body" not in failed.text
    assert retry.status_code == 409
    state = (await auth_client.get(f"/api/v1/post-drafts/{draft_id}")).json()
    assert state["status"] == "unknown" and state["sent"] is None
    sender.assert_awaited_once()


async def test_inflight_publication_cannot_send_again(
    auth_client, db_session, test_user
):
    from app.models.post_draft import PostDraft

    chat = await make_chat(db_session, test_user)
    draft_id, _ = await save(auth_client, chat)
    draft = await db_session.get(PostDraft, draft_id)
    draft.status = "publishing"
    await db_session.flush()
    with patch(
        "app.services.post_draft_service._send_file_path", new_callable=AsyncMock
    ) as sender:
        result = await auth_client.post(
            f"/api/v1/post-drafts/{draft_id}/publish", json={"approve": True}
        )
    assert result.status_code == 409
    sender.assert_not_awaited()


async def test_stream_size_is_bounded(auth_client, db_session, test_user):
    chat = await make_chat(db_session, test_user)
    with patch("app.services.post_draft_service.MAX_IMAGE_BYTES", 4):
        _, result = await save(auth_client, chat)
    assert result.status_code == 413


async def test_media_requires_read_access(
    app, client, auth_client, db_session, test_user
):
    chat = await make_chat(db_session, test_user)
    draft_id, _ = await save(auth_client, chat)
    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        test_user, scopes={"draft"}
    )
    for suffix in ("", "/media"):
        assert (
            await client.get(f"/api/v1/post-drafts/{draft_id}{suffix}")
        ).status_code == 403
