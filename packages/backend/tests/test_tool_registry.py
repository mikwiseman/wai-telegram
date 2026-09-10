from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.core.security import compute_api_key_prefix, get_key_hint, hash_api_key
from app.models.api_key import ApiKey
import pytest
from sqlalchemy import select

from app.models.chat import ChatType, TelegramChat
from app.models.media import (
    MediaObject,
    MediaObjectStatus,
    MediaStage,
    TranscriptSegment,
)
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.services.tool_registry import ToolInputError, execute_data_tool


async def _seed(db_session, test_user):
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=-1001234567890,
        chat_type=ChatType.CHANNEL,
        title="Tool chat",
        username="tool_chat",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=42,
        text="Caption",
        has_media=True,
        media_type="audio",
        media_file_name="meeting.ogg",
        media_mime_type="audio/ogg",
        media_file_size=120,
        content_text="abcdefghij",
        content_summary="Summary",
        media_processing_status=MediaProcessingStatus.READY,
        hidden_urls=["https://hidden.example"],
        reply_to_message_id=41,
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()
    db_session.add(
        MediaObject(
            user_id=test_user.id,
            message_id=message.id,
            cache_key="1" * 64,
            relative_path="11/cache/original.ogg",
            file_name="meeting.ogg",
            mime_type="audio/ogg",
            size_bytes=120,
            sha256="2" * 64,
            byte_offset=120,
            status=MediaObjectStatus.READY,
            stage=MediaStage.COMPLETE,
        )
    )
    db_session.add_all(
        [
            TranscriptSegment(
                message_id=message.id,
                sequence=index,
                start_ms=index * 1000,
                end_ms=(index + 1) * 1000,
                speaker="A",
                confidence=0.9,
                language="ru",
                text=f"segment {index}",
            )
            for index in range(3)
        ]
    )
    await db_session.flush()
    return chat, message


async def test_shared_tools_expose_metadata_content_segments_and_download(
    db_session, test_user
):
    chat, message = await _seed(db_session, test_user)
    locator = {"chat_id": str(chat.id), "telegram_message_id": 42}

    metadata = await execute_data_tool(db_session, test_user.id, "get_message", locator)
    content = await execute_data_tool(
        db_session,
        test_user.id,
        "get_message_content",
        {**locator, "cursor": 0, "limit_chars": 1000},
    )
    segments = await execute_data_tool(
        db_session,
        test_user.id,
        "get_transcript_segments",
        {**locator, "limit": 2},
    )
    download = await execute_data_tool(
        db_session, test_user.id, "download_media", locator
    )

    assert metadata["id"] == str(message.id)
    assert metadata["hidden_urls"] == ["https://hidden.example"]
    assert metadata["reply_to_message_id"] == 41
    assert content["content_text"] == "abcdefghij"
    assert content["has_more"] is False
    assert len(segments["segments"]) == 2
    assert segments["has_more"] is True
    assert segments["next_cursor"] == 2
    assert download["media_sha256"] == "2" * 64
    assert "token=" in download["media_download_url"]
    assert download["telegram_message_url"] == "https://t.me/tool_chat/42"


async def test_tools_api_lists_all_shared_data_tools(auth_client):
    response = await auth_client.get("/api/v1/tools")

    assert response.status_code == 200
    tools = response.json()["tools"]
    names = {tool["name"] for tool in tools}
    assert names == {
        "get_inbox",
        "clear_draft",
        "search_messages",
        "get_files",
        "get_message",
        "list_drafts",
        "save_draft",
        "prepare_media",
        "download_media",
        "get_message_content",
        "get_transcript_segments",
        "get_data_status",
    }
    search_tool = next(tool for tool in tools if tool["name"] == "search_messages")
    properties = search_tool["parameters"]["properties"]
    assert properties["mode"]["enum"] == ["hybrid", "exact"]
    assert properties["chat_types"]["items"]["enum"] == [
        "private",
        "group",
        "supergroup",
        "channel",
    ]


async def test_save_draft_tool_preserves_text_and_uses_owner_chat_id(
    db_session, test_user
):
    chat_id = uuid4()
    text = "  Черновик\nс форматированием  "
    expected = {
        "chat_id": str(chat_id),
        "text": text,
        "saved": True,
        "sent": False,
        "replaces_existing_draft": True,
    }

    with patch(
        "app.services.tool_registry.save_telegram_draft",
        new_callable=AsyncMock,
        return_value=expected,
    ) as save:
        result = await execute_data_tool(
            db_session,
            test_user.id,
            "save_draft",
            {"chat_id": str(chat_id), "text": text},
        )

    assert result == expected
    save.assert_awaited_once_with(db_session, test_user.id, chat_id, text)


async def test_clear_draft_uses_draft_only_handler(db_session, test_user):
    chat_id = uuid4()
    expected = {
        "chat_id": str(chat_id),
        "cleared": True,
        "saved": True,
        "sent": False,
    }

    with patch(
        "app.services.tool_registry.clear_telegram_draft",
        new_callable=AsyncMock,
        return_value=expected,
    ) as clear:
        result = await execute_data_tool(
            db_session,
            test_user.id,
            "clear_draft",
            {"chat_id": str(chat_id)},
        )

    assert result == expected
    clear.assert_awaited_once_with(db_session, test_user.id, chat_id)


async def test_shared_registry_rejects_a_deactivated_user(db_session, test_user):
    test_user.is_active = False
    await db_session.flush()

    with pytest.raises(ToolInputError, match="inactive"):
        await execute_data_tool(db_session, test_user.id, "get_data_status", {})


async def test_prepare_media_tool_does_not_duplicate_pending_dispatch(
    db_session,
    test_user,
):
    chat, message = await _seed(db_session, test_user)
    media_object = (
        await db_session.execute(
            select(MediaObject).where(MediaObject.message_id == message.id)
        )
    ).scalar_one()
    message.media_processing_status = None
    media_object.status = MediaObjectStatus.FAILED
    media_object.relative_path = None
    media_object.sha256 = None
    locator = {"chat_id": str(chat.id), "telegram_message_id": 42}

    with patch("app.tasks.media_tasks.enqueue_media_processing") as enqueue:
        first = await execute_data_tool(
            db_session,
            test_user.id,
            "prepare_media",
            locator,
        )
        second = await execute_data_tool(
            db_session,
            test_user.id,
            "prepare_media",
            locator,
        )

    assert first["enqueued"] is True
    assert second["enqueued"] is False
    enqueue.assert_called_once_with([message.id])


async def test_prepare_media_tool_reports_deferred_pipeline(
    db_session,
    test_user,
):
    chat, _message = await _seed(db_session, test_user)
    locator = {"chat_id": str(chat.id), "telegram_message_id": 42}

    with (
        patch(
            "app.services.tool_registry.settings",
            SimpleNamespace(media_pipeline_enabled=False),
        ),
        patch("app.tasks.media_tasks.enqueue_media_processing") as enqueue,
    ):
        result = await execute_data_tool(
            db_session,
            test_user.id,
            "prepare_media",
            locator,
        )

    assert result == {
        "message_id": str(_message.id),
        "status": "unavailable",
        "stage": "deferred",
        "enqueued": False,
        "error_code": "media_pipeline_deferred",
        "error_detail": "Durable media processing is deferred until storage is attached",
        "retry_after": None,
        "media_download_url": None,
        "telegram_message_url": "https://t.me/tool_chat/42",
        "next_action": "Retry after the durable media pipeline is enabled",
    }
    enqueue.assert_not_called()


async def test_read_only_api_key_can_start_media_processing(
    client, db_session, test_user
):
    """Media processing pulls from Telegram into our own cache and enqueues our
    own workers — the same shape as sync, which is read-level. Nothing reaches
    the other side. Drafts do, so they stay gated."""
    raw_key = "wai_read_only_media_key_abcdefghijklmnopqrstuvwxyz"
    db_session.add(
        ApiKey(
            user_id=test_user.id,
            name="Read only",
            key_hash=hash_api_key(raw_key),
            key_prefix=compute_api_key_prefix(raw_key),
            key_hint=get_key_hint(raw_key),
            scopes="read",
            is_active=True,
        )
    )
    await db_session.flush()
    headers = {"Authorization": f"Bearer {raw_key}"}

    discovery = await client.get("/api/v1/tools", headers=headers)
    mutation = await client.post(
        "/api/v1/tools/prepare_media",
        headers=headers,
        json={"arguments": {}},
    )

    assert discovery.status_code == 200
    names = {tool["name"] for tool in discovery.json()["tools"]}
    assert "prepare_media" in names
    assert "list_drafts" in names
    assert "save_draft" not in names
    assert "clear_draft" not in names
    assert mutation.status_code != 403


async def test_read_only_api_key_cannot_save_draft(client, db_session, test_user):
    raw_key = "wai_read_only_draft_key_abcdefghijklmnopqrstuvwxyz"
    db_session.add(
        ApiKey(
            user_id=test_user.id,
            name="Read only",
            key_hash=hash_api_key(raw_key),
            key_prefix=compute_api_key_prefix(raw_key),
            key_hint=get_key_hint(raw_key),
            scopes="read",
            is_active=True,
        )
    )
    await db_session.flush()

    with patch(
        "app.services.tool_registry.save_telegram_draft",
        new_callable=AsyncMock,
    ) as save:
        response = await client.post(
            "/api/v1/tools/save_draft",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "arguments": {
                    "chat_id": str(uuid4()),
                    "text": "Must not be saved",
                }
            },
        )

    assert response.status_code == 403
    save.assert_not_awaited()


async def test_jwt_can_save_draft_through_shared_tools_api(
    auth_client, db_session, test_user
):
    chat_id = uuid4()
    text = "Черновик 🌱\nhttps://example.com"
    expected = {
        "chat_id": str(chat_id),
        "text": text,
        "saved": True,
        "sent": False,
        "replaces_existing_draft": True,
    }

    with patch(
        "app.services.tool_registry.save_telegram_draft",
        new_callable=AsyncMock,
        return_value=expected,
    ) as save:
        response = await auth_client.post(
            "/api/v1/tools/save_draft",
            json={"arguments": {"chat_id": str(chat_id), "text": text}},
        )

    assert response.status_code == 200
    assert response.json() == expected
    save.assert_awaited_once_with(db_session, test_user.id, chat_id, text)


async def test_write_scoped_api_key_can_save_draft(client, db_session, test_user):
    raw_key = "wai_write_draft_key_abcdefghijklmnopqrstuvwxyz"
    chat_id = uuid4()
    text = "Production MCP draft"
    expected = {
        "chat_id": str(chat_id),
        "text": text,
        "saved": True,
        "sent": False,
        "replaces_existing_draft": True,
    }
    db_session.add(
        ApiKey(
            user_id=test_user.id,
            name="MCP write",
            key_hash=hash_api_key(raw_key),
            key_prefix=compute_api_key_prefix(raw_key),
            key_hint=get_key_hint(raw_key),
            scopes="read,write",
            is_active=True,
        )
    )
    await db_session.flush()
    headers = {"Authorization": f"Bearer {raw_key}"}

    with patch(
        "app.services.tool_registry.save_telegram_draft",
        new_callable=AsyncMock,
        return_value=expected,
    ) as save:
        discovery = await client.get("/api/v1/tools", headers=headers)
        response = await client.post(
            "/api/v1/tools/save_draft",
            headers=headers,
            json={"arguments": {"chat_id": str(chat_id), "text": text}},
        )

    assert discovery.status_code == 200
    assert "save_draft" in {tool["name"] for tool in discovery.json()["tools"]}
    assert response.status_code == 200
    assert response.json() == expected
    save.assert_awaited_once_with(db_session, test_user.id, chat_id, text)


async def test_save_draft_api_rejects_blank_text(auth_client):
    response = await auth_client.post(
        "/api/v1/tools/save_draft",
        json={
            "arguments": {
                "chat_id": str(uuid4()),
                "text": " \n\t ",
            }
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "text must be a non-empty string"


async def test_draft_scope_allows_drafting_but_not_sending(
    client, db_session, test_user
):
    """The hiring agent must prepare a reply for a person to look at, and must not
    be able to send it. 'draft' buys exactly that and nothing more."""
    raw_key = "wai_draft_only_key_abcdefghijklmnopqrstuvwxyz"
    db_session.add(
        ApiKey(
            user_id=test_user.id,
            name="Draft only",
            key_hash=hash_api_key(raw_key),
            key_prefix=compute_api_key_prefix(raw_key),
            key_hint=get_key_hint(raw_key),
            scopes="read,draft",
            is_active=True,
        )
    )
    await db_session.flush()
    headers = {"Authorization": f"Bearer {raw_key}"}

    discovery = await client.get("/api/v1/tools", headers=headers)
    assert discovery.status_code == 200
    names = {tool["name"] for tool in discovery.json()["tools"]}
    assert "list_drafts" in names
    assert "save_draft" in names
    assert "clear_draft" in names

    drafted = await client.post(
        "/api/v1/tools/save_draft", headers=headers, json={"arguments": {}}
    )
    assert drafted.status_code != 403

    sent = await client.post(
        f"/api/v1/messages/{uuid4()}/send",
        headers=headers,
        json={"text": "hello"},
    )
    assert sent.status_code == 403
    assert sent.json()["detail"] == "API key lacks 'write' permission"


async def test_write_scope_still_covers_drafting(client, db_session, test_user):
    """Existing keys were issued before 'draft' existed; write keeps covering it."""
    raw_key = "wai_write_covers_draft_abcdefghijklmnopqrstuvw"
    db_session.add(
        ApiKey(
            user_id=test_user.id,
            name="Write",
            key_hash=hash_api_key(raw_key),
            key_prefix=compute_api_key_prefix(raw_key),
            key_hint=get_key_hint(raw_key),
            scopes="read,write",
            is_active=True,
        )
    )
    await db_session.flush()
    headers = {"Authorization": f"Bearer {raw_key}"}

    discovery = await client.get("/api/v1/tools", headers=headers)
    assert "save_draft" in {tool["name"] for tool in discovery.json()["tools"]}

    drafted = await client.post(
        "/api/v1/tools/save_draft", headers=headers, json={"arguments": {}}
    )
    assert drafted.status_code != 403


async def test_prepare_media_wakes_video_stuck_in_extraction_retries(
    db_session,
    test_user,
):
    chat, message = await _seed(db_session, test_user)
    media_object = (
        await db_session.execute(
            select(MediaObject).where(MediaObject.message_id == message.id)
        )
    ).scalar_one()
    message.media_type = "video"
    message.media_mime_type = "video/mp4"
    message.media_processing_status = MediaProcessingStatus.FAILED
    media_object.status = MediaObjectStatus.RETRY_WAIT
    media_object.retry_after = datetime.now(UTC)
    media_object.error_code = "document_extraction_error"
    media_object.error_detail = "Document extraction failed: empty_document"
    await db_session.flush()
    locator = {"chat_id": str(chat.id), "telegram_message_id": 42}

    with (
        patch(
            "app.services.media_cache_service.MEDIA_TRANSCRIPTION_TYPES",
            frozenset({"voice", "video_note"}),
        ),
        patch("app.tasks.media_tasks.enqueue_media_processing") as enqueue,
    ):
        result = await execute_data_tool(
            db_session,
            test_user.id,
            "prepare_media",
            locator,
        )

    assert result["enqueued"] is True
    assert media_object.transcription_requested_at is not None
    assert media_object.retry_after is None
    enqueue.assert_called_once_with([message.id])


async def test_prepare_media_requests_transcription_for_skipped_video(
    db_session,
    test_user,
):
    chat, message = await _seed(db_session, test_user)
    media_object = (
        await db_session.execute(
            select(MediaObject).where(MediaObject.message_id == message.id)
        )
    ).scalar_one()
    message.media_type = "video"
    message.media_mime_type = "video/mp4"
    message.content_model = "download-only"
    message.media_processing_status = MediaProcessingStatus.READY
    media_object.status = MediaObjectStatus.READY_DOWNLOAD_ONLY
    await db_session.flush()
    locator = {"chat_id": str(chat.id), "telegram_message_id": 42}

    with (
        patch(
            "app.services.media_cache_service.MEDIA_TRANSCRIPTION_TYPES",
            frozenset({"voice", "video_note"}),
        ),
        patch("app.tasks.media_tasks.enqueue_media_processing") as enqueue,
    ):
        result = await execute_data_tool(
            db_session,
            test_user.id,
            "prepare_media",
            locator,
        )

    assert result["enqueued"] is True
    assert media_object.transcription_requested_at is not None
    assert message.media_processing_status == MediaProcessingStatus.PENDING
    enqueue.assert_called_once_with([message.id])
