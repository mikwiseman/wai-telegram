from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult
from telegram_wai_mcp import server


@pytest.mark.asyncio
async def test_inbox_uses_one_shared_tool_call_and_preserves_evidence():
    api = AsyncMock()
    api.execute_data_tool.return_value = {
        "checked_at": "2026-09-10T09:00:00+00:00",
        "scope": {"chat_types": ["private"], "active_since": None, "chat_ids": None},
        "listener_active": True,
        "has_more": True,
        "next_cursor": "next-chats",
        "conversations": [
            {
                "chat_id": "chat-a",
                "title": "Colleague",
                "chat_type": "private",
                "unread_count": 0,
                "last_sync_at": "2026-08-01T00:00:00+00:00",
                "sync_recommended": False,
                "latest_known_message_available": True,
                "has_more_messages": True,
                "next_message_cursor": "older-messages",
                "messages": [
                    {
                        "telegram_message_id": 42,
                        "sent_at": "2026-09-10T08:00:00+00:00",
                        "sender_name": "Colleague",
                        "is_outgoing": False,
                        "text": "Can you send the proposal?",
                        "text_truncated": False,
                        "reply_to_message_id": 40,
                        "thread_id": 10,
                        "has_media": True,
                        "media_type": "voice",
                        "media_file_name": None,
                        "media_processing_status": "ready",
                        "content_preview": "By Friday.",
                        "content_truncated": True,
                        "telegram_message_url": "https://t.me/example/42",
                    }
                ],
            }
        ],
    }
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool("get_inbox", {"limit": 20})
    assert not isinstance(result, CallToolResult)
    text = result[0].text
    for expected in (
        "Can you send the proposal?",
        "By Friday.",
        "https://t.me/example/42",
        "chat-a",
        "#42",
        "reply_to=40",
        "thread=10",
        "older-messages",
        "next-chats",
        "truncated",
        "oldest-first",
    ):
        assert expected in text
    assert "STALE" not in text
    api.execute_data_tool.assert_awaited_once_with("get_inbox", {"limit": 20})
    api.refresh_chats.assert_not_awaited()
    api.get_settings.assert_not_awaited()
    api.close.assert_awaited_once()


def test_inbox_warns_on_missing_messages_and_offline_listener():
    result = {
        "checked_at": "now",
        "scope": {},
        "listener_active": False,
        "conversations": [
            {
                "chat_id": "a",
                "title": "Empty",
                "chat_type": "private",
                "sync_recommended": True,
                "latest_known_message_id": 12,
                "messages": [],
            }
        ],
    }
    text = server.format_inbox(result)[0].text
    assert "offline" in text and "sync_chat" in text and "12" in text
