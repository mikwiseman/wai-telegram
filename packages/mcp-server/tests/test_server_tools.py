import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult
from telegram_wai_mcp import server


def _registry_client():
    client = AsyncMock()
    names = (
        "get_files",
        "search_messages",
        "get_message",
        "list_drafts",
        "save_draft",
        "clear_draft",
        "prepare_media",
        "download_media",
        "get_message_content",
        "get_transcript_segments",
        "get_data_status",
    )
    client.list_data_tools.return_value = {
        "tools": [
            {
                "name": name,
                "description": f"Shared {name}",
                "parameters": {
                    "type": "object",
                    "properties": (
                        {
                            "query": {"type": "string"},
                            "chat_ids": {
                                "type": "array",
                                "items": {"type": "string", "format": "uuid"},
                            },
                        }
                        if name == "search_messages"
                        else {}
                    ),
                },
            }
            for name in names
        ]
    }
    return client


class TestToolList:
    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_tools(self):
        client = _registry_client()
        with patch("telegram_wai_mcp.server.get_client", return_value=client):
            tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        expected_tools = {
            "get_data_status",
            "search_messages",
            "get_message",
            "list_drafts",
            "save_draft",
            "clear_draft",
            "prepare_media",
            "download_media",
            "search_chats",
            "find_chats",
            "list_chats",
            "refresh_chats",
            "get_chat_messages",
            "get_links",
            "get_message_content",
            "get_transcript_segments",
            "sync_chat",
            "get_sync_status",
            "get_daily_digest",
        }
        assert expected_tools.issubset(tool_names)

        search_tool = next(tool for tool in tools if tool.name == "search_messages")
        assert search_tool.inputSchema["properties"]["chat_ids"]["type"] == "array"
        assert "chat_id" not in search_tool.inputSchema["properties"]
        links_tool = next(tool for tool in tools if tool.name == "get_links")
        assert links_tool.inputSchema["required"] == ["chat_id"]
        assert links_tool.inputSchema["properties"]["max_pages"]["maximum"] == 10
        list_chats_tool = next(tool for tool in tools if tool.name == "list_chats")
        assert list_chats_tool.inputSchema["properties"]["unread_only"]["type"] == "boolean"
        client.list_data_tools.assert_awaited_once()
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_message_content_returns_summary_and_full_transcript(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "telegram_message_id": 42,
            "text": "Исходная подпись",
            "media_type": "video",
            "media_file_name": "meeting.mp4",
            "media_processing_status": "ready",
            "content_summary": "Обсудили сроки и ответственных.",
            "content_text": "Полная транскрипция встречи без сокращений.",
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "get_message_content",
                {"chat_id": "chat-1", "telegram_message_id": 42},
            )

        text = result[0].text
        assert "Обсудили сроки и ответственных." in text
        assert "Полная транскрипция встречи без сокращений." in text
        assert "Исходная подпись" in text
        mock_api.execute_data_tool.assert_awaited_once_with(
            "get_message_content",
            {"chat_id": "chat-1", "telegram_message_id": 42},
        )
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_each_tool_has_description(self):
        with patch("telegram_wai_mcp.server.get_client", return_value=_registry_client()):
            tools = await server.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"

    @pytest.mark.asyncio
    async def test_each_tool_has_input_schema(self):
        with patch("telegram_wai_mcp.server.get_client", return_value=_registry_client()):
            tools = await server.list_tools()
        for tool in tools:
            assert tool.inputSchema, f"Tool {tool.name} has no input schema"


class TestCallTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = await server.call_tool("nonexistent_tool", {})
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert len(result.content) == 1
        assert "Unknown tool" in result.content[0].text

    @pytest.mark.asyncio
    async def test_search_messages_requires_query(self):
        result = await server.call_tool("search_messages", {"query": ""})
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert len(result.content) == 1
        assert "non-empty" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_chats_requires_query(self):
        result = await server.call_tool("search_chats", {"query": ""})
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert len(result.content) == 1
        assert "non-empty" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_send_message_surfaces_backend_errors_as_mcp_errors(self):
        mock_api = AsyncMock()
        mock_api.send_message.side_effect = RuntimeError(
            "Backend returned HTTP 400 for POST /api/v1/messages/chat/send: Telegram error"
        )

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "send_message",
                {"chat_id": "chat-123", "text": "hello"},
            )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert "Backend returned HTTP 400" in result.content[0].text
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_draft_uses_shared_registry_and_reports_no_send(self):
        mock_api = AsyncMock()
        text = "Черновик 🌱\nhttps://example.com"
        mock_api.execute_data_tool.return_value = {
            "chat_id": "chat-123",
            "text": text,
            "saved": True,
            "sent": False,
            "replaces_existing_draft": True,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "save_draft",
                {"chat_id": "chat-123", "text": text},
            )

        assert "Draft saved successfully" in result[0].text
        assert "No Telegram message was sent" in result[0].text
        mock_api.execute_data_tool.assert_awaited_once_with(
            "save_draft",
            {"chat_id": "chat-123", "text": text},
        )
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_chats_reports_fresh_unread_counts_without_reading(self):
        mock_api = AsyncMock()
        mock_api.refresh_chats.return_value = {
            "chats": [{"id": "chat-1", "unread_count": 3}],
            "has_more": False,
            "total": 1,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool("refresh_chats", {})

        assert "Refreshed 1 chat" in result[0].text
        assert "No messages were marked read" in result[0].text
        mock_api.refresh_chats.assert_awaited_once_with()
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_draft_uses_shared_registry_and_reports_no_send(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "chat_id": "chat-123",
            "cleared": True,
            "saved": True,
            "sent": False,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool("clear_draft", {"chat_id": "chat-123"})

        assert "Draft cleared successfully" in result[0].text
        assert "No Telegram message was sent" in result[0].text
        mock_api.execute_data_tool.assert_awaited_once_with("clear_draft", {"chat_id": "chat-123"})
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_drafts_returns_live_backend_state(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "drafts": [
                {
                    "chat_id": "chat-123",
                    "text": "Current draft",
                    "date": "2026-09-04T16:00:00+00:00",
                }
            ],
            "count": 1,
            "unmatched_count": 0,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool("list_drafts", {})

        payload = result[0].text
        assert "Current draft" in payload
        assert '"count": 1' in payload
        mock_api.execute_data_tool.assert_awaited_once_with("list_drafts", {})
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_draft_rejects_blank_text(self):
        result = await server.call_tool(
            "save_draft",
            {"chat_id": "chat-123", "text": " \n "},
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert '"text" must be a non-empty string' in result.content[0].text

    @pytest.mark.asyncio
    async def test_save_draft_result_does_not_truncate_text(self):
        mock_api = AsyncMock()
        text = "A" * 500
        mock_api.execute_data_tool.return_value = {
            "chat_id": "chat-123",
            "text": text,
            "saved": True,
            "sent": False,
            "replaces_existing_draft": True,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "save_draft",
                {"chat_id": "chat-123", "text": text},
            )

        assert text in result[0].text

    @pytest.mark.asyncio
    async def test_mcp_save_draft_rejects_unconfirmed_backend_result(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "chat_id": "chat-123",
            "text": "Draft",
            "saved": False,
            "sent": False,
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "save_draft",
                {"chat_id": "chat-123", "text": "Draft"},
            )

        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert "did not confirm" in result.content[0].text

    @pytest.mark.asyncio
    async def test_get_data_status_uses_shared_backend_registry(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "chats": 250,
            "messages": 250,
            "queue_depths": {"media-fetch": 0},
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool("get_data_status", {})

        assert '"chats": 250' in result[0].text
        assert '"messages": 250' in result[0].text
        mock_api.execute_data_tool.assert_awaited_once_with("get_data_status")
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_messages_date_filters_expand_date_only_inputs(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {"results": [], "total": 0, "query": "test"}

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "search_messages",
                {
                    "query": "test",
                    "date_from": "2026-01-29",
                    "date_to": "2026-01-29",
                },
            )

        assert result[0].text.startswith("No messages found")
        tool_name, arguments = mock_api.execute_data_tool.await_args.args
        assert tool_name == "search_messages"
        assert arguments["date_from"] == datetime(2026, 1, 29, 0, 0, tzinfo=UTC).isoformat()
        assert (
            arguments["date_to"]
            == datetime(2026, 1, 29, 23, 59, 59, 999999, tzinfo=UTC).isoformat()
        )
        assert arguments["limit"] == 20
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_messages_forwards_multiple_chat_ids_and_cursor(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "results": [],
            "total": 0,
            "query": "roadmap",
        }
        chat_ids = [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ]

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            await server.call_tool(
                "search_messages",
                {
                    "query": "roadmap",
                    "chat_ids": chat_ids,
                    "cursor": "next-page",
                },
            )

        mock_api.execute_data_tool.assert_awaited_once_with(
            "search_messages",
            {
                "query": "roadmap",
                "limit": 20,
                "chat_ids": chat_ids,
                "cursor": "next-page",
            },
        )
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_messages_forwards_mode_and_chat_types(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "results": [],
            "total": 0,
            "query": "Альфа-Банк",
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            await server.call_tool(
                "search_messages",
                {
                    "query": "Альфа-Банк",
                    "mode": "exact",
                    "chat_types": ["group", "supergroup"],
                },
            )

        mock_api.execute_data_tool.assert_awaited_once_with(
            "search_messages",
            {
                "query": "Альфа-Банк",
                "limit": 20,
                "mode": "exact",
                "chat_types": ["group", "supergroup"],
            },
        )

    @pytest.mark.asyncio
    async def test_find_chats_groups_message_hits_and_merges_title_matches(self):
        mock_api = AsyncMock()
        mock_api.list_chats.return_value = {
            "chats": [
                {
                    "id": "chat-title",
                    "title": "Альфабанк/ обучение",
                    "chat_type": "group",
                    "total_messages_synced": 48,
                },
                {
                    "id": "chat-content",
                    "title": "Wai School",
                    "chat_type": "group",
                    "total_messages_synced": 100,
                },
            ],
            "total": 2,
            "next_cursor": None,
        }
        mock_api.execute_data_tool.return_value = {
            "results": [
                {
                    "chat_id": "chat-content",
                    "chat_title": "Wai School",
                    "chat_type": "group",
                    "telegram_message_id": 12,
                    "text": "Диана из Альфа Банка заинтересовалась",
                    "sent_at": "2026-06-01T10:00:00Z",
                    "similarity": 0.9,
                    "telegram_message_url": "https://t.me/c/1/12",
                },
                {
                    "chat_id": "chat-content",
                    "chat_title": "Wai School",
                    "chat_type": "group",
                    "telegram_message_id": 11,
                    "text": "Обсудили обучение риэлторов",
                    "sent_at": "2026-05-31T10:00:00Z",
                    "similarity": 0.8,
                },
            ],
            "total": 2,
            "has_more": False,
            "next_cursor": None,
            "query": "Альфа-Банк обучение риэлторов",
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "find_chats",
                {
                    "query": "Альфа-Банк обучение риэлторов",
                    "chat_types": ["group", "supergroup"],
                    "limit": 10,
                    "messages_per_chat": 1,
                },
            )

        text = result[0].text
        assert "Wai School" in text
        assert "Альфабанк/ обучение" in text
        assert text.count("Диана из Альфа Банка") == 1
        assert "Coverage: 2 message hits scanned" in text
        mock_api.execute_data_tool.assert_awaited_once_with(
            "search_messages",
            {
                "query": "Альфа-Банк обучение риэлторов",
                "limit": 100,
                "mode": "hybrid",
                "chat_types": ["group", "supergroup"],
            },
        )

    @pytest.mark.asyncio
    async def test_find_chats_fetches_inventory_and_first_search_page_concurrently(self):
        mock_api = AsyncMock()
        inventory_started = asyncio.Event()
        search_started = asyncio.Event()

        async def list_chats(**_kwargs):
            inventory_started.set()
            await asyncio.wait_for(search_started.wait(), timeout=0.2)
            return {"chats": [], "total": 0, "next_cursor": None}

        async def execute_data_tool(_name, _arguments):
            search_started.set()
            await asyncio.wait_for(inventory_started.wait(), timeout=0.2)
            return {
                "results": [],
                "total": 0,
                "has_more": False,
                "next_cursor": None,
            }

        mock_api.list_chats.side_effect = list_chats
        mock_api.execute_data_tool.side_effect = execute_data_tool

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            await asyncio.wait_for(
                server.call_tool("find_chats", {"query": "AI training"}),
                timeout=1,
            )

        assert inventory_started.is_set()
        assert search_started.is_set()

    @pytest.mark.asyncio
    async def test_find_chats_exact_mode_paginates_until_complete(self):
        mock_api = AsyncMock()
        mock_api.list_chats.return_value = {
            "chats": [],
            "total": 0,
            "next_cursor": None,
        }
        mock_api.execute_data_tool.side_effect = [
            {
                "results": [
                    {
                        "chat_id": "chat-1",
                        "chat_title": "First",
                        "chat_type": "group",
                        "telegram_message_id": 1,
                        "text": "Альфа-Банк",
                        "sent_at": "2026-01-01T00:00:00Z",
                        "similarity": 1.0,
                    }
                ],
                "has_more": True,
                "next_cursor": "page-2",
            },
            {
                "results": [
                    {
                        "chat_id": "chat-2",
                        "chat_title": "Second",
                        "chat_type": "supergroup",
                        "telegram_message_id": 2,
                        "text": "Альфа-Банк",
                        "sent_at": "2026-02-01T00:00:00Z",
                        "similarity": 1.0,
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
        ]

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "find_chats",
                {"query": "Альфа-Банк", "mode": "exact"},
            )

        assert "First" in result[0].text
        assert "Second" in result[0].text
        assert "complete" in result[0].text
        assert mock_api.execute_data_tool.await_count == 2
        assert mock_api.execute_data_tool.await_args_list[1].args[1]["cursor"] == "page-2"

    @pytest.mark.asyncio
    async def test_search_chats_matches_title_and_username(self):
        mock_api = AsyncMock()
        page_1 = {
            "chats": [
                {
                    "title": "Not Alice",
                    "id": "chat-1",
                    "chat_type": "private",
                    "username": "someone_else",
                    "total_messages_synced": 10,
                    "last_sync_at": "2026-04-01T00:00:00+00:00",
                }
            ],
            "total": 2,
            "next_cursor": "cursor-2",
        }
        page_2 = {
            "chats": [
                {
                    "title": "Alice Example",
                    "id": "chat-2",
                    "chat_type": "private",
                    "username": "alice_ush",
                    "total_messages_synced": 50,
                    "last_sync_at": "2026-04-10T00:00:00+00:00",
                }
            ],
            "total": 2,
            "next_cursor": None,
        }
        mock_api.list_chats.side_effect = [page_1, page_2]

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "search_chats",
                {"query": "alice", "limit": 10},
            )

        assert 'Found 2 chats for query: "alice"' in result[0].text
        assert result[0].text.index("Alice Example") < result[0].text.index("Not Alice")
        assert "@alice_ush" in result[0].text
        assert mock_api.list_chats.await_count == 2
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_messages_reranks_exact_text_match_within_chat(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "results": [
                {
                    "text": "Да кто же спорит",
                    "chat_title": "Алиса Лисичкаааааа",
                    "chat_username": "alice_ush",
                    "sender_name": "Алиса Лисичкаааааа",
                    "sent_at": "2026-04-10T10:00:00Z",
                    "similarity": 0.28,
                    "is_outgoing": False,
                    "has_media": False,
                    "chat_id": "chat-1",
                    "telegram_message_id": 1,
                },
                {
                    "text": "В Псковской области застрелили двух собак поисково-спасательного отряда «ЛизаАлерт»",
                    "chat_title": "Алиса Лисичкаааааа",
                    "chat_username": "alice_ush",
                    "sender_name": "Алиса Лисичкаааааа",
                    "sent_at": "2026-04-10T11:00:00Z",
                    "similarity": 0.24,
                    "is_outgoing": False,
                    "has_media": False,
                    "chat_id": "chat-1",
                    "telegram_message_id": 2,
                },
            ],
            "total": 2,
            "query": "ЛизаАлерт",
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "search_messages",
                {
                    "query": "ЛизаАлерт",
                    "chat_id": "chat-1",
                    "limit": 1,
                    "date_from": "2026-04-10",
                    "date_to": "2026-04-10",
                },
            )

        assert "ЛизаАлерт" in result[0].text
        _tool_name, arguments = mock_api.execute_data_tool.await_args.args
        assert arguments["limit"] == 1
        mock_api.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_messages_reranks_person_match_globally(self):
        mock_api = AsyncMock()
        mock_api.execute_data_tool.return_value = {
            "results": [
                {
                    "text": "а альта сегодня работает ещё, кто знает?",
                    "chat_title": "Белградские бродяги",
                    "sender_name": "beautiful james",
                    "sent_at": "2026-04-10T09:00:00Z",
                    "similarity": 0.41,
                    "is_outgoing": False,
                    "has_media": False,
                    "chat_id": "chat-2",
                    "telegram_message_id": 3,
                },
                {
                    "text": "Сижу сдерживаюсь, чтобы не плакать",
                    "chat_title": "Алиса Лисичкаааааа",
                    "chat_username": "alice_ush",
                    "sender_name": "Алиса Лисичкаааааа",
                    "sent_at": "2026-04-10T11:00:00Z",
                    "similarity": 0.27,
                    "is_outgoing": False,
                    "has_media": False,
                    "chat_id": "chat-1",
                    "telegram_message_id": 4,
                },
            ],
            "total": 2,
            "query": "что мне писала Алиса сегодня",
        }

        with patch("telegram_wai_mcp.server.get_client", return_value=mock_api):
            result = await server.call_tool(
                "search_messages",
                {
                    "query": "что мне писала Алиса сегодня",
                    "limit": 1,
                    "date_from": "2026-04-10",
                    "date_to": "2026-04-10",
                },
            )

        assert "Алиса Лисичкаааааа" in result[0].text
        _tool_name, arguments = mock_api.execute_data_tool.await_args.args
        assert arguments["limit"] == 1
        mock_api.close.assert_awaited_once()


class TestFormatHelpers:
    def test_format_search_results_with_results(self):
        result = {
            "results": [
                {
                    "text": "hello world",
                    "chat_title": "Test Chat",
                    "chat_username": "test_chat",
                    "sender_name": "John",
                    "sent_at": "2024-01-01T12:00:00Z",
                    "similarity": 0.95,
                    "is_outgoing": False,
                    "has_media": False,
                }
            ],
            "total": 1,
            "query": "hello",
        }
        content = server.format_search_results(result)
        assert len(content) >= 1
        assert "Found" in content[0].text
        assert "@test_chat" in content[0].text
        assert "https://t.me/test_chat" in content[0].text

    def test_format_search_results_private_supergroup_link_fallback(self):
        result = {
            "results": [
                {
                    "text": "hello world",
                    "chat_title": "Private Team",
                    "chat_type": "supergroup",
                    "chat_telegram_id": -1001234567890,
                    "telegram_message_id": 321,
                    "sender_name": "John",
                    "sent_at": "2024-01-01T12:00:00Z",
                    "similarity": 0.95,
                    "is_outgoing": False,
                    "has_media": False,
                }
            ],
            "total": 1,
            "query": "hello",
        }
        content = server.format_search_results(result)
        assert "Open: https://t.me/c/1234567890/321" in content[0].text

    def test_format_search_results_empty(self):
        result = {"results": [], "total": 0, "query": "nothing"}
        content = server.format_search_results(result)
        assert len(content) == 1

    def test_format_search_results_tolerates_missing_fields(self):
        result = {"results": [{"text": "hello"}]}
        content = server.format_search_results(result)
        assert len(content) >= 1

    def test_format_search_results_keeps_full_text(self):
        long_text = "A" * 320
        result = {
            "results": [
                {
                    "text": long_text,
                    "chat_title": "Test Chat",
                    "sender_name": "John",
                    "sent_at": "2024-01-01T12:00:00Z",
                    "similarity": 0.95,
                    "is_outgoing": False,
                    "has_media": False,
                }
            ],
            "total": 1,
            "query": "hello",
        }
        content = server.format_search_results(result)
        assert long_text in content[0].text


class TestFormatChatList:
    def test_shows_count_header(self):
        result = {
            "chats": [
                {
                    "title": "Chat A",
                    "id": "1",
                    "chat_type": "private",
                    "username": "chat_a",
                }
            ],
            "total": 50,
            "has_more": True,
            "next_cursor": "cursor_abc",
        }
        content = server.format_chat_list(result)
        assert "Showing 1 of 50" in content[0].text
        assert "@chat_a" in content[0].text
        assert "https://t.me/chat_a" in content[0].text

    def test_shows_unread_count(self):
        result = {
            "chats": [
                {
                    "title": "Needs Reply",
                    "id": "1",
                    "chat_type": "private",
                    "unread_count": 4,
                }
            ],
            "total": 1,
        }
        content = server.format_chat_list(result)
        assert "Unread: 4" in content[0].text

    def test_shows_last_message_preview_without_opening_each_chat(self):
        result = {
            "chats": [
                {
                    "title": "Needs Reply",
                    "id": "1",
                    "chat_type": "private",
                    "username": "candidate",
                    "last_message_id": 1661287,
                    "last_message_sender_name": "Georgiy Shashkov",
                    "last_message_text": "Хотел бы продолжить обсуждение вакансии",
                    "last_activity_at": "2026-08-26T13:20:00Z",
                    "unread_count": 1,
                }
            ],
            "total": 1,
        }

        content = server.format_chat_list(result)
        text = content[0].text
        assert "Last message #1661287" in text
        assert "Georgiy Shashkov" in text
        assert "Хотел бы продолжить обсуждение вакансии" in text
        assert "2026-08-26" in text


class TestFormatChatSearch:
    def test_format_chat_search_results_with_results(self):
        result = {
            "query": "alice",
            "total": 1,
            "chats": [
                {
                    "title": "Alice Example",
                    "id": "chat-1",
                    "chat_type": "private",
                    "username": "alice_ush",
                    "total_messages_synced": 12,
                }
            ],
        }
        content = server.format_chat_search_results(result)
        assert 'Found 1 chats for query: "alice"' in content[0].text
        assert "@alice_ush" in content[0].text

    def test_format_chat_search_results_empty(self):
        content = server.format_chat_search_results({"query": "alice", "total": 0, "chats": []})
        assert 'No chats found for query: "alice"' == content[0].text

    def test_shows_private_supergroup_link_when_username_missing(self):
        result = {
            "chats": [
                {
                    "title": "Private Team",
                    "id": "1",
                    "chat_type": "supergroup",
                    "telegram_chat_id": -1001234567890,
                    "last_message_id": 654,
                }
            ],
            "total": 1,
            "has_more": False,
        }
        content = server.format_chat_list(result)
        assert "Open: https://t.me/c/1234567890/654" in content[0].text

    def test_pagination_footer_when_has_more(self):
        result = {
            "chats": [{"title": "Chat A", "id": "1", "chat_type": "private"}],
            "total": 100,
            "has_more": True,
            "next_cursor": "cursor_xyz",
        }
        content = server.format_chat_list(result)
        text = content[0].text
        assert 'cursor="cursor_xyz"' in text
        assert "More chats available" in text

    def test_no_pagination_footer_when_no_more(self):
        result = {
            "chats": [{"title": "Chat A", "id": "1", "chat_type": "private"}],
            "total": 1,
            "has_more": False,
        }
        content = server.format_chat_list(result)
        assert "More chats available" not in content[0].text

    def test_empty_chats(self):
        result = {"chats": [], "total": 0}
        content = server.format_chat_list(result)
        assert "No chats synced" in content[0].text


class TestFormatDataStatus:
    def _make_chats(self, n: int) -> list[dict]:
        return [
            {
                "title": f"Chat {i}",
                "id": f"id-{i}",
                "chat_type": "private" if i % 2 == 0 else "group",
                "total_messages_synced": i * 10,
                "last_sync_at": f"2026-03-0{min(i, 9)}T12:00:00+00:00",
            }
            for i in range(1, n + 1)
        ]

    def test_shows_summary_not_full_list(self):
        chats = self._make_chats(20)
        result = {"chats": chats, "total": 20}
        settings = {"listener_active": False, "realtime_sync_enabled": True}
        content = server.format_data_status(settings, result)
        text = content[0].text
        # Should have summary stats
        assert "Total chats: 20" in text
        assert "Total messages synced:" in text
        assert "Chat types:" in text
        assert "Data freshness:" in text
        # Should only show 10 chats in the preview, not all 20
        assert text.count("ID: id-") == 10

    def test_top_10_cap(self):
        chats = self._make_chats(15)
        result = {"chats": chats, "total": 15}
        settings = {"listener_active": False, "realtime_sync_enabled": False}
        content = server.format_data_status(settings, result)
        text = content[0].text
        assert "Top 10" in text

    def test_footer_guidance(self):
        chats = self._make_chats(5)
        result = {"chats": chats, "total": 5}
        settings = {"listener_active": False, "realtime_sync_enabled": False}
        content = server.format_data_status(settings, result)
        text = content[0].text
        assert "list_chats" in text
        assert "search_messages" in text

    def test_empty_chats(self):
        result = {"chats": [], "total": 0}
        settings = {"listener_active": False, "realtime_sync_enabled": False}
        content = server.format_data_status(settings, result)
        assert "No chats synced" in content[0].text


class TestFormatChatMessages:
    def test_keeps_full_text(self):
        long_text = "B" * 360
        result = {
            "messages": [
                {
                    "telegram_message_id": 123,
                    "text": long_text,
                    "sender_name": "Jane",
                    "sent_at": "2024-01-01T12:00:00Z",
                    "is_outgoing": False,
                }
            ],
            "has_more": False,
            "total_messages_synced": 1,
            "last_sync_at": "2024-01-01T12:00:00Z",
        }
        content = server.format_chat_messages(result)
        assert long_text in content[0].text


async def test_get_files_is_routed_to_the_shared_registry():
    """list_tools advertises it from the backend, so call_tool must dispatch it too."""
    from unittest.mock import AsyncMock, patch

    from telegram_wai_mcp import server

    api = AsyncMock()
    api.execute_data_tool = AsyncMock(return_value={"files": [], "total": 0})
    api.close = AsyncMock()

    with patch.object(server, "get_client", return_value=api):
        result = await server.call_tool("get_files", {"query": "смета", "limit": 5})

    api.execute_data_tool.assert_awaited_once_with("get_files", {"query": "смета", "limit": 5})
    api.close.assert_awaited_once()
    assert "No files found" in result[0].text


async def test_every_advertised_tool_has_a_dispatch_branch():
    """A registry name with no branch answers "Unknown tool" to every caller."""
    from unittest.mock import patch

    from telegram_wai_mcp import server

    api = _registry_client()
    with patch.object(server, "get_client", return_value=api):
        advertised = [tool.name for tool in await server.list_tools()]

    class _EmptyBackend:
        """Every backend call answers empty, so only the routing is under test."""

        def __getattr__(self, _name):
            async def call(*_args, **_kwargs):
                return {}

            return call

    for name in advertised:
        with patch.object(server, "get_client", return_value=_EmptyBackend()):
            result = await server.call_tool(name, {})
        content = result.content if hasattr(result, "content") else result
        text = " ".join(item.text for item in content if hasattr(item, "text"))
        assert "Unknown tool" not in text, name
