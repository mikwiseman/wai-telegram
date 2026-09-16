from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult
from telegram_wai_mcp import server
from telegram_wai_mcp.links import message_urls


def test_markdown_link_without_following_space_keeps_only_the_link_target():
    assert message_urls(
        {
            "text": "[статью](https://example.com/a/123)об агентах и https://en.wikipedia.org/wiki/AI_(film)."
        }
    ) == [
        ("https://example.com/a/123", "example.com"),
        ("https://en.wikipedia.org/wiki/AI_(film)", "en.wikipedia.org"),
    ]


def test_url_normalization_preserves_case_sensitive_userinfo():
    assert message_urls({"hidden_urls": ["HTTPS://User:Pass@EXAMPLE.COM/File?Token=ABC#Page"]}) == [
        ("https://User:Pass@example.com/File?Token=ABC#Page", "example.com")
    ]


def message(n, **kwargs):
    return {
        "telegram_message_id": n,
        "sent_at": "2025-05-20T12:00:00Z",
        "telegram_message_url": f"https://t.me/test/{n}",
        "text": "",
        **kwargs,
    }


@pytest.mark.asyncio
async def test_links_extracts_hidden_and_visible_and_retains_occurrences_across_pages():
    api = AsyncMock()
    api.get_messages.side_effect = [
        {
            "messages": [
                message(
                    3,
                    text="[Документ](https://docs.google.com/document/d/abc/edit).",
                    hidden_urls=["https://docs.google.com/document/d/abc/edit"],
                    visible_urls=["https://example.com/page?q=one"],
                )
            ],
            "has_more": True,
            "next_cursor": "older",
        },
        {
            "messages": [message(2, hidden_urls=["https://docs.google.com/document/d/abc/edit"])],
            "has_more": False,
            "next_cursor": None,
        },
    ]
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool("get_links", {"chat_id": "chat", "max_pages": 2})
    assert isinstance(result, CallToolResult) and not result.isError
    data = result.structuredContent
    assert len(data["links"]) == 2
    doc = next(x for x in data["links"] if "docs.google.com" in x["url"])
    assert [x["telegram_message_id"] for x in doc["occurrences"]] == [3, 2]
    assert data["coverage"]["messages_scanned"] == 2
    assert data["coverage"]["synced_history_exhausted"] is True
    assert "Telegram" in data["coverage"]["note"]
    assert data["next_cursor"] is None
    api.get_messages.assert_any_await(chat_id="chat", limit=500, before="older")
    api.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_filtered_page_keeps_cursor_and_does_not_claim_no_history():
    api = AsyncMock()
    api.get_messages.return_value = {
        "messages": [
            message(
                2,
                visible_urls=[
                    "https://notdocs.google.com/x",
                    "https://docs.google.com.evil.test/x",
                ],
            )
        ],
        "has_more": True,
        "next_cursor": "next",
    }
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool(
            "get_links",
            {
                "chat_id": "chat",
                "domains": ["docs.google.com"],
                "max_pages": 1,
            },
        )
    assert not result.isError
    assert result.structuredContent["links"] == []
    assert result.structuredContent["has_more"] is True
    assert result.structuredContent["next_cursor"] == "next"
    assert result.structuredContent["coverage"]["synced_history_exhausted"] is False


@pytest.mark.asyncio
async def test_domain_boundary_dates_query_strings_and_deleted_messages():
    api = AsyncMock()
    api.get_messages.return_value = {
        "messages": [
            message(4, sent_at="2025-06-01T12:00:00Z", visible_urls=["https://example.com/later"]),
            message(
                3,
                visible_urls=[
                    "https://sub.example.com/path?a=1#two",
                    "https://example.com.evil.test/x",
                ],
            ),
            message(
                2, deleted_at="2025-05-21T00:00:00Z", visible_urls=["https://example.com/deleted"]
            ),
        ],
        "has_more": False,
    }
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool(
            "get_links",
            {
                "chat_id": "chat",
                "domains": ["EXAMPLE.COM"],
                "date_from": "2025-05-20",
                "date_to": "2025-05-20",
            },
        )
    assert not result.isError
    assert [x["url"] for x in result.structuredContent["links"]] == [
        "https://sub.example.com/path?a=1#two"
    ]


@pytest.mark.asyncio
async def test_repeated_cursor_is_an_error_not_an_infinite_loop():
    api = AsyncMock()
    api.get_messages.return_value = {"messages": [], "has_more": True, "next_cursor": "same"}
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool("get_links", {"chat_id": "chat", "before": "same"})
    assert result.isError
    assert "cursor" in result.content[0].text
    api.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"max_pages": True},
        {"max_pages": 0},
        {"max_pages": 11},
        {"domains": "example.com"},
        {"domains": ["https://example.com/path"]},
        {"date_from": "bad"},
        {"date_from": "2025-06-01", "date_to": "2025-05-01"},
    ],
)
async def test_invalid_filters_fail_before_reading_messages(args):
    api = AsyncMock()
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool("get_links", {"chat_id": "chat", **args})
    assert result.isError
    api.get_messages.assert_not_awaited()
