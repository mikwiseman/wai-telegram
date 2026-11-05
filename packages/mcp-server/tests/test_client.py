from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from telegram_wai_mcp.client import TelegramAIClient


@pytest.fixture
def client():
    return TelegramAIClient(base_url="http://test:8000", api_key="wai_test_key")


class TestTelegramAIClientInit:
    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_AI_URL", raising=False)
        monkeypatch.delenv("TELEGRAM_AI_KEY", raising=False)
        c = TelegramAIClient()
        assert c.base_url == "http://localhost:8000"

    def test_env_base_url(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_AI_URL", "http://custom:9000")
        monkeypatch.setenv("TELEGRAM_AI_KEY", "wai_env_key")
        c = TelegramAIClient()
        assert c.base_url == "http://custom:9000"
        assert c.api_key == "wai_env_key"

    def test_explicit_params_override_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_AI_URL", "http://env:9000")
        c = TelegramAIClient(base_url="http://explicit:8000", api_key="explicit_key")
        assert c.base_url == "http://explicit:8000"
        assert c.api_key == "explicit_key"


class TestClamp:
    def test_clamp_within_range(self):
        assert TelegramAIClient._clamp(50, 1, 100) == 50

    def test_clamp_below_min(self):
        assert TelegramAIClient._clamp(-5, 1, 100) == 1

    def test_clamp_above_max(self):
        assert TelegramAIClient._clamp(200, 1, 100) == 100


@pytest.mark.asyncio
async def test_chat_lookup_passes_exact_provider_id(client):
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        await client.list_chats(telegram_chat_id=-1001564999252, limit=10)
    request.assert_awaited_once_with(
        "GET", "/api/v1/chats", params={"limit": 10, "telegram_chat_id": -1001564999252}
    )


class TestSearchMessages:
    @pytest.mark.asyncio
    async def test_search_basic(self, client):
        mock_response = httpx.Response(
            200,
            json={"results": [], "query": "hello", "total": 0},
            request=httpx.Request("POST", "http://test:8000/api/v1/search"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.search_messages("hello")
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, client):
        mock_response = httpx.Response(
            200,
            json={"results": [], "query": "test", "total": 0},
            request=httpx.Request("POST", "http://test:8000/api/v1/search"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.search_messages(
                "test",
                chat_ids=["chat-1"],
                date_from=datetime(2024, 1, 1, tzinfo=UTC),
                limit=10,
            )
            call_kwargs = mock_req.call_args
            payload = call_kwargs.kwargs.get("json", {})
            assert payload["limit"] == 10
            assert payload["chat_ids"] == ["chat-1"]

    @pytest.mark.asyncio
    async def test_search_clamps_limit(self, client):
        mock_response = httpx.Response(
            200,
            json={"results": [], "query": "test", "total": 0},
            request=httpx.Request("POST", "http://test:8000/api/v1/search"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.search_messages("test", limit=500)
            payload = mock_req.call_args.kwargs.get("json", {})
            assert payload["limit"] == 100  # clamped to MAX_LIMIT


class TestListChats:
    @pytest.mark.asyncio
    async def test_list_chats(self, client):
        mock_response = httpx.Response(
            200,
            json={"chats": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.list_chats()
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_chats_with_type(self, client):
        mock_response = httpx.Response(
            200,
            json={"chats": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.list_chats(chat_type="private")
            params = mock_req.call_args.kwargs.get("params", {})
            assert params["chat_type"] == "private"

    @pytest.mark.asyncio
    async def test_list_chats_with_cursor(self, client):
        mock_response = httpx.Response(
            200,
            json={"chats": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.list_chats(limit=25, cursor="abc123")
            params = mock_req.call_args.kwargs.get("params", {})
            assert params["cursor"] == "abc123"
            assert params["limit"] == 25

    @pytest.mark.asyncio
    async def test_list_chats_unread_only(self, client):
        mock_response = httpx.Response(
            200,
            json={"chats": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.list_chats(unread_only=True)
            params = mock_req.call_args.kwargs.get("params", {})
            assert params["unread_only"] is True

    @pytest.mark.asyncio
    async def test_refresh_chats(self, client):
        mock_response = httpx.Response(
            200,
            json={"chats": [], "has_more": False, "total": 12},
            request=httpx.Request("POST", "http://test:8000/api/v1/chats/refresh"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            result = await client.refresh_chats()

        assert result["total"] == 12
        assert mock_req.call_args.args == ("POST", "/api/v1/chats/refresh")


class TestGetMessages:
    @pytest.mark.asyncio
    async def test_get_messages(self, client):
        mock_response = httpx.Response(
            200,
            json={"messages": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats/chat-1/messages"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.get_messages("chat-1")
            assert "messages" in result

    @pytest.mark.asyncio
    async def test_get_messages_with_cursor(self, client):
        mock_response = httpx.Response(
            200,
            json={"messages": [], "has_more": False, "total": 0},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats/chat-1/messages"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ) as mock_req:
            await client.get_messages("chat-1", before="cursor123")
            params = mock_req.call_args.kwargs.get("params", {})
            assert params["before"] == "cursor123"


class TestSyncChat:
    @pytest.mark.asyncio
    async def test_sync_chat(self, client):
        mock_response = httpx.Response(
            200,
            json={"id": "job-1", "status": "pending"},
            request=httpx.Request("POST", "http://test:8000/api/v1/sync/chats/chat-1"),
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await client.sync_chat("chat-1")
            assert result["status"] == "pending"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error_raises_runtime_error(self, client):
        mock_response = httpx.Response(
            404,
            json={"detail": "Chat not found"},
            request=httpx.Request("GET", "http://test:8000/api/v1/chats/bad"),
        )
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Not found", request=mock_response.request, response=mock_response
            )
        )
        with patch.object(
            client._client, "request", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(RuntimeError, match="Chat not found"):
                await client.get_chat("bad")

    @pytest.mark.asyncio
    async def test_timeout_error_raises_runtime_error(self, client):
        with patch.object(
            client._client,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("Timed out", request=httpx.Request("GET", "http://test")),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await client.list_chats()

    @pytest.mark.asyncio
    async def test_request_error_raises_runtime_error(self, client):
        with patch.object(
            client._client,
            "request",
            new_callable=AsyncMock,
            side_effect=httpx.RequestError(
                "Connection refused", request=httpx.Request("GET", "http://test")
            ),
        ):
            with pytest.raises(RuntimeError, match="request failed"):
                await client.list_chats()
