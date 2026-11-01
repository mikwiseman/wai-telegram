from starlette.testclient import TestClient
from telegram_wai_mcp import server as srv
from telegram_wai_mcp.sse_server import create_app


def test_public_mcp_never_inherits_process_owner_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_AI_KEY", "wai_owner_private")
    with TestClient(create_app()) as client:
        response = client.post(
            "/mcp",
            headers=_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
    assert response.status_code == 401


def test_existing_session_rejects_a_different_users_key(monkeypatch):
    monkeypatch.setattr(srv, "TelegramAIClient", FakeTelegramAIClient)
    with TestClient(create_app()) as client:
        session_id, protocol = _initialize(client, "/mcp", authorization="Bearer wai_alpha")
        response = client.post(
            "/mcp",
            headers=_headers(
                session_id=session_id,
                protocol_version=protocol,
                authorization="Bearer wai_beta",
            ),
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
    assert response.status_code == 403


class FakeTelegramAIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key

    async def close(self) -> None:
        return None

    async def list_data_tools(self) -> dict:
        return {
            "tools": [
                {
                    "name": "search_messages",
                    "description": "Shared search tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ]
        }

    async def search_messages(
        self,
        query: str,
        chat_ids: list[str] | None = None,
        date_from=None,
        date_to=None,
        limit: int = 20,
    ) -> dict:
        return {
            "results": [
                {
                    "text": f"query={query} key={self.api_key}",
                    "chat_title": "Test Chat",
                    "sender_name": "Tester",
                    "sent_at": "2026-03-10T10:00:00+00:00",
                    "similarity": 0.99,
                    "is_outgoing": False,
                    "chat_id": "chat-1",
                    "telegram_message_id": 123,
                }
            ],
            "total": 1,
            "query": query,
        }

    async def execute_data_tool(self, name: str, arguments: dict | None = None) -> dict:
        if name != "search_messages":
            raise AssertionError(name)
        arguments = arguments or {}
        return await self.search_messages(
            arguments.get("query", ""),
            chat_ids=arguments.get("chat_ids"),
            limit=arguments.get("limit", 20),
        )


def _headers(
    *,
    session_id: str | None = None,
    protocol_version: str | None = None,
    authorization: str | None = None,
) -> dict[str, str]:
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
        "host": "telegram.waiwai.is",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    if protocol_version:
        headers["mcp-protocol-version"] = protocol_version
    if authorization:
        headers["authorization"] = authorization
    return headers


def _initialize(client: TestClient, path: str, authorization: str | None = None) -> tuple[str, str]:
    response = client.post(
        path,
        headers=_headers(authorization=authorization),
        json={
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    protocol_version = response.json()["result"]["protocolVersion"]
    return session_id, protocol_version


def _send_initialized(client: TestClient, session_id: str, protocol_version: str) -> None:
    response = client.post(
        "/mcp",
        headers=_headers(session_id=session_id, protocol_version=protocol_version),
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert response.status_code == 202


def _tools_call_search_messages(
    client: TestClient,
    *,
    session_id: str,
    protocol_version: str,
    query: str,
) -> str:
    response = client.post(
        "/mcp",
        headers=_headers(session_id=session_id, protocol_version=protocol_version),
        json={
            "jsonrpc": "2.0",
            "id": f"call-{query}",
            "method": "tools/call",
            "params": {
                "name": "search_messages",
                "arguments": {"query": query},
            },
        },
    )
    assert response.status_code == 200
    return response.json()["result"]["content"][0]["text"]


def test_initialize_returns_session_id_and_allows_follow_up_requests(monkeypatch) -> None:
    monkeypatch.setattr(srv, "TelegramAIClient", FakeTelegramAIClient)
    srv._session_api_keys.clear()

    with TestClient(create_app()) as client:
        session_id, protocol_version = _initialize(client, "/mcp?key=wai_alpha")
        _send_initialized(client, session_id, protocol_version)

        response = client.post(
            "/mcp",
            headers=_headers(session_id=session_id, protocol_version=protocol_version),
            json={"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}},
        )

    assert response.status_code == 200
    tool_names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert "search_messages" in tool_names


def test_invalid_session_id_returns_not_found(monkeypatch) -> None:
    monkeypatch.setattr(srv, "TelegramAIClient", FakeTelegramAIClient)
    srv._session_api_keys.clear()

    with TestClient(create_app()) as client:
        _, protocol_version = _initialize(client, "/mcp?key=wai_alpha")
        response = client.post(
            "/mcp",
            headers=_headers(session_id="does-not-exist", protocol_version=protocol_version),
            json={"jsonrpc": "2.0", "id": "tools-2", "method": "tools/list", "params": {}},
        )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Session not found"


def test_authorization_header_is_bound_to_the_session(monkeypatch) -> None:
    monkeypatch.setattr(srv, "TelegramAIClient", FakeTelegramAIClient)
    srv._session_api_keys.clear()

    with TestClient(create_app()) as client:
        session_id, protocol_version = _initialize(
            client,
            "/mcp",
            authorization="Bearer wai_header",
        )
        _send_initialized(client, session_id, protocol_version)
        text = _tools_call_search_messages(
            client,
            session_id=session_id,
            protocol_version=protocol_version,
            query="presentation",
        )

    assert "wai_header" in text


def test_two_sessions_keep_api_keys_isolated(monkeypatch) -> None:
    monkeypatch.setattr(srv, "TelegramAIClient", FakeTelegramAIClient)
    srv._session_api_keys.clear()

    with TestClient(create_app()) as client:
        session_a, protocol_a = _initialize(client, "/mcp?key=wai_alpha")
        _send_initialized(client, session_a, protocol_a)
        session_b, protocol_b = _initialize(client, "/mcp?key=wai_beta")
        _send_initialized(client, session_b, protocol_b)

        text_a = _tools_call_search_messages(
            client,
            session_id=session_a,
            protocol_version=protocol_a,
            query="presentation",
        )
        text_b = _tools_call_search_messages(
            client,
            session_id=session_b,
            protocol_version=protocol_b,
            query="presentation",
        )

    assert session_a != session_b
    assert "wai_alpha" in text_a
    assert "wai_beta" in text_b
