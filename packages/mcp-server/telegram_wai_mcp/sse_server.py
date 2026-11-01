"""
Streamable HTTP MCP server for remote clients.

Usage:
    TELEGRAM_AI_URL=https://telegram.waiwai.is \
    telegram-wai-mcp-http --port 8808 --host 127.0.0.1

Authentication:
    - Authorization: Bearer wai_xxx
    - or ?key=wai_xxx on the initialize request

Public endpoint:
    POST /mcp
    GET  /mcp
"""

import argparse
import logging
import os

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from telegram_wai_mcp import server as srv
from telegram_wai_mcp.server import server

logger = logging.getLogger(__name__)

# Domain used for DNS rebinding protection when behind reverse proxy
DOMAIN = os.environ.get("MCP_ALLOWED_HOST", "telegram.waiwai.is")


def _allowed_origins() -> list[str]:
    configured = os.environ.get("MCP_ALLOWED_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        f"https://{DOMAIN}",
        "http://127.0.0.1:8808",
        "http://localhost:8808",
    ]


def _build_mcp() -> FastMCP:
    mcp = FastMCP(
        "telegram-wai-mcp",
        streamable_http_path="/mcp",
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[DOMAIN, "127.0.0.1:8808", "localhost:8808"],
            allowed_origins=_allowed_origins(),
        ),
    )
    mcp._mcp_server = server
    return mcp


def _extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()

    query_key = request.query_params.get("key", "").strip()
    return query_key or None


class SessionAuthMiddleware:
    """Bind MCP sessions to an API key without using process-global request state."""

    def __init__(self, app, mcp: FastMCP):
        self.app = app
        self.mcp = mcp

    def _session_exists(self, session_id: str) -> bool:
        return session_id in self.mcp.session_manager._server_instances

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_session_id = request.headers.get("mcp-session-id", "").strip()
        api_key = _extract_api_key(request)

        if request_session_id and not self._session_exists(request_session_id):
            response = JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "server-error",
                    "error": {"code": -32600, "message": "Session not found"},
                },
                status_code=404,
            )
            await response(scope, receive, send)
            return

        if request_session_id:
            bound_key = srv.get_session_api_key(request_session_id)
            if api_key and bound_key and api_key != bound_key:
                await Response("API key does not match this MCP session", status_code=403)(
                    scope, receive, send
                )
                return
            api_key = api_key or bound_key

        if api_key:
            scope["telegram_ai_api_key"] = api_key
        else:
            response = Response(
                "Missing API key. Use Authorization: Bearer <key> or ?key=... when initializing the MCP session.",
                status_code=401,
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message.setdefault("headers", []))
                response_session_id = headers.get("mcp-session-id", "").strip()
                if api_key and response_session_id:
                    srv.remember_session_api_key(response_session_id, api_key)
                if (
                    request.method == "DELETE"
                    and request_session_id
                    and 200 <= message["status"] < 300
                ):
                    srv.forget_session_api_key(request_session_id)
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app():
    mcp = _build_mcp()
    return SessionAuthMiddleware(mcp.streamable_http_app(), mcp)


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="MCP HTTP Server for Telegram WAI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8808, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
