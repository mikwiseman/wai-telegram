import logging
import os
from datetime import date, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_LIMIT = 100
MAX_LOOKBACK_HOURS = 24 * 30
MAX_LOOKBACK_DAYS = 180


class TelegramAIClient:
    """Client for the Telegram AI backend API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url or os.environ.get("TELEGRAM_AI_URL", "http://localhost:8000")
        self.api_key = api_key or os.environ.get("TELEGRAM_AI_KEY", "")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
            headers=headers,
        )

    async def close(self):
        await self._client.aclose()

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()[:400]

        if response.text:
            return response.text[:400]
        return "no response body"

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = self._response_detail(e.response)
            raise RuntimeError(
                f"Backend returned HTTP {e.response.status_code} for {method} {path}: {detail}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("Backend timeout on %s %s: %s", method, path, e)
            raise RuntimeError(f"Backend timed out for {method} {path}") from e
        except httpx.RequestError as e:
            logger.error("Backend request error on %s %s: %s", method, path, e)
            raise RuntimeError(f"Backend request failed for {method} {path}: {e}") from e

    @staticmethod
    def _clamp(value: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, value))

    async def search_messages(
        self,
        query: str,
        chat_ids: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Semantic search across Telegram messages."""
        limit = self._clamp(limit, 1, MAX_LIMIT)
        payload = {
            "query": query,
            "limit": limit,
        }
        if chat_ids:
            payload["chat_ids"] = chat_ids
        if date_from:
            payload["date_from"] = date_from.isoformat()
        if date_to:
            payload["date_to"] = date_to.isoformat()
        if cursor:
            payload["cursor"] = cursor

        return await self._request("POST", "/api/v1/search", json=payload)

    async def execute_data_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the backend's shared owner-scoped tool implementation."""
        return await self._request(
            "POST",
            f"/api/v1/tools/{name}",
            json={"arguments": arguments or {}},
        )

    async def list_data_tools(self) -> dict[str, Any]:
        """Fetch the canonical shared tool contract from the backend registry."""
        return await self._request("GET", "/api/v1/tools")

    async def list_chats(
        self,
        chat_type: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        unread_only: bool = False,
        telegram_chat_id: int | None = None,
    ) -> dict[str, Any]:
        """List synced Telegram chats with cursor pagination."""
        params: dict[str, Any] = {"limit": limit}
        if telegram_chat_id is not None:
            params["telegram_chat_id"] = telegram_chat_id
        if chat_type:
            params["chat_type"] = chat_type
        if cursor:
            params["cursor"] = cursor
        if unread_only:
            params["unread_only"] = True
        return await self._request("GET", "/api/v1/chats", params=params)

    async def refresh_chats(self) -> dict[str, Any]:
        """Refresh chat metadata, including unread counts, from Telegram."""
        return await self._request("POST", "/api/v1/chats/refresh")

    async def get_chat(self, chat_id: str) -> dict[str, Any]:
        """Get details for a specific chat."""
        return await self._request("GET", f"/api/v1/chats/{chat_id}")

    async def get_messages(
        self,
        chat_id: str,
        limit: int = 50,
        before: str | None = None,
    ) -> dict[str, Any]:
        """Get messages from a specific chat with cursor pagination."""
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        return await self._request("GET", f"/api/v1/chats/{chat_id}/messages", params=params)

    async def get_message_content(
        self,
        chat_id: str,
        telegram_message_id: int,
    ) -> dict[str, Any]:
        """Get the complete transcript/extracted text and summary for one message."""
        return await self._request(
            "GET",
            f"/api/v1/chats/{chat_id}/messages/{telegram_message_id}/content",
        )

    async def sync_chat(
        self,
        chat_id: str,
        message_limit: int | None = None,
    ) -> dict[str, Any]:
        """Trigger message sync for a chat. message_limit=None means unlimited."""
        params: dict[str, Any] = {}
        if message_limit is not None:
            params["limit"] = message_limit
        return await self._request("POST", f"/api/v1/sync/chats/{chat_id}", params=params)

    async def get_sync_status(self, job_id: str) -> dict[str, Any]:
        """Get sync job status."""
        return await self._request("GET", f"/api/v1/sync/jobs/{job_id}")

    async def get_daily_digest(
        self,
        digest_date: date | None = None,
    ) -> dict[str, Any]:
        """Get or generate daily digest."""
        if digest_date:
            try:
                return await self._request("GET", f"/api/v1/digests/{digest_date.isoformat()}")
            except RuntimeError as e:
                if "HTTP 404" in str(e):
                    # Generate if not found
                    pass
                else:
                    raise

        # Generate digest
        payload = {"date": digest_date.isoformat()} if digest_date else {}
        return await self._request("POST", "/api/v1/digests/generate", json=payload)

    async def get_settings(self) -> dict[str, Any]:
        """Get user settings including listener status."""
        return await self._request("GET", "/api/v1/settings")

    async def get_sync_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent sync jobs."""
        return await self._request("GET", "/api/v1/sync/jobs", params={"limit": limit})

    async def send_message(
        self,
        chat_id: str,
        text: str,
    ) -> dict[str, Any]:
        """Send a text message to a Telegram chat."""
        return await self._request(
            "POST",
            f"/api/v1/messages/{chat_id}/send",
            json={"text": text},
        )

    async def send_file(
        self,
        chat_id: str,
        file_url: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        """Send a file to a Telegram chat."""
        payload: dict[str, Any] = {"file_url": file_url}
        if caption:
            payload["caption"] = caption
        if file_name:
            payload["file_name"] = file_name
        return await self._request(
            "POST",
            f"/api/v1/messages/{chat_id}/send-file",
            json=payload,
        )

    async def reply_to_message(
        self,
        chat_id: str,
        telegram_message_id: int,
        text: str,
    ) -> dict[str, Any]:
        """Reply to a specific message in a Telegram chat."""
        return await self._request(
            "POST",
            f"/api/v1/messages/{chat_id}/reply",
            json={
                "telegram_message_id": telegram_message_id,
                "text": text,
            },
        )
