"""Unsupported attachments must not silently replace a text draft.

Telegram rejected both InputMediaPhoto and InputMediaUploadedDocument in live
SaveDraftRequest checks on 2026-10-13. Mocked RPC success is not support.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.services.tool_registry import ToolInputError, execute_data_tool


@pytest.mark.parametrize(
    "field,value",
    [
        ("file_url", "https://example.com/chart.jpg"),
        ("file_name", "chart.jpg"),
    ],
)
async def test_media_draft_request_fails_before_replacing_text(
    db_session, test_user, field, value
):
    with patch(
        "app.services.tool_registry.save_telegram_draft", new_callable=AsyncMock
    ) as save:
        with pytest.raises(ToolInputError, match="attachments"):
            await execute_data_tool(
                db_session,
                test_user.id,
                "save_draft",
                {
                    "chat_id": str(uuid4()),
                    "text": "Keep this draft",
                    field: value,
                },
            )
    save.assert_not_awaited()
