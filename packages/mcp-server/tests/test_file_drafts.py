from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, ResourceLink
from telegram_wai_mcp import server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["save_file_draft", "get_file_draft", "list_file_drafts", "send_file_draft"]
)
async def test_file_draft_tools_return_actual_attachment_and_forward_arguments(name):
    api = AsyncMock()
    api.base_url = "https://telegram.waiwai.is"
    info = {
        "id": "draft-id",
        "chat_id": "owner-chat",
        "caption": "Text",
        "file_name": "contract.docx",
        "file_size": 123,
        "sha256": "abc",
        "sent": False,
        "synced_to_telegram": False,
        "storage": "wai_telegram",
        "media_file_name": "contract.docx",
        "media_file_size": 123,
        "media_mime_type": "application/octet-stream",
        "media_download_url": "/api/v1/post-drafts/draft-id/download?token=capability",
    }
    expected = {"drafts": [info]} if name == "list_file_drafts" else info
    api.execute_data_tool.return_value = expected
    args = {"draft_id": "draft-id"}
    with patch("telegram_wai_mcp.server.get_client", return_value=api):
        result = await server.call_tool(name, args)
    assert isinstance(result, CallToolResult)
    assert result.structuredContent == expected
    link = next(item for item in result.content if isinstance(item, ResourceLink))
    assert link.name == "contract.docx"
    assert link.size == 123
    assert str(link.uri).startswith("https://telegram.waiwai.is/api/v1/post-drafts/")
    api.execute_data_tool.assert_awaited_once_with(name, args)
    api.close.assert_awaited_once()
