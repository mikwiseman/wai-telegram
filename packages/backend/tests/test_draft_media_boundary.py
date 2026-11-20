"""A native file draft must round-trip its text and attachment without sending."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from telethon import functions, types

from app.services.telegram_rich_drafts import draft_state, save_with_attachment
from app.services.tool_registry import ToolInputError, execute_data_tool


async def test_complete_attachment_reaches_native_draft_service(db_session, test_user):
    chat_id = uuid4()
    with patch(
        "app.services.tool_registry.save_telegram_draft", new_callable=AsyncMock
    ) as save:
        await execute_data_tool(
            db_session,
            test_user.id,
            "save_draft",
            {
                "chat_id": str(chat_id),
                "text": "An invitation",
                "file_name": "invite.pdf",
                "file_base64": "JVBERg==",
            },
        )
    save.assert_awaited_once_with(
        db_session,
        test_user.id,
        chat_id,
        "An invitation",
        file_name="invite.pdf",
        file_base64="JVBERg==",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("file_url", "https://example.com/chart.jpg"),
        ("file_name", "chart.jpg"),
        ("file_base64", "YWJj"),
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


@pytest.mark.parametrize("photo", [False, True])
@pytest.mark.parametrize(
    "outcome", ["saved", "changed", "missing_attachment", "wrong_text"]
)
async def test_verified_editor_draft(tmp_path, photo, outcome):
    text = "Первый абзац\n\nВторой абзац"
    path = tmp_path / ("card.png" if photo else "card.pdf")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" if photo else b"%PDF-1.7")
    peer = types.InputPeerUser(10, 11)
    previous = types.DraftMessage(text, None)
    document = types.Document(
        7,
        8,
        b"ref",
        None,
        "application/pdf",
        path.stat().st_size,
        1,
        [types.DocumentAttributeFilename(path.name)],
    )
    picture = types.Photo(7, 8, b"ref", None, [], 1)
    current = (
        types.DraftMessage("a new draft", None) if outcome == "changed" else previous
    )
    requests = []

    async def telegram(request):
        nonlocal current
        requests.append(type(request))
        if isinstance(request, functions.messages.GetPeerDialogsRequest):
            return SimpleNamespace(
                dialogs=[SimpleNamespace(peer=types.PeerUser(10), draft=current)]
            )
        if isinstance(request, functions.messages.UploadMediaRequest):
            if photo:
                assert isinstance(request.media, types.InputMediaUploadedPhoto)
                return types.MessageMediaPhoto(photo=picture)
            assert request.media.attributes[0].file_name == path.name
            return types.MessageMediaDocument(document=document)
        assert isinstance(request, functions.messages.SaveDraftRequest)
        expected_block = types.PageBlockPhoto if photo else types.PageBlockDocument
        assert isinstance(request.rich_message.blocks[-1], expected_block)
        assert request.media is None and request.message == text
        blocks = list(request.rich_message.blocks)
        if outcome == "wrong_text":
            blocks[0] = types.PageBlockParagraph(types.TextPlain("Different"))
        rich = types.RichMessage(
            blocks, [picture] if photo else [], [] if photo else [document]
        )
        current = types.DraftMessage(
            text, None, rich_message=None if outcome == "missing_attachment" else rich
        )
        return True

    client = AsyncMock(side_effect=telegram)
    client.upload_file.return_value = types.InputFile(1, 1, path.name, "")
    if outcome == "saved":
        saved = await save_with_attachment(client, peer, text, path, previous)
        assert saved["telegram_file_id"] == "7"
        assert saved["kind"] == ("photo" if photo else "document")
    else:
        with pytest.raises(ValueError, match="changed|did not return"):
            await save_with_attachment(client, peer, text, path, previous)
    if outcome == "changed":
        assert functions.messages.SaveDraftRequest not in requests
    client.send_file.assert_not_awaited()
    client.send_message.assert_not_awaited()


def test_rotating_references_do_not_report_a_user_edit():
    document = types.Document(7, 8, b"reference", None, "application/pdf", 20, 1, [])
    block = types.PageBlockDocument(
        7, types.PageCaption(types.TextEmpty(), types.TextEmpty())
    )
    draft = types.DraftMessage(
        "caption", None, rich_message=types.RichMessage([block], [], [document])
    )
    original = draft_state(draft)
    document.file_reference = b"renewed"
    assert draft_state(draft) == original
    draft.rich_message.blocks.clear()
    assert draft_state(draft) != original
