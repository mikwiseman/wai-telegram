"""Local uploads preserve original bytes in Wai and cannot implicitly send."""

import hashlib
import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "plugins/wai-telegram/skills/telegram-drafts/scripts"
)


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("downloaded", [b"document", b"wrong bytes"])
def test_file_uploader_uses_verified_wai_storage_without_publishing(
    tmp_path, monkeypatch, downloaded
):
    uploader = load("save_draft_file")
    path = tmp_path / "contract.docx"
    path.write_bytes(b"document")
    draft_id, chat_id = str(uuid4()), str(uuid4())
    state = {
        "id": draft_id,
        "chat_id": chat_id,
        "caption": "Caption",
        "file_name": path.name,
        "file_size": 8,
        "sha256": hashlib.sha256(b"document").hexdigest(),
        "status": "draft",
        "sent": False,
        "media_download_url": "/api/v1/post-drafts/file/download?token=private",
    }

    def response(body):
        value = MagicMock()
        value.__enter__.return_value = value
        value.headers = {"Mcp-Session-Id": "test-session"}
        value.read.return_value = body
        return value

    replies = [
        response(json.dumps({"result": {"protocolVersion": "2025-11-05"}}).encode()),
        response(b""),
        response(json.dumps({"result": {"structuredContent": state}}).encode()),
        response(json.dumps({"result": {"structuredContent": state}}).encode()),
        response(downloaded),
        response(b""),
    ]
    monkeypatch.setenv("TELEGRAM_AI_KEY", "test-key")
    with patch.object(uploader, "urlopen", side_effect=replies) as request:
        if downloaded == b"document":
            result = uploader.save_draft_file(
                path, "Caption", draft_id=draft_id, chat_id=chat_id
            )
            assert result["sent"] is False
            assert result["readback_verified"] is True and result["transport"] == "mcp"
        else:
            with pytest.raises(ValueError, match="Readback differs"):
                uploader.save_draft_file(
                    path, "Caption", draft_id=draft_id, chat_id=chat_id
                )
    payloads = [
        json.loads(c.args[0].data) for c in request.call_args_list if c.args[0].data
    ]
    calls = [p["params"] for p in payloads if p["method"] == "tools/call"]
    assert [c["name"] for c in calls] == ["save_file_draft", "get_file_draft"]
    assert base64.b64decode(calls[0]["arguments"]["file_base64"]) == b"document"


def test_local_post_keeps_exact_caption_and_attachment(tmp_path):
    preparer = load("prepare_post_draft")
    path = tmp_path / "chart.jpg"
    data = b"image-content"
    path.write_bytes(data)
    output = tmp_path / "draft"
    caption = "Подпись\nhttps://t.me/wisemantalks/882"
    result = preparer.prepare_post_draft(
        path,
        caption,
        chat_id="34d62d3f-9070-4189-8f26-97bdf0158cd2",
        output=output,
    )
    assert (output / result["attachment"]).read_bytes() == data
    assert (output / "post.txt").read_text() == caption
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["prepared_locally"] is True
    assert result["saved_to_telegram"] is False
    assert result["sent"] is False
    assert json.loads((output / "draft.json").read_text()) == result
    with pytest.raises(FileExistsError):
        preparer.prepare_post_draft(
            path, "Different", chat_id=result["chat_id"], output=output
        )
    assert (output / "post.txt").read_text() == caption
