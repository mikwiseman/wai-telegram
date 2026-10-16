"""Local uploads preserve original bytes in Wai and cannot implicitly send."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
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


def test_file_uploader_uses_verified_wai_storage_without_publishing(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    uploader = load("save_draft_file")
    path = tmp_path / "contract.docx"
    path.write_bytes(b"document")
    client = Mock()
    client.save.return_value = {"sent": False, "readback_verified": True}
    factory = Mock(return_value=client)
    monkeypatch.setattr(uploader, "DraftClient", factory)
    draft_id, chat_id = str(uuid4()), str(uuid4())
    result = uploader.save_draft_file(
        path, "Caption", draft_id=draft_id, chat_id=chat_id
    )
    client.save.assert_called_once_with(
        draft_id, chat_id, path, "Caption", media_type="file"
    )
    client.publish.assert_not_called()
    assert result["sent"] is False


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
