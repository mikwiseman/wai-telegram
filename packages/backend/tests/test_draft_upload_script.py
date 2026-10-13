"""Media preparation stays local; the obsolete upload entry point cannot send."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

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


def test_obsolete_uploader_fails_before_connecting(tmp_path):
    uploader = load("save_draft_file")
    path = tmp_path / "chart.jpg"
    path.write_bytes(b"image")
    with patch("http.client.HTTPSConnection") as connection:
        with pytest.raises(ValueError, match="MEDIA_INVALID"):
            uploader.save_draft_file(
                path,
                "Caption",
                chat_id="34d62d3f-9070-4189-8f26-97bdf0158cd2",
                base_url="https://telegram.waiwai.is",
                api_key="unused",
            )
        connection.assert_not_called()


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
