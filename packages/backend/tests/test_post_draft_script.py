"""The skill must upload privately to Wai, verify bytes, and never substitute a send."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "plugins/wai-telegram/skills/telegram-drafts/scripts/post_draft.py"
)
spec = importlib.util.spec_from_file_location("post_draft_script", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def client(monkeypatch):
    monkeypatch.setenv("TELEGRAM_AI_KEY", "wai_test_secret")
    return module.DraftClient()


def test_save_verifies_caption_destination_and_image_without_publishing(
    monkeypatch, tmp_path
):
    c = client(monkeypatch)
    path = tmp_path / "chart.jpg"
    path.write_bytes(b"image")
    chat_id = str(uuid4())
    state = {
        "chat_id": chat_id,
        "caption": "caption",
        "sha256": hashlib.sha256(b"image").hexdigest(),
        "status": "draft",
        "sent": False,
    }
    c.request = Mock(
        side_effect=[json.dumps(state).encode(), json.dumps(state).encode(), b"image"]
    )
    result = c.save("draft-id", chat_id, path, "caption")
    assert result["readback_verified"] is True
    assert [call.args[0] for call in c.request.call_args_list] == [
        "draft-id",
        "draft-id",
        "draft-id/media",
    ]
    assert c.request.call_args_list[0].kwargs["method"] == "PUT"


def test_save_rejects_wrong_media_readback(monkeypatch, tmp_path):
    c = client(monkeypatch)
    path = tmp_path / "chart.jpg"
    path.write_bytes(b"image")
    state = {
        "chat_id": "chat",
        "caption": "caption",
        "sha256": hashlib.sha256(b"image").hexdigest(),
        "status": "draft",
        "sent": False,
    }
    c.request = Mock(
        side_effect=[
            json.dumps(state).encode(),
            json.dumps(state).encode(),
            b"different",
        ]
    )
    with pytest.raises(ValueError, match="Readback"):
        c.save("draft-id", "chat", path, "caption")


def test_publish_does_not_contact_server_without_approval(monkeypatch):
    c = client(monkeypatch)
    c.request = Mock()
    with pytest.raises(ValueError, match="approval"):
        c.publish("draft-id", False)
    c.request.assert_not_called()


def test_published_and_uncertain_posts_are_not_resent(monkeypatch):
    c = client(monkeypatch)
    c.request = Mock(return_value=b'{"status":"published","sent":true}')
    assert c.publish("draft-id", True)["sent"] is True
    c.request.assert_called_once_with("draft-id")
    c.request = Mock(return_value=b'{"status":"unknown","sent":null}')
    with pytest.raises(ValueError, match="uncertain"):
        c.publish("draft-id", True)
    c.request.assert_called_once_with("draft-id")
