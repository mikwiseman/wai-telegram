"""Exercise the local-file-to-draft helper over real HTTP, without Telegram."""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "plugins/wai-telegram/skills/telegram-drafts/scripts/save_draft_file.py"
)
spec = importlib.util.spec_from_file_location("save_draft_file", SCRIPT)
uploader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uploader)


def test_draft_uploader_streams_file_and_verifies_no_send(tmp_path):
    path = tmp_path / "Оффер.pdf"
    content = b"pdf-data" * 100
    path.write_bytes(content)
    chat_id = "34d62d3f-9070-4189-8f26-97bdf0158cd2"
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), data))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "chat_id": chat_id,
                        "text": "Коротко",
                        "file_name": path.name,
                        "file_size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "has_media": True,
                        "saved": True,
                        "sent": False,
                        "replaces_existing_draft": True,
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = uploader.save_draft_file(
            path,
            "Коротко",
            chat_id=chat_id,
            base_url=f"http://127.0.0.1:{server.server_port}",
            api_key="test-key",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result["saved"] is True
    assert result["sent"] is False
    assert result["has_media"] is True
    assert len(calls) == 1
    route, headers, data = calls[0]
    assert route == f"/api/v1/messages/{chat_id}/draft-file"
    assert data == content
    assert unquote(headers["X-File-Name"]) == path.name
    assert unquote(headers["X-Telegram-Caption"]) == "Коротко"


def test_draft_uploader_rejects_non_uuid_chat_id(tmp_path):
    path = tmp_path / "file.pdf"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="chat_id must be a UUID"):
        uploader.save_draft_file(
            path,
            chat_id="chat-123",
            base_url="http://127.0.0.1:8000",
            api_key="test-key",
        )
