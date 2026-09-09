"""Exercise the distributed uploader over real HTTP, without sending to Telegram."""

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
    / "plugins/wai-telegram/skills/telegram-save/scripts/send_to_saved.py"
)
spec = importlib.util.spec_from_file_location("send_to_saved", SCRIPT)
uploader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uploader)


@pytest.mark.parametrize("outcome", ["success", "forbidden", "mismatch"])
def test_uploader_streams_the_selected_file_and_never_retries(tmp_path, outcome):
    path = tmp_path / "Отчёт.bin"
    content = b"\x00\xfforiginal" * 180000
    path.write_bytes(content)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), data))
            self.send_response(403 if outcome == "forbidden" else 200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "chat_id": "me",
                        "telegram_message_id": 42,
                        "file_size": len(data),
                        "sha256": hashlib.sha256(
                            data if outcome != "mismatch" else b"bad"
                        ).hexdigest(),
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        kwargs = {
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "api_key": "test-key",
        }
        if outcome == "success":
            assert (
                uploader.send_file(path, "Готово 🌱", **kwargs)["telegram_message_id"]
                == 42
            )
        else:
            with pytest.raises(RuntimeError, match="HTTP 403|mismatch"):
                uploader.send_file(path, "Готово 🌱", **kwargs)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(calls) == 1
    route, headers, data = calls[0]
    assert route == "/api/v1/messages/me/upload"
    assert data == content
    assert unquote(headers["X-File-Name"]) == path.name
    assert unquote(headers["X-Telegram-Caption"]) == "Готово 🌱"


def test_uploader_does_not_send_token_over_remote_plain_http(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        uploader.send_file(
            tmp_path / "file", base_url="http://example.com", api_key="test"
        )
