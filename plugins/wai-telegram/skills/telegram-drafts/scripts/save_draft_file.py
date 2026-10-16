#!/usr/bin/env python3
"""Save a local attachment through MCP and verify its original bytes. Never send by default."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from uuid import UUID


class MCPDraftClient:
    def __init__(self, *, base_url=None, api_key=None):
        self.origin = (
            base_url or os.environ.get("TELEGRAM_AI_URL", "https://telegram.waiwai.is")
        ).rstrip("/")
        if urlsplit(self.origin).scheme != "https":
            raise ValueError("Wai Telegram URL must use HTTPS")
        key = api_key or os.environ.get("TELEGRAM_AI_KEY")
        if not key:
            raise ValueError("Set TELEGRAM_AI_KEY through the configured secure environment")
        self.headers = {
            "Authorization": "Bearer " + key,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        self.started = False
        self.sequence = 0

    def _rpc(self, method, params=None, *, notification=False):
        self.sequence += 1
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notification:
            payload["id"] = self.sequence
        request = Request(
            self.origin + "/mcp",
            data=json.dumps(payload).encode(),
            headers=self.headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                if session_id := response.headers.get("Mcp-Session-Id"):
                    self.headers["Mcp-Session-Id"] = session_id
                body = response.read().decode()
        except (HTTPError, URLError, TimeoutError):
            raise ValueError("MCP request failed; read the same draft ID before retrying") from None
        if notification or not body:
            return {}
        if body.lstrip().startswith("{"):
            result = json.loads(body)
        else:
            frames = [
                json.loads(line[5:].strip())
                for line in body.splitlines()
                if line.startswith("data:")
            ]
            result = next(frame for frame in frames if frame.get("id") == self.sequence)
        if "error" in result:
            raise ValueError("MCP request was rejected")
        return result["result"]

    def call(self, name, arguments):
        if not self.started:
            init = self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "wai-file-draft", "version": "1.0"},
                },
            )
            self.headers["Mcp-Protocol-Version"] = init["protocolVersion"]
            self._rpc("notifications/initialized", notification=True)
            self.started = True
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise ValueError("MCP tool failed: " + name + "; read the draft before retrying")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        return json.loads(
            next(item["text"] for item in result["content"] if item["type"] == "text")
        )

    def download(self, url):
        url = urljoin(self.origin + "/", url)
        if urlsplit(url).netloc != urlsplit(self.origin).netloc or urlsplit(url).scheme != "https":
            raise ValueError("Draft download must stay on the configured server")
        with urlopen(
            Request(url, headers={"Authorization": self.headers["Authorization"]}), timeout=120
        ) as response:
            return response.read(10 * 1024 * 1024 + 1)

    def close(self):
        if not self.started:
            return
        try:
            with urlopen(
                Request(self.origin + "/mcp", headers=self.headers, method="DELETE"), timeout=15
            ):
                pass
        except (HTTPError, URLError, TimeoutError):
            pass


def save_draft_file(
    path: Path, caption: str, *, draft_id: str, chat_id: str, base_url=None, api_key=None
) -> dict:
    draft_id, chat_id = str(UUID(draft_id)), str(UUID(chat_id))
    if not path.is_file() or not 0 < path.stat().st_size <= 10 * 1024 * 1024:
        raise ValueError("Choose one nonempty attachment up to 10 MiB")
    if not caption.strip() or len(caption.encode("utf-16-le")) // 2 > 1024:
        raise ValueError("Caption must contain 1–1024 UTF-16 characters")
    content = path.read_bytes()
    client = MCPDraftClient(base_url=base_url, api_key=api_key)
    try:
        saved = client.call(
            "save_file_draft",
            {
                "draft_id": draft_id,
                "chat_id": chat_id,
                "text": caption,
                "file_name": path.name,
                "file_base64": base64.b64encode(content).decode("ascii"),
            },
        )
        current = client.call("get_file_draft", {"draft_id": draft_id})
        actual = client.download(current["media_download_url"])
        if (
            saved["id"] != draft_id
            or current["id"] != draft_id
            or current["chat_id"] != chat_id
            or current["caption"] != caption
            or current["file_name"] != path.name
            or current["file_size"] != len(content)
            or current["sha256"] != hashlib.sha256(content).hexdigest()
            or actual != content
            or current["status"] != "draft"
            or current["sent"] is not False
        ):
            raise ValueError(
                "Readback differs from the requested draft; inspect it before continuing"
            )
        return {**current, "readback_verified": True, "transport": "mcp"}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser(
        "save", help="Save a local file and text through MCP without sending"
    )
    save.add_argument("--id", required=True, type=UUID)
    save.add_argument("--chat-id", required=True, type=UUID)
    save.add_argument("--file", required=True, type=Path)
    save.add_argument("--caption-file", required=True, type=Path)
    save.add_argument("--receipt", required=True, type=Path)
    read = commands.add_parser("read")
    read.add_argument("--id", required=True, type=UUID)
    send = commands.add_parser(
        "send", help="Send only after the user explicitly approves this draft"
    )
    send.add_argument("--id", required=True, type=UUID)
    send.add_argument("--approve", required=True, action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "save":
            result = save_draft_file(
                args.file.expanduser(),
                args.caption_file.expanduser().read_text().strip(),
                draft_id=str(args.id),
                chat_id=str(args.chat_id),
            )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            args.receipt.chmod(0o600)
        else:
            client = MCPDraftClient()
            try:
                name = "get_file_draft" if args.command == "read" else "send_file_draft"
                arguments = {"draft_id": str(args.id)}
                if args.command == "send":
                    arguments["approve"] = args.approve
                result = client.call(name, arguments)
            finally:
                client.close()
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
