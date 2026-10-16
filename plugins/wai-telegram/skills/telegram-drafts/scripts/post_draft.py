#!/usr/bin/env python3
"""Save/read a complete Wai post draft; publish only with explicit --approve."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import UUID


class DraftClient:
    def __init__(self, *, base_url=None, api_key=None):
        self.origin = (base_url or os.environ.get("TELEGRAM_AI_URL", "https://telegram.waiwai.is")).rstrip("/")
        if urlsplit(self.origin).scheme != "https":
            raise ValueError("Wai Telegram URL must use HTTPS")
        self.key = api_key or os.environ.get("TELEGRAM_AI_KEY")
        if not self.key:
            raise ValueError("Set TELEGRAM_AI_KEY through the configured secure environment")

    def request(self, path, *, method="GET", data=None, headers=None):
        request = Request(
            self.origin + "/api/v1/post-drafts/" + path,
            data=data,
            method=method,
            headers={"Authorization": "Bearer " + self.key, **(headers or {})},
        )
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except HTTPError as error:
            # Do not echo request headers, tokens, private captions, or server traces.
            raise ValueError(
                f"Wai Telegram returned HTTP {error.code}; read the draft before retrying"
            ) from None
        except (URLError, TimeoutError):
            raise ValueError("Connection failed; read the same draft ID before retrying") from None

    def read(self, draft_id):
        return json.loads(self.request(draft_id))

    def save(self, draft_id, chat_id, file, caption, *, media_type="image"):
        if not file.is_file() or not 0 < file.stat().st_size <= 10 * 1024 * 1024:
            raise ValueError("Choose one nonempty attachment up to 10 MiB")
        if not caption.strip() or len(caption.encode("utf-16-le")) // 2 > 1024:
            raise ValueError("Caption must contain 1–1024 UTF-16 characters")
        content = file.read_bytes()
        saved = json.loads(
            self.request(
                draft_id,
                method="PUT",
                data=content,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Chat-ID": chat_id,
                    "X-File-Name": quote(file.name),
                    "X-Telegram-Caption": quote(caption),
                    "X-Draft-Media-Type": media_type,
                },
            )
        )
        current = self.read(draft_id)
        actual = self.request(draft_id + "/media")
        digest = hashlib.sha256(content).hexdigest()
        if (
            saved != current
            or current["chat_id"] != chat_id
            or current["caption"] != caption
            or current["sha256"] != digest
            or actual != content
            or current["status"] != "draft"
            or current["sent"] is not False
        ):
            raise ValueError(
                "Readback differs from the requested draft; inspect it before continuing"
            )
        return {**current, "readback_verified": True}

    def publish(self, draft_id, approve):
        if not approve:
            raise ValueError("Publication requires explicit user approval and --approve")
        current = self.read(draft_id)
        if current["status"] == "published":
            return current
        if current["status"] != "draft":
            raise ValueError(
                "Publication is in progress or uncertain; inspect Telegram, do not resend"
            )
        return json.loads(
            self.request(
                draft_id + "/publish",
                method="POST",
                data=b'{"approve":true}',
                headers={"Content-Type": "application/json"},
            )
        )


def main(*, media_type="image"):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save", help="Store caption and image in Wai without sending")
    save.add_argument(
        "--id", required=True, type=UUID, help="Fresh UUID; reuse exactly after a timeout"
    )
    save.add_argument("--chat-id", required=True, type=UUID)
    save.add_argument("--file", required=True, type=Path)
    save.add_argument("--caption-file", required=True, type=Path, help="Telegram Markdown caption")
    save.add_argument(
        "--receipt", required=True, type=Path, help="Save server readback for later review"
    )
    read = commands.add_parser("read")
    read.add_argument("--id", required=True, type=UUID)
    publish = commands.add_parser("publish")
    publish.add_argument("--id", required=True, type=UUID)
    publish.add_argument("--approve", required=True, action="store_true")
    args = parser.parse_args()
    try:
        client = DraftClient()
        draft_id = str(args.id)
        if args.command == "save":
            result = client.save(
                draft_id,
                str(args.chat_id),
                args.file.expanduser(),
                args.caption_file.expanduser().read_text().strip(),
                media_type=media_type,
            )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            args.receipt.chmod(0o600)
        elif args.command == "read":
            result = client.read(draft_id)
        else:
            result = client.publish(draft_id, args.approve)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
