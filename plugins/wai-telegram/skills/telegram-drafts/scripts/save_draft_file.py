#!/usr/bin/env python3
"""Attach one local file to a Telegram draft. The endpoint never sends a message."""

import argparse
import hashlib
import http.client
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import UUID


def save_draft_file(
    path: Path,
    caption: str | None = None,
    *,
    chat_id: str,
    base_url: str,
    api_key: str,
) -> dict:
    try:
        normalized_chat_id = str(UUID(chat_id))
    except ValueError as error:
        raise ValueError("chat_id must be a UUID") from error

    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("TELEGRAM_AI_URL must be an HTTP(S) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("TELEGRAM_AI_URL must not contain credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Use HTTPS for a remote Wai Telegram server")
    if not api_key:
        raise ValueError("Set TELEGRAM_AI_KEY in the process environment; draft permission is required")
    if not path.is_file():
        raise ValueError("The selected path must be a regular file")
    if caption and len(caption.encode("utf-16-le")) // 2 > 1024:
        raise ValueError("Caption exceeds 1024 characters")

    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=300)
    endpoint = (
        parsed.path.rstrip("/")
        + f"/api/v1/messages/{normalized_chat_id}/draft-file"
    )
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/octet-stream",
        "X-File-Name": quote(path.name, safe=""),
    }
    if caption:
        headers["X-Telegram-Caption"] = quote(caption, safe="")
    try:
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if not size:
                raise ValueError("Cannot attach an empty file")
            headers["Content-Length"] = str(size)
            connection.putrequest("POST", endpoint)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(
                        "File changed during upload; draft state is uncertain. Check Telegram before retrying"
                    )
                digest.update(chunk)
                connection.send(chunk)
                remaining -= len(chunk)
            response = connection.getresponse()
            raw = response.read(65536)
        if response.status != 200:
            raise RuntimeError(
                f"Wai Telegram returned HTTP {response.status}. Check access or service status"
            )
        result = json.loads(raw)
        if (
            result.get("chat_id") != normalized_chat_id
            or result.get("saved") is not True
            or result.get("sent") is not False
            or result.get("has_media") is not True
        ):
            raise RuntimeError(
                "Server did not confirm the file draft; check Telegram before retrying"
            )
        if result.get("file_size") != size or result.get("sha256") != digest.hexdigest():
            raise RuntimeError(
                "Draft upload verification mismatch; inspect Telegram before repeating"
            )
        return result
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--caption")
    args = parser.parse_args()
    try:
        result = save_draft_file(
            args.file.expanduser(),
            args.caption,
            chat_id=args.chat_id,
            base_url=os.environ.get("TELEGRAM_AI_URL", "https://telegram.waiwai.is"),
            api_key=os.environ.get("TELEGRAM_AI_KEY", ""),
        )
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
        if isinstance(error, (OSError, http.client.HTTPException)):
            message = "Connection interrupted; draft state is uncertain. Check Telegram before retrying"
        else:
            message = str(error)
        print(message, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
