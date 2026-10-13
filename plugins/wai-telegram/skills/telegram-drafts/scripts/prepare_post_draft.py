#!/usr/bin/env python3
"""Prepare one local post with its attachment. No network access or sending."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import UUID


def prepare_post_draft(file: Path, caption: str, *, chat_id: str, output: Path) -> dict:
    chat_id = str(UUID(chat_id))
    if not file.is_file():
        raise ValueError("Attachment must be a regular file")
    if not caption.strip():
        raise ValueError("Caption must not be empty")
    if len(caption.encode("utf-16-le")) // 2 > 1024:
        raise ValueError("Caption exceeds 1024 UTF-16 characters")
    if not file.stat().st_size:
        raise ValueError("Attachment is empty")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    attachment_dir = output / "attachments"
    attachment_dir.mkdir(mode=0o700)
    target = attachment_dir / file.name
    shutil.copyfile(file, target)
    target.chmod(0o600)
    (output / "post.txt").write_text(caption, encoding="utf-8")
    result = {
        "chat_id": chat_id,
        "prepared_locally": True,
        "saved_to_telegram": False,
        "sent": False,
        "caption_file": "post.txt",
        "attachment": str(target.relative_to(output)),
        "file_size": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (output / "draft.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--caption-file", type=Path, required=True)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_post_draft(
            args.file.expanduser(),
            args.caption_file.expanduser().read_text(encoding="utf-8"),
            chat_id=args.chat_id,
            output=args.output.expanduser(),
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
