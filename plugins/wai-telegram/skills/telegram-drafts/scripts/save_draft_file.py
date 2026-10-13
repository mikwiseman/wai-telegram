#!/usr/bin/env python3
"""Compatibility entry point: Telegram cannot sync photo/file draft attachments."""

import sys
from pathlib import Path

ERROR = (
    "Telegram rejects photo and file attachments in server-synced drafts "
    "(MEDIA_INVALID). Nothing was uploaded or sent. Use prepare_post_draft.py "
    "to keep the text and media together locally, and save only the text "
    "through the save_draft MCP tool."
)


def save_draft_file(
    path: Path,
    caption: str | None = None,
    *,
    chat_id: str,
    base_url: str,
    api_key: str,
) -> dict:
    """Fail before network access, including for callers of the old helper."""
    raise ValueError(ERROR)


if __name__ == "__main__":
    print(ERROR, file=sys.stderr)
    sys.exit(1)
