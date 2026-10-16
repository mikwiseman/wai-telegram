#!/usr/bin/env python3
"""Upload a local file+text to Wai's MCP-visible draft storage; never send."""

from pathlib import Path
from uuid import UUID

from post_draft import DraftClient, main


def save_draft_file(
    path: Path,
    caption: str,
    *,
    draft_id: str,
    chat_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Store immutable content, then verify the recipient, caption and original bytes."""
    client = DraftClient(base_url=base_url, api_key=api_key)
    return client.save(
        str(UUID(draft_id)), str(UUID(chat_id)), path, caption, media_type="file"
    )


if __name__ == "__main__":
    main(media_type="file")
