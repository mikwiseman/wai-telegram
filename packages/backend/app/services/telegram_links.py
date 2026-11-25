import mimetypes
import re
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from app.models.chat import ChatType
from app.services.media_access import create_media_download_token


def build_telegram_chat_url(
    *, chat_type: ChatType | str | None, username: str | None
) -> str | None:
    """Private peer usernames open the conversation, without a message locator."""
    handle = (username or "").strip().removeprefix("@")
    if chat_type == ChatType.PRIVATE and re.fullmatch(r"[A-Za-z0-9_]+", handle):
        return f"https://t.me/{handle}"
    return None


def build_telegram_message_url(
    *,
    chat_type: ChatType | str | None,
    telegram_chat_id: int | None,
    username: str | None,
    message_id: int | None,
) -> str | None:
    """Build the canonical Telegram message URL when Telegram exposes one."""
    if chat_type not in {ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL}:
        return None
    if not isinstance(message_id, int) or message_id <= 0:
        return None

    normalized_username = (username or "").strip().removeprefix("@")
    if normalized_username:
        return f"https://t.me/{normalized_username}/{message_id}"

    normalized_type = (
        chat_type.value if isinstance(chat_type, ChatType) else str(chat_type or "")
    )
    if normalized_type not in {ChatType.SUPERGROUP.value, ChatType.CHANNEL.value}:
        return None
    if not isinstance(telegram_chat_id, int) or telegram_chat_id >= 0:
        return None

    internal_id = abs(telegram_chat_id)
    if internal_id >= 10**12:
        internal_id -= 10**12
    if internal_id <= 0:
        return None
    return f"https://t.me/c/{internal_id}/{message_id}"


def build_media_download_url(
    *,
    base_path: str,
    user_id: UUID,
    chat_id: UUID,
    telegram_message_id: int,
) -> str:
    """Build a relative, short-lived media URL for API/MCP clients."""
    token = create_media_download_token(
        user_id=user_id,
        chat_id=chat_id,
        telegram_message_id=telegram_message_id,
    )
    return f"{base_path}?token={quote(token, safe='')}"


def media_download_filename(name: str | None, mime_type: str | None = None) -> str:
    """Keep a Telegram filename safe for Content-Disposition."""
    candidate = Path(name).name if name else "telegram-media"
    candidate = candidate.replace("\x00", "_").lstrip("-.")
    if name is None and mime_type:
        candidate += mimetypes.guess_extension(mime_type) or ""
    return candidate or "telegram-media"
