"""Store complete posts without Telegram side effects; publish only on a separate call."""

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.post_draft import PostDraft
from app.services.messaging_service import (
    _get_chat,
    _outbound_directory,
    _sanitize_file_name,
    _send_file_path,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class PostDraftError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


async def get_post_draft(db: AsyncSession, user_id: UUID, draft_id: UUID) -> PostDraft:
    draft = await db.scalar(
        select(PostDraft).where(PostDraft.id == draft_id, PostDraft.user_id == user_id)
    )
    if draft is None:
        raise PostDraftError(404, "Post draft not found")
    return draft


def metadata(draft: PostDraft) -> dict:
    return {
        "id": str(draft.id),
        "chat_id": str(draft.chat_id),
        "caption": draft.caption,
        "parse_mode": "markdown",
        "file_name": draft.file_name,
        "mime_type": draft.mime_type,
        "file_size": len(draft.media),
        "sha256": draft.sha256,
        "status": draft.status,
        "storage": "wai_telegram",
        "synced_to_telegram": False,
        "sent": None
        if draft.status in {"publishing", "unknown"}
        else draft.status == "published",
        "telegram_message_id": draft.telegram_message_id,
        "created_at": draft.created_at.replace(tzinfo=UTC).isoformat(),
    }


async def save_post_draft(
    db: AsyncSession,
    user_id: UUID,
    draft_id: UUID,
    chat_id: UUID,
    chunks: AsyncIterator[bytes],
    file_name: str,
    caption: str,
) -> PostDraft:
    try:
        await _get_chat(db, user_id, chat_id)
    except ValueError:
        raise PostDraftError(404, "Chat not found") from None
    if not caption.strip() or len(caption.encode("utf-16-le")) // 2 > 1024:
        raise PostDraftError(400, "Caption must contain 1–1024 UTF-16 characters")
    data = bytearray()
    async for chunk in chunks:
        if len(data) + len(chunk) > MAX_IMAGE_BYTES:
            raise PostDraftError(413, "Post image exceeds 10 MiB")
        data.extend(chunk)
    if data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        mime_type, suffixes = "image/jpeg", {".jpg", ".jpeg"}
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type, suffixes = "image/png", {".png"}
    else:
        raise PostDraftError(400, "A JPEG or PNG image is required")
    file_name = _sanitize_file_name(file_name)
    if Path(file_name).suffix.lower() not in suffixes:
        raise PostDraftError(400, "Image extension does not match its content")
    digest = hashlib.sha256(data).hexdigest()

    async def existing() -> PostDraft:
        draft = await get_post_draft(db, user_id, draft_id)
        if (draft.chat_id, draft.caption, draft.sha256, draft.file_name) != (
            chat_id,
            caption,
            digest,
            file_name,
        ):
            raise PostDraftError(
                409, "This draft ID already has different content; use a new ID"
            )
        return draft

    if await db.scalar(select(PostDraft.id).where(PostDraft.id == draft_id)):
        return await existing()
    draft = PostDraft(
        id=draft_id,
        user_id=user_id,
        chat_id=chat_id,
        caption=caption,
        file_name=file_name,
        mime_type=mime_type,
        media=bytes(data),
        sha256=digest,
    )
    try:
        async with db.begin_nested():
            db.add(draft)
            await db.flush()
    except IntegrityError:
        return await existing()
    return draft


async def publish_post_draft(
    db: AsyncSession, user_id: UUID, draft_id: UUID
) -> PostDraft:
    draft = await get_post_draft(db, user_id, draft_id)
    if draft.status == "published":
        return draft
    # Commit the reservation before the remote side effect. A timeout or process
    # crash must never turn into an automatic second channel post.
    claimed = await db.execute(
        update(PostDraft)
        .where(
            PostDraft.id == draft_id,
            PostDraft.user_id == user_id,
            PostDraft.status == "draft",
        )
        .values(status="publishing")
    )
    if claimed.rowcount != 1:
        raise PostDraftError(
            409,
            "Publication is in progress or uncertain; inspect Telegram before retrying",
        )
    await db.commit()
    try:
        chat = await _get_chat(db, user_id, draft.chat_id)
        with _outbound_directory() as work:
            path = Path(work) / draft.file_name
            path.write_bytes(draft.media)
            result = await _send_file_path(
                db, user_id, draft.chat_id, path, draft.caption, chat
            )
    except Exception:
        draft.status = "unknown"
        await db.commit()
        raise PostDraftError(
            503, "Publication could not be confirmed; inspect Telegram, do not resend"
        ) from None
    draft.status = "published"
    draft.telegram_message_id = result["telegram_message_id"]
    await db.commit()
    return draft
