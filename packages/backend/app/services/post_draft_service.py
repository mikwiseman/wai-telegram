"""Store complete posts without Telegram side effects; publish only on a separate call."""

import base64
import binascii
import hashlib
import mimetypes
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import httpx
import jwt
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.post_draft import PostDraft
from app.services.messaging_service import (
    _download_url_to_path,
    _get_chat,
    _outbound_directory,
    _sanitize_file_name,
    _send_file_path,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_FILE_BYTES = 10 * 1024 * 1024
DRAFT_DOWNLOAD_TTL = timedelta(minutes=15)


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


def create_draft_download_token(
    user_id: UUID, draft_id: UUID, *, expires_at: datetime | None = None
) -> str:
    return jwt.encode(
        {
            "typ": "wai-file-draft",
            "sub": str(user_id),
            "draft_id": str(draft_id),
            "exp": expires_at or datetime.now(UTC) + DRAFT_DOWNLOAD_TTL,
        },
        get_settings().secret_key,
        algorithm="HS256",
    )


def draft_download_owner(token: str, draft_id: UUID) -> UUID:
    try:
        claims = jwt.decode(
            token,
            get_settings().secret_key,
            algorithms=["HS256"],
            options={"require": ["typ", "sub", "draft_id", "exp"]},
        )
        if claims["typ"] != "wai-file-draft" or claims["draft_id"] != str(draft_id):
            raise ValueError
        return UUID(claims["sub"])
    except (jwt.InvalidTokenError, ValueError, TypeError, KeyError):
        raise PostDraftError(401, "Invalid or expired draft download token") from None


def file_draft_metadata(draft: PostDraft) -> dict:
    expiry = datetime.now(UTC) + DRAFT_DOWNLOAD_TTL
    token = create_draft_download_token(draft.user_id, draft.id, expires_at=expiry)
    return {
        **metadata(draft),
        "has_media": True,
        "media_file_name": draft.file_name,
        "media_mime_type": draft.mime_type,
        "media_file_size": len(draft.media),
        "media_download_url": f"/api/v1/post-drafts/{draft.id}/download?token={quote(token, safe='')}",
        "download_url_expires_at": expiry.isoformat(),
    }


async def save_post_draft(
    db: AsyncSession,
    user_id: UUID,
    draft_id: UUID,
    chat_id: UUID,
    chunks: AsyncIterator[bytes],
    file_name: str,
    caption: str,
    *,
    image_only: bool = True,
) -> PostDraft:
    try:
        await _get_chat(db, user_id, chat_id)
    except ValueError:
        raise PostDraftError(404, "Chat not found") from None
    if not caption.strip() or len(caption.encode("utf-16-le")) // 2 > 1024:
        raise PostDraftError(400, "Caption must contain 1–1024 UTF-16 characters")
    data = bytearray()
    limit = MAX_IMAGE_BYTES if image_only else MAX_FILE_BYTES
    async for chunk in chunks:
        if len(data) + len(chunk) > limit:
            raise PostDraftError(413, "Draft attachment exceeds 10 MiB")
        data.extend(chunk)
    file_name = _sanitize_file_name(file_name)
    if not image_only:
        if not data:
            raise PostDraftError(400, "Draft attachment must not be empty")
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    elif data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        mime_type, suffixes = "image/jpeg", {".jpg", ".jpeg"}
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type, suffixes = "image/png", {".png"}
    else:
        raise PostDraftError(400, "A JPEG or PNG image is required")
    if image_only and Path(file_name).suffix.lower() not in suffixes:
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


async def save_file_draft_from_url(
    db: AsyncSession,
    user_id: UUID,
    draft_id: UUID,
    chat_id: UUID,
    file_url: str,
    file_name: str,
    text: str,
) -> PostDraft:
    # Authorize the recipient before fetching any remote content.
    try:
        await _get_chat(db, user_id, chat_id)
    except ValueError:
        raise PostDraftError(404, "Chat not found") from None
    with _outbound_directory() as work:
        path = Path(work) / _sanitize_file_name(file_name)
        try:
            await _download_url_to_path(file_url, path, max_bytes=MAX_FILE_BYTES)
        except httpx.HTTPError:
            raise PostDraftError(
                400, "Could not download the draft attachment"
            ) from None
        except ValueError as exc:
            raise PostDraftError(400, str(exc)) from None

        async def chunks():
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    yield chunk

        return await save_post_draft(
            db, user_id, draft_id, chat_id, chunks(), file_name, text, image_only=False
        )


async def save_file_draft_from_base64(
    db: AsyncSession,
    user_id: UUID,
    draft_id: UUID,
    chat_id: UUID,
    encoded: str,
    file_name: str,
    text: str,
) -> PostDraft:
    if len(encoded) > ((MAX_FILE_BYTES + 2) // 3) * 4:
        raise PostDraftError(413, "Draft attachment exceeds 10 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise PostDraftError(400, "file_base64 must contain valid base64") from None

    async def chunks():
        yield data

    return await save_post_draft(
        db, user_id, draft_id, chat_id, chunks(), file_name, text, image_only=False
    )


async def list_file_drafts(
    db: AsyncSession, user_id: UUID, chat_id: UUID | None, limit: int
) -> dict:
    query = select(PostDraft).where(
        PostDraft.user_id == user_id, PostDraft.status == "draft"
    )
    if chat_id is not None:
        query = query.where(PostDraft.chat_id == chat_id)
    rows = (
        await db.scalars(
            query.order_by(PostDraft.created_at.desc(), PostDraft.id.desc()).limit(
                limit + 1
            )
        )
    ).all()
    return {
        "drafts": [file_draft_metadata(draft) for draft in rows[:limit]],
        "has_more": len(rows) > limit,
    }


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
