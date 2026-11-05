from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, OptionalUser, RequireWrite
from app.core.cursor import (
    CursorError,
    decode_cursor,
    encode_cursor,
    parse_cursor_datetime,
)
from app.core.config import get_settings
from app.core.database import get_db
from app.core.limiter import limiter
from app.models.chat import ChatType, TelegramChat
from app.models.media import MediaObject, MediaObjectStatus, TranscriptSegment
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.schemas.chat import ChatListResponse, ChatResponse
from app.schemas.message import (
    MessageContentResponse,
    MessageListResponse,
    MessageResponse,
    MediaPrepareResponse,
    TranscriptSegmentPage,
    TranscriptSegmentResponse,
)
from app.services.media_cache_service import (
    MediaCacheError,
    get_cached_media_for_download,
    get_or_create_media_object,
    media_preparation_needs_enqueue,
    request_transcription,
)
from app.services.media_access import decode_media_download_token
from app.services.media_stream_service import (
    MediaStreamUnavailable,
    open_media_stream,
)
from app.services.telegram_client import NoActiveTelegramSessionError
from app.services.sync_service import sync_chats
from app.services.telegram_links import (
    build_media_download_url,
    build_telegram_message_url,
    media_download_filename,
)
from app.tasks.media_tasks import enqueue_media_processing

router = APIRouter()
_CURSOR_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _decode_chat_cursor(cursor: str) -> dict[str, Any]:
    try:
        payload = decode_cursor(cursor)
        raw_last_message_id = payload.get("last_message_id")
        if raw_last_message_id is not None:
            raw_last_message_id = int(raw_last_message_id)
        return {
            "last_activity_at": parse_cursor_datetime(payload.get("last_activity_at")),
            "last_message_id": raw_last_message_id,
            "id": UUID(payload["id"]),
        }
    except (KeyError, TypeError, ValueError, CursorError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat cursor",
        ) from exc


def _encode_chat_cursor(chat: TelegramChat) -> str:
    return encode_cursor(
        {
            "last_activity_at": chat.last_activity_at.isoformat()
            if chat.last_activity_at
            else None,
            "last_message_id": chat.last_message_id,
            "id": str(chat.id),
        }
    )


def _decode_message_cursor(cursor: str) -> dict[str, Any]:
    try:
        payload = decode_cursor(cursor)
        sent_at = parse_cursor_datetime(payload["sent_at"])
        if sent_at is None:
            raise CursorError("Missing sent_at")
        return {
            "sent_at": sent_at,
            "telegram_message_id": int(payload["telegram_message_id"]),
            "id": UUID(payload["id"]),
        }
    except (KeyError, TypeError, ValueError, CursorError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid message cursor",
        ) from exc


def _encode_message_cursor(message: TelegramMessage) -> str:
    return encode_cursor(
        {
            "sent_at": message.sent_at.isoformat(),
            "telegram_message_id": message.telegram_message_id,
            "id": str(message.id),
        }
    )


def _message_download_url(
    chat_id: UUID, telegram_message_id: int, user_id: UUID
) -> str:
    return build_media_download_url(
        base_path=(f"/api/v1/chats/{chat_id}/messages/{telegram_message_id}/media"),
        user_id=user_id,
        chat_id=chat_id,
        telegram_message_id=telegram_message_id,
    )


@router.get("", response_model=ChatListResponse)
async def list_chats(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_type: ChatType | None = None,
    telegram_chat_id: int | None = Query(default=None, ge=-(2**63), le=2**63 - 1),
    unread_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
    cursor: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
) -> ChatListResponse:
    """List user's synced chats (cursor pagination, offset fallback)."""
    query = select(TelegramChat).where(TelegramChat.user_id == user.id)

    if telegram_chat_id is not None:
        query = query.where(TelegramChat.telegram_chat_id == telegram_chat_id)
    if chat_type:
        query = query.where(TelegramChat.chat_type == chat_type)
    if unread_only:
        query = query.where(func.coalesce(TelegramChat.unread_count, 0) > 0)

    query = query.order_by(
        TelegramChat.last_activity_at.desc().nulls_last(),
        TelegramChat.last_message_id.desc().nulls_last(),
        TelegramChat.id.desc(),
    )

    if cursor:
        c = _decode_chat_cursor(cursor)
        query = query.where(
            tuple_(
                func.coalesce(TelegramChat.last_activity_at, _CURSOR_EPOCH),
                func.coalesce(TelegramChat.last_message_id, -1),
                TelegramChat.id,
            )
            < (
                c["last_activity_at"] or _CURSOR_EPOCH,
                c["last_message_id"] if c["last_message_id"] is not None else -1,
                c["id"],
            )
        )
    elif offset:
        # Backward compatibility fallback for clients still using offset.
        query = query.offset(offset)

    query = query.limit(limit + 1)

    result = await db.execute(query)
    chats = result.scalars().all()
    has_more = len(chats) > limit
    if has_more:
        chats = chats[:limit]
    next_cursor = _encode_chat_cursor(chats[-1]) if has_more and chats else None

    # Backward compatibility total count.
    count_query = (
        select(func.count())
        .select_from(TelegramChat)
        .where(TelegramChat.user_id == user.id)
    )
    if telegram_chat_id is not None:
        count_query = count_query.where(
            TelegramChat.telegram_chat_id == telegram_chat_id
        )
    if chat_type:
        count_query = count_query.where(TelegramChat.chat_type == chat_type)
    if unread_only:
        count_query = count_query.where(func.coalesce(TelegramChat.unread_count, 0) > 0)
    total = (await db.execute(count_query)).scalar()

    return ChatListResponse(
        chats=[ChatResponse.model_validate(chat) for chat in chats],
        has_more=has_more,
        next_cursor=next_cursor,
        total=total,
    )


@router.post("/refresh", response_model=ChatListResponse)
@limiter.limit("10/minute")
async def refresh_chats(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatListResponse:
    """Refresh chat list from Telegram."""
    try:
        chats = await sync_chats(db, user.id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return ChatListResponse(
        chats=[ChatResponse.model_validate(chat) for chat in chats],
        has_more=False,
        next_cursor=None,
        total=len(chats),
    )


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatResponse:
    """Get chat details."""
    result = await db.execute(
        select(TelegramChat).where(
            TelegramChat.id == chat_id,
            TelegramChat.user_id == user.id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found"
        )
    return ChatResponse.model_validate(chat)


@router.get("/{chat_id}/messages", response_model=MessageListResponse)
async def get_chat_messages(
    chat_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=500),
    before: str | None = Query(default=None),
    after: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
) -> MessageListResponse:
    """Get messages for a chat (cursor pagination, offset fallback)."""
    # Verify chat ownership and get chat metadata
    result = await db.execute(
        select(TelegramChat).where(
            TelegramChat.id == chat_id,
            TelegramChat.user_id == user.id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found"
        )

    if after:
        # Fetch messages NEWER than the cursor (ascending, then reverse)
        c = _decode_message_cursor(after)
        query = (
            select(TelegramMessage)
            .where(TelegramMessage.chat_id == chat_id)
            .where(
                tuple_(
                    TelegramMessage.sent_at,
                    TelegramMessage.telegram_message_id,
                    TelegramMessage.id,
                )
                > (c["sent_at"], c["telegram_message_id"], c["id"])
            )
            .order_by(
                TelegramMessage.sent_at.asc(),
                TelegramMessage.telegram_message_id.asc(),
                TelegramMessage.id.asc(),
            )
            .limit(limit + 1)
        )
        result = await db.execute(query)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        # Reverse back to newest-first for consistent response format
        messages.reverse()
        next_cursor = None  # after-based queries don't paginate backwards
        newest_cursor = _encode_message_cursor(messages[0]) if messages else None
    else:
        # Standard newest-first pagination
        query = (
            select(TelegramMessage)
            .where(TelegramMessage.chat_id == chat_id)
            .order_by(
                TelegramMessage.sent_at.desc(),
                TelegramMessage.telegram_message_id.desc(),
                TelegramMessage.id.desc(),
            )
            .limit(limit + 1)
        )

        if before:
            c = _decode_message_cursor(before)
            query = query.where(
                tuple_(
                    TelegramMessage.sent_at,
                    TelegramMessage.telegram_message_id,
                    TelegramMessage.id,
                )
                < (c["sent_at"], c["telegram_message_id"], c["id"])
            )
        elif offset:
            query = query.offset(offset)

        result = await db.execute(query)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        next_cursor = (
            _encode_message_cursor(messages[-1]) if has_more and messages else None
        )
        # newest_cursor is the cursor of the first (newest) message on the first page
        newest_cursor = (
            _encode_message_cursor(messages[0])
            if messages and not before and not offset
            else None
        )

    media_objects = {}
    if messages:
        media_objects = {
            item.message_id: item
            for item in (
                (
                    await db.execute(
                        select(MediaObject).where(
                            MediaObject.message_id.in_([msg.id for msg in messages])
                        )
                    )
                )
                .scalars()
                .all()
            )
        }

    return MessageListResponse(
        messages=[
            MessageResponse(
                id=msg.id,
                telegram_message_id=msg.telegram_message_id,
                text=msg.text,
                has_media=msg.has_media,
                media_type=msg.media_type,
                media_file_name=msg.media_file_name,
                media_mime_type=msg.media_mime_type,
                media_file_size=msg.media_file_size,
                media_duration_seconds=msg.media_duration_seconds,
                content_summary=msg.content_summary,
                content_preview=msg.content_text[:1200] if msg.content_text else None,
                media_processing_status=msg.media_processing_status,
                media_processing_error_code=msg.media_processing_error_code,
                sender_id=msg.sender_id,
                sender_name=msg.sender_name,
                is_outgoing=msg.is_outgoing,
                sent_at=msg.sent_at,
                has_embedding=msg.embedding is not None,
                transcribed_at=msg.transcribed_at,
                telegram_message_url=build_telegram_message_url(
                    chat_type=chat.chat_type,
                    telegram_chat_id=chat.telegram_chat_id,
                    username=chat.username,
                    message_id=msg.telegram_message_id,
                ),
                media_download_url=(
                    _message_download_url(chat.id, msg.telegram_message_id, user.id)
                    if (
                        msg.id in media_objects
                        and media_objects[msg.id].relative_path
                        and media_objects[msg.id].sha256
                    )
                    else None
                ),
                media_cache_status=(
                    str(media_objects[msg.id].status)
                    if msg.id in media_objects
                    else None
                ),
                media_cache_stage=(
                    str(media_objects[msg.id].stage)
                    if msg.id in media_objects
                    else None
                ),
                media_cached_bytes=(
                    media_objects[msg.id].byte_offset
                    if msg.id in media_objects
                    else None
                ),
                media_sha256=(
                    media_objects[msg.id].sha256 if msg.id in media_objects else None
                ),
                visible_urls=msg.visible_urls or [],
                hidden_urls=msg.hidden_urls or [],
                edited_at=msg.edited_at,
                deleted_at=msg.deleted_at,
            )
            for msg in messages
        ],
        total=None,
        has_more=has_more,
        next_cursor=next_cursor,
        newest_cursor=newest_cursor,
        total_messages_synced=chat.total_messages_synced,
        last_sync_at=chat.last_sync_at,
    )


@router.get(
    "/{chat_id}/messages/{telegram_message_id}/transcript",
    response_model=TranscriptSegmentPage,
)
async def get_message_transcript(
    chat_id: UUID,
    telegram_message_id: int,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> TranscriptSegmentPage:
    message_id = (
        await db.execute(
            select(TelegramMessage.id)
            .join(TelegramChat)
            .where(
                TelegramMessage.chat_id == chat_id,
                TelegramMessage.telegram_message_id == telegram_message_id,
                TelegramChat.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if message_id is None:
        raise HTTPException(status_code=404, detail="Message not found")
    rows = list(
        (
            await db.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.message_id == message_id,
                    TranscriptSegment.sequence >= cursor,
                )
                .order_by(TranscriptSegment.sequence)
                .limit(limit + 1)
            )
        ).scalars()
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return TranscriptSegmentPage(
        segments=[TranscriptSegmentResponse.model_validate(row) for row in rows],
        has_more=has_more,
        next_cursor=rows[-1].sequence + 1 if has_more and rows else None,
    )


@router.get(
    "/{chat_id}/messages/{telegram_message_id}/content",
    response_model=MessageContentResponse,
)
async def get_message_content(
    chat_id: UUID,
    telegram_message_id: int,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageContentResponse:
    """Return the complete processed content for one media message."""
    result = await db.execute(
        select(TelegramMessage, TelegramChat, MediaObject)
        .join(TelegramChat)
        .outerjoin(MediaObject, MediaObject.message_id == TelegramMessage.id)
        .where(
            TelegramMessage.chat_id == chat_id,
            TelegramMessage.telegram_message_id == telegram_message_id,
            TelegramChat.user_id == user.id,
        )
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )
    message, chat, media_object = row

    cache_ready = bool(
        media_object and media_object.relative_path and media_object.sha256
    )
    if cache_ready:
        next_action = None
    elif media_object and media_object.status in {
        MediaObjectStatus.FETCHING,
        MediaObjectStatus.EXTRACTING,
        MediaObjectStatus.INDEXING,
        MediaObjectStatus.PROCESSING,
        MediaObjectStatus.RETRY_WAIT,
    }:
        next_action = "Call prepare_media again after retry_after to refresh progress"
    elif message.has_media:
        next_action = "Call prepare_media to fetch and process the original media"
    else:
        next_action = None

    return MessageContentResponse(
        id=message.id,
        telegram_message_id=message.telegram_message_id,
        text=message.text,
        media_type=message.media_type,
        media_file_name=message.media_file_name,
        media_mime_type=message.media_mime_type,
        media_file_size=message.media_file_size,
        media_duration_seconds=message.media_duration_seconds,
        content_text=message.content_text,
        content_summary=message.content_summary,
        media_processing_status=message.media_processing_status,
        media_processing_error_code=message.media_processing_error_code,
        media_processing_error=message.media_processing_error,
        transcribed_at=message.transcribed_at,
        media_processed_at=message.media_processed_at,
        content_model=message.content_model,
        summary_model=message.summary_model,
        telegram_message_url=build_telegram_message_url(
            chat_type=chat.chat_type,
            telegram_chat_id=chat.telegram_chat_id,
            username=chat.username,
            message_id=message.telegram_message_id,
        ),
        media_download_url=(
            _message_download_url(chat.id, message.telegram_message_id, user.id)
            if cache_ready
            else None
        ),
        media_cache_status=str(media_object.status) if media_object else None,
        media_cache_stage=str(media_object.stage) if media_object else None,
        media_sha256=media_object.sha256 if media_object else None,
        media_cached_bytes=media_object.byte_offset if media_object else None,
        next_action=next_action,
    )


@router.post(
    "/{chat_id}/messages/{telegram_message_id}/prepare",
    response_model=MediaPrepareResponse,
)
async def prepare_message_media(
    chat_id: UUID,
    telegram_message_id: int,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MediaPrepareResponse:
    user = ctx.user
    message = (
        await db.execute(
            select(TelegramMessage)
            .join(TelegramChat)
            .where(
                TelegramMessage.chat_id == chat_id,
                TelegramMessage.telegram_message_id == telegram_message_id,
                TelegramChat.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if not message.has_media:
        raise HTTPException(status_code=404, detail="Message has no media")
    if not get_settings().media_pipeline_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "media_pipeline_deferred",
                "message": (
                    "Durable media processing is deferred until storage is attached"
                ),
            },
        )

    media_object = await get_or_create_media_object(db, user.id, message.id)
    request_transcription(message, media_object)
    cache_ready = bool(media_object.relative_path and media_object.sha256)
    processing_ready = media_object.status in {
        MediaObjectStatus.READY,
        MediaObjectStatus.READY_DOWNLOAD_ONLY,
    }
    if media_preparation_needs_enqueue(message, media_object):
        message.media_processing_status = MediaProcessingStatus.PENDING
        message.media_processing_error_code = None
        message.media_processing_error = None
        media_object.status = MediaObjectStatus.PENDING
        media_object.error_code = None
        media_object.error_detail = None
        await db.commit()
        enqueue_media_processing([message.id])
    elif not processing_ready:
        await db.commit()

    return MediaPrepareResponse(
        message_id=message.id,
        status=str(media_object.status),
        stage=str(media_object.stage),
        byte_offset=media_object.byte_offset,
        size_bytes=media_object.size_bytes,
        sha256=media_object.sha256,
        retry_after=media_object.retry_after,
        error_code=media_object.error_code,
        error_detail=media_object.error_detail,
        media_download_url=(
            _message_download_url(chat_id, telegram_message_id, user.id)
            if cache_ready
            else None
        ),
        next_action=(
            "Call download_media"
            if cache_ready
            else "Call prepare_media again to refresh progress"
        ),
    )


def _content_disposition(file_name: str) -> str:
    ascii_name = file_name.encode("ascii", "ignore").decode() or "telegram-media"
    return (
        f'attachment; filename="{ascii_name.replace(chr(34), "")}"; '
        f"filename*=UTF-8''{quote(file_name)}"
    )


def _requested_offset(request: Request) -> int:
    """Read a resume point out of a Range header, ignoring anything exotic."""
    header = request.headers.get("range") or ""
    if not header.startswith("bytes="):
        return 0
    start = header.removeprefix("bytes=").split("-", 1)[0].strip()
    return int(start) if start.isdigit() else 0


async def _stream_media_response(
    request: Request,
    db: AsyncSession,
    owner_id: UUID,
    chat_id: UUID,
    telegram_message_id: int,
) -> Response:
    """Pipe the original from Telegram, so nothing has to be staged first."""
    offset = _requested_offset(request)
    try:
        media, chunks = await open_media_stream(
            db, owner_id, chat_id, telegram_message_id, offset=offset
        )
    except MediaStreamUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except NoActiveTelegramSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reconnect Telegram to download originals",
        ) from exc

    headers = {
        "Content-Disposition": _content_disposition(media.file_name),
        "Cache-Control": "private, no-store",
        "Accept-Ranges": "bytes",
    }
    code = status.HTTP_200_OK
    if media.size_bytes is not None:
        remaining = max(media.size_bytes - offset, 0)
        headers["Content-Length"] = str(remaining)
        if offset:
            headers["Content-Range"] = (
                f"bytes {offset}-{media.size_bytes - 1}/{media.size_bytes}"
            )
            code = status.HTTP_206_PARTIAL_CONTENT
    if request.method == "HEAD":
        return Response(status_code=code, headers=headers, media_type=media.mime_type)
    return StreamingResponse(
        chunks, status_code=code, headers=headers, media_type=media.mime_type
    )


@router.get("/{chat_id}/messages/{telegram_message_id}/media")
@router.head("/{chat_id}/messages/{telegram_message_id}/media")
async def download_message_media(
    request: Request,
    chat_id: UUID,
    telegram_message_id: int,
    user: OptionalUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    token: str | None = Query(default=None),
) -> Response:
    """Authorize a cached object; Nginx serves bytes with Range/sendfile."""
    claims = None
    if token:
        try:
            claims = decode_media_download_token(token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired media download link",
            ) from exc
        if (
            claims.chat_id != chat_id
            or claims.telegram_message_id != telegram_message_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Media download link does not match this message",
            )

    if claims is not None and user is not None and claims.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Media download link belongs to another user",
        )
    owner_id = claims.user_id if claims is not None else (user.id if user else None)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    from app.services.single_user import is_user_active

    if not await is_user_active(db, owner_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired media download link",
        )

    try:
        cached = await get_cached_media_for_download(
            db,
            owner_id,
            chat_id,
            telegram_message_id,
        )
    except MediaCacheError:
        # The row claimed a file the volume no longer has. Telegram still does.
        cached = None
    if cached is None or cached.relative_path is None or cached.sha256 is None:
        return await _stream_media_response(
            request, db, owner_id, chat_id, telegram_message_id
        )

    file_name = media_download_filename(cached.file_name, cached.mime_type)
    ascii_name = file_name.encode("ascii", "ignore").decode() or "telegram-media"
    content_disposition = (
        f'attachment; filename="{ascii_name.replace(chr(34), "")}"; '
        f"filename*=UTF-8''{quote(file_name)}"
    )
    internal_uri = (
        f"{get_settings().media_internal_uri_prefix.rstrip('/')}/{cached.relative_path}"
    )
    return Response(
        status_code=status.HTTP_200_OK,
        headers={
            "X-Accel-Redirect": internal_uri,
            "Content-Type": cached.mime_type or "application/octet-stream",
            "Content-Disposition": content_disposition,
            "ETag": f'"{cached.sha256}"',
            "Cache-Control": "private, no-store",
        },
    )
