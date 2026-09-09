from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import RequireWrite
from app.core.database import get_db
from app.core.limiter import limiter
from app.schemas.messaging import (
    ReplyMessageRequest,
    SavedFileResponse,
    SendFileRequest,
    SendFileResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.messaging_service import (
    SendTarget,
    reply_to_message,
    send_file,
    send_message,
    upload_to_saved_messages,
)

router = APIRouter()


@router.post("/{chat_id}/send", response_model=SendMessageResponse)
@limiter.limit("20/minute")
async def send_message_endpoint(
    request: Request,
    chat_id: SendTarget,
    body: SendMessageRequest,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SendMessageResponse:
    """Send a text message to a Telegram chat."""
    try:
        result = await send_message(db, ctx.user.id, chat_id, body.text)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return SendMessageResponse(**result)


@router.post("/{chat_id}/send-file", response_model=SendFileResponse)
@limiter.limit("10/minute")
async def send_file_endpoint(
    request: Request,
    chat_id: SendTarget,
    body: SendFileRequest,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SendFileResponse:
    """Download a file from URL and send it to a Telegram chat."""
    try:
        result = await send_file(
            db, ctx.user.id, chat_id, body.file_url, body.caption, body.file_name
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return SendFileResponse(**result)


@router.post("/me/upload", response_model=SavedFileResponse)
@limiter.limit("10/minute")
async def upload_saved_file_endpoint(
    request: Request,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
    file_name: Annotated[
        str, Header(alias="X-File-Name", min_length=1, max_length=2048)
    ],
    caption: Annotated[
        str | None, Header(alias="X-Telegram-Caption", max_length=16384)
    ] = None,
) -> SavedFileResponse:
    """Send raw file bytes to Saved Messages. Header values are percent-encoded UTF-8."""
    try:
        result = await upload_to_saved_messages(
            db,
            ctx.user.id,
            request.stream(),
            unquote(file_name),
            unquote(caption) if caption else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return SavedFileResponse(**result)


@router.post("/{chat_id}/reply", response_model=SendMessageResponse)
@limiter.limit("20/minute")
async def reply_message_endpoint(
    request: Request,
    chat_id: UUID,
    body: ReplyMessageRequest,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SendMessageResponse:
    """Reply to a specific message in a Telegram chat."""
    try:
        result = await reply_to_message(
            db, ctx.user.id, chat_id, body.telegram_message_id, body.text
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return SendMessageResponse(**result)
