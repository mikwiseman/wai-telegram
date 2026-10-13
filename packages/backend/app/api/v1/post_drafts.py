from typing import Annotated, Literal
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, RequireDraft, RequireWrite, require_scope
from app.core.database import get_db
from app.core.limiter import limiter
from app.services.post_draft_service import (
    PostDraftError,
    get_post_draft,
    metadata,
    publish_post_draft,
    save_post_draft,
)

router = APIRouter()
ReadAccess = Annotated[AuthContext, Depends(require_scope("read"))]
Database = Annotated[AsyncSession, Depends(get_db)]


class PublishApproval(BaseModel):
    approve: Literal[True]


@router.put("/{draft_id}")
@limiter.limit("10/minute")
async def store_post(
    request: Request,
    draft_id: UUID,
    ctx: RequireDraft,
    db: Database,
    chat_id: Annotated[UUID, Header(alias="X-Chat-ID")],
    file_name: Annotated[str, Header(alias="X-File-Name", max_length=2048)],
    caption: Annotated[str, Header(alias="X-Telegram-Caption", max_length=16384)],
):
    """Save immutable image+Markdown caption in Wai. Never contacts Telegram."""
    try:
        draft = await save_post_draft(
            db,
            ctx.user.id,
            draft_id,
            chat_id,
            request.stream(),
            unquote(file_name),
            unquote(caption),
        )
        return metadata(draft)
    except PostDraftError as error:
        raise HTTPException(error.status_code, str(error)) from None


@router.get("/{draft_id}")
async def read_post(draft_id: UUID, ctx: ReadAccess, db: Database):
    try:
        return metadata(await get_post_draft(db, ctx.user.id, draft_id))
    except PostDraftError as error:
        raise HTTPException(error.status_code, str(error)) from None


@router.get("/{draft_id}/media")
async def read_post_media(draft_id: UUID, ctx: ReadAccess, db: Database):
    try:
        draft = await get_post_draft(db, ctx.user.id, draft_id)
        return Response(
            draft.media,
            media_type=draft.mime_type,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    except PostDraftError as error:
        raise HTTPException(error.status_code, str(error)) from None


@router.post("/{draft_id}/publish")
@limiter.limit("5/minute")
async def publish_post(
    request: Request,
    draft_id: UUID,
    body: PublishApproval,
    ctx: RequireWrite,
    db: Database,
):
    """Send the exact stored post after the owner explicitly approves publication."""
    try:
        return metadata(await publish_post_draft(db, ctx.user.id, draft_id))
    except PostDraftError as error:
        raise HTTPException(error.status_code, str(error)) from None
