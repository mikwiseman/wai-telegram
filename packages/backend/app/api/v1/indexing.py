import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, RequireWrite
from app.core.database import get_db
from app.core.limiter import limiter
from app.models.chat import TelegramChat
from app.models.settings import UserSettings
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User
from app.services.sync_service import create_sync_job
from app.services.indexing_service import get_indexing_status
from app.services.telegram_client import start_listener
from app.tasks.sync_tasks import sync_all_chats_task

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatSelection(BaseModel):
    chat_ids: list[UUID] = Field(max_length=1000)


@router.get("/status")
async def indexing_status(
    user: CurrentUser, db: Annotated[AsyncSession, Depends(get_db)]
) -> dict:
    return await get_indexing_status(db, user.id)


@router.put("/chats")
@limiter.limit("10/minute")
async def select_chats(
    request: Request,
    body: ChatSelection,
    ctx: RequireWrite,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    selected = set(body.chat_ids)
    await db.execute(select(User.id).where(User.id == ctx.user.id).with_for_update())
    chats = (
        (
            await db.execute(
                select(TelegramChat)
                .where(
                    TelegramChat.user_id == ctx.user.id,
                )
                .with_for_update()
            )
        )
        .scalars()
        .all()
    )
    if not selected.issubset({chat.id for chat in chats}):
        raise HTTPException(404, "Chat not found")
    for chat in chats:
        if chat.id in selected and not chat.indexing_enabled:
            # New messages were ignored while disabled; rescan with deduplication.
            chat.history_synced_at = None
            chat.history_cursor = None
        chat.indexing_enabled = chat.id in selected
    preferences = (
        await db.execute(
            select(UserSettings).where(
                UserSettings.user_id == ctx.user.id,
            )
        )
    ).scalar_one_or_none()
    if preferences is None:
        preferences = UserSettings(user_id=ctx.user.id, digest_enabled=False)
        db.add(preferences)
    preferences.embedding_backfill_enabled = bool(selected)
    if selected:
        preferences.realtime_sync_enabled = True
    await db.flush()
    pending = (
        await db.execute(
            select(SyncJob)
            .where(
                SyncJob.user_id == ctx.user.id,
                SyncJob.chat_id.is_(None),
                SyncJob.status.in_([SyncStatus.PENDING, SyncStatus.IN_PROGRESS]),
            )
            .order_by(SyncJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    needs_history = any(
        chat.id in selected and chat.history_synced_at is None for chat in chats
    )
    start = needs_history and pending is None
    if start:
        pending = await create_sync_job(db, ctx.user.id)
    await db.commit()
    if selected:
        await start_listener(ctx.user.id)
    if start:
        try:
            sync_all_chats_task.delay(str(ctx.user.id), str(pending.id), 0, True)
        except Exception:
            logger.warning("History dispatch unavailable; durable job will be retried")
    return {"selected": len(selected), "job_id": str(pending.id) if pending else None}
