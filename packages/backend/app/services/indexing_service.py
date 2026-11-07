from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import TelegramChat
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.sync_job import SyncJob
from app.models.user import User

import shutil

# Leave room for PostgreSQL WAL, backups and the other services on this host.
MIN_FREE_BYTES = 8_000_000_000


def indexing_has_capacity() -> bool:
    return shutil.disk_usage("/").free >= MIN_FREE_BYTES


async def get_indexing_status(db: AsyncSession, user_id: UUID) -> dict:
    account = await db.get(User, user_id)
    eligible = (
        func.length(
            func.trim(func.coalesce(TelegramMessage.text, ""), " \t\n\r\v\f\u00a0")
        )
        > 0
    )
    counts = (
        (
            await db.execute(
                select(
                    TelegramMessage.chat_id,
                    func.count().label("messages"),
                    func.count().filter(eligible).label("text_messages"),
                    func.count()
                    .filter(eligible, TelegramMessage.embedding.is_not(None))
                    .label("indexed_texts"),
                    func.count()
                    .filter(
                        TelegramMessage.media_processing_status.in_(
                            [
                                MediaProcessingStatus.PENDING,
                                MediaProcessingStatus.QUEUED,
                                MediaProcessingStatus.PROCESSING,
                            ]
                        )
                    )
                    .label("pending_media"),
                    func.count()
                    .filter(
                        TelegramMessage.media_processing_status
                        == MediaProcessingStatus.FAILED,
                    )
                    .label("failed_media"),
                )
                .join(TelegramChat)
                .where(
                    TelegramChat.user_id == user_id,
                    TelegramChat.indexing_enabled.is_(True),
                )
                .group_by(TelegramMessage.chat_id)
            )
        )
        .mappings()
        .all()
    )
    chats = (
        (
            await db.execute(
                select(TelegramChat).where(
                    TelegramChat.user_id == user_id,
                    TelegramChat.indexing_enabled.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    by_chat = {row["chat_id"]: row for row in counts}
    jobs = (
        (
            await db.execute(
                select(SyncJob)
                .where(
                    SyncJob.user_id == user_id,
                    SyncJob.chat_id.is_(None),
                )
                .order_by(SyncJob.created_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .all()
    )
    from app.core.config import get_settings

    memory = None
    if get_settings().memory_batch_enabled:
        from app.services.memory_batch_service import memory_status

        memory = await memory_status(user_id)
        memory["search_active"] = get_settings().memory_search_enabled
    return {
        "memory": memory,
        "auto_index_since": account.auto_index_since if account else None,
        "pause_reason": None if indexing_has_capacity() else "disk_space",
        "chats": [
            {
                "id": str(chat.id),
                "title": chat.title,
                "history_complete": chat.history_synced_at is not None,
                **{
                    key: by_chat.get(chat.id, {}).get(key, 0)
                    for key in (
                        "messages",
                        "text_messages",
                        "indexed_texts",
                        "pending_media",
                        "failed_media",
                    )
                },
            }
            for chat in chats
        ],
        "sync_status": jobs[0].status.value if jobs else None,
    }
