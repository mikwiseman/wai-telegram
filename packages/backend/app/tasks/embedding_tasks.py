"""Small, resumable repair passes through selected chats; no extra service."""

import redis
from celery import shared_task
from sqlalchemy import select

from app.core.async_runner import run_async
from app.core.config import get_settings
from app.core.database import get_db_context
from app.models.settings import UserSettings
from app.models.user import User
from app.services.embedding_service import embed_unembedded_messages
from app.services.indexing_service import indexing_has_capacity


async def repair_selected_embeddings() -> int:
    indexed = 0
    async with get_db_context() as db:
        user_ids = (
            (
                await db.execute(
                    select(User.id)
                    .join(UserSettings)
                    .where(
                        User.is_active.is_(True),
                        UserSettings.embedding_backfill_enabled.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
    for user_id in user_ids:
        if not indexing_has_capacity():
            break
        # Commit each small batch, releasing ORM objects and surviving worker restarts.
        for _ in range(10):
            if not indexing_has_capacity():
                return indexed
            async with get_db_context() as db:
                count = await embed_unembedded_messages(
                    db, user_id, limit=get_settings().embedding_batch_size
                )
            indexed += count
            if count == 0:
                break
    return indexed


@shared_task(soft_time_limit=240, time_limit=270)
def repair_embeddings() -> dict:
    client = redis.from_url(get_settings().redis_url)
    try:
        with client.lock("embedding:repair", timeout=300, blocking_timeout=0):
            return {"indexed": run_async(repair_selected_embeddings())}
    except redis.exceptions.LockError:
        return {"indexed": 0, "busy": True}
    finally:
        client.close()


@shared_task(soft_time_limit=540, time_limit=570)
def advance_memory_batches() -> dict:
    from app.services.memory_batch_service import run_memory_batch_pass

    if not get_settings().memory_batch_enabled:
        return {"enabled": False}
    client = redis.from_url(get_settings().redis_url)
    try:
        with client.lock("memory:batch", timeout=600, blocking_timeout=0):
            return run_async(run_memory_batch_pass())
    except redis.exceptions.LockError:
        return {"busy": True}
    finally:
        client.close()
