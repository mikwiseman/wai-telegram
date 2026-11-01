import asyncio
import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from telethon.tl.types import (
    Channel,
    Chat,
    Message,
)
from telethon.tl.types import (
    User as TelegramUser,
)
from telethon.utils import get_peer_id

from app.core.config import get_settings
from app.models.chat import ChatType, TelegramChat
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User
from app.services.embedding_service import embed_messages
from app.services.indexing_service import indexing_has_capacity
from app.services.media_content_service import get_media_info
from app.services.messaging_service import _resolve_chat_entity
from app.services.rate_limiter import record_request
from app.services.telegram_client import (
    SESSION_EXPIRED_MESSAGE,
    TelegramSessionUnauthorizedError,
    get_client,
    invalidate_client_authorization,
    is_session_authorization_error,
)
from app.services.telegram_metadata import extract_message_metadata

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_chat_type(dialog) -> ChatType:
    """Determine chat type from Telethon dialog entity."""
    entity = dialog.entity
    if isinstance(entity, TelegramUser):
        return ChatType.PRIVATE
    elif isinstance(entity, Chat):
        return ChatType.GROUP
    elif isinstance(entity, Channel):
        if entity.megagroup:
            return ChatType.SUPERGROUP
        return ChatType.CHANNEL
    return ChatType.GROUP


def _get_chat_title(dialog) -> str:
    """Get chat title from dialog entity."""
    entity = dialog.entity
    if isinstance(entity, TelegramUser):
        parts = [entity.first_name or "", entity.last_name or ""]
        return " ".join(p for p in parts if p) or "Unknown"
    return getattr(entity, "title", "Unknown")


def _get_sender_name(message: Message) -> str | None:
    """Extract sender name from message."""
    if not message.sender:
        return None
    sender = message.sender
    if isinstance(sender, TelegramUser):
        parts = [sender.first_name or "", sender.last_name or ""]
        return " ".join(p for p in parts if p) or None
    return getattr(sender, "title", None)


def _get_media_type(message: Message) -> str | None:
    """Get media type from message."""
    info = get_media_info(message)
    return info.media_type if info else None


def _media_values(message: Message, *, queue_processing: bool = False) -> dict:
    info = get_media_info(message)
    if info is None:
        return {
            "has_media": bool(message.media),
            "media_type": None,
            "media_file_name": None,
            "media_mime_type": None,
            "media_file_size": None,
            "media_duration_seconds": None,
            "media_processing_status": None,
        }
    processable = queue_processing and info.media_type != "other"
    return {
        "has_media": True,
        "media_type": info.media_type,
        "media_file_name": info.file_name,
        "media_mime_type": info.mime_type,
        "media_file_size": info.file_size,
        "media_duration_seconds": info.duration_seconds,
        "media_processing_status": (
            MediaProcessingStatus.PENDING if processable else None
        ),
    }


async def _insert_message_batch(
    db: AsyncSession,
    values: list[dict],
) -> list[UUID]:
    stmt = (
        pg_insert(TelegramMessage)
        .values(values)
        .on_conflict_do_nothing(constraint="uq_telegram_messages_chat_msg")
        .returning(TelegramMessage.id)
    )
    rows = (await db.execute(stmt)).all()
    return [row.id for row in rows]


async def _jittered_sleep(base: float, jitter: float) -> None:
    """Sleep for base +/- jitter seconds."""
    delay = base + random.uniform(-jitter, jitter)
    await asyncio.sleep(max(0.1, delay))


async def sync_chats(db: AsyncSession, user_id: UUID) -> list[TelegramChat]:
    """Sync user's chat list from Telegram using upsert for atomicity."""
    client = await get_client(user_id, db)
    chats = []
    account = await db.get(User, user_id)
    automatic = bool(account and account.auto_index_since is not None)
    index_new_chats = bool(account and (account.invited_by_id is None or automatic))
    try:
        record_request()  # Track iter_dialogs API call

        dialog_limit = settings.sync_dialog_limit or None
        async for dialog in client.iter_dialogs(limit=dialog_limit):
            # Extract last message preview
            last_msg_text = None
            last_msg_sender = None
            if dialog.message:
                msg = dialog.message
                if msg.message:
                    last_msg_text = msg.message[:200]
                elif msg.media:
                    media_label = _get_media_type(msg)
                    last_msg_text = f"[{media_label}]" if media_label else "[media]"
                last_msg_sender = _get_sender_name(msg)

            values = {
                "user_id": user_id,
                "indexing_enabled": index_new_chats,
                # Store the canonical peer ID used by Telethon events. Raw
                # entity IDs are positive for legacy groups and channels,
                # while event.chat_id is negative (or -100-prefixed). Mixing
                # the two forms creates duplicate rows for the same dialog.
                "telegram_chat_id": get_peer_id(dialog.entity),
                "access_hash": getattr(dialog.entity, "access_hash", None),
                "chat_type": _get_chat_type(dialog),
                "title": _get_chat_title(dialog),
                "username": getattr(dialog.entity, "username", None),
                "last_activity_at": dialog.date,
                "last_message_text": last_msg_text,
                "last_message_sender_name": last_msg_sender,
                "unread_count": dialog.unread_count,
            }
            stmt = pg_insert(TelegramChat).values(**values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_telegram_chats_user_chat",
                set_={
                    **({"indexing_enabled": True} if automatic else {}),
                    "access_hash": func.coalesce(
                        stmt.excluded.access_hash,
                        TelegramChat.access_hash,
                    ),
                    "title": stmt.excluded.title,
                    "username": stmt.excluded.username,
                    "last_activity_at": stmt.excluded.last_activity_at,
                    # Telegram dialog payloads sometimes omit preview fields.
                    # Keep existing preview when the incoming value is null.
                    "last_message_text": func.coalesce(
                        stmt.excluded.last_message_text,
                        TelegramChat.last_message_text,
                    ),
                    "last_message_sender_name": func.coalesce(
                        stmt.excluded.last_message_sender_name,
                        TelegramChat.last_message_sender_name,
                    ),
                    "unread_count": func.coalesce(
                        stmt.excluded.unread_count,
                        TelegramChat.unread_count,
                        0,
                    ),
                },
            ).returning(TelegramChat)
            result = await db.execute(stmt)
            chat = result.scalar_one()
            chats.append(chat)

        await db.flush()
    except Exception as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        raise
    finally:
        await client.disconnect()
    return chats


async def sync_messages(
    db: AsyncSession,
    user_id: UUID,
    chat_id: UUID,
    job_id: UUID,
    limit: int | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
    client: "TelegramClient | None" = None,
) -> int:
    """Sync messages for a specific chat. Returns count of messages synced.

    If client is provided (e.g. from the listener service), it is used directly
    instead of calling get_client().
    """

    # Get chat
    result = await db.execute(
        select(TelegramChat).where(
            TelegramChat.id == chat_id, TelegramChat.user_id == user_id
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise ValueError("Chat not found")
    if not chat.indexing_enabled:
        raise ValueError("Indexing is disabled for this chat")
    account = await db.get(User, user_id)
    since = account.auto_index_since if account else None
    if since is not None:
        since = since.replace(tzinfo=UTC)

    # Get job
    result = await db.execute(
        select(SyncJob).where(
            SyncJob.id == job_id,
            SyncJob.user_id == user_id,
            SyncJob.chat_id == chat_id,
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise ValueError("Sync job not found")

    owns_client = False
    if client is None:
        client = await get_client(user_id, db)
        owns_client = True
    messages_synced = 0
    batch_values = []
    batch_count = 0
    last_id = chat.last_message_id
    oldest_seen_id = chat.history_cursor

    messages_seen = 0

    # Always fetch newest-first with the requested limit.
    # When limit is None ("Sync All"): Telethon fetches entire chat history.
    # When limit is N ("Sync Latest N"): Telethon fetches the N newest messages.
    # Deduplication is handled by the DB constraint. Every newly discovered media
    # message is persisted as pending and processed independently in the media queue.
    iter_kwargs = {"limit": limit, "wait_time": 0.5}
    if limit is None and chat.history_synced_at is None and chat.history_cursor:
        iter_kwargs["offset_id"] = chat.history_cursor
    total_reported = False

    try:
        # Resolve a proper InputPeer (with fresh access_hash) before GetHistoryRequest.
        # Passing a bare chat_id fails for channels/supergroups when the worker's
        # Telethon client has no warm entity cache — the listener and the worker
        # are separate clients, so the worker must resolve itself.
        peer = await _resolve_chat_entity(client, db, chat)
        iterator = client.iter_messages(peer, **iter_kwargs)
        async for message in iterator:
            # Telegram returns newest first. Stop before saving/processing older media.
            if since is not None and message.date < since:
                break
            messages_seen += 1
            oldest_seen_id = message.id

            # Report total from Telegram after first batch (Telethon populates iterator.total)
            if not total_reported and on_progress:
                iter_total = getattr(iterator, "total", None)
                on_progress(messages_seen, iter_total)
                total_reported = True

            if (
                not message.text
                and not message.media
                and not getattr(message, "action", None)
            ):
                continue

            media_values = _media_values(message, queue_processing=True)
            metadata_values = extract_message_metadata(
                message,
                file_name=media_values["media_file_name"],
            )
            msg_values = {
                "chat_id": chat_id,
                "telegram_message_id": message.id,
                "text": message.text,
                "sender_id": message.sender_id,
                "sender_name": _get_sender_name(message),
                "is_outgoing": message.out,
                "sent_at": message.date,
                "transcribed_at": None,
                **media_values,
                **metadata_values,
            }

            batch_values.append(msg_values)
            messages_synced += 1

            # Update last_id
            if not last_id or message.id > last_id:
                last_id = message.id

            # Persist metadata only. Expensive content processing runs in background.
            if len(batch_values) >= settings.sync_batch_size:
                if not indexing_has_capacity():
                    raise RuntimeError(
                        "History loading paused: insufficient free disk space"
                    )
                inserted_ids = await _insert_message_batch(
                    db,
                    batch_values,
                )
                await embed_messages(db, inserted_ids)
                skipped = len(batch_values) - len(inserted_ids)
                messages_synced -= skipped

                batch_values = []
                batch_count += 1
                record_request()  # Track batch API call

                if on_progress:
                    on_progress(messages_seen, None)

                # Commit batch so progress is visible and work survives interruptions
                job.messages_processed = messages_synced
                job.last_processed_id = last_id
                if limit is None and chat.history_synced_at is None:
                    chat.history_cursor = oldest_seen_id
                await db.commit()
                await db.refresh(chat)
                if not chat.indexing_enabled:
                    raise ValueError("Indexing was paused for this chat")

                # Progressive jittered delay — increases every N batches
                progressive_extra = (
                    batch_count // settings.sync_progressive_delay_interval
                ) * settings.sync_progressive_delay_step
                base_delay = settings.sync_delay_seconds + progressive_extra
                await _jittered_sleep(base_delay, settings.sync_delay_jitter)

        # Insert remaining batch
        if batch_values:
            if not indexing_has_capacity():
                raise RuntimeError(
                    "History loading paused: insufficient free disk space"
                )
            inserted_ids = await _insert_message_batch(
                db,
                batch_values,
            )
            await embed_messages(db, inserted_ids)
            skipped = len(batch_values) - len(inserted_ids)
            messages_synced -= skipped

        if on_progress:
            on_progress(messages_seen, None)

        if limit is None:
            chat.history_synced_at = datetime.now(UTC)
            chat.history_cursor = None

        # Update chat and job
        chat.last_message_id = last_id
        chat.last_sync_at = datetime.now(UTC)
        chat.total_messages_synced = (
            await db.execute(
                select(func.count()).where(TelegramMessage.chat_id == chat_id)
            )
        ).scalar()

        # Update chat preview from latest DB message
        latest_msg = (
            await db.execute(
                select(TelegramMessage)
                .where(TelegramMessage.chat_id == chat_id)
                .order_by(
                    TelegramMessage.sent_at.desc(),
                    TelegramMessage.telegram_message_id.desc(),
                    TelegramMessage.id.desc(),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest_msg:
            preview = latest_msg.text[:200] if latest_msg.text else None
            if not preview and latest_msg.media_type:
                preview = f"[{latest_msg.media_type}]"
            chat.last_message_text = preview
            chat.last_message_sender_name = latest_msg.sender_name
            chat.last_activity_at = latest_msg.sent_at

        job.messages_processed = messages_synced
        job.last_processed_id = last_id
        await db.commit()

        return messages_synced
    except Exception as e:
        if owns_client and is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        raise
    finally:
        if owns_client:
            await client.disconnect()


async def create_sync_job(
    db: AsyncSession, user_id: UUID, chat_id: UUID | None = None
) -> SyncJob:
    """Create a new sync job."""
    job = SyncJob(user_id=user_id, chat_id=chat_id, status=SyncStatus.PENDING)
    db.add(job)
    await db.flush()
    return job
