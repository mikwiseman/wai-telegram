import asyncio
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel
from telethon.tl.types import (
    Chat as TelegramGroupChat,
)
from telethon.tl.types import (
    User as TelegramUser,
)

from app.core.config import get_settings
from app.core.database import get_db_context
from app.models.chat import ChatType, TelegramChat
from app.models.message import (
    MediaProcessingStatus,
    MessageContentChunk,
    MessageRevision,
    TelegramMessage,
)
from app.models.media import (
    MediaObject,
    MediaObjectStatus,
    MediaStage,
    TranscriptSegment,
)
from app.models.settings import UserSettings
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User
from app.services.embedding_service import embed_messages
from app.services.correspondence_events import enqueue_event, dispatch_events
from app.services.media_content_service import get_media_info
from app.services.media_cache_service import _cache_key, telegram_media_identity
from app.services.single_user import is_user_active, lock_active_user
from app.services.sync_service import sync_messages
from app.services.telegram_client import (
    TelegramSessionUnauthorizedError,
    get_client_session_id,
    get_client,
    invalidate_client_authorization,
    is_session_authorization_error,
)
from app.services.telegram_metadata import extract_message_metadata

logger = logging.getLogger(__name__)
settings = get_settings()

HEARTBEAT_INTERVAL = 30
ACTIVE_KEY_TTL = 60
HEALTH_CHECK_INTERVAL = 300  # 5 minutes
SYNC_PROGRESS_TTL = 3600
SYNC_HEARTBEAT_TTL = 180


def _get_media_type(media) -> str | None:
    """Get media type from message media."""
    info = get_media_info(SimpleNamespace(media=media))
    return info.media_type if info else None


def _get_sender_name(sender) -> str | None:
    """Extract sender name from sender entity."""
    if not sender:
        return None
    if isinstance(sender, TelegramUser):
        parts = [sender.first_name or "", sender.last_name or ""]
        return " ".join(p for p in parts if p) or None
    return getattr(sender, "title", None)


class TelegramListener:
    def __init__(self):
        self.clients: dict[UUID, TelegramClient] = {}
        self.redis: aioredis.Redis | None = None

    async def run(self):
        """Main entry point."""
        self.redis = aioredis.from_url(settings.redis_url)
        logger.info("Listener starting — loading enabled users")
        await self._load_enabled_users()
        logger.info(f"Listener running with {len(self.clients)} active client(s)")
        await asyncio.gather(
            self._redis_command_loop(),
            self._heartbeat_loop(),
            self._health_check_loop(),
        )

    async def _load_enabled_users(self):
        """Connect Telethon clients for all users with realtime_sync_enabled."""
        async with get_db_context() as db:
            result = await db.execute(
                select(UserSettings)
                .join(User, User.id == UserSettings.user_id)
                .where(
                    UserSettings.realtime_sync_enabled == True,
                    User.is_active.is_(True),
                )
            )
            for user_settings in result.scalars().all():
                try:
                    await self._start_user(user_settings.user_id)
                except TelegramSessionUnauthorizedError as e:
                    logger.warning(
                        "Skipped listener startup for unauthorized user %s: %s",
                        user_settings.user_id,
                        e,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to start listener for user {user_settings.user_id}: {e}"
                    )

    async def _start_user(self, user_id: UUID):
        """Create Telethon client with event handlers."""
        if user_id in self.clients:
            logger.info(f"Client already active for user {user_id}")
            return

        async with get_db_context() as db:
            if not await is_user_active(db, user_id):
                logger.info("Ignored listener start for inactive user %s", user_id)
                return
            client = await get_client(user_id, db, receive_updates=True)

        @client.on(events.NewMessage)
        async def on_message(event):
            await self._handle_message(user_id, event)

        @client.on(events.MessageEdited)
        async def on_message_edited(event):
            await self._handle_edit(user_id, event)

        @client.on(events.MessageDeleted)
        async def on_message_deleted(event):
            await self._handle_delete(user_id, event)

        # Start receiving updates
        try:
            await client.catch_up()
        except Exception as e:
            if is_session_authorization_error(e):
                await invalidate_client_authorization(client, user_id, e)
                raise TelegramSessionUnauthorizedError(
                    "Telegram session expired. Reconnect Telegram and try again."
                ) from e
            await client.disconnect()
            raise

        self.clients[user_id] = client
        await self.redis.set(f"listener:active:{user_id}", "1", ex=ACTIVE_KEY_TTL)
        logger.info(f"Listener started for user {user_id}")

    async def _stop_user(self, user_id: UUID):
        """Disconnect a user's Telethon client."""
        client = self.clients.pop(user_id, None)
        if client:
            await client.disconnect()
        await self.redis.delete(f"listener:active:{user_id}")
        logger.info(f"Listener stopped for user {user_id}")

    async def _handle_unauthorized_client(
        self,
        user_id: UUID,
        client: TelegramClient,
        error: Exception,
    ) -> None:
        logger.warning(
            "Listener client lost authorization for user %s (session_id=%s): %s",
            user_id,
            get_client_session_id(client),
            error,
        )
        await invalidate_client_authorization(client, user_id, error)
        await self._stop_user(user_id)

    async def _auto_create_chat(
        self, db, user_id: UUID, entity, chat_id_tg: int
    ) -> TelegramChat | None:
        """Auto-create a TelegramChat record from a Telethon entity."""
        if isinstance(entity, TelegramUser):
            chat_type = ChatType.PRIVATE
            title = _get_sender_name(entity) or f"User {chat_id_tg}"
            username = entity.username
        elif isinstance(entity, Channel):
            chat_type = ChatType.SUPERGROUP if entity.megagroup else ChatType.CHANNEL
            title = entity.title or f"Channel {chat_id_tg}"
            username = entity.username
        elif isinstance(entity, TelegramGroupChat):
            chat_type = ChatType.GROUP
            title = entity.title or f"Group {chat_id_tg}"
            username = None
        else:
            logger.warning(f"Unknown entity type for chat {chat_id_tg}: {type(entity)}")
            return None

        account = await db.get(User, user_id)
        stmt = pg_insert(TelegramChat).values(
            user_id=user_id,
            indexing_enabled=bool(
                account and (account.invited_by_id is None or account.auto_index_since)
            ),
            telegram_chat_id=chat_id_tg,
            access_hash=getattr(entity, "access_hash", None),
            chat_type=chat_type,
            title=title,
            username=username,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_telegram_chats_user_chat",
            set_={
                "access_hash": func.coalesce(
                    stmt.excluded.access_hash,
                    TelegramChat.access_hash,
                ),
                "title": stmt.excluded.title,
                "username": func.coalesce(
                    stmt.excluded.username,
                    TelegramChat.username,
                ),
            },
        ).returning(TelegramChat.id)
        result = await db.execute(stmt)
        row = result.fetchone()
        if row:
            await db.flush()
            # Re-fetch the full object
            result = await db.execute(
                select(TelegramChat).where(TelegramChat.id == row[0])
            )
            chat = result.scalar_one()
            logger.info(
                f"Auto-created chat '{title}' ({chat_id_tg}) for user {user_id}"
            )
            return chat
        return None

    async def _handle_message(self, user_id: UUID, event):
        """Save incoming message to DB."""
        message = event.message
        if (
            not message.text
            and not message.media
            and not getattr(message, "action", None)
        ):
            return

        try:
            chat_id_tg = event.chat_id
            inserted_id = None
            correspondence = False

            async with get_db_context() as db:
                if not await lock_active_user(db, user_id):
                    await self._stop_user(user_id)
                    return
                # Find our chat record by telegram_chat_id
                result = await db.execute(
                    select(TelegramChat).where(
                        TelegramChat.user_id == user_id,
                        TelegramChat.telegram_chat_id == chat_id_tg,
                    )
                )
                chat = result.scalar_one_or_none()
                if not chat:
                    try:
                        entity = await event.get_chat()
                        chat = await self._auto_create_chat(
                            db, user_id, entity, chat_id_tg
                        )
                        if not chat:
                            return
                    except Exception as e:
                        if is_session_authorization_error(e):
                            raise
                        logger.warning(f"Could not auto-create chat {chat_id_tg}: {e}")
                        return

                if not chat or not chat.indexing_enabled:
                    return

                account = await db.get(User, user_id)
                if (
                    account
                    and account.auto_index_since is not None
                    and message.date < account.auto_index_since.replace(tzinfo=UTC)
                ):
                    return

                try:
                    sender = await event.get_sender()
                except FloodWaitError as e:
                    logger.warning(
                        f"FloodWait in listener for user {user_id}: {e.seconds}s"
                    )
                    await asyncio.sleep(e.seconds)
                    sender = await event.get_sender()

                media_info = get_media_info(message)
                media_type = media_info.media_type if media_info else None
                metadata_values = extract_message_metadata(
                    message,
                    file_name=media_info.file_name if media_info else None,
                )
                values = {
                    "chat_id": chat.id,
                    "telegram_message_id": message.id,
                    "text": message.text,
                    "has_media": media_info is not None,
                    "media_type": media_type,
                    "media_file_name": media_info.file_name if media_info else None,
                    "media_mime_type": media_info.mime_type if media_info else None,
                    "media_file_size": media_info.file_size if media_info else None,
                    "media_duration_seconds": (
                        media_info.duration_seconds if media_info else None
                    ),
                    "media_processing_status": (
                        MediaProcessingStatus.PENDING
                        if media_info and media_info.media_type != "other"
                        else None
                    ),
                    "sender_id": message.sender_id,
                    "sender_name": _get_sender_name(sender),
                    "is_outgoing": message.out,
                    "sent_at": message.date,
                    **metadata_values,
                }

                stmt = pg_insert(TelegramMessage).values(**values)
                stmt = stmt.on_conflict_do_nothing(
                    constraint="uq_telegram_messages_chat_msg"
                ).returning(TelegramMessage.id)
                result = await db.execute(stmt)
                inserted = result.fetchone()
                if inserted:
                    inserted_id = inserted[0]
                    if not getattr(sender, "bot", False):
                        correspondence = await enqueue_event(
                            db,
                            chat,
                            message.id,
                            "message.new",
                            str(message.id),
                            message.date,
                        )

                # Update chat's last_message_id and preview
                if not chat.last_message_id or message.id > chat.last_message_id:
                    chat.last_message_id = message.id
                    text = values.get("text")
                    preview = text[:200] if text else None
                    if not preview and message.media:
                        preview = f"[{media_type}]" if media_type else "[media]"
                    chat.last_message_text = preview
                    chat.last_message_sender_name = _get_sender_name(sender)
                    chat.last_activity_at = message.date
                chat.last_sync_at = datetime.now(UTC)
                chat.total_messages_synced = (
                    await db.execute(
                        select(func.count()).where(TelegramMessage.chat_id == chat.id)
                    )
                ).scalar()

            if correspondence:
                dispatch_events()

            # Best-effort caption/text embedding in a separate session. Media
            # processing replaces it atomically with summary + content chunks.
            if inserted_id:
                try:
                    async with get_db_context() as db:
                        await embed_messages(db, [inserted_id])
                except Exception as e:
                    logger.debug(f"Embedding failed for real-time message: {e}")

        except Exception as e:
            client = self.clients.get(user_id)
            if client is not None and is_session_authorization_error(e):
                await self._handle_unauthorized_client(user_id, client, e)
                return
            logger.error(f"Error handling real-time message for user {user_id}: {e}")

    async def _handle_edit(self, user_id: UUID, event) -> None:
        """Keep the previous version and update all searchable metadata."""
        correspondence = False
        async with get_db_context() as db:
            if not await lock_active_user(db, user_id):
                await self._stop_user(user_id)
                return
            chat = (
                await db.execute(
                    select(TelegramChat).where(
                        TelegramChat.user_id == user_id,
                        TelegramChat.telegram_chat_id == event.chat_id,
                    )
                )
            ).scalar_one_or_none()
            if chat is None or not chat.indexing_enabled:
                logger.warning(
                    "Ignored edit for unknown chat %s and active user %s",
                    event.chat_id,
                    user_id,
                )
                return
            current = (
                await db.execute(
                    select(TelegramMessage)
                    .where(
                        TelegramMessage.chat_id == chat.id,
                        TelegramMessage.telegram_message_id == event.message.id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if current is None:
                logger.warning(
                    "Ignored edit for unsynced Telegram message %s in chat %s",
                    event.message.id,
                    event.chat_id,
                )
                return
            next_revision = (
                await db.execute(
                    select(func.coalesce(func.max(MessageRevision.revision), 0)).where(
                        MessageRevision.message_id == current.id
                    )
                )
            ).scalar_one() + 1
            db.add(
                MessageRevision(
                    message_id=current.id,
                    revision=next_revision,
                    text=current.text,
                    entities=current.entities,
                    edited_at=current.edited_at or current.sent_at,
                )
            )
            info = get_media_info(event.message)
            metadata = extract_message_metadata(
                event.message,
                file_name=info.file_name if info else current.media_file_name,
            )
            current.text = event.message.text
            current.edited_at = event.message.edit_date or datetime.now(UTC)
            correspondence = await enqueue_event(
                db,
                chat,
                current.telegram_message_id,
                "message.updated",
                current.edited_at.isoformat(),
                current.edited_at,
            )
            for field, value in metadata.items():
                setattr(current, field, value)
            if info is not None:
                media_object = (
                    await db.execute(
                        select(MediaObject).where(MediaObject.message_id == current.id)
                    )
                ).scalar_one_or_none()
                current_identity = telegram_media_identity(event.message)
                media_changed = bool(
                    media_object is not None
                    and media_object.telegram_media_id != current_identity
                )
                if media_changed:
                    media_object.cache_key = _cache_key(
                        user_id,
                        chat.telegram_chat_id,
                        current.telegram_message_id,
                        current_identity,
                    )
                    media_object.telegram_media_id = current_identity
                    media_object.relative_path = None
                    media_object.sha256 = None
                    media_object.byte_offset = 0
                    media_object.status = MediaObjectStatus.PENDING
                    media_object.stage = MediaStage.FETCH
                    media_object.error_code = None
                    media_object.error_detail = None
                    current.content_text = None
                    current.content_summary = None
                    current.content_model = None
                    current.summary_model = None
                    current.embedding = None
                    current.embedded_at = None
                    current.transcribed_at = None
                    current.media_processed_at = None
                    await db.execute(
                        delete(MessageContentChunk).where(
                            MessageContentChunk.message_id == current.id
                        )
                    )
                    await db.execute(
                        delete(TranscriptSegment).where(
                            TranscriptSegment.message_id == current.id
                        )
                    )
                current.has_media = True
                current.media_type = info.media_type
                current.media_file_name = info.file_name
                current.media_mime_type = info.mime_type
                current.media_file_size = info.file_size
                current.media_duration_seconds = info.duration_seconds
                if media_changed or current.media_processing_status is None:
                    current.media_processing_status = MediaProcessingStatus.PENDING

        if correspondence:
            dispatch_events()

    async def _handle_delete(self, user_id: UUID, event) -> None:
        """Tombstone Telegram deletes; never erase the local archive."""
        deleted_ids = list(event.deleted_ids or [])
        if not deleted_ids:
            return
        correspondence = False
        async with get_db_context() as db:
            if not await lock_active_user(db, user_id):
                await self._stop_user(user_id)
                return
            owner_chat_ids = select(TelegramChat.id).where(
                TelegramChat.user_id == user_id
            )
            if event.chat_id is None:
                # Telethon omits Peer information for private-chat deletions.
                # Message IDs are safe to use only when they resolve to exactly
                # one chat in this owner's archive; ambiguous IDs stay untouched.
                message_ids = (
                    select(TelegramMessage.telegram_message_id)
                    .where(
                        TelegramMessage.chat_id.in_(owner_chat_ids),
                        TelegramMessage.telegram_message_id.in_(deleted_ids),
                    )
                    .group_by(TelegramMessage.telegram_message_id)
                    .having(func.count(func.distinct(TelegramMessage.chat_id)) == 1)
                )
                predicate = TelegramMessage.telegram_message_id.in_(message_ids)
            else:
                owner_chat_ids = owner_chat_ids.where(
                    TelegramChat.telegram_chat_id == event.chat_id
                )
                predicate = TelegramMessage.telegram_message_id.in_(deleted_ids)

            affected = (
                await db.execute(
                    select(TelegramMessage, TelegramChat)
                    .join(TelegramChat)
                    .where(
                        TelegramMessage.chat_id.in_(owner_chat_ids),
                        predicate,
                        TelegramMessage.deleted_at.is_(None),
                    )
                )
            ).all()
            for message, chat in affected:
                queued = await enqueue_event(
                    db,
                    chat,
                    message.telegram_message_id,
                    "message.deleted",
                    "deleted",
                    datetime.now(UTC),
                )
                correspondence = correspondence or queued
            await db.execute(
                update(TelegramMessage)
                .where(
                    TelegramMessage.chat_id.in_(owner_chat_ids),
                    predicate,
                )
                .values(deleted_at=datetime.now(UTC))
                .execution_options(synchronize_session="fetch")
            )
        if correspondence:
            dispatch_events()

    async def _handle_sync(self, cmd: dict):
        """Perform manual sync using existing sync_messages() function."""
        user_id = UUID(cmd["user_id"])
        chat_id = UUID(cmd["chat_id"])
        job_id = UUID(cmd["job_id"])
        limit = cmd.get("limit")

        client = self.clients.get(user_id)
        if not client:
            logger.error(f"No active client for user {user_id} during sync command")
            async with get_db_context() as db:
                result = await db.execute(select(SyncJob).where(SyncJob.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = SyncStatus.FAILED
                    job.error_message = "Listener client is not active for this user"
                    await db.commit()
            return

        try:
            async with get_db_context() as db:
                if not await is_user_active(db, user_id):
                    result = await db.execute(
                        select(SyncJob).where(
                            SyncJob.id == job_id,
                            SyncJob.user_id == user_id,
                        )
                    )
                    job = result.scalar_one_or_none()
                    if job:
                        job.status = SyncStatus.CANCELLED
                        job.error_message = "User is inactive"
                        await db.commit()
                    await self._stop_user(user_id)
                    return

                result = await db.execute(
                    select(SyncJob).where(
                        SyncJob.id == job_id,
                        SyncJob.user_id == user_id,
                        SyncJob.chat_id == chat_id,
                    )
                )
                job = result.scalar_one_or_none()
                if not job:
                    logger.error(f"Sync job {job_id} not found")
                    return
                job.status = SyncStatus.IN_PROGRESS
                job.error_message = None
                job.completed_at = None
                await db.commit()

                await self.redis.setex(
                    f"sync:{job_id}:heartbeat", SYNC_HEARTBEAT_TTL, "1"
                )

                def _on_progress(seen: int, total: int | None = None) -> None:
                    if total is not None:
                        asyncio.create_task(
                            self.redis.setex(
                                f"sync:{job_id}:total", SYNC_PROGRESS_TTL, total
                            )
                        )
                    asyncio.create_task(
                        self.redis.setex(f"sync:{job_id}:seen", SYNC_PROGRESS_TTL, seen)
                    )
                    asyncio.create_task(
                        self.redis.setex(
                            f"sync:{job_id}:heartbeat", SYNC_HEARTBEAT_TTL, "1"
                        )
                    )

                count = await sync_messages(
                    db,
                    user_id,
                    chat_id,
                    job_id,
                    limit,
                    on_progress=_on_progress,
                    client=client,
                )

                job.status = SyncStatus.COMPLETED
                job.completed_at = datetime.now(UTC)
                job.messages_processed = count
                await db.commit()

                logger.info(
                    f"Listener sync completed: chat {chat_id}, {count} messages"
                )
        except Exception as e:
            if is_session_authorization_error(e):
                await self._handle_unauthorized_client(user_id, client, e)
                error_message = (
                    "Telegram session expired. Reconnect Telegram and try again."
                )
                logger.warning(
                    "Listener sync failed due to expired Telegram session for user %s: %s",
                    user_id,
                    e,
                )
            else:
                error_message = str(e)[:500]
                logger.error(f"Listener sync failed for chat {chat_id}: {e}")
            async with get_db_context() as db:
                result = await db.execute(
                    select(SyncJob).where(
                        SyncJob.id == job_id,
                        SyncJob.user_id == user_id,
                    )
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = SyncStatus.FAILED
                    job.error_message = error_message
                    await db.commit()
            await self.redis.delete(
                f"sync:{job_id}:total",
                f"sync:{job_id}:seen",
                f"sync:{job_id}:heartbeat",
            )

    async def _redis_command_loop(self):
        """Listen for commands from API."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("listener:cmd:global")

        # Also subscribe to per-user channels for active users
        for user_id in list(self.clients.keys()):
            await pubsub.subscribe(f"listener:cmd:{user_id}")

        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                cmd = json.loads(msg["data"])
                match cmd.get("command"):
                    case "start_user":
                        uid = UUID(cmd["user_id"])
                        await self._start_user(uid)
                        await pubsub.subscribe(f"listener:cmd:{uid}")
                    case "stop_user":
                        uid = UUID(cmd["user_id"])
                        await pubsub.unsubscribe(f"listener:cmd:{uid}")
                        await self._stop_user(uid)
                    case "sync_chat":
                        await self._handle_sync(cmd)
                    case _:
                        logger.warning(f"Unknown listener command: {cmd}")
            except TelegramSessionUnauthorizedError as e:
                logger.warning("Ignored listener start for unauthorized session: %s", e)
            except Exception as e:
                logger.error(f"Error processing listener command: {e}")

    async def _heartbeat_loop(self):
        """Refresh Redis active keys every 30s, only for connected clients."""
        while True:
            try:
                for user_id, client in list(self.clients.items()):
                    async with get_db_context() as db:
                        active = await is_user_active(db, user_id)
                    if not active:
                        await self._stop_user(user_id)
                        continue
                    if client.is_connected():
                        await self.redis.set(
                            f"listener:active:{user_id}", "1", ex=ACTIVE_KEY_TTL
                        )
                    else:
                        logger.warning(f"Client disconnected for user {user_id}")
                        await self.redis.delete(f"listener:active:{user_id}")
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _health_check_loop(self):
        """Periodically verify client health and catch up missed updates."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            for user_id, client in list(self.clients.items()):
                try:
                    async with get_db_context() as db:
                        active = await is_user_active(db, user_id)
                    if not active:
                        await self._stop_user(user_id)
                        continue
                    if client.is_connected():
                        await client.catch_up()
                        logger.debug(
                            f"Health check catch_up completed for user {user_id}"
                        )
                except Exception as e:
                    if is_session_authorization_error(e):
                        await self._handle_unauthorized_client(user_id, client, e)
                    else:
                        logger.error(f"Health check failed for user {user_id}: {e}")
