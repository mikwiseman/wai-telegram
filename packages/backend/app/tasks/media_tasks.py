import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from uuid import UUID

from celery import shared_task
from sqlalchemy import and_, case, func, or_, select, update
from telethon.errors import FloodWaitError

from app.core.async_runner import run_async
from app.core.config import get_settings
from app.core.database import get_db_context
from app.models.chat import TelegramChat
from app.models.media import MediaObject, MediaObjectStatus, MediaStage
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.user import User
from app.services.media_cache_service import MediaDiskFull, MediaSourceDeleted
from app.services.media_content_service import (
    MediaNoSpeechError,
    MediaProcessingConfigurationError,
)
from app.services.media_processing_service import (
    _mark_failed,
    extract_media_stage,
    fetch_media_stage,
    index_media_stage,
    mark_media_retry,
)

logger = logging.getLogger(__name__)
settings = get_settings()
RETRY_DELAYS_SECONDS = (15, 60, 300, 900, 3600, 21_600, 86_400)


def enqueue_media_processing(message_ids: list[UUID]) -> None:
    """Start each durable media pipeline at the resumable fetch queue."""
    if not settings.media_pipeline_enabled:
        raise RuntimeError(
            "Durable media processing is deferred until storage is attached"
        )
    for message_id in message_ids:
        fetch_message_media_task.apply_async(
            args=[str(message_id)],
            queue="media-fetch",
        )


def _exact_retry_after(error: Exception) -> int | None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, FloodWaitError):
            return max(1, int(current.seconds))
        provider_delay = getattr(current, "retry_after_seconds", None)
        if isinstance(provider_delay, (int, float)) and provider_delay > 0:
            return max(1, int(provider_delay))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = headers.get("Retry-After") if headers is not None else None
        if isinstance(retry_after, str) and retry_after.strip():
            try:
                return max(1, int(float(retry_after)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    return max(
                        1,
                        int((retry_at - datetime.now(UTC)).total_seconds()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        current = current.__cause__
    return None


def _is_terminal_error(error: Exception) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(
            current,
            (
                MediaDiskFull,
                MediaSourceDeleted,
                MediaNoSpeechError,
                MediaProcessingConfigurationError,
            ),
        ):
            return True
        # A link preview or a poll is not a file: Telethon cannot build a download location for it.
        if isinstance(current, TypeError) and "InputFileLocation" in str(current):
            return True
        current = current.__cause__
    return False


def _retry_or_finish(task, message_uuid: UUID, error: Exception) -> None:
    retries = task.request.retries
    if _is_terminal_error(error) or retries >= len(RETRY_DELAYS_SECONDS):
        run_async(_mark_failed(message_uuid, error))
        return
    countdown = _exact_retry_after(error) or RETRY_DELAYS_SECONDS[retries]
    run_async(mark_media_retry(message_uuid, error, countdown))
    raise task.retry(exc=error, countdown=countdown)


def _is_redelivered(task) -> bool:
    delivery_info = getattr(task.request, "delivery_info", None) or {}
    return bool(delivery_info.get("redelivered"))


async def recover_interrupted_media_stage(message_id: UUID, stage: str) -> bool:
    """Reset only the durable claim represented by a broker redelivery."""
    async with get_db_context() as db:
        active_message_ids = (
            select(TelegramMessage.id)
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .join(User, User.id == TelegramChat.user_id)
            .where(User.is_active.is_(True), TelegramChat.indexing_enabled.is_(True))
        )
        if stage == "process":
            recovered = (
                await db.execute(
                    update(MediaObject)
                    .where(
                        MediaObject.message_id == message_id,
                        MediaObject.message_id.in_(active_message_ids),
                        MediaObject.stage == MediaStage.EXTRACTION,
                        MediaObject.status == MediaObjectStatus.EXTRACTING,
                    )
                    .values(status=MediaObjectStatus.CACHED)
                    .returning(MediaObject.id)
                )
            ).scalar_one_or_none()
            if recovered is not None:
                await db.execute(
                    update(TelegramMessage)
                    .where(
                        TelegramMessage.id == message_id,
                        TelegramMessage.media_processing_status
                        == MediaProcessingStatus.PROCESSING,
                    )
                    .values(
                        media_processing_status=MediaProcessingStatus.QUEUED,
                        media_processing_started_at=None,
                    )
                )
                return True
            return False
        if stage == "index":
            recovered = (
                await db.execute(
                    update(MediaObject)
                    .where(
                        MediaObject.message_id == message_id,
                        MediaObject.message_id.in_(active_message_ids),
                        MediaObject.stage == MediaStage.INDEX,
                        MediaObject.status == MediaObjectStatus.INDEXING,
                    )
                    .values(status=MediaObjectStatus.CACHED)
                    .returning(MediaObject.id)
                )
            ).scalar_one_or_none()
            return recovered is not None
    raise ValueError(f"Unknown media recovery stage: {stage}")


@shared_task(
    bind=True,
    max_retries=len(RETRY_DELAYS_SECONDS),
    name="app.tasks.media_tasks.fetch_message_media",
)
def fetch_message_media_task(self, message_id: str) -> dict[str, str]:
    """Fetch bytes with resume, then hand off the durable cache checkpoint."""
    message_uuid = UUID(message_id)
    try:
        status = run_async(fetch_media_stage(message_uuid))
        if status == "cached":
            process_message_media_task.apply_async(
                args=[message_id],
                queue="media-process",
            )
    except Exception as error:
        _retry_or_finish(self, message_uuid, error)
        return {"message_id": message_id, "status": "failed"}
    return {"message_id": message_id, "status": status}


@shared_task(
    bind=True,
    max_retries=len(RETRY_DELAYS_SECONDS),
    name="app.tasks.media_tasks.process_message_media",
)
def process_message_media_task(self, message_id: str) -> dict[str, str]:
    """Extract/transcribe/summarize, then hand off the saved checkpoint."""
    message_uuid = UUID(message_id)
    try:
        if _is_redelivered(self):
            run_async(recover_interrupted_media_stage(message_uuid, "process"))
        status = run_async(extract_media_stage(message_uuid))
        if status == "checkpointed":
            index_message_media_task.apply_async(
                args=[message_id],
                queue="media-index",
            )
    except Exception as error:
        _retry_or_finish(self, message_uuid, error)
        return {"message_id": message_id, "status": "failed"}
    return {"message_id": message_id, "status": status}


@shared_task(
    bind=True,
    max_retries=len(RETRY_DELAYS_SECONDS),
    name="app.tasks.media_tasks.index_message_media",
)
def index_message_media_task(self, message_id: str) -> dict[str, str]:
    """Build search indexes without repeating fetch or transcription."""
    message_uuid = UUID(message_id)
    try:
        if _is_redelivered(self):
            run_async(recover_interrupted_media_stage(message_uuid, "index"))
        status = run_async(index_media_stage(message_uuid))
    except Exception as error:
        _retry_or_finish(self, message_uuid, error)
        return {"message_id": message_id, "status": "failed"}
    return {"message_id": message_id, "status": status}


async def _find_pending_and_reap_stale() -> list[UUID]:
    now = datetime.now(UTC)
    queued_before = now - timedelta(minutes=settings.media_queue_stale_minutes)
    stale_before = now - timedelta(minutes=settings.media_processing_stale_minutes)
    async with get_db_context() as db:
        active_message_ids = (
            select(TelegramMessage.id)
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .join(User, User.id == TelegramChat.user_id)
            .where(User.is_active.is_(True), TelegramChat.indexing_enabled.is_(True))
        )
        # Recover only very old dispatch claims and retries that never ran.
        # Normal broker backlog remains queued; duplicate delivery is still
        # protected by the processing claim.
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id.in_(active_message_ids),
                TelegramMessage.media_processing_status == MediaProcessingStatus.QUEUED,
                TelegramMessage.media_processing_started_at < queued_before,
            )
            .values(
                media_processing_status=MediaProcessingStatus.PENDING,
                media_processing_started_at=None,
            )
        )
        stale_ids = list(
            (
                await db.execute(
                    update(TelegramMessage)
                    .where(
                        TelegramMessage.id.in_(active_message_ids),
                        TelegramMessage.media_processing_status
                        == MediaProcessingStatus.PROCESSING,
                        TelegramMessage.media_processing_started_at < stale_before,
                    )
                    .values(
                        media_processing_status=MediaProcessingStatus.PENDING,
                        media_processing_started_at=None,
                        media_processing_error_code=None,
                        media_processing_error=None,
                        media_processed_at=None,
                    )
                    .returning(TelegramMessage.id)
                )
            )
            .scalars()
            .all()
        )
        if stale_ids:
            await db.execute(
                update(MediaObject)
                .where(
                    MediaObject.message_id.in_(stale_ids),
                    MediaObject.status.in_(
                        (MediaObjectStatus.EXTRACTING, MediaObjectStatus.INDEXING)
                    ),
                )
                .values(
                    status=MediaObjectStatus.CACHED,
                    retry_after=None,
                )
            )
        # A queued row counts once its start is due. A row waiting for a retry
        # carries the retry time as its start, so a day-long wait holds no slot:
        # twenty failing files once stopped every new voice message.
        active_count = (
            await db.execute(
                select(func.count()).where(
                    TelegramMessage.id.in_(active_message_ids),
                    or_(
                        TelegramMessage.media_processing_status
                        == MediaProcessingStatus.PROCESSING,
                        and_(
                            TelegramMessage.media_processing_status
                            == MediaProcessingStatus.QUEUED,
                            TelegramMessage.media_processing_started_at <= now,
                        ),
                    ),
                )
            )
        ).scalar_one()
        dispatch_limit = max(
            0,
            settings.media_dispatch_target_depth - active_count,
        )
        if dispatch_limit == 0:
            return []

        media_priority = case(
            (TelegramMessage.transcribed_at.isnot(None), 0),
            (
                TelegramMessage.media_type.in_(
                    ("voice", "video_note", "audio", "video")
                ),
                1,
            ),
            (TelegramMessage.media_type == "document", 2),
            (TelegramMessage.media_type == "photo", 3),
            else_=4,
        )
        dispatchable = (
            TelegramMessage.media_processing_status == MediaProcessingStatus.PENDING
        )
        pending_subquery = (
            select(TelegramMessage.id)
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .join(User, User.id == TelegramChat.user_id)
            .where(
                dispatchable,
                User.is_active.is_(True),
                TelegramChat.indexing_enabled.is_(True),
            )
            .order_by(
                media_priority.asc(),
                TelegramMessage.sent_at.desc(),
                TelegramMessage.id.desc(),
            )
            .limit(dispatch_limit)
            .with_for_update(skip_locked=True)
        )
        pending_ids = list(
            (
                await db.execute(
                    update(TelegramMessage)
                    .where(
                        TelegramMessage.id.in_(pending_subquery),
                        dispatchable,
                    )
                    .values(
                        media_processing_status=MediaProcessingStatus.QUEUED,
                        media_processing_started_at=now,
                    )
                    .returning(TelegramMessage.id)
                )
            )
            .scalars()
            .all()
        )

    if stale_ids:
        logger.warning(
            "Recovered %s interrupted media jobs for resumable dispatch",
            len(stale_ids),
        )
    return pending_ids


async def _return_queued_to_pending(message_ids: list[UUID]) -> None:
    if not message_ids:
        return
    async with get_db_context() as db:
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id.in_(message_ids),
                TelegramMessage.media_processing_status == MediaProcessingStatus.QUEUED,
            )
            .values(
                media_processing_status=MediaProcessingStatus.PENDING,
                media_processing_started_at=None,
            )
        )


@shared_task(max_retries=0, name="app.tasks.media_tasks.dispatch_pending_media")
def dispatch_pending_media() -> dict[str, int]:
    """Recover pending jobs missed by immediate dispatch after ingestion."""
    if not settings.media_pipeline_enabled:
        return {"dispatched": 0, "deferred": 1}
    message_ids = run_async(_find_pending_and_reap_stale())
    for index, message_id in enumerate(message_ids):
        try:
            enqueue_media_processing([message_id])
        except Exception:
            run_async(_return_queued_to_pending(message_ids[index:]))
            raise
    return {"dispatched": len(message_ids)}
