import hashlib
import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.chat import ChatType, TelegramChat
from app.models.correspondence import CorrespondenceEvent
from app.models.user import User

logger = logging.getLogger(__name__)


def destination():
    settings = get_settings()
    url, token, owner = (
        settings.correspondence_webhook_url,
        settings.correspondence_webhook_token,
        settings.owner_user_id,
    )
    if not url or not token or not owner:
        return None
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/hooks/correspondence/")
        or len(token) < 32
    ):
        raise ValueError("Invalid correspondence webhook configuration")
    fingerprint = hashlib.sha256(f"{url}\0{token}\0{owner}".encode()).hexdigest()
    return url, token, owner, fingerprint


async def enqueue_event(
    db: AsyncSession,
    chat: TelegramChat,
    message_id: int,
    kind: str,
    revision: str,
    occurred_at: datetime,
) -> bool:
    target = destination()
    if (
        not target
        or chat.user_id != target[2]
        or not chat.indexing_enabled
        or chat.chat_type == ChatType.CHANNEL
    ):
        return False
    event_id = hashlib.sha256(
        f"{chat.user_id}:{chat.telegram_chat_id}:{message_id}:{kind}:{revision}".encode()
    ).hexdigest()
    await db.execute(
        insert(CorrespondenceEvent)
        .values(
            id=event_id,
            user_id=chat.user_id,
            chat_id=chat.id,
            destination_fingerprint=target[3],
            payload={
                "id": event_id,
                "threadId": str(chat.telegram_chat_id),
                "messageId": str(message_id),
                "type": kind,
                "occurredAt": occurred_at.isoformat(),
            },
            attempts=0,
            next_attempt_at=datetime.now(UTC),
            revoked=False,
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )
    return True


async def enqueue_media_ready(
    db: AsyncSession, chat: TelegramChat, message_id: int, at: datetime
) -> bool:
    if not destination():
        return False
    # Late transcription wakes a conversation already observed live. Enabling
    # this workflow must not replay the historical media-indexing backlog.
    seen = await db.scalar(
        select(CorrespondenceEvent.id)
        .where(
            CorrespondenceEvent.user_id == chat.user_id,
            CorrespondenceEvent.payload["threadId"].as_string()
            == str(chat.telegram_chat_id),
            CorrespondenceEvent.payload["messageId"].as_string() == str(message_id),
        )
        .limit(1)
    )
    return bool(seen) and await enqueue_event(
        db, chat, message_id, "message.updated", "media:" + at.isoformat(), at
    )


def dispatch_events() -> None:
    """Called after the source transaction commits. Beat recovers a lost broker wakeup."""
    if not destination():
        return
    from app.tasks.correspondence_tasks import deliver_correspondence_events

    try:
        deliver_correspondence_events.delay()
    except Exception:
        logger.warning(
            "Correspondence outbox wakeup failed; durable retry remains pending"
        )


async def deliver_batch(db: AsyncSession, client: httpx.AsyncClient) -> int:
    target = destination()
    if not target:
        return 0
    url, token, owner, fingerprint = target
    now = datetime.now(UTC)
    rows = (
        await db.scalars(
            select(CorrespondenceEvent)
            .join(User, User.id == CorrespondenceEvent.user_id)
            .join(TelegramChat, TelegramChat.id == CorrespondenceEvent.chat_id)
            .where(
                CorrespondenceEvent.delivered_at.is_(None),
                CorrespondenceEvent.revoked.is_(False),
                CorrespondenceEvent.next_attempt_at <= now,
                CorrespondenceEvent.user_id == owner,
                User.is_active.is_(True),
                TelegramChat.indexing_enabled.is_(True),
            )
            .order_by(CorrespondenceEvent.next_attempt_at, CorrespondenceEvent.id)
            .limit(10)
            .with_for_update(skip_locked=True, of=CorrespondenceEvent)
        )
    ).all()
    delivered = 0
    for event in rows:
        if event.destination_fingerprint != fingerprint:
            event.revoked = True
            continue
        event.attempts += 1
        try:
            response = await client.post(
                url,
                headers={"Authorization": "Bearer " + token},
                json=event.payload,
                timeout=5,
                follow_redirects=False,
            )
            if (
                response.status_code not in (200, 202)
                or response.json().get("accepted") is not True
            ):
                raise ValueError("Event not acknowledged")
        except (httpx.HTTPError, ValueError):
            event.next_attempt_at = now + timedelta(
                seconds=min(3600, 15 * 2 ** min(event.attempts, 8))
            )
            continue
        event.delivered_at = datetime.now(UTC)
        delivered += 1
    return delivered
