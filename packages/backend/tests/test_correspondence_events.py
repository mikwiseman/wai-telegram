from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
from sqlalchemy import select

from app.core.config import get_settings
from app.models.chat import ChatType, TelegramChat
from app.models.correspondence import CorrespondenceEvent
from app.services.correspondence_events import enqueue_event, deliver_batch


def configure(monkeypatch, owner):
    settings = get_settings()
    monkeypatch.setattr(settings, "owner_user_id", owner)
    monkeypatch.setattr(
        settings,
        "correspondence_webhook_url",
        "https://example.com/hooks/correspondence/telegram",
    )
    monkeypatch.setattr(
        settings,
        "correspondence_webhook_token",
        "private-test-token-32-characters-long",
    )


async def test_event_is_transactional_deduplicated_and_contains_only_references(
    db_session, test_user, monkeypatch
):
    configure(monkeypatch, test_user.id)
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=123,
        chat_type=ChatType.PRIVATE,
        indexing_enabled=True,
        title="Person",
    )
    db_session.add(chat)
    await db_session.flush()
    at = datetime.now(UTC)
    for _ in range(2):
        await enqueue_event(db_session, chat, 7, "message.new", "revision-one", at)
    rows = list((await db_session.scalars(select(CorrespondenceEvent))).all())
    assert len(rows) == 1
    assert rows[0].payload == {
        "id": rows[0].id,
        "threadId": "123",
        "messageId": "7",
        "type": "message.new",
        "occurredAt": at.isoformat(),
    }
    await db_session.rollback()
    assert not (await db_session.scalars(select(CorrespondenceEvent))).all()


async def test_foreign_owner_and_broadcast_channels_never_enqueue(
    db_session, test_user, monkeypatch
):
    from uuid import uuid4

    configure(monkeypatch, uuid4())
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=123,
        chat_type=ChatType.PRIVATE,
        indexing_enabled=True,
        title="Person",
    )
    db_session.add(chat)
    await db_session.flush()
    assert not await enqueue_event(
        db_session, chat, 7, "message.new", "v1", datetime.now(UTC)
    )
    configure(monkeypatch, test_user.id)
    chat.chat_type = ChatType.CHANNEL
    assert not await enqueue_event(
        db_session, chat, 7, "message.new", "v1", datetime.now(UTC)
    )
    assert not (await db_session.scalars(select(CorrespondenceEvent))).all()


async def test_delivery_retries_the_same_event_after_timeout_and_accepts_readback(
    db_session, test_user, monkeypatch
):
    configure(monkeypatch, test_user.id)
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=123,
        chat_type=ChatType.PRIVATE,
        indexing_enabled=True,
        title="Person",
    )
    db_session.add(chat)
    await db_session.flush()
    await enqueue_event(db_session, chat, 7, "message.updated", "v2", datetime.now(UTC))
    post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
    client = type("Client", (), {"post": post})()
    assert await deliver_batch(db_session, client) == 0
    row = (await db_session.scalars(select(CorrespondenceEvent))).one()
    assert row.delivered_at is None and row.attempts == 1
    first = post.call_args.kwargs["json"]
    row.next_attempt_at = datetime.now(UTC)
    post.side_effect = None
    post.return_value = httpx.Response(202, json={"accepted": True, "duplicate": True})
    assert await deliver_batch(db_session, client) == 1
    assert post.call_args.kwargs["json"] == first
    assert row.delivered_at is not None


async def test_first_delivery_after_an_outage_makes_the_whole_backlog_due(
    db_session, test_user, monkeypatch
):
    configure(monkeypatch, test_user.id)
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=124,
        chat_type=ChatType.PRIVATE,
        indexing_enabled=True,
        title="Person",
    )
    db_session.add(chat)
    await db_session.flush()
    for message_id in (1, 2, 3):
        await enqueue_event(
            db_session, chat, message_id, "message.new", "v1", datetime.now(UTC)
        )
    post = AsyncMock(side_effect=httpx.ConnectError("receiver is off"))
    client = type("Client", (), {"post": post})()
    assert await deliver_batch(db_session, client) == 0
    rows = (await db_session.scalars(select(CorrespondenceEvent))).all()
    # Each failed event backs off on its own, so they would come due at scattered times.
    assert all(row.next_attempt_at > datetime.now(UTC) for row in rows)
    rows[0].next_attempt_at = datetime.now(UTC)
    post.side_effect = None
    post.return_value = httpx.Response(202, json={"accepted": True})
    assert await deliver_batch(db_session, client) == 1
    # The receiver answers again: the rest is due now, so one conversation is read once, not once per event.
    assert all(
        row.delivered_at or row.next_attempt_at <= datetime.now(UTC) for row in rows
    )
    assert await deliver_batch(db_session, client) == 2


async def test_destination_change_does_not_replay_private_backlog_to_a_new_destination(
    db_session, test_user, monkeypatch
):
    configure(monkeypatch, test_user.id)
    chat = TelegramChat(
        user_id=test_user.id,
        telegram_chat_id=123,
        chat_type=ChatType.PRIVATE,
        indexing_enabled=True,
        title="Person",
    )
    db_session.add(chat)
    await db_session.flush()
    await enqueue_event(db_session, chat, 7, "message.new", "v1", datetime.now(UTC))
    monkeypatch.setattr(
        get_settings(),
        "correspondence_webhook_token",
        "a-different-private-source-token-long",
    )
    client = type("Client", (), {"post": AsyncMock()})()
    assert await deliver_batch(db_session, client) == 0
    client.post.assert_not_called()
    assert (await db_session.scalars(select(CorrespondenceEvent))).one().revoked is True
