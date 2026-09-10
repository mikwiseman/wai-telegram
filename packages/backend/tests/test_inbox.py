from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import event

from app.models.chat import ChatType, TelegramChat
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.user import User


@pytest.fixture(autouse=True)
def listener_status():
    redis = AsyncMock()
    redis.exists.return_value = 1
    with patch("app.services.tool_registry.aioredis.from_url", return_value=redis):
        yield redis


async def seed_chat(db, owner, *, kind=ChatType.PRIVATE, count=4, at=None):
    at = at or datetime.now(UTC)
    chat = TelegramChat(
        user_id=owner.id,
        telegram_chat_id=int(uuid4().int % 10**10),
        chat_type=kind,
        title="Conversation",
        last_activity_at=at,
        last_message_id=count,
        last_sync_at=at - timedelta(days=30),
        total_messages_synced=count,
        unread_count=0,
    )
    db.add(chat)
    await db.flush()
    messages = [
        TelegramMessage(
            chat_id=chat.id,
            telegram_message_id=i,
            text=f"Message {i}",
            sender_name="Me" if i % 2 == 0 else "Colleague",
            is_outgoing=i % 2 == 0,
            sent_at=at - timedelta(minutes=count - i),
        )
        for i in range(1, count + 1)
    ]
    db.add_all(messages)
    await db.flush()
    return chat, messages


async def inbox(client, **arguments):
    response = await client.post(
        "/api/v1/tools/get_inbox", json={"arguments": arguments}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_inbox_batches_read_and_unread_conversations_without_classifying_them(
    auth_client, db_session, test_user, listener_status
):
    chat, messages = await seed_chat(db_session, test_user)
    messages[-1].text = "Thanks! I will send it tomorrow."
    group, _ = await seed_chat(db_session, test_user, kind=ChatType.SUPERGROUP)
    group.unread_count = 3
    await seed_chat(db_session, test_user, kind=ChatType.CHANNEL)
    await db_session.flush()

    result = await inbox(auth_client)
    assert {c["chat_id"] for c in result["conversations"]} == {
        str(chat.id),
        str(group.id),
    }
    assert result["listener_active"] is True
    assert result["has_more"] is False
    assert result["scope"]["chat_types"] == ["private", "group", "supergroup"]
    conversation = next(
        c for c in result["conversations"] if c["chat_id"] == str(chat.id)
    )
    assert [m["telegram_message_id"] for m in conversation["messages"]] == [1, 2, 3, 4]
    assert conversation["messages"][-1]["text"] == "Thanks! I will send it tomorrow."
    assert conversation["messages"][-1]["is_outgoing"] is True
    assert conversation["sync_recommended"] is False  # Quiet is not stale.
    assert conversation["latest_known_message_available"] is True
    listener_status.exists.assert_awaited_once()


async def test_inbox_does_not_return_another_owners_chat(
    auth_client, db_session, test_user
):
    owner = User(email="another@example.com", password_hash="unused", is_active=False)
    db_session.add(owner)
    await db_session.flush()
    other, _ = await seed_chat(db_session, owner)
    own, _ = await seed_chat(db_session, test_user)
    result = await inbox(auth_client)
    assert [c["chat_id"] for c in result["conversations"]] == [str(own.id)]
    response = await auth_client.post(
        "/api/v1/tools/get_inbox", json={"arguments": {"chat_ids": [str(other.id)]}}
    )
    assert response.status_code == 400
    assert "not found" in response.text.lower()


async def test_inbox_message_cursor_reads_older_messages_without_repeating_the_tail(
    auth_client, db_session, test_user
):
    chat, _ = await seed_chat(db_session, test_user, count=8)
    result = await inbox(auth_client, chat_ids=[str(chat.id)], messages_per_chat=3)
    conversation = result["conversations"][0]
    assert [m["telegram_message_id"] for m in conversation["messages"]] == [6, 7, 8]
    assert conversation["has_more_messages"] is True
    older = await auth_client.get(
        f"/api/v1/chats/{chat.id}/messages",
        params={"before": conversation["next_message_cursor"], "limit": 10},
    )
    assert older.status_code == 200
    assert [m["telegram_message_id"] for m in older.json()["messages"]] == [
        5,
        4,
        3,
        2,
        1,
    ]


async def test_inbox_chat_pagination_and_activity_filter(
    auth_client, db_session, test_user
):
    now = datetime.now(UTC)
    chats = [
        (await seed_chat(db_session, test_user, at=now - timedelta(days=i)))[0]
        for i in range(4)
    ]
    args = {"limit": 2, "active_since": (now - timedelta(days=2, hours=1)).isoformat()}
    first = await inbox(auth_client, **args)
    second = await inbox(auth_client, **args, cursor=first["next_cursor"])
    assert [c["chat_id"] for c in first["conversations"]] == [
        str(c.id) for c in chats[:2]
    ]
    assert [c["chat_id"] for c in second["conversations"]] == [str(chats[2].id)]
    assert second["has_more"] is False


async def test_inbox_reports_missing_latest_message_but_hides_deleted_content(
    auth_client, db_session, test_user
):
    chat, messages = await seed_chat(db_session, test_user)
    messages[-1].deleted_at = datetime.now(UTC)
    await db_session.flush()
    result = await inbox(auth_client, chat_ids=[str(chat.id)])
    conversation = result["conversations"][0]
    assert [m["telegram_message_id"] for m in conversation["messages"]] == [1, 2, 3]
    assert conversation["sync_recommended"] is False  # Known deletion is not a gap.
    chat.last_message_id = 10
    await db_session.flush()
    result = await inbox(auth_client, chat_ids=[str(chat.id)])
    assert result["conversations"][0]["sync_recommended"] is True
    assert result["conversations"][0]["latest_known_message_available"] is False


async def test_inbox_media_and_clipped_text_remain_visible(
    auth_client, db_session, test_user
):
    chat, messages = await seed_chat(db_session, test_user, count=1)
    message = messages[0]
    message.text = "x" * 10000
    message.has_media = True
    message.media_type = "voice"
    message.media_processing_status = MediaProcessingStatus.READY
    message.content_text = "The deadline is Friday. " * 100
    message.reply_to_message_id = 99
    message.thread_id = 7
    await db_session.flush()
    result = await inbox(auth_client, chat_ids=[str(chat.id)])
    row = result["conversations"][0]["messages"][0]
    assert len(row["text"]) <= 1200 and row["text_truncated"] is True
    assert row["content_preview"].startswith("The deadline")
    assert row["content_truncated"] is True
    assert row["media_processing_status"] == "ready"
    assert row["reply_to_message_id"] == 99 and row["thread_id"] == 7


async def test_inbox_uses_constant_number_of_database_queries(
    auth_client, db_session, test_user
):
    for _ in range(20):
        await seed_chat(db_session, test_user)
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    engine = db_session.bind.sync_engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        result = await inbox(auth_client, limit=20)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert len(result["conversations"]) == 20
    assert len(statements) <= 4  # Auth + active owner + chats + batched message tails.


@pytest.mark.parametrize(
    "arguments",
    [
        {"chat_ids": []},
        {"chat_ids": ["invalid"]},
        {"limit": 41},
        {"limit": True},
        {"messages_per_chat": 31},
        {"cursor": "invalid"},
        {"chat_types": []},
        {"active_since": "invalid"},
    ],
)
async def test_inbox_rejects_invalid_input(auth_client, arguments):
    response = await auth_client.post(
        "/api/v1/tools/get_inbox", json={"arguments": arguments}
    )
    assert response.status_code == 400


async def test_inbox_requires_authentication(client):
    response = await client.post("/api/v1/tools/get_inbox", json={"arguments": {}})
    assert response.status_code == 401


async def test_inbox_selected_channel_keeps_context_before_activity_bound(
    auth_client, db_session, test_user
):
    now = datetime.now(UTC)
    channel, messages = await seed_chat(db_session, test_user, kind=ChatType.CHANNEL)
    messages[0].sent_at = now - timedelta(days=100)
    await db_session.flush()
    selected = await inbox(auth_client, chat_ids=[str(channel.id)])
    filtered = await inbox(
        auth_client,
        chat_types=["channel"],
        active_since=(now - timedelta(days=1)).isoformat(),
    )
    for result in (selected, filtered):
        assert [
            m["telegram_message_id"] for m in result["conversations"][0]["messages"]
        ] == [1, 2, 3, 4]


async def test_inbox_rejects_cursor_with_changed_scope(
    auth_client, db_session, test_user
):
    for _ in range(2):
        await seed_chat(db_session, test_user)
    first = await inbox(auth_client, limit=1)
    response = await auth_client.post(
        "/api/v1/tools/get_inbox",
        json={"arguments": {"cursor": first["next_cursor"], "chat_types": ["channel"]}},
    )
    assert response.status_code == 400


async def test_inbox_keeps_evidence_when_listener_status_is_unavailable(
    auth_client, db_session, test_user, listener_status
):
    await seed_chat(db_session, test_user)
    listener_status.exists.side_effect = OSError("Unavailable")
    result = await inbox(auth_client)
    assert result["listener_active"] is None
    assert len(result["conversations"]) == 1


async def test_inbox_exposes_unsynced_chat_as_gap(auth_client, db_session, test_user):
    chat, _ = await seed_chat(db_session, test_user, count=0)
    chat.last_message_id = 12
    await db_session.flush()
    result = await inbox(auth_client)
    conversation = result["conversations"][0]
    assert conversation["messages"] == []
    assert conversation["sync_recommended"] is True
