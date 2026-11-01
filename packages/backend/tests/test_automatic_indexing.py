from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from telethon.tl.types import Message, PeerUser, User as TelegramUser

from app.models.message import TelegramMessage
from app.models.settings import UserSettings
from app.models.sync_job import SyncJob, SyncStatus
from app.services.onboarding import create_invitation
from app.services.sync_service import sync_chats, sync_messages
from tests.factories import (
    SyncJobFactory,
    TelegramChatFactory,
    TelegramSessionFactory,
    UserSettingsFactory,
)

SINCE = datetime(2026, 1, 1, tzinfo=UTC)


async def test_automatic_invitation_exposes_period_without_enabling_owner(
    client, db_session, test_user
):
    guest, token = await create_invitation(
        db_session, test_user.id, 42, auto_index_since=SINCE
    )
    await db_session.commit()
    response = await client.post("/api/v1/onboarding/accept", json={"token": token})
    me = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {response.json()['access_token']}"},
    )
    assert me.json()["auto_index_since"].startswith("2026-01-01")
    preferences = (
        await db_session.execute(
            select(UserSettings).where(UserSettings.user_id == guest.id)
        )
    ).scalar_one()
    assert preferences.embedding_backfill_enabled is True
    assert test_user.auto_index_since is None


async def test_automatic_history_starts_without_chat_selection_and_refreshes_later(
    db_session, test_user
):
    from app.tasks.sync_tasks import resume_selected_history

    test_user.auto_index_since = SINCE
    db_session.add_all(
        [
            TelegramSessionFactory.create(user_id=test_user.id),
            UserSettingsFactory.create(
                user_id=test_user.id, embedding_backfill_enabled=True
            ),
        ]
    )
    await db_session.commit()

    @asynccontextmanager
    async def database():
        yield db_session
        await db_session.commit()

    with (
        patch("app.tasks.sync_tasks.get_db_context", database),
        patch("app.tasks.sync_tasks.indexing_has_capacity", return_value=True),
        patch("app.tasks.sync_tasks.sync_all_chats_task.delay") as dispatch,
    ):
        assert await resume_selected_history() == 1
        job = (await db_session.execute(select(SyncJob))).scalar_one()
        assert await resume_selected_history() == 0
        job.status = SyncStatus.COMPLETED
        await db_session.commit()
        assert await resume_selected_history() == 0
        job.updated_at = datetime.now(UTC) - timedelta(minutes=16)
        await db_session.commit()
        assert await resume_selected_history() == 1
        assert dispatch.call_count == 2


async def test_automatic_dialog_discovery_enables_every_chat(db_session, test_user):
    test_user.auto_index_since = SINCE
    test_user.invited_by_id = test_user.id
    await db_session.commit()
    dialog = MagicMock(
        entity=TelegramUser(id=42, first_name="Test"),
        message=None,
        date=SINCE,
        unread_count=0,
    )

    async def dialogs(**kwargs):
        yield dialog

    telegram = AsyncMock()
    telegram.iter_dialogs = MagicMock(side_effect=dialogs)
    with (
        patch("app.services.sync_service.get_client", return_value=telegram),
        patch("app.services.sync_service.record_request"),
    ):
        chats = await sync_chats(db_session, test_user.id)
    assert len(chats) == 1
    assert chats[0].indexing_enabled is True
    telegram.iter_dialogs.assert_called_once_with(limit=None)


async def test_history_stops_at_period_boundary_before_saving_or_embedding(
    db_session, test_user
):
    test_user.auto_index_since = SINCE
    chat = TelegramChatFactory.create(user_id=test_user.id)
    job = SyncJobFactory.create(user_id=test_user.id, chat_id=chat.id)
    db_session.add_all([chat, job])
    await db_session.commit()
    seen = []

    async def messages(*args, **kwargs):
        for number, date in [
            (4, SINCE + timedelta(days=1)),
            (3, SINCE),
            (2, SINCE - timedelta(seconds=1)),
            (1, SINCE - timedelta(days=300)),
        ]:
            seen.append(number)
            message = Message(
                id=number, peer_id=PeerUser(42), message=f"text {number}", date=date
            )
            message._text = message.message
            yield message

    telegram = AsyncMock()
    telegram.iter_messages = MagicMock(side_effect=messages)
    with (
        patch("app.services.sync_service._resolve_chat_entity", new_callable=AsyncMock),
        patch(
            "app.services.sync_service.embed_messages", new_callable=AsyncMock
        ) as embed,
        patch("app.services.sync_service.indexing_has_capacity", return_value=True),
    ):
        assert (
            await sync_messages(
                db_session, test_user.id, chat.id, job.id, client=telegram
            )
            == 2
        )
    stored = (await db_session.execute(select(TelegramMessage))).scalars().all()
    assert {message.telegram_message_id for message in stored} == {4, 3}
    assert set(embed.call_args.args[1]) == {message.id for message in stored}
    assert seen == [4, 3, 2]
    assert chat.history_synced_at is not None
    assert chat.history_cursor is None


async def test_bulk_sync_discovers_chats_before_loading_history(db_session, test_user):
    from app.tasks.sync_tasks import _run_bulk_sync

    test_user.auto_index_since = SINCE
    job = SyncJobFactory.create(user_id=test_user.id)
    db_session.add(job)
    await db_session.commit()

    @asynccontextmanager
    async def database():
        yield db_session

    with (
        patch("app.tasks.sync_tasks.get_db_context", database),
        patch("app.tasks.sync_tasks.redis_client"),
        patch("app.tasks.sync_tasks.sync_chats", new_callable=AsyncMock) as discover,
    ):
        await _run_bulk_sync(test_user.id, job.id, 0, True)
    discover.assert_awaited_once_with(db_session, test_user.id)


async def test_listener_auto_connects_new_chat_and_ignores_old_events(
    db_session, test_user
):
    from app.listener.main import TelegramListener

    test_user.auto_index_since = SINCE
    test_user.invited_by_id = test_user.id
    await db_session.commit()
    listener = TelegramListener()
    entity = TelegramUser(id=42, first_name="New chat")
    chat = await listener._auto_create_chat(db_session, test_user.id, entity, 42)
    assert chat.indexing_enabled is True
    await db_session.commit()

    @asynccontextmanager
    async def database():
        yield db_session

    event = SimpleNamespace(
        chat_id=42,
        message=SimpleNamespace(
            id=1,
            text="old",
            media=None,
            action=None,
            date=SINCE - timedelta(seconds=1),
        ),
        get_sender=AsyncMock(return_value=None),
    )
    with (
        patch("app.listener.main.get_db_context", database),
        patch("app.listener.main.embed_messages", new_callable=AsyncMock) as embed,
    ):
        await listener._handle_message(test_user.id, event)
    assert (await db_session.execute(select(TelegramMessage))).scalars().all() == []
    embed.assert_not_called()
    event.get_sender.assert_not_called()
