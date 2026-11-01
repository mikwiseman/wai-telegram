from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from sqlalchemy import select

from app.models.sync_job import SyncJob, SyncStatus
from tests.factories import (
    SyncJobFactory,
    TelegramChatFactory,
    TelegramSessionFactory,
    UserSettingsFactory,
)


async def test_history_recovers_lost_dispatch_and_new_selection(db_session, test_user):
    from app.tasks.sync_tasks import resume_selected_history

    chat = TelegramChatFactory.create(user_id=test_user.id)
    job = SyncJobFactory.create(
        user_id=test_user.id,
        updated_at=datetime.now(UTC) - timedelta(hours=1),
    )
    db_session.add_all(
        [
            chat,
            job,
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
        dispatch.assert_called_once_with(str(test_user.id), str(job.id), 0, True)
        # A queued task is not duplicated on every beat.
        assert await resume_selected_history() == 0
        # A new selection made during the previous run is picked up automatically.
        job.status = SyncStatus.COMPLETED
        await db_session.commit()
        assert await resume_selected_history() == 1
        jobs = (await db_session.execute(select(SyncJob))).scalars().all()
        assert len(jobs) == 2
        assert dispatch.call_count == 2


async def test_history_recovery_respects_disk_and_opt_in(db_session, test_user):
    from app.tasks.sync_tasks import resume_selected_history

    preferences = UserSettingsFactory.create(
        user_id=test_user.id, embedding_backfill_enabled=False
    )
    db_session.add_all(
        [
            preferences,
            TelegramChatFactory.create(user_id=test_user.id),
            TelegramSessionFactory.create(user_id=test_user.id),
        ]
    )
    await db_session.commit()

    @asynccontextmanager
    async def database():
        yield db_session
        await db_session.commit()

    with (
        patch("app.tasks.sync_tasks.get_db_context", database),
        patch(
            "app.tasks.sync_tasks.indexing_has_capacity", return_value=True
        ) as capacity,
        patch("app.tasks.sync_tasks.sync_all_chats_task.delay") as dispatch,
    ):
        assert await resume_selected_history() == 0
        preferences.embedding_backfill_enabled = True
        await db_session.commit()
        capacity.return_value = False
        assert await resume_selected_history() == 0
        dispatch.assert_not_called()


async def test_history_resumes_committed_batches_after_interruption(
    db_session, test_user
):
    from unittest.mock import AsyncMock, MagicMock
    from telethon.tl.types import Message, PeerUser
    from app.models.message import TelegramMessage
    from app.services.sync_service import sync_messages

    chat = TelegramChatFactory.create(user_id=test_user.id)
    job = SyncJobFactory.create(user_id=test_user.id, chat_id=chat.id)
    db_session.add_all([chat, job])
    await db_session.commit()
    telegram = AsyncMock()

    async def interrupted(*args, **kwargs):
        for number in [5, 4]:
            message = Message(
                id=number,
                peer_id=PeerUser(42),
                message=f"school {number}",
                date=datetime.now(UTC),
            )
            message._text = message.message
            yield message
        raise RuntimeError("worker interrupted")

    async def remaining(*args, **kwargs):
        for number in [3, 2, 1]:
            message = Message(
                id=number,
                peer_id=PeerUser(42),
                message=f"school {number}",
                date=datetime.now(UTC),
            )
            message._text = message.message
            yield message

    telegram.iter_messages = MagicMock(side_effect=interrupted)
    with (
        patch("app.services.sync_service._resolve_chat_entity", new_callable=AsyncMock),
        patch(
            "app.services.sync_service.embed_messages",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch("app.services.sync_service._jittered_sleep", new_callable=AsyncMock),
        patch("app.services.sync_service.record_request"),
        patch("app.services.sync_service.indexing_has_capacity", return_value=True),
        patch("app.services.sync_service.settings.sync_batch_size", 2),
    ):
        import pytest

        with pytest.raises(RuntimeError, match="worker interrupted"):
            await sync_messages(
                db_session, test_user.id, chat.id, job.id, client=telegram
            )
        assert chat.history_cursor == 4
        assert chat.history_synced_at is None
        assert (
            len((await db_session.execute(select(TelegramMessage))).scalars().all())
            == 2
        )
        telegram.iter_messages.side_effect = remaining
        assert (
            await sync_messages(
                db_session, test_user.id, chat.id, job.id, client=telegram
            )
            == 3
        )
        assert telegram.iter_messages.call_args.kwargs["offset_id"] == 4
        assert chat.history_cursor is None
        assert chat.history_synced_at is not None
        assert (
            len((await db_session.execute(select(TelegramMessage))).scalars().all())
            == 5
        )
