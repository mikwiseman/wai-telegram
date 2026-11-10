from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.chat import ChatType, TelegramChat
from app.models.media import MediaObject, MediaObjectStatus, MediaStage
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.user import User
from app.services.media_processing_service import mark_media_retry
from app.tasks.media_tasks import (
    RETRY_DELAYS_SECONDS,
    _exact_retry_after,
    _find_pending_and_reap_stale,
    _is_terminal_error,
    _retry_or_finish,
    dispatch_pending_media,
    enqueue_media_processing,
    fetch_message_media_task,
    index_message_media_task,
    process_message_media_task,
    recover_interrupted_media_stage,
)


def test_retry_policy_uses_the_full_bounded_schedule():
    assert RETRY_DELAYS_SECONDS == (15, 60, 300, 900, 3600, 21_600, 86_400)


def test_retry_after_provider_value_overrides_the_schedule():
    provider_error = RuntimeError("rate limited")
    provider_error.retry_after_seconds = 321
    assert _exact_retry_after(provider_error) == 321

    header_error = RuntimeError("rate limited")
    header_error.response = SimpleNamespace(headers={"Retry-After": "654"})
    assert _exact_retry_after(header_error) == 654


def test_retry_path_checkpoints_before_rescheduling():
    message_id = uuid4()
    error = RuntimeError("temporary")
    task = SimpleNamespace(
        request=SimpleNamespace(retries=2),
        retry=MagicMock(side_effect=RuntimeError("celery retry")),
    )
    mark = MagicMock(return_value="checkpoint")
    with (
        patch("app.tasks.media_tasks.mark_media_retry", new=mark),
        patch("app.tasks.media_tasks.run_async") as run,
        pytest.raises(RuntimeError, match="celery retry"),
    ):
        _retry_or_finish(task, message_id, error)

    mark.assert_called_once_with(message_id, error, 300)
    run.assert_called_once_with("checkpoint")
    task.retry.assert_called_once_with(exc=error, countdown=300)


def test_terminal_and_exhausted_errors_finish_without_rescheduling():
    from app.services.media_cache_service import MediaDiskFull, MediaSourceDeleted
    from app.services.media_content_service import (
        MediaNoSpeechError,
        MediaProcessingConfigurationError,
    )

    for error in (
        MediaDiskFull("full"),
        MediaSourceDeleted("gone"),
        MediaNoSpeechError("silent"),
        MediaProcessingConfigurationError("unsupported"),
    ):
        assert _is_terminal_error(error) is True

    message_id = uuid4()
    task = SimpleNamespace(
        request=SimpleNamespace(retries=len(RETRY_DELAYS_SECONDS)),
        retry=MagicMock(),
    )
    error = RuntimeError("exhausted")
    failed = MagicMock(return_value="failed")
    with (
        patch("app.tasks.media_tasks._mark_failed", new=failed),
        patch("app.tasks.media_tasks.run_async") as run,
    ):
        _retry_or_finish(task, message_id, error)

    failed.assert_called_once_with(message_id, error)
    run.assert_called_once_with("failed")
    task.retry.assert_not_called()


def test_media_pipeline_chains_three_isolated_queues():
    message_id = uuid4()
    with (
        patch(
            "app.tasks.media_tasks.fetch_media_stage",
            new=MagicMock(return_value=object()),
        ),
        patch("app.tasks.media_tasks.run_async", return_value="cached"),
        patch.object(process_message_media_task, "apply_async") as process_apply,
    ):
        fetch_result = fetch_message_media_task.run(str(message_id))

    process_apply.assert_called_once_with(
        args=[str(message_id)],
        queue="media-process",
    )
    assert fetch_result["status"] == "cached"

    with (
        patch(
            "app.tasks.media_tasks.extract_media_stage",
            new=MagicMock(return_value=object()),
        ),
        patch("app.tasks.media_tasks.run_async", return_value="checkpointed"),
        patch.object(index_message_media_task, "apply_async") as index_apply,
    ):
        process_result = process_message_media_task.run(str(message_id))

    index_apply.assert_called_once_with(
        args=[str(message_id)],
        queue="media-index",
    )
    assert process_result["status"] == "checkpointed"
    assert fetch_message_media_task.max_retries == len(RETRY_DELAYS_SECONDS)
    assert process_message_media_task.max_retries == len(RETRY_DELAYS_SECONDS)
    assert index_message_media_task.max_retries == len(RETRY_DELAYS_SECONDS)


def test_enqueue_routes_fetch_to_persistent_client_worker():
    message_id = uuid4()
    with patch.object(fetch_message_media_task, "apply_async") as apply:
        enqueue_media_processing([message_id])

    apply.assert_called_once_with(
        args=[str(message_id)],
        queue="media-fetch",
    )


def test_deferred_pipeline_does_not_dispatch_media_jobs():
    with patch(
        "app.tasks.media_tasks.settings",
        SimpleNamespace(media_pipeline_enabled=False),
    ):
        result = dispatch_pending_media.run()

    assert result == {"dispatched": 0, "deferred": 1}


def test_fetch_handoff_publish_failure_uses_durable_retry_path():
    message_id = uuid4()
    publish_error = RuntimeError("broker unavailable")
    with (
        patch(
            "app.tasks.media_tasks.fetch_media_stage",
            new=MagicMock(return_value=object()),
        ),
        patch("app.tasks.media_tasks.run_async", return_value="cached"),
        patch.object(
            process_message_media_task,
            "apply_async",
            side_effect=publish_error,
        ),
        patch("app.tasks.media_tasks._retry_or_finish") as retry,
    ):
        fetch_message_media_task.run(str(message_id))

    retry.assert_called_once_with(fetch_message_media_task, message_id, publish_error)


def test_process_handoff_publish_failure_uses_durable_retry_path():
    message_id = uuid4()
    publish_error = RuntimeError("broker unavailable")
    with (
        patch(
            "app.tasks.media_tasks.extract_media_stage",
            new=MagicMock(return_value=object()),
        ),
        patch("app.tasks.media_tasks.run_async", return_value="checkpointed"),
        patch.object(
            index_message_media_task,
            "apply_async",
            side_effect=publish_error,
        ),
        patch("app.tasks.media_tasks._retry_or_finish") as retry,
    ):
        process_message_media_task.run(str(message_id))

    retry.assert_called_once_with(process_message_media_task, message_id, publish_error)


async def test_dispatch_claims_pending_rows_once(db_session, test_user):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=321,
        chat_type=ChatType.PRIVATE,
        title="Queue",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=654,
        text=None,
        has_media=True,
        media_type="video",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch(
            "app.tasks.media_tasks.get_db_context",
            side_effect=test_db_context,
        ),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 20
        first = await _find_pending_and_reap_stale()
        second = await _find_pending_and_reap_stale()

    assert first == [message.id]
    assert second == []
    assert message.media_processing_status == MediaProcessingStatus.QUEUED


async def test_dispatch_does_not_backfill_uninitialized_historical_media(
    db_session,
    test_user,
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=322,
        chat_type=ChatType.PRIVATE,
        title="Historical queue",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=655,
        text="Старая расшифровка.",
        has_media=True,
        media_type="voice",
        media_processing_status=None,
        transcribed_at=datetime.now(UTC),
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch(
            "app.tasks.media_tasks.get_db_context",
            side_effect=test_db_context,
        ),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 20
        claimed = await _find_pending_and_reap_stale()

    assert claimed == []
    assert message.media_processing_status is None


async def test_dispatch_ignores_pending_media_for_inactive_user(
    db_session,
    test_user,
):
    test_user.is_active = False
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=324,
        chat_type=ChatType.PRIVATE,
        title="Archived queue",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=658,
        has_media=True,
        media_type="video",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch("app.tasks.media_tasks.get_db_context", side_effect=test_db_context),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 20
        claimed = await _find_pending_and_reap_stale()

    assert claimed == []
    assert message.media_processing_status == MediaProcessingStatus.PENDING


async def test_inactive_queued_media_does_not_consume_owner_queue_capacity(
    db_session,
    test_user,
):
    archived_user = User(
        email="archived-media@example.com",
        password_hash="not-used",
        is_active=False,
    )
    db_session.add(archived_user)
    await db_session.flush()
    owner_chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=325,
        chat_type=ChatType.PRIVATE,
        title="Owner queue",
    )
    archive_chat = TelegramChat(
        indexing_enabled=True,
        user_id=archived_user.id,
        telegram_chat_id=326,
        chat_type=ChatType.PRIVATE,
        title="Archived queue",
    )
    db_session.add_all([owner_chat, archive_chat])
    await db_session.flush()
    now = datetime.now(UTC)
    owner_pending = TelegramMessage(
        chat_id=owner_chat.id,
        telegram_message_id=659,
        has_media=True,
        media_type="video",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=now,
    )
    archived_queued = TelegramMessage(
        chat_id=archive_chat.id,
        telegram_message_id=660,
        has_media=True,
        media_type="video",
        media_processing_status=MediaProcessingStatus.QUEUED,
        media_processing_started_at=now,
        sent_at=now,
    )
    db_session.add_all([owner_pending, archived_queued])
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch("app.tasks.media_tasks.get_db_context", side_effect=test_db_context),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 1
        claimed = await _find_pending_and_reap_stale()

    assert claimed == [owner_pending.id]
    assert owner_pending.media_processing_status == MediaProcessingStatus.QUEUED
    assert archived_queued.media_processing_status == MediaProcessingStatus.QUEUED


async def test_dispatch_prioritizes_live_pending_media_over_historical_backfill(
    db_session,
    test_user,
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=323,
        chat_type=ChatType.PRIVATE,
        title="Live queue priority",
    )
    db_session.add(chat)
    await db_session.flush()
    now = datetime.now(UTC)
    live_photo = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=656,
        text=None,
        has_media=True,
        media_type="photo",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=now - timedelta(days=1),
    )
    historical_voice = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=657,
        text="Старая расшифровка.",
        has_media=True,
        media_type="voice",
        media_processing_status=None,
        transcribed_at=now,
        sent_at=now,
    )
    db_session.add_all([live_photo, historical_voice])
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch(
            "app.tasks.media_tasks.get_db_context",
            side_effect=test_db_context,
        ),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 1
        claimed = await _find_pending_and_reap_stale()

    assert claimed == [live_photo.id]
    assert live_photo.media_processing_status == MediaProcessingStatus.QUEUED
    assert historical_voice.media_processing_status is None


async def test_media_waiting_for_a_retry_holds_no_dispatch_slot(db_session, test_user):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=327,
        chat_type=ChatType.PRIVATE,
        title="Retry wait",
    )
    db_session.add(chat)
    await db_session.flush()
    now = datetime.now(UTC)
    failing_photo = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=661,
        has_media=True,
        media_type="photo",
        media_processing_status=MediaProcessingStatus.PROCESSING,
        media_processing_started_at=now,
        sent_at=now,
    )
    # Its retry was due seven hours ago and never ran: the broker lost it.
    lost_retry = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=662,
        has_media=True,
        media_type="photo",
        media_processing_status=MediaProcessingStatus.QUEUED,
        media_processing_started_at=now - timedelta(hours=7),
        sent_at=now,
    )
    new_voice = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=663,
        has_media=True,
        media_type="voice",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=now,
    )
    db_session.add_all([failing_photo, lost_retry, new_voice])
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with patch(
        "app.services.media_processing_service.get_db_context",
        side_effect=test_db_context,
    ):
        await mark_media_retry(failing_photo.id, RuntimeError("unsupported"), 86_400)
    await db_session.refresh(failing_photo)

    # The retry is a day away: the row records when it is due.
    assert failing_photo.media_processing_status == MediaProcessingStatus.QUEUED
    retry_due = failing_photo.media_processing_started_at
    assert retry_due is not None
    assert retry_due.replace(tzinfo=UTC) > now + timedelta(hours=23)

    with (
        patch("app.tasks.media_tasks.get_db_context", side_effect=test_db_context),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 1
        first = await _find_pending_and_reap_stale()
        task_settings.media_dispatch_target_depth = 2
        second = await _find_pending_and_reap_stale()
    await db_session.refresh(failing_photo)

    # One slot, and the waiting photo does not take it: the new voice message goes.
    assert first == [new_voice.id]
    # The lost retry came back to pending and is dispatched when a slot opens.
    assert second == [lost_retry.id]
    assert failing_photo.media_processing_status == MediaProcessingStatus.QUEUED
    assert failing_photo.media_processing_started_at == retry_due


async def test_stale_processing_job_is_requeued_instead_of_failed(
    db_session, test_user
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=325,
        chat_type=ChatType.PRIVATE,
        title="Crash recovery",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=659,
        has_media=True,
        media_type="video",
        media_processing_status=MediaProcessingStatus.PROCESSING,
        media_processing_started_at=datetime.now(UTC) - timedelta(hours=3),
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()
    db_session.add(
        MediaObject(
            user_id=test_user.id,
            message_id=message.id,
            cache_key="9" * 64,
            relative_path="99/cache/original.mp4",
            sha256="8" * 64,
            status=MediaObjectStatus.EXTRACTING,
            stage=MediaStage.EXTRACTION,
        )
    )
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with (
        patch("app.tasks.media_tasks.get_db_context", side_effect=test_db_context),
        patch("app.tasks.media_tasks.settings") as task_settings,
    ):
        task_settings.media_queue_stale_minutes = 360
        task_settings.media_processing_stale_minutes = 120
        task_settings.media_dispatch_target_depth = 20
        claimed = await _find_pending_and_reap_stale()

    assert claimed == [message.id]
    assert message.media_processing_status == MediaProcessingStatus.QUEUED
    assert message.media_processing_error_code is None


async def test_redelivered_index_stage_resets_only_interrupted_index_claim(
    db_session, test_user
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=326,
        chat_type=ChatType.PRIVATE,
        title="Index recovery",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=660,
        has_media=True,
        media_type="document",
        media_processing_status=MediaProcessingStatus.PROCESSING,
        sent_at=datetime.now(UTC),
    )
    db_session.add(message)
    await db_session.flush()
    media_object = MediaObject(
        user_id=test_user.id,
        message_id=message.id,
        cache_key="7" * 64,
        status=MediaObjectStatus.INDEXING,
        stage=MediaStage.INDEX,
    )
    db_session.add(media_object)
    await db_session.flush()

    @asynccontextmanager
    async def test_db_context():
        yield db_session

    with patch("app.tasks.media_tasks.get_db_context", side_effect=test_db_context):
        recovered = await recover_interrupted_media_stage(message.id, "index")

    assert recovered is True
    assert media_object.status == MediaObjectStatus.CACHED
