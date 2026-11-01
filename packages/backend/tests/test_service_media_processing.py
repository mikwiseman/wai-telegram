from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from sqlalchemy import select

from app.models.chat import TelegramChat
from app.models.media import (
    MediaObject,
    MediaObjectStatus,
    MediaStage,
    TranscriptSegment,
)
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.services.media_content_service import TranscriptPart, TranscriptionResult
from app.services.media_processing_service import (
    ClaimedMediaMessage,
    ProcessedMediaContent,
    _content_from_checkpoint,
    _transcribe_resumable,
    _save_content_checkpoint,
    claim_media_message,
    process_local_media,
    split_audio_for_transcription,
    summarize_existing_transcript,
    index_media_stage,
)


async def test_claim_is_atomic_and_prevents_duplicate_billable_work(
    db_session,
    test_user,
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=123,
        chat_type="private",
        title="Media",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=456,
        text="caption",
        has_media=True,
        media_type="audio",
        media_processing_status=MediaProcessingStatus.PENDING,
        sent_at=chat.created_at,
    )
    db_session.add(message)
    await db_session.flush()

    claimed = await claim_media_message(db_session, message.id)
    duplicate = await claim_media_message(db_session, message.id)

    assert claimed is not None
    assert claimed.user_id == test_user.id
    assert claimed.telegram_message_id == 456
    assert duplicate is None
    assert message.media_processing_status == MediaProcessingStatus.PROCESSING
    assert message.media_processing_attempts == 1


async def test_claim_reuses_legacy_transcript_stored_in_text(
    db_session,
    test_user,
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=124,
        chat_type="private",
        title="Legacy media",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=457,
        text="Старая расшифровка.",
        has_media=True,
        media_type="voice",
        media_processing_status=MediaProcessingStatus.PENDING,
        transcribed_at=datetime.now(UTC),
        sent_at=chat.created_at,
    )
    db_session.add(message)
    await db_session.flush()

    claimed = await claim_media_message(db_session, message.id)

    assert claimed is not None
    assert claimed.caption is None
    assert claimed.existing_content_text == "Старая расшифровка."
    assert claimed.legacy_transcript_in_text is True


async def test_audio_is_transcribed_once_then_summarized(tmp_path):
    source = tmp_path / "voice.ogg"
    source.write_bytes(b"audio")

    with (
        patch(
            "app.services.media_processing_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
            return_value=TranscriptionResult(
                text="Полная расшифровка.",
                segments=(
                    TranscriptPart(0, 1500, "Полная расшифровка.", "0", 0.98, "ru"),
                ),
            ),
        ) as transcribe,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Короткое резюме.",
        ) as summarize,
    ):
        result = await process_local_media(source, "audio", "audio/ogg")

    assert result.content_text == "Полная расшифровка."
    assert result.content_summary == "Короткое резюме."
    assert result.content_model == "deepgram:nova-3"
    assert result.segments[0].speaker == "0"
    transcribe.assert_awaited_once_with(source, "audio/ogg")
    summarize.assert_awaited_once_with("Полная расшифровка.")


async def test_video_extracts_compact_audio_before_transcription(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    extracted = tmp_path / "meeting.ogg"

    with (
        patch(
            "app.services.media_processing_service.extract_video_audio",
            new_callable=AsyncMock,
            return_value=extracted,
        ) as extract,
        patch(
            "app.services.media_processing_service.analyze_video_timeline",
            new_callable=AsyncMock,
            return_value="[0.000s] Meeting room.",
        ),
        patch(
            "app.services.media_processing_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
            return_value=TranscriptionResult(text="Transcript", segments=()),
        ) as transcribe,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Summary",
        ),
    ):
        await process_local_media(source, "video", "video/mp4")

    extract.assert_awaited_once_with(source)
    transcribe.assert_awaited_once_with(extracted, "audio/ogg")


async def test_silent_video_is_ready_from_visual_timeline(tmp_path):
    from app.services.media_processing_service import MediaConversionError

    source = tmp_path / "silent.mp4"
    source.write_bytes(b"video")
    with (
        patch(
            "app.services.media_processing_service.analyze_video_timeline",
            new_callable=AsyncMock,
            return_value="[0.000s] Slide: roadmap.",
        ),
        patch(
            "app.services.media_processing_service.extract_video_audio",
            new_callable=AsyncMock,
            side_effect=MediaConversionError("no audio"),
        ),
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Roadmap slide.",
        ),
    ):
        result = await process_local_media(source, "video", "video/mp4")

    assert result.transcribed is False
    assert "Visual timeline" in result.content_text
    assert "roadmap" in result.content_text
    assert result.content_model.endswith(":vision")


async def test_archives_remain_downloadable_without_failed_extraction(tmp_path):
    source = tmp_path / "archive.zip"
    source.write_bytes(b"PK")

    result = await process_local_media(source, "document", "application/zip")

    assert result.download_only is True
    assert result.content_text is None
    assert result.content_model == "download-only"


async def test_unknown_documents_remain_downloadable_when_converter_rejects_format(
    tmp_path,
):
    from app.services.media_content_service import MediaUnsupportedFormatError

    source = tmp_path / "firmware.bin"
    source.write_bytes(b"binary")
    with patch(
        "app.services.media_processing_service.extract_document_text",
        new_callable=AsyncMock,
        side_effect=MediaUnsupportedFormatError("unsupported"),
    ):
        result = await process_local_media(
            source, "document", "application/octet-stream"
        )

    assert result.download_only is True
    assert result.content_text is None
    assert result.content_model == "download-only"


def test_download_only_flag_survives_the_index_checkpoint_boundary():
    message = TelegramMessage(
        content_summary="Download-only Telegram attachment (application/zip).",
        content_model="download-only",
        summary_model="none",
    )

    checkpoint = _content_from_checkpoint(message, ())

    assert checkpoint.download_only is True


async def test_long_audio_is_chunked_to_avoid_provider_request_timeout(tmp_path):
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"audio")
    chunks = [tmp_path / "chunk-0000.ogg", tmp_path / "chunk-0001.ogg"]

    with (
        patch(
            "app.services.media_processing_service.split_audio_for_transcription",
            new_callable=AsyncMock,
            return_value=chunks,
        ) as split,
        patch(
            "app.services.media_processing_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
            side_effect=[
                TranscriptionResult(
                    text="Первая часть.",
                    segments=(TranscriptPart(100, 500, "Первая часть."),),
                ),
                TranscriptionResult(
                    text="Вторая часть.",
                    segments=(TranscriptPart(200, 700, "Вторая часть."),),
                ),
            ],
        ) as transcribe,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Summary",
        ),
    ):
        result = await process_local_media(
            source,
            "audio",
            "audio/mp4",
            duration_seconds=1_201,
        )

    assert result.content_text == "Первая часть.\n\nВторая часть."
    assert [part.start_ms for part in result.segments] == [100, 600_200]
    split.assert_awaited_once_with(source)
    assert transcribe.await_count == 2
    assert all(call.args[1] == "audio/ogg" for call in transcribe.await_args_list)


async def test_transcription_chunk_checkpoint_skips_completed_provider_work(tmp_path):
    source = tmp_path / "chunk-0000.ogg"
    source.write_bytes(b"audio")
    completed = TranscriptionResult(
        text="Already done.",
        segments=(TranscriptPart(0, 500, "Already done.", "0", 0.9, "en"),),
    )
    with patch(
        "app.services.media_processing_service.transcribe_media_file_detailed",
        new_callable=AsyncMock,
        return_value=completed,
    ) as provider:
        first = await _transcribe_resumable(source, "audio/ogg", 0)
        second = await _transcribe_resumable(source, "audio/ogg", 0)

    assert first == completed
    assert second == completed
    provider.assert_awaited_once_with(source, "audio/ogg")


async def test_audio_split_has_only_one_output_target(tmp_path):
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"audio")

    async def create_fake_process(*args, **_kwargs):
        bitrate_index = args.index("-b:a")
        assert args[bitrate_index + 1] == "32k"
        assert args[bitrate_index + 2] == "-f"
        (tmp_path / "chunk-0000.ogg").write_bytes(b"chunk")
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (b"", b"")
        return process

    with (
        patch(
            "app.services.media_processing_service.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ),
        patch(
            "app.services.media_processing_service.asyncio.create_subprocess_exec",
            side_effect=create_fake_process,
        ),
    ):
        chunks = await split_audio_for_transcription(source)

    assert chunks == [tmp_path / "chunk-0000.ogg"]


async def test_document_text_is_extracted_then_summarized(tmp_path):
    source = tmp_path / "brief.pdf"
    source.write_bytes(b"pdf")

    with (
        patch(
            "app.services.media_processing_service.extract_document_text",
            new_callable=AsyncMock,
            return_value="Extracted text",
        ) as extract,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Summary",
        ),
    ):
        result = await process_local_media(source, "document", "application/pdf")

    assert result.content_text == "Extracted text"
    assert result.content_summary == "Summary"
    assert result.content_model == "markitdown:0.1.7"
    extract.assert_awaited_once_with(source)


async def test_photo_uses_single_vision_call_without_second_summary(tmp_path):
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"jpeg")

    analysis = type(
        "Analysis",
        (),
        {"visible_text": "Встреча 10:00", "summary": "Фото расписания."},
    )()
    with (
        patch(
            "app.services.media_processing_service.analyze_image",
            new_callable=AsyncMock,
            return_value=analysis,
        ) as analyze,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
        ) as summarize,
    ):
        result = await process_local_media(source, "photo", "image/jpeg")

    assert result.content_text == "Встреча 10:00"
    assert result.content_summary == "Фото расписания."
    assert result.content_model == "gpt-5.6-luna"
    analyze.assert_awaited_once_with(source, "image/jpeg")
    summarize.assert_not_awaited()


async def test_legacy_transcript_is_summarized_without_buying_transcription_again():
    job = ClaimedMediaMessage(
        id=uuid4(),
        user_id=uuid4(),
        chat_id=uuid4(),
        telegram_message_id=1,
        caption=None,
        media_type="voice",
        file_name="voice.ogg",
        mime_type="audio/ogg",
        file_size=123,
        duration_seconds=5,
        existing_content_text="Уже готовая расшифровка.",
        transcribed_at=datetime.now(UTC),
    )

    with (
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Резюме.",
        ),
        patch(
            "app.services.media_processing_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
        ) as transcribe,
    ):
        reused = await summarize_existing_transcript(job)

    assert reused is not None
    content, info = reused
    assert content.content_text == "Уже готовая расшифровка."
    assert content.content_summary == "Резюме."
    assert info.file_name == "voice.ogg"
    transcribe.assert_not_awaited()


async def test_content_checkpoint_persists_transcript_segments(
    db_session,
    test_user,
):
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=812,
        chat_type="private",
        title="Transcript checkpoint",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=813,
        has_media=True,
        media_type="audio",
        media_processing_status=MediaProcessingStatus.PROCESSING,
        sent_at=chat.created_at,
    )
    db_session.add(message)
    await db_session.commit()
    job = ClaimedMediaMessage(
        id=message.id,
        user_id=test_user.id,
        chat_id=chat.id,
        telegram_message_id=message.telegram_message_id,
        caption=None,
        media_type="audio",
        file_name="meeting.ogg",
        mime_type="audio/ogg",
        file_size=10,
        duration_seconds=4,
        existing_content_text=None,
        transcribed_at=None,
    )
    content = ProcessedMediaContent(
        content_text="Один. Два.",
        content_summary="Два спикера.",
        content_model="deepgram:nova-3",
        summary_model="gpt-test",
        transcribed=True,
        segments=(
            TranscriptPart(0, 1200, "Один.", "0", 0.9, "ru"),
            TranscriptPart(1300, 2400, "Два.", "1", 0.8, "ru"),
        ),
    )

    @asynccontextmanager
    async def use_test_session():
        yield db_session
        await db_session.commit()

    with patch(
        "app.services.media_processing_service.get_db_context",
        use_test_session,
    ):
        await _save_content_checkpoint(
            job,
            type(
                "Info",
                (),
                {
                    "media_type": "audio",
                    "file_name": "meeting.ogg",
                    "mime_type": "audio/ogg",
                    "file_size": 10,
                    "duration_seconds": 4,
                },
            )(),
            content,
        )

    rows = (
        await db_session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.message_id == message.id)
            .order_by(TranscriptSegment.sequence)
        )
    ).scalars()
    rows = list(rows)
    assert [(row.start_ms, row.end_ms, row.speaker) for row in rows] == [
        (0, 1200, "0"),
        (1300, 2400, "1"),
    ]


async def test_index_stage_does_not_mutate_deactivated_archive(db_session, test_user):
    test_user.is_active = False
    chat = TelegramChat(
        indexing_enabled=True,
        user_id=test_user.id,
        telegram_chat_id=814,
        chat_type="private",
        title="Archived media",
    )
    db_session.add(chat)
    await db_session.flush()
    message = TelegramMessage(
        chat_id=chat.id,
        telegram_message_id=815,
        has_media=True,
        media_type="document",
        content_summary="Existing checkpoint",
        media_processing_status=MediaProcessingStatus.QUEUED,
        sent_at=chat.created_at,
    )
    db_session.add(message)
    await db_session.flush()
    media_object = MediaObject(
        user_id=test_user.id,
        message_id=message.id,
        cache_key="4" * 64,
        status=MediaObjectStatus.CACHED,
        stage=MediaStage.INDEX,
    )
    db_session.add(media_object)
    await db_session.flush()

    @asynccontextmanager
    async def use_test_session():
        yield db_session

    with patch(
        "app.services.media_processing_service.get_db_context",
        use_test_session,
    ):
        result = await index_media_stage(message.id)

    assert result == "skipped"
    assert media_object.status == MediaObjectStatus.CACHED
    assert message.media_processing_status == MediaProcessingStatus.QUEUED


async def test_video_outside_configured_types_stays_download_only(tmp_path):
    source = tmp_path / "submission.mov"
    source.write_bytes(b"video")

    with patch(
        "app.services.media_processing_service.MEDIA_TRANSCRIPTION_TYPES",
        frozenset({"voice", "video_note"}),
    ):
        result = await process_local_media(source, "video", "video/mp4")

    assert result.download_only is True
    assert result.transcribed is False
    assert result.content_model == "download-only"


async def test_video_outside_configured_types_transcribes_when_requested(tmp_path):
    source = tmp_path / "submission.mov"
    source.write_bytes(b"video")
    extracted = tmp_path / "submission.ogg"

    with (
        patch(
            "app.services.media_processing_service.MEDIA_TRANSCRIPTION_TYPES",
            frozenset({"voice", "video_note"}),
        ),
        patch(
            "app.services.media_processing_service.extract_video_audio",
            new_callable=AsyncMock,
            return_value=extracted,
        ) as extract,
        patch(
            "app.services.media_processing_service.analyze_video_timeline",
            new_callable=AsyncMock,
            return_value="[0.000s] Talking head.",
        ),
        patch(
            "app.services.media_processing_service.transcribe_media_file_detailed",
            new_callable=AsyncMock,
            return_value=TranscriptionResult(text="Transcript", segments=()),
        ) as transcribe,
        patch(
            "app.services.media_processing_service.summarize_text",
            new_callable=AsyncMock,
            return_value="Summary",
        ),
    ):
        result = await process_local_media(
            source,
            "video",
            "video/mp4",
            transcription_requested=True,
        )

    assert result.transcribed is True
    extract.assert_awaited_once_with(source)
    transcribe.assert_awaited_once_with(extracted, "audio/ogg")
