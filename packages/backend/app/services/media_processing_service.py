import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Awaitable, Callable
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from telethon import TelegramClient

from app.core.config import get_settings
from app.core.database import get_db_context
from app.models.chat import TelegramChat
from app.services.correspondence_events import enqueue_media_ready, dispatch_events
from app.models.message import (
    MediaProcessingStatus,
    MessageContentChunk,
    TelegramMessage,
)
from app.models.media import (
    MediaObject,
    MediaObjectStatus,
    MediaStage,
    TranscriptSegment,
)
from app.models.user import User
from app.services.embedding_service import (
    PreparedMediaContentIndex,
    prepare_media_content_index,
)
from app.services.media_content_service import (
    ALL_TRANSCRIPTION_TYPES,
    IMAGE_MEDIA_TYPES,
    MEDIA_TRANSCRIPTION_TYPES,
    MediaDocumentExtractionError,
    MediaInfo,
    MediaNoSpeechError,
    MediaProcessingConfigurationError,
    MediaProcessingError,
    MediaProviderRequestError,
    MediaProviderResponseError,
    MediaUnsupportedFormatError,
    TranscriptPart,
    TranscriptionResult,
    analyze_image,
    analyze_video_timeline,
    extract_document_text,
    summarize_text,
    transcribe_media_file_detailed,
)
from app.services.media_cache_service import (
    MediaCacheError,
    MediaDiskFull,
    MediaDownloadStalled,
    MediaSourceDeleted,
    fetch_media_to_cache,
)
from app.services.telegram_client import (
    TelegramSessionUnauthorizedError,
    get_client,
)

logger = logging.getLogger(__name__)
settings = get_settings()
_media_clients: dict[UUID, TelegramClient] = {}


def _active_media_message_ids():
    return (
        select(TelegramMessage.id)
        .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
        .join(User, User.id == TelegramChat.user_id)
        .where(User.is_active.is_(True), TelegramChat.indexing_enabled.is_(True))
    )


class MediaDownloadError(MediaProcessingError):
    """Telegram did not provide the expected media file."""


class MediaConversionError(MediaProcessingError):
    """A video audio track could not be converted for transcription."""


class MediaIndexingError(MediaProcessingError):
    """Search embeddings could not be prepared completely."""


async def _get_media_client(user_id: UUID, db: AsyncSession) -> TelegramClient:
    """Reuse one Telethon client per user inside a persistent worker process."""
    cached = _media_clients.get(user_id)
    if cached is not None:
        if cached.is_connected():
            return cached
        _media_clients.pop(user_id, None)
        await cached.disconnect()

    client = await get_client(user_id, db)
    _media_clients[user_id] = client
    return client


async def _discard_media_client(user_id: UUID, expected_client: TelegramClient) -> None:
    """Remove and close a client after a network or authorization failure."""
    if _media_clients.get(user_id) is expected_client:
        _media_clients.pop(user_id, None)
    await expected_client.disconnect()


async def disconnect_media_clients() -> None:
    """Close all worker-local Telethon clients before process shutdown."""
    clients = list(_media_clients.values())
    _media_clients.clear()
    await asyncio.gather(*(client.disconnect() for client in clients))


@dataclass(frozen=True)
class ClaimedMediaMessage:
    id: UUID
    user_id: UUID
    chat_id: UUID
    telegram_message_id: int
    caption: str | None
    media_type: str
    file_name: str | None
    mime_type: str | None
    file_size: int | None
    duration_seconds: int | None
    existing_content_text: str | None
    transcribed_at: datetime | None
    legacy_transcript_in_text: bool = False
    transcription_requested: bool = False


@dataclass(frozen=True)
class ProcessedMediaContent:
    content_text: str | None
    content_summary: str
    content_model: str
    summary_model: str
    transcribed: bool
    segments: tuple[TranscriptPart, ...] = ()
    download_only: bool = False


def _content_from_checkpoint(
    message: TelegramMessage,
    segments: tuple[TranscriptPart, ...],
) -> ProcessedMediaContent:
    """Restore every terminal processing flag from the persisted checkpoint."""
    content_model = message.content_model or "unknown"
    return ProcessedMediaContent(
        content_text=message.content_text,
        content_summary=message.content_summary or "",
        content_model=content_model,
        summary_model=message.summary_model or "unknown",
        transcribed=message.transcribed_at is not None,
        segments=segments,
        download_only=content_model == "download-only",
    )


DOWNLOAD_ONLY_EXTENSIONS = frozenset(
    {".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".tgs", ".xz", ".zip"}
)
DOWNLOAD_ONLY_MIME_TYPES = frozenset(
    {
        "application/gzip",
        "application/vnd.rar",
        "application/x-7z-compressed",
        "application/x-rar-compressed",
        "application/x-tar",
        "application/x-tgsticker",
        "application/zip",
    }
)


def _is_download_only_media(path: Path, media_type: str, mime_type: str | None) -> bool:
    return (
        media_type == "other"
        or path.suffix.lower() in DOWNLOAD_ONLY_EXTENSIONS
        or (mime_type or "").lower() in DOWNLOAD_ONLY_MIME_TYPES
    )


def _transcription_checkpoint_path(path: Path, chunk_index: int) -> Path:
    safe_model = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in settings.deepgram_model
    )
    return path.parent / f"transcript-{chunk_index:06d}-{safe_model}.json"


async def _transcribe_resumable(
    path: Path,
    mime_type: str | None,
    chunk_index: int,
) -> TranscriptionResult:
    checkpoint = _transcription_checkpoint_path(path, chunk_index)
    if checkpoint.is_file():
        try:
            payload = json.loads(await asyncio.to_thread(checkpoint.read_text, "utf-8"))
            return TranscriptionResult(
                text=str(payload["text"]),
                segments=tuple(
                    TranscriptPart(
                        start_ms=int(segment["start_ms"]),
                        end_ms=int(segment["end_ms"]),
                        text=str(segment["text"]),
                        speaker=segment.get("speaker"),
                        confidence=segment.get("confidence"),
                        language=segment.get("language"),
                    )
                    for segment in payload.get("segments", [])
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediaProcessingError(
                f"Invalid transcription checkpoint {checkpoint.name}"
            ) from exc
    result = await transcribe_media_file_detailed(path, mime_type)
    payload = {
        "text": result.text,
        "segments": [
            {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
                "speaker": segment.speaker,
                "confidence": segment.confidence,
                "language": segment.language,
            }
            for segment in result.segments
        ],
    }
    temporary = checkpoint.with_suffix(".json.part")
    await asyncio.to_thread(
        temporary.write_text,
        json.dumps(payload, ensure_ascii=False),
        "utf-8",
    )
    await asyncio.to_thread(temporary.replace, checkpoint)
    return result


async def claim_media_message(
    db: AsyncSession,
    message_id: UUID,
) -> ClaimedMediaMessage | None:
    """Atomically claim pending media so duplicate tasks cannot spend twice."""
    active_message_ids = _active_media_message_ids()
    claimed_id = (
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id == message_id,
                TelegramMessage.id.in_(active_message_ids),
                TelegramMessage.has_media.is_(True),
                TelegramMessage.media_processing_status.in_(
                    (
                        MediaProcessingStatus.PENDING,
                        MediaProcessingStatus.QUEUED,
                    )
                ),
            )
            .values(
                media_processing_status=MediaProcessingStatus.PROCESSING,
                media_processing_attempts=TelegramMessage.media_processing_attempts + 1,
                media_processing_started_at=datetime.now(UTC),
                media_processing_error_code=None,
                media_processing_error=None,
            )
            .returning(TelegramMessage.id)
        )
    ).scalar_one_or_none()
    if claimed_id is None:
        return None

    await db.execute(
        update(MediaObject)
        .where(MediaObject.message_id == claimed_id)
        .values(
            status=MediaObjectStatus.EXTRACTING,
            stage=MediaStage.EXTRACTION,
            retry_after=None,
            error_code=None,
            error_detail=None,
        )
    )

    row = (
        await db.execute(
            select(
                TelegramMessage.id,
                TelegramChat.user_id,
                TelegramMessage.chat_id,
                TelegramMessage.telegram_message_id,
                TelegramMessage.text,
                TelegramMessage.media_type,
                TelegramMessage.media_file_name,
                TelegramMessage.media_mime_type,
                TelegramMessage.media_file_size,
                TelegramMessage.media_duration_seconds,
                TelegramMessage.content_text,
                TelegramMessage.transcribed_at,
                MediaObject.transcription_requested_at,
            )
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .outerjoin(MediaObject, MediaObject.message_id == TelegramMessage.id)
            .where(TelegramMessage.id == claimed_id)
        )
    ).one()
    legacy_transcript_in_text = bool(
        row.transcribed_at is not None
        and row.content_text is None
        and row.text
        and row.media_type in MEDIA_TRANSCRIPTION_TYPES
    )
    return ClaimedMediaMessage(
        id=row.id,
        user_id=row.user_id,
        chat_id=row.chat_id,
        transcription_requested=row.transcription_requested_at is not None,
        telegram_message_id=row.telegram_message_id,
        caption=None if legacy_transcript_in_text else row.text,
        media_type=row.media_type or "other",
        file_name=row.media_file_name,
        mime_type=row.media_mime_type,
        file_size=row.media_file_size,
        duration_seconds=row.media_duration_seconds,
        existing_content_text=(
            row.text if legacy_transcript_in_text else row.content_text
        ),
        transcribed_at=row.transcribed_at,
        legacy_transcript_in_text=legacy_transcript_in_text,
    )


async def extract_video_audio(source: Path) -> Path:
    """Extract a small mono Opus stream so large videos do not waste bandwidth."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MediaProcessingConfigurationError("ffmpeg is required for video")

    output = source.with_name(f"{source.stem}.audio.ogg")
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-y",
        str(output),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:300]
        suffix = f": {detail}" if detail else ""
        raise MediaConversionError(f"Video has no usable audio track{suffix}")
    return output


async def split_audio_for_transcription(source: Path) -> list[Path]:
    """Normalize and split long media into bounded Nova-3 batch requests."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MediaProcessingConfigurationError(
            "ffmpeg is required for long audio and video"
        )

    pattern = source.with_name("chunk-%04d.ogg")
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-f",
        "segment",
        "-segment_time",
        str(settings.media_transcription_chunk_seconds),
        "-reset_timestamps",
        "1",
        "-y",
        str(pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    chunks = sorted(source.parent.glob("chunk-*.ogg"))
    if (
        process.returncode != 0
        or not chunks
        or any(chunk.stat().st_size == 0 for chunk in chunks)
    ):
        detail = stderr.decode("utf-8", errors="replace").strip()[:300]
        suffix = f": {detail}" if detail else ""
        raise MediaConversionError(f"Could not split media audio{suffix}")
    return chunks


async def process_local_media(
    path: Path,
    media_type: str,
    mime_type: str | None,
    duration_seconds: int | None = None,
    on_extracted: Callable[[], Awaitable[None]] | None = None,
    transcription_requested: bool = False,
) -> ProcessedMediaContent:
    """Produce full searchable content plus one concise summary."""
    if _is_download_only_media(path, media_type, mime_type):
        media_label = mime_type or path.suffix.lower().lstrip(".") or "binary file"
        return ProcessedMediaContent(
            content_text=None,
            content_summary=f"Download-only Telegram attachment ({media_label}).",
            content_model="download-only",
            summary_model="none",
            transcribed=False,
            download_only=True,
        )

    if media_type in IMAGE_MEDIA_TYPES:
        analysis = await analyze_image(path, mime_type)
        if on_extracted is not None:
            await on_extracted()
        return ProcessedMediaContent(
            content_text=analysis.visible_text or None,
            content_summary=analysis.summary,
            content_model=settings.media_summary_model,
            summary_model=settings.media_summary_model,
            transcribed=False,
        )

    wants_transcription = media_type in MEDIA_TRANSCRIPTION_TYPES or (
        transcription_requested and media_type in ALL_TRANSCRIPTION_TYPES
    )
    if wants_transcription:
        visual_timeline = (
            await analyze_video_timeline(path)
            if media_type in {"video", "video_note"}
            else ""
        )
        if (
            duration_seconds is not None
            and duration_seconds > settings.media_transcription_chunk_seconds
        ):
            try:
                transcription_paths = await split_audio_for_transcription(path)
            except MediaConversionError:
                if not visual_timeline:
                    raise
                transcription_paths = []
            transcription_mime_type = "audio/ogg"
        elif media_type in {"video", "video_note"}:
            try:
                transcription_paths = [await extract_video_audio(path)]
            except MediaConversionError:
                if not visual_timeline:
                    raise
                transcription_paths = []
            transcription_mime_type = "audio/ogg"
        else:
            transcription_paths = [path]
            transcription_mime_type = mime_type
        transcription_results: list[TranscriptionResult] = []
        for chunk_index, transcription_path in enumerate(transcription_paths):
            try:
                transcription_results.append(
                    await _transcribe_resumable(
                        transcription_path,
                        transcription_mime_type,
                        chunk_index,
                    )
                )
            except MediaNoSpeechError:
                if not visual_timeline:
                    raise
        transcript = "\n\n".join(result.text for result in transcription_results)
        segment_offset_ms = settings.media_transcription_chunk_seconds * 1000
        transcript_segments = tuple(
            TranscriptPart(
                start_ms=segment.start_ms + chunk_index * segment_offset_ms,
                end_ms=segment.end_ms + chunk_index * segment_offset_ms,
                text=segment.text,
                speaker=segment.speaker,
                confidence=segment.confidence,
                language=segment.language,
            )
            for chunk_index, result in enumerate(transcription_results)
            for segment in result.segments
        )
        if visual_timeline:
            content_sections = []
            if transcript:
                content_sections.append(f"## Transcript\n\n{transcript}")
            content_sections.append(f"## Visual timeline\n\n{visual_timeline}")
            combined_content = "\n\n".join(content_sections)
        else:
            combined_content = transcript
        if not combined_content:
            raise MediaNoSpeechError("Media contains no speech or analyzable frames")
        if on_extracted is not None:
            await on_extracted()
        return ProcessedMediaContent(
            content_text=combined_content,
            content_summary=await summarize_text(combined_content),
            content_model=(
                f"deepgram:{settings.deepgram_model}+{settings.media_summary_model}:vision"
                if transcript and visual_timeline
                else (
                    f"deepgram:{settings.deepgram_model}"
                    if transcript
                    else f"{settings.media_summary_model}:vision"
                )
            ),
            summary_model=settings.media_summary_model,
            transcribed=bool(transcript),
            segments=transcript_segments,
        )

    if media_type in ALL_TRANSCRIPTION_TYPES:
        # Transcription was not bought for this type and nobody asked for this
        # file; the document extractor cannot read audio or video.
        media_label = mime_type or path.suffix.lower().lstrip(".") or "binary file"
        return ProcessedMediaContent(
            content_text=None,
            content_summary=f"Download-only Telegram attachment ({media_label}).",
            content_model="download-only",
            summary_model="none",
            transcribed=False,
            download_only=True,
        )

    try:
        extracted_text = await extract_document_text(path)
    except MediaUnsupportedFormatError:
        media_label = mime_type or path.suffix.lower().lstrip(".") or "binary file"
        return ProcessedMediaContent(
            content_text=None,
            content_summary=f"Download-only Telegram attachment ({media_label}).",
            content_model="download-only",
            summary_model="none",
            transcribed=False,
            download_only=True,
        )
    if on_extracted is not None:
        await on_extracted()
    return ProcessedMediaContent(
        content_text=extracted_text,
        content_summary=await summarize_text(extracted_text),
        content_model="markitdown:0.1.7",
        summary_model=settings.media_summary_model,
        transcribed=False,
    )


async def summarize_existing_transcript(
    job: ClaimedMediaMessage,
) -> tuple[ProcessedMediaContent, MediaInfo] | None:
    """Reuse legacy transcripts so migration does not buy them a second time."""
    transcript = (job.existing_content_text or "").strip()
    if (
        not transcript
        or job.transcribed_at is None
        or job.media_type not in MEDIA_TRANSCRIPTION_TYPES
    ):
        return None
    return (
        ProcessedMediaContent(
            content_text=transcript,
            content_summary=await summarize_text(transcript),
            content_model=f"deepgram:{settings.deepgram_model}",
            summary_model=settings.media_summary_model,
            transcribed=True,
        ),
        MediaInfo(
            media_type=job.media_type,
            file_name=job.file_name,
            mime_type=job.mime_type,
            file_size=job.file_size,
            duration_seconds=job.duration_seconds,
        ),
    )


async def _download_telegram_media(
    job: ClaimedMediaMessage,
    _directory: Path | None = None,
) -> tuple[Path, MediaInfo]:
    try:
        cached = await _fetch_media_cache(job)
        if cached.path is None or not cached.path.is_file():
            raise MediaDownloadError("Telegram media is not cached yet")
        return (
            cached.path,
            MediaInfo(
                media_type=job.media_type,
                file_name=cached.file_name or job.file_name,
                mime_type=cached.mime_type or job.mime_type,
                file_size=cached.size_bytes or job.file_size,
                duration_seconds=job.duration_seconds,
            ),
        )
    except TelegramSessionUnauthorizedError:
        raise
    except MediaCacheError as exc:
        raise MediaDownloadError(
            f"Telegram media cache failed ({exc.code}: {exc})"
        ) from exc


async def _fetch_media_cache(job: ClaimedMediaMessage):
    selected_client: TelegramClient | None = None

    async def provide_client(user_id: UUID, db: AsyncSession) -> TelegramClient:
        nonlocal selected_client
        selected_client = await _get_media_client(user_id, db)
        return selected_client

    try:
        return await fetch_media_to_cache(
            job.user_id,
            job.id,
            get_media_client=provide_client,
        )
    except (MediaDownloadStalled, TelegramSessionUnauthorizedError, ConnectionError):
        if selected_client is not None:
            await _discard_media_client(job.user_id, selected_client)
        raise


def _error_details(error: Exception) -> tuple[str, str]:
    if isinstance(error, TelegramSessionUnauthorizedError):
        return "telegram_session_unauthorized", "Reconnect Telegram and try again"
    if isinstance(error, MediaProcessingConfigurationError):
        return "configuration_error", str(error)
    if isinstance(error, MediaDownloadError):
        return "download_error", str(error)
    if isinstance(error, MediaConversionError):
        return "conversion_error", str(error)
    if isinstance(error, MediaDocumentExtractionError):
        return "document_extraction_error", str(error)
    if isinstance(error, MediaProviderRequestError):
        return "provider_request_error", str(error)
    if isinstance(error, MediaNoSpeechError):
        return "no_speech", str(error)
    if isinstance(error, MediaSourceDeleted):
        return "source_deleted", "The sender deleted this file in Telegram"
    if isinstance(error, MediaProviderResponseError):
        return "provider_response_error", str(error)
    if isinstance(error, MediaIndexingError):
        return "indexing_error", str(error)
    if isinstance(error, MediaProcessingError):
        return "processing_error", str(error)
    return "unexpected_error", f"Unexpected {type(error).__name__}"


def _is_nothing_to_extract(error: Exception) -> bool:
    """Outcomes that are final and blameless: retrying can never produce text."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (MediaSourceDeleted, MediaNoSpeechError)):
            return True
        current = current.__cause__
    return False


async def _mark_failed(message_id: UUID, error: Exception) -> None:
    code, detail = _error_details(error)
    status = (
        MediaProcessingStatus.SKIPPED
        if _is_nothing_to_extract(error)
        else MediaProcessingStatus.FAILED
    )
    async with get_db_context() as db:
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id == message_id,
                TelegramMessage.id.in_(_active_media_message_ids()),
            )
            .values(
                media_processing_status=status,
                media_processing_error_code=code,
                media_processing_error=detail,
                media_processed_at=datetime.now(UTC),
            )
        )
        media_object = (
            await db.execute(
                select(MediaObject).where(
                    MediaObject.message_id == message_id,
                    MediaObject.message_id.in_(_active_media_message_ids()),
                )
            )
        ).scalar_one_or_none()
        if media_object is not None:
            cause = error.__cause__
            if isinstance(error, MediaSourceDeleted) or isinstance(
                cause, MediaSourceDeleted
            ):
                media_object.status = MediaObjectStatus.SOURCE_DELETED
            elif isinstance(error, MediaDiskFull) or isinstance(cause, MediaDiskFull):
                media_object.status = MediaObjectStatus.DISK_FULL
            elif isinstance(error, MediaNoSpeechError) or isinstance(
                cause, MediaNoSpeechError
            ):
                media_object.status = MediaObjectStatus.NO_SPEECH
            else:
                media_object.status = MediaObjectStatus.FAILED
            media_object.retry_after = None
            media_object.error_code = code
            media_object.error_detail = detail


async def _save_content_checkpoint(
    job: ClaimedMediaMessage,
    info: MediaInfo,
    content: ProcessedMediaContent,
) -> None:
    """Persist expensive extraction before the separately retryable index stage."""
    now = datetime.now(UTC)
    async with get_db_context() as db:
        message = (
            await db.execute(
                select(TelegramMessage).where(
                    TelegramMessage.id == job.id,
                    TelegramMessage.id.in_(_active_media_message_ids()),
                )
            )
        ).scalar_one_or_none()
        if message is None:
            raise MediaProcessingError("Media owner is inactive")
        message.media_type = info.media_type
        message.media_file_name = info.file_name or job.file_name
        message.media_mime_type = info.mime_type or job.mime_type
        message.media_file_size = info.file_size
        message.media_duration_seconds = info.duration_seconds
        message.content_text = content.content_text
        if job.legacy_transcript_in_text:
            message.text = None
        message.content_summary = content.content_summary
        message.content_model = content.content_model
        message.summary_model = content.summary_model
        message.transcribed_at = now if content.transcribed else None

        await db.execute(
            delete(TranscriptSegment).where(TranscriptSegment.message_id == job.id)
        )
        for sequence, segment in enumerate(content.segments):
            db.add(
                TranscriptSegment(
                    message_id=job.id,
                    sequence=sequence,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    speaker=segment.speaker,
                    confidence=segment.confidence,
                    language=segment.language,
                    text=segment.text,
                )
            )

        media_object = (
            await db.execute(
                select(MediaObject).where(MediaObject.message_id == job.id)
            )
        ).scalar_one_or_none()
        if media_object is not None:
            media_object.status = MediaObjectStatus.CACHED
            media_object.stage = MediaStage.INDEX
            media_object.extracted_at = now
            media_object.summarized_at = now


async def _mark_summary_stage(message_id: UUID) -> None:
    async with get_db_context() as db:
        media_object = (
            await db.execute(
                select(MediaObject).where(
                    MediaObject.message_id == message_id,
                    MediaObject.message_id.in_(_active_media_message_ids()),
                )
            )
        ).scalar_one_or_none()
        if media_object is None:
            raise MediaProcessingError("Media owner is inactive")
        media_object.stage = MediaStage.SUMMARY


async def _finalize_media(
    job: ClaimedMediaMessage,
    info: MediaInfo,
    content: ProcessedMediaContent,
    index: PreparedMediaContentIndex,
) -> None:
    now = datetime.now(UTC)
    correspondence = False
    async with get_db_context() as db:
        message = (
            await db.execute(
                select(TelegramMessage)
                .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
                .join(User, User.id == TelegramChat.user_id)
                .where(
                    TelegramMessage.id == job.id,
                    TelegramMessage.media_processing_status
                    == MediaProcessingStatus.PROCESSING,
                    User.is_active.is_(True),
                    TelegramChat.indexing_enabled.is_(True),
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if message is None:
            raise MediaProcessingError("Media claim is no longer active")

        await db.execute(
            delete(MessageContentChunk).where(
                MessageContentChunk.message_id == message.id
            )
        )
        for chunk_index, chunk in enumerate(index.chunks):
            db.add(
                MessageContentChunk(
                    message_id=message.id,
                    chunk_index=chunk_index,
                    text=chunk.text,
                    embedding=chunk.embedding,
                    embedded_at=now,
                )
            )

        message.media_type = info.media_type
        message.media_file_name = info.file_name or job.file_name
        message.media_mime_type = info.mime_type or job.mime_type
        message.media_file_size = info.file_size
        message.media_duration_seconds = info.duration_seconds
        message.content_text = content.content_text
        if job.legacy_transcript_in_text:
            message.text = None
        message.content_summary = content.content_summary
        message.content_model = content.content_model
        message.summary_model = content.summary_model
        message.embedding = index.message_embedding
        message.embedded_at = now
        message.transcribed_at = now if content.transcribed else None
        message.media_processing_status = MediaProcessingStatus.READY
        message.media_processing_error_code = None
        message.media_processing_error = None
        message.media_processed_at = now
        chat = await db.get(TelegramChat, message.chat_id)
        if chat is not None:
            correspondence = await enqueue_media_ready(
                db, chat, message.telegram_message_id, now
            )
        media_object = (
            await db.execute(
                select(MediaObject).where(MediaObject.message_id == job.id)
            )
        ).scalar_one_or_none()
        if media_object is not None:
            media_object.status = (
                MediaObjectStatus.READY_DOWNLOAD_ONLY
                if content.download_only
                else MediaObjectStatus.READY
            )
            media_object.stage = MediaStage.COMPLETE
            media_object.indexed_at = now
            media_object.error_code = None
            media_object.error_detail = None

    if correspondence:
        dispatch_events()


async def _load_active_job(
    db: AsyncSession,
    message_id: UUID,
) -> ClaimedMediaMessage | None:
    row = (
        await db.execute(
            select(
                TelegramMessage.id,
                TelegramChat.user_id,
                TelegramMessage.chat_id,
                TelegramMessage.telegram_message_id,
                TelegramMessage.text,
                TelegramMessage.media_type,
                TelegramMessage.media_file_name,
                TelegramMessage.media_mime_type,
                TelegramMessage.media_file_size,
                TelegramMessage.media_duration_seconds,
                TelegramMessage.content_text,
                TelegramMessage.transcribed_at,
                MediaObject.transcription_requested_at,
            )
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .join(User, User.id == TelegramChat.user_id)
            .outerjoin(MediaObject, MediaObject.message_id == TelegramMessage.id)
            .where(
                TelegramMessage.id == message_id,
                TelegramMessage.has_media.is_(True),
                User.is_active.is_(True),
                TelegramChat.indexing_enabled.is_(True),
            )
        )
    ).one_or_none()
    if row is None:
        return None
    legacy_transcript_in_text = bool(
        row.transcribed_at is not None
        and row.content_text is None
        and row.text
        and row.media_type in MEDIA_TRANSCRIPTION_TYPES
    )
    return ClaimedMediaMessage(
        id=row.id,
        user_id=row.user_id,
        chat_id=row.chat_id,
        transcription_requested=row.transcription_requested_at is not None,
        telegram_message_id=row.telegram_message_id,
        caption=None if legacy_transcript_in_text else row.text,
        media_type=row.media_type or "other",
        file_name=row.media_file_name,
        mime_type=row.media_mime_type,
        file_size=row.media_file_size,
        duration_seconds=row.media_duration_seconds,
        existing_content_text=(
            row.text if legacy_transcript_in_text else row.content_text
        ),
        transcribed_at=row.transcribed_at,
        legacy_transcript_in_text=legacy_transcript_in_text,
    )


async def fetch_media_stage(message_id: UUID) -> str:
    """Fetch/cache one owner-scoped Telegram object without processing it."""
    async with get_db_context() as db:
        job = await _load_active_job(db, message_id)
    if job is None:
        return "inactive_or_missing"
    cached = await _fetch_media_cache(job)
    if cached.path is not None and cached.path.is_file():
        return "cached"
    if cached.status in {
        MediaObjectStatus.FETCHING.value,
        MediaObjectStatus.RETRY_WAIT.value,
    }:
        raise MediaDownloadError(f"Telegram media cache is not ready ({cached.status})")
    raise MediaDownloadError("Telegram media cache returned no file")


async def extract_media_stage(message_id: UUID) -> str:
    """Extract/transcribe/summarize and persist an index-ready checkpoint."""
    async with get_db_context() as db:
        checkpoint = (
            await db.execute(
                select(MediaObject.id)
                .join(TelegramMessage, TelegramMessage.id == MediaObject.message_id)
                .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
                .join(User, User.id == TelegramChat.user_id)
                .where(
                    MediaObject.message_id == message_id,
                    MediaObject.stage == MediaStage.INDEX,
                    TelegramMessage.content_summary.isnot(None),
                    User.is_active.is_(True),
                    TelegramChat.indexing_enabled.is_(True),
                )
            )
        ).scalar_one_or_none()
        if checkpoint is not None:
            return "checkpointed"
        job = await claim_media_message(db, message_id)
    if job is None:
        return "skipped"

    reused = await summarize_existing_transcript(job)
    if reused is not None:
        content, info = reused
    else:
        source, info = await _download_telegram_media(job)
        content = await process_local_media(
            source,
            info.media_type,
            info.mime_type,
            info.duration_seconds,
            on_extracted=lambda: _mark_summary_stage(job.id),
            transcription_requested=job.transcription_requested,
        )
    await _save_content_checkpoint(job, info, content)
    return "checkpointed"


async def index_media_stage(message_id: UUID) -> str:
    """Claim and build search vectors from an already persisted checkpoint."""
    async with get_db_context() as db:
        ready = (
            await db.execute(
                select(TelegramMessage.media_processing_status).where(
                    TelegramMessage.id == message_id
                )
            )
        ).scalar_one_or_none()
        if ready == MediaProcessingStatus.READY:
            return "ready"
        claimed = (
            await db.execute(
                update(MediaObject)
                .where(
                    MediaObject.message_id == message_id,
                    MediaObject.message_id.in_(_active_media_message_ids()),
                    MediaObject.stage == MediaStage.INDEX,
                    MediaObject.status.in_(
                        (
                            MediaObjectStatus.CACHED,
                            MediaObjectStatus.RETRY_WAIT,
                            MediaObjectStatus.FAILED,
                        )
                    ),
                )
                .values(
                    status=MediaObjectStatus.INDEXING,
                    retry_after=None,
                    error_code=None,
                    error_detail=None,
                )
                .returning(MediaObject.id)
            )
        ).scalar_one_or_none()
        if claimed is None:
            return "skipped"
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id == message_id,
                TelegramMessage.id.in_(_active_media_message_ids()),
            )
            .values(
                media_processing_status=MediaProcessingStatus.PROCESSING,
                media_processing_started_at=datetime.now(UTC),
                media_processing_error_code=None,
                media_processing_error=None,
            )
        )
        job = await _load_active_job(db, message_id)
        message = await db.get(TelegramMessage, message_id)
        segments = tuple(
            TranscriptPart(
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                text=row.text,
                speaker=row.speaker,
                confidence=row.confidence,
                language=row.language,
            )
            for row in (
                (
                    await db.execute(
                        select(TranscriptSegment)
                        .where(TranscriptSegment.message_id == message_id)
                        .order_by(TranscriptSegment.sequence)
                    )
                )
                .scalars()
                .all()
            )
        )
    if job is None or message is None or not message.content_summary:
        raise MediaProcessingError("Media content checkpoint is missing")
    content = _content_from_checkpoint(message, segments)
    info = MediaInfo(
        media_type=message.media_type or job.media_type,
        file_name=message.media_file_name,
        mime_type=message.media_mime_type,
        file_size=message.media_file_size,
        duration_seconds=message.media_duration_seconds,
    )
    try:
        index = await prepare_media_content_index(
            caption=job.caption,
            summary=content.content_summary,
            content_text=content.content_text,
        )
    except Exception as exc:
        raise MediaIndexingError(
            f"Search indexing failed ({type(exc).__name__})"
        ) from exc
    await _finalize_media(job, info, content, index)
    return "ready"


async def mark_media_retry(
    message_id: UUID,
    error: Exception,
    retry_after_seconds: int,
) -> None:
    code, detail = _error_details(error)
    retry_at = datetime.now(UTC) + timedelta(seconds=retry_after_seconds)
    async with get_db_context() as db:
        await db.execute(
            update(TelegramMessage)
            .where(
                TelegramMessage.id == message_id,
                TelegramMessage.id.in_(_active_media_message_ids()),
            )
            .values(
                media_processing_status=MediaProcessingStatus.QUEUED,
                media_processing_error_code=code,
                media_processing_error=detail,
                # When the retry is due. Until then the row holds no dispatch
                # slot, and the dispatcher recovers it if the retry never runs.
                media_processing_started_at=retry_at,
            )
        )
        obj = (
            await db.execute(
                select(MediaObject).where(
                    MediaObject.message_id == message_id,
                    MediaObject.message_id.in_(_active_media_message_ids()),
                )
            )
        ).scalar_one_or_none()
        if obj is not None:
            obj.status = MediaObjectStatus.RETRY_WAIT
            obj.retry_count += 1
            obj.retry_after = retry_at
            obj.error_code = code
            obj.error_detail = detail


async def process_media_message(message_id: UUID) -> str:
    """Run one idempotent background media job without automatic retries."""
    async with get_db_context() as db:
        job = await claim_media_message(db, message_id)
    if job is None:
        return "skipped"

    try:
        reused = await summarize_existing_transcript(job)
        if reused is not None:
            content, info = reused
        else:
            source, info = await _download_telegram_media(job)
            content = await process_local_media(
                source,
                info.media_type,
                info.mime_type,
                info.duration_seconds,
                transcription_requested=job.transcription_requested,
            )
        await _save_content_checkpoint(job, info, content)
        try:
            index = await prepare_media_content_index(
                caption=job.caption,
                summary=content.content_summary,
                content_text=content.content_text,
            )
        except Exception as exc:
            raise MediaIndexingError(
                f"Search indexing failed ({type(exc).__name__})"
            ) from exc
        await _finalize_media(job, info, content, index)
        return "ready"
    except Exception as exc:
        await _mark_failed(job.id, exc)
        code, _ = _error_details(exc)
        logger.error(
            "Media processing failed for message %s with code %s",
            job.id,
            code,
        )
        return "failed"
