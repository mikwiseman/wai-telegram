"""One owner-scoped Telegram data tool registry for every agent surface."""

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.cursor import CursorError
from app.models.api_key import ApiKey
from app.models.chat import ChatType, TelegramChat
from app.models.media import MediaObject, MediaObjectStatus, TranscriptSegment
from app.models.message import MediaProcessingStatus, TelegramMessage
from app.models.session import TelegramSession
from app.models.user import User
from app.schemas.search import SearchRequest
from app.services.media_cache_service import (
    get_or_create_media_object,
    media_preparation_needs_enqueue,
    request_transcription,
)
from app.services.media_access import MEDIA_DOWNLOAD_TOKEN_TTL
from app.services.messaging_service import clear_draft as clear_telegram_draft
from app.services.messaging_service import list_drafts as list_telegram_drafts
from app.services.messaging_service import save_draft as save_telegram_draft
from app.services.file_browse_service import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MEDIA_TYPES,
    FILE_MEDIA_TYPES,
    MAX_CONTEXT_WINDOW,
    MAX_IN_FLIGHT_MESSAGES,
    MAX_LOCATORS,
    MAX_PREPARE_PER_CALL,
    MIN_FREE_BYTES,
    build_file_entry,
    decode_file_cursor,
    encode_file_cursor,
    list_files,
    list_files_by_locators,
    normalize_extensions,
    normalize_media_types,
)
from app.services.file_search_service import find_files
from app.services.inbox_service import get_inbox
from app.services.search_service import semantic_search
from app.services.telegram_links import (
    build_media_download_url,
    build_telegram_message_url,
)

settings = get_settings()


class ToolInputError(ValueError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def responses_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


_MESSAGE_LOCATOR = {
    "chat_id": {"type": "string", "format": "uuid"},
    "telegram_message_id": {"type": "integer", "minimum": 1},
}

TOOL_DEFINITIONS = (
    ToolDefinition(
        "get_inbox",
        "Read recent conversation tails in one fast batch for unanswered questions, "
        "open promises, waiting-for replies, or context across several chats. "
        "Returns original messages in chronological order, sender direction, reply/thread IDs, "
        "media previews, source links and per-chat older-message cursors. "
        "Default: 20 most recently active private/group/supergroup chats, 12 messages each, "
        "including already-read chats. Set active_since to bound discovery or chat_ids "
        "to read up to 40 chosen conversations. Channels require explicit chat_types. "
        "This is evidence, not an automatic needs-reply classification. "
        "No Telegram RPCs, sync or mark-read; do not call refresh_chats/get_data_status first. "
        "sync_recommended flags a missing known latest message; an old last_sync_at alone "
        "does not make a quiet chat stale. More history: get_chat_messages(before=next_message_cursor). "
        "Use cursor with the same discovery filters for more chats; this is live pagination, "
        "not a frozen account snapshot. Never describe one page of tails as the entire inbox.",
        {
            "type": "object",
            "properties": {
                "chat_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 40,
                    "items": {"type": "string", "format": "uuid"},
                },
                "chat_types": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": ["private", "group", "supergroup", "channel"],
                    },
                },
                "active_since": {"type": "string", "format": "date-time"},
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 40,
                    "default": 20,
                },
                "messages_per_chat": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 12,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "get_files",
        "List and download the files shared in Telegram chats - documents, "
        "photos, videos, voice notes - with their metadata. Filter by chat_ids, "
        "sender, date_from/date_to, media_types, extensions, file_name and "
        "from_me; results are newest first and cursor-paginated. No "
        'query is needed, so this is how you answer "what was shared in this chat '
        'last week" or "every PDF from that group in March". Add query only when '
        'the file is remembered through the talk around it ("the estimate Andrey '
        'sent in spring"): that reaches captionless photos through neighbouring '
        "messages, ranks by relevance instead of date, and returns no cursor. "
        "Every media_download_url works right away - the file is served from "
        "Telegram when it is not already on disk - and stops working at "
        "download_url_expires_at, though any later call mints a fresh one. The "
        "exception is download_state unavailable, which means Telegram itself no "
        "longer has the original and nothing brings it back. Downloading needs no "
        "preparation. Pass prepare=true only to have text pulled out of up to "
        f"{MAX_PREPARE_PER_CALL} files on this page - a transcript for a recording, "
        "extracted text for a document - which is what get_message_content reads; "
        "call again in a minute to see it land. "
        "Pass files=[{chat_id, telegram_message_id}] to act on "
        "an exact set another tool handed you; that ignores every filter, query, "
        "cursor and order. Use search_messages for conversation text and "
        "get_message_content for what is inside one file.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_LOCATORS,
                    "items": {
                        "type": "object",
                        "properties": _MESSAGE_LOCATOR,
                        "required": ["chat_id", "telegram_message_id"],
                        "additionalProperties": False,
                    },
                },
                "chat_ids": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                },
                "chat_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["private", "group", "supergroup", "channel"],
                    },
                },
                "media_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(FILE_MEDIA_TYPES)},
                    "default": list(DEFAULT_MEDIA_TYPES),
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 16},
                },
                "file_name": {"type": "string", "minLength": 3},
                "sender": {"type": "string", "minLength": 3},
                "from_me": {"type": "boolean"},
                "date_from": {"type": "string", "format": "date-time"},
                "date_to": {"type": "string", "format": "date-time"},
                "max_size_bytes": {"type": "integer", "minimum": 1},
                "context_window": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_CONTEXT_WINDOW,
                    "default": DEFAULT_CONTEXT_WINDOW,
                },
                "order": {
                    "type": "string",
                    "enum": ["newest", "oldest"],
                    "default": "newest",
                },
                "prepare": {"type": "boolean", "default": False},
                "cursor": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "search_messages",
        "Search synced messages, links, filenames, transcripts and extracted documents. Use mode=hybrid for a natural-language description and mode=exact for a known literal phrase. Filter by chat_types to restrict private chats, groups, supergroups or channels. Results are cursor-paginated.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "exact"],
                    "default": "hybrid",
                },
                "chat_ids": {
                    "type": "array",
                    "items": {"type": "string", "format": "uuid"},
                },
                "chat_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["private", "group", "supergroup", "channel"],
                    },
                },
                "date_from": {"type": "string", "format": "date-time"},
                "date_to": {"type": "string", "format": "date-time"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "cursor": {"type": "string"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        "get_message",
        "Return full message metadata, URLs, lifecycle, reply/thread/forward/album and media state.",
        {
            "type": "object",
            "properties": _MESSAGE_LOCATOR,
            "required": ["chat_id", "telegram_message_id"],
        },
    ),
    ToolDefinition(
        "list_drafts",
        "List current Telegram drafts with live text, chat identity and date. Read-only; does not mark messages read.",
        {"type": "object", "properties": {}},
    ),
    ToolDefinition(
        "save_draft",
        "Save or replace a server-synced Telegram text draft in a chat. This never sends a message and replaces any existing draft in that chat.",
        {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "format": "uuid"},
                "text": {"type": "string", "minLength": 1},
            },
            "required": ["chat_id", "text"],
        },
    ),
    ToolDefinition(
        "clear_draft",
        "Clear one Telegram draft by chat ID, verify it is empty, and never send a message.",
        {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "format": "uuid"},
            },
            "required": ["chat_id"],
        },
    ),
    ToolDefinition(
        "prepare_media",
        "Idempotently fetch and process the original Telegram media for one "
        "message, and report progress. Returns status, byte_offset and "
        "retry_after; historical media reaches disk only through this call or "
        "through get_files with prepare=true. For several files at once, call "
        "get_files with prepare=true, or with files=[{chat_id, "
        "telegram_message_id}].",
        {
            "type": "object",
            "properties": _MESSAGE_LOCATOR,
            "required": ["chat_id", "telegram_message_id"],
        },
    ),
    ToolDefinition(
        "download_media",
        "Return filename, MIME, size, SHA-256, a freshly signed download URL and "
        "the Telegram link for one cached original. Call it again whenever a "
        "signed URL has expired. On a cache miss call prepare_media first, or "
        "get_files with prepare=true for a whole page of files.",
        {
            "type": "object",
            "properties": _MESSAGE_LOCATOR,
            "required": ["chat_id", "telegram_message_id"],
        },
    ),
    ToolDefinition(
        "get_message_content",
        "Return summary and a cursor-paginated slice of transcript or extracted text, with an explicit next action.",
        {
            "type": "object",
            "properties": {
                **_MESSAGE_LOCATOR,
                "cursor": {"type": "integer", "minimum": 0},
                "limit_chars": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                },
            },
            "required": ["chat_id", "telegram_message_id"],
        },
    ),
    ToolDefinition(
        "get_transcript_segments",
        "Return timestamped transcript segments with speaker, confidence and language using sequence cursor pagination.",
        {
            "type": "object",
            "properties": {
                **_MESSAGE_LOCATOR,
                "cursor": {"type": "integer", "minimum": 0},
                "start_ms": {"type": "integer", "minimum": 0},
                "end_ms": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["chat_id", "telegram_message_id"],
        },
    ),
    ToolDefinition(
        "get_data_status",
        "Diagnostic account-wide counts, queue depths, cache usage and auth state. "
        "Can be expensive on large archives; not a prerequisite for reading or searching. "
        "Use get_inbox for conversation evidence and missing-latest-message checks.",
        {"type": "object", "properties": {}},
    ),
)

_DEFINITION_BY_NAME = {definition.name: definition for definition in TOOL_DEFINITIONS}
# Reading is free. A draft appears in Telegram on the other side, so it needs its
# own permission — narrower than 'write', which also buys sending. Fetching and
# processing media only fills our own cache, like sync, and stays read-level.
TOOL_SCOPES = {"save_draft": "draft", "clear_draft": "draft"}


def required_scope(name: str) -> str | None:
    return TOOL_SCOPES.get(name)


def responses_tool_definitions(names: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        definition.responses_schema()
        for definition in TOOL_DEFINITIONS
        if names is None or definition.name in names
    ]


def _required_uuid(arguments: dict[str, Any], name: str) -> UUID:
    try:
        return UUID(str(arguments[name]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be a UUID") from exc


def _required_int(arguments: dict[str, Any], name: str) -> int:
    try:
        value = int(arguments[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ToolInputError(f"{name} must be positive")
    return value


def _required_text(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} must be a non-empty string")
    return value


async def _message_row(
    db: AsyncSession,
    user_id: UUID,
    arguments: dict[str, Any],
) -> tuple[TelegramMessage, TelegramChat, MediaObject | None]:
    chat_id = _required_uuid(arguments, "chat_id")
    telegram_message_id = _required_int(arguments, "telegram_message_id")
    row = (
        await db.execute(
            select(TelegramMessage, TelegramChat, MediaObject)
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .outerjoin(MediaObject, MediaObject.message_id == TelegramMessage.id)
            .where(
                TelegramChat.user_id == user_id,
                TelegramMessage.chat_id == chat_id,
                TelegramMessage.telegram_message_id == telegram_message_id,
            )
        )
    ).one_or_none()
    if row is None:
        raise ToolInputError("Message not found")
    return row


def _telegram_url(message: TelegramMessage, chat: TelegramChat) -> str | None:
    return build_telegram_message_url(
        chat_type=chat.chat_type,
        telegram_chat_id=chat.telegram_chat_id,
        username=chat.username,
        message_id=message.telegram_message_id,
    )


def _download_url(
    user_id: UUID,
    message: TelegramMessage,
    media_object: MediaObject | None,
) -> str | None:
    if not media_object or not media_object.relative_path or not media_object.sha256:
        return None
    return build_media_download_url(
        base_path=(
            f"/api/v1/chats/{message.chat_id}/messages/"
            f"{message.telegram_message_id}/media"
        ),
        user_id=user_id,
        chat_id=message.chat_id,
        telegram_message_id=message.telegram_message_id,
    )


def _free_media_bytes() -> int | None:
    """Free space on the media volume, measured through the nearest real parent.

    The media root does not exist before the first fetch, and "the directory is
    missing" is not a reason to stop checking whether the disk has room.
    """
    for candidate in (settings.media_root, *settings.media_root.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free
    return None


def _optional_datetime(
    arguments: dict[str, Any], name: str, *, end_of_day: bool = False
) -> datetime | None:
    """Parse a date bound into a real timestamp the database can compare.

    A bare date in date_to means the whole of that day. Reading it as midnight
    drops everything sent on the day the caller explicitly asked for, which is
    the kind of empty result an agent reports as "there are no files".
    """
    value = arguments.get(name)
    if value is None or isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} must be an ISO 8601 date or date-time")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolInputError(
            f"{name} must be an ISO 8601 date or date-time, got {value!r}"
        ) from exc
    if end_of_day and len(text) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _optional_bounded_int(
    arguments: dict[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ToolInputError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ToolInputError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ToolInputError(f"{name} must be at most {maximum}")
    return parsed


def _optional_bool(arguments: dict[str, Any], name: str) -> bool | None:
    value = arguments.get(name)
    if value is None or isinstance(value, bool):
        return value
    raise ToolInputError(f"{name} must be true or false")


def _optional_text(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} must be a non-empty string")
    return value.strip()


def _download_url_expiry() -> str:
    """When the URL just minted stops working, so a caller can plan around it."""
    return (datetime.now(UTC) + MEDIA_DOWNLOAD_TOKEN_TTL).isoformat()


async def _search_messages(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ToolInputError("query is required")
    request = SearchRequest(
        query=query,
        mode=arguments.get("mode", "hybrid"),
        chat_ids=arguments.get("chat_ids"),
        chat_types=arguments.get("chat_types"),
        date_from=arguments.get("date_from"),
        date_to=arguments.get("date_to"),
        limit=arguments.get("limit", 20),
        cursor=arguments.get("cursor"),
    )
    response = await semantic_search(db, user_id, request)
    return response.model_dump(mode="json")


_FILTER_KEYS = (
    "chat_ids",
    "chat_types",
    "media_types",
    "extensions",
    "file_name",
    "sender",
    "from_me",
    "date_from",
    "date_to",
    "max_size_bytes",
    "order",
    "context_window",
)


def _locator_pairs(values: Any) -> list[tuple[UUID, int]]:
    if not isinstance(values, list) or not values:
        raise ToolInputError("files must be a non-empty array of message locators")
    if len(values) > MAX_LOCATORS:
        raise ToolInputError(f"files accepts at most {MAX_LOCATORS} message locators")
    pairs: list[tuple[UUID, int]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ToolInputError(
                "each files entry must be {chat_id, telegram_message_id}"
            )
        pair = (
            _required_uuid(value, "chat_id"),
            _required_int(value, "telegram_message_id"),
        )
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _download_state_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "ready": 0,
        "fetching": 0,
        "queued": 0,
        "not_prepared": 0,
        "unavailable": 0,
    }
    for entry in entries:
        counts[entry["download_state"]] += 1
    return counts


async def _stage_files(
    db: AsyncSession,
    user_id: UUID,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Start extracting text for everything on this page that has none yet.

    Downloading no longer waits on any of this - the endpoint streams straight
    from Telegram - so preparing is about what get_message_content will find:
    a transcript for a recording, extracted text for a document. A file whose
    original Telegram deleted is skipped, because nothing can be read from it.
    """
    counts = _download_state_counts(entries)
    pending = [
        entry["media_file_size"] or 0
        for entry in entries
        if entry["media_processing_status"] != "ready"
    ]
    pending_bytes = sum(pending)
    largest_pending = max(pending, default=0)
    report: dict[str, Any] = {
        "requested": True,
        "enqueued": 0,
        "ready_now": sum(1 for e in entries if e["media_processing_status"] == "ready"),
        "already_in_progress": sum(
            1
            for e in entries
            if e["media_processing_status"] in {"pending", "queued", "processing"}
        ),
        "skipped_over_cap": 0,
        "unavailable": counts["unavailable"],
        "pending_bytes": pending_bytes,
        "largest_pending_bytes": largest_pending,
        "error_code": None,
        "error_detail": None,
    }
    candidates = [
        entry
        for entry in entries
        if entry["download_state"] != "unavailable"
        and entry["media_type"] != "other"
        and entry["media_processing_status"]
        not in {"ready", "pending", "queued", "processing"}
    ]
    report["skipped_over_cap"] = max(0, len(candidates) - MAX_PREPARE_PER_CALL)
    batch = candidates[:MAX_PREPARE_PER_CALL]
    if not batch:
        return report

    if not settings.media_pipeline_enabled:
        report["error_code"] = "media_pipeline_deferred"
        report["error_detail"] = (
            "Durable media processing is deferred until storage is attached"
        )
        return report

    # Only dispatch claims count. PENDING is the standing backlog the dispatcher
    # meters down to media_dispatch_target_depth on its own, and it runs to
    # hundreds of thousands of historical files - counting it would mean this
    # ceiling is always exceeded and prepare never starts anything.
    in_flight = (
        await db.execute(
            select(func.count())
            .select_from(TelegramMessage)
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .where(
                TelegramChat.user_id == user_id,
                TelegramMessage.media_processing_status.in_(
                    [
                        MediaProcessingStatus.QUEUED,
                        MediaProcessingStatus.PROCESSING,
                    ]
                ),
            )
        )
    ).scalar_one()
    if in_flight >= MAX_IN_FLIGHT_MESSAGES:
        report["error_code"] = "too_many_in_flight"
        report["error_detail"] = f"{in_flight} files are already being fetched"
        return report

    # The worker turns a full volume into a terminal DISK_FULL per file, so a
    # batch that would fill it is refused whole rather than half-applied.
    needed = sum(entry["media_file_size"] or 0 for entry in batch)
    free = _free_media_bytes()
    if free is not None and free - needed < MIN_FREE_BYTES:
        report["error_code"] = "insufficient_disk"
        report["error_detail"] = f"{needed} bytes needed, {free} bytes free"
        return report

    rows = (
        await db.execute(
            select(TelegramMessage, MediaObject)
            .outerjoin(MediaObject, MediaObject.message_id == TelegramMessage.id)
            .where(TelegramMessage.id.in_([UUID(e["message_id"]) for e in batch]))
        )
    ).all()
    started: list[UUID] = []
    for message, media_object in rows:
        if media_object is None:
            media_object = await get_or_create_media_object(db, user_id, message.id)
        request_transcription(message, media_object)
        if media_object.status == MediaObjectStatus.SOURCE_DELETED:
            continue
        if not media_preparation_needs_enqueue(message, media_object):
            continue
        message.media_processing_status = MediaProcessingStatus.PENDING
        message.media_processing_error_code = None
        message.media_processing_error = None
        media_object.status = MediaObjectStatus.PENDING
        media_object.error_code = None
        media_object.error_detail = None
        started.append(message.id)
    await db.commit()
    if started:
        from app.tasks.media_tasks import enqueue_media_processing

        enqueue_media_processing(started)

    staged = {str(message_id) for message_id in started}
    for entry in entries:
        if entry["message_id"] in staged:
            # The file was already downloadable; what changed is that its text
            # is now on the way.
            entry["media_processing_status"] = str(MediaProcessingStatus.PENDING)
            entry["media_cache_status"] = str(MediaObjectStatus.PENDING)
    report["enqueued"] = len(started)
    return report


def _files_next_action(result: dict[str, Any]) -> str:
    """One sentence for the state of these files, one for the rest of the set.

    Staging and paging are different axes: a page that still needs fetching also
    needs paging, and dropping either sentence sends the caller away with half
    the story.
    """
    prepare = result["prepare"]
    counts = result["counts"]
    error_code = prepare["error_code"]
    parts: list[str] = []

    if error_code == "media_pipeline_deferred":
        parts.append("Durable media processing is deferred until storage is attached.")
    elif error_code == "insufficient_disk":
        parts.append(
            f"Not enough free disk to fetch these files ({prepare['error_detail']}). "
            "Narrow the filters or free space."
        )
    elif error_code == "too_many_in_flight":
        parts.append(
            f"{prepare['error_detail']}. Call get_files again with the same "
            "arguments in about 60 seconds before starting more."
        )
    elif prepare["skipped_over_cap"]:
        parts.append(
            f"{prepare['enqueued']} files started ({MAX_PREPARE_PER_CALL} per call "
            f"is the maximum, {prepare['skipped_over_cap']} still waiting). Call "
            "get_files again with the same arguments to collect them and start "
            "the rest."
        )
    elif prepare["enqueued"]:
        noun = "file is" if prepare["enqueued"] == 1 else "files are"
        parts.append(
            f"Every link below works now. {prepare['enqueued']} {noun} also having "
            "text extracted; call get_files again in about 60 seconds if you need "
            "get_message_content to have something to read."
        )
    elif not result["files"]:
        parts.append(
            "No files matched. Widen the date range, drop media_types, or add a "
            "query to reach files remembered only by the talk around them."
        )
    elif counts["ready"]:
        parts.append(
            "Every link below works now; they expire in 60 minutes, and calling "
            "get_files again mints fresh ones."
        )

    if result["has_more"]:
        parts.append(
            f'Call get_files again with cursor="{result["next_cursor"]}" for the '
            "next page."
        )
    elif result["truncated"]:
        parts.append(
            f"{result['matched_total']} files matched and {result['total']} were "
            "returned. Raise limit (max 100) or narrow the filters - relevance "
            "mode has no cursor."
        )
    return " ".join(parts) or "Nothing further is needed for these files."


def _uuid_list(arguments: dict[str, Any], name: str) -> list[UUID] | None:
    values = arguments.get(name)
    if not values:
        return None
    if not isinstance(values, list):
        raise ToolInputError(f"{name} must be an array of UUID strings")
    try:
        return [UUID(str(value)) for value in values]
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be an array of UUID strings") from exc


def _chat_type_list(arguments: dict[str, Any], name: str) -> list[str] | None:
    values = arguments.get(name)
    if not values:
        return None
    if not isinstance(values, list):
        raise ToolInputError(f"{name} must be an array of chat types")
    allowed = [chat_type.value for chat_type in ChatType]
    unknown = sorted({str(value) for value in values} - set(allowed))
    if unknown:
        raise ToolInputError(
            f"{name} contains an unknown value: "
            + ", ".join(unknown)
            + ". Allowed: "
            + ", ".join(allowed)
        )
    return [str(value) for value in values]


def _echo_filter(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_echo_filter(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    return value


async def _get_files(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    locators = arguments.get("files")
    query = str(arguments.get("query") or "").strip()
    cursor = arguments.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ToolInputError("cursor must be the next_cursor string from a prior page")
    order = str(arguments.get("order") or "newest")
    limit = min(100, _optional_bounded_int(arguments, "limit", maximum=100) or 20)

    if locators is not None and (
        query or cursor or any(key in arguments for key in _FILTER_KEYS)
    ):
        raise ToolInputError(
            "files cannot be combined with query, cursor, order or any filter"
        )
    if query and cursor:
        raise ToolInputError(
            "query mode ranks by relevance and has no cursor; drop the cursor or "
            "the query"
        )
    if order not in {"newest", "oldest"}:
        raise ToolInputError('order must be "newest" or "oldest"')

    try:
        media_types = normalize_media_types(arguments.get("media_types"))
        filters = {
            "chat_ids": _uuid_list(arguments, "chat_ids"),
            "chat_types": _chat_type_list(arguments, "chat_types"),
            "date_from": _optional_datetime(arguments, "date_from"),
            "date_to": _optional_datetime(arguments, "date_to", end_of_day=True),
            "extensions": normalize_extensions(arguments.get("extensions")),
            "file_name": _optional_text(arguments, "file_name"),
            "sender": _optional_text(arguments, "sender"),
            "from_me": _optional_bool(arguments, "from_me"),
            "max_size_bytes": _optional_bounded_int(arguments, "max_size_bytes"),
        }
        context_window = (
            _optional_bounded_int(
                arguments, "context_window", minimum=0, maximum=MAX_CONTEXT_WINDOW
            )
            if arguments.get("context_window") is not None
            else DEFAULT_CONTEXT_WINDOW
        )
        cursor_values = decode_file_cursor(cursor, order) if cursor else None
    except (CursorError, ValueError) as exc:
        raise ToolInputError(str(exc)) from exc

    not_found: list[dict[str, Any]] = []
    searched_messages = 0
    matched_total = 0
    has_more = False
    next_cursor: str | None = None

    if locators is not None:
        mode = "locators"
        rows, not_found = await list_files_by_locators(
            db, user_id, _locator_pairs(locators)
        )
        entries = [build_file_entry(user_id, row) for row in rows]
    elif query:
        mode = "query"
        entries, searched_messages, matched_total = await find_files(
            db,
            user_id,
            query=query,
            media_types=media_types,
            context_window=context_window,
            limit=limit,
            **filters,
        )
    else:
        mode = "browse"
        rows, has_more = await list_files(
            db,
            user_id,
            media_types=media_types,
            order=order,
            cursor_values=cursor_values,
            limit=limit,
            **filters,
        )
        entries = [build_file_entry(user_id, row) for row in rows]
        if has_more and rows:
            next_cursor = encode_file_cursor(rows[-1], order)

    counts = {
        "ready": 0,
        "fetching": 0,
        "queued": 0,
        "not_prepared": 0,
        "unavailable": 0,
    }
    prepare: dict[str, Any] = {
        "requested": False,
        "enqueued": 0,
        "ready_now": 0,
        "already_in_progress": 0,
        "skipped_over_cap": 0,
        "unavailable": 0,
        "pending_bytes": 0,
        "largest_pending_bytes": 0,
        "error_code": None,
        "error_detail": None,
    }
    if arguments.get("prepare"):
        prepare = await _stage_files(db, user_id, entries)
    for entry in entries:
        counts[entry["download_state"]] += 1
    if not arguments.get("prepare"):
        prepare["ready_now"] = counts["ready"]
        prepare["already_in_progress"] = counts["fetching"] + counts["queued"]
        prepare["unavailable"] = counts["unavailable"]
        idle_pending = [
            entry["media_file_size"] or 0
            for entry in entries
            if entry["download_state"] != "ready"
        ]
        prepare["pending_bytes"] = sum(idle_pending)
        prepare["largest_pending_bytes"] = max(idle_pending, default=0)

    result = {
        "files": entries,
        "mode": mode,
        "query": query or None,
        "total": len(entries),
        "matched_total": matched_total if mode == "query" else len(entries),
        "counts": counts,
        # Relevance order cannot be resumed from a keyset, so query mode says it
        # truncated instead of offering a cursor it would only loop on.
        "has_more": has_more,
        "next_cursor": next_cursor,
        "truncated": mode == "query" and matched_total > len(entries),
        "searched_messages": searched_messages,
        "not_found": not_found,
        "filters_applied": {
            "media_types": media_types,
            "order": order if mode == "browse" else None,
            "limit": limit,
            **{
                key: _echo_filter(filters[key])
                for key in (
                    "chat_ids",
                    "chat_types",
                    "date_from",
                    "date_to",
                    "extensions",
                    "file_name",
                    "sender",
                    "from_me",
                    "max_size_bytes",
                )
            },
        },
        "prepare": prepare,
    }
    result["next_action"] = _files_next_action(result)
    return result


async def _get_message(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    message, chat, media_object = await _message_row(db, user_id, arguments)
    return {
        "id": str(message.id),
        "chat_id": str(chat.id),
        "chat_title": chat.title,
        "telegram_message_id": message.telegram_message_id,
        "text": message.text,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "is_outgoing": message.is_outgoing,
        "sent_at": message.sent_at.isoformat(),
        "edited_at": message.edited_at.isoformat() if message.edited_at else None,
        "deleted_at": message.deleted_at.isoformat() if message.deleted_at else None,
        "entities": message.entities or [],
        "visible_urls": message.visible_urls or [],
        "hidden_urls": message.hidden_urls or [],
        "buttons": message.buttons or [],
        "webpage_preview": message.webpage_preview,
        "reply_to_message_id": message.reply_to_message_id,
        "thread_id": message.thread_id,
        "forward_origin": message.forward_origin,
        "album_id": message.album_id,
        "reactions": message.reactions or [],
        "poll": message.poll,
        "contact": message.contact,
        "location": message.location,
        "service_event": message.service_event,
        "has_media": message.has_media,
        "media_type": message.media_type,
        "media_file_name": message.media_file_name,
        "media_mime_type": message.media_mime_type,
        "media_file_size": message.media_file_size,
        "media_processing_status": str(message.media_processing_status)
        if message.media_processing_status
        else None,
        "media_cache_status": str(media_object.status) if media_object else None,
        "media_cache_stage": str(media_object.stage) if media_object else None,
        "media_cached_bytes": media_object.byte_offset if media_object else 0,
        "media_sha256": media_object.sha256 if media_object else None,
        "telegram_message_url": _telegram_url(message, chat),
        "media_download_url": _download_url(user_id, message, media_object),
        "download_url_expires_at": (
            _download_url_expiry()
            if media_object and media_object.relative_path and media_object.sha256
            else None
        ),
    }


async def _save_draft(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    chat_id = _required_uuid(arguments, "chat_id")
    text = _required_text(arguments, "text")
    try:
        return await save_telegram_draft(db, user_id, chat_id, text)
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc


async def _list_drafts(
    db: AsyncSession, user_id: UUID, _arguments: dict[str, Any]
) -> dict[str, Any]:
    return await list_telegram_drafts(db, user_id)


async def _clear_draft(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    chat_id = _required_uuid(arguments, "chat_id")
    try:
        return await clear_telegram_draft(db, user_id, chat_id)
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc


async def _prepare_media(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    message, chat, media_object = await _message_row(db, user_id, arguments)
    if not message.has_media:
        raise ToolInputError("Message has no media")
    if not settings.media_pipeline_enabled:
        return {
            "message_id": str(message.id),
            "status": "unavailable",
            "stage": "deferred",
            "enqueued": False,
            "error_code": "media_pipeline_deferred",
            "error_detail": (
                "Durable media processing is deferred until storage is attached"
            ),
            "retry_after": None,
            "media_download_url": None,
            "telegram_message_url": _telegram_url(message, chat),
            "next_action": "Retry after the durable media pipeline is enabled",
        }
    if media_object is None:
        media_object = await get_or_create_media_object(db, user_id, message.id)
    request_transcription(message, media_object)

    enqueued = False
    # media_preparation_needs_enqueue says yes to a deleted original, so without
    # this a caller can re-queue a permanently gone file on every poll forever.
    gone = media_object.status == MediaObjectStatus.SOURCE_DELETED
    if not gone and media_preparation_needs_enqueue(message, media_object):
        from app.tasks.media_tasks import enqueue_media_processing

        message.media_processing_status = MediaProcessingStatus.PENDING
        message.media_processing_error_code = None
        message.media_processing_error = None
        media_object.status = MediaObjectStatus.PENDING
        media_object.error_code = None
        media_object.error_detail = None
        await db.commit()
        enqueue_media_processing([message.id])
        enqueued = True

    return {
        "message_id": str(message.id),
        "status": str(media_object.status),
        "stage": str(media_object.stage),
        "byte_offset": media_object.byte_offset,
        "size_bytes": media_object.size_bytes,
        "sha256": media_object.sha256,
        "retry_after": media_object.retry_after.isoformat()
        if media_object.retry_after
        else None,
        "error_code": media_object.error_code,
        "error_detail": media_object.error_detail,
        "enqueued": enqueued,
        "download_state": (
            "ready"
            if media_object.relative_path and media_object.sha256
            else "unavailable"
            if gone
            else "queued"
        ),
        "media_download_url": _download_url(user_id, message, media_object),
        "download_url_expires_at": (
            _download_url_expiry()
            if media_object.relative_path and media_object.sha256
            else None
        ),
        "telegram_message_url": _telegram_url(message, chat),
        "next_action": (
            "Call download_media"
            if media_object.relative_path and media_object.sha256
            else "The original was deleted from Telegram. No download is possible."
            if gone
            else "Call prepare_media again after retry_after to refresh progress"
        ),
    }


async def _download_media(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    message, chat, media_object = await _message_row(db, user_id, arguments)
    url = _download_url(user_id, message, media_object)
    if url is None:
        return {
            "status": "cache_miss",
            "telegram_message_id": message.telegram_message_id,
            "telegram_message_url": _telegram_url(message, chat),
            "next_action": (
                "Call prepare_media, or get_files with prepare=true, then retry "
                "download_media"
            ),
        }
    return {
        "status": "ready",
        "telegram_message_id": message.telegram_message_id,
        "media_file_name": media_object.file_name or message.media_file_name,
        "media_mime_type": media_object.mime_type or message.media_mime_type,
        "media_file_size": media_object.size_bytes or message.media_file_size,
        "media_sha256": media_object.sha256,
        "media_download_url": url,
        "download_url_expires_at": _download_url_expiry(),
        "telegram_message_url": _telegram_url(message, chat),
        "next_action": "Download the resource_link; call download_media again to refresh the signed URL",
    }


async def _get_message_content(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    message, chat, media_object = await _message_row(db, user_id, arguments)
    cursor = max(0, int(arguments.get("cursor", 0)))
    limit_chars = min(50_000, max(1_000, int(arguments.get("limit_chars", 20_000))))
    full_text = message.content_text or ""
    content = full_text[cursor : cursor + limit_chars]
    next_cursor = cursor + len(content)
    has_more = next_cursor < len(full_text)
    if message.media_processing_status == MediaProcessingStatus.READY:
        next_action = (
            "Call get_message_content with next_cursor"
            if has_more
            else "Content is complete"
        )
    elif media_object and media_object.status in {
        MediaObjectStatus.FETCHING,
        MediaObjectStatus.EXTRACTING,
        MediaObjectStatus.INDEXING,
        MediaObjectStatus.PROCESSING,
        MediaObjectStatus.RETRY_WAIT,
    }:
        next_action = "Call prepare_media again after retry_after"
    elif message.has_media:
        next_action = "Call prepare_media"
    else:
        next_action = "No media content is available"
    return {
        "message_id": str(message.id),
        "telegram_message_id": message.telegram_message_id,
        "text": message.text,
        "media_type": message.media_type,
        "media_file_name": message.media_file_name,
        "media_mime_type": message.media_mime_type,
        "media_file_size": message.media_file_size,
        "content_summary": message.content_summary,
        "content_text": content,
        "cursor": cursor,
        "has_more": has_more,
        "next_cursor": next_cursor if has_more else None,
        "media_processing_status": str(message.media_processing_status)
        if message.media_processing_status
        else None,
        "media_processing_error_code": message.media_processing_error_code,
        "media_cache_status": str(media_object.status) if media_object else None,
        "telegram_message_url": _telegram_url(message, chat),
        "media_download_url": _download_url(user_id, message, media_object),
        "next_action": next_action,
    }


async def _get_transcript_segments(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    message, _chat, _media_object = await _message_row(db, user_id, arguments)
    cursor = max(0, int(arguments.get("cursor", 0)))
    limit = min(500, max(1, int(arguments.get("limit", 100))))
    query = select(TranscriptSegment).where(
        TranscriptSegment.message_id == message.id,
        TranscriptSegment.sequence >= cursor,
    )
    if arguments.get("start_ms") is not None:
        query = query.where(TranscriptSegment.end_ms >= int(arguments["start_ms"]))
    if arguments.get("end_ms") is not None:
        query = query.where(TranscriptSegment.start_ms <= int(arguments["end_ms"]))
    rows = list(
        (
            await db.execute(
                query.order_by(TranscriptSegment.sequence).limit(limit + 1)
            )
        ).scalars()
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {
        "segments": [
            {
                "sequence": row.sequence,
                "start_ms": row.start_ms,
                "end_ms": row.end_ms,
                "speaker": row.speaker,
                "confidence": row.confidence,
                "language": row.language,
                "text": row.text,
            }
            for row in rows
        ],
        "has_more": has_more,
        "next_cursor": rows[-1].sequence + 1 if has_more and rows else None,
    }


async def _get_inbox(
    db: AsyncSession, user_id: UUID, arguments: dict[str, Any]
) -> dict[str, Any]:
    allowed = {
        "chat_ids",
        "chat_types",
        "active_since",
        "cursor",
        "limit",
        "messages_per_chat",
    }
    if set(arguments) - allowed:
        raise ToolInputError("Unknown get_inbox arguments")
    for name in ("chat_ids", "chat_types"):
        if name in arguments and not arguments[name]:
            raise ToolInputError(f"{name} must be a non-empty array")
    for name in ("limit", "messages_per_chat"):
        if name in arguments and (
            isinstance(arguments[name], bool) or not isinstance(arguments[name], int)
        ):
            raise ToolInputError(f"{name} must be an integer")
    chat_ids = _uuid_list(arguments, "chat_ids")
    if chat_ids:
        chat_ids = list(dict.fromkeys(chat_ids))
        if len(chat_ids) > 40:
            raise ToolInputError("At most 40 chat_ids per call")
        if any(
            arguments.get(name) is not None
            for name in ("chat_types", "active_since", "cursor")
        ):
            raise ToolInputError("Use chat_ids or discovery filters/cursor, not both")
    limit = _optional_bounded_int(arguments, "limit", maximum=40) or 20
    messages_per_chat = (
        _optional_bounded_int(arguments, "messages_per_chat", maximum=30) or 12
    )
    if (len(chat_ids) if chat_ids else limit) * messages_per_chat > 480:
        raise ToolInputError(
            "At most 480 messages per batch; reduce limit or messages_per_chat"
        )
    try:
        result = await get_inbox(
            db,
            user_id,
            chat_ids=chat_ids,
            chat_types=_chat_type_list(arguments, "chat_types"),
            active_since=_optional_datetime(arguments, "active_since"),
            cursor=_optional_text(arguments, "cursor"),
            limit=limit,
            messages_per_chat=messages_per_chat,
        )
    except (CursorError, ValueError) as exc:
        raise ToolInputError(str(exc)) from exc
    redis_client = aioredis.from_url(
        settings.redis_url, socket_connect_timeout=1, socket_timeout=1
    )
    try:
        result["listener_active"] = bool(
            await redis_client.exists(f"listener:active:{user_id}")
        )
    except (aioredis.RedisError, OSError):
        result["listener_active"] = None
    finally:
        await redis_client.aclose()
    return result


async def _get_data_status(
    db: AsyncSession, user_id: UUID, _arguments: dict[str, Any]
) -> dict[str, Any]:
    chat_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramChat)
            .where(TelegramChat.user_id == user_id)
        )
    ).scalar_one()
    message_count = (
        await db.execute(
            select(func.count())
            .select_from(TelegramMessage)
            .join(TelegramChat)
            .where(TelegramChat.user_id == user_id)
        )
    ).scalar_one()
    freshest_message = (
        await db.execute(
            select(func.max(TelegramMessage.sent_at))
            .join(TelegramChat)
            .where(TelegramChat.user_id == user_id)
        )
    ).scalar_one()
    processing_rows = (
        await db.execute(
            select(TelegramMessage.media_processing_status, func.count())
            .join(TelegramChat)
            .where(TelegramChat.user_id == user_id)
            .group_by(TelegramMessage.media_processing_status)
        )
    ).all()
    cache_bytes, cache_objects = (
        await db.execute(
            select(
                func.coalesce(func.sum(MediaObject.size_bytes), 0),
                func.count(MediaObject.id),
            ).where(MediaObject.user_id == user_id)
        )
    ).one()
    active_users = (
        await db.execute(
            select(func.count()).select_from(User).where(User.is_active.is_(True))
        )
    ).scalar_one()
    active_sessions = (
        await db.execute(
            select(func.count())
            .select_from(TelegramSession)
            .where(
                TelegramSession.user_id == user_id,
                TelegramSession.is_active.is_(True),
            )
        )
    ).scalar_one()
    active_keys = (
        await db.execute(
            select(func.count())
            .select_from(ApiKey)
            .where(
                ApiKey.user_id == user_id,
                ApiKey.is_active.is_(True),
            )
        )
    ).scalar_one()

    redis_client = aioredis.from_url(settings.redis_url)
    try:
        queue_names = ("celery", "media-fetch", "media-process", "media-index")
        queue_depths = {
            name: int(await redis_client.llen(name)) for name in queue_names
        }
        metrics = await redis_client.hgetall("media:metrics")
    finally:
        await redis_client.aclose()
    hits = int(metrics.get(b"hits", metrics.get("hits", 0)))
    misses = int(metrics.get(b"misses", metrics.get("misses", 0)))
    attempts = hits + misses

    volume = None
    if settings.media_root.exists():
        usage = shutil.disk_usage(settings.media_root)
        volume = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round((usage.used / usage.total) * 100, 2),
        }
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "chats": int(chat_count),
        "messages": int(message_count),
        "freshest_message_at": freshest_message.isoformat()
        if freshest_message
        else None,
        "queue_depths": queue_depths,
        "processing": {
            str(status) if status is not None else "uninitialized": int(count)
            for status, count in processing_rows
        },
        "cache": {
            "pipeline_enabled": settings.media_pipeline_enabled,
            "objects": int(cache_objects),
            "bytes": int(cache_bytes),
            "hits": hits,
            "misses": misses,
            "hit_ratio": (hits / attempts) if attempts else None,
            "volume": volume,
            "eviction_enabled": False,
        },
        "auth": {
            "active_users": int(active_users),
            "active_owner_sessions": int(active_sessions),
            "active_owner_api_keys": int(active_keys),
        },
    }


ToolHandler = Callable[[AsyncSession, UUID, dict[str, Any]], Awaitable[dict[str, Any]]]
_HANDLERS: dict[str, ToolHandler] = {
    "get_inbox": _get_inbox,
    "search_messages": _search_messages,
    "get_files": _get_files,
    "get_message": _get_message,
    "list_drafts": _list_drafts,
    "save_draft": _save_draft,
    "clear_draft": _clear_draft,
    "prepare_media": _prepare_media,
    "download_media": _download_media,
    "get_message_content": _get_message_content,
    "get_transcript_segments": _get_transcript_segments,
    "get_data_status": _get_data_status,
}


async def execute_data_tool(
    db: AsyncSession,
    user_id: UUID,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if name not in _DEFINITION_BY_NAME:
        raise ToolInputError(f"Unknown data tool: {name}")
    active_user_id = (
        await db.execute(
            select(User.id).where(
                User.id == user_id,
                User.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()
    if active_user_id is None:
        raise ToolInputError("User is inactive")
    return await _HANDLERS[name](db, user_id, arguments)
