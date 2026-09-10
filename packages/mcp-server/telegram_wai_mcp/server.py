import asyncio
import json
import os
import re
from datetime import UTC, date, datetime, time
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ResourceLink, TextContent, Tool
from starlette.requests import Request

from telegram_wai_mcp.client import TelegramAIClient

# Initialize MCP server
server = Server("telegram-wai-mcp")
MAX_LIMIT = 500
MAX_LOOKBACK_DAYS = 180
_session_api_keys: dict[str, str] = {}

# Media type display labels
MEDIA_LABELS = {
    "photo": "Photo",
    "video": "Video",
    "audio": "Audio",
    "document": "Document",
    "voice": "Voice message",
    "video_note": "Video note",
}


def remember_session_api_key(session_id: str, api_key: str) -> None:
    if session_id and api_key:
        _session_api_keys[session_id] = api_key


def forget_session_api_key(session_id: str) -> None:
    _session_api_keys.pop(session_id, None)


def get_session_api_key(session_id: str) -> str | None:
    return _session_api_keys.get(session_id)


def _current_request() -> Request | None:
    try:
        request = server.request_context.request
    except LookupError:
        return None
    return request if isinstance(request, Request) else None


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _resolve_api_key(request: Request | None) -> str | None:
    if request is not None:
        scope_api_key = request.scope.get("telegram_ai_api_key")
        if isinstance(scope_api_key, str) and scope_api_key:
            return scope_api_key

        bearer = _extract_bearer_token(request.headers.get("authorization"))
        if bearer:
            return bearer

        query_key = request.query_params.get("key", "").strip()
        if query_key:
            return query_key

        session_id = request.headers.get("mcp-session-id", "").strip()
        if session_id:
            return get_session_api_key(session_id)

    env_api_key = os.environ.get("TELEGRAM_AI_KEY", "").strip()
    return env_api_key or None


def get_client() -> TelegramAIClient:
    request = _current_request()
    api_key = _resolve_api_key(request)
    if request is not None and not api_key:
        raise RuntimeError(
            "Missing API key. Use Authorization: Bearer <key> or ?key=... when initializing the MCP session."
        )
    base_url = os.environ.get("TELEGRAM_AI_URL", "http://localhost:8000")
    return TelegramAIClient(base_url=base_url, api_key=api_key)


def _error(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=f"Error: {message}")]


def _tool_error(message: str) -> CallToolResult:
    return CallToolResult(content=_error(message), isError=True)


def _as_dict(arguments: dict[str, Any] | None) -> dict[str, Any]:
    return arguments or {}


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'"{key}" must be a non-empty string')
    return value.strip()


def _optional_int(
    arguments: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = arguments.get(key, default)
    if not isinstance(raw, int):
        raise ValueError(f'"{key}" must be an integer')
    return max(minimum, min(maximum, raw))


def _optional_iso_datetime(
    arguments: dict[str, Any],
    key: str,
    *,
    end_of_day: bool = False,
) -> datetime | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'"{key}" must be an ISO 8601 datetime string')
    try:
        if len(value) == 10:
            parsed_date = date.fromisoformat(value)
            boundary = time.max if end_of_day else time.min
            return datetime.combine(parsed_date, boundary, tzinfo=UTC)
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f'"{key}" must be an ISO 8601 datetime string') from e


def _optional_iso_date(arguments: dict[str, Any], key: str) -> date | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'"{key}" must be a YYYY-MM-DD date string')
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f'"{key}" must be a YYYY-MM-DD date string') from e


def _format_date(value: Any) -> str:
    if isinstance(value, str):
        return value[:10]
    if isinstance(value, datetime):
        return value.date().isoformat()
    return "unknown"


def _format_media_label(msg: dict) -> str:
    """Format a compact, searchable representation of a media message."""
    media_type = msg.get("media_type")
    text = msg.get("text")
    summary = msg.get("content_summary")
    preview = msg.get("content_preview")
    status = msg.get("media_processing_status")

    if not media_type:
        return text or "[Media]"

    label = MEDIA_LABELS.get(media_type, "Media")
    parts: list[str] = []
    if text:
        parts.append(text)
    if summary:
        parts.append(f"Summary: {summary}")
    elif preview:
        parts.append(f"Content: {preview}")

    if status in {"pending", "queued"}:
        state = "processing queued"
    elif status == "processing":
        state = "processing"
    elif status == "failed":
        state = f"processing failed: {msg.get('media_processing_error_code') or 'unknown error'}"
    else:
        state = None

    prefix = f"[{label}{f' — {state}' if state else ''}]"
    return f"{prefix} {' | '.join(parts)}" if parts else prefix


def format_message_content(
    result: dict,
    base_url: str = "https://telegram.waiwai.is",
) -> list:
    """Format a full media transcript/extraction returned by the backend."""
    message_id = result.get("telegram_message_id", "unknown")
    media_type = result.get("media_type")
    label = MEDIA_LABELS.get(media_type, "Media")
    status = result.get("media_processing_status") or "unknown"
    lines = [f"{label} content (msg#{message_id})", f"Status: {status}"]

    file_name = result.get("media_file_name")
    if file_name:
        lines.append(f"File: {file_name}")

    caption = result.get("text")
    if caption:
        lines.extend(["", "Caption:", caption])

    telegram_message_url = result.get("telegram_message_url")
    if telegram_message_url:
        lines.extend(["", f"Telegram message: {telegram_message_url}"])

    media_download_url = _absolute_url(result.get("media_download_url"), base_url)
    if media_download_url:
        lines.extend(["", f"Download media: {media_download_url}"])

    summary = result.get("content_summary")
    if summary:
        lines.extend(["", "Summary:", summary])

    content = result.get("content_text")
    if content:
        section = (
            "Full transcript:"
            if media_type in {"voice", "audio", "video", "video_note"}
            else "Extracted content:"
        )
        lines.extend(["", section, content])

    if result.get("has_more"):
        lines.extend(
            [
                "",
                f"More content is available. next_cursor={result.get('next_cursor')}",
            ]
        )
    next_action = result.get("next_action")
    if next_action:
        lines.extend(["", f"Next action: {next_action}"])

    if status == "failed":
        error_code = result.get("media_processing_error_code") or "unknown"
        lines.extend(["", f"Processing error: {error_code}"])
    elif status in {"pending", "queued", "processing"}:
        lines.extend(["", "The file is being processed in the background."])

    content: list = [TextContent(type="text", text="\n".join(lines))]
    resource = _media_resource_link(result, base_url)
    if resource:
        content.append(resource)
    return content


async def _list_all_chats(api: TelegramAIClient) -> dict[str, Any]:
    """Collect all chats through cursor pagination for summary tools."""
    chats: list[dict[str, Any]] = []
    cursor: str | None = None
    total = 0

    while True:
        page = await api.list_chats(limit=200, cursor=cursor)
        chats.extend(page.get("chats", []))
        total = page.get("total", total)
        cursor = page.get("next_cursor")
        if not cursor:
            break

    return {
        "chats": chats,
        "total": total or len(chats),
        "has_more": False,
        "next_cursor": None,
    }


def _normalize_chat_search_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().removeprefix("@").casefold()


def _tokenize_chat_search(value: str) -> list[str]:
    return [token for token in re.findall(r"\w+", value, flags=re.UNICODE) if token]


def _chat_match_score(chat: dict[str, Any], query: str) -> int:
    """Score a chat title/username match for search_chats."""
    query_norm = _normalize_chat_search_text(query)
    if not query_norm:
        return 0

    title = _normalize_chat_search_text(chat.get("title"))
    username = _normalize_chat_search_text(chat.get("username"))
    score = 0

    if query_norm == title:
        score += 220
    if query_norm == username:
        score += 240

    if title and query_norm in title:
        score += 140
    if username and query_norm in username:
        score += 180

    query_tokens = _tokenize_chat_search(query_norm)
    if not query_tokens:
        return score

    title_tokens = set(_tokenize_chat_search(title))
    username_tokens = set(_tokenize_chat_search(username))

    for token in query_tokens:
        if token == title:
            score += 80
        if token == username:
            score += 100
        if token in title:
            score += 35
        if token in username:
            score += 55
        if token in title_tokens:
            score += 25
        if token in username_tokens:
            score += 35

    return score


def _chat_sort_key(chat: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(chat.get("last_activity_at") or chat.get("last_sync_at") or ""),
        int(chat.get("total_messages_synced") or 0),
        str(chat.get("title") or ""),
    )


def _normalize_search_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().casefold().split())


def _tokenize_search_text(value: str) -> list[str]:
    return [token for token in re.findall(r"\w+", value, flags=re.UNICODE) if token]


def _search_fetch_limit(requested_limit: int, chat_id: str | None) -> int:
    minimum = 50 if chat_id else 100
    return min(100, max(requested_limit, minimum))


def _search_result_lexical_score(result: dict[str, Any], query: str) -> int:
    query_norm = _normalize_search_text(query)
    if not query_norm:
        return 0

    query_tokens = _tokenize_search_text(query_norm)
    text = _normalize_search_text(
        " ".join(
            value
            for value in (
                result.get("text"),
                result.get("content_summary"),
                result.get("content_preview"),
            )
            if isinstance(value, str)
        )
    )
    sender = _normalize_search_text(result.get("sender_name"))
    chat_title = _normalize_search_text(result.get("chat_title"))
    username = _normalize_search_text(result.get("chat_username")).removeprefix("@")

    score = 0
    if query_norm and query_norm in text:
        score += 220
    if query_norm and query_norm in sender:
        score += 420
    if query_norm and query_norm in chat_title:
        score += 420
    if query_norm and query_norm in username:
        score += 460

    for token in query_tokens:
        if len(token) < 3:
            continue
        if token in username:
            score += 180
        if token in sender:
            score += 150
        if token in chat_title:
            score += 150
        if token in text:
            score += 35

    return score


def _rerank_search_results(
    query: str, results: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    def sort_key(result: dict[str, Any]) -> tuple[int, float, str]:
        lexical_score = _search_result_lexical_score(result, query)
        similarity = float(result.get("similarity") or 0)
        sent_at = str(result.get("sent_at") or "")
        return (lexical_score, similarity, sent_at)

    reranked = sorted(results, key=sort_key, reverse=True)
    return reranked[:limit]


async def _search_chats(
    api: TelegramAIClient,
    query: str,
    limit: int,
) -> dict[str, Any]:
    """Search chats by title/username using MCP-side filtering."""
    chats_result = await _list_all_chats(api)
    matches: list[tuple[int, dict[str, Any]]] = []

    for chat in chats_result.get("chats", []):
        score = _chat_match_score(chat, query)
        if score > 0:
            matches.append((score, chat))

    matches.sort(key=lambda item: (item[0], *_chat_sort_key(item[1])), reverse=True)
    return {
        "query": query,
        "total": len(matches),
        "chats": [chat for _, chat in matches[:limit]],
    }


_CHAT_TYPES = {"private", "group", "supergroup", "channel"}
_SEARCH_MODES = {"hybrid", "exact"}


def _validated_choice(value: Any, name: str, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f'"{name}" must be one of: {choices}')
    return value


def _validated_chat_types(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError('"chat_types" must be a non-empty array')
    if not all(isinstance(chat_type, str) and chat_type in _CHAT_TYPES for chat_type in value):
        raise ValueError('"chat_types" values must be private, group, supergroup, or channel')
    return list(dict.fromkeys(value))


async def _find_chats(
    api: TelegramAIClient,
    *,
    query: str,
    mode: str,
    chat_types: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    messages_per_chat: int,
) -> dict[str, Any]:
    """Find unique chats from title matches and representative message hits."""
    inventory_task = asyncio.create_task(_list_all_chats(api))
    selected_types = set(chat_types or _CHAT_TYPES)
    chats_by_id: dict[str, dict[str, Any]] = {}

    cursor: str | None = None
    seen_cursors: set[str] = set()
    pages_scanned = 0
    message_hits_scanned = 0
    capped = False
    complete = False
    max_pages = 10 if mode == "exact" else 1

    try:
        while pages_scanned < max_pages:
            search_arguments: dict[str, Any] = {
                "query": query,
                "limit": 100,
                "mode": mode,
            }
            if chat_types:
                search_arguments["chat_types"] = chat_types
            if date_from:
                search_arguments["date_from"] = date_from
            if date_to:
                search_arguments["date_to"] = date_to
            if cursor:
                search_arguments["cursor"] = cursor

            page = await api.execute_data_tool("search_messages", search_arguments)
            pages_scanned += 1
            page_results = page.get("results", [])
            if not isinstance(page_results, list):
                raise RuntimeError("Backend returned invalid search results")
            message_hits_scanned += len(page_results)

            for hit in page_results:
                if not isinstance(hit, dict):
                    continue
                chat_id = str(hit.get("chat_id") or "")
                if not chat_id or hit.get("chat_type") not in selected_types:
                    continue
                chat = chats_by_id.setdefault(
                    chat_id,
                    {
                        "id": chat_id,
                        "title": hit.get("chat_title") or "Unknown",
                        "chat_type": hit.get("chat_type") or "unknown",
                        "username": hit.get("chat_username"),
                        "telegram_chat_id": hit.get("chat_telegram_id"),
                        "title_match_score": 0,
                        "message_hit_count": 0,
                        "relevance": 0.0,
                        "representative_messages": [],
                    },
                )
                chat["message_hit_count"] += 1
                chat["relevance"] = max(
                    float(chat.get("relevance") or 0),
                    float(hit.get("similarity") or 0),
                )
                chat["representative_messages"].append(hit)

            next_cursor = page.get("next_cursor")
            if not page.get("has_more") or not next_cursor:
                complete = True
                break
            if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                raise RuntimeError("Backend returned an invalid search cursor sequence")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        inventory = await inventory_task
    except BaseException:
        if not inventory_task.done():
            inventory_task.cancel()
        try:
            await inventory_task
        except (Exception, asyncio.CancelledError):
            pass
        raise

    for inventory_chat in inventory.get("chats", []):
        chat_id = str(inventory_chat.get("id") or "")
        if not chat_id or inventory_chat.get("chat_type") not in selected_types:
            continue
        title_score = _chat_match_score(inventory_chat, query)
        if title_score <= 0:
            continue
        chat = chats_by_id.setdefault(
            chat_id,
            {
                **inventory_chat,
                "message_hit_count": 0,
                "relevance": 0.0,
                "representative_messages": [],
            },
        )
        chat["title_match_score"] = title_score

    if not complete:
        capped = True

    chats = list(chats_by_id.values())
    for chat in chats:
        messages = chat["representative_messages"]
        messages.sort(
            key=lambda hit: (
                float(hit.get("similarity") or 0),
                str(hit.get("sent_at") or ""),
            ),
            reverse=True,
        )
        chat["representative_messages"] = messages[:messages_per_chat]
    chats.sort(
        key=lambda chat: (
            int(int(chat.get("message_hit_count") or 0) > 0),
            float(chat.get("relevance") or 0),
            int(chat.get("title_match_score") or 0),
            int(chat.get("message_hit_count") or 0),
            _chat_sort_key(chat),
        ),
        reverse=True,
    )

    return {
        "query": query,
        "mode": mode,
        "total": len(chats),
        "chats": chats[:limit],
        "coverage": {
            "inventory_chats_scanned": len(inventory.get("chats", [])),
            "message_hits_scanned": message_hits_scanned,
            "pages_scanned": pages_scanned,
            "complete": complete,
            "capped": capped,
        },
    }


def _format_username_ref(username: Any) -> str:
    if not isinstance(username, str):
        return ""
    normalized = username.strip().removeprefix("@")
    if not normalized:
        return ""
    return f"@{normalized} | https://t.me/{normalized}"


def _private_chat_link(chat_type: Any, telegram_chat_id: Any, message_id: Any) -> str:
    if chat_type not in {"supergroup", "channel"}:
        return ""
    if not isinstance(telegram_chat_id, int) or not isinstance(message_id, int):
        return ""
    if message_id <= 0:
        return ""
    channel_id = abs(telegram_chat_id)
    if channel_id >= 10**12:
        channel_id -= 10**12
    if channel_id <= 0:
        return ""
    return f"https://t.me/c/{channel_id}/{message_id}"


def _absolute_url(url: Any, base_url: str) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    if url.startswith(("http://", "https://")):
        return url
    return f"{base_url.rstrip('/')}/{url.lstrip('/')}"


def _client_base_url(api: TelegramAIClient) -> str:
    public_url = os.environ.get("TELEGRAM_AI_PUBLIC_URL", "").strip()
    if public_url:
        return public_url.rstrip("/")
    base_url = getattr(api, "base_url", None)
    return base_url if isinstance(base_url, str) and base_url else "https://telegram.waiwai.is"


def _media_resource_link(result: dict, base_url: str) -> ResourceLink | None:
    url = _absolute_url(result.get("media_download_url"), base_url)
    if not url:
        return None
    message_id = result.get("telegram_message_id", "unknown")
    file_name = result.get("media_file_name") or f"telegram-media-{message_id}"
    size = result.get("media_file_size")
    if not isinstance(size, int) or size < 0:
        size = None
    mime_type = result.get("media_mime_type")
    if not isinstance(mime_type, str) or not mime_type.strip():
        mime_type = None
    return ResourceLink(
        type="resource_link",
        name=str(file_name),
        title=f"Download {file_name}",
        uri=url,
        description="Short-lived authenticated Telegram media download",
        mimeType=mime_type,
        size=size,
    )


def format_media_download(result: dict, base_url: str) -> list:
    """Return a compact download link plus an MCP resource link."""
    url = _absolute_url(result.get("media_download_url"), base_url)
    if not url:
        return [
            TextContent(type="text", text="No downloadable media is available for this message.")
        ]

    lines = [
        f"Download URL: {url}",
        f"File: {result.get('media_file_name') or 'telegram-media'}",
    ]
    resource = _media_resource_link(result, base_url)
    return (
        [TextContent(type="text", text="\n".join(lines)), resource]
        if resource
        else [TextContent(type="text", text="\n".join(lines))]
    )


def _format_bytes(size: Any) -> str | None:
    if not isinstance(size, int) or size <= 0:
        return None
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return None


def _format_duration(seconds: Any) -> str | None:
    if not isinstance(seconds, int) or seconds <= 0:
        return None
    return f"{seconds // 60}:{seconds % 60:02d}"


def _missing_lines(missing: list) -> list[str]:
    """Name every locator that no longer resolves, rather than dropping it."""
    return [
        f"Not found: chat {entry.get('chat_id')} msg#{entry.get('telegram_message_id')}"
        for entry in missing
    ]


def format_file_results(result: dict, base_url: str) -> list:
    """Render a file listing so an agent can judge, download, or wait.

    Every ready file also leaves a resource_link, so the binary can be fetched
    without the file itself ever entering the conversation.
    """
    files = result.get("files") or []
    next_action = result.get("next_action")
    missing = result.get("not_found") or []
    if not files:
        query = result.get("query")
        if query:
            headline = f'No files found for query: "{query}".'
        elif result.get("mode") == "locators":
            # Locator mode applies no filters, so blaming filters would be a lie.
            headline = "None of the requested messages carry a file any more."
        else:
            headline = "No files found for these filters."
        lines = [headline]
        # Every locator being stale is exactly when the caller most needs to be
        # told which ones, and an empty page must not swallow that.
        lines.extend(_missing_lines(missing))
        if next_action:
            lines.extend(["", f"Next action: {next_action}"])
        return [TextContent(type="text", text="\n".join(lines))]

    counts = result.get("counts") or {}
    summary = ", ".join(
        f"{counts[state]} {label}"
        for state, label in (
            ("ready", "ready to download"),
            ("fetching", "fetching"),
            ("queued", "queued"),
            ("not_prepared", "not prepared"),
            ("unavailable", "unavailable"),
        )
        if counts.get(state)
    )
    total = result.get("total", len(files))
    noun = "file" if total == 1 else "files"
    lines = [f"Found {total} {noun}. {summary}." if summary else f"Found {total} {noun}."]
    lines.append("")

    resources: list[ResourceLink] = []
    for entry in files:
        sender = entry.get("sender_name") or ("You" if entry.get("is_outgoing") else "Unknown")
        label = MEDIA_LABELS.get(entry.get("media_type"), "Media")
        head = [f"[{label}] {entry.get('media_file_name') or '(no filename)'}"]
        size = _format_bytes(entry.get("media_file_size"))
        if size:
            head.append(size)
        if entry.get("media_mime_type"):
            head.append(entry["media_mime_type"])
        duration = _format_duration(entry.get("media_duration_seconds"))
        if duration:
            head.append(duration)

        state = str(entry.get("download_state") or "unknown").replace("_", " ").upper()
        details = [
            f"Sent: {_format_date(entry.get('sent_at'))}",
            state,
            f"Chat ID: {entry.get('chat_id', '')}",
            f"msg#{entry.get('telegram_message_id', '')}",
        ]
        download_url = _absolute_url(entry.get("media_download_url"), base_url)
        if download_url:
            details.append(f"Download: {download_url}")
            details.append(f"Link expires: {entry.get('download_url_expires_at')}")
            resource = _media_resource_link(entry, base_url)
            if resource:
                resources.append(resource)
        if entry.get("telegram_message_url"):
            details.append(f"Open: {entry['telegram_message_url']}")
        if entry.get("download_state") == "unavailable" and entry.get("error_code"):
            details.append(f"Error: {entry['error_code']}")

        chat_title = entry.get("chat_title") or "Unknown"
        block = [f"[{chat_title}] {sender}: {' | '.join(head)}", f"  - {' | '.join(details)}"]
        # A summary is what tells an agent which of twelve PDFs is the invoice,
        # so it gets its own line rather than being lost in the detail run.
        if entry.get("content_summary"):
            block.append(f"  - Summary: {entry['content_summary']}")
        elif entry.get("caption"):
            block.append(f"  - Caption: {entry['caption']}")
        if entry.get("matched_because"):
            block.append(f"  - Matched: {entry['matched_because']}")
        # Only the states the page-level next action cannot speak to per row.
        if entry.get("download_state") in {"not_prepared", "unavailable"}:
            block.append(f"  - Next: {entry.get('next_action')}")
        lines.append("\n".join(block))
        lines.append("")

    lines.extend(_missing_lines(missing))
    if result.get("has_more") and result.get("next_cursor"):
        lines.append(
            f'--- More files available. Use cursor="{result["next_cursor"]}" '
            "to load the next page ---"
        )
    elif result.get("truncated"):
        lines.append(
            f"--- {result.get('matched_total')} files matched and {total} were "
            "returned. Raise limit (max 100) or narrow the filters - relevance "
            "mode has no cursor. ---"
        )
    if next_action:
        lines.extend(["", f"Next action: {next_action}"])
    return [TextContent(type="text", text="\n".join(lines)), *resources]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    api = get_client()
    try:
        payload = await api.list_data_tools()
    finally:
        await api.close()
    definitions = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(definitions, list):
        raise RuntimeError("Backend returned an invalid shared tool registry")
    shared_tools: list[Tool] = []
    for definition in definitions:
        if not isinstance(definition, dict):
            raise RuntimeError("Backend returned an invalid shared tool definition")
        name = definition.get("name")
        description = definition.get("description")
        parameters = definition.get("parameters")
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(parameters, dict)
        ):
            raise RuntimeError("Backend returned an invalid shared tool definition")
        shared_tools.append(Tool(name=name, description=description, inputSchema=parameters))
    shared_names = {tool.name for tool in shared_tools}
    legacy_tools = [
        Tool(
            name="get_data_status",
            description=(
                "Check the status of your Telegram data. Returns a compact summary: total chats/messages, "
                "chat type breakdown, data freshness distribution, and top 10 most recently active chats. "
                "Account-wide diagnostics can be expensive; not needed before ordinary retrieval. "
                "Use list_chats to browse all chats, or search_messages to find specific content."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="search_messages",
            description=(
                "Semantic search across synced message text, media summaries, full "
                "transcripts, and extracted document content using vector embeddings. "
                "Finds messages by meaning, not just keywords. Only searches already-synced data — "
                "if results seem incomplete, sync the relevant chat first. "
                "If you're trying to find a person/chat by name or username, use search_chats first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - describe what you're looking for",
                    },
                    "chat_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Optional: Limit search to one or more chat IDs",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (1-100, default: 20)",
                        "default": 20,
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional: Only return messages sent after this date (ISO 8601, e.g. 2025-01-15)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional: Only return messages sent before this date (ISO 8601, e.g. 2025-02-15)",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Opaque next_cursor returned by the previous search page",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_chats",
            description=(
                "Find a chat by title, person name, or Telegram username. "
                "Use this before get_chat_messages when the user asks about a specific person/chat. "
                "This is more reliable than search_messages for exact person or username lookup."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Chat/person name or Telegram username to find",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum chats to return (1-50, default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="find_chats",
            description=(
                "Find unique Telegram chats by title and by what their messages discuss. "
                "Use mode=hybrid for a natural-language description; use mode=exact for "
                "a known literal phrase and exhaustive cursor scanning (up to 1,000 hits). "
                "Returns representative messages and direct Telegram links. Use chat_types "
                "to restrict the search to groups/supergroups or other chat kinds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Chat title, person, literal phrase, or natural-language topic description",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "exact"],
                        "default": "hybrid",
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
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "messages_per_chat": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_chats",
            description=(
                "List synced Telegram chats with unread and message counts, the latest message "
                "ID/sender/text preview, sync status, and freshness. "
                "Returns paginated results — use the cursor from the response to load more pages. "
                "Set unread_only=true to return only chats with unread messages. "
                "Use to discover chat IDs. If you're looking for a specific chat or person, prefer "
                "search_chats first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_type": {
                        "type": "string",
                        "description": "Filter by chat type: private, group, supergroup, channel",
                        "enum": ["private", "group", "supergroup", "channel"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of chats per page (1-200, default: 50)",
                        "default": 50,
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor from previous response — pass to load the next page",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Return only chats with unread_count greater than zero",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="refresh_chats",
            description=(
                "Refresh chat metadata and unread counts from Telegram without marking any "
                "messages as read. Call this before list_chats with unread_only=true when "
                "current unread state matters."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_chat_messages",
            description=(
                "Read messages from a chat, newest-first, with cursor pagination (up to 500 per page). "
                "Pass 'before' cursor from previous response to page backwards through history. "
                "When you reach 'End of synced messages', use sync_chat with message_limit=0 to "
                "download older history from Telegram."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The chat ID to read messages from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to return per page (1-500, default: 50)",
                        "default": 50,
                    },
                    "before": {
                        "type": "string",
                        "description": "Pagination cursor from previous response's next_cursor — pass this to get the next (older) page of messages",
                    },
                },
                "required": ["chat_id"],
            },
        ),
        Tool(
            name="get_message",
            description=("Return complete metadata, links and lifecycle for one Telegram message."),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "telegram_message_id": {"type": "integer"},
                },
                "required": ["chat_id", "telegram_message_id"],
            },
        ),
        Tool(
            name="prepare_media",
            description=(
                "Idempotently fetch/process original media and return progress, retry_after and next action."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "telegram_message_id": {"type": "integer"},
                },
                "required": ["chat_id", "telegram_message_id"],
            },
        ),
        Tool(
            name="get_message_content",
            description=(
                "Get the complete background-processed content for one media message: "
                "its original caption, summary, and full transcript or extracted document text. "
                "Use chat_id and the numeric msg# shown by search_messages or get_chat_messages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The chat ID containing the media message",
                    },
                    "telegram_message_id": {
                        "type": "integer",
                        "description": "The numeric Telegram message ID shown as msg#",
                    },
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
        Tool(
            name="get_transcript_segments",
            description=(
                "Read timestamped transcript segments with speaker, confidence and language."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "telegram_message_id": {"type": "integer"},
                    "cursor": {"type": "integer", "minimum": 0},
                    "start_ms": {"type": "integer", "minimum": 0},
                    "end_ms": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "required": ["chat_id", "telegram_message_id"],
            },
        ),
        Tool(
            name="download_media",
            description=(
                "Get a short-lived authenticated download link for the original Telegram media file. "
                "Supports documents, photos, videos, audio, voice messages, and video notes. "
                "The MCP result includes a resource_link so an agent can fetch the binary without "
                "putting the file itself into the conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The chat ID containing the media message",
                    },
                    "telegram_message_id": {
                        "type": "integer",
                        "description": "The numeric Telegram message ID shown as msg#",
                    },
                },
                "required": ["chat_id", "telegram_message_id"],
            },
        ),
        Tool(
            name="get_daily_digest",
            description=(
                "Get an AI-generated daily digest summarizing Telegram activity for a specific date. "
                "Covers the top active chats with message counts and key discussion points. "
                "Defaults to yesterday if no date is specified."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (defaults to yesterday)",
                    },
                },
            },
        ),
        Tool(
            name="sync_chat",
            description=(
                "Download messages from Telegram into the database. Use message_limit=0 for full history "
                "(recommended). Returns a job_id — poll get_sync_status every 10-15 seconds until completed. "
                "The progress will show messages fetched out of total (e.g., '362 of 1,500 messages'). "
                "After completion, use get_chat_messages to read the synced messages. "
                "File transcription, extraction, image analysis, and summaries continue "
                "independently in the background after messages are saved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The chat ID to sync",
                    },
                    "message_limit": {
                        "type": "integer",
                        "description": "Maximum messages to download. 0 = unlimited (full history). Default: 0 (download all messages).",
                        "default": 0,
                    },
                },
                "required": ["chat_id"],
            },
        ),
        Tool(
            name="get_sync_status",
            description=(
                "Check sync job progress. Poll every 10-15 seconds until status is 'completed'. "
                "Returns status, messages fetched (seen) out of total from Telegram, "
                "messages saved to database, and progress percentage. "
                "The total becomes available after the first batch is fetched from Telegram."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by sync_chat",
                    },
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="send_message",
            description=(
                "Send a text message to a Telegram chat as the connected user account. "
                'Use chat_id="me" to send to your own Saved Messages (Избранное); no chat lookup needed. '
                "For another chat, get its ID from list_chats or search_messages. "
                "Saved Messages preserves literal text, up to 4096 characters; send longer text as a file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": 'Use "me" for Saved Messages, or the internal chat UUID for another chat',
                    },
                    "text": {
                        "type": "string",
                        "description": "The message text to send",
                    },
                },
                "required": ["chat_id", "text"],
            },
        ),
        Tool(
            name="send_file",
            description=(
                "Download a file from a URL and send it to a Telegram chat as the connected user account. "
                'Use chat_id="me" for Saved Messages (Избранное), preserving the original as a document. '
                "Supports any file type (PDF, images, documents, etc.). "
                "The file is downloaded server-side and sent via Telegram. "
                "To pass on a file from another chat, use the media_download_url from get_files or "
                "download_media, and always set file_name to that file's media_file_name - a signed "
                "media URL ends in /media, so the name is otherwise lost and the file arrives as "
                '"media" with no extension.'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": 'Use "me" for Saved Messages, or the internal chat UUID for another chat',
                    },
                    "file_url": {
                        "type": "string",
                        "description": "URL of the file to download and send",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption text for the file",
                    },
                    "file_name": {
                        "type": "string",
                        "description": (
                            "File name to send it under. Required in practice for a signed media "
                            "URL, whose path carries no name - pass media_file_name from get_files."
                        ),
                    },
                },
                "required": ["chat_id", "file_url"],
            },
        ),
        Tool(
            name="reply_to_message",
            description=(
                "Reply to a specific message in a Telegram chat. "
                "Requires telegram_message_id — the numeric Telegram message ID shown in "
                "search_messages and get_chat_messages results as 'msg#'. "
                "The reply appears as a quoted reply in the chat."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The chat ID containing the message to reply to",
                    },
                    "telegram_message_id": {
                        "type": "integer",
                        "description": "The Telegram message ID to reply to (from search/chat message results)",
                    },
                    "text": {
                        "type": "string",
                        "description": "The reply text",
                    },
                },
                "required": ["chat_id", "telegram_message_id", "text"],
            },
        ),
        Tool(
            name="search_today_requests",
            description=(
                "Search today's messages for specific requests or topics. "
                "Automatically filters to today only. "
                "Useful for finding who asked for something today (e.g., 'who asked for the presentation')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in today's messages",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (1-100, default: 20)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
    ]
    return shared_tools + [tool for tool in legacy_tools if tool.name not in shared_names]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent] | CallToolResult:
    """Handle tool calls."""
    args = _as_dict(arguments)
    api: TelegramAIClient | None = None

    try:
        api = get_client()
        if name == "get_data_status":
            result = await api.execute_data_tool("get_data_status")
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "get_inbox":
            result = await api.execute_data_tool("get_inbox", dict(args))
            return format_inbox(result)

        elif name == "get_files":
            # Arguments are validated by the backend against the schema this server
            # already advertises from the shared registry, so pass them through.
            result = await api.execute_data_tool("get_files", dict(args))
            return format_file_results(result, _client_base_url(api))

        elif name == "search_messages":
            query = _require_str(args, "query")
            limit = _optional_int(args, "limit", default=20, minimum=1, maximum=100)
            mode = _validated_choice(args.get("mode"), "mode", _SEARCH_MODES, "hybrid")
            chat_types = _validated_chat_types(args.get("chat_types"))
            date_from = _optional_iso_datetime(args, "date_from")
            date_to = _optional_iso_datetime(args, "date_to", end_of_day=True)
            chat_ids = args.get("chat_ids")
            if chat_ids is not None and (
                not isinstance(chat_ids, list)
                or not all(isinstance(chat_id, str) for chat_id in chat_ids)
            ):
                raise ValueError('"chat_ids" must be an array of UUID strings')
            tool_arguments = {
                "query": query,
                "limit": limit,
                "mode": mode if "mode" in args else None,
                "chat_types": chat_types,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
                "chat_ids": chat_ids,
                "cursor": args.get("cursor"),
            }
            result = await api.execute_data_tool(
                "search_messages",
                {key: value for key, value in tool_arguments.items() if value is not None},
            )
            return format_search_results(result, _client_base_url(api))

        elif name == "find_chats":
            query = _require_str(args, "query")
            mode = _validated_choice(args.get("mode"), "mode", _SEARCH_MODES, "hybrid")
            chat_types = _validated_chat_types(args.get("chat_types"))
            date_from = _optional_iso_datetime(args, "date_from")
            date_to = _optional_iso_datetime(args, "date_to", end_of_day=True)
            limit = _optional_int(args, "limit", default=10, minimum=1, maximum=50)
            messages_per_chat = _optional_int(
                args,
                "messages_per_chat",
                default=3,
                minimum=1,
                maximum=5,
            )
            result = await _find_chats(
                api,
                query=query,
                mode=mode,
                chat_types=chat_types,
                date_from=date_from.isoformat() if date_from else None,
                date_to=date_to.isoformat() if date_to else None,
                limit=limit,
                messages_per_chat=messages_per_chat,
            )
            return format_find_chats_results(result)

        elif name == "search_chats":
            query = _require_str(args, "query")
            limit = _optional_int(args, "limit", default=10, minimum=1, maximum=50)
            result = await _search_chats(api, query=query, limit=limit)
            return format_chat_search_results(result)

        elif name == "list_chats":
            chat_type = args.get("chat_type")
            if chat_type is not None and not isinstance(chat_type, str):
                raise ValueError('"chat_type" must be a string')
            limit = _optional_int(args, "limit", default=50, minimum=1, maximum=200)
            cursor = args.get("cursor")
            if cursor is not None and not isinstance(cursor, str):
                raise ValueError('"cursor" must be a string')
            unread_only = args.get("unread_only", False)
            if not isinstance(unread_only, bool):
                raise ValueError('"unread_only" must be a boolean')
            result = await api.list_chats(
                chat_type=chat_type,
                limit=limit,
                cursor=cursor,
                unread_only=unread_only,
            )
            return format_chat_list(result)

        elif name == "refresh_chats":
            result = await api.refresh_chats()
            total = int(result.get("total") or len(result.get("chats", [])))
            noun = "chat" if total == 1 else "chats"
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Refreshed {total} {noun} from Telegram. Unread counts are current.\n"
                        "No messages were marked read."
                    ),
                )
            ]

        elif name == "get_chat_messages":
            chat_id = _require_str(args, "chat_id")
            limit = _optional_int(args, "limit", default=50, minimum=1, maximum=MAX_LIMIT)
            before = args.get("before")
            if before is not None and not isinstance(before, str):
                raise ValueError('"before" must be a string cursor')
            result = await api.get_messages(
                chat_id=chat_id,
                limit=limit,
                before=before,
            )
            return format_chat_messages(result, _client_base_url(api))

        elif name in {
            "get_message",
            "prepare_media",
            "get_message_content",
            "get_transcript_segments",
        }:
            chat_id = _require_str(args, "chat_id")
            telegram_message_id = args.get("telegram_message_id")
            if not isinstance(telegram_message_id, int):
                raise ValueError('"telegram_message_id" must be an integer')
            tool_arguments = dict(args)
            tool_arguments["chat_id"] = chat_id
            result = await api.execute_data_tool(name, tool_arguments)
            if name == "get_message_content":
                return format_message_content(result, _client_base_url(api))
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "list_drafts":
            result = await api.execute_data_tool("list_drafts", {})
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False),
                )
            ]

        elif name == "save_draft":
            chat_id = _require_str(args, "chat_id")
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError('"text" must be a non-empty string')
            result = await api.execute_data_tool(
                "save_draft",
                {"chat_id": chat_id, "text": text},
            )
            return format_draft_result(result)

        elif name == "clear_draft":
            chat_id = _require_str(args, "chat_id")
            result = await api.execute_data_tool(
                "clear_draft",
                {"chat_id": chat_id},
            )
            return format_clear_draft_result(result)

        elif name == "download_media":
            chat_id = _require_str(args, "chat_id")
            telegram_message_id = args.get("telegram_message_id")
            if not isinstance(telegram_message_id, int):
                raise ValueError('"telegram_message_id" must be an integer')
            result = await api.execute_data_tool(
                "download_media",
                {
                    "chat_id": chat_id,
                    "telegram_message_id": telegram_message_id,
                },
            )
            return format_media_download(result, _client_base_url(api))

        elif name == "get_daily_digest":
            digest_date = _optional_iso_date(args, "date")
            result = await api.get_daily_digest(digest_date)
            return format_digest(result)

        elif name == "sync_chat":
            chat_id = _require_str(args, "chat_id")
            message_limit = _optional_int(
                args, "message_limit", default=0, minimum=0, maximum=10000
            )
            result = await api.sync_chat(
                chat_id=chat_id,
                message_limit=message_limit if message_limit > 0 else None,
            )
            return format_sync_started(result)

        elif name == "get_sync_status":
            job_id = _require_str(args, "job_id")
            result = await api.get_sync_status(job_id)
            return format_sync_status(result)

        elif name == "send_message":
            chat_id = _require_str(args, "chat_id")
            text = _require_str(args, "text")
            result = await api.send_message(chat_id=chat_id, text=text)
            return format_send_result(result, "Message sent")

        elif name == "send_file":
            chat_id = _require_str(args, "chat_id")
            file_url = _require_str(args, "file_url")
            caption = args.get("caption")
            file_name = args.get("file_name")
            result = await api.send_file(
                chat_id=chat_id,
                file_url=file_url,
                caption=caption,
                file_name=file_name,
            )
            return format_send_result(result, "File sent")

        elif name == "reply_to_message":
            chat_id = _require_str(args, "chat_id")
            telegram_message_id = args.get("telegram_message_id")
            if not isinstance(telegram_message_id, int):
                raise ValueError('"telegram_message_id" must be an integer')
            text = _require_str(args, "text")
            result = await api.reply_to_message(
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                text=text,
            )
            return format_send_result(result, "Reply sent")

        elif name == "search_today_requests":
            query = _require_str(args, "query")
            limit = _optional_int(args, "limit", default=20, minimum=1, maximum=100)
            from datetime import UTC

            today = datetime.now(UTC).date()
            result = await api.search_messages(
                query=query,
                limit=limit,
                date_from=datetime(today.year, today.month, today.day, tzinfo=UTC),
            )
            return format_search_results(result, _client_base_url(api))

        else:
            return _tool_error(f"Unknown tool: {name}")

    except ValueError as e:
        return _tool_error(str(e))
    except Exception as e:
        return _tool_error(str(e))
    finally:
        if api is not None:
            await api.close()


def format_search_results(
    result: dict,
    base_url: str = "https://telegram.waiwai.is",
) -> list:
    """Format search results for display."""
    if not result.get("results"):
        return [TextContent(type="text", text="No messages found matching your query.")]

    total = result.get("total", 0)
    query = result.get("query", "")
    lines = [f'Found {total} messages for query: "{query}"\n']
    resources: list[ResourceLink] = []
    for r in result.get("results", []):
        sender = r.get("sender_name") or ("You" if r.get("is_outgoing") else "Unknown")
        text = _format_media_label(r)
        if r.get("content_summary") and r.get("content_preview"):
            text += f" | Matching content: {r['content_preview']}"
        similarity = r.get("similarity", 0) * 100
        sent_at = _format_date(r.get("sent_at"))
        chat_title = r.get("chat_title") or "Unknown"
        username_ref = _format_username_ref(r.get("chat_username"))
        private_link = _private_chat_link(
            r.get("chat_type"),
            r.get("chat_telegram_id"),
            r.get("telegram_message_id"),
        )
        chat_id = r.get("chat_id", "")
        msg_id = r.get("telegram_message_id", "")
        details = [
            f"Sent: {sent_at}",
            f"Relevance: {similarity:.0f}%",
            f"Chat ID: {chat_id}",
            f"msg#{msg_id}",
        ]
        if username_ref:
            details.append(f"Username: {username_ref}")
        message_url = r.get("telegram_message_url") or private_link
        if message_url:
            details.append(f"Open: {message_url}")
        media_url = _absolute_url(r.get("media_download_url"), base_url)
        if media_url:
            details.append(f"Download media: {media_url}")
            resource = _media_resource_link(r, base_url)
            if resource:
                resources.append(resource)
        urls = list(dict.fromkeys([*(r.get("visible_urls") or []), *(r.get("hidden_urls") or [])]))
        if urls:
            details.append(f"Links: {', '.join(urls)}")
        if r.get("deleted_at"):
            details.append(f"Deleted in Telegram: {_format_date(r['deleted_at'])}")
        lines.append(f"[{chat_title}] {sender}: {text}\n  - {' | '.join(details)}\n")
    if result.get("has_more"):
        lines.append(
            "More results are available. Call search_messages again with "
            f"cursor={result.get('next_cursor')}."
        )
    return [TextContent(type="text", text="\n".join(lines)), *resources]


def _freshness_label(last_sync_at: Any, listener_active: bool = False) -> str:
    """Compute freshness label from last_sync_at timestamp."""
    if not last_sync_at:
        return "NEVER"
    try:
        if isinstance(last_sync_at, str):
            sync_dt = datetime.fromisoformat(last_sync_at)
        elif isinstance(last_sync_at, datetime):
            sync_dt = last_sync_at
        else:
            return "NEVER"
        from datetime import UTC, timedelta

        now = datetime.now(UTC)
        # Ensure sync_dt is timezone-aware
        if sync_dt.tzinfo is None:
            sync_dt = sync_dt.replace(tzinfo=UTC)
        age = now - sync_dt
        if listener_active and age < timedelta(minutes=5):
            return "LIVE"
        if age < timedelta(hours=1):
            return "FRESH"
        return "OLD_SYNC"
    except (ValueError, TypeError):
        return "NEVER"


def format_inbox(result: dict) -> list[TextContent]:
    """Compact primary evidence, with one chat header instead of repeated metadata."""
    chats = result.get("conversations", [])
    lines = [
        f"{len(chats)} conversation tails; messages oldest-first. Checked: {result.get('checked_at')}",
        f"Scope: {json.dumps(result.get('scope', {}), ensure_ascii=False)}",
        "These are original message excerpts, not a classified or complete list of obligations.",
    ]
    listener = result.get("listener_active")
    if listener is True:
        lines.append("Listener connected. Old last_sync_at alone is not a history gap.")
    else:
        state = "offline" if listener is False else "unknown"
        lines.append(
            f"Listener {state}; newest Telegram activity may be missing. Verify selected relevant chats."
        )
    for chat in chats:
        lines.extend(
            [
                "",
                f"## {chat.get('title')} ({chat.get('chat_type')}) | chat_id={chat.get('chat_id')}",
                f"Unread={chat.get('unread_count', 0)} | last_sync_at={chat.get('last_sync_at')}",
            ]
        )
        if chat.get("sync_recommended"):
            lines.append(
                f"Known latest #{chat.get('latest_known_message_id')} is missing: sync_chat for this chat if relevant."
            )
        elif chat.get("latest_known_message_available") is None:
            lines.append(
                "Latest-message coverage unknown; do not infer completeness from an empty tail."
            )
        for message in chat.get("messages", []):
            sender = (
                "Me →"
                if message.get("is_outgoing")
                else f"← {message.get('sender_name') or 'Unknown'}"
            )
            metadata = []
            for field, label in (("reply_to_message_id", "reply_to"), ("thread_id", "thread")):
                if message.get(field) is not None:
                    metadata.append(f"{label}={message[field]}")
            lines.append(
                f"#{message.get('telegram_message_id')} {message.get('sent_at')} {sender} {' '.join(metadata)}".rstrip()
            )
            if message.get("text"):
                suffix = (
                    " [text truncated; get_message for the full text]"
                    if message.get("text_truncated")
                    else ""
                )
                lines.append(f"  {message['text']}{suffix}")
            if message.get("has_media"):
                lines.append(
                    f"  Media: {message.get('media_type')} {message.get('media_file_name') or ''} "
                    f"status={message.get('media_processing_status')}".rstrip()
                )
                if message.get("content_preview"):
                    suffix = (
                        " [content truncated; get_message_content for more]"
                        if message.get("content_truncated")
                        else ""
                    )
                    lines.append(f"  Extracted content: {message['content_preview']}{suffix}")
                else:
                    lines.append(
                        "  No extracted content yet; use get_message_content if this attachment matters."
                    )
            if message.get("telegram_message_url"):
                lines.append(f"  {message['telegram_message_url']}")
        if chat.get("has_more_messages"):
            lines.append(
                f"Older context: get_chat_messages before={chat.get('next_message_cursor')}"
            )
    if result.get("has_more"):
        lines.append(
            f"\nMore chats: get_inbox cursor={result.get('next_cursor')} with the same filters."
        )
    else:
        lines.append(
            "\nEnd of chats in this scope; each returned conversation is still only a tail."
        )
    return [TextContent(type="text", text="\n".join(lines))]


def format_chat_list(result: dict, listener_active: bool = False) -> list[TextContent]:
    """Format chat list for display."""
    if not result.get("chats"):
        return [TextContent(type="text", text="No chats synced yet.")]

    chats = result.get("chats", [])
    total = result.get("total", len(chats))
    lines = [f"Showing {len(chats)} of {total} total chats:\n"]
    lines.append(
        "OLD_SYNC is an old timestamp, not proof of missing messages. Use get_inbox for conversation tails and coverage.\n"
    )
    for chat in chats:
        synced = chat.get("total_messages_synced", 0)
        title = chat.get("title", "Unknown")
        chat_type = chat.get("chat_type", "unknown")
        chat_id = chat.get("id", "unknown")
        username_ref = _format_username_ref(chat.get("username"))
        private_link = _private_chat_link(
            chat_type,
            chat.get("telegram_chat_id"),
            chat.get("last_message_id"),
        )
        last_sync = chat.get("last_sync_at")
        freshness = _freshness_label(last_sync, listener_active)
        sync_info = f"Last synced: {_format_date(last_sync)}" if last_sync else "Never synced"
        unread_count = int(chat.get("unread_count") or 0)
        last_message_id = chat.get("last_message_id")
        last_message_sender = " ".join(
            str(chat.get("last_message_sender_name") or "Unknown sender").split()
        )
        last_message_text = " ".join(str(chat.get("last_message_text") or "").split())
        if len(last_message_text) > 240:
            last_message_text = f"{last_message_text[:237]}..."
        last_activity = chat.get("last_activity_at")
        details = [
            f"ID: {chat_id}",
            f"Unread: {unread_count}",
            f"Messages synced: {synced}",
            sync_info,
        ]
        if username_ref:
            details.append(f"Username: {username_ref}")
        elif private_link:
            details.append(f"Open: {private_link}")
        latest = ""
        if last_message_id is not None:
            activity = f" at {_format_date(last_activity)}" if last_activity else ""
            preview = last_message_text or "[no text preview]"
            latest = (
                f"\n  Last message #{last_message_id}{activity} from "
                f"{last_message_sender}: {preview}"
            )
        lines.append(f"- {title} ({chat_type}) [{freshness}]\n  {' | '.join(details)}{latest}\n")

    has_more = result.get("has_more", False)
    next_cursor = result.get("next_cursor")
    if has_more and next_cursor:
        lines.append(
            f'\n--- More chats available. Use cursor="{next_cursor}" to load the next page ---'
        )

    return [TextContent(type="text", text="\n".join(lines))]


def format_chat_search_results(result: dict) -> list[TextContent]:
    """Format chat search results for display."""
    chats = result.get("chats", [])
    query = result.get("query", "")
    total = result.get("total", len(chats))

    if not chats:
        return [TextContent(type="text", text=f'No chats found for query: "{query}"')]

    lines = [f'Found {total} chats for query: "{query}"\n']
    for chat in chats:
        title = chat.get("title", "Unknown")
        chat_type = chat.get("chat_type", "unknown")
        chat_id = chat.get("id", "unknown")
        synced = chat.get("total_messages_synced", 0)
        username_ref = _format_username_ref(chat.get("username"))
        private_link = _private_chat_link(
            chat_type,
            chat.get("telegram_chat_id"),
            chat.get("last_message_id"),
        )
        details = [
            f"ID: {chat_id}",
            f"Messages synced: {synced}",
        ]
        if username_ref:
            details.append(f"Username: {username_ref}")
        elif private_link:
            details.append(f"Open: {private_link}")
        lines.append(f"- {title} ({chat_type})\n  {' | '.join(details)}\n")

    return [TextContent(type="text", text="\n".join(lines))]


def format_find_chats_results(result: dict[str, Any]) -> list[TextContent]:
    """Format topic-aware unique-chat search with explicit coverage."""
    chats = result.get("chats", [])
    query = result.get("query", "")
    coverage = result.get("coverage", {})
    hits = int(coverage.get("message_hits_scanned") or 0)
    pages = int(coverage.get("pages_scanned") or 0)
    coverage_label = "complete" if coverage.get("complete") else "capped/partial"
    page_label = "page" if pages == 1 else "pages"
    coverage_line = (
        f"Coverage: {hits} message hits scanned across {pages} {page_label}; "
        f"{coverage_label}. {int(coverage.get('inventory_chats_scanned') or 0)} "
        "chat titles scanned."
    )

    if not chats:
        return [
            TextContent(
                type="text",
                text=f'No chats found for query: "{query}"\n{coverage_line}',
            )
        ]

    lines = [
        f'Found {result.get("total", len(chats))} unique chats for query: "{query}"',
        coverage_line,
        "",
    ]
    for chat in chats:
        title = chat.get("title", "Unknown")
        chat_type = chat.get("chat_type", "unknown")
        details = [
            f"ID: {chat.get('id', 'unknown')}",
            f"Message hits: {chat.get('message_hit_count', 0)}",
        ]
        username_ref = _format_username_ref(chat.get("username"))
        if username_ref:
            details.append(f"Username: {username_ref}")
        else:
            private_link = _private_chat_link(
                chat_type,
                chat.get("telegram_chat_id"),
                chat.get("last_message_id"),
            )
            if private_link:
                details.append(f"Open: {private_link}")
        lines.append(f"- {title} ({chat_type})\n  {' | '.join(details)}")
        for message in chat.get("representative_messages", []):
            text_preview = _format_media_label(message)
            message_url = message.get("telegram_message_url") or _private_chat_link(
                message.get("chat_type"),
                message.get("chat_telegram_id"),
                message.get("telegram_message_id"),
            )
            suffix = f" | Open: {message_url}" if message_url else ""
            lines.append(f"  · {_format_date(message.get('sent_at'))}: {text_preview}{suffix}")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


def format_chat_messages(
    result: dict,
    base_url: str = "https://telegram.waiwai.is",
) -> list:
    """Format paginated chat messages for display."""
    messages = result.get("messages", [])
    if not messages:
        return [TextContent(type="text", text="No messages found in this chat.")]

    total_synced = result.get("total_messages_synced")
    last_sync = result.get("last_sync_at")

    lines = [f"Messages ({len(messages)} returned):\n"]
    resources: list[ResourceLink] = []
    for msg in messages:
        sender = msg.get("sender_name") or ("You" if msg.get("is_outgoing") else "Unknown")
        text = _format_media_label(msg)
        sent_at = _format_date(msg.get("sent_at"))
        msg_id = msg.get("telegram_message_id", "")
        links: list[str] = []
        if msg.get("telegram_message_url"):
            links.append(f"Open: {msg['telegram_message_url']}")
        media_url = _absolute_url(msg.get("media_download_url"), base_url)
        if media_url:
            links.append(f"Download media: {media_url}")
            resource = _media_resource_link(msg, base_url)
            if resource:
                resources.append(resource)
        link_suffix = f"\n  - {' | '.join(links)}" if links else ""
        lines.append(f"[{sent_at}] {sender} (msg#{msg_id}): {text}{link_suffix}\n")

    has_more = result.get("has_more", False)
    next_cursor = result.get("next_cursor")
    if has_more and next_cursor:
        lines.append(
            f'\n--- More messages available. Use before="{next_cursor}" to load the next page ---'
        )
    else:
        # End of synced messages — provide context
        sync_parts = []
        if total_synced is not None:
            sync_parts.append(f"{total_synced} messages synced in total")
        if last_sync:
            sync_parts.append(f"last synced: {_format_date(last_sync)}")
        sync_info = " (" + ", ".join(sync_parts) + ")" if sync_parts else ""
        lines.append(
            f"\n--- End of synced messages{sync_info}. "
            f"There may be older messages in Telegram not yet downloaded. "
            f"Use sync_chat with message_limit=0 to download full history. ---"
        )

    return [TextContent(type="text", text="\n".join(lines)), *resources]


def format_digest(result: dict) -> list[TextContent]:
    """Format digest for display."""
    lines = [
        f"Daily Digest for {result.get('digest_date', 'unknown')}\n",
        "=" * 40 + "\n",
        result.get("content", "No digest content available."),
        "\n" + "=" * 40,
        f"\nStats: {result.get('summary_stats', {})}",
    ]
    return [TextContent(type="text", text="\n".join(lines))]


def format_data_status(settings: dict, chats_result: dict) -> list[TextContent]:
    """Format compact data status overview for display."""
    listener_active = settings.get("listener_active", False)
    realtime_sync = settings.get("realtime_sync_enabled", False)

    lines = [
        "Telegram Data Status\n",
        "=" * 40 + "\n",
        f"Real-time sync enabled: {realtime_sync}",
        f"Listener active: {listener_active}\n",
    ]

    chats = chats_result.get("chats", [])
    total_chats = chats_result.get("total", len(chats))
    if not chats:
        lines.append("No chats synced yet. Use sync_chat to download messages.")
    else:
        # Summary stats
        total_messages = sum(c.get("total_messages_synced", 0) for c in chats)
        type_counts: dict[str, int] = {}
        freshness_counts: dict[str, int] = {"LIVE": 0, "FRESH": 0, "STALE": 0, "NEVER": 0}
        for chat in chats:
            ct = chat.get("chat_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
            freshness = _freshness_label(chat.get("last_sync_at"), listener_active)
            freshness_counts[freshness] = freshness_counts.get(freshness, 0) + 1

        lines.append(f"Total chats: {total_chats}")
        lines.append(f"Total messages synced: {total_messages:,}\n")

        # Type breakdown
        type_parts = [f"{v} {k}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]
        lines.append(f"Chat types: {', '.join(type_parts)}\n")

        # Freshness distribution
        freshness_parts = []
        for label in ("LIVE", "FRESH", "STALE", "NEVER"):
            count = freshness_counts.get(label, 0)
            if count > 0:
                freshness_parts.append(f"{count} {label}")
        lines.append(f"Data freshness: {', '.join(freshness_parts)}\n")

        # Top 10 most recently active chats
        sorted_chats = sorted(
            chats,
            key=lambda c: c.get("last_sync_at") or "",
            reverse=True,
        )
        top_chats = sorted_chats[:10]
        lines.append(f"Top {len(top_chats)} most recently active chats:\n")
        for chat in top_chats:
            title = chat.get("title", "Unknown")
            chat_type = chat.get("chat_type", "unknown")
            chat_id = chat.get("id", "unknown")
            synced = chat.get("total_messages_synced", 0)
            freshness = _freshness_label(chat.get("last_sync_at"), listener_active)
            lines.append(
                f"- {title} ({chat_type}) [{freshness}] | ID: {chat_id} | Messages: {synced}"
            )

        lines.append(
            "\nUse list_chats to browse all chats, or search_messages to find specific content."
        )

    return [TextContent(type="text", text="\n".join(lines))]


def format_send_result(result: dict, action: str) -> list[TextContent]:
    """Format send/reply result for display."""
    msg_id = result.get("telegram_message_id", "unknown")
    chat_id = result.get("chat_id", "unknown")
    lines = [
        f"{action} successfully.\n",
        f"Message ID: {msg_id}",
        f"Chat ID: {chat_id}",
    ]
    if chat_id == "me":
        lines.append("Destination: Saved Messages (Избранное) of the connected account")
    file_name = result.get("file_name")
    if file_name:
        lines.append(f"File: {file_name}")
    text = result.get("text")
    if text:
        lines.append(f"Text: {text[:200]}")
    return [TextContent(type="text", text="\n".join(lines))]


def format_draft_result(result: dict) -> list[TextContent]:
    """Format a draft mutation while making the no-send guarantee explicit."""
    if result.get("saved") is not True:
        raise ValueError("Backend did not confirm that the draft was saved")
    if result.get("sent") is not False:
        raise ValueError("Backend did not confirm the draft-only no-send invariant")
    chat_id = result.get("chat_id", "unknown")
    text = result.get("text", "")
    lines = [
        "Draft saved successfully.",
        "No Telegram message was sent.",
        f"Chat ID: {chat_id}",
    ]
    if text:
        lines.append(f"Draft: {text}")
    return [TextContent(type="text", text="\n".join(lines))]


def format_clear_draft_result(result: dict) -> list[TextContent]:
    """Format a verified draft deletion with the same no-send guarantee."""
    if result.get("cleared") is not True or result.get("saved") is not True:
        raise ValueError("Backend did not confirm that the draft was cleared")
    if result.get("sent") is not False:
        raise ValueError("Backend did not confirm the draft-only no-send invariant")
    chat_id = result.get("chat_id", "unknown")
    return [
        TextContent(
            type="text",
            text=(
                f"Draft cleared successfully.\nNo Telegram message was sent.\nChat ID: {chat_id}"
            ),
        )
    ]


def format_sync_started(result: dict) -> list[TextContent]:
    """Format sync started response."""
    job_id = result.get("id") or result.get("job_id", "unknown")
    status = result.get("status", "unknown")
    lines = [
        "Sync started successfully.\n",
        f"Job ID: {job_id}\n",
        f"Status: {status}\n",
        f'\nUse get_sync_status with job_id="{job_id}" to check progress.',
    ]
    return [TextContent(type="text", text="\n".join(lines))]


def format_sync_status(result: dict) -> list[TextContent]:
    """Format sync status response."""
    job_id = result.get("job_id", "unknown")
    status = result.get("status", "unknown")
    messages_processed = result.get("messages_processed", 0)
    messages_seen = result.get("messages_seen")
    messages_total = result.get("messages_total")
    progress = result.get("progress_percent")
    error = result.get("error_message")

    lines = [
        f"Sync Job: {job_id}\n",
        f"Status: {status}\n",
    ]
    if messages_seen is not None and messages_total is not None:
        lines.append(
            f"Progress: {messages_seen:,} of {messages_total:,} messages ({progress:.0f}%)\n"
            if progress is not None
            else f"Progress: {messages_seen:,} of {messages_total:,} messages\n"
        )
    elif messages_seen is not None:
        lines.append(f"Progress: {messages_seen:,} messages fetched\n")
    elif progress is not None:
        lines.append(f"Progress: {progress}%\n")
    lines.append(f"Messages saved: {messages_processed:,}\n")
    if error:
        lines.append(f"Error: {error}\n")
    if status == "in_progress":
        lines.append("\nSync is still running. Check again in 10-15 seconds.")
    elif status == "completed":
        lines.append("\nSync completed. You can now read the messages with get_chat_messages.")

    return [TextContent(type="text", text="\n".join(lines))]


def main():
    """Run the MCP server."""
    asyncio.run(run_server())


async def run_server():
    """Run the server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    main()
