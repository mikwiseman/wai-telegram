"""Read-only, bounded link inventory over the existing authenticated message API."""

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from telegram_wai_mcp.client import TelegramAIClient

_URL = re.compile(r"https?://[^\s<>\"'\[\]]+", re.IGNORECASE)


def validate_domains(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError('"domains" must be a list of at most 20 hostnames')
    result = []
    for domain in value:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError('"domains" must contain non-empty hostnames')
        domain = domain.strip().lower().rstrip(".")
        if any(c in domain for c in "/:@?#% ") or not all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", part)
            for part in domain.encode("idna").decode().split(".")
        ):
            raise ValueError('"domains" must contain hostnames, not URLs')
        result.append(domain.encode("idna").decode())
    return list(dict.fromkeys(result))


def _url(value: str, *, from_text: bool = False) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if from_text:
        # Archived Markdown may have no space after a link: [title](url)text.
        # Keep balanced parentheses inside URLs, but stop at the wrapper's close.
        depth = 0
        for index, char in enumerate(value):
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    value = value[:index]
                    break
                depth -= 1
        value = value.rstrip(".,;:!?")
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").encode("idna").decode().lower()
        if parts.scheme.lower() not in {"http", "https"} or not host:
            return None
        # Preserve the query and fragment: they can identify different documents.
        userinfo, separator, authority = parts.netloc.rpartition("@")
        netloc = userinfo + separator + authority.lower()
        return urlunsplit(
            (parts.scheme.lower(), netloc, parts.path, parts.query, parts.fragment)
        ), host
    except (ValueError, UnicodeError):
        return None


def message_urls(message: dict[str, Any]) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for field in ("visible_urls", "hidden_urls"):
        for raw in message.get(field) or []:
            if item := _url(raw):
                found[item[0]] = item[1]
    for raw in _URL.findall(message.get("text") or ""):
        if item := _url(raw, from_text=True):
            found[item[0]] = item[1]
    return list(found.items())


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


async def collect_links(
    api: TelegramAIClient,
    *,
    chat_id: str,
    before: str | None,
    max_pages: int,
    domains: list[str],
    date_from: datetime | None,
    date_to: datetime | None,
) -> dict[str, Any]:
    start = _utc(date_from) if date_from else None
    end = _utc(date_to) if date_to else None
    if start and end and start > end:
        raise ValueError('"date_from" must not be after "date_to"')
    cursor = before
    seen_cursors = {before} if before else set()
    links: dict[str, dict[str, Any]] = {}
    scanned = 0
    dates = []
    has_more = False
    missing_dates = 0
    for page_number in range(1, max_pages + 1):
        page = await api.get_messages(chat_id=chat_id, limit=500, before=cursor)
        for message in page.get("messages", []):
            scanned += 1
            if message.get("deleted_at"):
                continue
            sent_at = message.get("sent_at")
            try:
                sent = _utc(datetime.fromisoformat(sent_at))
            except (ValueError, TypeError):
                sent = None
            if sent:
                dates.append(sent.isoformat())
            if (start or end) and sent is None:
                missing_dates += 1
                continue
            if sent and ((start and sent < start) or (end and sent > end)):
                continue
            for url, host in message_urls(message):
                if domains and not any(host == d or host.endswith("." + d) for d in domains):
                    continue
                entry = links.setdefault(url, {"url": url, "domain": host, "occurrences": []})
                occurrence = {
                    "telegram_message_id": message.get("telegram_message_id"),
                    "sent_at": sent_at,
                    "telegram_message_url": message.get("telegram_message_url"),
                }
                if occurrence not in entry["occurrences"]:
                    entry["occurrences"].append(occurrence)
        has_more = bool(page.get("has_more"))
        cursor = page.get("next_cursor") if has_more else None
        if not has_more:
            break
        if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
            raise RuntimeError("Backend returned an invalid message cursor sequence")
        seen_cursors.add(cursor)
    return {
        "chat_id": chat_id,
        "links": list(links.values()),
        "total_unique_urls_in_batch": len(links),
        "has_more": has_more,
        "next_cursor": cursor,
        "coverage": {
            "pages_scanned": page_number,
            "messages_scanned": scanned,
            "oldest_scanned_at": min(dates) if dates else None,
            "newest_scanned_at": max(dates) if dates else None,
            "messages_skipped_missing_date": missing_dates,
            "synced_history_exhausted": not has_more,
            "note": "Coverage is the synced archive, not all Telegram history. "
            "Resume with before=next_cursor and the same filters, even when links is empty. "
            "URLs are deduplicated within this batch; merge batches by url. "
            "Hidden and visible HTTP(S) links are included; attachments are listed by get_files.",
        },
    }
