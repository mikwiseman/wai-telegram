"""Find shared files through the conversation around them.

Photos carry no filename, and plenty of documents are sent with no caption, so
searching message text alone never reaches them. What people do remember is the
exchange: "the estimate Andrey sent in March". This locates that exchange with
the existing hybrid search, then returns the files sitting next to it, so the
caller gets links to download rather than a transcription bill.

This is the expensive half of the file surface and it earns that cost only when
the file is remembered through what was said. An ask shaped like attributes - a
chat, a week, a person, a file type - belongs in file_browse_service, which
answers it off the message index with no embedding at all.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import Row, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import TelegramChat
from app.models.message import TelegramMessage
from app.schemas.search import SearchRequest
from app.services.file_browse_service import (
    DEFAULT_CONTEXT_WINDOW,
    MAX_CONTEXT_WINDOW,
    apply_file_filters,
    build_file_entry,
    context_snippet,
    file_select,
)
from app.services.search_service import semantic_search


@dataclass(frozen=True)
class _Hit:
    chat_id: UUID
    telegram_message_id: int
    text: str | None
    rank: int


async def find_files(
    db: AsyncSession,
    user_id: UUID,
    *,
    query: str,
    media_types: list[str],
    chat_ids: list[UUID] | None = None,
    chat_types: list[str] | None = None,
    date_from: Any = None,
    date_to: Any = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    limit: int = 20,
    **filters: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return the files sitting next to the messages that matched the query.

    Yields the entries for this page, how many messages were scanned to find
    them, and how many files matched in total, so the caller can say plainly
    that it truncated rather than offer a cursor relevance cannot honour.
    """
    window = max(0, min(int(context_window), MAX_CONTEXT_WINDOW))

    search = await semantic_search(
        db,
        user_id,
        SearchRequest(
            query=query,
            chat_ids=chat_ids,
            chat_types=chat_types,
            date_from=date_from,
            date_to=date_to,
            # Widen the conversational net: each hit may contribute several files,
            # and files far from any hit are worse answers than a shallower sweep.
            limit=min(100, max(limit * 3, 30)),
        ),
    )
    hits = [
        _Hit(item.chat_id, item.telegram_message_id, item.text, rank)
        for rank, item in enumerate(search.results)
    ]
    if not hits:
        return [], 0, 0

    conditions = [
        and_(
            TelegramMessage.chat_id == hit.chat_id,
            TelegramMessage.telegram_message_id.between(
                hit.telegram_message_id - window,
                hit.telegram_message_id + window,
            ),
        )
        for hit in hits
    ]
    stmt = apply_file_filters(
        file_select(user_id),
        media_types=media_types,
        chat_ids=chat_ids,
        chat_types=chat_types,
        date_from=date_from,
        date_to=date_to,
        **filters,
    ).where(or_(*conditions))
    rows = (await db.execute(stmt)).all()

    # A file that matched directly often has no caption of its own, and a photo
    # never has a filename either, so the result would carry nothing a person can
    # judge. Collect the talking around each window and quote the nearest line.
    context_rows = (
        await db.execute(
            select(
                TelegramMessage.chat_id,
                TelegramMessage.telegram_message_id,
                TelegramMessage.text,
            )
            .join(TelegramChat, TelegramChat.id == TelegramMessage.chat_id)
            .where(
                TelegramChat.user_id == user_id,
                TelegramChat.indexing_enabled.is_(True),
                TelegramMessage.text.isnot(None),
                TelegramMessage.text != "",
                TelegramMessage.deleted_at.is_(None),
                or_(*conditions),
            )
        )
    ).all()
    context_by_chat: dict[UUID, list[tuple[int, str]]] = {}
    for chat_id, telegram_message_id, text in context_rows:
        context_by_chat.setdefault(chat_id, []).append((telegram_message_id, text))

    def nearest_context(chat_id: UUID, telegram_message_id: int) -> str | None:
        candidates = context_by_chat.get(chat_id) or []
        if not candidates:
            return None
        tid, text = min(candidates, key=lambda row: abs(row[0] - telegram_message_id))
        if abs(tid - telegram_message_id) > window:
            return None
        return context_snippet(text)

    best: dict[UUID, tuple[tuple[int, int], dict[str, Any]]] = {}
    for row in rows:
        nearest = _nearest_hit(hits, row, window)
        if nearest is None:
            continue
        distance = abs(nearest.telegram_message_id - row.telegram_message_id)
        entry = build_file_entry(
            user_id,
            row,
            matched_because=(
                context_snippet(nearest.text)
                or context_snippet(row.text)
                or nearest_context(row.chat_id, row.telegram_message_id)
            ),
            matched_distance=distance,
        )
        rank = (distance, nearest.rank)
        previous = best.get(row.message_id)
        if previous is None or rank < previous[0]:
            best[row.message_id] = (rank, entry)

    ordered = [
        entry for _rank, entry in sorted(best.values(), key=lambda item: item[0])
    ]
    return ordered[:limit], len(hits), len(ordered)


def _nearest_hit(hits: list[_Hit], row: Row, window: int) -> _Hit | None:
    return min(
        (
            hit
            for hit in hits
            if hit.chat_id == row.chat_id
            and abs(hit.telegram_message_id - row.telegram_message_id) <= window
        ),
        key=lambda hit: (
            abs(hit.telegram_message_id - row.telegram_message_id),
            hit.rank,
        ),
        default=None,
    )
