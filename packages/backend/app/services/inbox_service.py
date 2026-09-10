"""Read bounded conversation tails in one indexed batch, without Telegram RPCs."""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, tuple_, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cursor import (
    CursorError,
    decode_cursor,
    encode_cursor,
    parse_cursor_datetime,
)
from app.models.chat import ChatType, TelegramChat
from app.models.message import TelegramMessage
from app.services.telegram_links import build_telegram_message_url

DEFAULT_CHAT_TYPES = ["private", "group", "supergroup"]
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=value.tzinfo or UTC).isoformat()


def _message_cursor(row: dict[str, Any]) -> str:
    # Same cursor contract as get_chat_messages/before.
    return encode_cursor(
        {
            "sent_at": _iso(row["sent_at"]),
            "telegram_message_id": row["telegram_message_id"],
            "id": str(row["id"]),
        }
    )


async def get_inbox(
    db: AsyncSession,
    user_id: UUID,
    *,
    chat_ids: list[UUID] | None = None,
    chat_types: list[str] | None = None,
    active_since: datetime | None = None,
    cursor: str | None = None,
    limit: int = 20,
    messages_per_chat: int = 12,
) -> dict[str, Any]:
    types = chat_types if chat_types is not None else DEFAULT_CHAT_TYPES
    scope = {
        "chat_types": types if chat_ids is None else None,
        "active_since": _iso(active_since),
        "chat_ids": [str(chat_id) for chat_id in chat_ids] if chat_ids else None,
    }
    known_message = (
        select(TelegramMessage.id)
        .where(
            TelegramMessage.chat_id == TelegramChat.id,
            TelegramMessage.telegram_message_id == TelegramChat.last_message_id,
        )
        .exists()
    )
    query = select(TelegramChat, known_message).where(TelegramChat.user_id == user_id)
    if chat_ids is not None:
        query = query.where(TelegramChat.id.in_(chat_ids))
        limit = len(chat_ids)
    else:
        query = query.where(TelegramChat.chat_type.in_([ChatType(t) for t in types]))
        if active_since is not None:
            query = query.where(TelegramChat.last_activity_at >= active_since)
    if cursor:
        try:
            c = decode_cursor(cursor)
            if c.get("kind") != "inbox" or c.get("scope") != scope:
                raise CursorError("Cursor belongs to a different inbox scope")
            activity = parse_cursor_datetime(c["activity"])
            chat_id = UUID(c["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CursorError("Invalid inbox cursor or changed filters") from exc
        query = query.where(
            tuple_(
                func.coalesce(TelegramChat.last_activity_at, _EPOCH), TelegramChat.id
            )
            < (activity or _EPOCH, chat_id)
        )
    query = query.order_by(
        TelegramChat.last_activity_at.desc().nulls_last(), TelegramChat.id.desc()
    ).limit(limit + 1)
    chat_rows = (await db.execute(query)).all()
    if chat_ids is not None and len(chat_rows) != len(chat_ids):
        raise ValueError("One or more chats not found")
    has_more = len(chat_rows) > limit
    chat_rows = chat_rows[:limit]
    next_cursor = None
    if has_more:
        last_chat = chat_rows[-1][0]
        next_cursor = encode_cursor(
            {
                "kind": "inbox",
                "scope": scope,
                "activity": _iso(last_chat.last_activity_at),
                "id": str(last_chat.id),
            }
        )

    # Each subquery uses the existing (chat_id, sent_at) index and LIMIT. A window
    # over whole histories would scan millions of rows to return a few messages.
    columns = [
        TelegramMessage.id,
        TelegramMessage.chat_id,
        TelegramMessage.telegram_message_id,
        TelegramMessage.sent_at,
        TelegramMessage.sender_name,
        TelegramMessage.is_outgoing,
        TelegramMessage.has_media,
        TelegramMessage.media_type,
        TelegramMessage.media_file_name,
        TelegramMessage.media_processing_status,
        TelegramMessage.reply_to_message_id,
        TelegramMessage.thread_id,
        func.substr(TelegramMessage.text, 1, 1201).label("text"),
        func.substr(TelegramMessage.content_text, 1, 601).label("content_preview"),
    ]
    tails = [
        select(
            select(*columns)
            .where(
                TelegramMessage.chat_id == chat.id,
                TelegramMessage.deleted_at.is_(None),
            )
            .order_by(
                TelegramMessage.sent_at.desc(),
                TelegramMessage.telegram_message_id.desc(),
                TelegramMessage.id.desc(),
            )
            .limit(messages_per_chat + 1)
            .subquery()
        )
        for chat, _known in chat_rows
    ]
    grouped: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
    if tails:
        rows = (await db.execute(union_all(*tails))).mappings().all()
        for row in rows:
            grouped[row["chat_id"]].append(dict(row))

    conversations = []
    for chat, known in chat_rows:
        rows = sorted(
            grouped[chat.id],
            key=lambda row: (row["sent_at"], row["telegram_message_id"], row["id"]),
            reverse=True,
        )
        more_messages = len(rows) > messages_per_chat
        rows = rows[:messages_per_chat]
        messages = []
        for row in reversed(rows):
            text, content = row["text"], row["content_preview"]
            messages.append(
                {
                    "telegram_message_id": row["telegram_message_id"],
                    "sent_at": _iso(row["sent_at"]),
                    "sender_name": row["sender_name"],
                    "is_outgoing": row["is_outgoing"],
                    "text": text[:1200] if text else None,
                    "text_truncated": bool(text and len(text) > 1200),
                    "has_media": row["has_media"],
                    "media_type": row["media_type"],
                    "media_file_name": row["media_file_name"],
                    "media_processing_status": row["media_processing_status"],
                    "content_preview": content[:600] if content else None,
                    "content_truncated": bool(content and len(content) > 600),
                    "reply_to_message_id": row["reply_to_message_id"],
                    "thread_id": row["thread_id"],
                    "telegram_message_url": build_telegram_message_url(
                        chat_type=chat.chat_type,
                        telegram_chat_id=chat.telegram_chat_id,
                        username=chat.username,
                        message_id=row["telegram_message_id"],
                    ),
                }
            )
        conversations.append(
            {
                "chat_id": str(chat.id),
                "title": chat.title,
                "chat_type": chat.chat_type.value,
                "username": chat.username,
                "unread_count": int(chat.unread_count or 0),
                "last_activity_at": _iso(chat.last_activity_at),
                "last_sync_at": _iso(chat.last_sync_at),
                "latest_known_message_id": chat.last_message_id,
                "latest_known_message_available": bool(known)
                if chat.last_message_id is not None
                else None,
                "sync_recommended": chat.last_message_id is not None and not known,
                "messages": messages,
                "has_more_messages": more_messages,
                "next_message_cursor": _message_cursor(rows[-1])
                if more_messages and rows
                else None,
            }
        )
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "scope": scope,
        "message_order": "oldest_first",
        "conversations": conversations,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }
