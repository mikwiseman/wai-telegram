import asyncio
import hashlib
import ipaddress
import logging
import socket
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal, NoReturn
from urllib.parse import urljoin, urlparse
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telethon.errors import (
    ChatWriteForbiddenError,
    FloodWaitError,
    RPCError,
    UserBannedInChannelError,
)
from telethon.tl.functions.messages import SaveDraftRequest
from telethon.tl.types import (
    Channel,
    Chat,
    InputPeerChannel,
    InputPeerChat,
    InputPeerSelf,
    InputPeerUser,
    User as TelegramUser,
)
from telethon.utils import get_peer_id

from app.core.config import get_settings
from app.models.chat import ChatType, TelegramChat
from app.services.telegram_client import get_client
from app.services.telegram_client import (
    SESSION_EXPIRED_MESSAGE,
    TelegramSessionUnauthorizedError,
    invalidate_client_authorization,
    is_session_authorization_error,
)

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_DOWNLOAD_REDIRECTS = 5
SendTarget = UUID | Literal["me"]


class TelegramEntityNotFoundError(ValueError):
    """A stored dialog can no longer be resolved by the owner session."""


def _validate_url(url: str) -> None:
    """Validate URL to prevent SSRF attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a hostname")

    # Resolve hostname and check for private IPs
    try:
        results = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from e

    for _family, _type, _proto, _canonname, sockaddr in results:
        # Strip IPv6 zone ID (e.g. "fe80::1%eth0" → "fe80::1")
        addr = sockaddr[0].split("%")[0]
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                "URLs pointing to private/internal networks are not allowed"
            )


def _sanitize_file_name(name: str) -> str:
    """Sanitize file name to remove dangerous characters."""
    # Take only the last path component
    name = Path(name).name
    # Remove dangerous characters
    for ch in "\x00/\\|<>:\"'":
        name = name.replace(ch, "_")
    # Remove leading dashes/dots (prevent command option injection / hidden files)
    name = name.lstrip("-.")
    # Limit length
    if len(name) > 200:
        suffix = Path(name).suffix[:20]
        name = name[: 200 - len(suffix)] + suffix
    return name or "file"


async def _get_chat(db: AsyncSession, user_id: UUID, chat_id: UUID) -> TelegramChat:
    """Look up a Telegram chat by our internal UUID."""
    result = await db.execute(
        select(TelegramChat).where(
            TelegramChat.id == chat_id,
            TelegramChat.user_id == user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise ValueError(f"Chat {chat_id} not found")
    return chat


def _entity_chat_type(entity: object) -> ChatType | None:
    if isinstance(entity, TelegramUser):
        return ChatType.PRIVATE
    if isinstance(entity, Chat):
        return ChatType.GROUP
    if isinstance(entity, Channel):
        return ChatType.SUPERGROUP if entity.megagroup else ChatType.CHANNEL
    return None


def _dialog_matches_chat(dialog: object, chat: TelegramChat) -> bool:
    entity = getattr(dialog, "entity", None)
    if entity is None:
        return False

    entity_ids = {
        getattr(entity, "id", None),
        get_peer_id(entity),
    }
    if chat.telegram_chat_id not in entity_ids:
        return False

    return _entity_chat_type(entity) == chat.chat_type


def _raw_chat_id(chat: TelegramChat) -> int:
    chat_id = chat.telegram_chat_id
    if chat.chat_type == ChatType.GROUP:
        return abs(chat_id)
    if chat.chat_type in (ChatType.SUPERGROUP, ChatType.CHANNEL) and chat_id < 0:
        return abs(chat_id) - 10**12
    return chat_id


def _stored_input_peer(chat: TelegramChat):
    raw_chat_id = _raw_chat_id(chat)
    if chat.chat_type == ChatType.GROUP:
        return InputPeerChat(raw_chat_id)
    if chat.access_hash is None:
        return None
    if chat.chat_type == ChatType.PRIVATE:
        return InputPeerUser(raw_chat_id, chat.access_hash)
    if chat.chat_type in (ChatType.SUPERGROUP, ChatType.CHANNEL):
        return InputPeerChannel(raw_chat_id, chat.access_hash)
    return None


async def _remember_access_hash(
    db: AsyncSession, chat: TelegramChat, entity: object | None
) -> None:
    access_hash = getattr(entity, "access_hash", None)
    if access_hash is not None and chat.access_hash != access_hash:
        chat.access_hash = access_hash
        await db.flush()


async def _resolve_chat_entity(
    client,
    db: AsyncSession,
    chat: TelegramChat,
):
    """Resolve a sendable Telethon entity for a stored chat.

    Prefer a stored InputPeer when we already know access_hash. Fall back to
    dialog warm-up for legacy rows that predate access_hash persistence.
    """
    stored_peer = _stored_input_peer(chat)
    if stored_peer is not None:
        return stored_peer

    try:
        entity = await client.get_input_entity(chat.telegram_chat_id)
        await _remember_access_hash(db, chat, entity)
        return entity
    except ValueError:
        pass

    normalized_username = (chat.username or "").strip().removeprefix("@")

    async for dialog in client.iter_dialogs():
        if _dialog_matches_chat(dialog, chat):
            entity = getattr(dialog, "entity", None)
            await _remember_access_hash(db, chat, entity)
            return getattr(dialog, "input_entity", entity)

        if normalized_username:
            entity = getattr(dialog, "entity", None)
            if (
                entity is not None
                and getattr(entity, "username", None) == normalized_username
            ):
                await _remember_access_hash(db, chat, entity)
                return getattr(dialog, "input_entity", entity)

    if normalized_username:
        try:
            entity = await client.get_input_entity(normalized_username)
            await _remember_access_hash(db, chat, entity)
            return entity
        except (RPCError, ValueError) as e:
            if is_session_authorization_error(e):
                raise
            pass

    hint = chat.title or normalized_username or str(chat.telegram_chat_id)
    raise TelegramEntityNotFoundError(
        "Could not resolve Telegram entity for chat "
        f"'{hint}'. Re-sync chats or open the dialog in Telegram, then try again."
    )


def _handle_telethon_error(e: Exception) -> NoReturn:
    """Convert Telethon exceptions to user-friendly ValueErrors."""
    if isinstance(e, FloodWaitError):
        raise ValueError(f"Telegram rate limit: please wait {e.seconds} seconds") from e
    if isinstance(e, ChatWriteForbiddenError):
        raise ValueError("You don't have permission to write in this chat") from e
    if isinstance(e, UserBannedInChannelError):
        raise ValueError("You are banned from writing in this channel") from e
    raise ValueError(f"Telegram error: {e}") from e


async def send_message(
    db: AsyncSession,
    user_id: UUID,
    chat_id: SendTarget,
    text: str,
) -> dict:
    """Send a text message to a Telegram chat via user's Telethon client."""
    if chat_id == "me" and (
        not text.strip() or len(text.encode("utf-16-le")) // 2 > 4096
    ):
        raise ValueError(
            "Saved Messages text must contain 1–4096 characters; upload longer text as a file"
        )
    chat = None if chat_id == "me" else await _get_chat(db, user_id, chat_id)
    client = await get_client(user_id, db)
    try:
        entity = (
            InputPeerSelf()
            if chat_id == "me"
            else await _resolve_chat_entity(client, db, chat)
        )
        options = {"parse_mode": None} if chat_id == "me" else {}
        result = await client.send_message(entity, text, **options)
        return {
            "telegram_message_id": result.id,
            "chat_id": str(chat_id),
            "text": text,
        }
    except (
        FloodWaitError,
        ChatWriteForbiddenError,
        UserBannedInChannelError,
        RPCError,
        ConnectionError,
        OSError,
    ) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()


async def save_draft(
    db: AsyncSession,
    user_id: UUID,
    chat_id: UUID,
    text: str,
) -> dict:
    """Save a server-synced Telegram draft without sending a message."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Draft text must not be empty")

    chat = await _get_chat(db, user_id, chat_id)
    client = await get_client(user_id, db)
    try:
        entity = await _resolve_chat_entity(client, db, chat)
        saved = await client(SaveDraftRequest(peer=entity, message=text))
        if not saved:
            raise ValueError("Telegram did not confirm that the draft was saved")
        return {
            "chat_id": str(chat_id),
            "text": text,
            "saved": True,
            "sent": False,
            "replaces_existing_draft": True,
        }
    except (
        FloodWaitError,
        ChatWriteForbiddenError,
        UserBannedInChannelError,
        RPCError,
        ConnectionError,
        OSError,
    ) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()


async def list_drafts(db: AsyncSession, user_id: UUID) -> dict:
    """Return every current server-synced Telegram draft for the owner."""
    chats = (
        (await db.execute(select(TelegramChat).where(TelegramChat.user_id == user_id)))
        .scalars()
        .all()
    )
    chats_by_peer_id = {chat.telegram_chat_id: chat for chat in chats}

    client = await get_client(user_id, db)
    try:
        live_drafts = await client.get_drafts()
        drafts = []
        for draft in live_drafts:
            if draft.is_empty:
                continue
            peer_id = get_peer_id(draft.entity)
            chat = chats_by_peer_id.get(peer_id)
            drafts.append(
                {
                    "chat_id": str(chat.id) if chat else None,
                    "telegram_chat_id": peer_id,
                    "chat_type": chat.chat_type.value if chat else None,
                    "title": chat.title if chat else None,
                    "username": chat.username if chat else None,
                    "text": draft.text,
                    "date": draft.date.isoformat() if draft.date else None,
                    "reply_to_message_id": draft.reply_to_msg_id,
                    "link_preview": draft.link_preview,
                }
            )
        drafts.sort(key=lambda item: item["date"] or "", reverse=True)
        return {
            "drafts": drafts,
            "count": len(drafts),
            "unmatched_count": sum(item["chat_id"] is None for item in drafts),
        }
    except (RPCError, ConnectionError, OSError) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()


async def clear_draft(
    db: AsyncSession,
    user_id: UUID,
    chat_id: UUID,
) -> dict:
    """Clear one server-synced Telegram draft and verify it stayed unsent."""
    chat = await _get_chat(db, user_id, chat_id)
    client = await get_client(user_id, db)
    try:
        entity = await _resolve_chat_entity(client, db, chat)
        cleared = await client(SaveDraftRequest(peer=entity, message=""))
        if not cleared:
            raise ValueError("Telegram did not confirm that the draft was cleared")
        current = await client.get_drafts(entity)
        if not current.is_empty:
            raise ValueError("Telegram still reports a draft after clearing it")
        return {
            "chat_id": str(chat_id),
            "cleared": True,
            "saved": True,
            "sent": False,
        }
    except (
        FloodWaitError,
        ChatWriteForbiddenError,
        UserBannedInChannelError,
        RPCError,
        ConnectionError,
        OSError,
    ) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()


async def send_file(
    db: AsyncSession,
    user_id: UUID,
    chat_id: SendTarget,
    file_url: str,
    caption: str | None = None,
    file_name: str | None = None,
) -> dict:
    """Download a file from URL and send it to a Telegram chat."""
    await asyncio.to_thread(_validate_url, file_url)
    chat = None if chat_id == "me" else await _get_chat(db, user_id, chat_id)
    settings = get_settings()

    if not file_name:
        path = urlparse(file_url).path
        file_name = Path(path).name or "file"
    file_name = _sanitize_file_name(file_name)
    with _outbound_directory() as temp_dir:
        temp_path = Path(temp_dir) / file_name
        timeout = httpx.Timeout(
            connect=30.0,
            read=settings.media_download_stall_timeout_seconds,
            write=30.0,
            pool=30.0,
        )
        current_url = file_url
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as http:
            for redirect_count in range(_MAX_DOWNLOAD_REDIRECTS + 1):
                async with http.stream("GET", current_url) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Download redirect is missing Location")
                        if redirect_count == _MAX_DOWNLOAD_REDIRECTS:
                            raise ValueError("Too many download redirects")
                        current_url = urljoin(current_url, location)
                        await asyncio.to_thread(_validate_url, current_url)
                        continue

                    response.raise_for_status()
                    with temp_path.open("wb") as output:
                        async for chunk in response.aiter_bytes(
                            chunk_size=settings.media_download_chunk_bytes
                        ):
                            output.write(chunk)
                    break

        return await _send_file_path(db, user_id, chat_id, temp_path, caption, chat)


def _outbound_directory():
    settings = get_settings()
    work_parent = None
    if settings.environment == "production":
        work_root = settings.media_root / "outbound-work"
        work_root.mkdir(parents=True, exist_ok=True)
        work_parent = str(work_root)
    return tempfile.TemporaryDirectory(prefix="wai-outbound-", dir=work_parent)


async def _send_file_path(
    db: AsyncSession,
    user_id: UUID,
    chat_id: SendTarget,
    path: Path,
    caption: str | None,
    chat: TelegramChat | None = None,
) -> dict:
    if chat_id == "me" and caption and len(caption.encode("utf-16-le")) // 2 > 1024:
        raise ValueError("Saved Messages file caption must be at most 1024 characters")
    client = await get_client(user_id, db)
    try:
        entity = (
            InputPeerSelf()
            if chat_id == "me"
            else await _resolve_chat_entity(client, db, chat)
        )
        # Saved files are originals: no photo compression or implicit Markdown parsing.
        options = (
            {"force_document": True, "parse_mode": None} if chat_id == "me" else {}
        )
        result = await client.send_file(
            entity, str(path), caption=caption, file_name=path.name, **options
        )
        return {
            "telegram_message_id": result.id,
            "chat_id": str(chat_id),
            "file_name": path.name,
        }
    except (RPCError, ConnectionError, OSError) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()


async def upload_to_saved_messages(
    db: AsyncSession,
    user_id: UUID,
    chunks: AsyncIterator[bytes],
    file_name: str,
    caption: str | None = None,
) -> dict:
    """Stream a local original to private temporary storage, then to the owner's self peer."""
    file_name = _sanitize_file_name(file_name)
    with _outbound_directory() as temp_dir:
        path = Path(temp_dir) / file_name
        digest = hashlib.sha256()
        size = 0
        with path.open("wb") as output:
            async for chunk in chunks:
                await asyncio.to_thread(output.write, chunk)
                digest.update(chunk)
                size += len(chunk)
        if not size:
            raise ValueError("Cannot send an empty file")
        result = await _send_file_path(db, user_id, "me", path, caption)
        return {**result, "file_size": size, "sha256": digest.hexdigest()}


async def reply_to_message(
    db: AsyncSession,
    user_id: UUID,
    chat_id: UUID,
    telegram_message_id: int,
    text: str,
) -> dict:
    """Reply to a specific message in a Telegram chat."""
    chat = await _get_chat(db, user_id, chat_id)
    client = await get_client(user_id, db)
    try:
        entity = await _resolve_chat_entity(client, db, chat)
        result = await client.send_message(
            entity,
            text,
            reply_to=telegram_message_id,
        )
        return {
            "telegram_message_id": result.id,
            "chat_id": str(chat_id),
            "text": text,
        }
    except (
        FloodWaitError,
        ChatWriteForbiddenError,
        UserBannedInChannelError,
        RPCError,
        ConnectionError,
        OSError,
    ) as e:
        if is_session_authorization_error(e):
            await invalidate_client_authorization(client, user_id, e)
            raise TelegramSessionUnauthorizedError(SESSION_EXPIRED_MESSAGE) from e
        _handle_telethon_error(e)
    finally:
        await client.disconnect()
