import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import asyncio
from collections import OrderedDict

from openai import AsyncOpenAI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.message import TelegramMessage

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class PreparedContentChunk:
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class PreparedMediaContentIndex:
    message_text: str
    message_embedding: list[float]
    chunks: list[PreparedContentChunk]


_client: AsyncOpenAI | None = None
_client_lock = asyncio.Lock()


async def get_openai_client() -> AsyncOpenAI:
    """Return the shared OpenAI client, building it once.

    A fresh AsyncOpenAI per call meant a fresh TCP connection and TLS handshake
    for every embedding, and the discarded clients were never closed. Query
    embedding dominated search latency because of it - about two seconds against
    the quarter second the same request takes over a warm connection.
    """
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def close_openai_client() -> None:
    """Release the shared client, for shutdown and for tests."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    if not texts:
        return []

    client = await get_openai_client()
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


async def prepare_media_content_index(
    *,
    caption: str | None,
    summary: str,
    content_text: str | None,
) -> PreparedMediaContentIndex:
    """Build a complete media index or fail before mutating database state."""
    from app.services.media_content_service import chunk_text

    message_text = "\n\n".join(
        part.strip()
        for part in (caption, summary)
        if isinstance(part, str) and part.strip()
    )
    if not message_text:
        raise RuntimeError("Media summary is required for indexing")

    chunks = chunk_text(
        content_text.strip() if content_text else "",
        max_chars=settings.media_embedding_chunk_chars,
        overlap_chars=settings.media_embedding_chunk_overlap_chars,
    )
    inputs = [message_text, *chunks]
    embeddings: list[list[float]] = []
    for start in range(0, len(inputs), settings.embedding_batch_size):
        batch = inputs[start : start + settings.embedding_batch_size]
        batch_embeddings = await generate_embeddings(batch)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "OpenAI returned "
                f"{len(batch_embeddings)} embeddings; expected {len(batch)} embeddings"
            )
        embeddings.extend(batch_embeddings)

    return PreparedMediaContentIndex(
        message_text=message_text,
        message_embedding=embeddings[0],
        chunks=[
            PreparedContentChunk(text=text, embedding=embedding)
            for text, embedding in zip(chunks, embeddings[1:], strict=True)
        ],
    )


async def embed_messages(db: AsyncSession, message_ids: list[UUID]) -> int:
    """Bound both database reads and provider calls; failed rows remain retryable."""
    embedded_count = 0
    size = settings.embedding_batch_size
    for start in range(0, len(message_ids), size):
        result = await db.execute(
            select(TelegramMessage)
            .where(
                TelegramMessage.id.in_(message_ids[start : start + size]),
                TelegramMessage.text.isnot(None),
                TelegramMessage.embedding.is_(None),
            )
            .with_for_update(skip_locked=True)
        )
        messages = [m for m in result.scalars().all() if m.text and m.text.strip()]
        if not messages:
            continue
        try:
            embeddings = await generate_embeddings([m.text[:8000] for m in messages])
            if len(embeddings) != len(messages):
                raise ValueError("Embedding provider returned an incomplete batch")
        except Exception:
            logger.warning("Embedding batch failed; rows remain pending for retry")
            continue
        for message, embedding in zip(messages, embeddings, strict=True):
            message.embedding = embedding
            message.embedded_at = datetime.now(UTC)
        await db.flush()
        embedded_count += len(messages)
    return embedded_count


async def embed_unembedded_messages(
    db: AsyncSession, user_id: UUID, limit: int = 1000
) -> int:
    """Find and embed messages without embeddings."""
    from app.models.chat import TelegramChat

    # Get messages without embeddings for this user
    result = await db.execute(
        select(TelegramMessage.id)
        .join(TelegramChat)
        .where(
            TelegramChat.user_id == user_id,
            TelegramChat.indexing_enabled.is_(True),
            func.length(func.trim(TelegramMessage.text, " \t\n\r\v\f\u00a0")) > 0,
            TelegramMessage.embedding.is_(None),
        )
        .limit(limit)
    )
    message_ids = [row[0] for row in result.all()]

    return await embed_messages(db, message_ids)


_QUERY_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_QUERY_CACHE_MAX = 512


def clear_query_embedding_cache() -> None:
    """Drop cached query vectors. Used by tests and after a model change."""
    _QUERY_CACHE.clear()


async def generate_query_embedding(query: str) -> list[float]:
    """Embed a search query, reusing the vector for a query already seen.

    Every hybrid search pays a round trip to the embeddings API - about 145ms on
    a warm connection - and the same phrasing recurs constantly: an agent
    refining a search, a person retrying, a saved query. The text fully
    determines the vector, so caching it is exact rather than approximate.
    """
    key = query.strip()
    if not key:
        return []
    cached = _QUERY_CACHE.get(key)
    if cached is not None:
        _QUERY_CACHE.move_to_end(key)
        return cached
    embeddings = await generate_embeddings([key])
    vector = embeddings[0] if embeddings else []
    if vector:
        _QUERY_CACHE[key] = vector
        if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)
    return vector
