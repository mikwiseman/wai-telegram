import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.cursor import CursorError, decode_cursor, encode_cursor
from app.models.chat import ChatType, TelegramChat
from app.models.message import TelegramMessage
from app.schemas.search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.services.embedding_service import (
    generate_query_embedding,
    generate_memory_query_embedding,
)
from app.services.telegram_links import (
    build_media_download_url,
    build_telegram_message_url,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchServiceError(RuntimeError):
    """Raised when the requested search mode cannot be completed."""


def _empty_response(query: str) -> SearchResponse:
    return SearchResponse(results=[], query=query, total=0)


def _search_offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        value = int(decode_cursor(cursor).get("offset", 0))
    except (CursorError, TypeError, ValueError) as exc:
        raise SearchServiceError("Invalid search cursor") from exc
    if value < 0:
        raise SearchServiceError("Invalid search cursor")
    return value


def _literal_like_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _base_where_clauses(
    user_id: UUID,
    request: SearchRequest,
) -> tuple[list[str], dict[str, object]]:
    where_clauses = [
        "c.user_id = :user_id",
        "c.indexing_enabled IS TRUE",
        "m.deleted_at IS NULL",
    ]
    offset = _search_offset(request.cursor)
    params: dict[str, object] = {
        "user_id": str(user_id),
        "query": request.query.strip(),
        "query_pattern": _literal_like_pattern(request.query.strip()),
        "limit": request.limit + 1,
        "offset": offset,
        "candidate_limit": max(100, (offset + request.limit + 1) * 10),
        "rrf_k": 60.0,
    }
    if request.chat_ids:
        where_clauses.append("m.chat_id = ANY(CAST(:chat_ids AS uuid[]))")
        params["chat_ids"] = [str(chat_id) for chat_id in request.chat_ids]
    if request.chat_types:
        where_clauses.append("c.chat_type::text = ANY(CAST(:chat_types AS text[]))")
        params["chat_types"] = [chat_type.name for chat_type in request.chat_types]
    if request.date_from:
        where_clauses.append("m.sent_at >= :date_from")
        params["date_from"] = request.date_from
    if request.date_to:
        where_clauses.append("m.sent_at <= :date_to")
        params["date_to"] = request.date_to
    return where_clauses, params


def _normalize_chat_type(value: object) -> ChatType | None:
    if value is None or isinstance(value, ChatType):
        return value
    if isinstance(value, str):
        try:
            return ChatType(value.strip().lower())
        except ValueError:
            logger.warning(
                "Unknown chat_type in search result", extra={"chat_type": value}
            )
    return None


def _rows_to_response(
    rows: list,
    query: str,
    user_id: UUID,
    *,
    limit: int,
    offset: int,
) -> SearchResponse:
    has_more = len(rows) > limit
    rows = rows[:limit]
    results = []
    for row in rows:
        chat_type = _normalize_chat_type(row.chat_type)
        media_cached = bool(getattr(row, "media_cached", False))
        results.append(
            SearchResultItem(
                id=row.id,
                chat_id=row.chat_id,
                chat_title=row.chat_title,
                chat_type=chat_type,
                chat_telegram_id=row.chat_telegram_id,
                chat_username=row.chat_username,
                telegram_message_id=row.telegram_message_id,
                text=row.text,
                sender_name=row.sender_name,
                is_outgoing=row.is_outgoing,
                sent_at=row.sent_at,
                similarity=float(row.similarity),
                has_media=row.has_media,
                media_type=row.media_type,
                content_summary=getattr(row, "content_summary", None),
                content_preview=getattr(row, "content_preview", None),
                media_processing_status=getattr(row, "media_processing_status", None),
                media_file_name=getattr(row, "media_file_name", None),
                media_mime_type=getattr(row, "media_mime_type", None),
                media_file_size=getattr(row, "media_file_size", None),
                visible_urls=getattr(row, "visible_urls", None) or [],
                hidden_urls=getattr(row, "hidden_urls", None) or [],
                deleted_at=getattr(row, "deleted_at", None),
                transcribed_at=row.transcribed_at,
                telegram_message_url=build_telegram_message_url(
                    chat_type=chat_type,
                    telegram_chat_id=row.chat_telegram_id,
                    username=row.chat_username,
                    message_id=row.telegram_message_id,
                ),
                media_download_url=(
                    build_media_download_url(
                        base_path=(
                            f"/api/v1/chats/{row.chat_id}/messages/"
                            f"{row.telegram_message_id}/media"
                        ),
                        user_id=user_id,
                        chat_id=row.chat_id,
                        telegram_message_id=row.telegram_message_id,
                    )
                    if media_cached
                    else None
                ),
            )
        )
    return SearchResponse(
        results=results,
        query=query,
        total=len(results),
        has_more=has_more,
        next_cursor=(
            encode_cursor({"offset": offset + len(results)}) if has_more else None
        ),
    )


def _hybrid_search_sql(dimensions: int, where_sql: str, *, memory: bool = False):
    """Build simple-FTS + trigram + pgvector retrieval fused with RRF.

    The trigram operator is applied to media_file_name only. Running it against
    searchable_metadata as well cost 1.5s of every hybrid search at the 0.2
    threshold this query sets, and returned nothing: metadata is long text, so a
    natural-language query never reaches the similarity cut-off. Filenames are
    short and fuzzy-matching them is both cheap (~40ms) and the point. Metadata
    is still matched by full-text search and by ILIKE, which cost ~35ms each.
    """
    if memory:
        vector_sql = f"""
        memory_candidates AS (
            SELECT p.message_ids,p.text,
                1-(p.embedding <=> CAST(:embedding AS halfvec(3072))) AS vector_score
            FROM memory_passages p
            JOIN memory_days d ON d.chat_id=p.chat_id AND d.day=p.day
            JOIN telegram_chats c ON c.id=p.chat_id
            JOIN users u ON u.id=c.user_id
            WHERE p.embedding IS NOT NULL AND p.user_id=c.user_id
                AND c.indexing_enabled AND u.is_active
                AND p.revision=d.revision AND d.prepared_revision=d.revision
                AND EXISTS (
                    SELECT 1 FROM telegram_messages m
                    WHERE m.id=ANY(p.message_ids) AND m.chat_id=p.chat_id AND {where_sql}
                )
            ORDER BY p.embedding <=> CAST(:embedding AS halfvec(3072))
            LIMIT :candidate_limit
        ),
        vector_raw AS (
            SELECT chosen.id AS message_id,p.vector_score,left(p.text,1200) AS matched_content
            FROM memory_candidates p
            CROSS JOIN LATERAL (
                SELECT m.id FROM telegram_messages m
                JOIN telegram_chats c ON c.id=m.chat_id
                WHERE m.id=ANY(p.message_ids) AND {where_sql}
                ORDER BY ts_rank_cd(m.search_vector,websearch_to_tsquery('simple',:query),32) DESC,
                    m.sent_at DESC,m.telegram_message_id DESC
                LIMIT 1
            ) chosen
        ),
        vector_chunks AS (SELECT * FROM vector_raw WHERE false),
        """
    else:
        vector_sql = f"""
        vector_raw AS (
            SELECT
                m.id AS message_id,
                1 - (m.embedding <=> cast(:embedding AS halfvec({dimensions})))
                    AS vector_score,
                NULL::text AS matched_content
            FROM telegram_messages m
            JOIN telegram_chats c ON c.id = m.chat_id
            WHERE {where_sql} AND m.embedding IS NOT NULL
            ORDER BY m.embedding <=> cast(:embedding AS halfvec({dimensions}))
            LIMIT :candidate_limit
        ),
        vector_chunks AS (
            SELECT
                m.id AS message_id,
                1 - (mc.embedding <=> cast(:embedding AS halfvec({dimensions})))
                    AS vector_score,
                left(mc.text, 1200) AS matched_content
            FROM message_content_chunks mc
            JOIN telegram_messages m ON m.id = mc.message_id
            JOIN telegram_chats c ON c.id = m.chat_id
            WHERE {where_sql} AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> cast(:embedding AS halfvec({dimensions}))
            LIMIT :candidate_limit
        ),
        """
    return text(f"""
        WITH search_query AS (
            SELECT websearch_to_tsquery('simple', :query) AS tsq
        ),
        lexical_raw AS (
            SELECT
                m.id AS message_id,
                greatest(
                    ts_rank_cd(m.search_vector, q.tsq, 32),
                    similarity(m.media_file_name, :query),
                    CASE WHEN m.media_file_name ILIKE :query_pattern
                        THEN 1.5 ELSE 0 END,
                    CASE WHEN m.searchable_metadata ILIKE :query_pattern
                        THEN 1.5 ELSE 0 END
                ) AS lexical_score,
                left(m.content_text, 1200) AS matched_content
            FROM telegram_messages m
            JOIN telegram_chats c ON c.id = m.chat_id
            CROSS JOIN search_query q
            WHERE {where_sql}
              AND (
                m.search_vector @@ q.tsq
                OR m.media_file_name ILIKE :query_pattern
                OR m.searchable_metadata ILIKE :query_pattern
                OR m.media_file_name % :query
              )
            UNION ALL
            SELECT
                m.id AS message_id,
                ts_rank_cd(mc.search_vector, q.tsq, 32) AS lexical_score,
                left(mc.text, 1200) AS matched_content
            FROM message_content_chunks mc
            JOIN telegram_messages m ON m.id = mc.message_id
            JOIN telegram_chats c ON c.id = m.chat_id
            CROSS JOIN search_query q
            WHERE {where_sql} AND mc.search_vector @@ q.tsq
        ),
        lexical_best AS (
            SELECT DISTINCT ON (message_id)
                message_id, lexical_score, matched_content
            FROM lexical_raw
            ORDER BY message_id, lexical_score DESC
        ),
        lexical_ranked AS (
            SELECT
                message_id,
                row_number() OVER (
                    ORDER BY lexical_score DESC, message_id DESC
                ) AS lexical_rank,
                matched_content
            FROM lexical_best
            ORDER BY lexical_score DESC, message_id DESC
            LIMIT :candidate_limit
        ),
        {vector_sql}
        vector_best AS (
            SELECT DISTINCT ON (message_id)
                message_id, vector_score, matched_content
            FROM (
                SELECT * FROM vector_raw
                UNION ALL
                SELECT * FROM vector_chunks
            ) candidates
            ORDER BY message_id, vector_score DESC
        ),
        vector_ranked AS (
            SELECT
                message_id,
                row_number() OVER (
                    ORDER BY vector_score DESC, message_id DESC
                ) AS vector_rank,
                matched_content
            FROM vector_best
            ORDER BY vector_score DESC, message_id DESC
            LIMIT :candidate_limit
        ),
        fused AS (
            SELECT
                coalesce(l.message_id, v.message_id) AS message_id,
                (
                    coalesce(1.0 / (:rrf_k + l.lexical_rank), 0) +
                    coalesce(1.0 / (:rrf_k + v.vector_rank), 0)
                ) / (2.0 / (:rrf_k + 1.0)) AS similarity,
                coalesce(l.matched_content, v.matched_content) AS matched_content
            FROM lexical_ranked l
            FULL OUTER JOIN vector_ranked v ON v.message_id = l.message_id
        )
        SELECT
            m.id,
            m.chat_id,
            c.title AS chat_title,
            c.chat_type AS chat_type,
            c.telegram_chat_id AS chat_telegram_id,
            c.username AS chat_username,
            m.telegram_message_id,
            m.text,
            m.sender_name,
            m.is_outgoing,
            m.sent_at,
            fused.similarity,
            m.has_media,
            m.media_type,
            m.content_summary,
            coalesce(fused.matched_content, left(m.content_text, 1200))
                AS content_preview,
            m.media_processing_status,
            m.media_file_name,
            m.media_mime_type,
            m.media_file_size,
            m.visible_urls,
            m.hidden_urls,
            m.deleted_at,
            m.transcribed_at,
            (mo.relative_path IS NOT NULL AND mo.sha256 IS NOT NULL) AS media_cached
        FROM fused
        JOIN telegram_messages m ON m.id = fused.message_id
        JOIN telegram_chats c ON c.id = m.chat_id
        LEFT JOIN media_objects mo ON mo.message_id = m.id
        ORDER BY similarity DESC, m.sent_at DESC, m.telegram_message_id DESC
        LIMIT :limit
        OFFSET :offset
    """)


def _exact_search_sql(where_sql: str):
    """Build indexed candidate retrieval with literal phrase verification."""
    return text(f"""
        WITH search_query AS (
            SELECT plainto_tsquery('simple', :query) AS tsq
        ),
        exact_raw AS (
            SELECT
                m.id AS message_id,
                greatest(
                    CASE WHEN coalesce(m.text, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 1.0 ELSE 0 END,
                    CASE WHEN coalesce(m.content_summary, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 0.95 ELSE 0 END,
                    CASE WHEN coalesce(m.content_text, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 0.9 ELSE 0 END,
                    CASE WHEN coalesce(m.sender_name, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 0.85 ELSE 0 END,
                    CASE WHEN coalesce(m.media_file_name, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 0.8 ELSE 0 END,
                    CASE WHEN coalesce(m.searchable_metadata, '') ILIKE :query_pattern ESCAPE '\\'
                        THEN 0.75 ELSE 0 END
                ) AS exact_score,
                left(m.content_text, 1200) AS matched_content
            FROM telegram_messages m
            JOIN telegram_chats c ON c.id = m.chat_id
            CROSS JOIN search_query q
            WHERE {where_sql}
              AND m.search_vector @@ q.tsq
              AND (
                coalesce(m.text, '') ILIKE :query_pattern ESCAPE '\\'
                OR coalesce(m.content_summary, '') ILIKE :query_pattern ESCAPE '\\'
                OR coalesce(m.content_text, '') ILIKE :query_pattern ESCAPE '\\'
                OR coalesce(m.sender_name, '') ILIKE :query_pattern ESCAPE '\\'
                OR coalesce(m.media_file_name, '') ILIKE :query_pattern ESCAPE '\\'
                OR coalesce(m.searchable_metadata, '') ILIKE :query_pattern ESCAPE '\\'
              )
            UNION ALL
            SELECT
                m.id AS message_id,
                0.9 AS exact_score,
                left(mc.text, 1200) AS matched_content
            FROM message_content_chunks mc
            JOIN telegram_messages m ON m.id = mc.message_id
            JOIN telegram_chats c ON c.id = m.chat_id
            CROSS JOIN search_query q
            WHERE {where_sql}
              AND mc.search_vector @@ q.tsq
              AND mc.text ILIKE :query_pattern ESCAPE '\\'
        ),
        exact_best AS (
            SELECT DISTINCT ON (message_id)
                message_id, exact_score, matched_content
            FROM exact_raw
            ORDER BY message_id, exact_score DESC
        )
        SELECT
            m.id,
            m.chat_id,
            c.title AS chat_title,
            c.chat_type AS chat_type,
            c.telegram_chat_id AS chat_telegram_id,
            c.username AS chat_username,
            m.telegram_message_id,
            m.text,
            m.sender_name,
            m.is_outgoing,
            m.sent_at,
            exact_best.exact_score AS similarity,
            m.has_media,
            m.media_type,
            m.content_summary,
            coalesce(exact_best.matched_content, left(m.content_text, 1200))
                AS content_preview,
            m.media_processing_status,
            m.media_file_name,
            m.media_mime_type,
            m.media_file_size,
            m.visible_urls,
            m.hidden_urls,
            m.deleted_at,
            m.transcribed_at,
            (mo.relative_path IS NOT NULL AND mo.sha256 IS NOT NULL) AS media_cached
        FROM exact_best
        JOIN telegram_messages m ON m.id = exact_best.message_id
        JOIN telegram_chats c ON c.id = m.chat_id
        LEFT JOIN media_objects mo ON mo.message_id = m.id
        ORDER BY similarity DESC, m.sent_at DESC, m.telegram_message_id DESC
        LIMIT :limit
        OFFSET :offset
    """)


async def semantic_search(
    db: AsyncSession,
    user_id: UUID,
    request: SearchRequest,
) -> SearchResponse:
    """Run the explicitly requested exact or hybrid retrieval mode."""
    normalized_query = request.query.strip()
    if not normalized_query:
        logger.info(
            "Search skipped for blank query",
            extra={"user_id": str(user_id), "query_length": 0},
        )
        return _empty_response(request.query)
    where_clauses, params = _base_where_clauses(user_id, request)
    where_sql = " AND ".join(where_clauses)

    if request.mode == SearchMode.EXACT:
        try:
            result = await db.execute(_exact_search_sql(where_sql), params)
        except Exception as exc:
            logger.exception(
                "Exact search query failed",
                extra={"user_id": str(user_id), "query_length": len(normalized_query)},
            )
            raise SearchServiceError("Exact search is temporarily unavailable") from exc
        return _rows_to_response(
            result.fetchall(),
            request.query,
            user_id,
            limit=request.limit,
            offset=int(params["offset"]),
        )

    try:
        query_embedding = await (
            generate_memory_query_embedding(normalized_query)
            if settings.memory_search_enabled
            else generate_query_embedding(normalized_query)
        )
    except Exception as exc:
        logger.exception(
            "Hybrid search embedding generation failed",
            extra={"user_id": str(user_id), "query_length": len(normalized_query)},
        )
        raise SearchServiceError(
            "Hybrid search is unavailable because query embedding failed"
        ) from exc
    if not query_embedding:
        raise SearchServiceError(
            "Hybrid search is unavailable because query embedding was empty"
        )

    params["embedding"] = "[" + ",".join(str(value) for value in query_embedding) + "]"
    sql = _hybrid_search_sql(
        settings.embedding_dimensions,
        where_sql,
        memory=settings.memory_search_enabled,
    )
    try:
        await db.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        await db.execute(text("SET LOCAL pg_trgm.similarity_threshold = 0.2"))
        result = await db.execute(sql, params)
    except Exception as exc:
        logger.exception(
            "Hybrid lexical/vector query failed",
            extra={"user_id": str(user_id), "query_length": len(normalized_query)},
        )
        raise SearchServiceError("Hybrid search is temporarily unavailable") from exc
    return _rows_to_response(
        result.fetchall(),
        request.query,
        user_id,
        limit=request.limit,
        offset=int(params["offset"]),
    )


async def get_recent_messages(
    db: AsyncSession,
    user_id: UUID,
    chat_id: UUID | None = None,
    hours: int = 24,
    limit: int = 100,
) -> list[TelegramMessage]:
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    query = (
        select(TelegramMessage)
        .join(TelegramChat)
        .where(
            TelegramChat.user_id == user_id,
            TelegramMessage.sent_at >= cutoff,
        )
    )
    if chat_id:
        query = query.where(TelegramMessage.chat_id == chat_id)
    result = await db.execute(
        query.order_by(TelegramMessage.sent_at.desc()).limit(limit)
    )
    return list(result.scalars().all())
