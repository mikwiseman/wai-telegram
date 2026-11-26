import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4


class TestSemanticSearch:
    async def test_exact_mode_skips_embeddings_and_filters_chat_types(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        row = SimpleNamespace(
            id=uuid4(),
            chat_id=uuid4(),
            chat_title="Альфабанк/ обучение",
            chat_type="GROUP",
            chat_telegram_id=-123,
            chat_username=None,
            telegram_message_id=48,
            text="Альфа-Банк вернется с датами академии",
            sender_name="John",
            is_outgoing=False,
            sent_at="2026-07-02T12:00:00Z",
            similarity=1.0,
            has_media=False,
            media_type=None,
            transcribed_at=None,
        )
        mock_result = SimpleNamespace(fetchall=lambda: [row])
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
        ) as mock_embedding:
            result = await semantic_search(
                mock_db,
                test_user.id,
                SearchRequest(
                    query="Альфа-Банк",
                    mode="exact",
                    chat_types=["group", "supergroup"],
                ),
            )

        mock_embedding.assert_not_awaited()
        assert mock_db.execute.await_count == 1
        sql_text = str(mock_db.execute.await_args.args[0])
        params = mock_db.execute.await_args.args[1]
        assert "plainto_tsquery('simple', :query)" in sql_text
        assert "ILIKE :query_pattern" in sql_text
        assert "c.chat_type::text = ANY(CAST(:chat_types AS text[]))" in sql_text
        assert params["chat_types"] == ["GROUP", "SUPERGROUP"]
        assert result.results[0].chat_title == "Альфабанк/ обучение"

    async def test_exact_mode_escapes_like_wildcards(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        mock_result = SimpleNamespace(fetchall=list)
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        await semantic_search(
            mock_db,
            test_user.id,
            SearchRequest(query=r"100%_ready\now", mode="exact"),
        )

        params = mock_db.execute.await_args.args[1]
        assert params["query_pattern"] == r"%100\%\_ready\\now%"

    async def test_blank_query_returns_empty_without_embeddings(
        self, db_session, test_user
    ):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        request = SearchRequest(query="   ")
        with (
            patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
            ) as mock_embedding,
            patch("app.services.search_service.logger.info") as mock_logger_info,
        ):
            result = await semantic_search(db_session, test_user.id, request)

        assert result.results == []
        assert result.total == 0
        mock_embedding.assert_not_awaited()
        mock_logger_info.assert_called_once()
        assert mock_logger_info.call_args.args[0] == "Search skipped for blank query"

    async def test_empty_embedding_is_explicit_failure(self, db_session, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import SearchServiceError, semantic_search

        request = SearchRequest(query="test")
        with (
            patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            with pytest.raises(SearchServiceError, match="embedding was empty"):
                await semantic_search(db_session, test_user.id, request)

    async def test_includes_chat_username_when_available(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        request = SearchRequest(query="test")
        row = SimpleNamespace(
            id=uuid4(),
            chat_id=uuid4(),
            chat_title="Test Chat",
            chat_type="SUPERGROUP",
            chat_telegram_id=-1001234567890,
            chat_username="test_chat",
            telegram_message_id=42,
            text="hello",
            sender_name="John",
            is_outgoing=False,
            sent_at="2026-03-10T12:00:00Z",
            similarity=0.91,
            has_media=False,
            media_type=None,
            transcribed_at=None,
        )
        mock_result = SimpleNamespace(fetchall=lambda: [row])
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3],
        ):
            result = await semantic_search(mock_db, test_user.id, request)

        assert result.total == 1
        assert result.results[0].chat_type == "supergroup"
        assert result.results[0].chat_telegram_id == -1001234567890
        assert result.results[0].chat_username == "test_chat"

    async def test_semantic_search_sql_has_stable_tiebreakers(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        row = SimpleNamespace(
            id=uuid4(),
            chat_id=uuid4(),
            chat_title="Test Chat",
            chat_type="PRIVATE",
            chat_telegram_id=123,
            chat_username=None,
            telegram_message_id=7,
            text="wai message",
            sender_name="John",
            is_outgoing=False,
            sent_at="2026-03-10T12:00:00Z",
            similarity=1.0,
            has_media=False,
            media_type=None,
            transcribed_at=None,
        )
        mock_result = SimpleNamespace(fetchall=lambda: [row])
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3],
        ):
            result = await semantic_search(
                mock_db, test_user.id, SearchRequest(query="wai")
            )

        sql_text = str(mock_db.execute.call_args.args[0])
        assert (
            "ORDER BY similarity DESC, m.sent_at DESC, m.telegram_message_id DESC"
            in sql_text
        )
        assert result.results[0].chat_type == "private"

    async def test_semantic_search_includes_media_summary_and_content_chunks(
        self, test_user
    ):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        row = SimpleNamespace(
            id=uuid4(),
            chat_id=uuid4(),
            chat_title="Test Chat",
            chat_type="PRIVATE",
            chat_telegram_id=123,
            chat_username=None,
            telegram_message_id=7,
            text="caption",
            content_preview="full transcript preview",
            content_summary="meeting summary",
            media_processing_status="ready",
            media_file_name="meeting.mp4",
            media_mime_type="video/mp4",
            media_file_size=123,
            sender_name="John",
            is_outgoing=False,
            sent_at="2026-03-10T12:00:00Z",
            similarity=0.9,
            has_media=True,
            media_type="video",
            transcribed_at="2026-03-10T12:01:00Z",
        )
        mock_result = SimpleNamespace(fetchall=lambda: [row])
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3],
        ):
            result = await semantic_search(
                mock_db, test_user.id, SearchRequest(query="meeting")
            )

        sql_text = str(mock_db.execute.call_args.args[0])
        assert "message_content_chunks" in sql_text
        assert "left(mc.text, 1200) as matched_content" in sql_text.lower()
        assert "content_summary" in sql_text
        assert result.results[0].content_preview == "full transcript preview"
        assert result.results[0].content_summary == "meeting summary"
        assert result.results[0].media_processing_status == "ready"

    async def test_hybrid_sql_uses_simple_fts_trigram_vector_and_rrf(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import semantic_search

        row = SimpleNamespace(
            id=uuid4(),
            chat_id=uuid4(),
            chat_title="Test Chat",
            chat_type="PRIVATE",
            chat_telegram_id=123,
            chat_username=None,
            telegram_message_id=7,
            text="wai message",
            sender_name="John",
            is_outgoing=False,
            sent_at="2026-03-10T12:00:00Z",
            similarity=1.0,
            has_media=False,
            media_type=None,
            transcribed_at=None,
        )
        mock_result = SimpleNamespace(fetchall=lambda: [row])
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1, 0.2, 0.3],
        ):
            result = await semantic_search(
                mock_db, test_user.id, SearchRequest(query="wai")
            )

        sql_text = str(mock_db.execute.call_args.args[0])
        assert "websearch_to_tsquery('simple'" in sql_text
        assert "similarity(m.media_file_name" in sql_text
        assert "m.media_file_name % :query" in sql_text
        # Trigram matching is deliberately confined to filenames: against
        # searchable_metadata it cost 1.5s per search and matched nothing.
        assert "m.searchable_metadata % :query" not in sql_text
        assert "m.searchable_metadata ILIKE :query_pattern" in sql_text
        assert "similarity(m.media_file_name, :query) > 0.2" not in sql_text
        assert "coalesce(m.media_file_name" not in sql_text
        assert str(mock_db.execute.await_args_list[0].args[0]) == (
            "SET LOCAL hnsw.iterative_scan = 'strict_order'"
        )
        assert str(mock_db.execute.await_args_list[1].args[0]) == (
            "SET LOCAL pg_trgm.similarity_threshold = 0.2"
        )
        assert str(mock_db.execute.await_args_list[2].args[0]) == (
            "SET LOCAL jit = off"
        )
        assert "message_content_chunks" in sql_text
        assert "FULL OUTER JOIN vector_ranked" in sql_text
        assert ":rrf_k" in sql_text
        assert result.results[0].chat_type == "private"

    async def test_does_not_silently_degrade_when_embeddings_fail(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import SearchServiceError, semantic_search

        request = SearchRequest(query="test")
        mock_db = AsyncMock()

        with (
            patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
                side_effect=RuntimeError("openai failed"),
            ),
            patch("app.services.search_service.logger.exception"),
        ):
            with pytest.raises(SearchServiceError, match="query embedding failed"):
                await semantic_search(mock_db, test_user.id, request)
        mock_db.execute.assert_not_awaited()

    async def test_does_not_silently_degrade_when_hybrid_query_fails(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import SearchServiceError, semantic_search

        request = SearchRequest(query="test")
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=RuntimeError("vector failed"))

        with (
            patch(
                "app.services.search_service.generate_query_embedding",
                new_callable=AsyncMock,
                return_value=[0.1, 0.2, 0.3],
            ),
            patch("app.services.search_service.logger.exception"),
        ):
            with pytest.raises(SearchServiceError, match="temporarily unavailable"):
                await semantic_search(mock_db, test_user.id, request)

    async def test_invalid_cursor_is_explicit_failure(self, test_user):
        from app.schemas.search import SearchRequest
        from app.services.search_service import SearchServiceError, semantic_search

        request = SearchRequest(query="test", cursor="not-a-cursor")
        mock_db = AsyncMock()

        with patch(
            "app.services.search_service.generate_query_embedding",
            new_callable=AsyncMock,
            return_value=[0.1],
        ):
            with pytest.raises(SearchServiceError, match="Invalid search cursor"):
                await semantic_search(mock_db, test_user.id, request)
