import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.embedding_service import (
    embed_messages,
    generate_embeddings,
    generate_query_embedding,
)


class TestGenerateEmbeddings:
    async def test_empty_input(self):
        result = await generate_embeddings([])
        assert result == []

    async def test_calls_openai(self):
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [mock_item1, mock_item2]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "app.services.embedding_service.get_openai_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await generate_embeddings(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

    async def test_passes_model_and_dimensions_to_openai(self):
        """Verify the correct model/dimensions are forwarded to the API."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch(
                "app.services.embedding_service.get_openai_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.embedding_dimensions = 1536

            await generate_embeddings(["test"])

            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["test"],
                dimensions=1536,
            )

    async def test_single_text(self):
        """Single text input returns a list with one embedding."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5, 0.6])]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch(
            "app.services.embedding_service.get_openai_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await generate_embeddings(["hello"])
            assert result == [[0.5, 0.6]]


@pytest.fixture(autouse=True)
def _clear_query_cache():
    """Query vectors are cached by text, so tests must not inherit each other's."""
    from app.services.embedding_service import clear_query_embedding_cache

    clear_query_embedding_cache()
    yield
    clear_query_embedding_cache()


class TestGenerateQueryEmbedding:
    async def test_generates_single_embedding(self):
        with patch(
            "app.services.embedding_service.generate_embeddings",
            new_callable=AsyncMock,
            return_value=[[0.1, 0.2, 0.3]],
        ):
            result = await generate_query_embedding("test query")
            assert result == [0.1, 0.2, 0.3]

    async def test_empty_result(self):
        with patch(
            "app.services.embedding_service.generate_embeddings",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await generate_query_embedding("test query")
            assert result == []

    async def test_delegates_to_generate_embeddings(self):
        """Verify the query is wrapped in a list and passed to generate_embeddings."""
        mock_gen = AsyncMock(return_value=[[0.9, 0.8]])
        with patch(
            "app.services.embedding_service.generate_embeddings",
            mock_gen,
        ):
            await generate_query_embedding("search term")
            mock_gen.assert_called_once_with(["search term"])


class TestEmbedMessages:
    """Test embed_messages() with mocked DB and OpenAI API."""

    async def test_empty_message_ids_returns_zero(self):
        db = AsyncMock()
        result = await embed_messages(db, [])
        assert result == 0
        db.execute.assert_not_called()

    async def test_no_messages_found_returns_zero(self):
        """When DB returns no matching messages, embed_messages returns 0."""
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=mock_result)

        result = await embed_messages(db, [uuid4()])
        assert result == 0

    async def test_messages_with_no_text_returns_zero(self):
        """Messages that have None or whitespace-only text are skipped."""
        db = AsyncMock()

        msg1 = MagicMock()
        msg1.text = None
        msg1.embedding = None

        msg2 = MagicMock()
        msg2.text = "   "
        msg2.embedding = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg1, msg2]
        db.execute = AsyncMock(return_value=mock_result)

        result = await embed_messages(db, [uuid4(), uuid4()])
        assert result == 0

    async def test_successful_embedding(self):
        """Messages with text get embeddings assigned and count returned."""
        db = AsyncMock()

        msg1 = MagicMock()
        msg1.text = "Hello world"
        msg1.embedding = None
        msg1.embedded_at = None

        msg2 = MagicMock()
        msg2.text = "Goodbye world"
        msg2.embedding = None
        msg2.embedded_at = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg1, msg2]
        db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "app.services.embedding_service.generate_embeddings",
                new_callable=AsyncMock,
                return_value=[[0.1, 0.2], [0.3, 0.4]],
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.memory_batch_enabled = False
            mock_settings.embedding_batch_size = 100

            result = await embed_messages(db, [uuid4(), uuid4()])

        assert result == 2
        assert msg1.embedding == [0.1, 0.2]
        assert msg2.embedding == [0.3, 0.4]
        assert msg1.embedded_at is not None
        assert msg2.embedded_at is not None
        db.flush.assert_awaited_once()

    async def test_truncates_long_messages(self):
        """Messages longer than 8000 chars are truncated before embedding."""
        db = AsyncMock()

        long_text = "A" * 10000
        msg = MagicMock()
        msg.text = long_text
        msg.embedding = None
        msg.embedded_at = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg]
        db.execute = AsyncMock(return_value=mock_result)

        captured_texts = []

        async def capture_generate(texts):
            captured_texts.extend(texts)
            return [[0.1, 0.2]]

        with (
            patch(
                "app.services.embedding_service.generate_embeddings",
                side_effect=capture_generate,
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.memory_batch_enabled = False
            mock_settings.embedding_batch_size = 100

            result = await embed_messages(db, [uuid4()])

        assert result == 1
        assert len(captured_texts[0]) == 8000

    async def test_api_failure_continues_with_next_batch(self):
        """If generate_embeddings raises, that batch is skipped but processing continues."""
        db = AsyncMock()

        msg1 = MagicMock()
        msg1.text = "First"
        msg1.embedding = None
        msg1.embedded_at = None

        msg2 = MagicMock()
        msg2.text = "Second"
        msg2.embedding = None
        msg2.embedded_at = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg1, msg2]
        second_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg1]
        second_result.scalars.return_value.all.return_value = [msg2]
        db.execute = AsyncMock(side_effect=[mock_result, second_result])

        call_count = 0

        async def failing_then_ok(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("OpenAI API error")
            return [[0.5, 0.6]]

        with (
            patch(
                "app.services.embedding_service.generate_embeddings",
                side_effect=failing_then_ok,
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.memory_batch_enabled = False
            mock_settings.embedding_batch_size = 1  # Force one message per batch

            result = await embed_messages(db, [uuid4(), uuid4()])

        # First batch fails, second succeeds
        assert result == 1
        assert msg1.embedding is None  # First message not embedded (batch failed)
        assert msg2.embedding == [0.5, 0.6]
        db.flush.assert_awaited_once()

    async def test_batching_respects_batch_size(self):
        """Messages are batched according to embedding_batch_size setting."""
        db = AsyncMock()

        messages = []
        for i in range(5):
            msg = MagicMock()
            msg.text = f"Message {i}"
            msg.embedding = None
            msg.embedded_at = None
            messages.append(msg)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = messages
        batches = []
        for group in (messages[:2], messages[2:4], messages[4:]):
            result = MagicMock()
            result.scalars.return_value.all.return_value = group
            batches.append(result)
        db.execute = AsyncMock(side_effect=batches)

        call_texts = []

        async def capture_batches(texts):
            call_texts.append(texts)
            return [[0.1] for _ in texts]

        with (
            patch(
                "app.services.embedding_service.generate_embeddings",
                side_effect=capture_batches,
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.memory_batch_enabled = False
            mock_settings.embedding_batch_size = 2

            result = await embed_messages(db, [uuid4() for _ in range(5)])

        assert result == 5
        # 5 messages with batch_size=2 should produce 3 batches: [2, 2, 1]
        assert len(call_texts) == 3
        assert len(call_texts[0]) == 2
        assert len(call_texts[1]) == 2
        assert len(call_texts[2]) == 1

    async def test_mixed_text_and_none_messages(self):
        """Only messages with non-empty text get embedded; None/blank are skipped."""
        db = AsyncMock()

        msg_with_text = MagicMock()
        msg_with_text.text = "Real content"
        msg_with_text.embedding = None
        msg_with_text.embedded_at = None

        msg_none = MagicMock()
        msg_none.text = None
        msg_none.embedding = None

        msg_blank = MagicMock()
        msg_blank.text = "  "
        msg_blank.embedding = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            msg_with_text,
            msg_none,
            msg_blank,
        ]
        db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "app.services.embedding_service.generate_embeddings",
                new_callable=AsyncMock,
                return_value=[[0.1, 0.2]],
            ),
            patch("app.services.embedding_service.settings") as mock_settings,
        ):
            mock_settings.memory_batch_enabled = False
            mock_settings.embedding_batch_size = 100

            result = await embed_messages(db, [uuid4(), uuid4(), uuid4()])

        assert result == 1
        assert msg_with_text.embedding == [0.1, 0.2]
