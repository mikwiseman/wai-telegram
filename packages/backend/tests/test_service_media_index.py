from unittest.mock import AsyncMock, patch

import pytest

from app.services.embedding_service import prepare_media_content_index


async def test_media_index_embeds_summary_and_every_content_chunk():
    calls: list[list[str]] = []

    async def fake_embeddings(texts):
        calls.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]

    with (
        patch(
            "app.services.embedding_service.generate_embeddings",
            new_callable=AsyncMock,
            side_effect=fake_embeddings,
        ),
        patch("app.services.embedding_service.settings") as settings,
    ):
        settings.media_embedding_chunk_chars = 12
        settings.media_embedding_chunk_overlap_chars = 2
        settings.memory_batch_enabled = False
        settings.embedding_batch_size = 100
        prepared = await prepare_media_content_index(
            caption="Caption",
            summary="Summary",
            content_text="one two three four five six",
        )

    assert prepared.message_text == "Caption\n\nSummary"
    assert len(prepared.chunks) >= 2
    assert "\n".join(chunk.text for chunk in prepared.chunks)
    assert calls[0][0] == "Caption\n\nSummary"


async def test_media_index_rejects_partial_embedding_response():
    with (
        patch(
            "app.services.embedding_service.generate_embeddings",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.services.embedding_service.settings") as settings,
    ):
        settings.media_embedding_chunk_chars = 100
        settings.media_embedding_chunk_overlap_chars = 10
        settings.memory_batch_enabled = False
        settings.embedding_batch_size = 100
        with pytest.raises(RuntimeError, match="expected 2 embeddings"):
            await prepare_media_content_index(
                caption=None,
                summary="Summary",
                content_text="Full content",
            )
