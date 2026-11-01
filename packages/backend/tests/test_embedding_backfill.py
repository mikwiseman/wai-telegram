from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.embedding_service import embed_messages


async def test_embedding_reads_and_flushes_only_bounded_batches():
    ids = [uuid4() for _ in range(5)]
    messages = [
        MagicMock(id=i, text=f"message {n}", embedding=None) for n, i in enumerate(ids)
    ]
    results = []
    for group in (messages[:2], messages[2:4], messages[4:]):
        result = MagicMock()
        result.scalars.return_value.all.return_value = group
        results.append(result)
    db = AsyncMock()
    db.execute.side_effect = results
    with (
        patch("app.services.embedding_service.settings.embedding_batch_size", 2),
        patch(
            "app.services.embedding_service.generate_embeddings",
            new_callable=AsyncMock,
            side_effect=[[[1.0], [2.0]], [[3.0], [4.0]], [[5.0]]],
        ),
    ):
        count = await embed_messages(db, ids)
    assert count == 5
    assert db.execute.await_count == 3
    assert db.flush.await_count == 3


async def test_partial_provider_response_never_assigns_incomplete_batch():
    messages = [MagicMock(id=uuid4(), text="school", embedding=None) for _ in range(2)]
    result = MagicMock()
    result.scalars.return_value.all.return_value = messages
    db = AsyncMock()
    db.execute.return_value = result
    with patch(
        "app.services.embedding_service.generate_embeddings",
        new_callable=AsyncMock,
        return_value=[[1.0]],
    ):
        assert await embed_messages(db, [m.id for m in messages]) == 0
    assert all(m.embedding is None for m in messages)
