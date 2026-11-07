"""Batch failure boundaries and durable source lifecycle."""

import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.services import memory_batch_service as service
from app.services.memory_passages import DIMENSIONS, MODEL


@asynccontextmanager
async def mock_db(db):
    yield db


def client_with_output(lines):
    client = MagicMock()
    response = MagicMock()

    async def stream():
        for line in lines:
            yield json.dumps(line)

    response.iter_lines = stream
    client.files.with_streaming_response.content.return_value.__aenter__ = AsyncMock(
        return_value=response
    )
    return client


def output(identifier):
    return {
        "custom_id": identifier,
        "response": {
            "status_code": 200,
            "body": {
                "model": MODEL,
                "data": [{"index": 0, "embedding": [0.01] * DIMENSIONS}],
                "usage": {"total_tokens": 10},
            },
        },
    }


async def test_revoked_source_is_not_uploaded():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalar_one.return_value = 0
    client = MagicMock()
    with (
        patch.object(service, "get_db_context", lambda: mock_db(db)),
        patch.object(service, "_release", new_callable=AsyncMock) as release,
    ):
        await service._submit(client, {"id": uuid4(), "passages": [{"id": "x"}]})
    client.with_options.return_value.files.create.assert_not_called()
    assert release.call_args.kwargs == {"count_failure": False}


@pytest.mark.parametrize(
    "code,status", [("insufficient_quota", "blocked"), (None, "unknown")]
)
async def test_quota_and_uncertain_submission_are_not_blindly_retried(code, status):
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalar_one.return_value = 1
    client = MagicMock()
    exc = RuntimeError("provider unavailable")
    exc.code = code
    exc.status_code = 429 if code else None
    client.with_options.return_value.files.create = AsyncMock(side_effect=exc)
    with (
        patch.object(service, "get_db_context", lambda: mock_db(db)),
        patch.object(service, "_batch_update", new_callable=AsyncMock) as update,
    ):
        assert not await service._submit(
            client, {"id": uuid4(), "passages": [{"id": "x", "text": "source"}]}
        )
    assert update.call_args.kwargs["status"] == status
    client.with_options.assert_called_once_with(max_retries=0)


async def test_partial_results_retry_only_missing_passages():
    db = AsyncMock()
    client = client_with_output([output("a")])
    with (
        patch.object(service, "get_db_context", lambda: mock_db(db)),
        patch.object(service, "indexing_has_capacity", return_value=True),
        patch.object(service, "_batch_update", new_callable=AsyncMock) as update,
    ):
        await service._apply_output(
            client,
            {"id": uuid4(), "passage_ids": ["a", "b"]},
            SimpleNamespace(output_file_id="result"),
        )
    assert update.call_args.kwargs["failed"] == 1
    assert update.call_args.kwargs["actual_tokens"] == 10
    assert "embedding IS NULL" in str(db.execute.call_args_list[-1].args[0])


async def test_duplicate_output_is_not_marked_complete():
    with patch.object(service, "_batch_update", new_callable=AsyncMock) as update:
        with pytest.raises(ValueError, match="Duplicate"):
            await service._apply_output(
                client_with_output([output("a"), output("a")]),
                {"id": uuid4(), "passage_ids": ["a"]},
                SimpleNamespace(output_file_id="result"),
            )
    update.assert_not_called()


async def test_final_small_output_still_checks_disk_capacity():
    with patch.object(service, "indexing_has_capacity", return_value=False):
        with pytest.raises(RuntimeError, match="disk capacity"):
            await service._apply_output(
                client_with_output([output("a")]),
                {"id": uuid4(), "passage_ids": ["a"]},
                SimpleNamespace(output_file_id="result"),
            )


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_CONTRACT_DATABASE_URL"),
    reason="PostgreSQL contract database required",
)
async def test_postgres_memory_edits_deletion_scope_and_idempotent_import():
    from app.models.user import User
    from app.models.chat import TelegramChat, ChatType
    from app.models.message import TelegramMessage

    engine = create_async_engine(os.environ["POSTGRES_CONTRACT_DATABASE_URL"])
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            db = AsyncSession(bind=connection, expire_on_commit=False)
            try:
                user = User(
                    id=uuid4(),
                    email=f"memory-{uuid4()}@example.com",
                    password_hash="unused",
                    is_active=True,
                )
                chat = TelegramChat(
                    id=uuid4(),
                    user_id=user.id,
                    telegram_chat_id=1199,
                    chat_type=ChatType.GROUP,
                    title="Family",
                    indexing_enabled=True,
                )
                db.add(user)
                await db.flush()
                db.add(chat)
                await db.flush()
                msg = TelegramMessage(
                    id=uuid4(),
                    chat_id=chat.id,
                    telegram_message_id=1,
                    sender_name="Alisa",
                    text="The visit is in the morning",
                    sent_at=datetime.now(UTC) - timedelta(days=1),
                )
                db.add(msg)
                await db.flush()
                with (
                    patch.object(service, "get_db_context", lambda: mock_db(db)),
                    patch.object(service, "indexing_has_capacity", return_value=True),
                ):
                    assert await service.prepare_day() == 1
                    assert await service.prepare_day() is None
                    status = await service.memory_status(user.id)
                    assert status["passages"] == 1 and status["embedded"] == 0
                    assert (await service.memory_status(uuid4()))["passages"] == 0
                    claim = await service._claim_batch()
                    old_id = claim["passages"][0]["id"]
                    # A correction arrives while OpenAI is processing the old passage.
                    msg.text = "Correction: the visit is in the evening"
                    await db.flush()
                    assert (await service.memory_status(user.id))["passages"] == 0
                    remote = SimpleNamespace(output_file_id="result")
                    old_batch = {"id": claim["id"], "passage_ids": [old_id]}
                    await service._apply_output(
                        client_with_output([output(old_id)]), old_batch, remote
                    )
                    assert (
                        await db.execute(
                            text(
                                "SELECT count(*) FROM memory_passages WHERE embedding IS NOT NULL"
                            )
                        )
                    ).scalar_one() == 0
                    assert await service.prepare_day() == 1
                    claim = await service._claim_batch()
                    new_id = claim["passages"][0]["id"]
                    assert new_id != old_id
                    batch = {"id": claim["id"], "passage_ids": [new_id]}
                    await service._apply_output(
                        client_with_output([output(new_id)]), batch, remote
                    )
                    await service._apply_output(
                        client_with_output([output(new_id)]), batch, remote
                    )
                    assert (await service.memory_status(user.id))["embedded"] == 1
                    from app.services import search_service
                    from app.schemas.search import SearchRequest

                    with (
                        patch.object(
                            search_service.settings, "memory_search_enabled", True
                        ),
                        patch.object(
                            search_service,
                            "generate_memory_query_embedding",
                            new_callable=AsyncMock,
                            return_value=[0.01] * DIMENSIONS,
                        ),
                    ):
                        found = await search_service.semantic_search(
                            db, user.id, SearchRequest(query="evening visit")
                        )
                        assert any(hit.id == msg.id for hit in found.results)
                        denied = await search_service.semantic_search(
                            db, uuid4(), SearchRequest(query="evening visit")
                        )
                        assert denied.results == []

                    chat.title = "Renamed family"
                    await db.flush()
                    assert (await service.memory_status(user.id))["passages"] == 0
                    assert await service.prepare_day() == 1
                    assert (await service.memory_status(user.id))["passages"] == 1
                    # Revoking access makes current memory disappear immediately.
                    chat.indexing_enabled = False
                    await db.flush()
                    assert (await service.memory_status(user.id))["embedded"] == 0
                    chat.indexing_enabled = True
                    msg.deleted_at = datetime.now(UTC)
                    await db.flush()
                    assert (await service.memory_status(user.id))["embedded"] == 0
                    assert await service.prepare_day() == 0
                    assert (
                        await db.execute(
                            text(
                                "SELECT count(*) FROM memory_passages WHERE chat_id=:chat"
                            ),
                            {"chat": chat.id},
                        )
                    ).scalar_one() == 0
                    # Deleting a chat must not resurrect days through its message trigger.
                    await db.execute(
                        text("DELETE FROM telegram_chats WHERE id=:chat"),
                        {"chat": chat.id},
                    )
                    assert (
                        await db.execute(
                            text(
                                "SELECT count(*) FROM memory_days WHERE chat_id=:chat"
                            ),
                            {"chat": chat.id},
                        )
                    ).scalar_one() == 0
            finally:
                await db.close()
                await transaction.rollback()
    finally:
        await engine.dispose()
