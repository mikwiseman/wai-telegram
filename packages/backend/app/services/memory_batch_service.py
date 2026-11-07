"""Incremental OpenAI Batch indexing with durable, source-checked receipts."""

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import get_db_context
from app.services.embedding_service import get_openai_client
from app.services.indexing_service import indexing_has_capacity
from app.services.memory_passages import (
    DIMENSIONS,
    MODEL,
    build_passages,
    decode_embedding,
)

logger = logging.getLogger(__name__)


async def seed_days() -> int:
    async with get_db_context() as db:
        result = await db.execute(
            text("""
            INSERT INTO memory_days(chat_id,day)
            SELECT m.chat_id,(m.sent_at AT TIME ZONE 'UTC')::date
            FROM telegram_messages m JOIN telegram_chats c ON c.id=m.chat_id
            JOIN users u ON u.id=c.user_id
            WHERE u.is_active AND c.indexing_enabled
            GROUP BY m.chat_id,(m.sent_at AT TIME ZONE 'UTC')::date
            ON CONFLICT DO NOTHING
        """)
        )
        return result.rowcount


async def prepare_day() -> int | None:
    async with get_db_context() as db:
        day = (
            (
                await db.execute(
                    text("""
            SELECT d.*,c.title,c.user_id FROM memory_days d
            JOIN telegram_chats c ON c.id=d.chat_id JOIN users u ON u.id=c.user_id
            WHERE d.revision<>d.prepared_revision AND c.indexing_enabled AND u.is_active
            ORDER BY d.changed_at,d.chat_id,d.day DESC
            LIMIT 1 FOR UPDATE OF d SKIP LOCKED
        """)
                )
            )
            .mappings()
            .first()
        )
        if day is None:
            return None
        start = datetime.combine(day["day"], datetime.min.time(), tzinfo=UTC)
        rows = (
            (
                await db.execute(
                    text("""
            SELECT id,telegram_message_id,sender_name,sent_at,text,content_text,
                   content_summary,media_file_name,thread_id,deleted_at
            FROM telegram_messages WHERE chat_id=:chat AND sent_at>=:start AND sent_at<:end
            ORDER BY sent_at,telegram_message_id
        """),
                    {
                        "chat": day["chat_id"],
                        "start": start,
                        "end": start + timedelta(days=1),
                    },
                )
            )
            .mappings()
            .all()
        )
        passages = build_passages(day["title"], [dict(row) for row in rows])
        common = {
            "chat": day["chat_id"],
            "day": day["day"],
            "revision": day["revision"],
            "user": day["user_id"],
        }
        await db.execute(
            text("""
            DELETE FROM memory_passages WHERE chat_id=:chat AND day=:day
            AND NOT(id=ANY(CAST(:ids AS text[])))
        """),
            {**common, "ids": [p.id for p in passages]},
        )
        if passages:
            await db.execute(
                text("""
                INSERT INTO memory_passages(id,user_id,chat_id,day,revision,message_ids,text,tokens)
                VALUES(:id,:user,:chat,:day,:revision,CAST(:messages AS uuid[]),:text,:tokens)
                ON CONFLICT(id) DO UPDATE SET revision=EXCLUDED.revision,user_id=EXCLUDED.user_id
            """),
                [
                    {
                        **common,
                        "id": p.id,
                        "messages": p.message_ids,
                        "text": p.text,
                        "tokens": p.tokens,
                    }
                    for p in passages
                ],
            )
        await db.execute(
            text("""
            UPDATE memory_days SET prepared_revision=:revision,prepared_at=now(),source_messages=:sources
            WHERE chat_id=:chat AND day=:day
        """),
            {**common, "sources": len({i for p in passages for i in p.message_ids})},
        )
        return len(passages)


async def memory_status(user_id=None) -> dict:
    scope = " AND c.user_id=:user" if user_id is not None else ""
    params = {"user": user_id} if user_id is not None else {}
    async with get_db_context() as db:
        counts = dict(
            (
                await db.execute(
                    text(
                        """
            SELECT count(*) AS passages,count(*) FILTER(WHERE p.embedding IS NOT NULL) AS embedded,
              coalesce(sum(p.tokens),0) AS tokens,
              count(*) FILTER(WHERE p.failures>=3) AS failed
            FROM memory_passages p JOIN telegram_chats c ON c.id=p.chat_id
            JOIN users u ON u.id=c.user_id JOIN memory_days d ON d.chat_id=p.chat_id AND d.day=p.day
            WHERE c.indexing_enabled AND u.is_active AND p.user_id=c.user_id
              AND p.revision=d.revision AND d.prepared_revision=d.revision
        """
                        + scope
                    ),
                    params,
                )
            )
            .mappings()
            .one()
        )
        counts["dirtyDays"] = (
            await db.execute(
                text(
                    """
            SELECT count(*) FROM memory_days d JOIN telegram_chats c ON c.id=d.chat_id
            JOIN users u ON u.id=c.user_id
            WHERE u.is_active AND c.indexing_enabled AND d.revision<>d.prepared_revision
        """
                    + scope
                ),
                params,
            )
        ).scalar_one()
        counts["batches"] = (
            []
            if user_id is not None
            else [
                dict(row)
                for row in (
                    await db.execute(
                        text("""
            SELECT status,count(*) AS count,sum(actual_tokens) AS billed_tokens
            FROM memory_batches GROUP BY status
        """)
                    )
                ).mappings()
            ]
        )
        counts["model"] = MODEL
        counts["dimensions"] = DIMENSIONS
        return counts


async def _batch_update(batch_id, **values):
    allowed = {
        "status",
        "provider_id",
        "input_file_id",
        "output_file_id",
        "actual_tokens",
        "applied",
        "failed",
        "error_code",
    }
    if not values or not set(values) <= allowed:
        raise ValueError("Unsupported batch fields")
    assignments = ",".join(f"{key}=:{key}" for key in values)
    async with get_db_context() as db:
        await db.execute(
            text(
                f"UPDATE memory_batches SET {assignments},updated_at=now() WHERE id=:id"
            ),
            {"id": batch_id, **values},
        )


async def _release(batch, code: str, *, count_failure=True):
    async with get_db_context() as db:
        await db.execute(
            text("""
            UPDATE memory_passages SET batch_id=NULL,failures=failures+:increment,error_code=:error
            WHERE batch_id=:id AND embedding IS NULL
        """),
            {"id": batch["id"], "increment": int(count_failure), "error": code[:100]},
        )
    await _batch_update(batch["id"], status="failed", error_code=code[:100])


async def _claim_batch():
    settings = get_settings()
    async with get_db_context() as db:
        queued = (
            await db.execute(
                text("""
            SELECT coalesce(sum(input_tokens),0) FROM memory_batches
            WHERE status NOT IN ('applied','failed','cancelled','expired')
        """)
            )
        ).scalar_one()
        available = settings.memory_batch_queue_tokens - int(queued)
        if available < 1400:
            return None
        rows = (
            (
                await db.execute(
                    text("""
            SELECT p.id,p.text,p.tokens FROM memory_passages p
            JOIN memory_days d ON d.chat_id=p.chat_id AND d.day=p.day
            JOIN telegram_chats c ON c.id=p.chat_id JOIN users u ON u.id=c.user_id
            WHERE p.embedding IS NULL AND p.batch_id IS NULL AND p.failures<3
              AND p.revision=d.revision AND d.prepared_revision=d.revision
              AND c.indexing_enabled AND u.is_active AND p.user_id=c.user_id
              AND (p.day < (now() AT TIME ZONE 'UTC')::date OR p.created_at < now()-interval '5 minutes')
            ORDER BY p.day DESC,p.id LIMIT 10000 FOR UPDATE OF p SKIP LOCKED
        """)
                )
            )
            .mappings()
            .all()
        )
        selected = []
        tokens = 0
        for row in rows:
            if tokens + row["tokens"] > available:
                break
            selected.append(dict(row))
            tokens += row["tokens"]
        if not selected:
            return None
        batch_id = uuid4()
        ids = [p["id"] for p in selected]
        await db.execute(
            text("""
            INSERT INTO memory_batches(id,status,passage_ids,input_tokens)
            VALUES(:id,'prepared',CAST(:ids AS text[]),:tokens)
        """),
            {"id": batch_id, "ids": ids, "tokens": tokens},
        )
        await db.execute(
            text(
                "UPDATE memory_passages SET batch_id=:batch WHERE id=ANY(CAST(:ids AS text[]))"
            ),
            {"batch": batch_id, "ids": ids},
        )
        return {"id": batch_id, "passages": selected, "input_tokens": tokens}


def batch_payload(passages: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                {
                    "custom_id": p["id"],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": MODEL,
                        "dimensions": DIMENSIONS,
                        "encoding_format": "base64",
                        "input": p["text"],
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for p in passages
        )
        + "\n"
    ).encode()


async def _submit(client, batch):
    # The receipt exists before either network write. An uncertain create is
    # reconciled by metadata; it is never blindly submitted for a second charge.
    try:
        client = client.with_options(max_retries=0)
        async with get_db_context() as db:
            current = (
                await db.execute(
                    text("""
                SELECT count(*) FROM memory_passages p
                JOIN memory_days d ON d.chat_id=p.chat_id AND d.day=p.day
                JOIN telegram_chats c ON c.id=p.chat_id JOIN users u ON u.id=c.user_id
                WHERE p.batch_id=:id AND p.revision=d.revision AND d.prepared_revision=d.revision
                  AND c.indexing_enabled AND u.is_active AND p.user_id=c.user_id
            """),
                    {"id": batch["id"]},
                )
            ).scalar_one()
        if current != len(batch["passages"]):
            await _release(batch, "source_changed_before_upload", count_failure=False)
            return
        data = batch_payload(batch["passages"])
        uploaded = await client.files.create(
            file=(f"memory-{batch['id']}.jsonl", data), purpose="batch"
        )
        await _batch_update(batch["id"], input_file_id=uploaded.id, status="submitting")
        remote = await client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"memory_batch": str(batch["id"]), "source": "wai-telegram"},
        )
        await _batch_update(batch["id"], provider_id=remote.id, status=remote.status)
        return True
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        code = str(getattr(exc, "code", "") or "")
        if code in {"insufficient_quota", "billing_hard_limit_reached"}:
            await _batch_update(batch["id"], status="blocked", error_code=code)
        elif status is not None and 400 <= status < 500:
            await _release(batch, f"http_{status}", count_failure=status != 429)
        else:
            await _batch_update(
                batch["id"], status="unknown", error_code=type(exc).__name__
            )
        logger.warning(
            "Memory batch submission deferred",
            extra={"batch_id": str(batch["id"]), "error_type": type(exc).__name__},
        )
        return False


def result_vector(line: dict, expected: set[str]):
    identifier = line.get("custom_id")
    if identifier not in expected:
        raise ValueError("Unexpected batch source identifier")
    response = line.get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") != 200 or line.get("error"):
        return identifier, None, 0
    if body.get("model") != MODEL:
        raise ValueError("Unexpected embedding model")
    data = body.get("data") or []
    if len(data) != 1 or data[0].get("index") != 0:
        raise ValueError("Incomplete embedding response")
    return (
        identifier,
        decode_embedding(data[0]["embedding"]),
        int((body.get("usage") or {}).get("total_tokens", 0)),
    )


async def _apply_output(client, batch, remote):
    expected = set(batch["passage_ids"])
    seen = set()
    tokens = 0
    valid = 0
    buffer = []

    async def flush():
        nonlocal buffer
        if buffer:
            if not indexing_has_capacity():
                raise RuntimeError("Insufficient disk capacity for memory import")
            async with get_db_context() as db:
                await db.execute(
                    text("""
                    UPDATE memory_passages p SET embedding=CAST(:vector AS halfvec(3072)),
                      embedded_at=now(),error_code=NULL
                    FROM memory_days d,telegram_chats c,users u
                    WHERE p.id=:id AND p.batch_id=:batch AND p.embedding IS NULL
                      AND d.chat_id=p.chat_id AND d.day=p.day AND d.revision=p.revision
                      AND d.prepared_revision=d.revision AND c.id=p.chat_id AND p.user_id=c.user_id
                      AND c.indexing_enabled AND u.id=c.user_id AND u.is_active
                """),
                    buffer,
                )
            buffer = []

    if remote.output_file_id:
        async with client.files.with_streaming_response.content(
            remote.output_file_id
        ) as response:
            async for line in response.iter_lines():
                if not line.strip():
                    continue
                item = json.loads(line)
                identifier, vector, usage = result_vector(item, expected)
                if identifier in seen:
                    raise ValueError("Duplicate batch source identifier")
                seen.add(identifier)
                tokens += usage
                if vector is not None:
                    valid += 1
                    buffer.append(
                        {
                            "id": identifier,
                            "batch": batch["id"],
                            "vector": "[" + ",".join(map(str, vector)) + "]",
                        }
                    )
                if len(buffer) >= 100:
                    if not indexing_has_capacity():
                        raise RuntimeError(
                            "Insufficient disk capacity for memory import"
                        )
                    await flush()
        await flush()
    # Failed/expired partial results are recoverable. Only missing vectors are
    # retried; confirmed results are never charged twice on a resumed import.
    async with get_db_context() as db:
        await db.execute(
            text("""
            UPDATE memory_passages SET batch_id=NULL,failures=failures+1,error_code='batch_item_failed'
            WHERE batch_id=:batch AND embedding IS NULL
        """),
            {"batch": batch["id"]},
        )
    await _batch_update(
        batch["id"],
        status="applied",
        output_file_id=remote.output_file_id,
        actual_tokens=tokens,
        applied=valid,
        failed=len(expected) - valid,
    )


async def reconcile_batches(client):
    async with get_db_context() as db:
        pending = [
            dict(row)
            for row in (
                await db.execute(
                    text("""
            SELECT * FROM memory_batches WHERE status NOT IN ('applied','failed','cancelled','expired','blocked')
            ORDER BY created_at
        """)
                )
            ).mappings()
        ]
    remote_by_receipt = None
    for batch in pending:
        if not batch["provider_id"]:
            if batch["status"] == "prepared":
                # No network write was begun; returning its claims is safe.
                await _release(batch, "interrupted_before_submit", count_failure=False)
                continue
            if remote_by_receipt is None:
                remote_by_receipt = {}
                async for item in client.batches.list(limit=100):
                    receipt = (item.metadata or {}).get("memory_batch")
                    if receipt:
                        remote_by_receipt[receipt] = item
            remote = remote_by_receipt.get(str(batch["id"]))
            if remote is None:
                continue
            await _batch_update(
                batch["id"], provider_id=remote.id, status=remote.status
            )
        else:
            remote = await client.batches.retrieve(batch["provider_id"])
        if remote.status in {"completed", "expired", "cancelled"}:
            await _apply_output(client, batch, remote)
        elif remote.status == "failed":
            errors = getattr(remote.errors, "data", []) or []
            code = errors[0].code if errors else "batch_failed"
            await _release(batch, code, count_failure="token_limit" not in code)
        else:
            await _batch_update(batch["id"], status=remote.status)


async def run_memory_batch_pass(*, prepare_limit=25, submit_limit=2):
    if not get_settings().memory_batch_enabled:
        return {"enabled": False}
    client = await get_openai_client()
    expected_org = get_settings().memory_batch_organization
    if expected_org:
        response = await client.batches.with_raw_response.list(limit=1)
        if response.headers.get("openai-organization") != expected_org:
            raise RuntimeError("OpenAI Batch organization does not match configuration")
    await reconcile_batches(client)
    status = await memory_status()
    if any(item["status"] == "blocked" for item in status["batches"]):
        return {"paused": "provider_quota", **status}
    prepared = 0
    started = time.monotonic()
    for _ in range(prepare_limit):
        if not indexing_has_capacity() or time.monotonic() - started > 90:
            break
        count = await prepare_day()
        if count is None:
            break
        prepared += count
    submitted = 0
    for _ in range(submit_limit):
        if not indexing_has_capacity():
            break
        batch = await _claim_batch()
        if batch is None:
            break
        if not await _submit(client, batch):
            break
        submitted += 1
    return {"prepared": prepared, "submitted": submitted, **await memory_status()}
