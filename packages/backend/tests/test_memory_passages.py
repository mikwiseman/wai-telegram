"""Source continuity, token limits and provider-result validation for memory."""

import base64
import struct
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.services.memory_passages import (
    DIMENSIONS,
    MAX_TOKENS,
    build_passages,
    decode_embedding,
)


def message(value, **overrides):
    return {
        "id": uuid4(),
        "telegram_message_id": 10,
        "sender_name": "Alisa",
        "sent_at": datetime(2026, 11, 7, 12, tzinfo=UTC),
        "text": value,
        "content_text": None,
        "content_summary": None,
        "media_file_name": None,
        "thread_id": None,
        "deleted_at": None,
        **overrides,
    }


def test_a_correction_retains_both_speakers_sources_and_order():
    old = message("Use the morning plan", sender_name="Mik")
    new = message("Correction: use the evening plan", telegram_message_id=11)
    result = build_passages("Family", [old, new])
    assert len(result) == 1
    assert result[0].message_ids == [old["id"], new["id"]]
    assert "Mik" in result[0].text and "Alisa" in result[0].text
    assert result[0].text.index("morning") < result[0].text.index("Correction")


def test_long_unicode_documents_preserve_every_character_with_bounded_inputs():
    source = "שלום мир 日本語 👨‍👩‍👧‍👦\n" * 5000
    item = message(None, content_text=source)
    result = build_passages("Family", [item])
    assert len(result) > 1
    assert all(0 < p.tokens <= MAX_TOKENS for p in result)
    assert "".join(p.source_text for p in result) == source
    assert all(p.message_ids == [item["id"]] for p in result)


def test_threads_and_extracted_documents_are_separate():
    a = message("first", thread_id=10)
    b = message("second", thread_id=20)
    c = message("caption", content_text="original document")
    result = build_passages("Work", [a, b, c])
    assert not any(
        a["id"] in p.message_ids and b["id"] in p.message_ids for p in result
    )
    assert any(p.source_text == "original document" for p in result)


def test_deleted_text_is_not_embedded_and_unchanged_sources_keep_identity():
    keep = message("hello")
    gone = message("secret", deleted_at=datetime.now(UTC))
    first = build_passages("Work", [keep, gone])
    second = build_passages("Work", [keep])
    assert first == second
    assert "secret" not in first[0].text
    changed = build_passages("Work", [{**keep, "text": "corrected"}])
    assert first[0].id != changed[0].id


def test_provider_vectors_must_be_complete_and_finite():
    encoded = base64.b64encode(
        struct.pack(f"<{DIMENSIONS}f", *([0.1] * DIMENSIONS))
    ).decode()
    assert len(decode_embedding(encoded)) == DIMENSIONS
    with pytest.raises(ValueError, match="dimension"):
        decode_embedding([0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        decode_embedding([float("nan")] * DIMENSIONS)


def test_batch_output_cannot_attach_another_source_or_model():
    from app.services.memory_batch_service import result_vector
    from app.services.memory_passages import MODEL

    valid = {
        "custom_id": "ours",
        "response": {
            "status_code": 200,
            "body": {
                "model": MODEL,
                "data": [{"index": 0, "embedding": [0.1] * DIMENSIONS}],
                "usage": {"total_tokens": 23},
            },
        },
    }
    assert result_vector(valid, {"ours"})[2] == 23
    with pytest.raises(ValueError, match="source identifier"):
        result_vector(valid, {"other"})
    valid["response"]["body"]["model"] = "text-embedding-3-small"
    with pytest.raises(ValueError, match="model"):
        result_vector(valid, {"ours"})


def test_batch_payload_uses_the_large_model_and_keeps_source_ids():
    import json
    from app.services.memory_batch_service import batch_payload

    payload = json.loads(
        batch_payload([{"id": "source-v1", "text": "Алиса: исправление"}])
    )
    assert payload["custom_id"] == "source-v1"
    assert payload["body"] == {
        "model": "text-embedding-3-large",
        "dimensions": DIMENSIONS,
        "encoding_format": "base64",
        "input": "Алиса: исправление",
    }


def test_search_rechecks_current_source_revision_and_scope():
    from app.services.search_service import _hybrid_search_sql

    sql = str(
        _hybrid_search_sql(
            DIMENSIONS, "c.user_id=:user_id AND m.deleted_at IS NULL", memory=True
        )
    )
    assert "p.revision=d.revision" in sql
    assert "d.prepared_revision=d.revision" in sql
    assert "p.user_id=c.user_id" in sql
    assert "c.indexing_enabled" in sql
    assert "m.deleted_at IS NULL" in sql
    assert "m.embedding <=>" not in sql
