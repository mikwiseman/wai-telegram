"""Authentication must not lock a shared API key during endpoint network I/O."""

from datetime import UTC, datetime

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import get_auth_context
from app.core.security import (
    compute_api_key_prefix,
    get_key_hint,
    hash_api_key,
    password_hash,
)
from app.models.api_key import ApiKey


@pytest.mark.parametrize("legacy", [False, True])
async def test_api_key_transaction_finishes_before_endpoint(
    db_session, test_user, legacy
):
    raw_key = "wai_transaction_boundary_test_key"
    api_key = ApiKey(
        user_id=test_user.id,
        name="Concurrent callers",
        key_hash=password_hash.hash(raw_key) if legacy else hash_api_key(raw_key),
        key_prefix=compute_api_key_prefix(raw_key),
        key_hint=get_key_hint(raw_key),
        is_active=True,
        scopes="read,draft",
    )
    db_session.add(api_key)
    await db_session.commit()
    key_id = api_key.id
    started = datetime.now(UTC)

    context = await get_auth_context(
        token=raw_key,
        api_key=HTTPAuthorizationCredentials(scheme="Bearer", credentials=raw_key),
        db=db_session,
    )

    assert context.user.id == test_user.id
    assert context.scopes == {"read", "draft"}
    # The endpoint can now await Telegram indefinitely. No key-row UPDATE lock
    # may survive into that wait and block other calls using the same key.
    assert not db_session.in_transaction()

    # A later endpoint failure must not roll back authentication bookkeeping.
    await db_session.rollback()
    db_session.expire_all()
    stored_key = await db_session.get(ApiKey, key_id)
    assert stored_key.key_hash == hash_api_key(raw_key)
    assert stored_key.last_used_at.replace(tzinfo=UTC) >= started
