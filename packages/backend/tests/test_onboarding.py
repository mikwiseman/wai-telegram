from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.user import User
from app.services.single_user import OwnerConfigurationError, validate_active_owner
from tests.factories import TelegramChatFactory, UserFactory


async def test_owner_and_invited_user_can_be_active(db_session, test_user):
    guest = UserFactory.create(invited_by_id=test_user.id, expected_telegram_user_id=42)
    db_session.add(guest)
    await db_session.flush()
    await validate_active_owner(db_session, test_user.id)


async def test_uninvited_active_user_is_rejected(db_session, test_user):
    db_session.add(UserFactory.create())
    await db_session.flush()
    with pytest.raises(OwnerConfigurationError):
        await validate_active_owner(db_session, test_user.id)


async def test_invite_is_single_use_and_has_no_password_step(
    client, db_session, test_user
):
    from app.services.onboarding import create_invitation

    guest, token = await create_invitation(db_session, test_user.id, 42)
    await db_session.commit()
    response = await client.post("/api/v1/onboarding/accept", json={"token": token})
    assert response.status_code == 200, response.text
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.json()["id"] == str(guest.id)
    assert (
        await client.post("/api/v1/onboarding/accept", json={"token": token})
    ).status_code == 401
    await db_session.refresh(guest)
    assert guest.invite_token_hash is None
    assert guest.is_active is True


async def test_expired_invite_does_not_activate_user(client, db_session, test_user):
    from app.services.onboarding import create_invitation

    guest, token = await create_invitation(db_session, test_user.id, 42)
    guest.invite_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()
    response = await client.post("/api/v1/onboarding/accept", json={"token": token})
    assert response.status_code == 401
    await db_session.refresh(guest)
    assert guest.is_active is False


async def test_invitation_cannot_read_data_or_mint_plugin_keys_before_telegram_login(
    client, db_session, test_user
):
    from app.services.onboarding import create_invitation

    guest, token = await create_invitation(db_session, test_user.id, 42)
    await db_session.commit()
    response = await client.post("/api/v1/onboarding/accept", json={"token": token})
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    assert response.json()["refresh_token"] == ""
    assert (await client.get("/api/v1/chats", headers=headers)).status_code == 403
    assert (
        await client.post(
            "/api/v1/auth/api-keys", headers=headers, json={"name": "stolen"}
        )
    ).status_code == 403
    assert (
        await client.delete("/api/v1/telegram/session", headers=headers)
    ).status_code == 403
    # Even after a legitimate Telegram session exists, this old bootstrap token stays restricted.
    from tests.factories import TelegramSessionFactory

    db_session.add(TelegramSessionFactory.create(user_id=guest.id, telegram_user_id=42))
    await db_session.commit()
    assert (
        await client.get("/api/v1/telegram/session", headers=headers)
    ).json() is None
    assert (await client.get("/api/v1/chats", headers=headers)).status_code == 403


async def test_telegram_login_exchanges_bootstrap_for_full_access(
    client, db_session, test_user
):
    from app.services.onboarding import create_invitation

    _, token = await create_invitation(db_session, test_user.id, 42)
    await db_session.commit()
    response = await client.post("/api/v1/onboarding/accept", json={"token": token})
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    with (
        patch(
            "app.services.telegram_client.request_code",
            return_value=(AsyncMock(), "hash", "app"),
        ),
        patch("app.services.telegram_client.verify_code", return_value=("session", 42)),
    ):
        await client.post(
            "/api/v1/telegram/request-code",
            headers=headers,
            json={"phone_number": "+1234567890"},
        )
        result = await client.post(
            "/api/v1/telegram/verify-code",
            headers=headers,
            json={
                "phone_number": "+1234567890",
                "phone_code_hash": "hash",
                "code": "12345",
            },
        )
    assert result.status_code == 200, result.text
    assert result.json()["refresh_token"]
    full_headers = {"Authorization": f"Bearer {result.json()['access_token']}"}
    assert (await client.get("/api/v1/chats", headers=full_headers)).status_code == 200


async def test_invite_cannot_reactivate_an_unrelated_archive(db_session, test_user):
    from app.services.onboarding import create_invitation

    first, _ = await create_invitation(db_session, test_user.id, 42)
    second, _ = await create_invitation(db_session, test_user.id, 42)
    assert first.id == second.id
    assert len((await db_session.execute(select(User))).scalars().all()) == 2


async def test_selection_rejects_other_users_chat(auth_client, db_session, test_user):
    guest = UserFactory.create(invited_by_id=test_user.id, expected_telegram_user_id=42)
    db_session.add(guest)
    await db_session.flush()
    foreign = TelegramChatFactory.create(user_id=guest.id, indexing_enabled=False)
    db_session.add(foreign)
    await db_session.commit()
    response = await auth_client.put(
        "/api/v1/indexing/chats", json={"chat_ids": [str(foreign.id)]}
    )
    assert response.status_code == 404
    await db_session.refresh(foreign)
    assert foreign.indexing_enabled is False


async def test_selection_only_changes_own_chats_and_schedules_full_history(
    auth_client, db_session, test_user
):
    own = TelegramChatFactory.create(user_id=test_user.id)
    other = TelegramChatFactory.create(user_id=test_user.id, indexing_enabled=True)
    db_session.add_all([own, other])
    await db_session.commit()
    with patch("app.api.v1.indexing.sync_all_chats_task.delay") as task:
        response = await auth_client.put(
            "/api/v1/indexing/chats", json={"chat_ids": [str(own.id)]}
        )
    assert response.status_code == 200, response.text
    await db_session.refresh(own)
    await db_session.refresh(other)
    assert own.indexing_enabled is True
    assert other.indexing_enabled is False
    assert task.call_args.args[0] == str(test_user.id)
    assert task.call_args.args[2] == 0


async def test_wrong_telegram_account_cannot_be_attached(
    auth_client, db_session, test_user
):
    from app.api.v1 import telegram

    test_user.expected_telegram_user_id = 42
    await db_session.commit()
    fake_client = AsyncMock()
    telegram._auth_clients[f"{test_user.id}:+1234567890"] = (
        fake_client,
        __import__("time").time(),
    )
    with (
        patch(
            "app.services.telegram_client.verify_code",
            return_value=("secret_session", 99),
        ),
        patch(
            "app.services.telegram_client.save_session", new_callable=AsyncMock
        ) as save,
    ):
        response = await auth_client.post(
            "/api/v1/telegram/verify-code",
            json={
                "phone_number": "+1234567890",
                "phone_code_hash": "hash",
                "code": "12345",
            },
        )
    assert response.status_code == 403
    save.assert_not_called()
    fake_client.disconnect.assert_awaited()


async def test_reselecting_chat_recovers_messages_missed_while_disabled(
    auth_client, db_session, test_user
):
    chat = TelegramChatFactory.create(
        user_id=test_user.id,
        indexing_enabled=False,
        history_synced_at=datetime.now(UTC),
        history_cursor=123,
    )
    db_session.add(chat)
    await db_session.commit()
    with (
        patch("app.api.v1.indexing.sync_all_chats_task.delay") as task,
        patch("app.api.v1.indexing.start_listener", new_callable=AsyncMock),
    ):
        response = await auth_client.put(
            "/api/v1/indexing/chats", json={"chat_ids": [str(chat.id)]}
        )
    assert response.status_code == 200
    await db_session.refresh(chat)
    assert chat.history_synced_at is None
    assert chat.history_cursor is None
    task.assert_called_once()
