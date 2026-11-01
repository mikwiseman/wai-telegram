from datetime import UTC, date, datetime
from uuid import uuid4

from app.core.security import (
    compute_api_key_prefix,
    get_key_hint,
    hash_api_key,
    hash_password,
)
from app.models.api_key import ApiKey
from app.models.chat import ChatType, TelegramChat
from app.models.digest import DailyDigest
from app.models.message import TelegramMessage
from app.models.session import TelegramSession
from app.models.settings import UserSettings
from app.models.sync_job import SyncJob, SyncStatus
from app.models.user import User


class UserFactory:
    @staticmethod
    def create(**kwargs):
        defaults = {
            "id": uuid4(),
            "email": f"user-{uuid4().hex[:8]}@example.com",
            "password_hash": hash_password("TestPassword1"),
            "is_active": True,
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return User(**defaults)


class TelegramChatFactory:
    @staticmethod
    def create(user_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "telegram_chat_id": abs(hash(uuid4())) % 10**9,
            "chat_type": ChatType.PRIVATE,
            "title": f"Chat {uuid4().hex[:6]}",
            "indexing_enabled": True,
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return TelegramChat(**defaults)


class TelegramMessageFactory:
    @staticmethod
    def create(chat_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "chat_id": chat_id or uuid4(),
            "telegram_message_id": abs(hash(uuid4())) % 10**9,
            "text": f"Test message {uuid4().hex[:6]}",
            "has_media": False,
            "sender_id": 12345,
            "sender_name": "Test User",
            "is_outgoing": False,
            "sent_at": datetime.now(UTC),
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return TelegramMessage(**defaults)


class TelegramSessionFactory:
    @staticmethod
    def create(user_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "phone_number": "+1234567890",
            "session_string": "encrypted_session_string",
            "telegram_user_id": 12345,
            "is_active": True,
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return TelegramSession(**defaults)


class SyncJobFactory:
    @staticmethod
    def create(user_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "status": SyncStatus.PENDING,
            "messages_processed": 0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return SyncJob(**defaults)


class ApiKeyFactory:
    @staticmethod
    def create(user_id=None, raw_key="wai_test_key_12345678", **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "name": f"Test Key {uuid4().hex[:6]}",
            "key_hash": hash_api_key(raw_key),
            "key_prefix": compute_api_key_prefix(raw_key),
            "key_hint": get_key_hint(raw_key),
            "is_active": True,
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return ApiKey(**defaults)


class UserSettingsFactory:
    @staticmethod
    def create(user_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "digest_enabled": True,
            "digest_hour_utc": 9,
            "digest_timezone": "UTC",
            "digest_telegram_enabled": False,
            "realtime_sync_enabled": False,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return UserSettings(**defaults)


class DailyDigestFactory:
    @staticmethod
    def create(user_id=None, **kwargs):
        defaults = {
            "id": uuid4(),
            "user_id": user_id or uuid4(),
            "digest_date": date.today(),
            "content": "Test digest content",
            "summary_stats": {"total_messages": 10, "chats": ["Test Chat"]},
            "created_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return DailyDigest(**defaults)
