"""Tests for app.tasks.sync_tasks — DistributedLock, heartbeat/cleanup helpers,
_mark_job_state, and reap_stale_sync_jobs."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


from app.models.sync_job import SyncStatus
from app.tasks.sync_tasks import DistributedLock


# ---------------------------------------------------------------------------
# FakeRedis (lightweight, in-process — no real Redis needed)
# ---------------------------------------------------------------------------
class FakeRedis:
    """Minimal fake Redis for testing DistributedLock and helpers."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def set(
        self, key: str, value: str, nx: bool = False, ex: int | None = None
    ) -> bool:
        if nx and key in self.data:
            return False
        self.data[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self.data[key] = str(value)
        self.ttls[key] = ttl
        return True

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count

    def eval(self, script: str, num_keys: int, key: str, *args) -> int:
        if "del" in script.lower():
            if self.data.get(key) == args[0]:
                del self.data[key]
                return 1
            return 0
        if "expire" in script.lower():
            if self.data.get(key) == args[0]:
                self.ttls[key] = int(args[1]) if len(args) > 1 else 180
                return 1
            return 0
        return 0

    def publish(self, channel: str, message: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# DistributedLock
# ---------------------------------------------------------------------------
class TestDistributedLock:
    def test_acquire_success(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        user_id = uuid4()
        lock = DistributedLock(user_id, owner=f"chat:{uuid4()}")
        assert lock.acquire() is True
        assert fake_redis.data[lock.lock_key] == lock.payload

        lock.release()

    def test_acquire_conflict(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        user_id = uuid4()
        lock1 = DistributedLock(user_id, owner="owner1")
        lock2 = DistributedLock(user_id, owner="owner2")

        assert lock1.acquire() is True
        assert lock2.acquire() is False

        lock1.release()

    def test_release_does_not_delete_foreign_owner(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        user_id = uuid4()
        lock = DistributedLock(user_id, owner=f"chat:{uuid4()}")
        assert lock.acquire() is True

        other_payload = json.dumps(
            {"owner": "other", "token": "other-token"},
            separators=(",", ":"),
            sort_keys=True,
        )
        fake_redis.data[lock.lock_key] = other_payload
        lock.release()

        assert fake_redis.data[lock.lock_key] == other_payload

    def test_refresh_extends_ttl(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        user_id = uuid4()
        lock = DistributedLock(user_id, owner="test")
        assert lock.acquire() is True
        assert lock.refresh() is True

        lock.release()

    def test_refresh_fails_if_not_owner(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        user_id = uuid4()
        lock = DistributedLock(user_id, owner="test")
        assert lock.acquire() is True

        fake_redis.data[lock.lock_key] = "foreign_payload"
        assert lock.refresh() is False

        lock.release()

    def test_lock_key_format(self):
        """Lock key follows sync:{user_id}:lock pattern."""
        uid = uuid4()
        lock = DistributedLock(uid, owner="x")
        assert lock.lock_key == f"sync:{uid}:lock"

    def test_payload_is_deterministic_json(self):
        """Payload is sorted-key JSON with compact separators."""
        uid = uuid4()
        lock = DistributedLock(uid, owner="myowner")
        parsed = json.loads(lock.payload)
        assert parsed["owner"] == "myowner"
        assert "token" in parsed
        # Verify compact separators (no spaces)
        assert " " not in lock.payload

    def test_custom_ttl(self, monkeypatch):
        """Custom TTL is stored and used during acquire."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        uid = uuid4()
        lock = DistributedLock(uid, owner="ttl-test", ttl=999)
        assert lock.ttl == 999
        lock.acquire()

        assert fake_redis.ttls[lock.lock_key] == 999
        lock.release()

    def test_acquire_after_release(self, monkeypatch):
        """After releasing, the same user_id can be re-locked."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        uid = uuid4()
        lock1 = DistributedLock(uid, owner="a")
        assert lock1.acquire() is True
        lock1.release()

        lock2 = DistributedLock(uid, owner="b")
        assert lock2.acquire() is True
        lock2.release()


class TestInactiveUserGate:
    @patch("app.tasks.sync_tasks.run_async", return_value=False)
    @patch("app.tasks.sync_tasks.check_budget")
    def test_single_sync_stops_before_budget_and_lock(self, budget, run):
        from app.tasks.sync_tasks import sync_chat_task

        result = sync_chat_task.run(str(uuid4()), str(uuid4()))

        assert result == {"status": "skipped", "reason": "inactive_user"}
        budget.assert_not_called()
        assert run.call_count == 1
        run.call_args.args[0].close()


# ---------------------------------------------------------------------------
# Heartbeat key helpers
# ---------------------------------------------------------------------------
class TestHeartbeatKeys:
    def test_single_heartbeat_key_format(self):
        from app.tasks.sync_tasks import _single_heartbeat_key

        job_id = uuid4()
        key = _single_heartbeat_key(job_id)
        assert key == f"sync:{job_id}:heartbeat"

    def test_bulk_heartbeat_key_format(self):
        from app.tasks.sync_tasks import _bulk_heartbeat_key

        job_id = uuid4()
        key = _bulk_heartbeat_key(job_id)
        assert key == f"bulk:{job_id}:heartbeat"


# ---------------------------------------------------------------------------
# Touch heartbeat helpers
# ---------------------------------------------------------------------------
class TestTouchHeartbeat:
    def test_touch_single_heartbeat(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _touch_single_heartbeat

        job_id = uuid4()
        _touch_single_heartbeat(job_id)

        key = f"sync:{job_id}:heartbeat"
        assert fake_redis.data[key] == "1"
        assert fake_redis.ttls[key] == 600  # JOB_HEARTBEAT_TTL

    def test_touch_bulk_heartbeat(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _touch_bulk_heartbeat

        job_id = uuid4()
        _touch_bulk_heartbeat(job_id)

        key = f"bulk:{job_id}:heartbeat"
        assert fake_redis.data[key] == "1"
        assert fake_redis.ttls[key] == 600


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
class TestCleanupProgress:
    def test_cleanup_single_progress(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _cleanup_single_progress

        job_id = uuid4()
        fake_redis.data[f"sync:{job_id}:total"] = "100"
        fake_redis.data[f"sync:{job_id}:seen"] = "50"
        fake_redis.data[f"sync:{job_id}:heartbeat"] = "1"

        _cleanup_single_progress(job_id)

        assert f"sync:{job_id}:total" not in fake_redis.data
        assert f"sync:{job_id}:seen" not in fake_redis.data
        assert f"sync:{job_id}:heartbeat" not in fake_redis.data

    def test_cleanup_bulk_progress(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _cleanup_bulk_progress

        job_id = uuid4()
        fake_redis.data[f"bulk:{job_id}:total"] = "10"
        fake_redis.data[f"bulk:{job_id}:completed"] = "5"
        fake_redis.data[f"bulk:{job_id}:current_chat"] = "Chat"
        fake_redis.data[f"bulk:{job_id}:heartbeat"] = "1"

        _cleanup_bulk_progress(job_id)

        assert f"bulk:{job_id}:total" not in fake_redis.data
        assert f"bulk:{job_id}:completed" not in fake_redis.data
        assert f"bulk:{job_id}:current_chat" not in fake_redis.data
        assert f"bulk:{job_id}:heartbeat" not in fake_redis.data

    def test_cleanup_single_idempotent(self, monkeypatch):
        """Cleaning up keys that don't exist should not error."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _cleanup_single_progress

        job_id = uuid4()
        _cleanup_single_progress(job_id)  # no-op, should not raise

    def test_cleanup_bulk_idempotent(self, monkeypatch):
        """Cleaning up keys that don't exist should not error."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _cleanup_bulk_progress

        job_id = uuid4()
        _cleanup_bulk_progress(job_id)  # no-op, should not raise


# ---------------------------------------------------------------------------
# _mark_job_state (async helper)
# ---------------------------------------------------------------------------
class TestMarkJobState:
    async def test_noop_when_job_id_is_none(self):
        """Passing None as job_id returns immediately."""
        from app.tasks.sync_tasks import _mark_job_state

        # Should not raise or interact with DB
        await _mark_job_state(None, SyncStatus.FAILED, "error")

    async def test_updates_status_and_error(self):
        """Job status and error_message are written to the DB."""
        job_id = uuid4()

        mock_job = MagicMock()
        mock_job.status = SyncStatus.PENDING
        mock_job.error_message = None
        mock_job.completed_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _mark_job_state

            await _mark_job_state(job_id, SyncStatus.FAILED, "sync_error: timeout")

        assert mock_job.status == SyncStatus.FAILED
        assert mock_job.error_message == "sync_error: timeout"
        assert mock_job.completed_at is None
        mock_db.commit.assert_awaited_once()

    async def test_sets_completed_at_on_completed_status(self):
        """When marking COMPLETED, completed_at is set."""
        job_id = uuid4()

        mock_job = MagicMock()
        mock_job.status = SyncStatus.IN_PROGRESS
        mock_job.completed_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _mark_job_state

            await _mark_job_state(job_id, SyncStatus.COMPLETED)

        assert mock_job.status == SyncStatus.COMPLETED
        assert mock_job.completed_at is not None

    async def test_does_not_downgrade_terminal_to_nonterminal(self):
        """A job in FAILED state should not be downgraded to PENDING."""
        job_id = uuid4()

        mock_job = MagicMock()
        mock_job.status = SyncStatus.FAILED
        mock_job.error_message = "original error"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _mark_job_state

            await _mark_job_state(job_id, SyncStatus.PENDING, "retry")

        # Status should remain FAILED (not downgraded to PENDING)
        assert mock_job.status == SyncStatus.FAILED

    async def test_noop_when_job_not_found(self):
        """If the job doesn't exist in the DB, returns silently."""
        job_id = uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _mark_job_state

            await _mark_job_state(job_id, SyncStatus.FAILED, "gone")

        mock_db.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# reap_stale_sync_jobs (Celery task + async helper)
# ---------------------------------------------------------------------------
class TestReapStaleSyncJobs:
    @patch("app.tasks.sync_tasks.run_async")
    def test_celery_wrapper_calls_async(self, mock_run_async):
        """reap_stale_sync_jobs delegates to the shared async runner."""
        mock_run_async.side_effect = [{"scanned": 0, "expired": 0}, 2]

        from app.tasks.sync_tasks import reap_stale_sync_jobs

        result = reap_stale_sync_jobs()
        assert result == {"scanned": 0, "expired": 0, "resumed": 2}
        for call in mock_run_async.call_args_list:
            call.args[0].close()

    async def test_expires_stale_job_without_heartbeat(self, monkeypatch):
        """A stale IN_PROGRESS job with no heartbeat is marked FAILED."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        job_id = uuid4()
        user_id = uuid4()
        chat_id = uuid4()

        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.user_id = user_id
        mock_job.chat_id = chat_id  # single-chat job
        mock_job.status = SyncStatus.IN_PROGRESS
        mock_job.updated_at = datetime.now(UTC) - timedelta(minutes=20)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_job]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _reap_stale_sync_jobs

            result = await _reap_stale_sync_jobs()

        assert result["scanned"] == 1
        assert result["expired"] == 1
        assert mock_job.status == SyncStatus.FAILED
        assert "stale" in mock_job.error_message.lower()

    async def test_skips_job_with_active_heartbeat(self, monkeypatch):
        """A stale job with an active heartbeat is NOT expired."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        job_id = uuid4()
        user_id = uuid4()

        # Set a heartbeat for this job
        fake_redis.data[f"sync:{job_id}:heartbeat"] = "1"

        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.user_id = user_id
        mock_job.chat_id = uuid4()
        mock_job.status = SyncStatus.IN_PROGRESS

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_job]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _reap_stale_sync_jobs

            result = await _reap_stale_sync_jobs()

        assert result["scanned"] == 1
        assert result["expired"] == 0
        # Status should remain IN_PROGRESS
        assert mock_job.status == SyncStatus.IN_PROGRESS

    async def test_no_stale_jobs(self, monkeypatch):
        """When no stale jobs exist, scanned and expired are both 0."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _reap_stale_sync_jobs

            result = await _reap_stale_sync_jobs()

        assert result == {"scanned": 0, "expired": 0}
        mock_db.commit.assert_not_awaited()

    async def test_bulk_job_uses_bulk_heartbeat_key(self, monkeypatch):
        """A bulk job (chat_id=None) checks the bulk heartbeat key."""
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        job_id = uuid4()
        user_id = uuid4()

        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.user_id = user_id
        mock_job.chat_id = None  # bulk job
        mock_job.status = SyncStatus.IN_PROGRESS

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_job]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tasks.sync_tasks.get_db_context", return_value=mock_db):
            from app.tasks.sync_tasks import _reap_stale_sync_jobs

            result = await _reap_stale_sync_jobs()

        # No bulk heartbeat → expired
        assert result["expired"] == 1
        assert mock_job.status == SyncStatus.FAILED


# ---------------------------------------------------------------------------
# _release_lock_if_owned
# ---------------------------------------------------------------------------
class TestReleaseLockIfOwned:
    def test_releases_when_owner_matches(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _release_lock_if_owned

        user_id = uuid4()
        lock_key = f"sync:{user_id}:lock"

        # FakeRedis eval checks simple string equality for the "del" script.
        # _release_lock_if_owned uses cjson.decode in the real Lua script,
        # but our FakeRedis just does get(key) == args[0].
        owner = "chat:abc"
        fake_redis.data[lock_key] = owner

        _release_lock_if_owned(user_id, owner)
        assert lock_key not in fake_redis.data

    def test_does_not_release_foreign_owner(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr("app.tasks.sync_tasks.redis_client", fake_redis)

        from app.tasks.sync_tasks import _release_lock_if_owned

        user_id = uuid4()
        lock_key = f"sync:{user_id}:lock"
        fake_redis.data[lock_key] = "owner_A"

        _release_lock_if_owned(user_id, "owner_B")
        assert fake_redis.data[lock_key] == "owner_A"


# ---------------------------------------------------------------------------
# listener_health_check
# ---------------------------------------------------------------------------
class TestListenerHealthCheck:
    @patch("app.tasks.sync_tasks.run_async")
    def test_celery_wrapper_calls_async(self, mock_run_async):
        mock_run_async.return_value = {"checked": 0, "restarted": 0}

        from app.tasks.sync_tasks import listener_health_check

        result = listener_health_check()
        assert result == {"checked": 0, "restarted": 0}
        mock_run_async.assert_called_once()
        mock_run_async.call_args.args[0].close()
