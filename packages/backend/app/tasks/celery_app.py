import logging

from celery import Celery, signals

from app.core.async_runner import run_async, start_async_runner, stop_async_runner
from app.core.config import get_settings
from app.core.database import dispose_engine
from app.core.observability import (
    build_runtime_summary,
    configure_logging,
    init_observability,
    log_event,
)

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "wai_telegram",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.tasks.sync_tasks",
        "app.tasks.digest_tasks",
        "app.tasks.agent_tasks",
        "app.tasks.media_tasks",
        "app.tasks.embedding_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        "app.tasks.media_tasks.fetch_message_media": {"queue": "media-fetch"},
        "app.tasks.media_tasks.process_message_media": {"queue": "media-process"},
        "app.tasks.media_tasks.index_message_media": {"queue": "media-index"},
    },
)


@signals.celeryd_init.connect
def configure_celery_worker_observability(**_kwargs):
    init_observability(
        settings,
        "wai-telegram-celery-worker",
        enable_celery_monitoring=True,
    )
    log_event(
        logger,
        logging.INFO,
        "Celery worker observability initialized",
        event_name="celery.worker.bootstrap",
        **build_runtime_summary(
            service_name="wai-telegram-celery-worker",
            settings=settings,
        ),
    )


@signals.worker_process_init.connect
def init_worker_process_runtime(**_kwargs):
    start_async_runner()


@signals.worker_process_shutdown.connect
def shutdown_worker_process_runtime(**_kwargs):
    try:
        start_async_runner()
        from app.services.media_processing_service import disconnect_media_clients

        run_async(disconnect_media_clients())
        run_async(dispose_engine())
    finally:
        stop_async_runner()


@signals.beat_init.connect
def configure_celery_beat_observability(**_kwargs):
    init_observability(
        settings,
        "wai-telegram-celery-beat",
        enable_celery_monitoring=True,
    )
    log_event(
        logger,
        logging.INFO,
        "Celery beat observability initialized",
        event_name="celery.beat.bootstrap",
        **build_runtime_summary(
            service_name="wai-telegram-celery-beat",
            settings=settings,
        ),
    )


# Beat schedule
beat_schedule = {
    "repair-selected-embeddings": {
        "task": "app.tasks.embedding_tasks.repair_embeddings",
        "schedule": 60,
    },
    "generate-daily-digests": {
        "task": "app.tasks.digest_tasks.generate_all_digests",
        "schedule": 3600,  # Every hour — per-user hour matching inside task
    },
    "listener-health-check": {
        "task": "app.tasks.sync_tasks.listener_health_check",
        "schedule": 300,  # Every 5 minutes
    },
    "reap-stale-sync-jobs": {
        "task": "app.tasks.sync_tasks.reap_stale_sync_jobs",
        "schedule": 120,  # Every 2 minutes
    },
    "run-due-agents": {
        "task": "app.tasks.agent_tasks.run_due_agents",
        "schedule": 60,  # Every minute
    },
}
if settings.media_pipeline_enabled:
    beat_schedule["dispatch-pending-media"] = {
        "task": "app.tasks.media_tasks.dispatch_pending_media",
        "schedule": 15,  # Keep a small durable queue full without flooding Redis.
    }
celery_app.conf.beat_schedule = beat_schedule
