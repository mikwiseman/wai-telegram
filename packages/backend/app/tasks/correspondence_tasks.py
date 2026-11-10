import httpx
from celery import shared_task

from app.core.async_runner import run_async
from app.core.database import get_db_context
from app.services.correspondence_events import BATCH_SIZE, deliver_batch


@shared_task
def deliver_correspondence_events():
    return run_async(_deliver())


async def _deliver():
    # Locks cover only outbox rows. A crash rolls them back; the stable event ID
    # makes a retry safe even when the receiver accepted it before a timeout.
    # A full batch means more may wait: keep going, so a backlog arrives together.
    delivered = 0
    async with httpx.AsyncClient() as client:
        for _ in range(200):
            async with get_db_context() as db:
                batch = await deliver_batch(db, client)
            delivered += batch
            if batch < BATCH_SIZE:
                break
    return {"delivered": delivered}
