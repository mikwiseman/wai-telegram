import httpx
from celery import shared_task

from app.core.async_runner import run_async
from app.core.database import get_db_context
from app.services.correspondence_events import deliver_batch


@shared_task
def deliver_correspondence_events():
    return run_async(_deliver())


async def _deliver():
    # Locks cover only outbox rows. A crash rolls them back; the stable event ID
    # makes a retry safe even when the receiver accepted it before a timeout.
    async with get_db_context() as db, httpx.AsyncClient() as client:
        return {"delivered": await deliver_batch(db, client)}
