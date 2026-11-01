"""Create or renew a one-use onboarding link without a WAI password."""

import argparse
import asyncio
from datetime import UTC, date, datetime, time

from app.core.config import get_settings
from app.core.database import get_db_context
from app.services.onboarding import create_invitation


async def run(telegram_user_id: int, base_url: str, since: date | None = None) -> None:
    settings = get_settings()
    if settings.owner_user_id is None:
        raise RuntimeError("OWNER_USER_ID is required")
    async with get_db_context() as db:
        _, token = await create_invitation(
            db,
            settings.owner_user_id,
            telegram_user_id,
            auto_index_since=datetime.combine(since, time.min, UTC) if since else None,
        )
    # Fragment avoids placing the login secret in access logs or Referer headers.
    print(f"{base_url.rstrip('/')}/connect#invite={token}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telegram-user-id", required=True, type=int)
    parser.add_argument("--base-url", default="https://telegram.waiwai.is")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        help="Automatically connect all chats and load history since YYYY-MM-DD (UTC)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.telegram_user_id, args.base_url, args.since))


if __name__ == "__main__":
    main()
