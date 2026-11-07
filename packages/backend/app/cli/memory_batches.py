"""Seed, inspect or advance the same durable Batch workflow used by Celery."""

import argparse
import asyncio
import json

import redis

from app.core.config import get_settings
from app.services.memory_batch_service import (
    memory_status,
    prepare_day,
    run_memory_batch_pass,
    seed_days,
)
from app.services.indexing_service import indexing_has_capacity


async def run(args):
    if args.seed:
        print(json.dumps({"seededDays": await seed_days()}))
    if args.prepare_all:
        days = 0
        while indexing_has_capacity():
            if await prepare_day() is None:
                break
            days += 1
            if days % 100 == 0:
                print(json.dumps({"preparedDays": days}), flush=True)
    if args.advance:
        client = redis.from_url(get_settings().redis_url)
        try:
            with client.lock("memory:batch", timeout=600, blocking_timeout=0):
                result = await run_memory_batch_pass(prepare_limit=args.prepare_limit)
        finally:
            client.close()
    else:
        result = await memory_status()
    print(json.dumps(result, default=str))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--advance", action="store_true")
    parser.add_argument("--prepare-all", action="store_true")
    parser.add_argument("--prepare-limit", type=int, default=100)
    args = parser.parse_args()
    if args.prepare_limit < 1:
        parser.error("prepare-limit must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
