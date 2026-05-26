"""
Rate limiting using a sliding window counter in Postgres.

Design decision: Postgres over Redis because:
- No extra dependency for a side project
- Transactional upsert is correct under concurrent requests
- At 10x scale with distributed deployments: move to Redis with INCR + EXPIRE
  (Redis atomic operations are more efficient for pure counting workloads)
"""

from datetime import datetime, timezone

from app.core.config import settings
from app.db.pool import get_pool


def _window_key(now: datetime) -> str:
    """1-minute tumbling window key."""
    return now.strftime("%Y%m%d%H%M")


async def check_rate_limit(agent_id: str, operation: str) -> tuple[bool, int]:
    """
    Returns (allowed, current_count).
    Increments counter and checks against limit in a single round-trip.
    """
    limit = (
        settings.rate_limit_writes_per_minute
        if operation == "write"
        else settings.rate_limit_searches_per_minute
    )
    window_key = _window_key(datetime.now(timezone.utc))

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO rate_limit_counters (agent_id, window_key, operation, count, window_start)
            VALUES ($1, $2, $3, 1, now())
            ON CONFLICT (agent_id, window_key, operation)
            DO UPDATE SET count = rate_limit_counters.count + 1
            RETURNING count
            """,
            agent_id,
            window_key,
            operation,
        )
    current = row["count"]
    return current <= limit, current
