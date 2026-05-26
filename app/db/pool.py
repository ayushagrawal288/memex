"""
Connection pool management using asyncpg directly.

Design decision: asyncpg over SQLAlchemy async because:
- Zero ORM overhead on the hot retrieval path
- Direct control over pool sizing (same instinct as HikariCP tuning)
- pgvector queries need raw SQL anyway for the <=> operator

At 10x scale: replace with PgBouncer in transaction mode in front of this,
and increase db_max_pool_size to match PgBouncer's server_pool_size.
"""

import asyncpg
from app.core.config import settings


_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.db_min_pool_size,
        max_size=settings.db_max_pool_size,
        command_timeout=settings.db_command_timeout,
        # server_settings tune pg behaviour per-connection
        server_settings={"application_name": "memex-api"},
    )


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_pool() first")
    return _pool


async def run_migrations() -> None:
    """
    Inline migrations — sufficient for a side project.
    In production: use Alembic with version-controlled migration files.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_id        TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                content         TEXT NOT NULL,
                embedding       vector(384),
                importance      FLOAT NOT NULL DEFAULT 1.0,
                memory_type     TEXT NOT NULL DEFAULT 'episodic',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        # ivfflat index for approximate nearest-neighbour search.
        # lists=100 is a reasonable default for up to ~1M rows.
        # At 10x scale: switch to HNSW (better recall, higher build cost).
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS memories_embedding_idx
            ON memories USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS memories_agent_user_idx
            ON memories (agent_id, user_id, created_at DESC)
        """)

        # Rate limiting table — sliding window counters
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limit_counters (
                agent_id    TEXT NOT NULL,
                window_key  TEXT NOT NULL,
                operation   TEXT NOT NULL,
                count       INT NOT NULL DEFAULT 0,
                window_start TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (agent_id, window_key, operation)
            )
        """)
