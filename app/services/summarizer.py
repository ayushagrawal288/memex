"""
Memory summarisation background job.

Condenses old episodic memories when an (agent_id, user_id) pair exceeds
SUMMARIZATION_THRESHOLD. Runs every SUMMARIZATION_INTERVAL_SECONDS.

Summarisation is fully local — no API calls, no external dependencies.
See local_summarizer.py for the algorithm (extractive, frequency-scored).

Concurrency safety:
    pg_try_advisory_xact_lock keyed on hashtext(agent_id|user_id).
    Lock is transaction-scoped (auto-releases on commit/rollback) and held
    only during DB writes — not during summarisation or embedding — so
    transaction time stays short. Safe under multi-replica deployments.

Flow per (agent_id, user_id) pair:
    1. SELECT oldest SUMMARIZATION_BATCH_SIZE episodic memories
    2. Summarise locally (CPU, ~1 ms)
    3. Embed the summary (asyncio.to_thread, ~10–20 ms)
    4. BEGIN TRANSACTION
       a. pg_try_advisory_xact_lock
       b. INSERT summary as 'semantic' memory
       c. DELETE originals by id (ANY() — silent on already-deleted ids)
       d. COMMIT
    On any error: log, increment error counter, continue to next pair.
"""

import asyncio
import logging

from app.core.config import settings
from app.db.pool import get_pool
from app.services.embeddings import embed
from app.services.local_summarizer import summarize
from app.services.metrics import (
    SUMMARIZER_MEMORIES_CONDENSED_TOTAL,
    SUMMARIZER_RUNS_TOTAL,
)

logger = logging.getLogger(__name__)


async def _find_pairs_above_threshold() -> list[tuple[str, str]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT agent_id, user_id
            FROM memories
            WHERE memory_type = 'episodic'
            GROUP BY agent_id, user_id
            HAVING COUNT(*) > $1
            """,
            settings.summarization_threshold,
        )
    return [(r["agent_id"], r["user_id"]) for r in rows]


async def _fetch_oldest_episodic(agent_id: str, user_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, created_at
            FROM memories
            WHERE agent_id = $1 AND user_id = $2 AND memory_type = 'episodic'
            ORDER BY created_at ASC
            LIMIT $3
            """,
            agent_id,
            user_id,
            settings.summarization_batch_size,
        )
    return [dict(r) for r in rows]


async def _commit_summary(
    agent_id: str,
    user_id: str,
    ids_to_delete: list,
    summary_content: str,
    embedding: list[float],
) -> bool:
    """Returns False if advisory lock was not acquired (another worker is busy)."""
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    pool = get_pool()
    async with pool.acquire() as conn:
        lock_key = await conn.fetchval("SELECT hashtext($1)", f"{agent_id}|{user_id}")
        async with conn.transaction():
            if not await conn.fetchval("SELECT pg_try_advisory_xact_lock($1)", lock_key):
                return False
            await conn.execute(
                """
                INSERT INTO memories
                    (agent_id, user_id, content, embedding, importance, memory_type)
                VALUES ($1, $2, $3, $4::vector, 1.0, 'semantic')
                """,
                agent_id,
                user_id,
                summary_content,
                embedding_str,
            )
            await conn.execute(
                "DELETE FROM memories WHERE id = ANY($1::uuid[])",
                ids_to_delete,
            )
    return True


async def _summarize_pair(agent_id: str, user_id: str) -> int:
    """Returns number of memories condensed (0 if skipped)."""
    rows = await _fetch_oldest_episodic(agent_id, user_id)
    if len(rows) < 2:
        return 0

    raw_texts = [r["content"] for r in rows]
    summary_text = summarize(raw_texts, max_sentences=5)
    embedding = await embed(summary_text)

    committed = await _commit_summary(
        agent_id,
        user_id,
        ids_to_delete=[r["id"] for r in rows],
        summary_content=f"[Summary of {len(rows)} memories] {summary_text}",
        embedding=embedding,
    )
    if not committed:
        logger.debug("summarizer: lock not acquired for %s/%s, skipping", agent_id, user_id)
        return 0
    return len(rows)


async def run_once() -> None:
    """One full summarisation pass. Exported for integration tests."""
    pairs = await _find_pairs_above_threshold()
    if not pairs:
        return

    logger.info("summarizer: %d pair(s) above threshold", len(pairs))
    total_condensed = 0

    for agent_id, user_id in pairs:
        try:
            n = await _summarize_pair(agent_id, user_id)
            if n:
                total_condensed += n
                logger.info("summarizer: condensed %d memories for %s/%s", n, agent_id, user_id)
        except Exception:
            logger.error("summarizer: error processing %s/%s", agent_id, user_id, exc_info=True)
            SUMMARIZER_RUNS_TOTAL.labels(status="error").inc()
            continue

    if total_condensed:
        SUMMARIZER_MEMORIES_CONDENSED_TOTAL.inc(total_condensed)
    SUMMARIZER_RUNS_TOTAL.labels(status="ok").inc()


async def _loop() -> None:
    logger.info(
        "summarizer: started (interval=%ds, threshold=%d, batch=%d)",
        settings.summarization_interval_seconds,
        settings.summarization_threshold,
        settings.summarization_batch_size,
    )
    while True:
        try:
            await run_once()
        except Exception:
            logger.error("summarizer: unexpected error in run_once", exc_info=True)
        await asyncio.sleep(settings.summarization_interval_seconds)


def start() -> asyncio.Task:
    return asyncio.create_task(_loop(), name="summarizer")
