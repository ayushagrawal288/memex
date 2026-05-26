"""
Memory service — the core of memex.

Retrieval scoring formula:
    score = α × cosine_similarity + (1 - α) × recency_weight

Where recency_weight uses exponential decay:
    recency_weight = exp(-λ × age_in_days)
    λ = ln(2) / half_life_days  (so weight = 0.5 at half_life_days)

α is configurable per-request (default 0.7).
This lets callers tune the tradeoff: task-focused agents want higher α
(pure semantic), conversational agents want lower α (recency matters more).

Design decision: scoring happens in Python, not SQL, because:
- The decay formula is easy to change without a migration
- At current scale (<10k memories per agent) the extra round-trip doesn't matter
- At 10x scale: push scoring into a Postgres function to eliminate Python overhead
"""

import math
from datetime import datetime, timezone
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.db.pool import get_pool
from app.models.schemas import MemoryCreate, MemoryResponse, MemoryType
from app.services.embeddings import embed
from app.services.metrics import MEMORY_OPS_TOTAL


_DECAY_LAMBDA = math.log(2) / settings.recency_decay_days


def _recency_weight(created_at: datetime) -> float:
    age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
    age_days = age_seconds / 86400
    return math.exp(-_DECAY_LAMBDA * age_days)


def _score(similarity: float, created_at: datetime, alpha: float) -> float:
    return alpha * similarity + (1 - alpha) * _recency_weight(created_at)


async def create_memory(data: MemoryCreate) -> MemoryResponse:
    embedding = await embed(data.content)
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories (agent_id, user_id, content, embedding, importance, memory_type)
            VALUES ($1, $2, $3, $4::vector, $5, $6)
            RETURNING id, agent_id, user_id, content, importance, memory_type, created_at
            """,
            data.agent_id,
            data.user_id,
            data.content,
            embedding_str,
            data.importance,
            data.memory_type.value,
        )

    MEMORY_OPS_TOTAL.labels(operation="create").inc()
    return MemoryResponse(
        id=row["id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        content=row["content"],
        importance=row["importance"],
        memory_type=MemoryType(row["memory_type"]),
        created_at=row["created_at"],
        score=None,
    )


async def search_memories(
    agent_id: str,
    user_id: str,
    query: str,
    top_k: int = 10,
    alpha: float = 0.7,
    memory_type: MemoryType | None = None,
) -> list[MemoryResponse]:
    query_embedding = await embed(query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    pool = get_pool()
    async with pool.acquire() as conn:
        # Fetch top_k * 3 candidates by similarity, then re-rank with decay in Python.
        # Fetching extra candidates ensures recency doesn't starve relevant older memories.
        candidate_limit = min(top_k * 3, 150)

        type_filter = "AND memory_type = $5" if memory_type else ""
        params: list = [agent_id, user_id, embedding_str, candidate_limit]
        if memory_type:
            params.append(memory_type.value)

        rows = await conn.fetch(
            f"""
            SELECT
                id, agent_id, user_id, content, importance, memory_type, created_at,
                1 - (embedding <=> $3::vector) AS similarity
            FROM memories
            WHERE agent_id = $1 AND user_id = $2
            {type_filter}
            ORDER BY embedding <=> $3::vector
            LIMIT $4
            """,
            *params,
        )

    scored = [
        (
            row,
            _score(
                similarity=float(row["similarity"]),
                created_at=row["created_at"],
                alpha=alpha,
            ),
        )
        for row in rows
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    MEMORY_OPS_TOTAL.labels(operation="search").inc()
    return [
        MemoryResponse(
            id=row["id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            content=row["content"],
            importance=float(row["importance"]),
            memory_type=MemoryType(row["memory_type"]),
            created_at=row["created_at"],
            score=round(score, 4),
        )
        for row, score in top
    ]


async def delete_memory(memory_id: UUID, agent_id: str) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM memories WHERE id = $1 AND agent_id = $2",
            memory_id,
            agent_id,
        )
    deleted = result == "DELETE 1"
    if deleted:
        MEMORY_OPS_TOTAL.labels(operation="delete").inc()
    return deleted


async def get_memory_count(agent_id: str, user_id: str) -> int:
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE agent_id = $1 AND user_id = $2",
            agent_id,
            user_id,
        )
