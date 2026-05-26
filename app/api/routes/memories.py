from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    MemoryCreate,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)
from app.services import memory as memory_svc
from app.services.rate_limit import check_rate_limit

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(payload: MemoryCreate) -> MemoryResponse:
    allowed, count = await check_rate_limit(payload.agent_id, "write")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Write rate limit exceeded ({count} writes this minute). Retry in the next minute.",
            headers={"Retry-After": "60"},
        )

    try:
        return await memory_svc.create_memory(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to create memory: {exc}",
        ) from exc


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(payload: MemorySearchRequest) -> MemorySearchResponse:
    allowed, count = await check_rate_limit(payload.agent_id, "search")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Search rate limit exceeded ({count} searches this minute).",
            headers={"Retry-After": "60"},
        )

    try:
        results = await memory_svc.search_memories(
            agent_id=payload.agent_id,
            user_id=payload.user_id,
            query=payload.query,
            top_k=payload.top_k,
            alpha=payload.alpha,
            memory_type=payload.memory_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Search failed: {exc}",
        ) from exc

    return MemorySearchResponse(results=results, query=payload.query, total=len(results))


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: UUID, agent_id: str) -> None:
    deleted = await memory_svc.delete_memory(memory_id, agent_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or does not belong to this agent.",
        )


@router.get("/count", response_model=dict)
async def count_memories(agent_id: str, user_id: str) -> dict:
    count = await memory_svc.get_memory_count(agent_id, user_id)
    return {"agent_id": agent_id, "user_id": user_id, "count": count}
