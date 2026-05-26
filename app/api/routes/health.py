from fastapi import APIRouter

from app.db.pool import get_pool
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
    )
