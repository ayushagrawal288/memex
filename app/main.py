"""
memex — production-grade memory service for AI agents

Startup sequence mirrors what you'd do for any service going to prod:
1. Validate config (pydantic-settings raises on missing required vars)
2. Initialise connection pool
3. Run migrations
4. Mount routes
5. Wire up observability
6. Start background tasks (summariser, pool-metrics collector)

Shutdown sequence is explicit — don't rely on GC to close pool connections.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import health, memories
from app.core.config import settings
from app.db.pool import close_pool, get_pool, init_pool, run_migrations
from app.services import summarizer
from app.services.metrics import POOL_IDLE, POOL_SIZE

logger = logging.getLogger(__name__)


async def _pool_metrics_loop() -> None:
    """Update pool gauges every 15 s — cheap enough to run continuously."""
    while True:
        pool = get_pool()
        POOL_SIZE.set(pool.get_size())
        POOL_IDLE.set(pool.get_idle_size())
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-size the thread pool that asyncio.to_thread() uses for embedding calls.
    # The default (min(32, cpu_count+4)) is too small under high concurrency.
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=settings.worker_thread_pool_size))

    await init_pool()
    await run_migrations()

    background_tasks: list[asyncio.Task] = []
    background_tasks.append(
        asyncio.create_task(_pool_metrics_loop(), name="pool-metrics")
    )
    if settings.summarization_enabled:
        background_tasks.append(summarizer.start())

    yield

    # shutdown — cancel tasks before closing pool so they don't race with close_pool
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)

    await close_pool()


app = FastAPI(
    title="memex",
    description="Production-grade persistent memory service for AI agents.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Prometheus metrics on /metrics
Instrumentator().instrument(app).expose(app)

app.include_router(health.router)
app.include_router(memories.router, prefix="/v1")


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
    }
