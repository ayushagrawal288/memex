"""
Standalone MCP server — runs as a separate single-process service on port 8001.

Why a separate process, not mounted in the REST API:
  The MCP Streamable HTTP transport is session-stateful — initialize, tools/call,
  and subsequent requests must all reach the same server instance. The REST API
  runs with multiple uvicorn workers (round-robin), which routes different requests
  to different processes, breaking session state.

  Running the MCP server as a single-worker service avoids sticky session
  complexity while keeping the REST API fully multi-worker.

Connection:
  POST http://localhost:8001/mcp/

Claude Desktop (~/.config/claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "memex": {
        "type": "streamable-http",
        "url": "http://localhost:8001/mcp/"
      }
    }
  }

Claude Code:
  claude mcp add --transport http memex http://localhost:8001/mcp/
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.mcp_tools import mcp_server
from app.core.config import settings
from app.db.pool import close_pool, init_pool, run_migrations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Thread pool for ONNX inference (same sizing as the REST API).
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=settings.worker_thread_pool_size))

    await init_pool()
    await run_migrations()

    # Start the MCP session manager's anyio task group.
    # This must wrap the yield so the task group is alive for the duration of the server.
    async with mcp_server.session_manager.run():
        yield

    await close_pool()


mcp_app = FastAPI(
    title="memex-mcp",
    description="MCP server — exposes memex tools to Claude agents.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,    # no docs on the MCP server
    redoc_url=None,
)


class _MCPHandler:
    """Thin ASGI wrapper around the MCP session manager's handle_request."""

    async def __call__(self, scope, receive, send):
        await mcp_server.session_manager.handle_request(scope, receive, send)


# Mount at /mcp/ (trailing slash avoids Starlette's 307 redirect on POST)
mcp_app.mount("/mcp/", _MCPHandler())
