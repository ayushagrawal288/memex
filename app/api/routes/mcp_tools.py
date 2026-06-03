"""
MCP server — exposes memex as a tool for Claude agents.

Transport: Streamable HTTP, mounted at /mcp inside the FastAPI app.
  POST /mcp  — send MCP requests (initialize, tools/list, tools/call, ...)
  GET  /mcp  — open a long-lived SSE stream for server-initiated messages

Multi-worker safe: stateless transport — no sticky sessions required.
The MCP session manager's task group is started in the FastAPI lifespan.

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "memex": {
        "type": "streamable-http",
        "url": "http://localhost:8000/mcp"
      }
    }
  }

Claude Code:
  claude mcp add --transport http memex http://localhost:8000/mcp

Tools exposed:
  store_memory    — embed + persist a memory
  search_memories — semantic + recency ranked retrieval
  delete_memory   — forget a specific memory by ID
  count_memories  — how many memories an agent/user has
"""

from uuid import UUID

from mcp.server.fastmcp import FastMCP

from app.models.schemas import MemoryCreate, MemoryType
from app.services import memory as memory_svc

mcp_server = FastMCP(
    "memex",
    instructions="""
memex is a persistent memory service for AI agents.

Use store_memory at the end of a session to persist important facts, user
preferences, and conversation highlights. Use search_memories at the start
of a session (or whenever context is needed) to recall relevant history.

memory_type guide:
  episodic   — specific events or past conversations ("user asked about X")
  semantic   — facts, preferences, general knowledge ("user prefers Python")
  procedural — workflows or how-to instructions ("deploy by running Y")

importance (0.0–2.0, default 1.0):
  Use >1.0 for facts the agent must prioritise in future sessions.
  Use <1.0 for low-signal observations.

alpha (0.0–1.0, default 0.7):
  Controls the semantic vs recency blend during search.
  Higher → semantic similarity dominates.
  Lower → recent memories rank higher.
""".strip(),
)


@mcp_server.tool()
async def store_memory(
    agent_id: str,
    user_id: str,
    content: str,
    memory_type: str = "episodic",
    importance: float = 1.0,
) -> str:
    """
    Store a memory for an agent/user pair. Content is embedded locally (ONNX)
    and persisted to Postgres. Returns the memory ID on success.

    Args:
        agent_id:     Unique identifier for the agent storing the memory.
        user_id:      Unique identifier for the user this memory belongs to.
        content:      The text to remember (max 8000 chars).
        memory_type:  'episodic' (events/conversations), 'semantic' (facts/preferences),
                      or 'procedural' (workflows). Default: episodic.
        importance:   Weight multiplier 0.0–2.0. Use >1.0 for critical facts. Default: 1.0.
    """
    try:
        mem_type = MemoryType(memory_type)
    except ValueError:
        return (
            f"Invalid memory_type '{memory_type}'. "
            "Must be one of: episodic, semantic, procedural."
        )

    payload = MemoryCreate(
        agent_id=agent_id,
        user_id=user_id,
        content=content,
        memory_type=mem_type,
        importance=max(0.0, min(2.0, importance)),
    )
    result = await memory_svc.create_memory(payload)
    return (
        f"Stored memory {result.id} "
        f"(type={result.memory_type.value}, importance={result.importance})"
    )


@mcp_server.tool()
async def search_memories(
    agent_id: str,
    user_id: str,
    query: str,
    top_k: int = 5,
    alpha: float = 0.7,
    memory_type: str | None = None,
) -> str:
    """
    Search memories using semantic similarity + recency decay. Returns top-k
    ranked results with scores.

    Args:
        agent_id:     Agent whose memories to search.
        user_id:      User whose memories to search.
        query:        Natural language description of what to recall.
        top_k:        Number of memories to return (1–50). Default: 5.
        alpha:        Blend weight 0.0–1.0. 1.0 = pure semantic, 0.0 = pure recency.
                      Default: 0.7.
        memory_type:  Filter to 'episodic', 'semantic', or 'procedural'. Omit to search all.
    """
    mem_type: MemoryType | None = None
    if memory_type is not None:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            return (
                f"Invalid memory_type '{memory_type}'. "
                "Must be one of: episodic, semantic, procedural."
            )

    results = await memory_svc.search_memories(
        agent_id=agent_id,
        user_id=user_id,
        query=query,
        top_k=max(1, min(50, top_k)),
        alpha=max(0.0, min(1.0, alpha)),
        memory_type=mem_type,
    )

    if not results:
        return (
            f"No memories found for agent '{agent_id}' / user '{user_id}' "
            f"matching '{query}'."
        )

    lines = [f"Found {len(results)} {'memory' if len(results) == 1 else 'memories'} "
             f"(query: \"{query}\"):"]
    for i, m in enumerate(results, 1):
        lines.append(
            f"\n{i}. [{m.memory_type.value}] score={m.score:.4f}  id={m.id}\n"
            f"   {m.content}"
        )
    return "\n".join(lines)


@mcp_server.tool()
async def delete_memory(agent_id: str, memory_id: str) -> str:
    """
    Delete a specific memory by ID. The agent_id is used as an ownership check —
    agents cannot delete each other's memories.

    Args:
        agent_id:   Agent that owns the memory.
        memory_id:  UUID of the memory to delete (from store_memory or search_memories).
    """
    try:
        uid = UUID(memory_id)
    except ValueError:
        return f"Invalid memory_id '{memory_id}'. Must be a valid UUID."

    deleted = await memory_svc.delete_memory(uid, agent_id)
    if deleted:
        return f"Deleted memory {memory_id}."
    return (
        f"Memory {memory_id} not found or does not belong to agent '{agent_id}'."
    )


@mcp_server.tool()
async def count_memories(agent_id: str, user_id: str) -> str:
    """
    Return the total number of stored memories for an agent/user pair.

    Args:
        agent_id:  Agent identifier.
        user_id:   User identifier.
    """
    count = await memory_svc.get_memory_count(agent_id, user_id)
    noun = "memory" if count == 1 else "memories"
    return f"Agent '{agent_id}' / User '{user_id}' has {count} stored {noun}."


# Initialize the streamable HTTP session manager eagerly so that
# mcp_server.session_manager is accessible before the lifespan runs.
# The actual task group is started in app/main.py lifespan via session_manager.run().
mcp_server.streamable_http_app()  # noqa: F841 — side-effect: creates session_manager
