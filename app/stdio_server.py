"""
Stdio transport entry point — used by Glama's mcp-proxy deployment system.

FastMCP supports both HTTP and stdio transports from the same tool definitions.
This module starts the same 4 tools (store_memory, search_memories, delete_memory,
count_memories) over stdio so Glama can run security checks and tool-list validation
without requiring the HTTP server or the PostgreSQL pool.

Note: tools that perform DB operations will return an error if DATABASE_URL is not
      reachable. Glama's build test only calls initialize + tools/list, which require
      no DB connection.

Usage (Glama mcp-proxy subprocess mode):
    mcp-proxy -- python3 -m app.stdio_server
"""

from app.api.routes.mcp_tools import mcp_server

if __name__ == "__main__":
    mcp_server.run()
