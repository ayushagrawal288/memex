"""
Custom Prometheus metrics for memex.

prometheus-fastapi-instrumentator handles HTTP-level metrics (latency by handler,
status codes, request rate). These metrics cover the internals that are invisible
at the HTTP layer:

  - Embedding API latency + errors — the most expensive external call, and the
    most likely bottleneck under load. A p99 spike here shows up as HTTP latency
    but without this histogram you can't tell whether the problem is embedding,
    DB, or re-ranking.

  - Memory op counters — write/search/delete throughput; useful for capacity
    planning and for correlating latency spikes with traffic patterns.

  - DB pool gauges — pool saturation is the first thing to check when latency
    climbs under load. Updated every 15s by a background task in main.py.

  - Summariser outcomes — how many runs, how many memories condensed, error rate.
"""

from prometheus_client import Counter, Gauge, Histogram

EMBEDDING_LATENCY = Histogram(
    "memex_embedding_duration_seconds",
    "Embedding API call duration (per attempt, excludes tenacity retry overhead)",
    labelnames=["operation"],  # "embed" | "embed_batch"
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

EMBEDDING_ERRORS_TOTAL = Counter(
    "memex_embedding_errors_total",
    "Embedding API failures",
    labelnames=["operation", "error_type"],
)

MEMORY_OPS_TOTAL = Counter(
    "memex_memory_operations_total",
    "Memory CRUD operations",
    labelnames=["operation"],  # "create" | "search" | "delete"
)

# Updated every 15 s by the pool-metrics background task in main.py.
POOL_SIZE = Gauge("memex_db_pool_size", "Total connections held by asyncpg pool")
POOL_IDLE = Gauge("memex_db_pool_idle", "Idle connections in asyncpg pool")

SUMMARIZER_RUNS_TOTAL = Counter(
    "memex_summarizer_runs_total",
    "Summariser job execution outcomes",
    labelnames=["status"],  # "ok" | "error"
)

SUMMARIZER_MEMORIES_CONDENSED_TOTAL = Counter(
    "memex_summarizer_memories_condensed_total",
    "Individual episodic memories condensed into summary memories",
)
