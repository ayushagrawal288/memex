# memex

A production-grade persistent memory service for AI agents. Agents forget everything between sessions by default — memex fixes that. It stores, retrieves, and ranks conversation memory using semantic search with recency decay, so agents surface what's relevant *and* recent, not just what's semantically closest.

```
POST /v1/memories          → store a memory, embed it, persist to Postgres
POST /v1/memories/search   → retrieve top-k memories ranked by similarity + recency
DELETE /v1/memories/{id}   → forget a specific memory
GET  /v1/memories/count    → how many memories does this agent/user have
GET  /health               → liveness + DB connectivity check
GET  /metrics              → Prometheus metrics
```

---

## Architecture

```
caller (agent / app)
        │
        ▼
  FastAPI (async)
        │
   ┌────┴────┐
   │         │
embeddings  asyncpg pool (min=5, max=20)
(Anthropic  │
 voyage-3)  ▼
        PostgreSQL 16
          pgvector extension
          ivfflat index (cosine)
```

**Write path:** content → Anthropic embedding API (with retry + jitter) → INSERT with 1024-dim vector → return memory ID.

**Read path:** query → embed → pgvector cosine search (top_k × 3 candidates) → re-rank with recency decay in Python → return top_k results with scores.

---

## Design decisions

### 1. Recency decay on top of semantic search

Pure vector similarity returns the most semantically similar memories, not the most useful ones. A fact from 90 days ago that's a 0.95 similarity match is often less useful than a 0.80 match from yesterday.

Score formula:

```
score = α × cosine_similarity + (1 − α) × exp(−λ × age_days)
```

Where `λ = ln(2) / half_life_days` (default: 30 days, so a 30-day-old memory has 50% recency weight).

`α` is configurable per request (default 0.7). Task-focused agents use higher α (semantic dominates). Conversational agents use lower α (recency matters more).

### 2. Fetch 3× candidates, re-rank in Python

The pgvector query returns `top_k × 3` candidates sorted by pure similarity. Python re-ranks with the decay formula and slices to `top_k`. This prevents recency decay from starving high-similarity older memories — they're still in the candidate pool.

At 10× scale (>1M memories per agent): push the scoring into a Postgres function using `pg_proc` to eliminate the Python re-ranking round-trip.

### 3. asyncpg + explicit pool sizing over SQLAlchemy async

SQLAlchemy adds ORM overhead on every query. The hot retrieval path — embed, query, re-rank — needs to be tight. asyncpg gives direct control over pool min/max (same instinct as tuning HikariCP in Java). pgvector queries require raw SQL for the `<=>` operator anyway.

Pool defaults: `min=5, max=20`. Right-size for a single-instance deployment. Override via `DB_MAX_POOL_SIZE` env var.

### 4. Rate limiting in Postgres, not Redis

Sliding window counter via upsert. One fewer dependency. Correct under concurrent requests (transactional upsert). At 10× scale with distributed deployments: replace with Redis `INCR + EXPIRE` — atomic operations, no lock contention.

### 5. ivfflat index, not HNSW

`ivfflat` has lower build cost and lower memory footprint — the right tradeoff at small-to-medium scale (<1M vectors). `lists=100` works well up to ~1M rows. At 10× scale: switch to HNSW (`m=16, ef_construction=64`) for better recall at the cost of higher memory and build time.

---

## Running locally

**Prerequisites:** Docker, Docker Compose, an Anthropic API key.

```bash
git clone https://github.com/ayushagrawal288/memex
cd memex
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
docker compose up
```

The API is live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API reference

### Store a memory

```bash
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "user_id": "user-123",
    "content": "User prefers concise responses and dislikes verbose explanations.",
    "memory_type": "semantic",
    "importance": 1.2
  }'
```

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "agent_id": "my-agent",
  "user_id": "user-123",
  "content": "User prefers concise responses and dislikes verbose explanations.",
  "importance": 1.2,
  "memory_type": "semantic",
  "created_at": "2026-05-26T10:30:00Z",
  "score": null
}
```

### Search memories

```bash
curl -X POST http://localhost:8000/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "user_id": "user-123",
    "query": "how does this user like to communicate",
    "top_k": 5,
    "alpha": 0.7
  }'
```

```json
{
  "results": [
    {
      "id": "3fa85f64-...",
      "content": "User prefers concise responses and dislikes verbose explanations.",
      "memory_type": "semantic",
      "created_at": "2026-05-26T10:30:00Z",
      "score": 0.8921
    }
  ],
  "query": "how does this user like to communicate",
  "total": 1
}
```

### Memory types

| Type | Use for |
|---|---|
| `episodic` | Specific events, past conversations |
| `semantic` | Facts, preferences, general knowledge |
| `procedural` | Workflows, how-to instructions |

---

## Load test results

Run on a MacBook M-series, Docker Desktop, single Postgres instance:

```bash
locust -f scripts/load_test.py --host=http://localhost:8000 \
       --headless -u 50 -r 10 -t 60s
```

**Realistic load** (50 users, 100–300 ms think time — models actual agent traffic):

| Endpoint | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|---|---|---|---|---|---|
| POST /v1/memories (write) | 27 | 160 | 270 | 330 | 0% |
| POST /v1/memories/search | 83 | 110 | 200 | 250 | 0% |
| Aggregated | **113** | 120 | 230 | 300 | **0%** |

**Saturation test** (500 users, minimal think time — finds the throughput ceiling):

| Endpoint | RPS (plateau) | p50 (ms) | p99 (ms) | Error rate |
|---|---|---|---|---|
| POST /v1/memories (write) | 28 | 3,900 | 6,100 | **0%** |
| POST /v1/memories/search | 91 | 3,600 | 5,800 | **0%** |
| Aggregated | **~120** | 3,700 | 5,900 | **0%** |

> Run on MacBook M-series, Docker Desktop (4 CPUs), 4 uvicorn workers, 16 threads/worker.  
> Embeddings: local ONNX (`BAAI/bge-small-en-v1.5`) — zero external API calls, zero cost.

**Why the ceiling is ~120 RPS:**  
Every write and every search requires one ONNX inference (~10–15 ms on CPU). With 4 Docker CPUs: `4 cores / 12 ms ≈ 333 embeddings/s` theoretical max. After Python overhead, DB queries, and asyncio scheduling: ~120 RPS actual.

**Path to higher throughput:**

| Approach | Expected gain | Complexity |
|---|---|---|
| Embedding cache (Redis, key = SHA256 of text) | 2–3× (40–60% hit rate on repeated agent queries) | Low |
| Horizontal scaling (N replicas behind a load balancer) | N× linear | Medium |
| GPU inference (swap ONNX runtime → CUDA) | 10–50× | Medium |
| Voyage-3 API (offload to Anthropic's inference fleet) | Scales to thousands of RPS, limited by API quota | Low code change |

---

## Project structure

```
memex/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   ├── core/
│   │   └── config.py            # All settings, loaded from env
│   ├── db/
│   │   └── pool.py              # asyncpg pool, migrations
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── services/
│   │   ├── embeddings.py        # Anthropic embedding API + retry
│   │   ├── memory.py            # Core write/search/scoring logic
│   │   └── rate_limit.py        # Sliding window rate limiter
│   └── api/routes/
│       ├── memories.py          # Memory endpoints
│       └── health.py            # Health + readiness
├── scripts/
│   └── load_test.py             # Locust load test
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Observability

`docker compose up` starts Prometheus and Grafana alongside the API:

| Service | URL | Credentials |
|---|---|---|
| API docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The Grafana dashboard is provisioned automatically. Panels:

- **HTTP request rate + latency p50/p99** — from `prometheus-fastapi-instrumentator`
- **Embedding API latency p50/p99** — per-attempt histogram by operation (`embed` / `embed_batch`)
- **Memory operations/s** — create, search, delete throughput
- **DB pool utilisation** — active vs idle connections (update interval: 15 s)
- **Summariser activity** — memories condensed per hour, run outcomes
- **Embedding errors/min** — by operation and error type

Custom metrics are in `app/services/metrics.py` and exposed on `/metrics` alongside the standard FastAPI instrumentator metrics.

---

## Memory summarisation

Runs as a background asyncio task on a configurable interval (default: every 5 minutes). Finds any `(agent_id, user_id)` pair where episodic memory count exceeds a threshold, condenses the oldest batch into a single `semantic` memory using Claude Haiku, then deletes the originals.

**Why episodic-only:** Episodic memories are conversation events with natural time-based obsolescence. Semantic and procedural memories encode facts and skills — silently condensing them risks precision loss; they age out via recency decay instead.

**Concurrency safety:** Uses `pg_try_advisory_xact_lock` keyed on `hashtext(agent_id|user_id)`. The lock is held only during the DB write transaction, not during the Claude or embedding API calls.

Tune via env vars:

| Var | Default | Description |
|---|---|---|
| `SUMMARIZATION_ENABLED` | `true` | Toggle the background job |
| `SUMMARIZATION_THRESHOLD` | `100` | Episodic count to trigger per pair |
| `SUMMARIZATION_BATCH_SIZE` | `50` | Oldest N memories to condense per run |
| `SUMMARIZATION_INTERVAL_SECONDS` | `300` | How often the job wakes up |
| `SUMMARIZATION_MODEL` | `claude-haiku-4-5-20251001` | Claude model for summarisation |

---

## What's next

- [x] **Memory summarisation** — background job to condense old memories using Claude when count exceeds threshold, keeping context windows lean
- [x] **Prometheus + Grafana** — p50/p99 latency dashboards, embedding API call duration, pool saturation
- [ ] **MCP-compatible endpoint** — expose memex as a Claude tool so any agent using MCP can plug in without custom integration
- [ ] **HNSW index option** — flag to switch from ivfflat to HNSW for deployments with >1M vectors
- [ ] **Importance-weighted retrieval** — factor `importance` score into ranking formula alongside similarity and recency

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async-first, fast, excellent OpenAPI generation |
| Embeddings | Anthropic voyage-3 | 1024-dim, strong semantic quality |
| Database | PostgreSQL 16 + pgvector | Relational + vector in one system, no extra infra |
| Vector index | ivfflat | Lower build cost than HNSW at this scale |
| Pool | asyncpg | Direct control, zero ORM overhead |
| Retry | tenacity | Jitter-based backoff on embedding API calls |
| Metrics | Prometheus + prometheus-fastapi-instrumentator | Standard observability |
| Load testing | Locust | Python-native, realistic user simulation |
