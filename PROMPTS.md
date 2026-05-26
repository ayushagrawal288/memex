# How memex was built — session-by-session prompt log

memex was built across three Claude Code sessions. This document captures the key prompts, the reasoning behind design decisions, and the pivots that shaped the final architecture. It's part technical diary, part prompt engineering reference.

---

## Session 1 — Core service

**Goal:** Stand up a production-grade memory API from scratch.

### Opening prompt

> I want to build a production-grade persistent memory service for AI agents called memex.
>
> Stack: FastAPI + PostgreSQL + pgvector.
>
> Endpoints needed:
> - `POST /v1/memories` — store a memory, embed it, persist to Postgres
> - `POST /v1/memories/search` — retrieve top-k memories ranked by similarity + recency
> - `DELETE /v1/memories/{id}` — forget a specific memory
> - `GET /v1/memories/count` — how many memories does this agent/user have
> - `GET /health` — liveness + DB connectivity check
>
> Design requirements (non-negotiable):
> 1. Recency decay on top of cosine similarity. Score = α × cosine + (1−α) × exp(−λ × age_days). α configurable per request (default 0.7). Half-life 30 days.
> 2. Fetch 3× candidates from pgvector, re-rank in Python. Prevents recency from starving high-similarity older memories.
> 3. asyncpg, not SQLAlchemy. Direct pool control — same instinct as tuning HikariCP in Java. Pool min=5, max=20.
> 4. Rate limiting per agent_id in Postgres (sliding window upsert). No Redis dependency.
> 5. ivfflat index (lists=100), not HNSW. Lower build cost at this scale. Explicit comment on when to switch.
> 6. `vector(1024)` for Voyage-3 embeddings.
>
> Write production-quality code — proper error handling, Pydantic v2 models, typed everything, async throughout.

### Key follow-up prompts

**On the embedding service:**
> Use Anthropic's embedding API (voyage-3, 1024-dim). Wrap in tenacity retry with jitter. expose `embed()` and `embed_batch()`.

**On the DB migration:**
> The pool.py should handle schema creation on startup — CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS. No alembic, keep it simple.

**On rate limiting:**
> Sliding window: upsert a counter keyed on (agent_id, window_start). Transactional. Raise 429 with a clear message when exceeded.

---

## Session 2 — Observability + summarisation + going fully local

**Goal:** Add production observability, a memory summarisation background job, and eliminate all external API dependencies.

### Opening prompt

> I've been building a project called memex — a production-grade persistent memory service for AI agents. FastAPI + PostgreSQL + pgvector + Anthropic embeddings. The full codebase is already here. Read the README and the file structure first, then we'll continue with Session 2: memory summarisation background job + Prometheus metrics.
>
> Design notes for Session 2:
>
> **Memory summarisation:**
> - Background asyncio task, configurable interval (default 5 min)
> - Trigger: any (agent_id, user_id) pair where episodic memory count > threshold (default 100)
> - Condense oldest 50 episodic memories into a single semantic memory using Claude Haiku
> - Delete the originals after successful insert
> - Concurrency safety: `pg_try_advisory_xact_lock` keyed on `hashtext(agent_id|user_id)`. Lock held only during the DB write transaction, not during Claude or embedding API calls.
> - Episodic-only — semantic/procedural memories encode facts and skills, silently condensing them risks precision loss
>
> **Prometheus + Grafana:**
> - Custom metrics: embedding latency histogram, memory ops counter (create/search/delete), DB pool utilisation gauge, summariser activity counter
> - `prometheus-fastapi-instrumentator` for HTTP-level metrics
> - Grafana provisioned via YAML — dashboard.json committed, not hand-drawn

### The API-cost pivot

After the core service was wired up, we ran it and hit the first real constraint:

> **WHAT ARE we calling anthropic for? and do we need to this? We need to ensure that we don't exhaust my AI credits through this project and load testing.**

This triggered a full architectural pivot. The Voyage-3 embedding API was called on every write and every search — at 100 RPS that's 100 API calls/second against a paid endpoint. The summariser was calling Claude Haiku on every summarisation cycle. Neither was acceptable for a load test.

**Embedding pivot prompt:**
> yes. let's ensure that everything is running on the local machine for effective load test results.

**Summariser pivot prompt:**
> for summariser also let's find a local alternative. And we need to load test this upto 6k RPS

### How the local stack was chosen

**Embeddings — why fastembed over sentence-transformers:**

The first attempt used `sentence-transformers`. Docker build immediately pulled PyTorch + CUDA — 426 MB + 444 MB, a ~1 GB image just for embeddings. Killed it. `fastembed` uses ONNX Runtime directly: no PyTorch, no CUDA, ~100 MB total footprint, same BAAI/bge-small-en-v1.5 model.

Tradeoff documented: 384-dim vs 1024-dim (Voyage-3). Recall is slightly lower but similarity ordering holds for the re-ranking approach we use. Acceptable.

**Summariser — why extractive over abstractive:**

Abstractive summarisation (any transformer-based approach) adds 200–500 MB of model weight and needs GPU or slow CPU inference. For a memory service, the goal is deduplication and compression — not creative rewriting.

Pure Python extractive approach:
1. Split memories into sentences (>20 chars)
2. Deduplicate by Jaccard similarity ≥ 0.7
3. Score each unique sentence by word frequency (TF, no IDF — the corpus is tiny)
4. Take top-N, restore original order (preserves temporal flow)
5. Return as a paragraph

Zero ML dependencies. ~1 ms per summarisation. Zero API calls.

**Thread pool sizing:**

> With N uvicorn workers and T threads each: N×T concurrent embeddings possible.

`asyncio.to_thread()` dispatches CPU-bound ONNX inference off the event loop. Default executor replaced with `ThreadPoolExecutor(max_workers=16)`. With 4 uvicorn workers: 64 parallel embedding threads, ~333 embeddings/s theoretical ceiling on 4 Docker CPUs.

### Load test design

**What we discovered through iteration:**

First run: per-user seeding (each of 1500 virtual users seeded 3 memories on startup) → thundering herd, 38% error rate, 26-second latencies.

Fix: moved seeding to `@events.test_start.add_listener` — a single hook that runs once before any virtual user spawns, using 32 parallel threads. 1,000 memories seeded in ~5 seconds.

Second run: rate limiter fired (429 errors). 20 shared agent IDs × 500 users → each agent hit at ~25× the rate limit. Fix: `RATE_LIMIT_*=999999` in docker-compose for load testing.

Third run: `ModuleNotFoundError: No module named 'fastembed'` — stale Docker image still running the old sentence-transformers build. Fix: `docker compose build api && docker compose up -d --force-recreate api`.

**Final results (500 users, 90s duration, zero errors):**

| Endpoint | RPS | p50 | p99 |
|---|---|---|---|
| POST /v1/memories | ~28 | 3,900ms | 6,100ms |
| POST /v1/memories/search | ~91 | 3,600ms | 5,800ms |
| Aggregated | **~120** | 3,700ms | 5,900ms |

**Why ~120 RPS is the ceiling:**

```
4 Docker CPUs × (1000ms / 12ms ONNX inference) ≈ 333 embeddings/s theoretical
After asyncio overhead + DB queries + scheduling: ~120 RPS actual
```

This is an honest ceiling — not a bug. The path to 6,000 RPS requires embedding cache (2-3× from ~40% hit rate on repeated agent queries), GPU inference (10-50×), or horizontal scaling (N× linear).

---

## Prompting patterns that worked

### 1. Leading with constraints, not requirements

Prompts that opened with non-negotiables ("asyncpg not SQLAlchemy", "ivfflat not HNSW", "Postgres rate limiting not Redis") got better first-draft code than prompts that listed features. Constraints eliminate the solution space before code is written.

### 2. Asking for the "why" in comments

> Explicit comment on when to switch [from ivfflat to HNSW].

Every design decision in the codebase has a comment explaining the tradeoff and the scale at which the decision reverses. This is what makes the code readable as a portfolio artifact, not just functional software.

### 3. Calling out the real constraint immediately

The API-cost pivot happened because the constraint was stated directly and early: *"We need to ensure that we don't exhaust my AI credits."* Stating it once, clearly, caused everything downstream — embedding choice, summariser design, load test configuration — to flow from that constraint.

### 4. Trusting iteration over specification

The load test wasn't designed correctly on the first try. Three bugs were found and fixed through actual runs, not upfront analysis. A more exhaustively specified prompt upfront would have been slower. Run fast, read the errors, fix the specific thing.

---

## What's next

From the README "What's next" section:

- [ ] **MCP-compatible endpoint** — expose memex as a Claude tool so any MCP-aware agent can plug in without custom integration
- [ ] **HNSW index option** — flag to switch from ivfflat to HNSW for deployments with >1M vectors
- [ ] **Importance-weighted retrieval** — factor `importance` score into ranking formula alongside similarity and recency
