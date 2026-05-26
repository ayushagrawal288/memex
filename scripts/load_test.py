"""
memex load test — two user classes:

  MemexUser           — realistic: 100-300 ms think time
                        Models actual agent traffic. Good for p50/p99 benchmarks.

  MemexSaturationUser — minimal think time
                        Finds the throughput ceiling. Good for RPS benchmarks.

Usage:

  # Realistic benchmark (p50/p99 under real agent load):
  locust -f scripts/load_test.py --host=http://localhost:8000 \\
         --headless -u 50 -r 10 -t 60s

  # Saturation / max-RPS benchmark:
  locust -f scripts/load_test.py --host=http://localhost:8000 \\
         --headless -u 500 -r 50 -t 60s

Design:
  - A test_start hook seeds a shared pool of (agent, user) pairs once before
    any virtual user runs. Virtual users draw from this pool and immediately
    start issuing reads/writes — no per-user seeding thundering-herd.
  - Traffic shape: 75 % reads (search), 25 % writes — matches real agent usage.
"""

import random

import requests
from locust import HttpUser, between, constant_throughput, events, task

# ---------------------------------------------------------------------------
# Shared agent/user pool — seeded once by test_start hook
# ---------------------------------------------------------------------------

_NUM_AGENTS = 20
_NUM_USERS = 10

AGENT_IDS = [f"load-agent-{i:03d}" for i in range(_NUM_AGENTS)]
USER_IDS = [f"load-user-{i:03d}" for i in range(_NUM_USERS)]

SAMPLE_MEMORIES = [
    "The user prefers concise responses and dislikes verbose explanations.",
    "User is building a FastAPI service and is comfortable with Python async.",
    "Last session: discussed connection pooling strategies for PostgreSQL.",
    "User's timezone is IST (UTC+5:30). Schedule references should account for this.",
    "User prefers code examples over abstract descriptions.",
    "Previous conversation touched on HikariCP tuning in Java Spring Boot.",
    "User is targeting backend engineering roles at product startups and FAANG.",
    "The user has 7 years of Java experience and is learning Python.",
    "User mentioned they worked at CRED and Wayfair previously.",
    "Preferred communication style: direct, no pleasantries, get to the point.",
    "User is interested in distributed systems and database internals.",
    "Last session covered pgvector ivfflat vs HNSW index trade-offs.",
    "User wants to showcase this project for senior backend roles.",
    "User understands async I/O well; explain blocking vs non-blocking concretely.",
    "Prefers minimal comments in code — trusts well-named identifiers.",
]

SAMPLE_QUERIES = [
    "what are the user's communication preferences",
    "what did we discuss last session",
    "what is the user's technical background",
    "does the user prefer Python or Java",
    "what timezone is the user in",
    "what projects is the user working on",
    "what databases does the user work with",
    "what is the user's experience level",
]


def _write_one(host: str, agent_id: str, user_id: str, content: str) -> bool:
    try:
        r = requests.post(
            f"{host}/v1/memories",
            json={"agent_id": agent_id, "user_id": user_id,
                  "content": content, "memory_type": "episodic"},
            timeout=30,
        )
        return r.status_code == 201
    except Exception:
        return False


@events.test_start.add_listener
def seed_shared_pool(environment, **kwargs):
    """
    Seed 20 agents × 10 users × 5 memories = 1,000 memories before ramp-up.
    Done in parallel (32 threads) so seeding takes ~5 s instead of ~40 s.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    host = environment.host
    jobs = [
        (host, agent_id, user_id, content)
        for agent_id in AGENT_IDS
        for user_id in USER_IDS
        for content in random.sample(SAMPLE_MEMORIES, 5)
    ]
    print(f"\n[seed] Writing {len(jobs)} memories with 32 threads...")
    seeded = errors = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_write_one, *j): j for j in jobs}
        for f in as_completed(futures):
            if f.result():
                seeded += 1
            else:
                errors += 1
    print(f"[seed] done — {seeded} seeded, {errors} errors\n")


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class MemexUser(HttpUser):
    """Realistic agent — think time models real inter-request gaps."""
    wait_time = between(0.1, 0.3)

    def on_start(self):
        self.agent_id = random.choice(AGENT_IDS)
        self.user_id = random.choice(USER_IDS)

    @task(3)
    def search_memories(self):
        self.client.post(
            "/v1/memories/search",
            json={
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "query": random.choice(SAMPLE_QUERIES),
                "top_k": 5,
                "alpha": 0.7,
            },
            name="/v1/memories/search",
        )

    @task(1)
    def create_memory(self):
        self.client.post(
            "/v1/memories",
            json={
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "content": random.choice(SAMPLE_MEMORIES),
                "importance": round(random.uniform(0.5, 1.5), 2),
                "memory_type": random.choice(["episodic", "semantic"]),
            },
            name="/v1/memories",
        )


class MemexSaturationUser(HttpUser):
    """No think time — drives the service to its throughput ceiling."""
    wait_time = constant_throughput(20)  # each user targets 20 req/s

    def on_start(self):
        self.agent_id = random.choice(AGENT_IDS)
        self.user_id = random.choice(USER_IDS)

    @task(3)
    def search_memories(self):
        self.client.post(
            "/v1/memories/search",
            json={
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "query": random.choice(SAMPLE_QUERIES),
                "top_k": 5,
                "alpha": 0.7,
            },
            name="/v1/memories/search",
        )

    @task(1)
    def create_memory(self):
        self.client.post(
            "/v1/memories",
            json={
                "agent_id": self.agent_id,
                "user_id": self.user_id,
                "content": random.choice(SAMPLE_MEMORIES),
                "importance": round(random.uniform(0.5, 1.5), 2),
                "memory_type": random.choice(["episodic", "semantic"]),
            },
            name="/v1/memories",
        )
