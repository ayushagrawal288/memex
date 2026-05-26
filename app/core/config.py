from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # database
    database_url: str = "postgresql://memex:memex@localhost:5432/memex"
    db_min_pool_size: int = 5
    db_max_pool_size: int = 20          # tunable — matches HikariCP instinct
    db_command_timeout: float = 10.0    # fail fast, don't queue forever

    # embeddings — local ONNX model, no API key required
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # retrieval
    default_top_k: int = 10
    default_alpha: float = 0.7          # weight: 0=pure recency, 1=pure similarity
    recency_decay_days: float = 30.0    # half-life for recency scoring

    # rate limiting (per agent_id, per minute)
    rate_limit_writes_per_minute: int = 60
    rate_limit_searches_per_minute: int = 120

    # summarisation background job (fully local — no API calls)
    summarization_enabled: bool = True
    summarization_threshold: int = 100       # trigger per (agent_id, user_id) pair
    summarization_batch_size: int = 50       # oldest N episodic memories to condense
    summarization_interval_seconds: int = 300

    # concurrency — tune for your machine
    # worker_thread_pool_size controls asyncio.to_thread() parallelism per worker process.
    # With N uvicorn workers and T threads each: N×T concurrent embeddings possible.
    # Default 16 threads × 4 workers = 64 parallel embeddings at ~12 ms each → ~5,000 emb/s.
    worker_thread_pool_size: int = 16

    # app
    app_name: str = "memex"
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # tolerate unknown env vars (e.g. ANTHROPIC_API_KEY from old .env)


settings = Settings()
