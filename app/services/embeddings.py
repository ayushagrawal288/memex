"""
Local embedding service using fastembed (ONNX runtime, no GPU required).

Zero API calls, zero cost — runs entirely in-process.

Model: BAAI/bge-small-en-v1.5
  - 384 dimensions
  - ~67 MB on disk (pre-baked into the Docker image via Dockerfile)
  - ~5-15 ms per text on CPU
  - fastembed uses ONNX runtime — no PyTorch, no CUDA, ~100 MB total footprint
    vs 1+ GB for sentence-transformers + torch

To switch to Voyage-3 for production:
  1. Replace embed() with Voyage API client
  2. Update embedding_dimensions = 1024 in config
  3. Rebuild + wipe DB volume (dimension change requires schema migration)
"""

import asyncio
import time
from functools import lru_cache

from fastembed import TextEmbedding

from app.services.metrics import EMBEDDING_ERRORS_TOTAL, EMBEDDING_LATENCY


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _encode_sync(text: str) -> list[float]:
    return next(_model().embed([text])).tolist()


def _encode_batch_sync(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]


async def embed(text: str) -> list[float]:
    t0 = time.perf_counter()
    try:
        # fastembed is CPU-bound + sync — run in thread to avoid blocking event loop
        return await asyncio.to_thread(_encode_sync, text)
    except Exception as exc:
        EMBEDDING_ERRORS_TOTAL.labels(operation="embed", error_type=type(exc).__name__).inc()
        raise
    finally:
        EMBEDDING_LATENCY.labels(operation="embed").observe(time.perf_counter() - t0)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    t0 = time.perf_counter()
    try:
        return await asyncio.to_thread(_encode_batch_sync, texts)
    except Exception as exc:
        EMBEDDING_ERRORS_TOTAL.labels(operation="embed_batch", error_type=type(exc).__name__).inc()
        raise
    finally:
        EMBEDDING_LATENCY.labels(operation="embed_batch").observe(time.perf_counter() - t0)
