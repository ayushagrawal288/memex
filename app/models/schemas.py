from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    episodic = "episodic"       # specific events / conversations
    semantic = "semantic"       # facts, preferences, general knowledge
    procedural = "procedural"   # how to do things / workflows


class MemoryCreate(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=8000)
    importance: float = Field(default=1.0, ge=0.0, le=2.0)
    memory_type: MemoryType = MemoryType.episodic


class MemoryResponse(BaseModel):
    id: UUID
    agent_id: str
    user_id: str
    content: str
    importance: float
    memory_type: MemoryType
    created_at: datetime
    score: float | None = None      # populated on search results


class MemorySearchRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Blend weight: 1.0 = pure semantic similarity, 0.0 = pure recency",
    )
    memory_type: MemoryType | None = None


class MemorySearchResponse(BaseModel):
    results: list[MemoryResponse]
    query: str
    total: int


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str = "0.1.0"
