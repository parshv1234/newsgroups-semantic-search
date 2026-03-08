"""
Pydantic request/response models.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="Natural language search query"
    )


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None       # query that was matched on hit
    similarity_score: Optional[float] = None  # cosine sim to matched query
    result: str                               # retrieved documents
    dominant_cluster: int                     # NMF cluster with highest weight


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheFlushResponse(BaseModel):
    message: str
    entries_cleared: int


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    cache_entries: int
    model: str