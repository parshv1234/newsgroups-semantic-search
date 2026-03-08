"""
FastAPI route handlers.

All shared state (cache, embedder, clusterer, vector store) lives in
module-level singletons initialised once at startup in main.py.
Each request gets the same instances via FastAPI dependency injection.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends

from app.models.schemas import (
    QueryRequest, QueryResponse,
    CacheStatsResponse, CacheFlushResponse, HealthResponse,
)
from app.core.embedder import get_embedder, Embedder
from app.core.vector_store import get_vector_store, VectorStore
from app.core.clustering import get_clusterer, FuzzyClusterer
from app.core.semantic_cache import get_semantic_cache, SemanticCache
from app.core.config import settings

router = APIRouter()


# ── Dependency injection ──────────────────────────────────────────────────────

def _embedder()      -> Embedder:      return get_embedder()
def _store()         -> VectorStore:   return get_vector_store()
def _clusterer()     -> FuzzyClusterer: return get_clusterer()
def _cache()         -> SemanticCache: return get_semantic_cache()


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    embedder:  Embedder       = Depends(_embedder),
    store:     VectorStore    = Depends(_store),
    clusterer: FuzzyClusterer = Depends(_clusterer),
    cache:     SemanticCache  = Depends(_cache),
) -> QueryResponse:
    """
    Semantic search with cache-first lookup.

    Flow:
      1. Embed query (sentence-transformer)
      2. Compute NMF cluster memberships
      3. Check semantic cache (cluster-bucketed)
      4a. HIT  → return cached result immediately (no ChromaDB call)
      4b. MISS → query ChromaDB, store result in cache, return
    """
    raw_query = body.query.strip()

    # Step 1: Embed
    query_embedding = embedder.embed_single(raw_query)

    # Step 2: Cluster assignment
    memberships = clusterer.transform_single(raw_query)
    dominant_cluster = int(memberships.argmax())

    # Step 3: Cache lookup
    cache_result = cache.lookup(
        query_embedding=query_embedding,
        cluster_id=dominant_cluster,
        memberships=memberships,
    )

    if cache_result.hit and cache_result.entry is not None:
        entry = cache_result.entry
        return QueryResponse(
            query=raw_query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(cache_result.similarity, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
        )

    # Step 4b: Cache miss — hit ChromaDB
    chroma_results = store.query(
        query_embedding=query_embedding,
        n_results=settings.RETRIEVAL_TOP_K,
    )
    result_text = _format_results(chroma_results)

    # Store in cache for future hits
    cache.store(
        query=raw_query,
        query_embedding=query_embedding,
        result=result_text,
        cluster_id=dominant_cluster,
        memberships=memberships,
    )

    return QueryResponse(
        query=raw_query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_text,
        dominant_cluster=dominant_cluster,
    )


# ── GET /cache/stats ──────────────────────────────────────────────────────────

@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(cache: SemanticCache = Depends(_cache)) -> CacheStatsResponse:
    """Return current cache performance metrics."""
    s = cache.stats
    return CacheStatsResponse(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


# ── DELETE /cache ─────────────────────────────────────────────────────────────

@router.delete("/cache", response_model=CacheFlushResponse)
def flush_cache(cache: SemanticCache = Depends(_cache)) -> CacheFlushResponse:
    """
    Flush the entire cache and reset all stats.

    Use after:
      - Corpus updates (cached results may be stale)
      - Changing the similarity threshold
      - Testing (start from clean state)
    """
    n = cache.total_entries
    cache.flush()
    return CacheFlushResponse(
        message="Cache flushed successfully.",
        entries_cleared=n,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(
    store: VectorStore   = Depends(_store),
    cache: SemanticCache = Depends(_cache),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store_count=store.count,
        cache_entries=cache.total_entries,
        model=settings.EMBEDDING_MODEL,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_results(chroma_results: dict) -> str:
    """Format ChromaDB results into a readable string."""
    documents = chroma_results.get("documents", [[]])[0]
    metadatas = chroma_results.get("metadatas", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]

    if not documents:
        return "No results found."

    parts = []
    for i, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances)
    ):
        sim = round(1 - dist, 4)
        newsgroup = meta.get("newsgroup", "unknown")
        cluster   = meta.get("dominant_cluster", "?")
        snippet   = doc[:300].replace("\n", " ").strip()
        if len(doc) > 300:
            snippet += "..."
        parts.append(
            f"[{i+1}] similarity={sim} | newsgroup={newsgroup} | cluster={cluster}\n{snippet}"
        )

    return "\n\n".join(parts)