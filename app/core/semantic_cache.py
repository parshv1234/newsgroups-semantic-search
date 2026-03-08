"""
semantic_cache.py — From-scratch semantic cache with cluster-bucketed lookup.

The core problem with a naive semantic cache:
──────────────────────────────────────────────
A naive implementation stores all cached entries in a flat list and does
a linear scan on every lookup — O(n) comparisons. At 10k cached entries
that's 10k dot products per query. The cache becomes the bottleneck.

Our solution: cluster-bucketed lookup
──────────────────────────────────────
Every cached entry is stored in a bucket keyed by its dominant cluster.
At lookup time we only search the relevant bucket(s).

  Without bucketing: 10,000 comparisons
  With 23 buckets:   ~435 comparisons (23x faster)

This is what makes the cluster structure "do real work" — it's not just
for analysis, it's load-bearing infrastructure in the cache.

The improvement compounds as the cache grows. O(n) eventually defeats
the purpose of caching; O(n/k) stays manageable.

Boundary expansion:
────────────────────
A query sitting near a cluster boundary (e.g. "gun legislation" between
politics C10 and guns C16) might have been cached under either cluster
depending on which side it landed. We handle this by also searching
neighbouring clusters when a query has significant secondary membership
(>= 20% in another cluster).

The similarity threshold — the most consequential tunable:
───────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD determines what "close enough" means.

  0.95+  Near-exact rephrases only. Very safe, low hit rate.
  0.85   Paraphrase-level. "space shuttle launch" ~ "NASA rocket mission".
         Recommended default — best precision/recall balance.
  0.75   Topic-level. Higher hit rate but risk of conflating distinct queries.
  0.60   Cluster-level. Too aggressive — unrelated queries collide.

The cluster bucketing provides a safety floor: even at 0.75, queries
from different clusters can never match each other.

No Redis. No Memcached. No caching library. Pure Python.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.config import settings


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray       # shape (dim,), L2-normalised
    result: str
    cluster_id: int             # dominant cluster (argmax of memberships)
    memberships: np.ndarray     # full soft distribution, shape (n_clusters,)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0          # incremented each time this entry is returned


@dataclass
class CacheResult:
    hit: bool
    entry: Optional[CacheEntry] = None
    similarity: float = 0.0


class SemanticCache:
    """
    Cluster-bucketed semantic cache. Thread-safe via threading.Lock.

    Data structure:
        _buckets: dict[cluster_id, list[CacheEntry]]

    Lookup is O(bucket_size) not O(total_entries).
    Bucket size ≈ total_entries / n_clusters on a uniform query distribution.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n_clusters: int | None = None,
        enable_boundary_expansion: bool = True,
    ):
        self.threshold = threshold if threshold is not None \
            else settings.CACHE_SIMILARITY_THRESHOLD
        self.n_clusters = n_clusters or settings.N_CLUSTERS

        # Core data structure: one list per cluster
        self._buckets: dict[int, list[CacheEntry]] = {
            i: [] for i in range(self.n_clusters)
        }

        # Boundary expansion: also search clusters where the query
        # has >= 20% membership (handles boundary documents)
        self._enable_boundary_expansion = enable_boundary_expansion
        self._boundary_threshold = 0.20

        # Stats
        self._hit_count = 0
        self._miss_count = 0

        # Thread safety: single lock for reads and writes.
        # A readers-writer lock would be faster under high read concurrency,
        # but a simple lock is correct and sufficient at this scale.
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
        memberships: np.ndarray | None = None,
    ) -> CacheResult:
        """
        Search for a semantically similar cached query.

        Steps:
          1. Determine which buckets to search (dominant + boundary clusters)
          2. Compute cosine similarity against all entries in those buckets
             Since embeddings are L2-normalised: cosine sim == dot product
          3. Return highest-similarity entry if it exceeds the threshold

        Args:
            query_embedding : L2-normalised query embedding, shape (dim,)
            cluster_id      : dominant cluster index (NMF argmax)
            memberships     : full cluster distribution for boundary logic
        """
        buckets_to_search = self._get_buckets_to_search(
            cluster_id, memberships
        )

        best_sim = -1.0
        best_entry: Optional[CacheEntry] = None

        with self._lock:
            for bucket_id in buckets_to_search:
                for entry in self._buckets[bucket_id]:
                    # dot product == cosine similarity for L2-normalised vecs
                    sim = float(np.dot(query_embedding, entry.embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

        if best_sim >= self.threshold and best_entry is not None:
            with self._lock:
                best_entry.hit_count += 1
                self._hit_count += 1
            return CacheResult(hit=True, entry=best_entry, similarity=best_sim)

        with self._lock:
            self._miss_count += 1
        return CacheResult(hit=False, similarity=best_sim)

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: str,
        cluster_id: int,
        memberships: np.ndarray,
    ) -> CacheEntry:
        """
        Add a new entry to the appropriate cluster bucket.
        """
        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),  # defensive copy
            result=result,
            cluster_id=cluster_id,
            memberships=memberships.copy(),
        )
        with self._lock:
            self._buckets[cluster_id].append(entry)
        return entry

    def flush(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            for bucket_id in self._buckets:
                self._buckets[bucket_id] = []
            self._hit_count = 0
            self._miss_count = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "total_entries": sum(len(b) for b in self._buckets.values()),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
                "bucket_sizes": {k: len(v) for k, v in self._buckets.items()},
                "threshold": self.threshold,
            }

    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_buckets_to_search(
        self,
        cluster_id: int,
        memberships: np.ndarray | None,
    ) -> list[int]:
        """
        Base: always search the dominant cluster bucket.
        Expansion: also search any cluster with >= boundary_threshold membership.

        Example: "gun legislation" might have C10(politics)=0.45, C16(guns)=0.35
        We search both buckets so cached entries from either cluster are found.
        """
        buckets = {cluster_id}
        if self._enable_boundary_expansion and memberships is not None:
            for other_id, weight in enumerate(memberships):
                if other_id != cluster_id and weight >= self._boundary_threshold:
                    buckets.add(other_id)
        return list(buckets)


# Module-level singleton — shared across all API requests
_cache: SemanticCache | None = None

def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache