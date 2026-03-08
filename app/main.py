"""
FastAPI application with lifespan startup.

All ML models are loaded ONCE at startup via the lifespan context manager.
Loading the sentence-transformer alone takes ~500ms — never per-request.

Startup sequence:
  1. Embedder      (sentence-transformer model)
  2. Cluster model (NMF + TF-IDF from disk)
  3. Vector store  (ChromaDB connection)
  4. Semantic cache (in-memory, fresh each restart)
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.embedder import get_embedder
from app.core.vector_store import get_vector_store
from app.core.clustering import get_clusterer
from app.core.semantic_cache import get_semantic_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("Starting Newsgroups Semantic Search API")
    print("=" * 50)

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading embedder: {settings.EMBEDDING_MODEL}")
    embedder = get_embedder()
    print(f"      dim={embedder.dim}")

    cluster_path = Path(settings.CLUSTER_MODEL_PATH)
    if not cluster_path.exists():
        raise RuntimeError(
            f"Cluster model not found at {cluster_path}.\n"
            "Run: python scripts/04_cluster.py"
        )
    print(f"[2/4] Loading cluster model")
    clusterer = get_clusterer()
    print(f"      clusters={clusterer.n_clusters}")

    print(f"[3/4] Connecting to ChromaDB")
    store = get_vector_store()
    print(f"      documents={store.count}")
    if store.count == 0:
        print("      WARNING: vector store is empty. Run scripts 01-03.")

    print(f"[4/4] Initialising semantic cache")
    cache = get_semantic_cache()
    print(f"      threshold={cache.threshold}, buckets={cache.n_clusters}")

    print("=" * 50)
    print("API ready → http://localhost:8000")
    print("Docs     → http://localhost:8000/docs")
    print("=" * 50)

    yield  # app runs here

    print("Shutting down.")


app = FastAPI(
    title="20 Newsgroups Semantic Search",
    description="Semantic search with NMF fuzzy clustering and cluster-bucketed semantic cache.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)