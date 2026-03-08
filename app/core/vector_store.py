"""
ChromaDB interface.

Why ChromaDB over alternatives:
─────────────────────────────────
  FAISS:
    - Best ANN performance but no built-in persistence, no metadata
      filtering, no document storage. We'd build all that ourselves.

  Pinecone / Qdrant / Weaviate:
    - Managed or heavy services. This system must run locally with
      a single uvicorn command.

  ChromaDB:
    - Persistent SQLite backend (no separate process needed)
    - Native cosine similarity
    - Metadata filtering — essential for cluster-based cache lookup
    - Python-native API
    - Perfect fit for 20k docs on a local machine

Metadata schema per document:
  newsgroup        : str   original category label
  dominant_cluster : int   argmax of NMF membership vector
  cluster_memberships: str JSON-encoded float list (all clusters)
  doc_id           : int   integer index
"""

from __future__ import annotations
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings


class VectorStore:

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # cosine is correct here: our embeddings are L2-normalised upstream
        # so cosine distance and dot-product distance are equivalent.
        # Being explicit guards against future changes to the embedder.
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 512,
    ) -> None:
        """Upsert in batches to avoid ChromaDB memory limits."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  Indexed {end}/{total}", end="\r")
        print()

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """
        Nearest-neighbour search.
        where: optional metadata filter e.g. {"dominant_cluster": 3}
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def reset(self) -> None:
        """Drop and recreate collection. Used in testing."""
        self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


_store: VectorStore | None = None

def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store