"""
Sentence-transformer wrapper with batched inference.

Model choice: all-MiniLM-L6-v2
────────────────────────────────
Why this over alternatives:

  all-MiniLM-L6-v2 (22M params, 384-dim):
    - ~2ms/doc on Apple M-series via MPS (Metal Performance Shaders)
    - Trained on 1B+ pairs: Reddit, NLI, QA, STS datasets
    - Informal/conversational training data matches newsgroup post style

  all-mpnet-base-v2 (109M params, 768-dim):
    - Better SBERT benchmarks but 4-5x slower, 2x larger embeddings
    - ChromaDB index grows proportionally with dim — not worth it here

  text-embedding-ada-002:
    - Requires OpenAI API, adds latency, costs money per embedding
    - Overkill for local inference at 20k doc scale

normalize_embeddings=True is critical: L2-normalised vectors mean
cosine similarity == dot product. This keeps the cache's similarity
comparisons consistent with ChromaDB's cosine distance metric.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class Embedder:
    """
    Thin wrapper around SentenceTransformer.

    Kept as a class (not a module function) so the model loads once
    and is reused across the app lifetime. Model loading takes ~500ms —
    we never want that happening per request.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        # device=None lets sentence-transformers auto-detect MPS/CUDA/CPU
        self._model = SentenceTransformer(model_name, device=None)
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[Embedder] Loaded '{model_name}' — dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str] | str) -> np.ndarray:
        """
        Embed one or more texts.
        Returns float32 array of shape (N, dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        return self._model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,  # L2-normalise → cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Single query embedding — returns 1-D array of shape (dim,)."""
        return self.embed([text])[0]


# Module-level singleton — loaded once on first import
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder