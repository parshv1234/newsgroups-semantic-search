"""
NMF-based fuzzy (soft) clustering.

Why NMF over alternatives:
───────────────────────────
The assignment requires a DISTRIBUTION over clusters per document — not
a hard label. Here's why NMF is the right tool:

  NMF (Non-negative Matrix Factorization):
    - Factorizes TF-IDF matrix V ≈ W × H
    - W (docs × topics): non-negative weights per document.
      After L1-normalisation → valid probability distribution.
    - H (topics × terms): interpretable — shows which words define each cluster.
    - Parts-based: each doc is a MIXTURE of topics, not assigned to one.
    - Natively probabilistic — no post-hoc hack needed.

  K-Means:
    - Hard assignments only. Soft K-Means assumes spherical geometry
      in high-dimensional space — embeddings don't satisfy this.

  LDA:
    - Also produces per-doc distributions but requires integer token
      counts (incompatible with TF-IDF sublinear weighting), slower
      (variational inference), poor on short informal texts.

  HDBSCAN:
    - Density-based; unreliable in high-dimensional TF-IDF space.

Why TF-IDF input (not embeddings) for NMF:
  - NMF on TF-IDF produces interpretable topic-term distributions (H matrix)
    which we use to name and validate clusters.
  - The clusters are used for CACHE BUCKETING, so topic-coherence
    matters more than embedding-geometry accuracy for this component.

Why NOT N_CLUSTERS=20:
  - The 20 labels are not semantically distinct. Several newsgroups
    share nearly identical vocabulary (comp.sys.mac.hardware ≈
    comp.sys.ibm.pc.hardware). NMF finds the real structure.
  - 23 was chosen from reconstruction error elbow analysis.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import settings


def _safe_l1_normalize(W: np.ndarray) -> np.ndarray:
    """
    L1-normalize each row, handling zero-rows gracefully.

    Zero rows occur when a document's NMF projection is all-zero —
    e.g. very short posts whose tokens don't appear in any component.
    sklearn's normalize() returns all-zeros for zero-norm rows, breaking
    the "memberships sum to 1" invariant.

    Fix: assign uniform distribution to zero rows (maximum entropy /
    most uncertain assignment) — semantically correct fallback.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    zero_mask = (row_sums == 0).flatten()

    W_out = np.where(row_sums > 0, W / row_sums, W)

    if zero_mask.any():
        n_zero = zero_mask.sum()
        print(f"  [normalize] {n_zero} zero-rows → assigned uniform distribution")
        W_out[zero_mask] = 1.0 / W.shape[1]

    return W_out.astype(np.float32)


class FuzzyClusterer:
    """
    NMF-based soft clusterer.

    Each document gets a membership distribution over clusters —
    a float array of shape (n_clusters,) that sums to 1.0.

    Training:
        clusterer = FuzzyClusterer()
        clusterer.fit(texts)
        clusterer.save()

    Inference:
        clusterer = FuzzyClusterer.load()
        memberships = clusterer.transform_single("my query")
        dominant   = memberships.argmax()
    """

    def __init__(self, n_clusters: int | None = None):
        self.n_clusters = n_clusters or settings.N_CLUSTERS
        self._tfidf: TfidfVectorizer | None = None
        self._nmf: NMF | None = None
        self._feature_names: list[str] = []

    def fit(self, texts: list[str]) -> "FuzzyClusterer":
        """
        Fit TF-IDF vectorizer then NMF model on the corpus.

        TF-IDF config choices:
          sublinear_tf=True  : log(1+tf) reduces dominance of high-freq terms
          ngram_range=(1,2)  : bigrams capture "space shuttle", "hard drive"
          token_pattern      : keeps "C++", "TCP/IP", "RS-6000" as single tokens
          min_df=5           : ignore tokens in fewer than 5 docs (noise)
          max_df=0.85        : ignore tokens in >85% of docs (stopwords)

        NMF config choices:
          init="nndsvda"     : deterministic, better than random for sparse input
          No alpha/l1_ratio  : regularization drove rows to zero in practice
                               (non-negativity constraint alone gives sparsity)
        """
        print(f"[Clusterer] Building TF-IDF over {len(texts)} documents...")
        self._tfidf = TfidfVectorizer(
            max_features=settings.TFIDF_MAX_FEATURES,
            min_df=settings.TFIDF_MIN_DF,
            max_df=settings.TFIDF_MAX_DF,
            sublinear_tf=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\-\.]{1,}\b",
        )
        tfidf_matrix = self._tfidf.fit_transform(texts)
        self._feature_names = self._tfidf.get_feature_names_out().tolist()
        print(f"[Clusterer] TF-IDF matrix: {tfidf_matrix.shape}")

        print(f"[Clusterer] Fitting NMF with {self.n_clusters} components...")
        self._nmf = NMF(
            n_components=self.n_clusters,
            init="nndsvda",
            max_iter=settings.NMF_MAX_ITER,
            random_state=settings.NMF_RANDOM_STATE,
        )
        W = self._nmf.fit_transform(tfidf_matrix)
        print(f"[Clusterer] Reconstruction error: {self._nmf.reconstruction_err_:.4f}")

        self._W_train = _safe_l1_normalize(W)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Project texts into cluster space.
        Returns float32 array shape (n_texts, n_clusters), rows sum to 1.
        """
        if self._tfidf is None or self._nmf is None:
            raise RuntimeError("Clusterer not fitted. Call fit() or load() first.")
        tfidf_matrix = self._tfidf.transform(texts)
        W = self._nmf.transform(tfidf_matrix)
        return _safe_l1_normalize(W)

    def transform_single(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    def dominant_cluster(self, text: str) -> int:
        return int(self.transform_single(text).argmax())

    def get_top_terms(self, cluster_idx: int, n_terms: int = 15) -> list[str]:
        """
        Top terms for a cluster from the H matrix.
        High H values = 'documents in this cluster use these words a lot.'
        These are the human-readable cluster labels.
        """
        if self._nmf is None:
            raise RuntimeError("Not fitted.")
        top_idx = self._nmf.components_[cluster_idx].argsort()[::-1][:n_terms]
        return [self._feature_names[i] for i in top_idx]

    def get_cluster_summary(self) -> list[dict]:
        return [
            {"cluster_id": i, "top_terms": self.get_top_terms(i)}
            for i in range(self.n_clusters)
        ]

    def save(self, path: Path | None = None) -> None:
        path = Path(path or settings.CLUSTER_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "tfidf": self._tfidf,
            "nmf": self._nmf,
            "n_clusters": self.n_clusters,
            "feature_names": self._feature_names,
        }, path)
        print(f"[Clusterer] Saved to {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "FuzzyClusterer":
        path = Path(path or settings.CLUSTER_MODEL_PATH)
        data = joblib.load(path)
        instance = cls(n_clusters=data["n_clusters"])
        instance._tfidf = data["tfidf"]
        instance._nmf = data["nmf"]
        instance._feature_names = data["feature_names"]
        print(f"[Clusterer] Loaded from {path}")
        return instance


_clusterer: FuzzyClusterer | None = None

def get_clusterer() -> FuzzyClusterer:
    global _clusterer
    if _clusterer is None:
        _clusterer = FuzzyClusterer.load()
    return _clusterer