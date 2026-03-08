"""
Fit NMF clustering and validate cluster quality.

Three things this script does:
  1. Finds optimal k via reconstruction error elbow analysis
  2. Fits the final model and saves it
  3. Validates quality: top terms, representative docs, boundary cases

Run BEFORE re-running 03_embed_and_index.py so cluster assignments
get stored in ChromaDB metadata.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.core.config import settings
from app.core.clustering import FuzzyClusterer


def load_corpus() -> list[dict]:
    path = settings.PROCESSED_DATA_PATH
    if not path.exists():
        print(f"ERROR: {path} not found. Run 02_preprocess.py first.")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def find_optimal_k(texts: list[str]) -> None:
    """
    Compute NMF reconstruction error for k in range(5, 30, 2).

    The elbow point — where the curve bends and further increases in k
    yield diminishing error reduction — is the evidence-based justification
    for N_CLUSTERS. We compute the second derivative to detect it.
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("=" * 55)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 55)

    print("Fitting TF-IDF (once, reused for all k)...")
    tfidf = TfidfVectorizer(
        max_features=settings.TFIDF_MAX_FEATURES,
        min_df=settings.TFIDF_MIN_DF,
        max_df=settings.TFIDF_MAX_DF,
        sublinear_tf=True,
        ngram_range=(1, 2),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\+\#\-\.]{1,}\b",
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}\n")

    k_range = list(range(5, 30, 2))
    errors = {}

    for k in k_range:
        nmf = NMF(n_components=k, init="nndsvda", max_iter=300, random_state=42)
        nmf.fit_transform(tfidf_matrix)
        errors[k] = nmf.reconstruction_err_
        print(f"  k={k:>3}: error = {errors[k]:.4f}")

    # Elbow via second derivative (max curvature)
    ks = sorted(errors)
    errs = [errors[k] for k in ks]
    if len(ks) >= 3:
        second_deriv = [
            errs[i-1] - 2*errs[i] + errs[i+1]
            for i in range(1, len(ks)-1)
        ]
        elbow_k = ks[1 + second_deriv.index(max(second_deriv))]
        print(f"\nElbow detected at k={elbow_k}")
    print(f"Configured N_CLUSTERS={settings.N_CLUSTERS}")


def analyse_clusters(
    clusterer: FuzzyClusterer,
    records: list[dict],
    memberships: np.ndarray,
) -> None:
    """
    Print cluster analysis to convince a sceptical reader.

    Shows:
      1. Top terms per cluster
      2. Newsgroup distribution per cluster (purity check)
      3. Most uncertain documents (highest entropy)
      4. Boundary documents (significant multi-cluster membership)
    """
    n = clusterer.n_clusters
    dominant = memberships.argmax(axis=1)

    # ── 1. Top terms ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TOP TERMS PER CLUSTER")
    print("=" * 55)
    for i in range(n):
        terms = ", ".join(clusterer.get_top_terms(i, 8))
        print(f"  C{i:>2}: {terms}")

    # ── 2. Newsgroup distribution ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("NEWSGROUP DISTRIBUTION PER CLUSTER")
    print("=" * 55)
    for i in range(n):
        mask = dominant == i
        if mask.sum() == 0:
            continue
        ngs = Counter(records[j]["newsgroup"] for j in range(len(records)) if mask[j])
        top = ngs.most_common(3)
        top_str = " | ".join(f"{ng.split('.')[-1]}:{c}" for ng, c in top)
        terms = ", ".join(clusterer.get_top_terms(i, 4))
        print(f"  C{i:>2} ({mask.sum():>4} docs) [{terms}]")
        print(f"       {top_str}")

    # ── 3. Most uncertain documents ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("MOST UNCERTAIN DOCUMENTS (highest membership entropy)")
    print("Boundary cases are the most semantically interesting")
    print("=" * 55)
    entropy = -np.sum(memberships * np.log(memberships + 1e-10), axis=1)
    top_uncertain = entropy.argsort()[::-1][:8]

    for idx in top_uncertain:
        rec = records[idx]
        top_clusters = memberships[idx].argsort()[::-1][:3]
        cluster_str = " + ".join(
            f"C{c}({memberships[idx, c]:.2f})" for c in top_clusters
        )
        snippet = rec["text"][:120].replace("\n", " ").strip()
        print(f"\n  [{rec['newsgroup']}]")
        print(f"  Clusters: {cluster_str}")
        print(f"  \"{snippet}...\"")

    # ── 4. Boundary documents ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BOUNDARY DOCUMENTS (>25% membership in 2+ clusters)")
    print("These show the fuzzy structure the labels miss")
    print("=" * 55)
    boundary = [
        (i, np.where(memberships[i] > 0.25)[0])
        for i in range(len(records))
        if (memberships[i] > 0.25).sum() >= 2
    ]
    print(f"\nTotal boundary docs: {len(boundary)} ({len(boundary)/len(records)*100:.1f}%)\n")

    for idx, sig_clusters in sorted(boundary, key=lambda x: -len(x[1]))[:8]:
        rec = records[idx]
        cluster_str = " + ".join(
            f"C{c}[{','.join(clusterer.get_top_terms(c, 3))}]({memberships[idx,c]:.2f})"
            for c in sig_clusters
        )
        snippet = rec["text"][:120].replace("\n", " ").strip()
        print(f"  [{rec['newsgroup']}]")
        print(f"  {cluster_str}")
        print(f"  \"{snippet}...\"\n")


def main():
    print("Loading corpus...")
    records = load_corpus()
    texts = [r["text"] for r in records]
    print(f"Loaded {len(records)} documents.\n")

    # ── Step 1: Elbow analysis ────────────────────────────────────────────────
    find_optimal_k(texts)

    # ── Step 2: Fit final model ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"FITTING FINAL MODEL (N_CLUSTERS={settings.N_CLUSTERS})")
    print("=" * 55)
    clusterer = FuzzyClusterer(n_clusters=settings.N_CLUSTERS)
    clusterer.fit(texts)
    clusterer.save()

    # ── Step 3: Compute memberships ───────────────────────────────────────────
    print("\nComputing memberships for all documents...")
    memberships = clusterer.transform(texts)
    print(f"Shape: {memberships.shape}")

    row_sums = memberships.sum(axis=1)
    print(f"Row sums — min: {row_sums.min():.4f}, max: {row_sums.max():.4f}, mean: {row_sums.mean():.4f}")

    # ── Step 4: Save memberships for 03_embed_and_index.py ───────────────────
    out_path = settings.DATA_DIR / "memberships.npy"
    np.save(out_path, memberships)
    print(f"Memberships saved to {out_path}")

    # ── Step 5: Validate ──────────────────────────────────────────────────────
    analyse_clusters(clusterer, records, memberships)

    print("\n" + "=" * 55)
    print("NEXT STEP: re-run 03_embed_and_index.py to store cluster")
    print("assignments in ChromaDB metadata.")
    print("=" * 55)


if __name__ == "__main__":
    main()