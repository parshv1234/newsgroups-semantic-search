"""
Embed all documents and populate ChromaDB.

Pipeline order note:
  We run clustering (04) BEFORE this script when possible, so cluster
  assignments can be stored as metadata in ChromaDB for filtered retrieval.
  If the cluster model doesn't exist yet, we store dominant_cluster=-1
  as a placeholder and you can re-run after 04_cluster.py.

Expected runtime on Apple M3 Pro (MPS):
  Embedding 19,898 docs: ~3-5 minutes
  ChromaDB indexing:     ~1 minute
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from app.core.config import settings
from app.core.embedder import Embedder
from app.core.vector_store import VectorStore


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


def main():
    print("Loading corpus...")
    records = load_corpus()
    print(f"Loaded {len(records)} documents.\n")

    # ── Try to load cluster model if it exists ───────────────────────────────
    cluster_model_path = Path(settings.CLUSTER_MODEL_PATH)
    clusterer = None
    if cluster_model_path.exists():
        print("Cluster model found — assigning memberships during indexing.")
        from app.core.clustering import FuzzyClusterer
        clusterer = FuzzyClusterer.load()
    else:
        print("No cluster model yet — storing placeholder dominant_cluster=-1.")
        print("Run 04_cluster.py first, then re-run this script.\n")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"Loading embedder: {settings.EMBEDDING_MODEL}")
    embedder = Embedder()

    texts = [r["text"] for r in records]
    print(f"\nEmbedding {len(texts)} documents...")
    print("(This takes ~3-5 min on M3 Pro via MPS)\n")

    all_embeddings = embedder.embed(texts)
    print(f"Embeddings shape: {all_embeddings.shape}\n")

    # ── Cluster assignments ───────────────────────────────────────────────────
    all_memberships = None
    if clusterer:
        print("Computing NMF memberships...")
        all_memberships = clusterer.transform(texts)
        print(f"Memberships shape: {all_memberships.shape}\n")

    # ── Build metadata ────────────────────────────────────────────────────────
    ids = [str(r["id"]) for r in records]
    # Truncate stored text to 2000 chars — ChromaDB stores this for display,
    # not for search. The full embedding captures the semantics.
    documents = [r["text"][:2000] for r in records]
    metadatas = []

    for i, record in enumerate(records):
        meta: dict = {
            "newsgroup": record["newsgroup"],
            "doc_id": record["id"],
        }
        if all_memberships is not None:
            meta["dominant_cluster"] = int(all_memberships[i].argmax())
            meta["cluster_memberships"] = json.dumps(
                all_memberships[i].tolist()
            )
        else:
            meta["dominant_cluster"] = -1
            meta["cluster_memberships"] = "[]"

        metadatas.append(meta)

    # ── Index into ChromaDB ───────────────────────────────────────────────────
    print(f"Connecting to ChromaDB at {settings.CHROMA_PERSIST_DIR}")
    store = VectorStore()

    existing = store.count
    if existing > 0:
        print(f"WARNING: ChromaDB already has {existing} documents.")
        resp = input("Re-index? (upserts/overwrites) [y/N]: ").strip().lower()
        if resp != "y":
            print("Skipping. Exiting.")
            return

    print(f"\nIndexing {len(ids)} documents into ChromaDB...")
    store.add_documents(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone. ChromaDB now contains {store.count} documents.")
    print(f"Persisted to: {settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()