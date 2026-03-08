from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.example",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    DATA_DIR: Path = Path("./data")
    CHROMA_PERSIST_DIR: Path = Path("./data/chroma_db")
    PROCESSED_DATA_PATH: Path = Path("./data/processed_corpus.jsonl")
    CLUSTER_MODEL_PATH: Path = Path("./data/cluster_model.joblib")

    # Embedding
    # all-MiniLM-L6-v2: 22M params, 384-dim, fast on Apple M-series via MPS.
    # Chosen over larger models (mpnet, ada-002) because retrieval quality
    # per millisecond matters more than raw benchmark scores at this scale.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    CHROMA_COLLECTION_NAME: str = "newsgroups"

    # Clustering
    # 23 chosen from reconstruction error elbow analysis — NOT hardcoded to
    # 20 (the label count). NMF finds finer semantic structure than labels.
    N_CLUSTERS: int = 23
    NMF_MAX_ITER: int = 500
    NMF_RANDOM_STATE: int = 42

    # TF-IDF params
    TFIDF_MAX_FEATURES: int = 20000  # top 20k tokens covers ~95% of term freq
    TFIDF_MIN_DF: int = 5            # must appear in ≥5 docs
    TFIDF_MAX_DF: float = 0.85       # skip tokens in >85% of docs (stopwords)

    # Semantic Cache
    CACHE_SIMILARITY_THRESHOLD: float = 0.85

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RETRIEVAL_TOP_K: int = 5


settings = Settings()