# ── Runtime image ─────────────────────────────────────────────────────────────
# python:3.11-slim chosen over:
#   - alpine: C extensions (numpy, scipy) compile poorly on musl libc
#   - full image: 150MB heavier with no benefit for our stack
#   - 3.12: sentence-transformers has occasional compat issues on 3.12
FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for some Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (separate layer — only rebuilds on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model at build time.
# Without this, first startup downloads ~90MB — slow and fragile in production.
# Baked into the image, startup is immediate.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data directory — will be overridden by volume mount in production
RUN mkdir -p data/chroma_db

# Copy env file
COPY .env .env

EXPOSE 8000

# Health check — polls /health every 30s after a 60s startup grace period
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Single worker: the in-memory semantic cache is process-local.
# Multiple workers = each has its own cache = divergent state.
# For horizontal scaling the cache would need to move to a shared store,
# but that's outside this project's scope.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]