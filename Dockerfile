# Slim Python 3.13 base
FROM python:3.13-slim

WORKDIR /app

# System dependencies needed by PyTorch / sentence-transformers / curl healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (Docker layer caching - this layer doesn't rebuild
# when you only change app code)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the Sentence Transformer model so first request is fast
# (otherwise the first /ask call downloads ~80MB inside the container)
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True})"

# Copy application code
COPY agent.py api.py Ingest.py fetch_corpus.py eval.py eval_set.json ./

# Expose the API port
EXPOSE 8000

# Container healthcheck - hits /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]