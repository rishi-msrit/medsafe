# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="MedSafe Research"
LABEL description="GNN-powered polypharmacy drug interaction safety API"

# System deps for RDKit, psycopg2, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html && \
    pip install --no-cache-dir -r requirements.txt


# Copy project source
COPY configs/     configs/
COPY models/      models/
COPY pipeline/    pipeline/
COPY explainability/ explainability/
COPY scoring/     scoring/
COPY serving/     serving/
COPY training/    training/
COPY evaluation/  evaluation/

# Copy checkpoints and data
COPY checkpoints/ checkpoints/
COPY data/ data/
RUN mkdir -p data/raw data/embeddings mlruns


# Non-root user for security
RUN useradd -m -u 1001 medsafe && chown -R medsafe:medsafe /app
USER medsafe

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
