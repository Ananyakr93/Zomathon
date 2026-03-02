# ── Multi-stage build for CSAO Recommendation Service ──────────────────
# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for LightGBM/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime — minimal image ──────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# libgomp required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY data/ data/

# Non-root user for security
RUN useradd -m -s /bin/bash csao && chown -R csao:csao /app
USER csao

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start with uvicorn — production settings
CMD ["python", "-m", "uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30", \
     "--access-log"]
