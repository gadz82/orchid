# agents-api — FastAPI + LangGraph agent backend
#
# Build context: agents/
#   docker build -t agents-api .
#
# Multi-stage: install deps → slim runtime

# ── Stage 1: build & install ───────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install ".[all-storage]"

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "orchid_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
