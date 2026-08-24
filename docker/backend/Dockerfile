ARG BACKEND=cpu

FROM python:3.12-slim AS base-cpu

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base-cuda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

FROM base-${BACKEND} AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN mkdir -p dashboard api && touch dashboard/__init__.py api/__init__.py README.md
ARG BACKEND=cpu
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BACKEND" = "cuda" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cu126 && \
        pip install --no-cache-dir ".[api,cuda]"; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir ".[api,cpu]"; \
    fi

COPY dashboard/ ./dashboard/
COPY api/ ./api/
COPY alembic/ ./alembic/
COPY alembic.ini ./
# The BSH sector geometry is versioned once, under frontend/public/geo/. The backend reads it
# from api/data/, so the image copies it rather than the repository duplicating 5 MB of JSON.
# Without it, api/services/sector_mapping.py raises FileNotFoundError at startup and the BSH
# sector layer stays empty (three tests fail on it too).
COPY frontend/public/geo/secteurs-bsh.geojson ./api/data/secteurs-bsh.geojson
COPY scripts/ ./scripts/
COPY docker/backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/data/prepared /app/checkpoints /app/results /app/logs

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["/app/entrypoint.sh"]
