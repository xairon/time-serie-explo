ARG BACKEND=cpu

# ---------- base image per backend ----------
FROM python:3.12-slim AS base-cpu
# nothing extra

FROM intel/intel-extension-for-pytorch:2.5.10-xpu AS base-xpu
# Intel XPU runtime already ships torch + IPEX

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

# ---------- select the right base ----------
FROM base-${BACKEND} AS runtime

WORKDIR /app

# System deps (slim images need curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python deps via pyproject.toml ----------
# Copy only what pip needs to resolve deps (cache-friendly layer)
COPY pyproject.toml ./
# hatchling needs the package dir + README to exist
RUN mkdir -p dashboard && touch dashboard/__init__.py && touch README.md

ARG BACKEND=cpu
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BACKEND" = "cuda" ]; then \
        pip install --no-cache-dir torch torchaudio torchvision \
            --index-url https://download.pytorch.org/whl/cu124 && \
        pip install --no-cache-dir ".[cuda]"; \
    elif [ "$BACKEND" = "xpu" ]; then \
        pip install --no-cache-dir ".[xpu]"; \
    else \
        pip install --no-cache-dir ".[cpu]"; \
    fi

# ---------- App code ----------
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/
COPY run_app.py ./

# Ensure writable dirs exist (volumes will overlay them)
RUN mkdir -p /app/data/prepared /app/checkpoints/darts /app/mlruns \
             /app/results /app/figs /app/logs/darts

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "dashboard/training/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
