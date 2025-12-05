# Multi-stage build for optimized image size
FROM python:3.11-slim AS builder

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies with uv (creates virtual environment in /app/.venv)
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim AS production

# Install system dependencies for PyTorch and visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

# Create directories for data persistence
RUN mkdir -p /app/checkpoints /app/logs /app/results

# Expose Streamlit default port (will be mapped in docker-compose)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command - run the training app
CMD ["streamlit", "run", "dashboard/training/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
