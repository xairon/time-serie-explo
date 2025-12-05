# Simple single-stage build
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.10.0 \
    tqdm>=4.65.0 \
    einops>=0.7.0 \
    streamlit>=1.28.0 \
    plotly>=5.17.0 \
    darts>=0.32.0 \
    pytorch-lightning>=2.0.0 \
    statsmodels>=0.14.0 \
    shap>=0.42.0 \
    optuna>=3.0.0 \
    torch>=2.0.0

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

# Create directories for data persistence
RUN mkdir -p /app/checkpoints /app/logs /app/results

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Use python -m to ensure streamlit is found
CMD ["python", "-m", "streamlit", "run", "dashboard/training/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
