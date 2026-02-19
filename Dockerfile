FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 scikit-learn>=1.3.0

RUN pip install --no-cache-dir \
    matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.17.0 streamlit>=1.28.0

RUN pip install --no-cache-dir \
    darts>=0.32.0 statsmodels>=0.14.0 pytorch-lightning>=2.0.0

RUN pip install --no-cache-dir \
    shap>=0.42.0 optuna>=3.0.0 tqdm>=4.65.0 einops>=0.7.0

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy application code
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p /app/checkpoints /app/logs /app/results

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "streamlit", "run", "dashboard/training/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
