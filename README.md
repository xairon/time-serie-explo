# Junon — Piezometric Forecasting Platform

Full-stack platform for groundwater level forecasting and analysis. React frontend + FastAPI backend + Darts (deep learning) + Pastas (transfer function-noise models).

## Features

### AI Lab (Darts)
- **12+ deep learning models**: TFT, Transformer, N-BEATS, N-HiTS, LSTM, GRU, TCN, TiDE, TSMixer, DLinear, NLinear
- **Hyperparameter optimization** with Optuna
- **Real-time training** via SSE streaming
- **Explainability**: SHAP, TimeSHAP, Captum gradients, attention weights
- **Counterfactual analysis**: PhysCF, CoMTE with dual validation
- **Latent space exploration**: SoftCLT/TS2Vec embeddings, UMAP, clustering

### Pastas Lab (TFN)
- **Transfer Function-Noise models** for piezometric time series
- **Auto-fit**: grid search across configurations with STOWA quality criteria
- **Calibration/validation** with train/test split
- **Results dashboard**: performance metrics, response functions, diagnostics, hydrological signatures
- **What-if scenarios**: synthetic pumping, climate change, trend modifications
- **Realistic referential**: pumping profiles per usage (AEP, irrigation, industrial) x aquifer family
- **Adaptive bounds**: drawdown estimation from calibrated step response

### Observatory
- **Station map** with BDLISA aquifer overlays
- **Drought indices** (SPI, SPLI)
- **Multi-station comparison** and regional analysis

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, React Router 7, TanStack Query 5, Plotly.js, Tailwind CSS 4, Vite |
| Backend | FastAPI, SSE (Server-Sent Events) |
| ML | Darts (PyTorch Lightning), Pastas, Optuna, SHAP/Captum |
| Tracking | MLflow |
| Database | PostgreSQL (BRGM data warehouse), SQLAlchemy |
| Deployment | Docker Compose (rootless), NVIDIA CUDA |

## Quick Start

### Docker (recommended)

```bash
git clone <repo-url>
cd junon-time-series

# Configure .env (see .env.example)
# COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml for GPU

docker compose up -d --build
```

Access: `http://localhost:49513`

MLflow UI: `http://localhost:49512`

### Port Mapping

| Service | Port |
|---------|------|
| App (Nginx) | 49513 |
| MLflow | 49512 |
| PostgreSQL | internal only |

## Architecture

```
frontend/                     # React SPA
  src/
    pages/                    # Route pages
    components/               # UI components (charts, cards, forms, pastas/*)
    hooks/                    # TanStack Query hooks
    lib/                      # API client, types

api/                          # FastAPI REST/SSE
  routers/                    # Endpoint modules
  schemas/                    # Pydantic models

dashboard/
  utils/                      # Pure Python (NO framework dependency)
    pastas/                   # Pastas pipeline (builder, fit, scenarios, diagnostics)
    explainability/           # SHAP, attention, gradients
    counterfactual/           # PhysCF, CoMTE
    training.py               # Darts training pipeline
    model_factory.py          # Model creation
    preprocessing.py          # Data prep
```

## Development

```bash
# Rebuild after code changes
docker compose up -d --build

# Rebuild specific service
docker compose up -d --build frontend
docker compose up -d --build backend

# Run tests
docker compose exec backend pytest tests/
cd frontend && npx tsc --noEmit  # TypeScript check (no Node.js on host)

# View logs
docker compose logs -f backend
```

**Important**: No Node.js on the host. Frontend builds and runs inside Docker only.

## Requirements

- Docker with Compose v2
- NVIDIA GPU + CUDA drivers (for GPU acceleration)
- ~8 GB disk for images

## License

MIT License

## Author

Nicolas Ringuet
