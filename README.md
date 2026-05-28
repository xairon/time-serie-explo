# Junon — Piezometric Forecasting Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)

Full-stack platform for groundwater level forecasting and scenario analysis, integrating physically-based transfer function-noise models ([Pastas](https://pastas.readthedocs.io/)) and deep learning forecasters ([Darts](https://unit8co.github.io/darts/)) under a unified MLflow registry. Built for the [BRGM](https://www.brgm.fr/) (French Geological Survey) data warehouse, exposed through a React frontend and a FastAPI backend.

## Features

### Pastas Lab — Transfer Function-Noise models
- TFN calibration on piezometric series with precipitation and evapotranspiration as forcings
- **Auto-fit** with STOWA quality criteria and configuration grid search
- **Calibration/validation split** with full diagnostics (NSE, KGE, RMSE, residual autocorrelation, normality)
- **Prospective scenarios** — synthetic pumping (AEP, irrigation, industrial), climate trends, stress scaling
- **Adaptive bounds** — drawdown estimation derived from the calibrated step response, with physically-bounded recommendations
- **BDLISA-aware presets** — aquifer-family-specific response function defaults (alluvial, sedimentary, karst, fractured, volcanic)

### AI Lab — Deep learning forecasters
- 12+ Darts models: TFT, Transformer, N-BEATS, N-HiTS, LSTM, GRU, TCN, TiDE, TSMixer, DLinear, NLinear
- Hyperparameter optimization with Optuna
- Real-time training monitoring via Server-Sent Events
- Explainability: SHAP, TimeSHAP, Captum gradients, attention weights
- Counterfactual analysis: PhysCF, CoMTE with dual validation

### Observatory
- Spatial exploration of piezometric and hydrometric stations
- BDLISA aquifer overlays, drought indices (SPI, SPLI, SSFI)
- Cross-station comparison with persistent selection

## Quick Start

### Requirements

- Docker Engine + Docker Compose v2
- NVIDIA GPU + CUDA 12.x drivers (optional, for accelerated DL training)
- ~8 GB disk space for images
- Access to the BRGM gold data warehouse (PostgreSQL, networked or local replica)

### Install and run

```bash
git clone https://scm.univ-tours.fr/ringuet/time-serie-explo.git
cd time-serie-explo
cp .env.example .env
# Edit .env: set BRGM_DB credentials, ALLOWED_ORIGINS, COMPOSE_FILE

docker compose up -d --build
```

For GPU acceleration, set `COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml` in `.env`.

Open `http://localhost:49513` for the application and `http://localhost:49512` for the MLflow UI.

### Verify installation

```bash
# API health
curl http://localhost:49513/api/v1/health

# Run the test suite
docker compose exec backend pytest tests/
```

## Architecture

```
frontend/                     React SPA (Vite, Tailwind, TanStack Query, Plotly)
  src/pages/                  Route pages
  src/components/             UI components
  src/hooks/                  Data-fetching hooks
  src/lib/                    API client, shared types

api/                          FastAPI REST + SSE
  routers/                    Endpoint modules
  schemas/                    Pydantic models

dashboard/utils/              Framework-free Python core
  pastas/                     TFN builder, fit service, scenarios, diagnostics
  explainability/             SHAP, attention, gradients, feature importance
  counterfactual/             PhysCF, CoMTE, dual validation, IPS
  training.py                 Darts training pipeline
  model_factory.py            Darts model instantiation
  preprocessing.py            Data preparation

tests/                        Pytest suite
docker/                       Per-service Dockerfiles
```

## Tech Stack

| Layer | Stack |
|---|---|
| Frontend | React 19, React Router 7, TanStack Query 5, Plotly.js, Tailwind CSS 4, Vite |
| Backend | FastAPI, Server-Sent Events, SQLAlchemy |
| Forecasting | Darts (PyTorch Lightning), Pastas |
| Tuning & XAI | Optuna, SHAP, TimeSHAP, Captum |
| Tracking | MLflow |
| Database | PostgreSQL (BRGM gold layer) |
| Deployment | Docker Compose (rootless), NVIDIA CUDA |

## Ports

| Service | Port | Notes |
|---|---|---|
| Application (Nginx) | 49513 | UI + `/api/v1/*` |
| MLflow | 49512 | Experiment tracking |
| PostgreSQL | — | Internal only |

## Development

```bash
# Rebuild after code changes
docker compose up -d --build

# Rebuild only one service
docker compose up -d --build frontend
docker compose up -d --build backend

# View backend logs
docker compose logs -f backend

# TypeScript type-check (runs inside the container, no Node on host)
docker compose run --rm frontend npx tsc --noEmit
```

The Python core under `dashboard/utils/` has no framework dependency: it is callable from notebooks, scripts, or the API layer without modification.

## Citation

If you use Junon in published work, please cite:

```bibtex
@software{ringuet_junon_2026,
  author       = {Ringuet, Nicolas},
  title        = {Junon: A piezometric forecasting platform combining Pastas TFN and deep learning},
  year         = {2026},
  url          = {https://scm.univ-tours.fr/ringuet/time-serie-explo}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built around [Pastas](https://github.com/pastas/pastas) (Collenteur et al.), [Darts](https://github.com/unit8co/darts) (Unit8), and the BRGM Hub'Eau / BDLISA / ADES open data services.
