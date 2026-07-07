# Junon — Piezometric Forecasting Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)

Full-stack platform for groundwater level forecasting and scenario analysis, integrating physically-based transfer function-noise models ([Pastas](https://pastas.readthedocs.io/)) and deep learning forecasters ([Darts](https://unit8co.github.io/darts/)) under a unified MLflow registry. Built for the [BRGM](https://www.brgm.fr/) (French Geological Survey) data warehouse, exposed through a React frontend and a FastAPI backend.

## 🚀 Deployment (split front / back)

Production is **split**: the **frontend** runs on the IT department's Kubernetes
cluster, while the **backend + GPU + databases** stay on `dib-2019006065`.

> 👉 **Frontend deployment guide: [`deploy/frontend/README.md`](deploy/frontend/README.md)**
> — everything is ready to wire up: GitLab CI, Kubernetes manifests, and configuration.

In short:
- **CI provided** (`.gitlab-ci.yml`): pushing to `main` builds and pushes the frontend image to the registry.
- **K8s manifests** ready (`deploy/frontend/k8s/`: Deployment, Service, Ingress, hostname `junon.univ-tours.fr` pre-filled).
- The frontend proxies `/api/` to the dib backend (`10.195.25.16:49514`) — **no persistent storage** required.
- **Critical network requirement**: open the route *cluster pods → `10.195.25.16:49514`*.
- Backend on dib: [`deploy/dib-backend/`](deploy/dib-backend/) · dev/prod environments: [`docs/dev-environment.md`](docs/dev-environment.md).

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

### Climat
- Dedicated page (`/climat`) with 3 views: **Situation** (France-wide grid map of SPI/STI/precip/ETP/water balance per 0.1° cell), **Point/Zone** (per-cell 1950→present history, precip vs. normal, SPI/STI multi-window, drought episode table, CSV export), **Comparaison** (multi-year rainfall overlays, SPI small multiples)
- Backend endpoints under `/api/v1/observatory/climat/*` (`range`, `grid-monthly`, `grid-indices`, `situation-summary`, `point-series`, `point-episodes`, `compare-years`, `export-point.csv`) — plain `SELECT`s (no on-the-fly statistics) over the BRGM data warehouse's precomputed ERA5 grid marts (`gold.fct_era5_monthly_grid`, `gold.fct_era5_climatology_grid`, `gold.fct_era5_indices_grid`), Redis-cached 24h
- Integrated into the Observatory map (cell popup deep-link) and the Station page (local SPI/rolling cumuls)
- The default month and the `MonthStepper` bounds come from `GET /observatory/climat/range` (`max_indices_month` / `max_monthly_complete_month` / `max_monthly_month` / `min_month`), not from `/observatory/era5/range` (the daily grid) — the daily grid's max is the current partial month, which has no SPI/STI yet. `situation-summary` returns `available: false` (instead of zeroed percentages) when no cell has an SPI for the requested month/window
- ⚠️ Temperature is currently a 00:00 UTC instantaneous read (cold bias ~2-4 °C) pending the warehouse's "daily statistics" cutover; SPI/STI drought/heat indices are already exact and unaffected

#### Notes de déploiement

Ce chantier reshape le payload de `situation-summary` (nouveau champ `available`) et
ajoute l'endpoint `range`. Après déploiement, vider dans Redis les clés :
- `junon:obs_climat_*` (situation-summary, range, grid-monthly, grid-indices, point-series,
  point-episodes, compare-years, export-point) — TTL 24h (`GRID_TTL`), donc une entrée
  déjà en cache continuerait sinon à servir l'ancienne forme du payload jusqu'à 24h
- `junon:obs_era5_*` — TTL 24h également ; l'ancien `/era5/range` (grille journalière)
  n'est plus consommé par la page Climat, mais une entrée obsolète en cache ne serait
  de toute façon plus lue
- `junon:obs_piezo_detail:*` / `junon:obs_hydro_detail:*` — TTL 1h (`DETAIL_TTL`), sinon la
  section « Contexte climatique » de la page Station peut rester masquée/obsolète
  jusqu'à 1h après le déploiement
- `junon:obs_piezo_spi:*` / `junon:obs_hydro_spi:*` — TTL 24h (`SPLI_TTL`/`SSFI_TTL`)

(Les clés Redis sont suffixées d'un hash des paramètres — voir `dashboard/utils/cache.py::_make_key` —
d'où les patterns `*` ci-dessous plutôt que des clés exactes.) Les autres clés SPI en
cache (préfixes non listés ci-dessus) ne sont pas affectées par ce changement et
expirent naturellement sur leur propre TTL — aucune action requise.

```bash
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_climat_*' | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_era5_*' | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_piezo_detail:*' | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_hydro_detail:*' | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_piezo_spi:*' | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_hydro_spi:*' | xargs -r docker exec -i junon-redis redis-cli DEL
```

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
