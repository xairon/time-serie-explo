# Junon — Piezometric Forecasting Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)

Full-stack platform for groundwater level forecasting and scenario analysis. It combines
physically-based transfer function-noise models ([Pastas](https://pastas.readthedocs.io/)) with
deep learning forecasters ([Darts](https://unit8co.github.io/darts/)) under a unified MLflow
registry, on top of the [BRGM](https://www.brgm.fr/) data warehouse, exposed through a React
frontend and a FastAPI backend.

**Status** — active. Documentation verified on 2026-08-24.

New here? Finish this page, then open [docs/README.md](docs/README.md) for the documentation map.

## Quick start

### Requirements

- Docker Engine + Docker Compose v2
- NVIDIA GPU + CUDA 12.x drivers (optional, for accelerated deep-learning training)
- ~8 GB disk for the images
- **The `hubeau_data_integration` stack, started at least once** — see below

> **Junon does not own its data, and it cannot start alone.** The Observatory reads the Gold
> tables of the [`hubeau_data_integration`](https://scm.univ-tours.fr/ringuet/hubeau_data_integration)
> warehouse, over the Docker network `hubeau_data_integration_default` at host
> `brgm-postgres`. That network is declared `external: true` here, so Compose refuses to start
> Junon until it exists:
>
> ```
> network hubeau_data_integration_default declared as external, but could not be found
> ```
>
> Bring the hubeau stack up first. Note that the network name is derived from hubeau's
> **directory name**: cloning it as anything other than `hubeau_data_integration`, or setting
> `COMPOSE_PROJECT_NAME`, renames the network and breaks Junon. Rename the network in
> `docker-compose.yml` if you must.

### 1. Install and run

```bash
git clone https://scm.univ-tours.fr/ringuet/time-serie-explo.git
cd time-serie-explo
cp .env.example .env

# JWT_SECRET has no default and the stack refuses to start without it:
sed -i "s|^JWT_SECRET=.*|JWT_SECRET=$(openssl rand -hex 32)|" .env
# Then edit .env: POSTGRES_PASSWORD, BRGM_DB_PASSWORD, ALLOWED_ORIGINS, COMPOSE_FILE

docker compose up -d --build
```

For GPU acceleration, set `COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml` in `.env`.

### 2. Create the first account — you cannot log in without this

Junon has **no self-registration and no SSO**: the admin creates every account. On a fresh
install there is no account at all, so the login screen is a dead end until you run:

```bash
docker compose exec backend python scripts/create_admin.py \
  --email you@univ-tours.fr --name "Your Name"
```

It prompts for a password without echoing it. `--password` also works but leaks through `ps`
and shell history, so prefer the prompt. Creating further accounts, roles and the secret
policy are covered in [docs/account-management.md](docs/account-management.md).

### 3. Open it

- Application: `http://localhost:49513`
- MLflow UI: `http://localhost:49512`

Both ports come from `.env` (`APP_PORT`, `MLFLOW_PORT`).

### 4. Verify

```bash
curl http://localhost:49513/api/v1/health      # API health (no auth needed)
```

Run the test suite **on the host**, not in the container:

```bash
uv run pytest tests/ -v --maxfail=5
```

`tests/` is not copied into the backend image (`docker/backend/Dockerfile`) and `pytest` lives
in the `dev` extra, which the image does not install — so `docker compose exec backend pytest`
cannot work.

> **Known gap: nothing runs these tests automatically.** The only test automation is
> `.github/workflows/test.yml`, a GitHub Actions workflow, and this repository is hosted on
> GitLab. `.gitlab-ci.yml` has a single `build` stage and never mentions pytest. Until a test
> job is added there, the suite only runs when someone runs it by hand.

## What it does

### Pastas Lab — transfer function-noise models

- TFN calibration on piezometric series, with precipitation and evapotranspiration as forcings
- **Auto-fit** with STOWA quality criteria and a configuration grid search
- **Calibration/validation split** with full diagnostics (NSE, KGE, RMSE, residual
  autocorrelation, normality)
- **Prospective scenarios** — synthetic pumping (drinking water, irrigation, industrial),
  climate trends, stress scaling
- **Adaptive bounds** — drawdown estimated from the calibrated step response, with
  physically-bounded recommendations
- **BDLISA-aware presets** — response-function defaults per aquifer family (alluvial,
  sedimentary, karst, fractured, volcanic)

### AI Lab — deep learning forecasters

- 12+ Darts models: TFT, Transformer, N-BEATS, N-HiTS, LSTM, GRU, TCN, TiDE, TSMixer, DLinear,
  NLinear
- Hyperparameter optimization with Optuna
- Real-time training monitoring over Server-Sent Events
- Explainability: SHAP, TimeSHAP, Captum gradients, attention weights
- Counterfactual analysis: PhysCF, CoMTE with dual validation

### Observatory

Spatial exploration of piezometric and hydrometric stations, BDLISA aquifer overlays, drought
indices, and a Climat tab over the France-wide 0.1° ERA5 grid. Full description, API surface
and the cache-purge procedure in [docs/observatory.md](docs/observatory.md); what the indices
mean and how they are validated in [docs/climate-indices.md](docs/climate-indices.md).

## Architecture

```
frontend/            React SPA (Vite, Tailwind, TanStack Query, Plotly)
api/                 FastAPI REST + SSE
dashboard/utils/     Framework-free Python core (Pastas, XAI, counterfactual, training)
tests/               Pytest suite
docker/              Per-service Dockerfiles
deploy/              Deployment manifests (frontend K8s + backend compose)
```

The Python core under `dashboard/utils/` has no framework dependency: it is callable from
notebooks, scripts or the API layer without modification. Details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Layer | Stack |
|---|---|
| Frontend | React 19, React Router 7, TanStack Query 5, Plotly.js, Tailwind CSS 4, Vite |
| Backend | FastAPI, Server-Sent Events, SQLAlchemy |
| Forecasting | Darts (PyTorch Lightning), Pastas |
| Tuning & XAI | Optuna, SHAP, TimeSHAP, Captum |
| Tracking | MLflow |
| Database | PostgreSQL (BRGM gold layer) |
| Deployment | Docker Compose (rootless), NVIDIA CUDA |

## Deployment

Production is **split**: the frontend runs on the IT department's Kubernetes cluster, the
backend, GPU and databases stay on `dib-2019006065`. The CI (`.gitlab-ci.yml`) builds and
pushes the frontend image on every push to `main`, and the K8s manifests are ready in
`deploy/frontend/k8s/`.

One network requirement gates everything: the route *cluster pods → `10.195.25.16:49514`*
must be open.

Full procedure in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) and
[deploy/frontend/README.md](deploy/frontend/README.md).

## Development

```bash
docker compose up -d --build             # rebuild after code changes
docker compose up -d --build frontend    # one service only
docker compose logs -f backend           # backend logs
```

TypeScript type-check without installing Node on the host — the `frontend` service image is
`nginx:alpine` and carries neither Node nor the sources, so it cannot run `tsc` itself:

```bash
docker run --rm -v "$PWD/frontend":/app -w /app node:20-alpine \
  sh -c "npm ci && npx tsc --noEmit"
```

## Documentation

The map is [docs/README.md](docs/README.md).

## Citation

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

Built around [Pastas](https://github.com/pastas/pastas) (Collenteur et al.),
[Darts](https://github.com/unit8co/darts) (Unit8), and the BRGM Hub'Eau / BDLISA / ADES open
data services.
