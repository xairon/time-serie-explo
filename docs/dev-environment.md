# Isolated development environment (on dib)

> **The dev stack requires an NVIDIA GPU and a large machine.**
> `docker-compose.dev.yml` reserves `driver: nvidia, count: all` unconditionally and sets a
> 24 GB memory limit on the backend. On a host without `nvidia-container-toolkit`, Compose
> refuses to start it:
>
> ```
> could not select device driver "nvidia" with capabilities: [[gpu]]
> ```
>
> There is no CPU variant of the dev stack. To work without a GPU, use the production compose
> (`COMPOSE_FILE=docker-compose.yml`, `BACKEND=cpu`), which carries no GPU reservation, and
> accept that deep-learning training will run on CPU.

Two **completely separate** environments run on `dib-2019006065`:

| | **PROD** | **DEV** |
|---|---|---|
| Code | `main` branch | `dev` branch |
| Compose | `docker-compose.yml` (+ `.cuda.yml`) | `docker-compose.dev.yml` |
| Docker project | `time-serie-explo` | `junon-dev` |
| Front | `:49513` (+ public K8s) | `:49518` |
| Backend API | `:49514` (← K8s front) | `:49516` |
| MLflow | `:49512` | `:49517` |
| PostgreSQL database | `postgres_data` volume | `postgres_data_dev` volume |
| Models / artifacts | `mlflow_artifacts`, `data/`, `checkpoints/` | `mlflow_artifacts_dev`, `data-dev/`, `checkpoints-dev/` |

→ A model trained, a migration, or an account created in **dev** **never** affects prod.

## Starting / stopping dev

```bash
# Start (GPU included in the file → the explicit -f is correct here)
docker compose -p junon-dev -f docker-compose.dev.yml up -d --build

# Stop (keeps dev data)
docker compose -p junon-dev -f docker-compose.dev.yml down

# Reset everything (also removes dev volumes)
docker compose -p junon-dev -f docker-compose.dev.yml down -v
```

The dev database is empty on first start: apply the migrations, then create an admin.

```bash
docker exec junon-backend-dev sh -lc 'cd /app && alembic upgrade head'
docker exec junon-backend-dev sh -lc 'cd /app && python3 scripts/create_admin.py --email <toi@ex.fr> --name "<Nom>"'
```

## Workflow

1. Work on `dev`, test on the dev environment (`:49518`).
2. Once it is good: merge `dev` → `main`.
3. Merging into `main` rebuilds the front image (CI) → the IT department (DSI) redeploys; on the dib side,
   rebuild the prod stack if the **backend** changed:
   `docker compose up -d --build` (uses the `COMPOSE_FILE` from `.env`, GPU preserved).

> ⚠️ Do **not** rebuild the prod stack (`docker compose up`) from the `dev` branch:
> the prod backend `:49514` (used by the K8s front) would then run the dev code.
