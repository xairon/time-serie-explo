# Deployment guide

The application consists of a FastAPI backend (`api/`) and a React frontend
(`frontend/`). The legacy Streamlit interface has been removed.

## Deploying the whole platform, in order

Junon is **not standalone**. It reads the Gold tables of the `hubeau_data_integration`
warehouse over a Docker network that the warehouse owns. Deploy in this order — steps 1 and 2
are not optional, and skipping either produces errors that look unrelated to their cause.

**1. The warehouse (`hubeau_data_integration`)**

```bash
git clone https://scm.univ-tours.fr/ringuet/hubeau_data_integration.git
cd hubeau_data_integration
bash scripts/init_volumes.sh          # external volumes, once
cp .env.example .env                  # set PG_PASSWORD, DAGSTER_PG_PASSWORD, COPERNICUS_API_KEY
docker compose up -d --build
```

Keep the directory name `hubeau_data_integration`: Compose derives the network name from it
(`hubeau_data_integration_default`), and Junon references that name literally.

**2. Load the data.** Dagster UI at `http://localhost:49500` → Jobs → `full_bootstrap`. Until
Gold tables exist, every Junon Observatory endpoint answers HTTP 500 with `UndefinedTable`.
A full load takes hours and tens of GB — restrict it first, see the warehouse's
`docs/OPERATIONS.md`.

**3. Junon (this repository)**

```bash
git clone https://scm.univ-tours.fr/ringuet/time-serie-explo.git
cd time-serie-explo
cp .env.example .env
sed -i "s|^JWT_SECRET=.*|JWT_SECRET=$(openssl rand -hex 32)|" .env
# BRGM_DB_PASSWORD must equal the warehouse's PG_PASSWORD
docker compose up -d --build
```

**4. Create the first account** — there is no self-registration:

```bash
docker compose exec backend python scripts/create_admin.py \
  --email you@univ-tours.fr --name "Your Name"
```

**5. Check.** `curl http://localhost:49513/api/v1/health` should answer
`{"status":"ok","db":"ok","redis":"ok",...}`, and logging in should return a
`junon_session` cookie.

### What breaks when the order is wrong

| Symptom | Cause |
|---------|-------|
| `network hubeau_data_integration_default declared as external, but could not be found` | Step 1 never ran, or the warehouse was cloned under a different directory name |
| Observatory 500 with `fe_sendauth: no password supplied` | `BRGM_DB_PASSWORD` empty or different from the warehouse's `PG_PASSWORD` |
| Observatory 500 with `relation "gold.…" does not exist` | Step 2 never ran |
| `junon-frontend` restarting in a loop at cold start | The backend was not healthy yet; it settles on its own once it is |

## Local development

### Backend (API)

```bash
# uv (recommended)
uv sync --extra cpu --extra api      # or --extra cuda on a GPU machine
uv run uvicorn api.main:app --reload

# Required variables (see deploy/dib-backend/.env.example):
#   JWT_SECRET (openssl rand -hex 32), DB_PASSWORD, ALLOWED_ORIGINS, DEBUG
```

Unless `DEBUG=true`, the API refuses to start if `JWT_SECRET` is missing, too
short (<32), or left at its default value, and if `DB_PASSWORD` is empty.

### Frontend

```bash
cd frontend
npm ci
npm run dev          # Vite dev server
npm run build        # production build (dist/)
```

### Database

```bash
uv run alembic upgrade head
uv run python -m scripts.create_admin       # create an admin account
```

## Deployed environments

There are **two distinct Docker Compose projects** on the `dib` server. Getting the project
name wrong recreates containers in the *other* environment, so always pass it explicitly for
dev.

| | Compose project | Files | Containers | Ports |
|---|---|---|---|---|
| **Production** | `time-serie-explo` | `docker-compose.yml` + `docker-compose.cuda.yml` (via `COMPOSE_FILE` in `.env`) | `junon-backend`, `junon-frontend`, `junon-postgres`, `junon-redis`, `junon-mlflow` | 49514 / 49513 |
| **Dev** | `junon-dev` | `docker-compose.dev.yml` only | `junon-*-dev` | 49516 / 49518 |

### Production (live stack on `dib`)

```bash
# from the repo root, on main
docker compose up -d --build
```

Starting the whole stack is now correct — there is no longer a second `nginx` service fighting
`frontend` for port 49513. `frontend` is the entry point and publishes `APP_PORT`.

> **Never pass `-f` here.** `.env` sets
> `COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml`; overriding it with an explicit
> `-f` drops the CUDA overlay and the backend loses the GPU. Service names are
> `backend`/`frontend` — *not* the container names `junon-backend`/`junon-frontend`.

### Dev

```bash
docker compose -p junon-dev -f docker-compose.dev.yml up -d --build backend-dev frontend-dev
```

> **`-p junon-dev` is mandatory.** Without it the project falls back to the directory name
> (`time-serie-explo`) and you would recreate **production** containers. This is the one
> exception to the "never pass `-f`" rule above, which applies to production only.
> Dev has its own Redis (`junon-redis-dev`) — purge its cache separately.

### Kubernetes target (`deploy/`)

`deploy/` holds the stacks for the DSI Kubernetes target, driven by GitLab CI → registry →
ArgoCD image-updater:

- `deploy/dib-backend/` — backend stack (API + Postgres + Redis + MLflow)
- `deploy/frontend/` — static frontend (nginx), `deploy/frontend/k8s/`, proxying `/api/`

See `deploy/README.md`, `deploy/frontend/README.md` and the matching `.env.example`.

> These are **not** what currently serves the live `dib` instance — that is the root
> Compose project above. Don't assume `deploy/` is the running production.

## GDPR / retention

Authentication log purging (365-day retention by default) runs via cron:

```bash
uv run python -m scripts.purge_expired
```
