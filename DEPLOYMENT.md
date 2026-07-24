# Deployment guide

The application consists of a FastAPI backend (`api/`) and a React frontend
(`frontend/`). The legacy Streamlit interface has been removed.

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
| **Production** | `time-serie-explo` | `docker-compose.yml` + `docker-compose.cuda.yml` (via `COMPOSE_FILE` in `.env`) | `junon-backend`, `junon-frontend` | 49514 / 49513 |
| **Dev** | `junon-dev` | `docker-compose.dev.yml` only | `junon-*-dev` | 49516 / 49518 |

### Production (live stack on `dib`)

```bash
# from the repo root, on main
docker compose up -d --build backend frontend
```

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
