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

## Production deployment

The canonical deployment is described under **`deploy/`**:

- `deploy/dib-backend/` — backend stack (API + Postgres + Redis + MLflow) via
  Docker Compose. Copy `.env.example` to `.env` and fill in the secrets.
- `deploy/frontend/` — static frontend (nginx) deployed on Kubernetes
  (`deploy/frontend/k8s/`), proxying `/api/` to the backend.

See `deploy/README.md`, `deploy/frontend/README.md`, and the matching
`.env.example` for up-to-date details.

> Note: the combined root `docker-compose.yml` is kept for development; in
> production, use the separate stacks under `deploy/`.

## GDPR / retention

Authentication log purging (365-day retention by default) runs via cron:

```bash
uv run python -m scripts.purge_expired
```
