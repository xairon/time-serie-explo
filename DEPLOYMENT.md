# Guide de déploiement

L'application se compose d'une API FastAPI (`api/`) et d'un frontend React
(`frontend/`). L'ancienne interface Streamlit a été retirée.

## Développement local

### Backend (API)

```bash
# uv (recommandé)
uv sync --extra cpu --extra api      # ou --extra cuda sur machine GPU
uv run uvicorn api.main:app --reload

# Variables requises (voir deploy/dib-backend/.env.example) :
#   JWT_SECRET (openssl rand -hex 32), DB_PASSWORD, ALLOWED_ORIGINS, DEBUG
```

En dehors de `DEBUG=true`, l'API refuse de démarrer si `JWT_SECRET` est absent,
trop court (<32) ou laissé à la valeur par défaut, et si `DB_PASSWORD` est vide.

### Frontend

```bash
cd frontend
npm ci
npm run dev          # serveur de dev Vite
npm run build        # build de production (dist/)
```

### Base de données

```bash
uv run alembic upgrade head
uv run python -m scripts.create_admin       # créer un compte admin
```

## Déploiement (production)

Le déploiement canonique est décrit dans **`deploy/`** :

- `deploy/dib-backend/` — stack backend (API + Postgres + Redis + MLflow) via
  Docker Compose. Copier `.env.example` vers `.env` et renseigner les secrets.
- `deploy/frontend/` — frontend statique (nginx) déployé sur Kubernetes
  (`deploy/frontend/k8s/`), proxy `/api/` vers le backend.

Voir `deploy/README.md`, `deploy/frontend/README.md` et le `.env.example`
correspondant pour les détails à jour.

> Note : le `docker-compose.yml` combiné à la racine est conservé pour le
> développement ; en production, utiliser les stacks séparées de `deploy/`.

## RGPD / rétention

La purge des journaux d'authentification (rétention 365 jours par défaut) se fait
via cron :

```bash
uv run python -m scripts.purge_expired
```
