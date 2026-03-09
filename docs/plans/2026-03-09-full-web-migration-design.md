# Design : Junon Time-Series Explorer — Full Web Migration

> **Date** : 2026-03-09
> **Status** : Approved
> **Objectif** : Remplacer l'app Streamlit par une app web React + FastAPI standalone

## Contexte

Ecosysteme sur le serveur dib-2019006065 :

| Projet | Role | Stack | Port |
|--------|------|-------|------|
| hubeau_data_integration | Data warehouse (Bronze>Silver>Gold) | Dagster + DLT + dbt + PostgreSQL/TimescaleDB + Superset | 49500-49508 |
| junondashboard | Observatoire hydrologique (consultation) | React 19 + FastAPI + Redis + Nginx | 49510 |
| **time-serie-explo** | **ML : training, forecasting, counterfactual** | **Streamlit (a migrer)** | **49509** |

time-serie-explo est le dernier projet encore en Streamlit. On le migre en full web avec la meme stack que junondashboard.

## Decisions

- **App separee** (option B) — pas de fusion dans junondashboard (GPU requis, serveur different a terme)
- **UX revue** — on repense les pages, pas une reproduction 1:1 du Streamlit
- **Stack frontend** : React 19 + Vite + TypeScript + Tailwind 4 + TanStack Query + React Router 7 + **Plotly React** (pas Recharts)
- **Stack backend** : FastAPI async + SQLAlchemy 2 + Redis (cache-aside) + Nginx reverse proxy
- **Taches longues** : SSE (Server-Sent Events) + TaskManager (threads + stop_event)
- **MLflow expose** sur port 49511 (UI consultable directement)
- **Pas d'auth** v1 (reseau interne BRGM)
- **Streamlit conserve** en parallele pendant migration, supprime apres validation

## Architecture

```
Nginx :49509 (reverse proxy)
  /           -> Frontend (React SPA)
  /api/*      -> Backend (FastAPI :8000)

Frontend (React 19 + TS)     Backend (FastAPI)          Services
  Vite                         Uvicorn                    Redis 7
  Tailwind 4                   SQLAlchemy 2 async         PostgreSQL 15
  TanStack Query v5            SSE streaming              MLflow :49511
  Plotly React                 TaskManager
  React Router 7               dashboard/utils/ (pure Python)
                                         |
                               brgm-postgres (gold schema)
                               via hubeau_data_integration_default
```

## Backend API — Routers & Endpoints

7 routers, ~30 endpoints wrappant dashboard/utils/ :

### /api/v1/health
- `GET /` — status, GPU info, versions

### /api/v1/datasets
- `GET /` — liste datasets prepares
- `POST /` — upload CSV
- `GET /{id}` — details dataset
- `DELETE /{id}` — suppression
- `POST /import-db` — import depuis brgm-postgres (gold schema)
- `GET /{id}/preview` — apercu des donnees
- `GET /{id}/profile` — profiling statistique

### /api/v1/training
- `POST /start` — lance un training (retourne task_id)
- `GET /{id}/stream` — SSE stream des metriques epoch par epoch
- `POST /{id}/cancel` — arrete un training en cours
- `GET /history` — historique des runs (via MLflow)

### /api/v1/models
- `GET /` — liste modeles entraines
- `GET /{id}` — details modele + metriques
- `DELETE /{id}` — suppression
- `GET /{id}/download` — export ZIP (checkpoint + config)
- `GET /available` — architectures disponibles (NBEATS, TFT, TCN, etc.)

### /api/v1/forecasting
- `POST /single` — forecast fenetre unique
- `POST /rolling` — rolling forecast
- `POST /comparison` — comparaison multi-modeles
- `POST /global` — forecast global

### /api/v1/explainability
- `POST /feature-importance` — correlation, permutation, SHAP
- `POST /attention` — attention weights (TFT)
- `POST /shap` — TimeSHAP detailed
- `POST /gradients` — Captum gradient attributions

### /api/v1/counterfactual
- `POST /generate` — PhysCF (physics-based)
- `POST /generate-optuna` — Optuna optimization
- `POST /generate-comet` — COMET hydrologique
- `GET /ips-reference` — IPS reference pour une station
- `POST /pastas-validate` — validation duale Pastas + TFT

## TaskManager

Remplace st.session_state pour les taches longues :

```python
class TaskManager:
    _tasks: Dict[str, TaskInfo]

    def create(task_type, config) -> str        # UUID
    def get_status(task_id) -> TaskStatus        # pending/running/completed/failed/cancelled
    def get_result(task_id) -> Optional[dict]
    def cancel(task_id) -> bool                  # Sets stop_event
    def stream_metrics(task_id) -> AsyncIterator  # SSE
```

Le training ecrit metrics.json via MetricsFileCallback (inchange).
Le SSE stream lit ce fichier et pousse les updates au client.

## Frontend — Pages

| Page | Route | Contenu |
|------|-------|---------|
| Dashboard | `/` | Vue d'ensemble : modeles, derniers runs, GPU, datasets |
| Data | `/data` | Import DB/CSV, exploration, profiling, config features |
| Training | `/training` | Config modele, lancement, monitoring SSE, resultats |
| Forecasting | `/forecasting` | 4 modes forecast, metriques, explainability integree |
| Counterfactual | `/counterfactual` | 3 methodes CF, IPS, radar, Pastas, export |

## Phase 0 : Nettoyage utils (pre-requis)

| Fichier | Action | Effort |
|---------|--------|--------|
| plots.py | Supprimer 16x @st.cache_data + import st | 10 min |
| statistics.py | Supprimer 7x @st.cache_data + import st | 10 min |
| data_loader.py | Supprimer 1x @st.cache_data + import st | 5 min |
| export.py | Extraire add_download_button() vers components/ | 30 min |
| training_monitor.py | Extraire UI vers page 2, garder read_metrics() | 1h |
| state.py | Supprimer entierement | 30 min |

+ Extraire ~1300 lignes de logique metier des pages Streamlit vers utils/.

## Docker Compose

```yaml
services:
  postgres:
    image: postgres:15-alpine
    # port interne only

  redis:
    image: redis:7-alpine
    # 900MB LRU, interne

  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.8.1
    ports: ["49511:5000"]
    command: mlflow server --host 0.0.0.0 --allowed-hosts '*'

  backend:
    build: docker/backend/
    # GPU access, volumes: data/, checkpoints/, results/
    depends_on: [postgres, redis, mlflow]

  frontend:
    build: docker/frontend/
    # Nginx serving React SPA build

  nginx:
    image: nginx:alpine
    ports: ["49509:80"]
    # Reverse proxy -> frontend + backend

networks:
  default:
  hubeau_data_integration_default:
    external: true
```

## Contraintes

- GPU sur le backend uniquement (NVIDIA RTX A6000)
- brgm-postgres accessible via reseau externe Docker
- MLflow expose pour consultation directe
- dashboard/utils/ reste la source de verite pour toute la logique metier
- Pas de duplication de logique entre API et utils
