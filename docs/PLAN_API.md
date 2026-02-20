# Plan : API-iser le projet Junon Time-Series

## Contexte

Le projet est une plateforme Streamlit de prévision piézométrique (4 pages). **87% du backend (`dashboard/utils/`) est déjà découplé de Streamlit** — les modules de training, forecasting, preprocessing, counterfactual, explainability sont du pur Python. Le travail consiste à créer une couche API FastAPI fine qui délègue à ces modules existants, et à extraire les ~13% de logique métier encore enfouie dans les pages Streamlit.

---

## Architecture cible

```
api/                          ← NOUVEAU PACKAGE
  main.py                     # FastAPI app, lifespan, CORS
  routers/                    # Endpoints REST + SSE
    health.py                 # GET /health, GET /config
    datasets.py               # CRUD datasets + upload CSV
    training.py               # Start/cancel/status/SSE stream
    models.py                 # Registry, download ZIP, delete
    forecasting.py            # Single window, rolling, comparison
    explainability.py         # SHAP, attention, importance, gradients
    counterfactual.py         # IPS, PhysCF, Optuna, COMET, Pastas
  schemas/                    # Pydantic models (request/response)
    common.py, datasets.py, training.py, models.py,
    forecasting.py, explainability.py, counterfactual.py
  services/                   # Couche métier (wraps dashboard/utils)
    task_manager.py           # Remplace st.session_state (thread-safe)
    dataset_service.py        # Upload, prepare, validate
    training_service.py       # Thread management + SSE progress
    forecasting_service.py    # Wrap forecasting.py
    explainability_service.py # Wrap explainability/*
    counterfactual_service.py # Wrap counterfactual/*
  sse/
    training_stream.py        # SSE event generator (lit metrics.json)
  middleware/
    auth.py                   # API key optionnelle
    error_handler.py          # Exception → JSON
dashboard/utils/              ← EXISTANT (partagé Streamlit + API)
  training.py, forecasting.py, preprocessing.py, etc.  # Déjà pur
```

**Principe** : `api/routers/` → `api/services/` → `dashboard/utils/` (aucune logique métier dans les routers)

---

## Endpoints (~25)

### Health & Config
- `GET /api/v1/health` → device, mlflow_uri, postgres status
- `GET /api/v1/config` → modèles disponibles, métriques, variables

### Datasets (7 endpoints)
- `POST /api/v1/datasets/upload` — Upload CSV + metadata (Form + File)
- `POST /api/v1/datasets/from-database` — Import depuis PostgreSQL
- `GET /api/v1/datasets` — Liste des datasets préparés
- `GET /api/v1/datasets/{id}` — Détails d'un dataset
- `GET /api/v1/datasets/{id}/preview` — 50 premières lignes + stats
- `POST /api/v1/datasets/{id}/prepare` — Preprocess + save (fill, normalize, split)
- `DELETE /api/v1/datasets/{id}`

### Training (5 endpoints)
- `POST /api/v1/training/start` → task_id (UUID)
- `GET /api/v1/training/{task_id}/status` → epoch, losses, ETA
- `GET /api/v1/training/{task_id}/stream` → **SSE** (text/event-stream)
- `POST /api/v1/training/{task_id}/cancel`
- `GET /api/v1/training/history`

### Models (4 endpoints)
- `GET /api/v1/models` — Liste (MLflow registry)
- `GET /api/v1/models/{run_id}` — Détails + métriques
- `GET /api/v1/models/{run_id}/download` → ZIP (StreamingResponse)
- `DELETE /api/v1/models/{run_id}`

### Forecasting (4 endpoints)
- `POST /api/v1/forecasting/single-window`
- `POST /api/v1/forecasting/rolling`
- `POST /api/v1/forecasting/comparison`
- `POST /api/v1/forecasting/global`

### Explainability (4 endpoints)
- `POST /api/v1/explainability/feature-importance`
- `POST /api/v1/explainability/attention`
- `POST /api/v1/explainability/shap`
- `POST /api/v1/explainability/summary`

### Counterfactual (5 endpoints)
- `POST /api/v1/counterfactual/ips-reference`
- `POST /api/v1/counterfactual/ips-classify`
- `POST /api/v1/counterfactual/generate`
- `POST /api/v1/counterfactual/generate-batch`
- `POST /api/v1/counterfactual/pastas-validate`

---

## Extractions critiques (logique enfouie dans les pages Streamlit)

### 1. `prepare_station_data()` → `dashboard/utils/preprocessing.py`
**Source** : `2_Train_Models.py` lignes 620-739
Fonction pure qui : filtre par station, fill missing, supprime doublons, convertit en Darts TimeSeries, split train/val/test, normalise.
~120 lignes, appelable par Streamlit ET API.

### 2. `run_training_thread()` → `api/services/training_service.py`
**Source** : `2_Train_Models.py` lignes 871-1023
Thread function quasi-verbatim. Remplacer `_write_log_to_state()` par `TaskManager.update_task()`, `st.session_state` par `TaskManager`.

### 3. Helpers CF → `api/services/counterfactual_service.py`
**Source** : `4_Counterfactual_Analysis.py`
- `_detect_columns()` (~lignes 140-180)
- `_build_full_df()` (~lignes 200-250)
- `_extract_real_scaler_params()` (~lignes 260-300)
- `_build_physcf_scaler()` (~lignes 310-330)
- Bloc IPS reference (lignes 329-421)
- Boucle CF generation (lignes 845-976)

### 4. `load_model_data()` → `api/services/forecasting_service.py`
**Source** : `3_Forecasting.py` lignes 181-234
Remplacer `@st.cache_resource` par `functools.lru_cache(maxsize=10)`.

---

## Training Progress : SSE (pas WebSocket)

L'architecture actuelle utilise déjà un fichier JSON temporaire écrit par `MetricsFileCallback` (dans le thread de training) et lu par le main thread Streamlit. SSE est le mapping naturel :

```
Client ←─ SSE ──── FastAPI ──── lit metrics.json ──── Thread Training
                                                         ↓
                                                   run_training_pipeline()
                                                   (dashboard/utils/training.py)
```

- `POST /training/start` → crée thread + metrics_file + task_id
- `GET /training/{id}/stream` → async generator qui lit metrics.json toutes les 1s
- `POST /training/{id}/cancel` → set threading.Event (pattern existant)

---

## TaskManager (remplace st.session_state)

```python
class TaskManager:
    """Thread-safe task registry pour utilisateurs concurrents."""
    _tasks: Dict[str, dict]    # task_id → {status, thread, metrics_file, results, ...}
    _lock: threading.Lock

    def create_task(task_type, config) -> str       # UUID
    def get_task(task_id) -> Optional[dict]
    def update_task(task_id, **kwargs)              # thread-safe
    def cancel_task(task_id) -> bool                # set stop_event
    def cleanup_old_tasks(max_age_hours=24)
```

---

## Nettoyage utils (6 fichiers, ~1 jour)

| Fichier | Action |
|---------|--------|
| `plots.py` | Retirer 15× `@st.cache_data` — les fonctions retournent déjà des `go.Figure` purs |
| `statistics.py` | Retirer 7× `@st.cache_data` |
| `data_loader.py` | Retirer 1× `@st.cache_data` sur `get_data_summary()` |
| `state.py` | Supprimer (remplacé par TaskManager) |
| `export.py` | Garder `create_model_archive()` (pur), ignorer `add_download_button()` |
| `training_monitor.py` | Garder `read_metrics()` (pur), ignorer `display_progress()` |

---

## Phases d'implémentation

| Phase | Scope | Jours | Dépendances |
|-------|-------|:-----:|-------------|
| **0** | Foundation : FastAPI app, health, Docker, middleware | **2** | — |
| **1** | Dataset Management : upload, prepare, CRUD | **3** | Phase 0 |
| **2** | Training Pipeline : start, SSE, cancel, TaskManager | **4** | Phases 0+1 |
| **3** | Model Registry : list, download, delete | **2** | Phase 0 |
| **4** | Forecasting : 4 types de forecast | **3** | Phases 0+3 |
| **5** | Explainability : importance, SHAP, attention | **3** | Phases 0+3+4 |
| **6** | Counterfactual : IPS, CF, Pastas | **3** | Phases 0+3+4 |
| **7** | Tests intégration + polish | **1** | Toutes |
| | **TOTAL** | **21 jours** | |

Chemin critique : Phase 0 → 1 → 2 → 7 = **10 jours minimum**
Phases 3-6 parallélisables si 2 devs.

---

## Nouvelles dépendances (pyproject.toml)

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "sse-starlette>=1.8.0",
]
```

---

## Docker (ajout service API)

Nouveau service dans `docker-compose.yml` :
```yaml
api:
  build:
    context: .
    dockerfile: docker/Dockerfile.api
  ports:
    - "8000:8000"
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
    - JUNON_API_KEY=${API_KEY:-}
  volumes:  # mêmes bind mounts que streamlit
  depends_on: [postgres, mlflow]
```

---

## Vérification

1. `pytest api/tests/` — tests unitaires par endpoint (httpx + TestClient)
2. `docker compose up -d --build` — vérifier les 4 services (postgres, mlflow, streamlit, api)
3. `curl http://localhost:8000/api/v1/health` — healthcheck
4. Upload CSV → prepare → train → forecast → explain → CF : workflow complet via API
5. SSE training stream : vérifier avec `curl -N http://localhost:8000/api/v1/training/{id}/stream`

---

## Fichiers critiques à modifier/créer

**Existants à modifier :**
- `dashboard/utils/preprocessing.py` — ajouter `prepare_station_data()`
- `pyproject.toml` — ajouter extra `api`
- `docker-compose.yml` — ajouter service `api`

**Nouveaux à créer :**
- `api/` — tout le package (~20 fichiers)
- `docker/Dockerfile.api` — image pour le service API
