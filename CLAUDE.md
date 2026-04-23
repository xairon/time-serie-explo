# CLAUDE.md - Junon Time-Series Explorer

## Project Overview

Plateforme de prévision piézométrique : React frontend + FastAPI backend + Darts (PyTorch Lightning) + Pastas (TFN).
Pages : Dashboard, Data, Training, Forecasting, Counterfactual Analysis, Observatory, Pastas Lab.

## Tech Stack

- **Frontend**: React 19, React Router 7, TanStack React Query 5, Plotly.js, Tailwind CSS 4, Vite
- **Backend**: FastAPI, SSE (Server-Sent Events) for async tasks
- **ML**: Darts (PyTorch Lightning), Optuna, SHAP/TimeSHAP/Captum, SoftCLT/TS2Vec
- **Tracking**: MLflow (experiments, model registry)
- **Database**: PostgreSQL (data warehouse via brgm-postgres), SQLAlchemy
- **Deployment**: Docker Compose (rootless), NVIDIA CUDA (RTX A6000)
- **Python**: 3.12, dependencies in `pyproject.toml`

## Architecture

```
frontend/
  src/
    pages/                    # React pages (UI only)
      DashboardPage.tsx       # Overview, health status
      DataPage.tsx            # Dataset import (CSV, DB), preview, profiling
      TrainingPage.tsx        # Model training config & monitoring
      ForecastingPage.tsx     # Inference, forecast modes, XAI panel
      CounterfactualPage.tsx  # Counterfactual analysis, IPS
      ObservatoryPage.tsx     # Advanced exploration/monitoring
    components/               # Reusable React components
      charts/                 # Plotly-based visualizations
      cards/                  # StatusCard, DatasetCard, ModelCard
      forms/                  # ModelConfigForm, ImportCSVForm, ImportDBForm
      forecasting/            # ModelSelector, ForecastView, ExplainabilityPanel
      counterfactual/         # CFComparisonView, CFTargetSelector, IPSBandsChart
      training/               # TrainingMonitor, TrainingResults
    hooks/                    # React Query hooks (data fetching)
      useDatasets.ts          # Dataset CRUD
      useModels.ts            # Model list
      useTraining.ts          # Training job management + SSE
      useForecasting.ts       # Forecast execution
      useCounterfactual.ts    # CF analysis + streaming
      useSSE.ts               # SSE wrapper for real-time updates
      useHealth.ts            # System health
    lib/
      api.ts                  # Centralized fetch-based API client → /api/v1/*

api/
  routers/                    # FastAPI routers
    datasets.py               # List, upload, validate datasets
    training.py               # Create/monitor training runs
    models.py                 # List trained models, load, config
    forecasting.py            # Single-step and rolling forecasts
    explainability.py         # Feature importance, SHAP, gradients, attention
    counterfactual.py         # PhysCF, Optuna, CoMTE + dual validation
    db_introspection.py       # BRGM data warehouse schema exploration
  schemas/                    # Pydantic request/response models

dashboard/
  utils/                      # Pure Python backend (NO framework dependency)
    training.py               # run_training_pipeline()
    model_factory.py          # ModelFactory (Darts models)
    preprocessing.py          # Data prep
    forecasting.py            # Prediction functions
    explainability/           # SHAP, attention, gradients, feature importance
    counterfactual/           # PhysCF, CoMTE, Pastas validation, IPS
    callbacks.py              # MetricsFileCallback (PyTorch Lightning)
    mlflow_client.py          # MLflow integration
    postgres_connector.py     # PostgreSQL via SQLAlchemy
    dataset_registry.py       # Dataset management
  config.py                   # Global configuration
```

**Key principles**:
- `dashboard/utils/` is pure Python with NO framework dependency
- `api/routers/` wraps utils as REST/SSE endpoints
- `frontend/` is pure React consuming `/api/v1/*` endpoints
- All visualizations use Plotly.js with dark theme

## Pastas Pipeline

- **Config**: `dashboard/utils/pastas/config.py` — BDLISA aquifer presets
- **Builder**: `dashboard/utils/pastas/builder.py` — model construction (accepts `add_trend` flag)
- **Fit**: `dashboard/utils/pastas/fit_service.py` — fitting + MLflow logging, returns `FitResult`
- **IO**: `dashboard/utils/pastas/io.py` — model load/save with LRU cache (maxsize=32)
- **Diagnostics**: `dashboard/utils/pastas/diagnostics.py` — QQ, PACF, normality tests
- **Outlier diagnostics**: `dashboard/utils/pastas/outlier_diagnostics.py` — classify outliers by climate/data/neighbors
- **Advanced analytics**: `recession.py`, `baseflow.py`, `spectral.py`, `signal_decomposition.py`, `cross_correlation.py`, `multi_station_residuals.py`, `input_quality.py`
- **Frontend results**: `FitResultsPanel.tsx` — main results dashboard with unified analysis chart
- **MLflow tags**: `cal_tmin`, `cal_tmax`, `val_tmin`, `val_tmax`, `station_id` — use `_get_cal_val_periods(run)` helper
- **Endpoints return `{cal: ..., val: ...}`** for period-dependent analyses (diagnostics, outliers, spectral, regional)

## Docker Setup

- **Port mapping**: 49513 (Nginx/App), 49512 (MLflow), PostgreSQL internal only
- **GPU**: BACKEND=cuda, image nvidia/cuda:12.6.3-runtime-ubuntu24.04
- **Network**: Connected to `hubeau_data_integration_default` for brgm-postgres access
- **Compose files**: `docker-compose.yml` + `docker-compose.cuda.yml` (GPU override)
- **COMPOSE_FILE** is set in `.env` → `docker compose` auto-loads both files
- **Rebuild**: `docker compose up -d --build` (NEVER use `-f` flags manually, `.env` handles it)
- **CRITICAL**: NEVER run `docker compose` with explicit `-f docker-compose.yml` only — this drops the CUDA overlay and disables GPU access. Always rely on the `COMPOSE_FILE` env var in `.env`.

## Database Connections

- **Junon internal DB**: host=postgres, port=5432, user=junon (inside Docker network)
- **BRGM data warehouse**: host=brgm-postgres, port=5432, user=postgres, db=postgres, schema=gold

## Development Conventions

- Keep `dashboard/utils/` free of any framework imports
- SQL queries use parameterized queries (SQLAlchemy `text()` with `:param` syntax)
- All SQL identifiers are validated with `_validate_identifier()` (prevent injection)
- Training uses `MetricsFileCallback` writing JSON, streamed via SSE to frontend
- Models are Darts `TorchForecastingModel` subclasses, created via `ModelFactory`
- Preprocessing outputs Darts `TimeSeries` objects
- Frontend hooks use TanStack React Query (5min stale time, 30min GC)
- API client in `frontend/src/lib/api.ts` — centralized fetch with timeout handling
- Long-running tasks (training, CF generation) use SSE streams

## Gotchas

- **No Node.js on host** — frontend runs in Docker only. Use `docker compose up -d --build frontend` to test changes
- **Plotly `fill` prop** — always use `as const` (e.g. `fill: 'tonexty' as const`) or TS build fails
- **React Query hooks returning `Record<string, unknown>`** — destructure with `as { data: any }` to avoid TS errors
- **Date format mismatch** — API dates may be `YYYY-MM-DD HH:MM:SS` while periods are `YYYY-MM-DD`. Compare by `.slice(0, 7)` for month matching
- **Residuals only cover cal period** in FitResponse — compute `obs - sim` on frontend for val period
- **Docker disk quota** — run `docker builder prune -f` if build fails with "disk quota exceeded"
- **`rtk` intercepts curl output** — use `rtk proxy curl` for raw JSON, or pipe to `python3 -c` for parsing

## Testing

```bash
pytest tests/
cd frontend && npm test
cd e2e && npx playwright test
```

## Skills Reference

### Workflow — Quand et comment travailler

| Skill | Quand l'utiliser | Fichiers/modules concernés |
|-------|-----------------|---------------------------|
| `superpowers:brainstorming` | **OBLIGATOIRE** avant toute conception (nouvelle feature, refacto, choix d'archi). Lancé automatiquement avant `EnterPlanMode`. | — |
| `superpowers:writing-plans` | Quand on a un spec/requirements et qu'il faut planifier l'implémentation multi-étapes. | — |
| `superpowers:executing-plans` | Pour exécuter un plan d'implémentation écrit (ex: suivre `docs/PLAN_API.md`). | — |
| `superpowers:subagent-driven-development` | Quand le plan contient des tâches indépendantes parallélisables. | — |
| `superpowers:dispatching-parallel-agents` | Quand on a 2+ tâches indépendantes à lancer en parallèle. | — |
| `feature-dev:feature-dev` | Développement guidé d'une feature complète (exploration → plan → implémentation → tests). | — |
| `superpowers:test-driven-development` | **OBLIGATOIRE** avant d'écrire du code : écrire les tests d'abord. | `tests/` |
| `superpowers:systematic-debugging` | Tout bug, erreur de test, comportement inattendu. Ne pas deviner — suivre le process. | — |
| `superpowers:using-git-worktrees` | Pour travailler sur une feature en isolation sans toucher `main`. | — |
| `superpowers:verification-before-completion` | **OBLIGATOIRE** avant de dire "c'est fini" — vérifier que tout passe. | — |
| `superpowers:requesting-code-review` | Après avoir terminé une feature/fix significative. | — |
| `superpowers:receiving-code-review` | Quand on reçoit un retour de code review — suivre le process de résolution. | — |
| `superpowers:finishing-a-development-branch` | Quand l'implémentation est terminée, tests passent, prêt à merger. | — |

### Git & CI

| Skill | Quand l'utiliser |
|-------|-----------------|
| `commit-commands:commit` | Créer un commit (raccourci `/commit`). |
| `commit-commands:commit-push-pr` | Commit + push + ouvrir une PR en une commande (`/commit-push-pr`). |
| `commit-commands:clean_gone` | Nettoyer les branches locales dont le remote a été supprimé. |
| `code-review:code-review` | Review une PR existante (`/code-review`). |

### ML & Training Pipeline

| Skill | Quand l'utiliser | Fichiers/modules concernés |
|-------|-----------------|---------------------------|
| `scientific-skills:pytorch-lightning` | Modifier le pipeline d'entraînement, callbacks, trainers, cycle de vie des modèles. | `dashboard/utils/training.py`, `dashboard/utils/callbacks.py`, `dashboard/utils/model_factory.py` |
| `scientific-skills:scikit-learn` | Preprocessing, feature engineering, métriques de validation, scalers. | `dashboard/utils/preprocessing.py` |
| `scientific-skills:statsmodels` | Analyse statistique, tests de stationnarité, décomposition de séries. | Analyses exploratoires |
| `scientific-skills:shap` | Explainability SHAP/TimeSHAP, feature importance, visualisations. | `dashboard/utils/explainability/` |
| `scientific-skills:transformers` | Si on utilise des modèles HuggingFace ou qu'on modifie les architectures Transformer/TFT. | `dashboard/utils/model_factory.py` |
| `scientific-skills:aeon` | Pour le machine learning sur séries temporelles (classification, clustering, anomalies). | Nouvelles features d'analyse |

### Visualisation & Frontend

| Skill | Quand l'utiliser | Fichiers/modules concernés |
|-------|-----------------|---------------------------|
| `scientific-skills:plotly` | **Toutes les visualisations du dashboard** — graphes forecast, métriques, explainability. | `frontend/src/components/charts/` |
| `scientific-skills:matplotlib` | Figures statiques pour export/rapport (pas pour le dashboard interactif). | Export PDF/PNG |
| `scientific-skills:seaborn` | Heatmaps de corrélation, distributions statistiques, pair plots. | Analyses exploratoires |
| `scientific-skills:scientific-visualization` | Figures publication-ready pour rapports ou présentations. | `figs/`, exports |
| `frontend-design:frontend-design` | Refonte UX des pages React, layout, composants. | `frontend/src/pages/`, `frontend/src/components/` |

### Analyse de données

| Skill | Quand l'utiliser | Fichiers/modules concernés |
|-------|-----------------|---------------------------|
| `scientific-skills:exploratory-data-analysis` | Analyser un nouveau jeu de données piézométrique avant entraînement. | `dashboard/utils/data_loader.py`, `dashboard/utils/preprocessing.py` |
| `scientific-skills:statistical-analysis` | Évaluation des métriques, validation croisée, tests d'hypothèses. | `dashboard/utils/training.py` |
| `scientific-skills:polars` | Si pandas est trop lent sur de gros datasets — alternative performante. | `dashboard/utils/data_loader.py`, `dashboard/utils/preprocessing.py` |
| `scientific-skills:pymc` | Modélisation bayésienne, incertitude des prédictions. | Nouvelles features |
| `scientific-skills:umap-learn` | Réduction de dimension pour visualiser l'espace des embeddings ou des features. | Analyses exploratoires |

### Hydrologie & Domaine

| Skill | Quand l'utiliser | Fichiers/modules concernés |
|-------|-----------------|---------------------------|
| `scientific-skills:networkx` | Graphes de réseau hydrographique, dépendances entre stations. | Nouvelles features |
| `scientific-skills:geopandas` | Cartographie des stations piézométriques, analyses spatiales. | Nouvelles features (carte) |
| `scientific-skills:sympy` | Équations physiques symboliques (loi de Darcy, bilan hydrique). | `dashboard/utils/counterfactual/physcf_optim.py` |

### Documents & Export

| Skill | Quand l'utiliser |
|-------|-----------------|
| `scientific-skills:pdf` | Générer/manipuler des rapports PDF. |
| `scientific-skills:xlsx` | Export Excel des résultats, métriques, datasets. |
| `scientific-skills:docx` | Rapports Word. |
| `scientific-skills:pptx` | Présentations des résultats. |
| `scientific-skills:scientific-slides` | Présentations scientifiques pour conférences/séminaires. |
| `scientific-skills:scientific-writing` | Rédaction de rapports/articles scientifiques. |
| `scientific-skills:latex-posters` | Posters de recherche LaTeX. |

### Maintenance CLAUDE.md

| Skill | Quand l'utiliser |
|-------|-----------------|
| `claude-md-management:revise-claude-md` | Fin de session — capturer les apprentissages (`/revise-claude-md`). |
| `claude-md-management:claude-md-improver` | Audit et amélioration des fichiers CLAUDE.md. |

### Recherche & Veille

| Skill | Quand l'utiliser |
|-------|-----------------|
| `scientific-skills:research-lookup` | Chercher de l'info récente sur un sujet (modèles, méthodes). |
| `scientific-skills:literature-review` | Revue de littérature structurée (ex: nouvelles approches de forecasting piézo). |
| `scientific-skills:scientific-brainstorming` | Idéation pour nouvelles features ou approches de recherche. |
