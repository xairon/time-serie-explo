# Architecture

## Vue d'ensemble

L'application est composée de trois couches :

1. **Frontend React/TypeScript** (`frontend/`) — l'interface utilisateur (SPA, build Vite, servie par nginx).
2. **API FastAPI** (`api/`) — l'API HTTP authentifiée : observatoire, entraînement, prévision, Pastas, explicabilité, détection de pompage, administration.
3. **Bibliothèque métier** (`dashboard/utils/`) — code Python pur (sans interface graphique) importé par l'API : préparation des données, entraînement, registres de modèles/datasets, Pastas, etc.

> L'ancienne interface Streamlit a été retirée. `dashboard/utils/` reste la bibliothèque métier partagée.

## Séparation des responsabilités (entraînement / interface)

Le code d'entraînement est **indépendant de toute interface** : il écrit sa
progression dans un fichier JSON, que la couche interface lit séparément.

### Composants principaux

#### 1. Callbacks standards (`dashboard/utils/callbacks.py`)

- **`MetricsFileCallback`** : écrit les métriques dans un fichier JSON.
- **`create_training_callbacks()`** : factory de callbacks PyTorch Lightning standards.

Aucune dépendance à une interface graphique : utilisable en CLI, backend ou notebook.

#### 2. Lecture des métriques (`dashboard/utils/training_monitor.py`)

`TrainingMonitor` lit et parse le fichier JSON. En production, le suivi temps réel
est exposé au frontend par l'API via **SSE** (`api/routers/training.py`,
endpoint `/api/v1/training/{task_id}/stream`).

#### 3. Pipeline d'entraînement (`dashboard/utils/training.py`)

`run_training_pipeline()` utilise les callbacks standards et le paramètre
`metrics_file` pour le suivi.

#### 4. Factory de modèles (`dashboard/utils/model_factory.py`)

`ModelFactory` instancie dynamiquement les modèles Darts avec validation des
hyperparamètres.

## Structure du projet

```
time-serie-explo/
├── api/                       # API FastAPI
│   ├── main.py                # App + routes + middleware
│   ├── routers/               # Endpoints (training, forecasting, pastas, admin, …)
│   ├── auth/                  # Auth (JWT, ownership, audit, rate limit, erasure)
│   ├── models_db/             # Modèles SQLAlchemy (User, AuthEvent)
│   └── schemas/               # Schémas Pydantic
│
├── frontend/                  # SPA React/TypeScript (Vite)
│   └── src/
│       ├── pages/             # Pages (observatoire, AI lab, pastas, admin, …)
│       ├── components/        # Composants UI
│       └── lib/, contexts/    # Client API, état auth
│
├── dashboard/
│   ├── utils/                 # Bibliothèque métier (code Python pur)
│   │   ├── callbacks.py       # Callbacks PyTorch Lightning
│   │   ├── training.py        # Pipeline d'entraînement
│   │   ├── training_monitor.py# Lecture des métriques JSON
│   │   ├── model_factory.py   # Factory de modèles
│   │   ├── model_registry.py  # Registre des modèles
│   │   ├── dataset_registry.py# Registre des datasets
│   │   ├── pastas/, pumping_detection/, counterfactual/, explainability/
│   │   └── …
│   └── models_config.py       # Catalogue des architectures
│
├── alembic/                   # Migrations de base de données
├── scripts/                   # Scripts utilitaires (create_admin, purge_expired, …)
├── tests/                     # Tests pytest
├── deploy/                    # Déploiement canonique (front K8s + back compose)
├── pyproject.toml             # Configuration du projet (uv)
└── docker-compose.yml
```

## Flux de données (entraînement)

```
┌─────────────────────────────────────────────┐
│             PROCESSUS D'ENTRAÎNEMENT          │
│  PyTorch Lightning Trainer                    │
│  ├── MetricsFileCallback → metrics.json       │
│  ├── EarlyStopping (standard)                 │
│  └── autres callbacks standards               │
│  Modèle entraîné → sauvegardé (MLflow)        │
└─────────────────────────────────────────────┘
                    │ (fichier JSON)
                    ▼
┌─────────────────────────────────────────────┐
│        API FastAPI → SSE → Frontend React     │
│  /training/{task_id}/stream lit metrics.json  │
│  et pousse la progression au navigateur       │
└─────────────────────────────────────────────┘
```

## Format du fichier JSON

```json
{
  "status": "training",
  "start_time": 1234567890.123,
  "current_epoch": 5,
  "total_epochs": 50,
  "train_losses": [0.5, 0.4, 0.35, 0.32, 0.30],
  "val_losses": [0.6, 0.5, 0.45, 0.42, 0.40],
  "epochs": [1, 2, 3, 4, 5],
  "elapsed_seconds": 120.5,
  "eta_seconds": 1080.0,
  "last_update": 1234568010.123
}
```

## Bonnes pratiques

### À faire

1. Garder `dashboard/utils/` **sans import d'interface graphique** (code pur).
2. Utiliser uniquement des callbacks standards dans `run_training_pipeline()`.
3. Passer `metrics_file` pour le suivi ; exposer la progression via l'API (SSE).
4. Nettoyer les modèles avant sauvegarde (automatique).

### À éviter

1. Référencer une interface (UI) dans le code d'entraînement.
2. Sérialiser des objets d'interface dans les modèles.
