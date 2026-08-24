# Architecture

## Overview

The application has three layers:

1. **React/TypeScript frontend** (`frontend/`) — the user interface (SPA, Vite build, served by nginx).
2. **FastAPI backend** (`api/`) — the authenticated HTTP API: observatory, training, forecasting, Pastas, explainability, pumping detection, administration.
3. **Business library** (`dashboard/utils/`) — pure Python (no GUI) imported by the API: data preparation, training, model/dataset registries, Pastas, etc.

> The legacy Streamlit interface has been removed. `dashboard/utils/` remains the shared business library.

## One entry point

The `frontend` container is the single HTTP entry point. Its image is built on `nginx:alpine`
and its config (`docker/frontend/nginx.conf`) does everything: serves the SPA, falls back to
`index.html` for client-side routes, proxies `/api/` to the backend with SSE buffering
disabled, applies the security headers and the rate limits (429 over the limit, matching the
backend's own convention).

A second, standalone `nginx` service used to sit in front of it. It was removed: it duplicated
this reverse proxy, published the same host port (so a full `docker compose up -d` always
failed with `port is already allocated`), and was never started in production — which meant
the rate limiting and security headers it carried were, in practice, never applied. Its only
unique payload was `observatory-bridge.js`, an iframe bridge to `junondashboard` that became
dead code when the Observatory was ported natively into this SPA. Everything worth keeping was
folded into `docker/frontend/nginx.conf`.

## Separation of concerns (training / interface)

Training code is **independent of any interface**: it writes its progress to a
JSON file, which the interface layer reads separately.

### Main components

#### 1. Standard callbacks (`dashboard/utils/callbacks.py`)

- **`MetricsFileCallback`**: writes metrics to a JSON file.
- **`create_training_callbacks()`**: factory for standard PyTorch Lightning callbacks.

No dependency on any GUI: usable from a CLI, the backend, or notebooks.

#### 2. Metrics reading (`dashboard/utils/training_monitor.py`)

`TrainingMonitor` reads and parses the JSON file. In production, real-time
progress is exposed to the frontend by the API via **SSE**
(`api/routers/training.py`, endpoint `/api/v1/training/{task_id}/stream`).

#### 3. Training pipeline (`dashboard/utils/training.py`)

`run_training_pipeline()` uses the standard callbacks and the `metrics_file`
parameter for progress tracking.

#### 4. Model factory (`dashboard/utils/model_factory.py`)

`ModelFactory` dynamically instantiates Darts models with hyperparameter
validation.

## Project structure

```
time-serie-explo/
├── api/                       # FastAPI backend
│   ├── main.py                # App + routes + middleware
│   ├── routers/               # Endpoints (training, forecasting, pastas, admin, …)
│   ├── auth/                  # Auth (JWT, ownership, audit, rate limit, erasure)
│   ├── models_db/             # SQLAlchemy models (User, AuthEvent)
│   └── schemas/               # Pydantic schemas
│
├── frontend/                  # React/TypeScript SPA (Vite)
│   └── src/
│       ├── pages/             # Pages (observatory, AI lab, pastas, admin, …)
│       ├── components/        # UI components
│       └── lib/, contexts/    # API client, auth state
│
├── dashboard/
│   ├── utils/                 # Business library (pure Python)
│   │   ├── callbacks.py       # PyTorch Lightning callbacks
│   │   ├── training.py        # Training pipeline
│   │   ├── training_monitor.py# JSON metrics reader
│   │   ├── model_factory.py   # Model factory
│   │   ├── model_registry.py  # Model registry
│   │   ├── dataset_registry.py# Dataset registry
│   │   ├── pastas/, pumping_detection/, counterfactual/, explainability/
│   │   └── …
│   └── models_config.py       # Architecture catalogue
│
├── alembic/                   # Database migrations
├── scripts/                   # Utility scripts (create_admin, purge_expired, …)
├── tests/                     # pytest suite
├── deploy/                    # Canonical deployment (frontend K8s + backend compose)
├── pyproject.toml             # Project configuration (uv)
└── docker-compose.yml
```

## Data flow (training)

```
┌─────────────────────────────────────────────┐
│               TRAINING PROCESS                │
│  PyTorch Lightning Trainer                    │
│  ├── MetricsFileCallback → metrics.json       │
│  ├── EarlyStopping (standard)                 │
│  └── other standard callbacks                 │
│  Trained model → saved (MLflow)               │
└─────────────────────────────────────────────┘
                    │ (JSON file)
                    ▼
┌─────────────────────────────────────────────┐
│       FastAPI backend → SSE → React frontend  │
│  /training/{task_id}/stream reads metrics.json│
│  and pushes progress to the browser           │
└─────────────────────────────────────────────┘
```

## JSON file format

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

## Best practices

### Do

1. Keep `dashboard/utils/` **free of any GUI import** (pure code).
2. Use only standard callbacks in `run_training_pipeline()`.
3. Pass `metrics_file` for progress; expose it to the frontend via the API (SSE).
4. Clean models before saving (done automatically).

### Avoid

1. Referencing a UI in training code.
2. Serializing UI objects into models.
