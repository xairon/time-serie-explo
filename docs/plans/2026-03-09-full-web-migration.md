# Full Web Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Streamlit app with a React + FastAPI web application that reproduces all ML features (dataset prep, training, forecasting, counterfactual) with a modern UX.

**Architecture:** FastAPI backend wrapping existing `dashboard/utils/` (pure Python), React 19 frontend with Plotly, Nginx reverse proxy, Redis cache. Same patterns as junondashboard (`/home/ringuet/junondashboard/`). SSE for training progress, TaskManager for long-running tasks.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, Redis 7, React 19, Vite, TypeScript, Tailwind 4, TanStack Query v5, Plotly React, Nginx, Docker Compose, CUDA GPU support.

**Reference project:** `/home/ringuet/junondashboard/` — copy patterns from there.

---

## Task Map & Dependencies

```
Phase 0: Utils Cleanup (no deps)
  Task 1: Decontaminate plots.py, statistics.py, data_loader.py
  Task 2: Split export.py and training_monitor.py
  Task 3: Delete state.py, extract business logic from pages

Phase 1: Backend Foundation (depends on Phase 0)
  Task 4: FastAPI app skeleton (main.py, config, database, cache, json_response)
  Task 5: Health + datasets router
  Task 6: Training router + TaskManager + SSE
  Task 7: Models router
  Task 8: Forecasting router
  Task 9: Explainability router
  Task 10: Counterfactual router

Phase 2: Frontend Foundation (no deps, parallel with Phase 1)
  Task 11: Vite + React + Tailwind scaffold
  Task 12: Layout, routing, API client, types

Phase 3: Frontend Pages (depends on Phase 1 + 2)
  Task 13: Dashboard page (home)
  Task 14: Data page (import, explore, configure)
  Task 15: Training page (config, launch, SSE monitor, results)
  Task 16: Forecasting page (4 modes, metrics, explainability)
  Task 17: Counterfactual page (3 methods, IPS, radar, Pastas)

Phase 4: Docker & Integration (depends on all)
  Task 18: Docker Compose (6 services + GPU + nginx)
  Task 19: Integration tests + smoke tests
```

**Parallelizable:** Tasks 1-3 are independent. Tasks 5-10 are independent. Tasks 11-12 can run parallel to Phase 1. Tasks 13-17 are independent once Phase 1+2 done.

---

## Phase 0: Utils Cleanup

### Task 1: Decontaminate simple utils files

Remove Streamlit imports and `@st.cache_data` decorators from 3 files. These are pure Python functions that just had caching decorators added.

**Files:**
- Modify: `dashboard/utils/plots.py` — remove `import streamlit as st` (line 11) and all `@st.cache_data(ttl=3600)` decorators (lines 42, 118, 157, 216, 284, 338, 389, 432, 490, 528, 572, 664, 727, 790)
- Modify: `dashboard/utils/statistics.py` — remove `import streamlit as st` (line 10) and all `@st.cache_data(ttl=3600)` decorators (lines 13, 51, 98, 162, 208, 253, 290)
- Modify: `dashboard/utils/data_loader.py` — remove `import streamlit as st` (line 9) and `@st.cache_data(ttl=3600)` decorator (line 279)

**Step 1:** Read each file, identify all `st.` references.

**Step 2:** Remove all `import streamlit as st` lines and all `@st.cache_data(...)` decorator lines. Do NOT remove the functions themselves.

**Step 3:** Verify no remaining `st.` references in any of the 3 files:
```bash
grep -n "import streamlit\|st\." dashboard/utils/plots.py dashboard/utils/statistics.py dashboard/utils/data_loader.py
```
Expected: no matches.

**Step 4:** Verify Python syntax is valid:
```bash
python -c "import dashboard.utils.plots; import dashboard.utils.statistics; import dashboard.utils.data_loader; print('OK')"
```

**Step 5:** Commit:
```bash
git add dashboard/utils/plots.py dashboard/utils/statistics.py dashboard/utils/data_loader.py
git commit -m "refactor: remove Streamlit decorators from plots, statistics, data_loader"
```

---

### Task 2: Split export.py and training_monitor.py

Separate pure Python logic from Streamlit UI code.

**Files:**
- Modify: `dashboard/utils/export.py` — keep only `create_model_archive()`, remove `add_download_button()` and `import streamlit`
- Create: `dashboard/components/export_button.py` — move `add_download_button()` here (with streamlit import)
- Modify: `dashboard/utils/training_monitor.py` — keep `TrainingMonitor.__init__()` and `read_metrics()`, remove `display_progress()` and `create_training_monitor_fragment()`
- Create: `dashboard/components/training_monitor_ui.py` — move UI rendering here
- Modify: `dashboard/components/sidebar/export_section.py` — update import path
- Modify: `dashboard/training/pages/2_Train_Models.py` — update import paths

**Step 1:** Read `dashboard/utils/export.py`. Split:
- KEEP in `dashboard/utils/export.py`: `create_model_archive()` (pure Python, uses `io.BytesIO`, `zipfile`, `Path`)
- MOVE to `dashboard/components/export_button.py`: `add_download_button()` (uses `st.warning`, `st.spinner`, `st.download_button`, `st.error`)

Remove `import streamlit as st` from `dashboard/utils/export.py`.
In `dashboard/components/export_button.py`, import from `dashboard.utils.export import create_model_archive`.

**Step 2:** Read `dashboard/utils/training_monitor.py`. Split:
- KEEP in `dashboard/utils/training_monitor.py`: `TrainingMonitor` class with `__init__()` and `read_metrics()` only. Remove `display_progress()`. Remove `import streamlit as st`.
- MOVE to `dashboard/components/training_monitor_ui.py`: `display_progress()` function (standalone, takes monitor + placeholders as args) and `create_training_monitor_fragment()`.

**Step 3:** Update imports in:
- `dashboard/components/sidebar/export_section.py` — change import to `from dashboard.components.export_button import add_download_button`
- `dashboard/training/pages/2_Train_Models.py` — change import to `from dashboard.components.training_monitor_ui import create_training_monitor_fragment`

**Step 4:** Verify no `st.` in utils files:
```bash
grep -n "import streamlit\|st\." dashboard/utils/export.py dashboard/utils/training_monitor.py
```
Expected: no matches.

**Step 5:** Verify Python imports:
```bash
python -c "from dashboard.utils.export import create_model_archive; print('OK')"
python -c "from dashboard.utils.training_monitor import TrainingMonitor; print('OK')"
```

**Step 6:** Commit:
```bash
git add dashboard/utils/export.py dashboard/utils/training_monitor.py dashboard/components/export_button.py dashboard/components/training_monitor_ui.py dashboard/components/sidebar/export_section.py dashboard/training/pages/2_Train_Models.py
git commit -m "refactor: split export.py and training_monitor.py into pure utils + UI components"
```

---

### Task 3: Delete state.py, extract business logic from pages

Extract pure Python business logic from Streamlit pages into `dashboard/utils/`.

**Files:**
- Delete: `dashboard/utils/state.py`
- Modify: `dashboard/utils/statistics.py` — add `nash_sutcliffe_efficiency()`, `kling_gupta_efficiency()`
- Modify: `dashboard/utils/preprocessing.py` — add `detect_date_column()`, `detect_station_column()`, `detect_columns_from_config()`, `build_complete_dataframe()`, `denormalize_data()`, `merge_covariates_with_splits()`
- Modify: `dashboard/utils/model_registry.py` — add `load_model_with_data()`
- Modify: `dashboard/training/pages/3_Forecasting.py` — remove extracted functions, import from utils
- Modify: `dashboard/training/pages/4_Counterfactual_Analysis.py` — remove extracted functions, import from utils

**Step 1:** Delete `dashboard/utils/state.py`. Verify nothing imports it:
```bash
grep -rn "from.*state import\|import.*state" dashboard/ --include="*.py" | grep -v __pycache__
```
If any page imports it, replace with direct `st.session_state` usage.

**Step 2:** Extract from `3_Forecasting.py` to `dashboard/utils/statistics.py`:

```python
def nash_sutcliffe_efficiency(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency (NSE). Perfect = 1.0."""
    mean_obs = np.mean(actual)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - mean_obs) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)

def kling_gupta_efficiency(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Kling-Gupta Efficiency (KGE). Perfect = 1.0."""
    r = np.corrcoef(actual, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(actual) if np.std(actual) > 0 else float('nan')
    beta = np.mean(predicted) / np.mean(actual) if np.mean(actual) != 0 else float('nan')
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
```

**Step 3:** Extract from `4_Counterfactual_Analysis.py` to `dashboard/utils/preprocessing.py`:

```python
def detect_columns_from_config(model_config: dict, data_dict: dict) -> tuple[str, list[str]]:
    """Detect target column and covariate list from model config or data."""
    # Copy logic from 4_Counterfactual_Analysis.py lines 157-192

def build_complete_dataframe(data_dict: dict, target_col: str, covariate_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate train/val/test splits with covariates into single DataFrame."""
    # Copy logic from 4_Counterfactual_Analysis.py lines 195-221

def denormalize_data(processed_df: pd.DataFrame, scalers: dict, target_col: str) -> pd.DataFrame:
    """Inverse-transform normalized data to physical units."""
    # Copy logic from 3_Forecasting.py lines 520-534

def merge_covariates_with_splits(data_dict: dict) -> dict:
    """Merge covariate DataFrames with base train/val/test splits."""
    # Copy logic from 3_Forecasting.py lines 486-495
```

**Step 4:** Extract from `3_Forecasting.py` to `dashboard/utils/model_registry.py`:

```python
def load_model_with_data(model_entry: dict) -> dict:
    """Load model, scalers, config, and all data splits from registry.
    Returns dict with keys: model, scalers, config, data_dict, covariates.
    """
    # Copy logic from 3_Forecasting.py lines 181-234
    # Remove all st.session_state references, return dict instead
```

**Step 5:** Update imports in page files to use the newly extracted functions.

**Step 6:** Verify:
```bash
python -c "from dashboard.utils.statistics import nash_sutcliffe_efficiency, kling_gupta_efficiency; print('OK')"
python -c "from dashboard.utils.preprocessing import detect_columns_from_config, build_complete_dataframe; print('OK')"
python -c "from dashboard.utils.model_registry import load_model_with_data; print('OK')"
```

**Step 7:** Commit:
```bash
git add -A
git commit -m "refactor: extract business logic from Streamlit pages to utils, delete state.py"
```

---

## Phase 1: Backend API

### Task 4: FastAPI app skeleton

Create the backend foundation following junondashboard patterns exactly. Reference: `/home/ringuet/junondashboard/backend/app/`.

**Files:**
- Create: `api/__init__.py`
- Create: `api/main.py` — FastAPI app with lifespan, CORS, exception handler
- Create: `api/config.py` — Pydantic Settings from .env
- Create: `api/database.py` — async SQLAlchemy engine + session
- Create: `api/cache.py` — Redis cache-aside pattern
- Create: `api/json_response.py` — orjson FastJSONResponse
- Create: `api/task_manager.py` — TaskManager for long-running tasks (training, CF)
- Create: `api/serializers.py` — torch/numpy/plotly serialization helpers

**Step 1:** Create `api/config.py`:

```python
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Internal Junon DB
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "junon_db"
    db_user: str = "junon"
    db_password: str = ""

    # BRGM data warehouse (gold schema)
    brgm_db_host: str = "brgm-postgres"
    brgm_db_port: int = 5432
    brgm_db_name: str = "postgres"
    brgm_db_user: str = "postgres"
    brgm_db_password: str = ""

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: list[str] = ["http://localhost:49509"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = False

    # Paths
    data_dir: str = "/app/data"
    checkpoints_dir: str = "/app/checkpoints"
    results_dir: str = "/app/results"

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def brgm_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.brgm_db_user}:{self.brgm_db_password}@{self.brgm_db_host}:{self.brgm_db_port}/{self.brgm_db_name}"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
```

**Step 2:** Create `api/database.py` (copy pattern from junondashboard):

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from api.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

brgm_engine = create_async_engine(
    settings.brgm_database_url,
    pool_size=3,
    max_overflow=5,
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
brgm_session = async_sessionmaker(brgm_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with async_session() as session:
        yield session


async def get_brgm_db():
    async with brgm_session() as session:
        yield session
```

**Step 3:** Create `api/cache.py` (copy pattern from junondashboard):

```python
import hashlib
import json
import logging

import redis.asyncio as redis
from starlette.responses import Response

from api.config import settings
from api.json_response import FastJSONResponse

logger = logging.getLogger(__name__)

pool: redis.ConnectionPool | None = None
_client: redis.Redis | None = None

try:
    pool = redis.ConnectionPool.from_url(
        settings.redis_url,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=10,
    )
    _client = redis.Redis(connection_pool=pool)
except Exception:
    logger.warning("Redis not configured, caching disabled")


def get_redis() -> redis.Redis | None:
    return _client


def cache_key(prefix: str, params: dict) -> str:
    normalized = {k: sorted(str(x) for x in v) if isinstance(v, list) else v for k, v in params.items()}
    raw = json.dumps(normalized, sort_keys=True, default=str)
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"junon:{prefix}:{h}"


async def cached_response(prefix: str, params: dict, ttl: int, fetch_fn) -> Response:
    r = get_redis()
    key = cache_key(prefix, params)
    if r is not None:
        try:
            cached_val = await r.get(key)
            if cached_val:
                return Response(content=cached_val, media_type="application/json")
        except Exception as e:
            logger.debug("Redis GET error: %s", e)

    result = await fetch_fn()
    resp = FastJSONResponse(result)
    body = resp.body

    if r is not None:
        try:
            await r.setex(key, ttl, body)
        except Exception as e:
            logger.debug("Redis SET error: %s", e)

    return resp
```

**Step 4:** Create `api/json_response.py`:

```python
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import orjson
from starlette.responses import Response


def _default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")


class FastJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, default=_default)
```

**Step 5:** Create `api/task_manager.py`:

```python
import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    config: dict = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    metrics_file: str | None = None
    thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


class TaskManager:
    """Manages long-running tasks (training, counterfactual generation)."""

    def __init__(self, max_tasks: int = 50):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._max_tasks = max_tasks

    def create(self, task_type: str, config: dict) -> str:
        task_id = uuid.uuid4().hex[:12]
        task = TaskInfo(task_id=task_id, task_type=task_type, config=config)
        with self._lock:
            self._cleanup_old()
            self._tasks[task_id] = task
        return task_id

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.stop_event.set()
        task.update(status=TaskStatus.CANCELLED)
        return True

    def list_tasks(self, task_type: str | None = None) -> list[dict]:
        with self._lock:
            tasks = self._tasks.values()
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [
                {"task_id": t.task_id, "task_type": t.task_type, "status": t.status.value, "error": t.error}
                for t in tasks
            ]

    def _cleanup_old(self):
        if len(self._tasks) >= self._max_tasks:
            completed = [
                tid for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for tid in completed[:len(completed) // 2]:
                del self._tasks[tid]


task_manager = TaskManager()
```

**Step 6:** Create `api/serializers.py`:

```python
"""Serialization helpers for ML objects → JSON-safe dicts."""
import numpy as np


def serialize_tensor(tensor) -> list:
    """Convert torch.Tensor or numpy array to list."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy().tolist()
    if isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return tensor


def serialize_timeseries(ts) -> list[dict]:
    """Convert Darts TimeSeries to list of records."""
    df = ts.pd_dataframe()
    df.index.name = "date"
    return df.reset_index().to_dict("records")


def serialize_figure(fig) -> dict:
    """Convert Plotly figure to JSON-serializable dict."""
    import json
    return json.loads(fig.to_json())


def clean_nans(d: dict) -> dict:
    """Replace NaN/inf values with None recursively."""
    result = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            result[k] = None
        elif isinstance(v, dict):
            result[k] = clean_nans(v)
        elif isinstance(v, list):
            result[k] = [clean_nans(x) if isinstance(x, dict) else (None if isinstance(x, float) and np.isnan(x) else x) for x in v]
        else:
            result[k] = v
    return result
```

**Step 7:** Create `api/main.py`:

```python
import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from api.cache import get_redis, pool as redis_pool
from api.config import settings
from api.database import engine, brgm_engine, get_db
from api.json_response import FastJSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level))
    logger = logging.getLogger(__name__)

    r = get_redis()
    if r is not None:
        try:
            await r.ping()
            logger.info("Redis connection OK")
        except Exception as e:
            logger.warning("Redis ping failed: %s", e)

    yield

    if redis_pool is not None:
        await redis_pool.aclose()
    await engine.dispose()
    await brgm_engine.dispose()


app = FastAPI(
    title="Junon Time-Series Explorer API",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=FastJSONResponse,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

# Routers will be added here as they are built:
# from api.routers import datasets, training, models, forecasting, explainability, counterfactual
# app.include_router(datasets.router)
# ...


@app.get("/api/v1/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "unavailable"

    r = get_redis()
    redis_status = "disabled"
    if r is not None:
        try:
            await r.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "unavailable"

    gpu_info = _get_gpu_info()

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "db": db_status,
        "redis": redis_status,
        "gpu": gpu_info,
    }


def _get_gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
            }
    except ImportError:
        pass
    return {"available": False}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).error("Unhandled error: %s", traceback.format_exc())
    return FastJSONResponse({"detail": "Internal server error"}, status_code=500)
```

**Step 8:** Verify:
```bash
cd /home/ringuet/time-serie-explo
python -c "from api.main import app; print('FastAPI app created:', app.title)"
```

**Step 9:** Commit:
```bash
git add api/
git commit -m "feat: FastAPI app skeleton (main, config, database, cache, task_manager)"
```

---

### Task 5: Datasets router

Wraps `dashboard/utils/postgres_connector.py`, `dashboard/utils/dataset_registry.py`, and `dashboard/utils/preprocessing.py`.

**Files:**
- Create: `api/routers/__init__.py`
- Create: `api/routers/datasets.py`
- Create: `api/schemas/__init__.py`
- Create: `api/schemas/datasets.py`
- Modify: `api/main.py` — include datasets router

**Step 1:** Create `api/schemas/datasets.py`:

```python
from pydantic import BaseModel
from datetime import datetime


class DatasetSummary(BaseModel):
    id: str
    name: str
    source: str  # "csv" or "db"
    stations: list[str]
    target_variable: str
    covariates: list[str]
    date_range: list[str]  # [start, end]
    n_rows: int
    created_at: str


class DatasetDetail(DatasetSummary):
    preprocessing: dict
    columns: list[str]
    missing_pct: dict[str, float]


class DatasetPreview(BaseModel):
    columns: list[str]
    data: list[dict]
    n_total: int


class DatasetProfile(BaseModel):
    descriptive_stats: dict
    correlations: dict
    missing_values: dict
    temporal_stats: dict


class ImportDBRequest(BaseModel):
    table: str = "hubeau_daily_chroniques"
    schema: str = "gold"
    station_codes: list[str]
    date_start: str | None = None
    date_end: str | None = None
    variables: list[str] | None = None


class DatasetCreateRequest(BaseModel):
    name: str
    target_variable: str
    covariates: list[str]
    date_column: str
    station_column: str | None = None
    preprocessing: dict = {}
```

**Step 2:** Create `api/routers/datasets.py`:

```python
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_brgm_db
from api.schemas.datasets import (
    DatasetSummary, DatasetDetail, DatasetPreview, DatasetProfile,
    ImportDBRequest, DatasetCreateRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("/", response_model=list[DatasetSummary])
async def list_datasets():
    """List all prepared datasets."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    datasets = registry.list_datasets()
    return [_to_summary(d) for d in datasets]


@router.post("/", response_model=DatasetSummary)
async def create_dataset_from_upload(
    file: UploadFile = File(...),
    config: str = Query(..., description="JSON config string"),
):
    """Upload CSV and create a prepared dataset."""
    import json
    import pandas as pd
    from io import BytesIO
    from dashboard.utils.dataset_registry import DatasetRegistry

    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    cfg = json.loads(config)

    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    dataset_id = registry.save_dataset(df, cfg)
    info = registry.get_dataset(dataset_id)
    return _to_summary(info)


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get dataset details."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    info = registry.get_dataset(dataset_id)
    if info is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return _to_detail(info)


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a prepared dataset."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    success = registry.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return {"status": "deleted", "id": dataset_id}


@router.post("/import-db", response_model=DatasetSummary)
async def import_from_database(req: ImportDBRequest, db: AsyncSession = Depends(get_brgm_db)):
    """Import data from BRGM PostgreSQL (gold schema)."""
    from dashboard.utils.postgres_connector import fetch_data
    from dashboard.utils.dataset_registry import DatasetRegistry
    import pandas as pd

    # Build query with parameterized station codes
    conditions = ["code_bss = ANY(:stations)"]
    params = {"stations": req.station_codes}
    if req.date_start:
        conditions.append("date >= :date_start")
        params["date_start"] = req.date_start
    if req.date_end:
        conditions.append("date <= :date_end")
        params["date_end"] = req.date_end

    where = " AND ".join(conditions)
    cols = "*"
    if req.variables:
        safe_cols = ["date", "code_bss"] + [v for v in req.variables if v.isalnum() or "_" in v]
        cols = ", ".join(safe_cols)

    query = f"SELECT {cols} FROM {req.schema}.{req.table} WHERE {where} ORDER BY code_bss, date"
    result = await db.execute(text(query), params)
    rows = [dict(r) for r in result.mappings().all()]

    if not rows:
        raise HTTPException(404, "No data found for specified stations/dates")

    df = pd.DataFrame(rows)
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    config = {"source": "db", "table": req.table, "stations": req.station_codes}
    dataset_id = registry.save_dataset(df, config)
    info = registry.get_dataset(dataset_id)
    return _to_summary(info)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(dataset_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Preview first N rows of a dataset."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    info = registry.get_dataset(dataset_id)
    if info is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    df = registry.load_data(dataset_id)
    return {
        "columns": list(df.columns),
        "data": df.head(limit).to_dict("records"),
        "n_total": len(df),
    }


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def profile_dataset(dataset_id: str):
    """Statistical profiling of a dataset."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    from dashboard.utils.data_loader import get_data_summary
    registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
    df = registry.load_data(dataset_id)
    if df is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    summary = get_data_summary(df)
    return summary


def _to_summary(info: dict) -> dict:
    return {
        "id": info.get("id", ""),
        "name": info.get("name", ""),
        "source": info.get("source", "csv"),
        "stations": info.get("stations", []),
        "target_variable": info.get("target_variable", ""),
        "covariates": info.get("covariates", []),
        "date_range": info.get("date_range", []),
        "n_rows": info.get("n_rows", 0),
        "created_at": info.get("created_at", ""),
    }


def _to_detail(info: dict) -> dict:
    base = _to_summary(info)
    base["preprocessing"] = info.get("preprocessing", {})
    base["columns"] = info.get("columns", [])
    base["missing_pct"] = info.get("missing_pct", {})
    return base
```

**Step 3:** Add router to `api/main.py`:

```python
from api.routers import datasets
app.include_router(datasets.router)
```

**Step 4:** Verify:
```bash
python -c "from api.routers.datasets import router; print(f'{len(router.routes)} routes')"
```

**Step 5:** Commit:
```bash
git add api/routers/ api/schemas/
git commit -m "feat: datasets API router (list, create, import-db, preview, profile)"
```

---

### Task 6: Training router + SSE streaming

The most complex router. Uses TaskManager for background training, SSE for live metrics streaming.

**Files:**
- Create: `api/routers/training.py`
- Create: `api/schemas/training.py`
- Modify: `api/main.py` — include training router

**Step 1:** Create `api/schemas/training.py`:

```python
from pydantic import BaseModel


class TrainingRequest(BaseModel):
    dataset_id: str
    model_type: str  # NBEATS, TFT, TCN, LSTM, etc.
    hyperparams: dict = {}
    train_split: float = 0.7
    val_split: float = 0.15
    max_epochs: int = 100
    early_stopping_patience: int = 10
    station: str | None = None  # specific station or None for all


class TrainingStatus(BaseModel):
    task_id: str
    status: str
    epoch: int | None = None
    total_epochs: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    best_val_loss: float | None = None
    metrics: dict | None = None
    error: str | None = None


class TrainingResult(BaseModel):
    task_id: str
    status: str
    model_id: str | None = None
    metrics: dict = {}
    training_time_seconds: float | None = None
    mlflow_run_id: str | None = None
```

**Step 2:** Create `api/routers/training.py`:

```python
import asyncio
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.config import settings
from api.schemas.training import TrainingRequest, TrainingStatus, TrainingResult
from api.task_manager import task_manager, TaskStatus, TaskInfo
from api.serializers import clean_nans

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.post("/start", response_model=TrainingStatus)
async def start_training(req: TrainingRequest):
    """Start a training task in background. Returns task_id for SSE streaming."""
    # Create metrics file for callback
    metrics_dir = Path(settings.results_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    task_id = task_manager.create("training", req.model_dump())
    task = task_manager.get(task_id)

    metrics_file = str(metrics_dir / f"{task_id}.json")
    task.update(metrics_file=metrics_file)

    # Run training in background thread
    thread = threading.Thread(
        target=_run_training,
        args=(task, req, metrics_file),
        daemon=True,
    )
    task.update(thread=thread, status=TaskStatus.RUNNING)
    thread.start()

    return TrainingStatus(task_id=task_id, status="running")


@router.get("/{task_id}/stream")
async def stream_training(task_id: str):
    """SSE stream of training metrics. Reads MetricsFileCallback output."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found")

    async def event_generator():
        last_epoch = -1
        while True:
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                yield {"event": "done", "data": json.dumps({
                    "status": task.status.value,
                    "result": clean_nans(task.result) if task.result else None,
                    "error": task.error,
                })}
                break

            # Read metrics file written by MetricsFileCallback
            metrics = _read_metrics_file(task.metrics_file)
            if metrics and metrics.get("epoch", -1) > last_epoch:
                last_epoch = metrics["epoch"]
                yield {"event": "metrics", "data": json.dumps(clean_nans(metrics))}

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@router.post("/{task_id}/cancel")
async def cancel_training(task_id: str):
    """Cancel a running training task."""
    success = task_manager.cancel(task_id)
    if not success:
        raise HTTPException(404, f"Task {task_id} not found")
    return {"status": "cancelled", "task_id": task_id}


@router.get("/history", response_model=list[dict])
async def training_history():
    """List all training tasks (recent first)."""
    return task_manager.list_tasks(task_type="training")


def _run_training(task: TaskInfo, req: TrainingRequest, metrics_file: str):
    """Execute training in background thread. Writes results to task."""
    try:
        from dashboard.utils.dataset_registry import DatasetRegistry
        from dashboard.utils.preprocessing import TimeSeriesPreprocessor
        from dashboard.utils.model_factory import ModelFactory
        from dashboard.utils.training import run_training_pipeline
        from dashboard.utils.callbacks import create_training_callbacks

        # Load dataset
        registry = DatasetRegistry(base_dir=settings.data_dir + "/prepared")
        df = registry.load_data(req.dataset_id)
        config = registry.get_dataset(req.dataset_id)

        # Preprocess
        preprocessor = TimeSeriesPreprocessor(config.get("preprocessing", {}))
        train_series, val_series, test_series = preprocessor.prepare(
            df, target=req.station or config.get("target_variable"),
            train_split=req.train_split, val_split=req.val_split,
        )

        # Create model
        model = ModelFactory.create_model(
            model_type=req.model_type,
            hyperparams=req.hyperparams,
        )

        # Create callbacks with metrics file
        callbacks = create_training_callbacks(metrics_file=metrics_file)

        # Run pipeline
        start = time.time()
        result = run_training_pipeline(
            model=model,
            train_series=train_series,
            val_series=val_series,
            test_series=test_series,
            callbacks=callbacks,
            max_epochs=req.max_epochs,
            early_stopping_patience=req.early_stopping_patience,
        )
        elapsed = time.time() - start

        task.update(
            status=TaskStatus.COMPLETED,
            result={
                "metrics": result.get("metrics", {}),
                "model_path": result.get("model_path", ""),
                "mlflow_run_id": result.get("mlflow_run_id", ""),
                "training_time_seconds": round(elapsed, 1),
            },
        )

    except Exception as e:
        logger.exception("Training failed for task %s", task.task_id)
        task.update(status=TaskStatus.FAILED, error=str(e))


def _read_metrics_file(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None
```

**Step 3:** Add to `api/main.py`:
```python
from api.routers import training
app.include_router(training.router)
```

**Step 4:** Commit:
```bash
git add api/routers/training.py api/schemas/training.py api/main.py
git commit -m "feat: training API router with TaskManager + SSE streaming"
```

---

### Task 7: Models router

Wraps `dashboard/utils/model_registry.py` and `dashboard/utils/models_config.py`.

**Files:**
- Create: `api/routers/models.py`
- Create: `api/schemas/models.py`
- Modify: `api/main.py` — include models router

**Step 1:** Create `api/schemas/models.py`:

```python
from pydantic import BaseModel


class ModelSummary(BaseModel):
    id: str
    name: str
    model_type: str
    station: str | None = None
    metrics: dict = {}
    created_at: str
    mlflow_run_id: str | None = None


class ModelDetail(ModelSummary):
    hyperparams: dict = {}
    training_config: dict = {}
    input_features: list[str] = []
    target_variable: str = ""


class AvailableModel(BaseModel):
    name: str
    description: str
    category: str
    default_hyperparams: dict = {}
```

**Step 2:** Create `api/routers/models.py`:

```python
import logging
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from api.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/available", response_model=list[dict])
async def list_available_models():
    """List available model architectures."""
    from dashboard.utils.models_config import get_available_models
    return get_available_models()


@router.get("/", response_model=list[dict])
async def list_trained_models():
    """List all trained models."""
    from dashboard.utils.model_registry import ModelRegistry
    registry = ModelRegistry(base_dir=settings.checkpoints_dir)
    return registry.list_models()


@router.get("/{model_id}", response_model=dict)
async def get_model(model_id: str):
    """Get model details + metrics."""
    from dashboard.utils.model_registry import ModelRegistry
    registry = ModelRegistry(base_dir=settings.checkpoints_dir)
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return model


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model."""
    from dashboard.utils.model_registry import ModelRegistry
    registry = ModelRegistry(base_dir=settings.checkpoints_dir)
    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"status": "deleted", "id": model_id}


@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """Download model as ZIP archive."""
    from dashboard.utils.model_registry import ModelRegistry
    from dashboard.utils.export import create_model_archive

    registry = ModelRegistry(base_dir=settings.checkpoints_dir)
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(404, f"Model {model_id} not found")

    model_path = model.get("path", "")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(404, "Model files not found on disk")

    archive_bytes = create_model_archive(model_path)
    return StreamingResponse(
        BytesIO(archive_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={model_id}.zip"},
    )
```

**Step 3:** Add to `api/main.py`, commit.

---

### Task 8: Forecasting router

Wraps `dashboard/utils/forecasting.py`.

**Files:**
- Create: `api/routers/forecasting.py`
- Create: `api/schemas/forecasting.py`
- Modify: `api/main.py` — include forecasting router

**Step 1:** Create `api/schemas/forecasting.py`:

```python
from pydantic import BaseModel


class ForecastRequest(BaseModel):
    model_id: str
    horizon: int = 30
    dataset_id: str | None = None  # if None, uses model's original data


class RollingForecastRequest(ForecastRequest):
    window_size: int = 365
    step: int = 30


class ComparisonForecastRequest(BaseModel):
    model_ids: list[str]
    horizon: int = 30
    dataset_id: str | None = None


class ForecastResult(BaseModel):
    dates: list[str]
    predictions: list[float | None]
    actuals: list[float | None] = []
    metrics: dict = {}
    confidence_low: list[float | None] = []
    confidence_high: list[float | None] = []
```

**Step 2:** Create `api/routers/forecasting.py`:

```python
import logging

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas.forecasting import (
    ForecastRequest, RollingForecastRequest, ComparisonForecastRequest, ForecastResult,
)
from api.serializers import serialize_timeseries, clean_nans

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/forecasting", tags=["forecasting"])


@router.post("/single", response_model=ForecastResult)
async def single_forecast(req: ForecastRequest):
    """Generate single-window forecast."""
    from dashboard.utils.model_registry import load_model_for_inference
    from dashboard.utils.forecasting import generate_single_window_forecast

    try:
        model, data = load_model_for_inference(req.model_id, base_dir=settings.checkpoints_dir)
    except Exception as e:
        raise HTTPException(404, f"Model load failed: {e}")

    result = generate_single_window_forecast(model, data, horizon=req.horizon)
    return _format_forecast_result(result)


@router.post("/rolling", response_model=list[ForecastResult])
async def rolling_forecast(req: RollingForecastRequest):
    """Generate rolling forecast across multiple windows."""
    from dashboard.utils.model_registry import load_model_for_inference
    from dashboard.utils.forecasting import generate_rolling_forecast

    try:
        model, data = load_model_for_inference(req.model_id, base_dir=settings.checkpoints_dir)
    except Exception as e:
        raise HTTPException(404, f"Model load failed: {e}")

    results = generate_rolling_forecast(
        model, data, horizon=req.horizon,
        window_size=req.window_size, step=req.step,
    )
    return [_format_forecast_result(r) for r in results]


@router.post("/comparison", response_model=dict)
async def comparison_forecast(req: ComparisonForecastRequest):
    """Compare forecasts from multiple models."""
    from dashboard.utils.model_registry import load_model_for_inference
    from dashboard.utils.forecasting import generate_single_window_forecast

    results = {}
    for model_id in req.model_ids:
        try:
            model, data = load_model_for_inference(model_id, base_dir=settings.checkpoints_dir)
            result = generate_single_window_forecast(model, data, horizon=req.horizon)
            results[model_id] = _format_forecast_result(result)
        except Exception as e:
            results[model_id] = {"error": str(e)}

    return results


@router.post("/global", response_model=ForecastResult)
async def global_forecast(req: ForecastRequest):
    """Generate global forecast (full dataset)."""
    from dashboard.utils.model_registry import load_model_for_inference
    from dashboard.utils.forecasting import generate_global_forecast

    try:
        model, data = load_model_for_inference(req.model_id, base_dir=settings.checkpoints_dir)
    except Exception as e:
        raise HTTPException(404, f"Model load failed: {e}")

    result = generate_global_forecast(model, data, horizon=req.horizon)
    return _format_forecast_result(result)


def _format_forecast_result(result: dict) -> dict:
    return clean_nans({
        "dates": result.get("dates", []),
        "predictions": result.get("predictions", []),
        "actuals": result.get("actuals", []),
        "metrics": result.get("metrics", {}),
        "confidence_low": result.get("confidence_low", []),
        "confidence_high": result.get("confidence_high", []),
    })
```

**Step 3:** Add to `api/main.py`, commit.

---

### Task 9: Explainability router

Wraps `dashboard/utils/explainability/`.

**Files:**
- Create: `api/routers/explainability.py`
- Create: `api/schemas/explainability.py`
- Modify: `api/main.py` — include explainability router

**Step 1:** Create schemas and router following same pattern as forecasting. Endpoints:

- `POST /api/v1/explainability/feature-importance` — calls `compute_correlation_importance()`, `compute_permutation_importance()`, `compute_shap_importance()`
- `POST /api/v1/explainability/attention` — calls attention weight extraction
- `POST /api/v1/explainability/shap` — calls TimeSHAP detailed analysis
- `POST /api/v1/explainability/gradients` — calls `compute_gradient_attributions()`

Each endpoint takes `model_id` + optional params, loads model via `load_model_for_inference()`, runs computation, returns JSON-serialized results.

**Step 2:** Add to `api/main.py`, commit.

---

### Task 10: Counterfactual router

Wraps `dashboard/utils/counterfactual/`.

**Files:**
- Create: `api/routers/counterfactual.py`
- Create: `api/schemas/counterfactual.py`
- Modify: `api/main.py` — include counterfactual router

**Step 1:** Create schemas and router. Endpoints:

- `POST /api/v1/counterfactual/generate` — calls `physcf_optim.generate_counterfactual()`, returns via TaskManager (long-running)
- `POST /api/v1/counterfactual/generate-optuna` — calls `optuna_optim.generate_counterfactual_optuna()`
- `POST /api/v1/counterfactual/generate-comte` — calls `comte.generate_counterfactual_comte()`
- `GET /api/v1/counterfactual/ips-reference` — calls `ips.compute_ips_reference()`
- `POST /api/v1/counterfactual/pastas-validate` — calls `pastas_validation.run_dual_validation_for_results()`

CF generation is long-running → use TaskManager + SSE (same pattern as training). Return task_id, client polls or streams.

**Step 2:** Add to `api/main.py`, commit.

---

## Phase 2: Frontend Foundation

### Task 11: Vite + React + Tailwind scaffold

Create the frontend project following junondashboard's exact structure.

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.app.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`

**Step 1:** Initialize frontend:
```bash
cd /home/ringuet/time-serie-explo
mkdir -p frontend/src
```

**Step 2:** Create `frontend/package.json`:

```json
{
  "name": "junon-timeseries-explorer",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint src/"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.0.0",
    "@tanstack/react-query": "^5.60.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js-dist-min": "^2.35.0",
    "lucide-react": "^0.460.0",
    "clsx": "^2.1.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "@tailwindcss/vite": "^4.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.9.0",
    "vite": "^6.0.0",
    "eslint": "^9.0.0"
  }
}
```

**Step 3:** Create `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-plotly': ['plotly.js-dist-min', 'react-plotly.js'],
          'vendor-query': ['@tanstack/react-query'],
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

**Step 4:** Create `frontend/tsconfig.app.json` (copy from junondashboard pattern, add `@/*` alias).

**Step 5:** Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Junon — Time-Series Explorer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap" rel="stylesheet" />
  </head>
  <body class="bg-bg-primary text-text-primary">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**Step 6:** Create `frontend/src/index.css` (Tailwind theme, dark mode, same as junondashboard but adapted for ML domain):

```css
@import "tailwindcss";

@theme {
  --color-bg-primary: #0a0e1a;
  --color-bg-card: #111827;
  --color-bg-hover: #1f2937;
  --color-bg-input: #1e293b;
  --color-text-primary: #e5e7eb;
  --color-text-secondary: #9ca3af;
  --color-accent-cyan: #06b6d4;
  --color-accent-indigo: #6366f1;
  --color-accent-green: #10b981;
  --color-accent-amber: #f59e0b;
  --color-accent-red: #ef4444;
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
}

body { margin: 0; min-height: 100vh; }
```

**Step 7:** Create `frontend/src/main.tsx` and `frontend/src/App.tsx` (copy patterns from junondashboard — StrictMode, QueryClient, ErrorBoundary, RouterProvider).

**Step 8:** Install and verify:
```bash
cd frontend && npm install && npm run build
```

**Step 9:** Commit:
```bash
git add frontend/
git commit -m "feat: frontend scaffold (React 19, Vite, Tailwind 4, Plotly)"
```

---

### Task 12: Layout, routing, API client, types

Set up the app shell, navigation, API layer, and TypeScript types.

**Files:**
- Create: `frontend/src/routes.tsx`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/constants.ts`
- Create: `frontend/src/components/layout/Layout.tsx`
- Create: `frontend/src/components/layout/TopNav.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/hooks/useApi.ts`

**Step 1:** Create `frontend/src/lib/types.ts` — TypeScript types for all API responses:

```typescript
// Dataset types
export interface DatasetSummary {
  id: string
  name: string
  source: 'csv' | 'db'
  stations: string[]
  target_variable: string
  covariates: string[]
  date_range: [string, string]
  n_rows: number
  created_at: string
}

// Model types
export interface ModelSummary {
  id: string
  name: string
  model_type: string
  station: string | null
  metrics: Record<string, number>
  created_at: string
  mlflow_run_id: string | null
}

// Training types
export interface TrainingConfig {
  dataset_id: string
  model_type: string
  hyperparams: Record<string, any>
  max_epochs: number
  early_stopping_patience: number
}

export interface TrainingMetrics {
  epoch: number
  total_epochs: number
  train_loss: number
  val_loss: number
  best_val_loss: number
}

// Forecast types
export interface ForecastResult {
  dates: string[]
  predictions: (number | null)[]
  actuals: (number | null)[]
  metrics: Record<string, number>
  confidence_low: (number | null)[]
  confidence_high: (number | null)[]
}

// Counterfactual types
export interface CounterfactualResult {
  original: number[]
  counterfactual: number[]
  dates: string[]
  theta: Record<string, number>
  metrics: Record<string, number>
}

// Health
export interface HealthStatus {
  status: string
  db: string
  redis: string
  gpu: { available: boolean; device?: string; memory_total_gb?: number }
}
```

**Step 2:** Create `frontend/src/lib/api.ts` — fetch client (same pattern as junondashboard):

```typescript
const API_BASE = '/api/v1'

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `API error ${res.status}`)
  }
  return res.json()
}

async function postJson<T>(path: string, body: any): Promise<T> {
  return fetchJson<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

export const api = {
  health: () => fetchJson<HealthStatus>('/health'),

  datasets: {
    list: () => fetchJson<DatasetSummary[]>('/datasets/'),
    get: (id: string) => fetchJson<any>(`/datasets/${id}`),
    delete: (id: string) => fetchJson<any>(`/datasets/${id}`, { method: 'DELETE' }),
    preview: (id: string, limit = 100) => fetchJson<any>(`/datasets/${id}/preview?limit=${limit}`),
    profile: (id: string) => fetchJson<any>(`/datasets/${id}/profile`),
    importDb: (params: any) => postJson<DatasetSummary>('/datasets/import-db', params),
  },

  training: {
    start: (config: TrainingConfig) => postJson<any>('/training/start', config),
    cancel: (id: string) => postJson<any>(`/training/${id}/cancel`, {}),
    history: () => fetchJson<any[]>('/training/history'),
    stream: (id: string) => new EventSource(`${API_BASE}/training/${id}/stream`),
  },

  models: {
    list: () => fetchJson<ModelSummary[]>('/models/'),
    get: (id: string) => fetchJson<any>(`/models/${id}`),
    delete: (id: string) => fetchJson<any>(`/models/${id}`, { method: 'DELETE' }),
    available: () => fetchJson<any[]>('/models/available'),
    downloadUrl: (id: string) => `${API_BASE}/models/${id}/download`,
  },

  forecasting: {
    single: (params: any) => postJson<ForecastResult>('/forecasting/single', params),
    rolling: (params: any) => postJson<ForecastResult[]>('/forecasting/rolling', params),
    comparison: (params: any) => postJson<any>('/forecasting/comparison', params),
    global: (params: any) => postJson<ForecastResult>('/forecasting/global', params),
  },

  explainability: {
    featureImportance: (params: any) => postJson<any>('/explainability/feature-importance', params),
    attention: (params: any) => postJson<any>('/explainability/attention', params),
    shap: (params: any) => postJson<any>('/explainability/shap', params),
    gradients: (params: any) => postJson<any>('/explainability/gradients', params),
  },

  counterfactual: {
    generate: (params: any) => postJson<any>('/counterfactual/generate', params),
    generateOptuna: (params: any) => postJson<any>('/counterfactual/generate-optuna', params),
    generateComte: (params: any) => postJson<any>('/counterfactual/generate-comte', params),
    ipsReference: (stationCode: string) => fetchJson<any>(`/counterfactual/ips-reference?station=${stationCode}`),
    pastasValidate: (params: any) => postJson<any>('/counterfactual/pastas-validate', params),
  },
}
```

**Step 3:** Create `frontend/src/routes.tsx` — 5 lazy-loaded pages:

```typescript
import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import { Layout } from './components/layout/Layout'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DataPage = lazy(() => import('./pages/DataPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'))
const CounterfactualPage = lazy(() => import('./pages/CounterfactualPage'))

function Loader({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen bg-bg-primary text-text-secondary">
        <div className="animate-pulse">Chargement...</div>
      </div>
    }>{children}</Suspense>
  )
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { index: true, element: <Loader><DashboardPage /></Loader> },
      { path: 'data', element: <Loader><DataPage /></Loader> },
      { path: 'training', element: <Loader><TrainingPage /></Loader> },
      { path: 'forecasting', element: <Loader><ForecastingPage /></Loader> },
      { path: 'counterfactual', element: <Loader><CounterfactualPage /></Loader> },
    ],
  },
])
```

**Step 4:** Create `Layout.tsx` + `TopNav.tsx` — navigation with 5 items (Dashboard, Data, Training, Forecasting, Counterfactual). Use same dark theme pattern as junondashboard. Navigation uses NavLink from react-router-dom with active state styling.

**Step 5:** Create placeholder pages (each exports `default function XPage() { return <div>TODO</div> }`).

**Step 6:** Verify build:
```bash
cd frontend && npm run build
```

**Step 7:** Commit:
```bash
git add frontend/
git commit -m "feat: frontend layout, routing, API client, TypeScript types"
```

---

## Phase 3: Frontend Pages

### Task 13: Dashboard page (home)

Overview page showing system status, available datasets, trained models, GPU info.

**Files:**
- Create: `frontend/src/pages/DashboardPage.tsx`
- Create: `frontend/src/hooks/useHealth.ts`
- Create: `frontend/src/components/cards/StatusCard.tsx`
- Create: `frontend/src/components/cards/DatasetCard.tsx`
- Create: `frontend/src/components/cards/ModelCard.tsx`

**Key features:**
- Health status bar (DB, Redis, GPU)
- Grid of dataset cards (name, stations, date range, row count)
- Grid of recent model cards (type, metrics, training date)
- Quick actions (import data, start training)
- All data via TanStack Query hooks

---

### Task 14: Data page

Dataset management: import from BRGM DB or CSV, explore data, configure features.

**Files:**
- Create: `frontend/src/pages/DataPage.tsx`
- Create: `frontend/src/components/data/ImportForm.tsx`
- Create: `frontend/src/components/data/DataExplorer.tsx`
- Create: `frontend/src/components/data/DataProfiler.tsx`
- Create: `frontend/src/components/data/FeatureConfig.tsx`
- Create: `frontend/src/components/charts/TimeseriesPlot.tsx` (Plotly React wrapper)
- Create: `frontend/src/components/charts/CorrelationMatrix.tsx`
- Create: `frontend/src/hooks/useDatasets.ts`

**Key features:**
- Tab layout: Import | Explore | Configure
- Import: form for BRGM DB connection (station codes, date range, variables) or CSV upload
- Explore: data preview table, descriptive stats, time series plot, correlation matrix, missing data report
- Configure: select target variable, covariates, date column, preprocessing options
- All visualizations use `react-plotly.js`
- TanStack Query for data fetching with loading/error states

---

### Task 15: Training page

Model configuration, training launch, live monitoring via SSE, results display.

**Files:**
- Create: `frontend/src/pages/TrainingPage.tsx`
- Create: `frontend/src/components/training/ModelConfig.tsx`
- Create: `frontend/src/components/training/TrainingMonitor.tsx`
- Create: `frontend/src/components/training/TrainingResults.tsx`
- Create: `frontend/src/components/charts/LossPlot.tsx`
- Create: `frontend/src/components/charts/MetricsRadar.tsx`
- Create: `frontend/src/hooks/useTraining.ts`
- Create: `frontend/src/hooks/useSSE.ts`

**Key features:**
- Model selection (dropdown of available architectures with descriptions)
- Hyperparameter form (dynamic based on model type)
- Dataset & station selection
- Start training button → creates task, opens SSE stream
- Live monitoring panel: epoch progress bar, loss curves (Plotly), current metrics
- Cancel button (sends POST to cancel endpoint)
- Results panel: final metrics table, metrics radar chart, link to MLflow run
- Training history list

**SSE Hook (`useSSE.ts`):**

```typescript
import { useCallback, useEffect, useRef, useState } from 'react'

export function useSSE<T>(url: string | null) {
  const [data, setData] = useState<T | null>(null)
  const [status, setStatus] = useState<'idle' | 'connected' | 'done' | 'error'>('idle')
  const sourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!url) return
    const source = new EventSource(url)
    sourceRef.current = source
    setStatus('connected')

    source.addEventListener('metrics', (e) => {
      setData(JSON.parse(e.data))
    })

    source.addEventListener('done', (e) => {
      setData(JSON.parse(e.data))
      setStatus('done')
      source.close()
    })

    source.onerror = () => {
      setStatus('error')
      source.close()
    }

    return () => source.close()
  }, [url])

  return { data, status }
}
```

---

### Task 16: Forecasting page

4 forecast modes + metrics + explainability integration.

**Files:**
- Create: `frontend/src/pages/ForecastingPage.tsx`
- Create: `frontend/src/components/forecasting/ModelSelector.tsx`
- Create: `frontend/src/components/forecasting/ForecastView.tsx`
- Create: `frontend/src/components/forecasting/MetricsPanel.tsx`
- Create: `frontend/src/components/forecasting/ExplainabilityPanel.tsx`
- Create: `frontend/src/components/charts/ForecastPlot.tsx`
- Create: `frontend/src/components/charts/FeatureImportancePlot.tsx`
- Create: `frontend/src/hooks/useForecasting.ts`

**Key features:**
- Model selector (dropdown of trained models with metrics preview)
- Tab layout: Single | Rolling | Comparison | Global
- Forecast plot (Plotly): predictions vs actuals, confidence intervals, date axis
- Metrics panel: MAE, RMSE, NSE, KGE, sMAPE etc. in cards + radar chart
- Explainability panel (collapsible): feature importance bars, attention heatmap, SHAP waterfall
- All fetched via TanStack Query mutations (POST endpoints)

---

### Task 17: Counterfactual page

3 CF generation methods, IPS visualization, radar chart, Pastas validation.

**Files:**
- Create: `frontend/src/pages/CounterfactualPage.tsx`
- Create: `frontend/src/components/counterfactual/CFGenerator.tsx`
- Create: `frontend/src/components/counterfactual/IPSPanel.tsx`
- Create: `frontend/src/components/counterfactual/ThetaRadar.tsx`
- Create: `frontend/src/components/counterfactual/PastasValidation.tsx`
- Create: `frontend/src/components/charts/CFOverlayPlot.tsx`
- Create: `frontend/src/components/charts/RadarPlot.tsx`
- Create: `frontend/src/hooks/useCounterfactual.ts`

**Key features:**
- Method selection tabs: PhysCF | Optuna | CoMTE
- Configuration panel (perturbation targets, constraints, presets)
- Generation button → TaskManager + SSE for progress
- CF overlay plot: original vs counterfactual time series
- Theta radar chart: perturbation parameters visualization
- IPS panel: reference computation, monthly classification, multi-window (1/3/6/12)
- Pastas validation: dual validation results with TFT comparison
- Export button: download results as CSV/JSON

---

## Phase 4: Docker & Integration

### Task 18: Docker Compose (6 services + GPU + Nginx)

**Files:**
- Create: `docker/backend/Dockerfile`
- Create: `docker/frontend/Dockerfile`
- Create: `nginx/nginx.conf`
- Modify: `docker-compose.yml` — rewrite with 6 services
- Modify: `docker-compose.cuda.yml` — GPU overlay for backend service

**Step 1:** Create `docker/backend/Dockerfile`:

```dockerfile
ARG BACKEND=cpu

FROM python:3.12-slim AS base-cpu
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base-cuda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

FROM base-${BACKEND} AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN mkdir -p dashboard api && touch dashboard/__init__.py api/__init__.py README.md
ARG BACKEND=cpu
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BACKEND" = "cuda" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cu126 && \
        pip install --no-cache-dir ".[api,cuda]"; \
    else \
        pip install --no-cache-dir ".[api,cpu]"; \
    fi

COPY dashboard/ ./dashboard/
COPY api/ ./api/

RUN mkdir -p /app/data/prepared /app/checkpoints /app/results /app/logs

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Step 2:** Create `docker/frontend/Dockerfile`:

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Step 3:** Create `docker/frontend/nginx.conf` (SPA fallback):

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**Step 4:** Create `nginx/nginx.conf` (reverse proxy):

```nginx
upstream backend_pool {
    server backend:8000;
    keepalive 32;
}

upstream frontend_pool {
    server frontend:80;
    keepalive 16;
}

limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
limit_req_zone $binary_remote_addr zone=general:10m rate=60r/s;
limit_conn_zone $binary_remote_addr zone=addr:10m;

server {
    listen 80;
    server_name _;

    gzip on;
    gzip_vary on;
    gzip_min_length 256;
    gzip_comp_level 5;
    gzip_types text/plain text/css application/json application/javascript text/xml;

    # Security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    # API — proxy to FastAPI backend
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://backend_pool;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;  # Long for training/CF

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    # Frontend — proxy to React SPA
    location / {
        limit_req zone=general burst=30 nodelay;
        proxy_pass http://frontend_pool;
        proxy_set_header Host $host;
    }
}
```

**Step 5:** Rewrite `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: junon-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-junon}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-junon}
      POSTGRES_DB: ${POSTGRES_DB:-junon_db}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-junon}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: junon-redis
    command: redis-server --maxmemory 900mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 1G

  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.8.1
    container_name: junon-mlflow
    ports:
      - "${MLFLOW_PORT:-49511}:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///app/db/mlflow.db
      --default-artifact-root /app/mlruns
      --allowed-hosts '*'
    volumes:
      - mlflow_db:/app/db
      - mlflow_artifacts:/app/mlruns
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/')"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile
      args:
        BACKEND: ${BACKEND:-cpu}
    container_name: junon-backend
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=${POSTGRES_USER:-junon}
      - DB_PASSWORD=${POSTGRES_PASSWORD:-junon}
      - DB_NAME=${POSTGRES_DB:-junon_db}
      - BRGM_DB_HOST=brgm-postgres
      - BRGM_DB_PORT=5432
      - BRGM_DB_USER=postgres
      - BRGM_DB_PASSWORD=
      - BRGM_DB_NAME=postgres
      - REDIS_URL=redis://redis:6379/0
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DEBUG=${DEBUG:-true}
    volumes:
      - ./data/prepared:/app/data/prepared
      - ./data/piezos:/app/data/piezos:ro
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
      - ./logs:/app/logs
      - mlflow_artifacts:/app/mlruns
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      mlflow:
        condition: service_healthy
    networks:
      - default
      - brgm_net
    deploy:
      resources:
        limits:
          memory: 4G

  frontend:
    build:
      context: .
      dockerfile: docker/frontend/Dockerfile
    container_name: junon-frontend
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://127.0.0.1/"]
      interval: 15s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: junon-nginx
    ports:
      - "${APP_PORT:-49509}:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      backend:
        condition: service_healthy
      frontend:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 256M

volumes:
  postgres_data:
  mlflow_db:
  mlflow_artifacts:

networks:
  brgm_net:
    external: true
    name: hubeau_data_integration_default
```

**Step 6:** Rewrite `docker-compose.cuda.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
        limits:
          memory: 8G
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - BACKEND=cuda
```

**Step 7:** Update `pyproject.toml` — add `[project.optional-dependencies] api = [...]`:

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "asyncpg>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "redis>=5.0.0",
    "pydantic-settings>=2.0.0",
    "orjson>=3.10.0",
    "sse-starlette>=1.8.0",
    "python-multipart>=0.0.9",
]
```

**Step 8:** Test build:
```bash
BACKEND=cpu docker compose build
```

**Step 9:** Commit:
```bash
git add docker/ nginx/ docker-compose.yml docker-compose.cuda.yml pyproject.toml
git commit -m "feat: Docker Compose with 6 services (postgres, redis, mlflow, backend, frontend, nginx)"
```

---

### Task 19: Integration tests + smoke tests

**Files:**
- Create: `tests/test_api_health.py`
- Create: `tests/test_api_datasets.py`
- Create: `tests/test_api_models.py`
- Create: `tests/test_api_training.py`
- Create: `tests/conftest.py`

**Step 1:** Create `tests/conftest.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from api.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

**Step 2:** Create `tests/test_api_health.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "gpu" in data
```

**Step 3:** Create tests for each router (datasets CRUD, models list, training start/cancel).

**Step 4:** Run tests:
```bash
pytest tests/ -v
```

**Step 5:** Commit:
```bash
git add tests/
git commit -m "feat: API integration tests (health, datasets, models, training)"
```

---

## Execution Order Summary

```
Independent (parallel):
  [Task 1] + [Task 2] + [Task 3]     → Phase 0 cleanup
  [Task 11] + [Task 12]               → Frontend foundation

Sequential (after Phase 0):
  [Task 4]                             → Backend skeleton
  [Task 5] + [Task 6] + [Task 7] + [Task 8] + [Task 9] + [Task 10]  → All routers (parallel)

Sequential (after Phase 1 + 2):
  [Task 13] + [Task 14] + [Task 15] + [Task 16] + [Task 17]  → All pages (parallel)

Sequential (after all):
  [Task 18]                            → Docker
  [Task 19]                            → Tests
```

**Critical path:** Tasks 1-3 → Task 4 → Tasks 5-10 → Task 18 → Task 19
**Parallel path:** Tasks 11-12 run alongside Phase 1.
