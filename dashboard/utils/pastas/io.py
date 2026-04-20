"""Load Pastas models from MLflow artifacts with LRU caching."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import mlflow
import pastas as ps

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("/tmp/pastas_models")


class ModelVersionMismatch(Exception):
    """Raised when stored Pastas version differs from current."""


@lru_cache(maxsize=32)
def _load_cached(run_id: str) -> ps.Model:
    """Download and parse .pas file from MLflow (cached)."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    stored_version = run.data.tags.get("pastas_version", "unknown")
    if stored_version != "unknown" and stored_version != ps.__version__:
        logger.warning(
            "Model %s was saved with Pastas %s, current is %s",
            run_id, stored_version, ps.__version__,
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _CACHE_DIR / f"{run_id}.pas"

    if not local_path.exists():
        artifacts = client.list_artifacts(run_id)
        pas_artifact = next(
            (a.path for a in artifacts if a.path.endswith(".pas")), None,
        )
        if pas_artifact is None:
            raise FileNotFoundError(f"No .pas artifact in run {run_id}")

        local_dir = client.download_artifacts(run_id, pas_artifact, str(_CACHE_DIR))
        downloaded = Path(local_dir)
        if downloaded != local_path:
            downloaded.rename(local_path)

    return ps.io.load(str(local_path))


def load_model(run_id: str) -> ps.Model:
    """Load a Pastas model from MLflow."""
    return _load_cached(run_id)


def evict_cache(run_id: str) -> None:
    """Remove a model from the LRU cache and disk."""
    _load_cached.cache_clear()
    local_path = _CACHE_DIR / f"{run_id}.pas"
    if local_path.exists():
        local_path.unlink()
