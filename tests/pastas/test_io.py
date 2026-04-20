"""Tests for Pastas model I/O."""
from __future__ import annotations

import pytest
import numpy as np

from dashboard.utils.pastas.fit_service import run_fit
from dashboard.utils.pastas.io import load_model, evict_cache, ModelVersionMismatch


def _fit_test_model(synthetic_station, tmp_path, monkeypatch) -> str:
    """Helper: fit and return run_id."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs={},
        tmin=None, tmax=None,
        dataset_id="test", name="test_io",
    )
    return result.run_id


@pytest.fixture(autouse=True)
def clear_io_cache():
    """Clear the LRU cache before each test to avoid cross-test contamination."""
    from dashboard.utils.pastas.io import _load_cached
    _load_cached.cache_clear()
    yield
    _load_cached.cache_clear()


def test_load_model_roundtrip(synthetic_station, tmp_path, monkeypatch):
    """Load a model from MLflow and simulate — should produce non-NaN values."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    model = load_model(run_id)

    sim = model.simulate()
    assert len(sim) > 0
    assert not np.all(np.isnan(sim.values))


def test_load_model_caching(synthetic_station, tmp_path, monkeypatch):
    """Second load returns the same object (cache hit)."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    m1 = load_model(run_id)
    m2 = load_model(run_id)
    assert m1 is m2


def test_evict_cache(synthetic_station, tmp_path, monkeypatch):
    """After eviction, next load returns a different object."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    m1 = load_model(run_id)
    evict_cache(run_id)
    m2 = load_model(run_id)
    assert m1 is not m2


def test_load_model_not_found(tmp_path, monkeypatch):
    """Loading a nonexistent run_id raises."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with pytest.raises(Exception):
        load_model("nonexistent_run_id_12345")
