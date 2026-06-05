"""Tests for Pastas scenario engine."""
from __future__ import annotations

import pytest
import numpy as np

from dashboard.utils.pastas.fit_service import run_fit
from dashboard.utils.pastas.io import evict_cache
from dashboard.utils.pastas.scenario import simulate_scenario, ScenarioResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_helper(synthetic_station, tmp_path, monkeypatch) -> str:
    """Fit a baseline model and return its MLflow run_id."""
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
        tmin=None,
        tmax=None,
        dataset_id="test",
        name="scenario_test",
    )
    return result.run_id


@pytest.fixture(autouse=True)
def clear_io_cache():
    """Clear the LRU cache in io module between tests to avoid stale state."""
    from dashboard.utils.pastas import io
    io._load_cached.cache_clear()
    yield
    io._load_cached.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scenario_no_modifications(synthetic_station, tmp_path, monkeypatch):
    """With no modifications, scenario equals baseline and delta is approximately 0."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=[])

    assert isinstance(result, ScenarioResult)
    assert len(result.baseline) > 0
    assert len(result.scenario) > 0
    assert len(result.delta) > 0

    # scenario == baseline when no mods applied
    np.testing.assert_allclose(
        result.scenario.values,
        result.baseline.values,
        atol=1e-6,
        err_msg="Scenario should equal baseline with no modifications",
    )
    np.testing.assert_allclose(
        result.delta.values,
        0.0,
        atol=1e-6,
        err_msg="Delta should be ~0 with no modifications",
    )


def test_scenario_pumping_synthetic_zero_rate(synthetic_station, tmp_path, monkeypatch):
    """A pumping stress with rate=0 should produce a scenario approximately equal to baseline."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    mods = [
        {
            "type": "pumping_synthetic",
            "name": "pump_zero",
            "start": "2016-01-01",
            "end": "2017-12-31",
            "pattern": "constant",
            "rate_m3d": 0.0,
            "distance_m": 500.0,
        }
    ]
    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=mods)

    assert isinstance(result, ScenarioResult)
    # With Q=0, no stress is added, scenario ≈ baseline
    common = result.baseline.index.intersection(result.scenario.index)
    np.testing.assert_allclose(
        result.scenario.loc[common].values,
        result.baseline.loc[common].values,
        atol=0.01,
        err_msg="Zero-rate pumping should not change GWL",
    )


def test_scenario_pumping_lowers_gwl(synthetic_station, tmp_path, monkeypatch):
    """A significant pumping rate should lower GWL (mean delta < 0)."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    mods = [
        {
            "type": "pumping_synthetic",
            "name": "pump_large",
            "start": "2015-01-01",
            "end": "2019-12-31",
            "pattern": "constant",
            "rate_m3d": 1000.0,
            "distance_m": 500.0,
        }
    ]
    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=mods)

    assert isinstance(result, ScenarioResult)
    assert result.delta.mean() < 0, (
        f"High pumping rate should lower GWL, but delta.mean() = {result.delta.mean():.4f}"
    )


def test_scenario_linear_trend(synthetic_station, tmp_path, monkeypatch):
    """A positive linear trend should raise GWL (mean delta > 0)."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    mods = [
        {
            "type": "linear_trend",
            "start": "2015-01-01",
            "end": "2019-12-31",
            "name": "upward_trend",
            "slope": 0.5,  # m/year
        }
    ]
    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=mods)

    assert isinstance(result, ScenarioResult)
    assert result.delta.mean() > 0, (
        f"Positive linear trend should raise GWL, but delta.mean() = {result.delta.mean():.4f}"
    )


def test_scenario_scale_stress_noop(synthetic_station, tmp_path, monkeypatch):
    """Scaling precip by factor=1.0 should leave scenario equal to baseline."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    mods = [
        {
            "type": "scale_stress",
            "stress_index": 0,  # precip
            "factor": 1.0,
        }
    ]
    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=mods)

    assert isinstance(result, ScenarioResult)
    common = result.baseline.index.intersection(result.scenario.index)
    np.testing.assert_allclose(
        result.scenario.loc[common].values,
        result.baseline.loc[common].values,
        atol=1e-4,
        err_msg="factor=1.0 should be a no-op",
    )


def test_scenario_scale_precip_decrease(synthetic_station, tmp_path, monkeypatch):
    """Scaling precip by 0.5 (less rain) should lower GWL (mean delta < 0)."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)

    mods = [
        {
            "type": "scale_stress",
            "stress_index": 0,  # precip
            "factor": 0.5,
        }
    ]
    result = simulate_scenario(run_id=run_id, tmin=None, tmax=None, modifications=mods)

    assert isinstance(result, ScenarioResult)
    assert result.delta.mean() < 0, (
        f"Halving precip should lower GWL, but delta.mean() = {result.delta.mean():.4f}"
    )
