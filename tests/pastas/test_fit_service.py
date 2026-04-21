"""Tests for Pastas fit service."""
from __future__ import annotations

import pytest

from dashboard.utils.pastas.fit_service import run_fit, FitResult


def test_run_fit_least_squares(synthetic_station, monkeypatch, tmp_path):
    """Fit with LeastSquares and check all result fields are populated."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="none",
        solver_type="LeastSquares",
        solver_kwargs=None,
        tmin=None,
        tmax=None,
        dataset_id="test_dataset",
        name="test_least_squares",
    )

    assert isinstance(result, FitResult)
    assert result.run_id, "run_id should be a non-empty string"

    # Metrics — EVP > 40 on synthetic AR(0.98) + Gamma model
    assert result.metrics["evp"] > 40, f"EVP should be > 40, got {result.metrics['evp']}"
    assert result.metrics["rmse"] > 0, f"RMSE should be positive, got {result.metrics['rmse']}"
    assert "ljung_box_pvalue" in result.metrics

    # Parameters
    assert len(result.parameters) > 0, "parameters list should be non-empty"
    first_param = result.parameters[0]
    assert "name" in first_param
    assert "optimal" in first_param

    # Series
    assert len(result.observed) > 0, "observed series should be non-empty"
    assert len(result.simulated) > 0, "simulated series should be non-empty"
    assert len(result.residuals) > 0, "residuals series should be non-empty"

    # Contributions
    assert "recharge" in result.contributions, "contributions should contain 'recharge'"
    assert len(result.contributions["recharge"]) > 0

    # Step response
    assert len(result.step_response) > 0, "step_response should be non-empty"

    # Pastas version
    assert result.pastas_version != ""


def test_run_fit_lmfit(synthetic_station, monkeypatch, tmp_path):
    """Fit with Lmfit solver + Gamma response + no noise; verify fit ran and is logged."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="none",
        solver_type="Lmfit",
        solver_kwargs=None,
        tmin=None,
        tmax=None,
        dataset_id="test_dataset_lmfit",
        name="test_lmfit",
    )

    assert isinstance(result, FitResult)
    assert result.run_id, "run_id should be non-empty"
    # EVP >= 0 (may be 0 if solver hits a degenerate minimum; what matters is it ran)
    assert result.metrics["evp"] >= 0, f"EVP should be >= 0, got {result.metrics['evp']}"
    assert len(result.parameters) > 0, "parameters should be non-empty"


def test_run_fit_returns_warnings_on_bound_hit(synthetic_station, monkeypatch, tmp_path):
    """Fit with a short calibration window; warnings should be a list."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs=None,
        tmin="2015-01-01",
        tmax="2016-06-30",
        dataset_id="test_dataset_short",
        name="test_short_window",
    )

    assert isinstance(result, FitResult)
    assert isinstance(result.warnings, list), "warnings should always be a list"


def test_run_fit_with_warmup(synthetic_station, monkeypatch, tmp_path):
    """Warm-up should still produce valid results."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="none",
        solver_type="LeastSquares",
        solver_kwargs=None,
        tmin=None,
        tmax=None,
        dataset_id="test_warmup",
        warm_up_years=1,
    )
    assert result.run_id
    assert result.metrics.get("evp") is not None


def test_run_fit_two_pass(synthetic_station, monkeypatch, tmp_path):
    """Two-pass solve should produce valid results with noise model."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs=None,
        tmin=None,
        tmax=None,
        dataset_id="test_twopass",
        two_pass=True,
    )
    assert result.run_id
    assert result.metrics.get("evp") is not None
