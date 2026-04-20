"""Fit a Pastas TFN model and persist results to MLflow."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import pastas as ps
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from dashboard.utils.pastas.builder import build_model
from dashboard.utils.pastas.config import SOLVER_REGISTRY


@dataclass
class FitResult:
    run_id: str
    metrics: dict[str, float]
    parameters: list[dict[str, Any]]
    observed: pd.Series
    simulated: pd.Series
    residuals: pd.Series
    contributions: dict[str, pd.Series]
    step_response: pd.Series
    block_response: pd.Series
    acf_stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    pastas_version: str = ""


def _series_hash(gwl: pd.Series, precip: pd.Series, evap: pd.Series) -> str:
    """SHA-256 over concatenated index+values of the three series."""
    parts = []
    for s in (gwl, precip, evap):
        parts.append(s.index.astype(str).str.cat())
        parts.append(np.array2string(s.values, separator=","))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _extract_metrics(model: ps.Model, tmin: Optional[str], tmax: Optional[str]) -> dict[str, float]:
    """Extract EVP, RMSE, AIC, BIC, Ljung-Box p-value, and n_obs from a solved model."""
    metrics: dict[str, float] = {}

    try:
        metrics["evp"] = float(model.stats.evp(tmin=tmin, tmax=tmax))
    except Exception:
        metrics["evp"] = float("nan")

    try:
        metrics["rmse"] = float(model.stats.rmse(tmin=tmin, tmax=tmax))
    except Exception:
        metrics["rmse"] = float("nan")

    try:
        metrics["aic"] = float(model.stats.aic())
    except Exception:
        metrics["aic"] = float("nan")

    try:
        metrics["bic"] = float(model.stats.bic())
    except Exception:
        metrics["bic"] = float("nan")

    for stat_name in ("nse", "kge", "rsq", "mae"):
        try:
            metrics[stat_name] = float(getattr(model.stats, stat_name)(tmin=tmin, tmax=tmax))
        except Exception:
            pass

    # Ljung-Box on residuals
    try:
        residuals = model.residuals(tmin=tmin, tmax=tmax).dropna()
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        metrics["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        metrics["ljung_box_pvalue"] = float("nan")

    try:
        metrics["n_obs"] = float(len(model.observations(tmin=tmin, tmax=tmax).dropna()))
    except Exception:
        metrics["n_obs"] = float("nan")

    return metrics


def _extract_parameters(model: ps.Model) -> list[dict[str, Any]]:
    """Extract parameters from the model's parameters DataFrame."""
    params = []
    df = model.parameters
    for name, row in df.iterrows():
        param = {
            "name": name,
            "optimal": float(row.get("optimal", float("nan"))),
            "stderr": float(row.get("stderr", float("nan"))) if pd.notna(row.get("stderr")) else None,
            "initial": float(row.get("initial", float("nan"))),
            "pmin": float(row.get("pmin", float("nan"))) if pd.notna(row.get("pmin")) else None,
            "pmax": float(row.get("pmax", float("nan"))) if pd.notna(row.get("pmax")) else None,
            "vary": bool(row.get("vary", True)),
        }
        params.append(param)
    return params


def _check_warnings(model: ps.Model) -> list[str]:
    """Check if any parameter optimal value hit its bounds (pmin or pmax)."""
    warnings: list[str] = []
    df = model.parameters
    for name, row in df.iterrows():
        optimal = row.get("optimal")
        pmin = row.get("pmin")
        pmax = row.get("pmax")

        if optimal is None or not pd.notna(optimal):
            continue

        if pmin is not None and pd.notna(pmin) and abs(float(optimal) - float(pmin)) < 1e-10:
            warnings.append(f"Parameter '{name}' hit lower bound (pmin={pmin}).")

        if pmax is not None and pd.notna(pmax) and abs(float(optimal) - float(pmax)) < 1e-10:
            warnings.append(f"Parameter '{name}' hit upper bound (pmax={pmax}).")

    return warnings


def _acf_stats(residuals: pd.Series) -> dict[str, Any]:
    """Compute ACF and PACF stats from residuals using statsmodels."""
    clean = residuals.dropna()
    result: dict[str, Any] = {}

    try:
        nlags = min(40, len(clean) // 2 - 1)
        acf_values, acf_confint = acf(clean, nlags=nlags, alpha=0.05, fft=True)
        result["acf"] = acf_values.tolist()
        result["acf_confint"] = acf_confint.tolist()
    except Exception:
        result["acf"] = []
        result["acf_confint"] = []

    try:
        nlags = min(40, len(clean) // 2 - 1)
        pacf_values, pacf_confint = pacf(clean, nlags=nlags, alpha=0.05)
        result["pacf"] = pacf_values.tolist()
        result["pacf_confint"] = pacf_confint.tolist()
    except Exception:
        result["pacf"] = []
        result["pacf_confint"] = []

    try:
        lb_result = acorr_ljungbox(clean, lags=[10], return_df=True)
        result["ljung_box_stat"] = float(lb_result["lb_stat"].iloc[0])
        result["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        result["ljung_box_stat"] = float("nan")
        result["ljung_box_pvalue"] = float("nan")

    return result


def _get_tracking_uri() -> str:
    """Resolve MLflow tracking URI from env var or settings."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        from api.config import settings
        return settings.mlflow_tracking_uri
    except Exception:
        return "http://mlflow:5000"


def run_fit(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    recharge_type: str,
    response_type: str,
    noise_type: str,
    solver_type: str,
    solver_kwargs: Optional[dict[str, Any]],
    tmin: Optional[str],
    tmax: Optional[str],
    dataset_id: str,
    name: Optional[str] = None,
) -> FitResult:
    """Fit a Pastas TFN model and persist to MLflow.

    Parameters
    ----------
    gwl:
        Piezometric head time series.
    precip:
        Precipitation stress time series.
    evap:
        Evapotranspiration stress time series.
    recharge_type:
        Key in RECHARGE_REGISTRY (e.g. "Linear", "FlexModel").
    response_type:
        Key in RFUNC_REGISTRY (e.g. "Gamma", "Exponential").
    noise_type:
        Key in NOISE_REGISTRY or "none".
    solver_type:
        Key in SOLVER_REGISTRY (e.g. "LeastSquares", "Lmfit").
    solver_kwargs:
        Optional extra kwargs forwarded to the solver.
    tmin:
        ISO date string for the calibration start (or None for auto).
    tmax:
        ISO date string for the calibration end (or None for auto).
    dataset_id:
        Identifier of the dataset (logged as MLflow param and tag).
    name:
        Optional MLflow run name.

    Returns
    -------
    FitResult
        Dataclass containing all fit outputs and MLflow run_id.
    """
    solver_kwargs = solver_kwargs or {}

    # Build model
    model, tmin, tmax = build_model(
        gwl=gwl,
        precip=precip,
        evap=evap,
        recharge_type=recharge_type,
        response_type=response_type,
        noise_type=noise_type,
        tmin=tmin,
        tmax=tmax,
    )

    # Solve — Pastas >= 0.23 requires an instance, not a class
    solver_cls = SOLVER_REGISTRY[solver_type]
    solver_instance = solver_cls()
    model.solve(
        tmin=tmin,
        tmax=tmax,
        solver=solver_instance,
        report=False,
        **solver_kwargs,
    )

    # Extract all outputs before MLflow context
    metrics = _extract_metrics(model, tmin, tmax)
    parameters = _extract_parameters(model)
    fit_warnings = _check_warnings(model)

    observed = model.observations(tmin=tmin, tmax=tmax)
    simulated = model.simulate(tmin=tmin, tmax=tmax)
    residuals = model.residuals(tmin=tmin, tmax=tmax)

    # Per-stress contributions
    contributions: dict[str, pd.Series] = {}
    for sm_name in model.stressmodels:
        try:
            contributions[sm_name] = model.get_contribution(sm_name, tmin=tmin, tmax=tmax)
        except Exception:
            pass

    # Step and block response for the first stressmodel (recharge)
    step_response = pd.Series(dtype=float)
    block_response = pd.Series(dtype=float)
    if model.stressmodels:
        sm_name = next(iter(model.stressmodels))
        try:
            step_response = model.get_step_response(sm_name)
        except Exception:
            pass
        try:
            block_response = model.get_block_response(sm_name)
        except Exception:
            pass

    acf_result = _acf_stats(residuals)

    # Compute series hash for tagging
    sh = _series_hash(gwl, precip, evap)

    # Station ID: use gwl.name if available
    station_id = str(gwl.name) if gwl.name else "unknown"

    # MLflow logging
    tracking_uri = _get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pastas")

    run_name = name or f"{station_id}_{solver_type}"

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_params({
            "recharge_type": recharge_type,
            "response_type": response_type,
            "noise_type": noise_type,
            "solver_type": solver_type,
            "dataset_id": dataset_id,
            "tmin": str(tmin) if tmin is not None else "auto",
            "tmax": str(tmax) if tmax is not None else "auto",
        })

        # Metrics (skip NaN — MLflow rejects them)
        loggable_metrics = {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}
        if loggable_metrics:
            mlflow.log_metrics(loggable_metrics)

        # Tags
        mlflow.set_tags({
            "station_id": station_id,
            "pastas_version": ps.__version__,
            "series_hash": sh,
        })

        # Artifact: save model.pas
        with tempfile.TemporaryDirectory() as tmpdir:
            pas_path = os.path.join(tmpdir, "model.pas")
            model.to_file(pas_path)
            mlflow.log_artifact(pas_path)

        run_id = run.info.run_id

    return FitResult(
        run_id=run_id,
        metrics=metrics,
        parameters=parameters,
        observed=observed,
        simulated=simulated,
        residuals=residuals,
        contributions=contributions,
        step_response=step_response,
        block_response=block_response,
        acf_stats=acf_result,
        warnings=fit_warnings,
        pastas_version=ps.__version__,
    )
