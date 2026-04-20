"""Pastas TFN model API router."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas.pastas import (
    FitParameter,
    FitRequest,
    FitResponse,
    PastasModelSummary,
    ScenarioRequest,
    ScenarioResponse,
    TimeSeriesData,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pastas", tags=["pastas"])


def _series_to_ts(s: pd.Series) -> TimeSeriesData:
    return TimeSeriesData(
        index=[str(d) for d in s.index],
        values=[float(v) if pd.notna(v) else 0.0 for v in s.values],
    )


# ---------------------------------------------------------------------------
# GET /options
# ---------------------------------------------------------------------------

@router.get("/options")
def get_options() -> dict:
    """Return available Pastas component options for UI dropdowns."""
    from dashboard.utils.pastas.config import get_p1_options
    return get_p1_options()


# ---------------------------------------------------------------------------
# POST /fit
# ---------------------------------------------------------------------------

@router.post("/fit", response_model=FitResponse)
def fit_model(req: FitRequest) -> FitResponse:
    """Fit a Pastas TFN model to the given dataset."""
    from dashboard.utils.dataset_registry import DatasetRegistry
    from dashboard.utils.pastas.builder import ValidationError
    from dashboard.utils.pastas.fit_service import run_fit

    # Locate dataset
    registry = DatasetRegistry(Path(settings.data_dir) / "prepared")
    datasets = registry.scan_datasets()
    ds = next(
        (d for d in datasets if d.name == req.dataset_id or d.path.suffix == req.dataset_id or str(d.path).endswith(req.dataset_id)),
        None,
    )
    if ds is None:
        raise HTTPException(404, f"Dataset not found: {req.dataset_id}")

    df, _config = registry.load_dataset(ds)

    # Filter by station if applicable
    if req.station_id and ds.station_column and ds.station_column in df.columns:
        df = df[df[ds.station_column] == req.station_id]
        if df.empty:
            raise HTTPException(404, f"Station '{req.station_id}' not found in dataset")

    # Validate columns
    for col in (req.precip_column, req.evap_column):
        if col not in df.columns:
            raise HTTPException(422, f"Column '{col}' not found in dataset")

    target_col = ds.target_column or df.columns[0]
    if target_col not in df.columns:
        raise HTTPException(422, f"Target column '{target_col}' not found in dataset")

    gwl = df[target_col].dropna()
    gwl.name = req.station_id or ds.name
    precip = df[req.precip_column]
    evap = df[req.evap_column]

    tmin_str = str(req.tmin) if req.tmin else None
    tmax_str = str(req.tmax) if req.tmax else None

    try:
        result = run_fit(
            gwl=gwl,
            precip=precip,
            evap=evap,
            recharge_type=req.recharge.type,
            response_type=req.response.type,
            noise_type=req.noise.type,
            solver_type=req.solver.type,
            solver_kwargs=req.solver.kwargs or None,
            tmin=tmin_str,
            tmax=tmax_str,
            dataset_id=req.dataset_id,
            name=req.name,
        )
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Pastas fit failed: %s", exc)
        raise HTTPException(500, f"Fit failed: {exc}") from exc

    parameters = [FitParameter(**p) for p in result.parameters]

    return FitResponse(
        run_id=result.run_id,
        metrics=result.metrics,
        parameters=parameters,
        observed=_series_to_ts(result.observed),
        simulated=_series_to_ts(result.simulated),
        residuals=_series_to_ts(result.residuals),
        contributions={k: _series_to_ts(v) for k, v in result.contributions.items()},
        step_response=_series_to_ts(result.step_response),
        block_response=_series_to_ts(result.block_response),
        acf=result.acf_stats,
        warnings=result.warnings,
        pastas_version=result.pastas_version,
    )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=list[PastasModelSummary])
def list_models(station_id: Optional[str] = None) -> list[PastasModelSummary]:
    """List Pastas models stored in MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("pastas")
    if experiment is None:
        return []

    filter_str = ""
    if station_id:
        filter_str = f"tags.station_id = '{station_id}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["start_time DESC"],
    )

    summaries: list[PastasModelSummary] = []
    for run in runs:
        tags = run.data.tags
        params = run.data.params
        metrics = run.data.metrics
        summaries.append(
            PastasModelSummary(
                run_id=run.info.run_id,
                name=run.info.run_name or run.info.run_id,
                station_id=tags.get("station_id", "unknown"),
                recharge_type=params.get("recharge_type", "unknown"),
                response_type=params.get("response_type", "unknown"),
                evp=metrics.get("evp"),
                rmse=metrics.get("rmse"),
                created_at=str(run.info.start_time),
                pastas_version=tags.get("pastas_version", "unknown"),
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# GET /models/{run_id}
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}", response_model=FitResponse)
def get_model(run_id: str) -> FitResponse:
    """Reconstruct FitResponse for a stored Pastas model."""
    from dashboard.utils.pastas.fit_service import _extract_parameters, _acf_stats
    from dashboard.utils.pastas.io import load_model

    try:
        model = load_model(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", run_id, exc)
        raise HTTPException(500, f"Failed to load model: {exc}") from exc

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = dict(run.data.metrics)
    tags = run.data.tags

    try:
        observed = model.observations()
        simulated = model.simulate()
        residuals = model.residuals()
    except Exception as exc:
        raise HTTPException(500, f"Failed to reconstruct series: {exc}") from exc

    contributions: dict[str, pd.Series] = {}
    for sm_name in model.stressmodels:
        try:
            contributions[sm_name] = model.get_contribution(sm_name)
        except Exception:
            pass

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
    parameters = [FitParameter(**p) for p in _extract_parameters(model)]

    return FitResponse(
        run_id=run_id,
        metrics=metrics,
        parameters=parameters,
        observed=_series_to_ts(observed),
        simulated=_series_to_ts(simulated),
        residuals=_series_to_ts(residuals),
        contributions={k: _series_to_ts(v) for k, v in contributions.items()},
        step_response=_series_to_ts(step_response),
        block_response=_series_to_ts(block_response),
        acf=acf_result,
        warnings=[],
        pastas_version=tags.get("pastas_version", "unknown"),
    )


# ---------------------------------------------------------------------------
# DELETE /models/{run_id}
# ---------------------------------------------------------------------------

@router.delete("/models/{run_id}")
def delete_model(run_id: str) -> dict:
    """Delete a Pastas model from MLflow and evict cache."""
    from dashboard.utils.pastas.io import evict_cache

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        client.get_run(run_id)
    except Exception:
        raise HTTPException(404, f"Run not found: {run_id}")

    evict_cache(run_id)

    try:
        client.delete_run(run_id)
    except Exception as exc:
        raise HTTPException(500, f"Failed to delete run: {exc}") from exc

    return {"deleted": run_id}


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=ScenarioResponse)
def simulate(req: ScenarioRequest) -> ScenarioResponse:
    """Apply what-if modifications to a calibrated model and simulate."""
    from dashboard.utils.pastas.scenario import simulate_scenario

    modifications = [m.model_dump() for m in req.modifications]

    try:
        result = simulate_scenario(
            run_id=req.run_id,
            tmin=str(req.tmin),
            tmax=str(req.tmax),
            modifications=modifications,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Scenario simulation failed: %s", exc)
        raise HTTPException(500, f"Simulation failed: {exc}") from exc

    return ScenarioResponse(
        baseline=_series_to_ts(result.baseline),
        scenario=_series_to_ts(result.scenario),
        delta=_series_to_ts(result.delta),
        contributions_baseline={k: _series_to_ts(v) for k, v in result.contributions_baseline.items()},
        contributions_scenario={k: _series_to_ts(v) for k, v in result.contributions_scenario.items()},
        warnings=result.warnings,
    )
