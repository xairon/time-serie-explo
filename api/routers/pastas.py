"""Pastas TFN model API router."""
from __future__ import annotations

import csv
import io
import logging
import tempfile
from typing import Optional

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from api.config import settings
from api.schemas.pastas import (
    FitParameter,
    FitRequest,
    FitResponse,
    PastasModelSummary,
    ScenarioRequest,
    ScenarioResponse,
    StationPreview,
    TimeSeriesData,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pastas", tags=["pastas"])


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


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
# GET /preview/{code_bss}
# ---------------------------------------------------------------------------

@router.get("/preview/{code_bss}")
def preview_station(code_bss: str):
    """Return raw series + statistics for a station before fitting."""
    from typing import Any
    from dashboard.utils.pastas.station_loader import load_station_series

    try:
        station = load_station_series(code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    piezo = station.piezo
    stats: dict[str, Any] = {
        "n_obs_piezo": len(piezo),
        "n_obs_precip": len(station.precip),
        "date_range": [str(piezo.index.min()), str(piezo.index.max())],
        "piezo_mean": round(float(piezo.mean()), 3),
        "piezo_std": round(float(piezo.std()), 3),
    }

    if len(piezo) > 1:
        gaps = piezo.index.to_series().diff().dt.days.dropna()
        stats["piezo_median_gap_days"] = float(gaps.median())
        stats["piezo_max_gap_days"] = float(gaps.max())
        stats["piezo_pct_daily"] = round(float((gaps == 1).mean() * 100), 1)

    return StationPreview(
        code_bss=code_bss,
        metadata=station.metadata,
        piezo=_series_to_ts(piezo),
        precip=_series_to_ts(station.precip),
        evap=_series_to_ts(station.evap),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# POST /fit
# ---------------------------------------------------------------------------

@router.post("/fit", response_model=FitResponse)
def fit_model(req: FitRequest) -> FitResponse:
    """Fit a Pastas TFN model to the given station from the BRGM data warehouse."""
    from dashboard.utils.pastas.station_loader import load_station_series
    from dashboard.utils.pastas.builder import ValidationError
    from dashboard.utils.pastas.fit_service import run_fit

    try:
        station = load_station_series(req.code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    # Build additional stresses from CSV rows
    extra_stresses = None
    if req.additional_stresses:
        extra_stresses = []
        for s in req.additional_stresses:
            dates = pd.to_datetime([r.date for r in s.csv_rows])
            values = [r.value for r in s.csv_rows]
            series = pd.Series(values, index=dates, name=s.name)
            # Regularize to daily
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="first")]
            if len(series) > 1:
                series = series.asfreq("D")
                series = series.interpolate(method="linear", limit=7).ffill().bfill()
            extra_stresses.append({
                "type": s.type,
                "name": s.name,
                "rfunc": s.rfunc,
                "series": series,
            })

    try:
        result = run_fit(
            gwl=station.piezo,
            precip=station.precip,
            evap=station.evap,
            recharge_type=req.recharge.type,
            response_type=req.response.type,
            noise_type=req.noise.type,
            solver_type=req.solver.type,
            solver_kwargs=req.solver.kwargs or None,
            tmin=str(req.tmin) if req.tmin else None,
            tmax=str(req.tmax) if req.tmax else None,
            dataset_id=req.code_bss,
            name=req.name,
            val_split=req.val_split,
            additional_stresses=extra_stresses,
        )
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Pastas fit failed: %s", exc)
        raise HTTPException(500, f"Fit failed: {exc}") from exc

    return FitResponse(
        run_id=result.run_id,
        metrics=result.metrics,
        parameters=[FitParameter(**p) for p in result.parameters],
        observed=_series_to_ts(result.observed),
        simulated=_series_to_ts(result.simulated),
        residuals=_series_to_ts(result.residuals),
        contributions={k: _series_to_ts(v) for k, v in result.contributions.items()},
        step_response=_series_to_ts(result.step_response),
        block_response=_series_to_ts(result.block_response),
        acf=result.acf_stats,
        warnings=result.warnings,
        pastas_version=result.pastas_version,
        validation_metrics=result.validation_metrics,
        cal_period=result.cal_period,
        val_period=result.val_period,
    )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=list[PastasModelSummary])
def list_models(code_bss: Optional[str] = None) -> list[PastasModelSummary]:
    """List Pastas models stored in MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("pastas")
    if experiment is None:
        return []

    filter_str = ""
    if code_bss:
        filter_str = f"tags.station_id = '{code_bss}'"

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
                code_bss=tags.get("station_id", "unknown"),
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
        validation_metrics=None,
        cal_period=None,
        val_period=None,
    )


# ---------------------------------------------------------------------------
# GET /models/{run_id}/signatures
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/signatures")
def get_signatures(run_id: str):
    """Compute hydrological signatures (observed vs simulated) for a stored Pastas model."""
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.signatures import compute_signatures

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin = run.data.params.get("tmin")
    tmax = run.data.params.get("tmax")

    try:
        obs = model.observations(tmin=tmin, tmax=tmax)
        sim = model.simulate(tmin=tmin, tmax=tmax)
    except Exception as exc:
        raise HTTPException(500, f"Failed to compute series: {exc}") from exc

    try:
        return compute_signatures(obs, sim)
    except Exception as exc:
        logger.exception("Signatures computation failed: %s", exc)
        raise HTTPException(500, f"Signatures computation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /models/{run_id}/diagnostics
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/diagnostics")
def get_diagnostics(run_id: str):
    """Compute full diagnostic statistics on residuals of a stored Pastas model."""
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.diagnostics import compute_diagnostics

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin = run.data.params.get("tmin")
    tmax = run.data.params.get("tmax")

    residuals = model.residuals(tmin=tmin, tmax=tmax)
    return compute_diagnostics(residuals)


# ---------------------------------------------------------------------------
# GET /models/{run_id}/export/pas
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/export/pas")
def export_pas(run_id: str):
    """Export a Pastas model as a .pas file."""
    from dashboard.utils.pastas.io import load_model

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")

    f = tempfile.NamedTemporaryFile(suffix=".pas", delete=False)
    model.to_file(f.name)
    return FileResponse(f.name, filename=f"pastas_{run_id[:8]}.pas", media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# GET /models/{run_id}/export/csv
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/export/csv")
def export_csv(run_id: str):
    """Export model params, metrics and tags as a CSV file."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        raise HTTPException(404, f"Run '{run_id}' not found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["category", "key", "value"])
    for k, v in run.data.params.items():
        writer.writerow(["param", k, v])
    for k, v in run.data.metrics.items():
        writer.writerow(["metric", k, v])
    for k, v in run.data.tags.items():
        if not k.startswith("mlflow."):
            writer.writerow(["tag", k, v])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=pastas_{run_id[:8]}.csv"},
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
