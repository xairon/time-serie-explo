"""Pastas TFN model API router."""
from __future__ import annotations

import csv
import io
import logging
import re
import tempfile
from typing import Optional

import mlflow
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from api.auth.deps import get_current_user
from api.auth.ownership import assert_owns_model
from api.config import settings
from api.models_db import User, UserRole
from api.schemas.pastas import (
    AdaptiveBoundsResponse,
    CompareRequest,
    CompareResponse,
    FitParameter,
    FitRequest,
    FitResponse,
    PastasModelSummary,
    SavedScenario,
    SaveScenarioRequest,
    ScenarioRequest,
    ScenarioResponse,
    StationPreview,
    TimeSeriesData,
    ValidateModificationsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pastas", tags=["pastas"])

_RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _validate_run_id(run_id: str) -> None:
    """Raise 400 if run_id is not a 32-char MLflow hex ID.

    Prevents path traversal in artifact/model paths and SSRF via MLflow client.
    """
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "Format run_id invalide")


def _clean_tmin_tmax(params: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract tmin/tmax from MLflow params, treating 'auto'/empty as None."""
    tmin = params.get("tmin")
    tmax = params.get("tmax")
    if tmin in (None, "", "auto", "None"):
        tmin = None
    if tmax in (None, "", "auto", "None"):
        tmax = None
    return tmin, tmax


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


def _rebuild_val_metrics(all_metrics: dict) -> Optional[dict[str, float]]:
    """Extract val_* metrics from MLflow metrics dict."""
    val = {k[4:]: v for k, v in all_metrics.items() if k.startswith("val_")}
    return val if val else None


def _rebuild_period(tags: dict, prefix: str) -> Optional[list[str]]:
    """Extract cal/val period from MLflow tags."""
    tmin = tags.get(f"{prefix}_tmin")
    tmax = tags.get(f"{prefix}_tmax")
    if tmin and tmax:
        return [tmin, tmax]
    return None


def _get_cal_val_periods(run) -> tuple[tuple[Optional[str], Optional[str]], Optional[tuple[str, str]]]:
    """Extract calibration and validation periods from an MLflow run.

    Returns (cal_period, val_period) where cal_period is (tmin, tmax)
    and val_period is (tmin, tmax) or None if no validation split.
    """
    tags = run.data.tags
    params = run.data.params
    cal_tmin = tags.get("cal_tmin")
    cal_tmax = tags.get("cal_tmax")
    val_tmin = tags.get("val_tmin")
    val_tmax = tags.get("val_tmax")

    if cal_tmin and cal_tmax:
        cal = (cal_tmin, cal_tmax)
    else:
        cal = _clean_tmin_tmax(params)

    val = (val_tmin, val_tmax) if val_tmin and val_tmax else None
    return cal, val


def _series_to_ts(s: pd.Series) -> TimeSeriesData:
    return TimeSeriesData(
        index=[str(d) for d in s.index],
        values=[float(v) if pd.notna(v) else None for v in s.values],
    )


# ---------------------------------------------------------------------------
# GET /options
# ---------------------------------------------------------------------------

@router.get("/options")
def get_options() -> dict:
    """Return available Pastas component options for UI dropdowns."""
    from dashboard.utils.pastas.config import get_options as _get_options
    return _get_options()


# ---------------------------------------------------------------------------
# GET /siblings?code_bss=...
# ---------------------------------------------------------------------------

@router.get("/siblings")
def get_siblings(code_bss: str, limit: int = 20):
    """Return nearby stations from the same BDLISA aquifer."""
    from sqlalchemy import create_engine, text as sql_text

    db_url = _brgm_url()
    engine = create_engine(db_url)
    try:
        # Get this station's BDLISA code
        with engine.connect() as conn:
            bdlisa_df = pd.read_sql(
                sql_text("SELECT codes_bdlisa FROM gold.int_station_era5_mapping WHERE code_bss = :code LIMIT 1"),
                conn, params={"code": code_bss},
            )
        if bdlisa_df.empty or pd.isna(bdlisa_df.iloc[0]["codes_bdlisa"]):
            return {"siblings": []}

        bdlisa_code = str(bdlisa_df.iloc[0]["codes_bdlisa"])

        # Find other stations with the same BDLISA code
        with engine.connect() as conn:
            siblings_df = pd.read_sql(
                sql_text("""
                    SELECT DISTINCT m.code_bss, s.nom_commune,
                           s.latitude, s.longitude
                    FROM gold.int_station_era5_mapping m
                    LEFT JOIN gold.dim_piezo_stations s ON s.code_bss = m.code_bss
                    WHERE m.codes_bdlisa = :bdlisa AND m.code_bss != :code
                      AND s.latitude IS NOT NULL AND s.longitude IS NOT NULL
                    LIMIT :lim
                """),
                conn, params={"bdlisa": bdlisa_code, "code": code_bss, "lim": limit},
            )

        siblings = [
            {
                "code_bss": row["code_bss"],
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "nom_commune": row.get("nom_commune"),
            }
            for _, row in siblings_df.iterrows()
        ]
        return {"siblings": siblings, "bdlisa_code": bdlisa_code}
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# GET /station-info?code_bss=...  (fast, metadata only, <50ms)
# ---------------------------------------------------------------------------

@router.get("/station-info")
def station_info(code_bss: str):
    """Return rich station metadata from dim_piezo_stations (instant)."""
    from dashboard.utils.pastas.station_loader import load_station_metadata
    from dashboard.utils.pastas.config import get_preset

    try:
        meta = load_station_metadata(code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    preset = get_preset(meta.get("nature_eh"), meta.get("milieu_eh"))
    return {**meta, "preset": preset}


# ---------------------------------------------------------------------------
# GET /preview?code_bss=...  (heavy, loads full series)
# ---------------------------------------------------------------------------

@router.get("/preview")
def preview_station(code_bss: str):
    """Return raw series + statistics for a station."""
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

    from dashboard.utils.pastas.config import get_preset
    preset = get_preset(
        station.metadata.get("nature_eh"),
        station.metadata.get("milieu_eh"),
    )

    return StationPreview(
        code_bss=code_bss,
        metadata=station.metadata,
        piezo=_series_to_ts(piezo),
        precip=_series_to_ts(station.precip),
        evap=_series_to_ts(station.evap),
        stats=stats,
        preset=preset,
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

    # Build additional stresses
    extra_stresses = None
    if req.include_temp:
        extra_stresses = [{
            "type": "custom",
            "name": "temperature",
            "rfunc": "Gamma",
            "series": station.temp,
        }]

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
            station_metadata=station.metadata,
        )
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Pastas fit failed: %s", exc)
        raise HTTPException(500, f"Échec de la calibration : {exc}") from exc

    return FitResponse(
        run_id=result.run_id,
        code_bss=req.code_bss,
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
def list_models(
    code_bss: Optional[str] = None,
    current: User = Depends(get_current_user),
) -> list[PastasModelSummary]:
    """List Pastas models stored in MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("pastas")
    if experiment is None:
        return []

    clauses = []
    if code_bss:
        if not re.fullmatch(r"[A-Za-z0-9/_.X-]+", code_bss):
            raise HTTPException(422, "Format code_bss invalide")
        clauses.append(f"tags.station_id = '{code_bss}'")
    if current.role != UserRole.admin:
        clauses.append(f"tags.owner_id = '{current.id}'")
    filter_str = " and ".join(clauses)

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
                noise_type=params.get("noise_type", "unknown"),
                solver_type=params.get("solver_type", "unknown"),
                evp=metrics.get("evp"),
                rmse=metrics.get("rmse"),
                nse=metrics.get("nse"),
                aic=metrics.get("aic"),
                val_nse=metrics.get("val_nse"),
                val_evp=metrics.get("val_evp"),
                has_validation="val_tmin" in tags,
                include_temp=params.get("include_temp") == "True",
                created_at=str(run.info.start_time),
                pastas_version=tags.get("pastas_version", "unknown"),
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# GET /models/{run_id}
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}", response_model=FitResponse)
def get_model(run_id: str, current: User = Depends(get_current_user)) -> FitResponse:
    """Reconstruct FitResponse for a stored Pastas model."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.fit_service import _extract_parameters, _acf_stats
    from dashboard.utils.pastas.io import load_model

    try:
        model = load_model(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", run_id, exc)
        raise HTTPException(500, f"Échec du chargement du modèle : {exc}") from exc

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = dict(run.data.metrics)
    tags = run.data.tags
    params = run.data.params

    tmin, tmax = _clean_tmin_tmax(params)
    val_period = _rebuild_period(tags, "val")
    # Simulate over full period (cal+val) so frontend can plot both
    tmax_full = val_period[1] if val_period else tmax

    try:
        observed = model.observations(tmin=tmin, tmax=tmax_full)
        simulated = model.simulate(tmin=tmin, tmax=tmax_full)
        residuals = model.residuals(tmin=tmin, tmax=tmax)
    except Exception as exc:
        raise HTTPException(500, f"Échec de la reconstruction des séries : {exc}") from exc

    contributions: dict[str, pd.Series] = {}
    for sm_name in model.stressmodels:
        try:
            contributions[sm_name] = model.get_contribution(sm_name, tmin=tmin, tmax=tmax_full)
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
        code_bss=run.data.params.get("dataset_id", tags.get("station_id", "")),
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
        validation_metrics=_rebuild_val_metrics(metrics),
        cal_period=_rebuild_period(tags, "cal"),
        val_period=_rebuild_period(tags, "val"),
    )


# ---------------------------------------------------------------------------
# GET /models/{run_id}/signatures
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/signatures")
def get_signatures(run_id: str, current: User = Depends(get_current_user)):
    """Compute hydrological signatures (observed vs simulated) for a stored Pastas model."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.signatures import compute_signatures

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin, tmax = _clean_tmin_tmax(run.data.params)

    try:
        obs = model.observations(tmin=tmin, tmax=tmax)
        sim = model.simulate(tmin=tmin, tmax=tmax)
    except Exception as exc:
        raise HTTPException(500, f"Échec du calcul des séries : {exc}") from exc

    try:
        return compute_signatures(obs, sim)
    except Exception as exc:
        logger.exception("Signatures computation failed: %s", exc)
        raise HTTPException(500, f"Échec du calcul des signatures : {exc}") from exc


# ---------------------------------------------------------------------------
# GET /models/{run_id}/diagnostics
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/diagnostics")
def get_diagnostics(run_id: str, current: User = Depends(get_current_user)):
    """Compute full diagnostic statistics on residuals — split by cal/val."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.diagnostics import compute_diagnostics

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)

    cal_residuals = model.residuals(tmin=cal[0], tmax=cal[1])
    result = {"cal": compute_diagnostics(cal_residuals), "val": None}
    if val:
        try:
            val_residuals = model.residuals(tmin=val[0], tmax=val[1])
            if len(val_residuals.dropna()) >= 10:
                result["val"] = compute_diagnostics(val_residuals)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# GET /models/{run_id}/outlier-diagnostics
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/outlier-diagnostics")
def get_outlier_diagnostics(run_id: str, current: User = Depends(get_current_user)):
    """Compute outlier diagnostics — split by cal/val."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.outlier_diagnostics import compute_outlier_diagnostics
    from sqlalchemy import create_engine

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    code_bss = run.data.params.get("dataset_id", run.data.tags.get("station_id", ""))

    engine = create_engine(_brgm_url())
    try:
        cal_result = compute_outlier_diagnostics(
            model=model, code_bss=code_bss,
            cal_tmin=cal[0], cal_tmax=cal[1], engine=engine,
        )
        cal_result["run_id"] = run_id
        cal_result["period"] = "cal"

        val_result = None
        if val:
            try:
                val_result = compute_outlier_diagnostics(
                    model=model, code_bss=code_bss,
                    cal_tmin=val[0], cal_tmax=val[1], engine=engine,
                )
                val_result["run_id"] = run_id
                val_result["period"] = "val"
            except Exception:
                pass
    finally:
        engine.dispose()

    return {"cal": cal_result, "val": val_result}


# ---------------------------------------------------------------------------
# GET /models/{run_id}/confidence-bands
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/confidence-bands")
def get_confidence_bands(run_id: str, current: User = Depends(get_current_user)):
    """Compute bootstrap confidence bands — full period (cal+val)."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.confidence_intervals import compute_confidence_bands

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    tmax_full = val[1] if val else cal[1]

    return compute_confidence_bands(model, cal[0], tmax_full)


# ---------------------------------------------------------------------------
# GET /models/{run_id}/recession
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/recession")
def get_recession(run_id: str, current: User = Depends(get_current_user)):
    """Compute recession analysis — full observed period."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.recession import compute_recession_analysis

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    tmax_full = val[1] if val else cal[1]

    return compute_recession_analysis(model, cal[0], tmax_full)


# ---------------------------------------------------------------------------
# GET /models/{run_id}/baseflow
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/baseflow")
def get_baseflow(run_id: str, current: User = Depends(get_current_user)):
    """Compute baseflow separation — full observed period."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.baseflow import compute_baseflow

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    tmax_full = val[1] if val else cal[1]

    return compute_baseflow(model, cal[0], tmax_full)


# ---------------------------------------------------------------------------
# GET /models/{run_id}/spectral
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/spectral")
def get_spectral(run_id: str, current: User = Depends(get_current_user)):
    """Compute spectral analysis — split by cal/val."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.spectral import compute_spectral_analysis

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)

    result = {"cal": compute_spectral_analysis(model, cal[0], cal[1]), "val": None}
    if val:
        try:
            val_spectral = compute_spectral_analysis(model, val[0], val[1])
            if val_spectral.get("frequencies"):
                result["val"] = val_spectral
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# GET /models/{run_id}/decomposition
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/decomposition")
def get_decomposition(run_id: str, current: User = Depends(get_current_user)):
    """Compute STL decomposition — full observed period."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.signal_decomposition import compute_stl_decomposition

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    tmax_full = val[1] if val else cal[1]

    return compute_stl_decomposition(model, cal[0], tmax_full)


# ---------------------------------------------------------------------------
# GET /models/{run_id}/cross-correlation
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/cross-correlation")
def get_cross_correlation(run_id: str, current: User = Depends(get_current_user)):
    """Compute precipitation-piezometry cross-correlogram — full period."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.cross_correlation import compute_cross_correlation
    from sqlalchemy import create_engine

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    tmax_full = val[1] if val else cal[1]
    code_bss = run.data.params.get("dataset_id", run.data.tags.get("station_id", ""))

    engine = create_engine(_brgm_url())
    try:
        return compute_cross_correlation(model, code_bss, cal[0], tmax_full, engine)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# GET /models/{run_id}/regional-residuals
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/regional-residuals")
def get_regional_residuals(run_id: str, current: User = Depends(get_current_user)):
    """Compare model residuals with neighbors — split by cal/val."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.multi_station_residuals import compute_regional_residuals
    from sqlalchemy import create_engine

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    cal, val = _get_cal_val_periods(run)
    code_bss = run.data.params.get("dataset_id", run.data.tags.get("station_id", ""))

    engine = create_engine(_brgm_url())
    try:
        result = {"cal": compute_regional_residuals(model, code_bss, cal[0], cal[1], engine), "val": None}
        if val:
            try:
                result["val"] = compute_regional_residuals(model, code_bss, val[0], val[1], engine)
            except Exception:
                pass
        return result
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# GET /models/{run_id}/input-quality
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/input-quality")
def get_input_quality(run_id: str, current: User = Depends(get_current_user)):
    """Detect anomalous months in input data using Isolation Forest."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.input_quality import detect_input_anomalies
    from sqlalchemy import create_engine

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        raise HTTPException(404, f"Exécution '{run_id}' introuvable")
    code_bss = run.data.params.get("dataset_id", run.data.tags.get("station_id", ""))

    engine = create_engine(_brgm_url())
    try:
        return detect_input_anomalies(code_bss, engine)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# GET /models/{run_id}/export/pas
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/export/pas")
def export_pas(run_id: str, current: User = Depends(get_current_user)):
    """Export a Pastas model as a .pas file."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    import os
    from starlette.background import BackgroundTask
    from dashboard.utils.pastas.io import load_model

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Modèle '{run_id}' introuvable")

    f = tempfile.NamedTemporaryFile(suffix=".pas", delete=False)
    f.close()
    model.to_file(f.name)
    return FileResponse(
        f.name,
        filename=f"pastas_{run_id[:8]}.pas",
        media_type="application/octet-stream",
        background=BackgroundTask(os.unlink, f.name),
    )


# ---------------------------------------------------------------------------
# GET /models/{run_id}/export/csv
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/export/csv")
def export_csv(run_id: str, current: User = Depends(get_current_user)):
    """Export model params, metrics and tags as a CSV file."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        raise HTTPException(404, f"Exécution '{run_id}' introuvable")

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
def delete_model(run_id: str, current: User = Depends(get_current_user)) -> dict:
    """Delete a Pastas model from MLflow and evict cache."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
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
# POST /compare
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=CompareResponse)
def compare_models_endpoint(req: CompareRequest, current: User = Depends(get_current_user)) -> CompareResponse:
    """Load N models and return side-by-side metrics + aligned series."""
    from dashboard.utils.pastas.comparison import compare_models

    if len(req.run_ids) < 2 or len(req.run_ids) > 5:
        raise HTTPException(422, "Provide 2-5 run IDs to compare")
    for rid in req.run_ids:
        assert_owns_model(current, rid)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        results = compare_models(req.run_ids)
    except Exception as exc:
        logger.exception("Comparison failed: %s", exc)
        raise HTTPException(500, str(exc)) from exc

    return CompareResponse(
        models=[
            {
                "run_id": m["run_id"],
                "name": m["name"],
                "code_bss": m["code_bss"],
                "params": m["params"],
                "metrics": m["metrics"],
                "observed": _series_to_ts(m["observed"]),
                "simulated": _series_to_ts(m["simulated"]),
            }
            for m in results
        ]
    )


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


# ---------------------------------------------------------------------------
# GET /scenario-presets
# ---------------------------------------------------------------------------

@router.get("/scenario-presets")
def scenario_presets(
    aquifer_family: str | None = Query(None),
    tmin: str | None = Query(None),
    tmax: str | None = Query(None),
):
    """Return the full scenario referential for frontend cache."""
    from dashboard.utils.pastas.scenario_presets import (
        AQUIFER_FAMILY_LABELS,
        SCALE_STRESS_LIMITS,
        LINEAR_TREND_LIMITS,
        get_all_profiles,
        build_preset_scenarios,
        _range_to_dict,
    )

    family = aquifer_family or "sedimentary"
    t0 = tmin or "2020-01-01"
    t1 = tmax or "2024-12-31"

    return {
        "aquifer_families": AQUIFER_FAMILY_LABELS,
        "pumping_profiles": get_all_profiles(),
        "non_pumping_limits": {
            "scale_stress": _range_to_dict(SCALE_STRESS_LIMITS),
            "linear_trend": _range_to_dict(LINEAR_TREND_LIMITS),
        },
        "presets": build_preset_scenarios(family, t0, t1),
        "detected_family": family,
    }


# ---------------------------------------------------------------------------
# POST /validate-modifications
# ---------------------------------------------------------------------------

@router.post("/validate-modifications")
def validate_modifications_endpoint(req: ValidateModificationsRequest):
    """Pre-validate modifications without running a simulation."""
    from dashboard.utils.pastas.scenario_presets import validate_modifications as _validate

    family = req.aquifer_family or "sedimentary"
    mods = [m.model_dump() for m in req.modifications]
    result = _validate(mods, family)

    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# GET /models/{run_id}/scenarios
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/scenarios")
def get_scenarios(run_id: str, current: User = Depends(get_current_user)):
    """List saved scenarios for a model."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.scenario_presets import list_scenarios
    return list_scenarios(run_id)


# ---------------------------------------------------------------------------
# POST /models/{run_id}/scenarios
# ---------------------------------------------------------------------------

@router.post("/models/{run_id}/scenarios", status_code=201)
def create_scenario(run_id: str, req: SaveScenarioRequest, current: User = Depends(get_current_user)):
    """Save a named scenario."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.scenario_presets import save_scenario
    from dashboard.utils.pastas.scenario import resolve_aquifer_family

    family = resolve_aquifer_family(run_id)
    mods = [m.model_dump() for m in req.modifications]
    save_scenario(
        run_id=run_id,
        name=req.name,
        modifications=mods,
        description=req.description,
        aquifer_family=family,
        tmin=str(req.tmin) if req.tmin else None,
        tmax=str(req.tmax) if req.tmax else None,
    )
    return {"status": "saved", "name": req.name}


# ---------------------------------------------------------------------------
# DELETE /models/{run_id}/scenarios/{name}
# ---------------------------------------------------------------------------

@router.delete("/models/{run_id}/scenarios/{name}")
def remove_scenario(run_id: str, name: str, current: User = Depends(get_current_user)):
    """Delete a saved scenario."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.scenario_presets import delete_scenario
    delete_scenario(run_id, name)
    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# POST /models/{run_id}/scenarios/{name}/apply
# ---------------------------------------------------------------------------

@router.post("/models/{run_id}/scenarios/{name}/apply")
def apply_scenario(run_id: str, name: str, target_run_id: str = Body(..., embed=True), current: User = Depends(get_current_user)):
    """Load a saved scenario, adjusting for cross-model reuse."""
    _validate_run_id(run_id)
    _validate_run_id(target_run_id)
    assert_owns_model(current, run_id)
    assert_owns_model(current, target_run_id)
    from dashboard.utils.pastas.scenario_presets import load_scenario, validate_modifications as _validate
    from dashboard.utils.pastas.scenario import resolve_aquifer_family
    from dashboard.utils.pastas.io import load_model

    scenario = load_scenario(run_id, name)
    target_family = resolve_aquifer_family(target_run_id)
    source_family = scenario.get("aquifer_family")

    extra_warnings = []
    if source_family and source_family != target_family:
        extra_warnings.append(
            f"Scénario calibré sur nappe {source_family}, appliqué sur nappe {target_family} "
            f"— vérifiez les ordres de grandeur"
        )

    target_model = load_model(target_run_id)
    model_tmin = str(target_model.get_tmin(use_oseries=True, use_stresses=True).date())
    model_tmax = str(target_model.get_tmax(use_oseries=True, use_stresses=True).date())

    for mod in scenario["modifications"]:
        if mod.get("start") and str(mod["start"]) < model_tmin:
            mod["start"] = model_tmin
        if mod.get("end") and str(mod["end"]) > model_tmax:
            mod["end"] = model_tmax

    validation = _validate(scenario["modifications"], target_family)
    extra_warnings.extend(validation.warnings)

    scenario["_warnings"] = extra_warnings
    return scenario


# ---------------------------------------------------------------------------
# GET /models/{run_id}/adaptive-bounds
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/adaptive-bounds", response_model=AdaptiveBoundsResponse)
def get_adaptive_bounds(run_id: str, t_final_days: Optional[int] = Query(None, ge=1), current: User = Depends(get_current_user)):
    """Compute adaptive pumping bounds from the calibrated model's step response."""
    _validate_run_id(run_id)
    assert_owns_model(current, run_id)
    from dashboard.utils.pastas.scenario_presets import compute_adaptive_bounds

    result = compute_adaptive_bounds(run_id, t_final_days=t_final_days)
    if result is None:
        raise HTTPException(404, "No step response available for this model")

    return AdaptiveBoundsResponse(
        gain_A=result.gain_A,
        t95_days=result.t95_days,
        step_response_at_t=result.step_response_at_t,
        t_final_days=result.t_final_days,
        soft_drawdown_m=result.soft_drawdown_m,
        hard_drawdown_m=result.hard_drawdown_m,
        Q_soft=result.Q_soft,
        Q_hard=result.Q_hard,
        source=result.source,
    )


# ---------------------------------------------------------------------------
# POST /auto-fit
# ---------------------------------------------------------------------------

@router.post("/auto-fit")
def auto_fit_endpoint(code_bss: str = Body(...), warm_up_years: int = Body(2), val_split: float = Body(0.2), include_temp: bool = Body(False), add_trend: Optional[bool] = Body(None)):
    from dashboard.utils.pastas.station_loader import load_station_series
    from dashboard.utils.pastas.auto_fit import run_auto_fit
    from dashboard.utils.pastas.config import get_preset
    from dashboard.utils.pastas.diagnostics_prefit import run_prefit_diagnostics

    try:
        station = load_station_series(code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    preset = get_preset(station.metadata.get("nature_eh"), station.metadata.get("milieu_eh"))

    detect_trend = add_trend
    if detect_trend is None:
        diag = run_prefit_diagnostics(station.piezo)
        detect_trend = diag.trend_detected

    result = run_auto_fit(
        gwl=station.piezo, precip=station.precip, evap=station.evap, temp=station.temp,
        code_bss=code_bss, db_url=_brgm_url(),
        bdlisa_preset=preset, warm_up_years=warm_up_years,
        add_trend=detect_trend, val_split=val_split, include_temp=include_temp,
    )

    candidates = []
    for c in result.candidates:
        m = c.fit_result.metrics if c.fit_result else {}
        candidates.append({
            "config": c.config,
            "nse": m.get("nse"),
            "rmse": m.get("rmse"),
            "kge": m.get("kge"),
            "run_id": c.fit_result.run_id if c.fit_result else None,
            "stowa": {
                "evp_pass": c.stowa.evp_pass, "evp_value": c.stowa.evp_value,
                "autocorrelation_pass": c.stowa.autocorrelation_pass, "runs_test_pvalue": c.stowa.runs_test_pvalue,
                "t95_pass": c.stowa.t95_pass, "t95_days": c.stowa.t95_days, "t95_threshold": c.stowa.t95_threshold,
                "gain_pass": c.stowa.gain_pass, "gain_significance": c.stowa.gain_significance,
                "overall_pass": c.stowa.overall_pass, "suggestions": c.stowa.suggestions,
            } if c.stowa else None,
            "error": c.error,
            "elapsed_s": round(c.elapsed_s, 1),
        })

    return {
        "candidates": candidates,
        "best_run_id": result.best.fit_result.run_id if result.best and result.best.fit_result else None,
        "best_config": result.best.config if result.best else None,
        "total_elapsed_s": round(result.total_elapsed_s, 1),
    }


# ---------------------------------------------------------------------------
# POST /compare-ai
# ---------------------------------------------------------------------------

@router.post("/compare-ai")
def compare_ai_endpoint(pastas_run_id: str = Body(...), ai_model_id: str = Body(...), current: User = Depends(get_current_user)):
    _validate_run_id(pastas_run_id)
    assert_owns_model(current, pastas_run_id)
    assert_owns_model(current, ai_model_id)
    from dashboard.utils.pastas.io import load_model as load_pastas_model
    import numpy as np

    try:
        ps_model = load_pastas_model(pastas_run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Pastas model not found: {exc}") from exc

    ps_sim = ps_model.simulate()
    ps_obs = ps_model.observations()

    # Load AI model predictions
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        ai_run = client.get_run(ai_model_id)
    except Exception:
        raise HTTPException(404, f"AI model {ai_model_id} not found")

    # For now, return comparison structure even if AI data is limited
    # Full implementation needs the forecast pipeline
    ps_dates = [str(d.date()) for d in ps_sim.index]
    ps_vals = [float(v) if pd.notna(v) else None for v in ps_sim.values]
    obs_vals = [float(ps_obs.get(d, float("nan"))) if d in ps_obs.index else None for d in ps_sim.index]

    def _nse(obs, pred):
        o = np.array([v for v, p in zip(obs, pred) if v is not None and p is not None], dtype=float)
        p = np.array([p for v, p in zip(obs, pred) if v is not None and p is not None], dtype=float)
        if len(o) < 10: return None
        ss_res = np.sum((o - p) ** 2)
        ss_tot = np.sum((o - np.mean(o)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    pastas_nse = _nse(obs_vals, ps_vals)

    return {
        "common_period": [ps_dates[0], ps_dates[-1]] if ps_dates else [],
        "dates": ps_dates,
        "observed": obs_vals,
        "pastas_simulated": ps_vals,
        "ai_predicted": [],  # populated when AI forecast pipeline is wired
        "metrics": [
            {"metric": "NSE", "pastas_value": pastas_nse, "ai_value": None, "best": "pastas"},
        ],
    }
