"""Forecasting API router.

Single-window, rolling, comparison, and global forecast endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.serializers import clean_nans, serialize_timeseries
from api.schemas.forecasting import (
    ComparisonForecastRequest,
    ForecastRequest,
    ForecastResult,
    GlobalForecastRequest,
    RollingForecastRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/forecasting", tags=["forecasting"])


def _load_model_and_data(model_id: str):
    """Load model, scalers, config, and data splits from the model registry."""
    from dashboard.utils.model_registry import ModelRegistry

    from api.config import settings
    from pathlib import Path

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))

    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model = registry.load_model(entry)
    scalers = registry.load_scalers(entry)
    config = registry.load_model_config(entry)

    # Load data splits and reconstruct full DataFrame
    import pandas as pd

    train_df = registry.load_data(entry, "train")
    val_df = registry.load_data(entry, "val")
    test_df = registry.load_data(entry, "test")

    if train_df is None or val_df is None or test_df is None:
        raise HTTPException(status_code=404, detail="Model data splits not found")

    full_df = pd.concat([train_df, val_df, test_df])
    full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

    # Extract column info from config
    columns = config.get("columns", {})
    target_col = columns.get("target") or (full_df.columns[0] if len(full_df.columns) > 0 else "")
    cov_cols = columns.get("covariates", [])

    preprocessing_config = config.get("preprocessing", {})
    is_global = config.get("type") == "global"

    return model, full_df, target_col, cov_cols, preprocessing_config, scalers, is_global, entry, train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/run", response_model=ForecastResult)
async def run_forecast(req: ForecastRequest):
    """Convenience endpoint - alias for /single forecast."""
    return await single_forecast(req)


@router.post("/single", response_model=ForecastResult)
async def single_forecast(req: ForecastRequest):
    """Generate a single-window forecast from a given start date."""
    import pandas as pd

    from dashboard.utils.forecasting import generate_single_window_forecast

    model, full_df, target_col, cov_cols, preproc, scalers, is_global, entry, train_df, val_df, test_df = _load_model_and_data(req.model_id)

    if req.start_date:
        try:
            start_date = pd.Timestamp(req.start_date)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid start_date: {req.start_date}")
    else:
        # Default to test set start
        start_date = test_df.index[0] if test_df is not None and len(test_df) > 0 else full_df.index[len(full_df) // 2]

    try:
        pred_auto, pred_onestep, target_series, metrics_auto, metrics_onestep, horizon = (
            generate_single_window_forecast(
                model=model,
                full_df=full_df,
                target_col=target_col,
                covariate_cols=cov_cols if req.use_covariates else None,
                preprocessing_config=preproc,
                scalers=scalers,
                start_date=start_date,
                use_covariates=req.use_covariates,
                already_processed=True,
                is_global_model=is_global,
                freq=req.freq,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")

    return ForecastResult(
        predictions=clean_nans(serialize_timeseries(pred_auto)),
        target=clean_nans(serialize_timeseries(target_series)),
        metrics=clean_nans(metrics_auto),
        horizon=horizon,
        predictions_onestep=clean_nans(serialize_timeseries(pred_onestep)),
        metrics_onestep=clean_nans(metrics_onestep),
    )


@router.post("/rolling", response_model=ForecastResult)
async def rolling_forecast(req: RollingForecastRequest):
    """Generate rolling (historical) forecasts."""
    import pandas as pd

    from dashboard.utils.forecasting import generate_rolling_forecast

    model, full_df, target_col, cov_cols, preproc, scalers, *_ = _load_model_and_data(req.model_id)

    try:
        start_date = pd.Timestamp(req.start_date)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid start_date: {req.start_date}")

    try:
        forecasts, full_series = generate_rolling_forecast(
            model=model,
            full_df=full_df,
            target_col=target_col,
            covariate_cols=cov_cols if req.use_covariates else None,
            preprocessing_config=preproc,
            scalers=scalers,
            start_date=start_date,
            forecast_horizon=req.forecast_horizon,
            stride=req.stride,
            use_covariates=req.use_covariates,
            freq=req.freq,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rolling forecast failed: {exc}")

    windows = [clean_nans(serialize_timeseries(w)) for w in forecasts]
    return ForecastResult(
        predictions=[],
        target=clean_nans(serialize_timeseries(full_series)),
        metrics={},
        forecast_windows=windows,
    )


@router.post("/comparison", response_model=ForecastResult)
async def comparison_forecast(req: ComparisonForecastRequest):
    """Compare autoregressive vs teacher-forcing forecasts."""
    import pandas as pd

    from dashboard.utils.forecasting import generate_comparison_forecast

    model, full_df, target_col, cov_cols, preproc, scalers, *_ = _load_model_and_data(req.model_id)

    try:
        start_date = pd.Timestamp(req.start_date)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid start_date: {req.start_date}")

    try:
        target_slice, autoregressive, exact_window, metrics_auto, metrics_exact = (
            generate_comparison_forecast(
                model=model,
                full_df=full_df,
                target_col=target_col,
                covariate_cols=cov_cols if req.use_covariates else None,
                preprocessing_config=preproc,
                scalers=scalers,
                start_date=start_date,
                forecast_horizon=req.forecast_horizon,
                use_covariates=req.use_covariates,
                freq=req.freq,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison forecast failed: {exc}")

    return ForecastResult(
        predictions=clean_nans(serialize_timeseries(autoregressive)),
        target=clean_nans(serialize_timeseries(target_slice)),
        metrics=clean_nans(metrics_auto),
        predictions_exact=clean_nans(serialize_timeseries(exact_window)),
        metrics_exact=clean_nans(metrics_exact),
    )


@router.post("/global", response_model=ForecastResult)
async def global_forecast(req: GlobalForecastRequest):
    """Generate a global forecast over the full test set."""
    import pandas as pd

    from dashboard.utils.forecasting import generate_global_forecast

    model, full_df, target_col, cov_cols, preproc, scalers, is_global, entry, train_df, val_df, test_df = (
        _load_model_and_data(req.model_id)
    )

    history_df = pd.concat([train_df, val_df])
    history_df = history_df[~history_df.index.duplicated(keep="first")].sort_index()

    try:
        pred_series, target_series, metrics = generate_global_forecast(
            model=model,
            history_df=history_df,
            target_df=test_df,
            target_col=target_col,
            covariate_cols=cov_cols if req.use_covariates else None,
            preprocessing_config=preproc,
            scalers=scalers,
            use_covariates=req.use_covariates,
            freq=req.freq,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Global forecast failed: {exc}")

    return ForecastResult(
        predictions=clean_nans(serialize_timeseries(pred_series)),
        target=clean_nans(serialize_timeseries(target_series)),
        metrics=clean_nans(metrics),
    )
