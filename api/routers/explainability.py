"""Explainability API router.

Feature importance (correlation, permutation), attention (TFT), SHAP, gradient-based methods.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.auth.deps import get_current_user
from api.auth.ownership import assert_owns_model
from api.config import settings
from api.models_db import User
from api.serializers import clean_nans
from api.utils import force_cpu_if_needed
from api.schemas.explainability import (
    ExplainRequest,
    ExplainResult,
    LagImportanceResult,
    ResidualAnalysisResult,
    SeasonalityResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/explainability", tags=["explainability"])


def _load_model_for_explain(model_id: str):
    """Load model, data, and metadata needed for explainability."""
    from dashboard.utils.model_registry import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model = registry.load_model(entry)
    force_cpu_if_needed(model)
    config = registry.load_model_config(entry)
    scalers = registry.load_scalers(entry)

    # Load train data for background/context
    train_df = registry.load_data(entry, "train")
    test_df = registry.load_data(entry, "test")

    columns = config.get("columns", {})
    target_col = columns.get("target", "")
    cov_cols = columns.get("covariates", [])

    # Merge covariate splits into target DataFrames (covariates saved separately)
    if cov_cols:
        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            if split_df is None:
                continue
            cov_df = registry.load_data(entry, f"{split_name}_cov")
            if cov_df is not None:
                for col in cov_df.columns:
                    if col not in split_df.columns:
                        split_df[col] = cov_df[col]
    hyperparams = config.get("hyperparams", {})

    input_chunk = int(hyperparams.get("input_chunk_length", 30))
    output_chunk = int(hyperparams.get("output_chunk_length", 7))

    return model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/{model_id}/feature-importance", response_model=ExplainResult)
async def feature_importance_get(model_id: str, current: User = Depends(get_current_user)):
    """GET endpoint for feature importance (correlation method)."""
    assert_owns_model(current, model_id)
    req = ExplainRequest(model_id=model_id, method="correlation")
    return await feature_importance(req, current)


@router.post("/feature-importance", response_model=ExplainResult)
async def feature_importance(req: ExplainRequest, current: User = Depends(get_current_user)):
    """Compute feature importance (correlation or permutation)."""
    assert_owns_model(current, req.model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    if req.method == "correlation":
        from dashboard.utils.explainability.feature_importance import compute_correlation_importance

        importance = compute_correlation_importance(
            df=train_df,
            target_col=target_col,
            covariate_cols=cov_cols,
            absolute=True,
        )
        return ExplainResult(
            method="correlation",
            success=True,
            feature_importance=clean_nans(importance),
            feature_names=list(importance.keys()),
        )

    elif req.method == "permutation":
        from dashboard.utils.explainability.feature_importance import compute_permutation_importance
        from dashboard.utils.preprocessing import prepare_dataframe_for_darts

        preprocessing_config = config.get("preprocessing", {})
        fill_method = preprocessing_config.get("fill_method", "Linear interpolation")

        series, covariates = prepare_dataframe_for_darts(
            train_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
        )

        importance = compute_permutation_importance(
            model=model,
            series=series,
            covariates=covariates,
            n_permutations=req.n_permutations,
            output_chunk_length=output_chunk,
        )

        if "_error" in importance:
            return ExplainResult(
                method="permutation",
                success=False,
                error_message=importance["_error"],
            )

        return ExplainResult(
            method="permutation",
            success=True,
            feature_importance=clean_nans(importance),
            feature_names=list(importance.keys()),
        )

    else:
        raise HTTPException(status_code=400, detail=f"Method must be 'correlation' or 'permutation' for this endpoint")


@router.post("/attention", response_model=ExplainResult)
async def attention_analysis(req: ExplainRequest, current: User = Depends(get_current_user)):
    """Extract attention weights (TFT/Transformer models only)."""
    assert_owns_model(current, req.model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    from dashboard.utils.explainability.base import ModelType

    model_type = ModelType.from_model(model)
    if model_type not in (ModelType.TFT, ModelType.TRANSFORMER):
        raise HTTPException(
            status_code=400,
            detail=f"Attention extraction not supported for model type: {model_type.name}",
        )

    from dashboard.utils.explainability.attention import extract_tft_attention
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Linear interpolation")

    series, covariates = prepare_dataframe_for_darts(
        train_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
    )

    result = extract_tft_attention(
        model=model,
        series=series,
        past_covariates=covariates,
        background_series=series,
        background_past_covariates=covariates,
    )

    attention_list = None
    if result.get("attention") is not None:
        import numpy as np

        att = result["attention"]
        attention_list = att.tolist() if isinstance(att, np.ndarray) else att

    return ExplainResult(
        method="attention",
        success=result.get("success", False),
        error_message=result.get("error"),
        attention_weights=attention_list,
        encoder_importance=clean_nans(result.get("encoder_importance")),
        decoder_importance=clean_nans(result.get("decoder_importance")),
        model_type=model_type.name,
    )


@router.post("/shap", response_model=ExplainResult)
async def shap_analysis(req: ExplainRequest, current: User = Depends(get_current_user)):
    """Compute SHAP values using TimeSHAP or perturbation fallback."""
    assert_owns_model(current, req.model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    from dashboard.utils.explainability.feature_importance import compute_shap_importance
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts
    import numpy as np

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Linear interpolation")

    series, covariates = prepare_dataframe_for_darts(
        train_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
    )

    # Build model wrapper and input data
    target_values = series.values()[-input_chunk:]
    features = [target_values]
    feature_names = [target_col]

    n_cov_features = 0
    if covariates is not None:
        cov_values = covariates.values()[-input_chunk:]
        if cov_values.ndim == 1:
            cov_values = cov_values.reshape(-1, 1)
        n_cov_features = cov_values.shape[1]
        features.append(cov_values)
        cov_df = covariates.to_dataframe()
        feature_names.extend(list(cov_df.columns))

    data = np.concatenate(features, axis=1).reshape(1, input_chunk, -1)

    # Check if model uses past covariates
    uses_past_cov = n_cov_features > 0 and (
        getattr(model, "_uses_past_covariates", False) or
        getattr(model, "uses_past_covariates", False)
    )

    # Track wrapper health: SHAP perturbs the input and re-predicts many times.
    # If EVERY prediction throws, the wrapper returns zeros and SHAP silently
    # produces all-zero attributions — which must be reported as a failure, not
    # as a successful (but meaningless) explanation.
    wrapper_stats = {"calls": 0, "failures": 0, "last_error": None}

    def model_wrapper(x):
        """Wrap the Darts model for SHAP-style (batch, seq, features) -> (batch, horizon)."""
        from darts import TimeSeries

        # Reconstruct target series from column 0
        target_ts = TimeSeries.from_values(x[0, :, 0:1])

        wrapper_stats["calls"] += 1
        try:
            predict_kwargs = {"n": output_chunk, "series": target_ts}

            # Reconstruct past covariates from remaining columns if present
            if uses_past_cov and x.shape[2] > 1:
                cov_ts = TimeSeries.from_values(x[0, :, 1:])
                predict_kwargs["past_covariates"] = cov_ts

            pred = model.predict(**predict_kwargs)
            return pred.values().reshape(1, -1)
        except Exception as exc:
            wrapper_stats["failures"] += 1
            wrapper_stats["last_error"] = str(exc)
            return np.zeros((1, output_chunk))

    result = compute_shap_importance(
        model_wrapper=model_wrapper,
        data=data,
        feature_names=feature_names,
        n_samples=req.n_samples,
    )

    # Surface an outright failure instead of returning a zero-filled "success".
    all_failed = wrapper_stats["calls"] > 0 and wrapper_stats["failures"] == wrapper_stats["calls"]
    if all_failed or result.get("_error"):
        detail = result.get("_error") or (
            f"La prédiction du modèle a échoué à chaque itération SHAP : {wrapper_stats['last_error']}"
        )
        return ExplainResult(
            method=result.get("method", "shap"),
            success=False,
            error_message=detail,
            feature_names=feature_names,
        )

    shap_values_list = None
    if result.get("shap_values") is not None:
        sv = result["shap_values"]
        shap_values_list = sv.tolist() if isinstance(sv, np.ndarray) else sv

    return ExplainResult(
        method=result.get("method", "shap"),
        success=True,
        feature_importance=clean_nans(result.get("feature_importance")),
        shap_values=shap_values_list,
        feature_names=feature_names,
    )


@router.post("/gradients", response_model=ExplainResult)
async def gradient_analysis(req: ExplainRequest, current: User = Depends(get_current_user)):
    """Compute gradient-based attributions (saliency, integrated gradients, or deeplift)."""
    assert_owns_model(current, req.model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    from dashboard.utils.explainability.gradients import compute_gradient_attributions
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Linear interpolation")

    series, covariates = prepare_dataframe_for_darts(
        train_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
    )

    method = req.method if req.method in ("saliency", "integrated_gradients", "deeplift") else "integrated_gradients"

    result = compute_gradient_attributions(
        model=model,
        series=series,
        past_covariates=covariates,
        method=method,
        target_step=req.target_step,
        input_chunk_length=input_chunk,
        n_steps=req.n_steps,
    )

    import numpy as np

    temporal = None
    if result.get("temporal_importance") is not None:
        t = result["temporal_importance"]
        temporal = t.tolist() if isinstance(t, np.ndarray) else t

    attributions = None
    if result.get("attributions") is not None:
        a = result["attributions"]
        attributions = a.tolist() if isinstance(a, np.ndarray) else a

    feature_imp = None
    if result.get("feature_importance") is not None:
        fi = result["feature_importance"]
        if isinstance(fi, np.ndarray):
            feature_names = [target_col] + cov_cols
            feature_imp = {
                feature_names[i] if i < len(feature_names) else f"feature_{i}": float(v)
                for i, v in enumerate(fi)
            }
        elif isinstance(fi, dict):
            feature_imp = fi

    from dashboard.utils.explainability.base import ModelType

    model_type = ModelType.from_model(model)

    return ExplainResult(
        method=result.get("method", method),
        success=result.get("success", False),
        error_message=result.get("error"),
        feature_importance=clean_nans(feature_imp),
        temporal_importance=temporal,
        gradient_attributions=attributions,
        model_type=model_type.name,
        feature_names=[target_col] + cov_cols,
    )


@router.get("/{model_id}/lag-importance", response_model=LagImportanceResult)
async def lag_importance(model_id: str, max_lag: int = 60, current: User = Depends(get_current_user)):
    """Compute lag importance (autocorrelation analysis) for the target variable."""
    assert_owns_model(current, model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(model_id)
    )

    if test_df is None or target_col not in test_df.columns:
        # Fallback to train
        df = train_df
    else:
        df = test_df

    if df is None or target_col not in df.columns:
        raise HTTPException(status_code=404, detail="Data not found for lag analysis")

    import numpy as np
    from statsmodels.tsa.stattools import acf, pacf

    values = df[target_col].dropna().values
    n_lags = min(max_lag, len(values) // 3)

    acf_values = acf(values, nlags=n_lags, fft=True)
    try:
        pacf_values = pacf(values, nlags=n_lags)
    except Exception:
        pacf_values = None

    lags = list(range(n_lags + 1))
    acf_list = [float(v) if not np.isnan(v) else 0.0 for v in acf_values]
    pacf_list = [float(v) if not np.isnan(v) else 0.0 for v in pacf_values] if pacf_values is not None else None

    # Significant lags (above 2/sqrt(n) threshold)
    threshold = 2.0 / np.sqrt(len(values))
    significant = [i for i, v in enumerate(acf_list) if abs(v) > threshold and i > 0]

    return LagImportanceResult(
        lags=lags,
        autocorrelations=acf_list,
        partial_autocorrelations=pacf_list,
        significant_lags=significant,
    )


@router.get("/{model_id}/residuals", response_model=ResidualAnalysisResult)
async def residual_analysis(model_id: str, current: User = Depends(get_current_user)):
    """Compute residual analysis on the test set using sliding window predictions."""
    assert_owns_model(current, model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(model_id)
    )

    if test_df is None:
        raise HTTPException(status_code=404, detail="Test data not found")

    import numpy as np
    import pandas as pd
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts
    from dashboard.utils.model_registry import ModelRegistry

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    val_df = registry.load_data(entry, "val")
    if val_df is None:
        raise HTTPException(status_code=404, detail="Validation data not found")

    # Merge covariates into val_df
    if cov_cols:
        val_cov_df = registry.load_data(entry, "val_cov")
        if val_cov_df is not None:
            for col in val_cov_df.columns:
                if col not in val_df.columns:
                    val_df[col] = val_cov_df[col]

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Linear interpolation")

    # Build full series (train+val+test) for historical_forecasts
    full_df = pd.concat([train_df, val_df, test_df])
    full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

    full_series, full_cov = prepare_dataframe_for_darts(
        full_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
    )

    target_preprocessor = scalers.get("target_preprocessor") if scalers else None
    cov_preprocessor = scalers.get("cov_preprocessor") if scalers else None

    if target_preprocessor:
        full_series = target_preprocessor.transform(full_series)
    if cov_preprocessor and full_cov is not None:
        full_cov = cov_preprocessor.transform(full_cov)

    # Use historical_forecasts with stride=output_chunk for efficiency
    import pandas as pd
    test_start_dt = test_df.index[input_chunk] if len(test_df) > input_chunk else test_df.index[0]
    # Ensure test_start is in full_series time index (nearest match)
    ts_idx = full_series.time_index
    nearest_idx = ts_idx.get_indexer([pd.Timestamp(test_start_dt)], method='nearest')[0]
    test_start = ts_idx[nearest_idx]

    try:
        hf_kwargs = {
            "series": full_series,
            "start": test_start,
            "forecast_horizon": 1,
            "stride": output_chunk,
            "retrain": False,
            "verbose": False,
        }
        if full_cov is not None and (getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False)):
            hf_kwargs["past_covariates"] = full_cov

        preds = model.historical_forecasts(**hf_kwargs)

        from darts import concatenate as darts_concat

        if isinstance(preds, list):
            pred_series = darts_concat(preds)
        else:
            pred_series = preds

        if target_preprocessor:
            pred_series = target_preprocessor.inverse_transform(pred_series)
            # Get test series in raw scale for comparison
            test_series, _ = prepare_dataframe_for_darts(
                test_df, target_col=target_col, covariate_cols=None, freq="D", fill_method=fill_method
            )
        else:
            test_series, _ = prepare_dataframe_for_darts(
                test_df, target_col=target_col, covariate_cols=None, freq="D", fill_method=fill_method
            )

        # Align pred and actual by date index (handles sparse stride timestamps)
        pred_df = pred_series.to_dataframe()
        actual_df = test_series.to_dataframe()
        aligned = actual_df.reindex(pred_df.index).dropna()
        if len(aligned) == 0:
            raise ValueError("No overlapping dates between predictions and test set")
        pred_aligned = pred_df.loc[aligned.index]
        residuals = aligned.values.flatten() - pred_aligned.values.flatten()
        dates_out = [d.isoformat() for d in aligned.index]

    except Exception as e:
        logger.warning(f"Historical forecasts failed for residuals: {e}")
        raise HTTPException(status_code=500, detail=f"Residual computation failed: {e}")

    from scipy import stats

    mean_err = float(np.mean(residuals))
    std_err = float(np.std(residuals))
    skew = float(stats.skew(residuals)) if len(residuals) > 3 else None
    kurt = float(stats.kurtosis(residuals)) if len(residuals) > 3 else None

    try:
        _, norm_p = stats.normaltest(residuals)
        norm_p = float(norm_p)
    except Exception:
        norm_p = None

    acf1 = None
    if len(residuals) > 2:
        from statsmodels.tsa.stattools import acf

        acf_vals = acf(residuals, nlags=1, fft=True)
        acf1 = float(acf_vals[1])

    bias_status = "balanced" if abs(mean_err) < 0.5 * std_err else "biased"

    return ResidualAnalysisResult(
        mean_error=mean_err,
        std_error=std_err,
        skewness=skew,
        kurtosis=kurt,
        normality_pvalue=norm_p,
        acf_lag1=acf1,
        bias_status=bias_status,
        residuals=[float(v) if not np.isnan(v) else 0.0 for v in residuals],
        dates=dates_out,
    )


@router.get("/{model_id}/seasonality", response_model=SeasonalityResult)
async def seasonality_analysis(model_id: str, current: User = Depends(get_current_user)):
    """Detect seasonality patterns and compute STL decomposition."""
    assert_owns_model(current, model_id)
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(model_id)
    )

    df = test_df if test_df is not None else train_df
    if df is None or target_col not in df.columns:
        raise HTTPException(status_code=404, detail="Data not found for seasonality analysis")

    import numpy as np

    values = df[target_col].dropna()

    # Detect periodicity using FFT
    detected_periods = []
    period_strengths: dict[str, float] = {}
    candidate_periods = [7, 30, 90, 365]

    try:
        fft_vals = np.fft.rfft(values.values - values.values.mean())
        freqs = np.fft.rfftfreq(len(values), d=1)
        power = np.abs(fft_vals) ** 2
        avg_power = float(np.mean(power[1:]))

        for period in candidate_periods:
            if period >= len(values) // 2:
                continue
            target_freq = 1.0 / period
            idx = np.argmin(np.abs(freqs - target_freq))
            strength = float(power[idx] / avg_power) if avg_power > 0 else 0.0
            period_strengths[str(period)] = round(strength, 2)
            if strength > 3.0:
                detected_periods.append(period)
    except Exception:
        pass

    # STL decomposition
    variance_trend = None
    variance_seasonal = None
    variance_residual = None
    decomp_dict = None
    decomp_dates = None

    try:
        from statsmodels.tsa.seasonal import STL

        period = 365 if len(values) > 730 else 7
        stl = STL(values, period=period, robust=True)
        result = stl.fit()

        total_var = float(np.var(values.values))
        if total_var > 0:
            variance_trend = float(np.var(result.trend) / total_var * 100)
            variance_seasonal = float(np.var(result.seasonal) / total_var * 100)
            variance_residual = float(np.var(result.resid) / total_var * 100)

        # Subsample for transfer (max 500 points)
        step = max(1, len(values) // 500)
        decomp_dict = {
            "trend": [float(v) for v in result.trend[::step]],
            "seasonal": [float(v) for v in result.seasonal[::step]],
            "residual": [float(v) for v in result.resid[::step]],
        }
        decomp_dates = [d.isoformat() for d in values.index[::step]]
    except Exception as e:
        logger.warning(f"STL decomposition failed: {e}")

    return SeasonalityResult(
        detected_periods=detected_periods,
        period_strengths=period_strengths,
        decomposition=decomp_dict,
        decomposition_dates=decomp_dates,
        variance_trend=variance_trend,
        variance_seasonal=variance_seasonal,
        variance_residual=variance_residual,
    )
