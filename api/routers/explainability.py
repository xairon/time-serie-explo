"""Explainability API router.

Feature importance (correlation, permutation), attention (TFT), SHAP, gradient-based methods.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.serializers import clean_nans
from api.schemas.explainability import ExplainRequest, ExplainResult

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
    config = registry.load_model_config(entry)
    scalers = registry.load_scalers(entry)

    # Load train data for background/context
    train_df = registry.load_data(entry, "train")
    test_df = registry.load_data(entry, "test")

    columns = config.get("columns", {})
    target_col = columns.get("target", "")
    cov_cols = columns.get("covariates", [])
    hyperparams = config.get("hyperparams", {})

    input_chunk = int(hyperparams.get("input_chunk_length", 30))
    output_chunk = int(hyperparams.get("output_chunk_length", 7))

    return model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/feature-importance", response_model=ExplainResult)
async def feature_importance(req: ExplainRequest):
    """Compute feature importance (correlation or permutation)."""
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
        fill_method = preprocessing_config.get("fill_method", "Interpolation linéaire")

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
async def attention_analysis(req: ExplainRequest):
    """Extract attention weights (TFT/Transformer models only)."""
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
    fill_method = preprocessing_config.get("fill_method", "Interpolation linéaire")

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
async def shap_analysis(req: ExplainRequest):
    """Compute SHAP values using TimeSHAP or perturbation fallback."""
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    from dashboard.utils.explainability.feature_importance import compute_shap_importance
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts
    import numpy as np

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Interpolation linéaire")

    series, covariates = prepare_dataframe_for_darts(
        train_df, target_col=target_col, covariate_cols=cov_cols, freq="D", fill_method=fill_method
    )

    # Build model wrapper and input data
    target_values = series.values()[-input_chunk:]
    features = [target_values]
    feature_names = [target_col]

    if covariates is not None:
        cov_values = covariates.values()[-input_chunk:]
        if cov_values.ndim == 1:
            cov_values = cov_values.reshape(-1, 1)
        features.append(cov_values)
        cov_df = covariates.to_dataframe()
        feature_names.extend(list(cov_df.columns))

    data = np.concatenate(features, axis=1).reshape(1, input_chunk, -1)

    def model_wrapper(x):
        """Wrap the Darts model for SHAP-style (batch, seq, features) -> (batch, horizon)."""
        from darts import TimeSeries
        import pandas as pd

        ts = TimeSeries.from_values(x[0, :, 0:1])
        try:
            pred = model.predict(n=output_chunk, series=ts)
            return pred.values().reshape(1, -1)
        except Exception:
            return np.zeros((1, output_chunk))

    result = compute_shap_importance(
        model_wrapper=model_wrapper,
        data=data,
        feature_names=feature_names,
        n_samples=req.n_samples,
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
async def gradient_analysis(req: ExplainRequest):
    """Compute gradient-based attributions (saliency, integrated gradients, or deeplift)."""
    model, entry, config, scalers, train_df, test_df, target_col, cov_cols, input_chunk, output_chunk = (
        _load_model_for_explain(req.model_id)
    )

    from dashboard.utils.explainability.gradients import compute_gradient_attributions
    from dashboard.utils.preprocessing import prepare_dataframe_for_darts

    if train_df is None:
        raise HTTPException(status_code=404, detail="Training data not found")

    preprocessing_config = config.get("preprocessing", {})
    fill_method = preprocessing_config.get("fill_method", "Interpolation linéaire")

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
