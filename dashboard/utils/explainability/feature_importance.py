"""Feature importance methods for time series models.

Includes:
- Correlation-based importance
- Permutation importance
- SHAP (waterfall, force, interactions)
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

logger = __import__('logging').getLogger(__name__)


def compute_correlation_importance(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    absolute: bool = True
) -> Dict[str, float]:
    """
    Calculate feature importance based on correlation with target.

    Simple, robust method that always works.

    Args:
        df: DataFrame containing target and covariates
        target_col: Name of target column
        covariate_cols: List of covariate column names
        absolute: If True, return absolute correlations

    Returns:
        Dictionary mapping feature names to importance scores
    """
    correlations = {}

    if target_col not in df.columns:
        return {}

    target_values = df[target_col].dropna()

    for cov in covariate_cols:
        if cov in df.columns and cov != target_col:
            try:
                cov_values = df[cov].dropna()
                # Align indices
                common_idx = target_values.index.intersection(cov_values.index)
                if len(common_idx) > 2:
                    corr = target_values.loc[common_idx].corr(cov_values.loc[common_idx])
                    if not pd.isna(corr):
                        correlations[cov] = abs(corr) if absolute else corr
                    else:
                        correlations[cov] = 0.0
                else:
                    correlations[cov] = 0.0
            except Exception:
                correlations[cov] = 0.0

    return correlations


def compute_permutation_importance(
    model,
    series,
    covariates,
    n_permutations: int = 5,
    output_chunk_length: int = 7,
    metric: str = "mae"
) -> Dict[str, float]:
    """
    Compute permutation importance by shuffling features.

    For each feature, shuffles its values and measures prediction degradation.

    Args:
        model: Darts forecasting model
        series: Target TimeSeries
        covariates: Past covariates TimeSeries
        n_permutations: Number of shuffle iterations per feature
        output_chunk_length: Prediction horizon
        metric: Degradation metric ('mae', 'mse', 'rmse')

    Returns:
        Dictionary of normalized importance scores
    """
    from darts import TimeSeries
    from darts.metrics import mae, mse, rmse

    if covariates is None:
        return {"target": 1.0}

    metrics = {"mae": mae, "mse": mse, "rmse": rmse}
    metric_fn = metrics.get(metric, mae)

    try:
        # Baseline prediction
        predict_kwargs = {"n": output_chunk_length, "series": series}

        # Check for covariate usage - try multiple attribute patterns
        uses_past = (
            getattr(model, "_uses_past_covariates", False) or
            getattr(model, "uses_past_covariates", False) or
            hasattr(model, "past_covariate_series")
        )
        uses_future = (
            getattr(model, "_uses_future_covariates", False) or
            getattr(model, "uses_future_covariates", False)
        )

        if uses_past:
            predict_kwargs["past_covariates"] = covariates
        if uses_future:
            predict_kwargs["future_covariates"] = covariates

        baseline_pred = model.predict(**predict_kwargs)

        # Compute baseline score against actual values (end of series)
        actual_end = series.slice_n_points_after(
            series.time_index[-output_chunk_length], output_chunk_length
        ) if len(series) > output_chunk_length else series
        baseline_score = float(metric_fn(actual_end, baseline_pred))

        cov_df = covariates.to_dataframe()
        feature_names = list(cov_df.columns)

        # Get frequency - try multiple methods
        cov_freq = getattr(covariates, 'freq_str', None) or getattr(covariates, 'freq', None)
        if cov_freq is None:
            # Infer from index
            cov_freq = pd.infer_freq(cov_df.index)

        importances = {}
        last_error = None

        for feature in feature_names:
            degradations = []

            for _ in range(n_permutations):
                try:
                    # Shuffle feature values
                    cov_permuted = cov_df.copy()
                    cov_permuted[feature] = np.random.permutation(
                        cov_permuted[feature].values
                    )

                    # Recreate TimeSeries with same index
                    cov_permuted_ts = TimeSeries.from_dataframe(
                        cov_permuted, freq=cov_freq
                    )

                    # Predict with shuffled covariates
                    perm_kwargs = {"n": output_chunk_length, "series": series}
                    if uses_past:
                        perm_kwargs["past_covariates"] = cov_permuted_ts
                    if uses_future:
                        perm_kwargs["future_covariates"] = cov_permuted_ts

                    permuted_pred = model.predict(**perm_kwargs)

                    # Measure degradation: how much worse is the permuted prediction
                    # compared to baseline, measured against actual values
                    permuted_score = float(metric_fn(actual_end, permuted_pred))
                    degradation = permuted_score - baseline_score
                    degradations.append(max(degradation, 0.0))

                except Exception as inner_e:
                    last_error = inner_e
                    continue

            if degradations:
                importances[feature] = np.mean(degradations)
            else:
                importances[feature] = 0.0
                if last_error:
                    logger.warning(f"  Feature {feature} failed: {last_error}")

        # Check if all features failed
        if all(v == 0.0 for v in importances.values()) and last_error:
            return {"_error": f"All permutations failed. Last error: {last_error}"}

        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    except Exception as e:
        logger.exception(f"Permutation importance failed: {e}")
        return {"_error": str(e)}


def compute_shap_importance(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
    use_timeshap: bool = True
) -> Dict[str, Any]:
    """
    Compute SHAP values using TimeSHAP or fallback perturbation.

    Args:
        model_wrapper: Callable f(X) -> Y where X is (n, seq_len, n_features)
        data: Input data array (1, seq_len, n_features)
        feature_names: List of feature names
        n_samples: Number of SHAP samples
        use_timeshap: Whether to try TimeSHAP API first

    Returns:
        Dictionary with:
            - feature_importance: Dict[str, float]
            - event_importance: np.ndarray (per timestep)
            - shap_values: np.ndarray (feature × time)
            - method: str
    """
    if use_timeshap:
        try:
            return _compute_timeshap(model_wrapper, data, feature_names, n_samples)
        except Exception as e:
            print(f"TimeSHAP failed, using perturbation fallback: {e}")

    return _compute_perturbation_shap(model_wrapper, data, feature_names, n_samples)


def _compute_timeshap(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int
) -> Dict[str, Any]:
    """Use TimeSHAP official API for SHAP computation."""
    from timeshap.explainer import local_report

    seq_len = data.shape[1]
    n_features = data.shape[2]

    # Create baseline (mean per feature)
    baseline = np.zeros_like(data)
    for f in range(n_features):
        baseline[0, :, f] = np.mean(data[0, :, f])

    # Prepare DataFrame format for TimeSHAP
    rows = []
    for t in range(seq_len):
        row = {"entity": "seq_0", "t": t}
        for f_idx, f_name in enumerate(feature_names):
            row[f_name] = data[0, t, f_idx]
        rows.append(row)
    data_df = pd.DataFrame(rows)

    baseline_rows = []
    for t in range(seq_len):
        row = {"entity": "baseline", "t": t}
        for f_idx, f_name in enumerate(feature_names):
            row[f_name] = baseline[0, t, f_idx]
        baseline_rows.append(row)
    baseline_df = pd.DataFrame(baseline_rows)

    # Call TimeSHAP
    report = local_report(
        f=model_wrapper,
        data=data_df,
        entity_uuid="seq_0",
        entity_col="entity",
        time_col="t",
        model_features=feature_names,
        baseline=baseline_df,
        pruning_dict={"tol": 0.025},
        event_dict={"rs": 42, "nsamples": n_samples},
        feature_dict={"rs": 42, "nsamples": n_samples},
    )

    # Extract results
    event_df = report[1] if len(report) > 1 else None
    feat_df = report[2] if len(report) > 2 else None
    cell_df = report[3] if len(report) > 3 else None

    # Convert to standard format
    feature_importance = {}
    if feat_df is not None and not feat_df.empty:
        for _, row in feat_df.iterrows():
            feature_importance[row["Feature"]] = abs(float(row["Shapley Value"]))

    event_importance = None
    if event_df is not None and not event_df.empty:
        event_importance = event_df["Shapley Value"].values.astype(float)

    shap_values = None
    if cell_df is not None and not cell_df.empty:
        pivot = cell_df.pivot(index="Feature", columns="t", values="Shapley Value")
        shap_values = pivot.values.astype(float)

    return {
        "feature_importance": feature_importance,
        "event_importance": event_importance,
        "shap_values": shap_values,
        "method": "timeshap",
        "raw_event_df": event_df,
        "raw_feat_df": feat_df,
        "raw_cell_df": cell_df,
    }


def _compute_perturbation_shap(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int
) -> Dict[str, Any]:
    """Fallback SHAP using simple perturbation method."""
    seq_len = data.shape[1]
    n_features = data.shape[2]

    # Baseline prediction
    baseline_pred = model_wrapper(data)[0, 0]

    # Feature means for perturbation
    feature_means = np.mean(data[0], axis=0)

    # Event-level importance (per timestep)
    event_importance = []
    for t in range(seq_len):
        perturbed = data.copy()
        perturbed[0, t, :] = feature_means
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = abs(baseline_pred - perturbed_pred)
        event_importance.append(importance)

    # Feature-level importance
    feature_importance = {}
    for f_idx, f_name in enumerate(feature_names):
        perturbed = data.copy()
        perturbed[0, :, f_idx] = feature_means[f_idx]
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = baseline_pred - perturbed_pred
        feature_importance[f_name] = float(importance)

    # Cell-level SHAP (feature × time) - sampled for speed
    shap_values = np.zeros((n_features, seq_len))
    sample_timesteps = list(range(0, seq_len, max(1, seq_len // 10)))

    for f_idx in range(n_features):
        for t_idx in sample_timesteps:
            perturbed = data.copy()
            perturbed[0, t_idx, f_idx] = feature_means[f_idx]
            perturbed_pred = model_wrapper(perturbed)[0, 0]
            shap_values[f_idx, t_idx] = baseline_pred - perturbed_pred

    return {
        "feature_importance": feature_importance,
        "event_importance": np.array(event_importance),
        "shap_values": shap_values,
        "baseline_pred": float(baseline_pred),
        "method": "perturbation",
    }


def compute_shap_waterfall(
    feature_importance: Dict[str, float],
    base_value: float = 0.0,
    prediction: Optional[float] = None
) -> Dict[str, Any]:
    """
    Prepare data for SHAP waterfall visualization.

    Args:
        feature_importance: Dict mapping feature names to SHAP values (signed)
        base_value: Expected value (E[f(x)])
        prediction: Actual prediction value

    Returns:
        Dictionary with waterfall plot data
    """
    # Sort by absolute value
    sorted_items = sorted(
        feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
    )

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Cumulative values for waterfall
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)

    return {
        "features": features,
        "values": values,
        "cumulative": cumulative,
        "base_value": base_value,
        "prediction": prediction if prediction is not None else cumulative[-1],
    }


def compute_shap_interactions(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    top_k: int = 5
) -> pd.DataFrame:
    """
    Compute SHAP interaction values between top features.

    Uses pairwise perturbation to estimate interactions.

    Args:
        model_wrapper: Model prediction function
        data: Input data (1, seq_len, n_features)
        feature_names: List of feature names
        top_k: Number of top features to compute interactions for

    Returns:
        DataFrame with interaction matrix
    """
    n_features = min(top_k, len(feature_names))
    feature_means = np.mean(data[0], axis=0)

    # First get single feature importance to select top features
    single_importance = {}
    baseline_pred = model_wrapper(data)[0, 0]

    for f_idx, f_name in enumerate(feature_names):
        perturbed = data.copy()
        perturbed[0, :, f_idx] = feature_means[f_idx]
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        single_importance[f_name] = abs(baseline_pred - perturbed_pred)

    # Select top features
    top_features = sorted(
        single_importance.items(), key=lambda x: x[1], reverse=True
    )[:n_features]
    top_names = [f[0] for f in top_features]
    top_indices = [feature_names.index(name) for name in top_names]

    # Compute pairwise interactions
    interactions = np.zeros((n_features, n_features))

    for i, idx_i in enumerate(top_indices):
        for j, idx_j in enumerate(top_indices):
            if i == j:
                interactions[i, j] = single_importance[top_names[i]]
            else:
                # Perturb both features
                perturbed_both = data.copy()
                perturbed_both[0, :, idx_i] = feature_means[idx_i]
                perturbed_both[0, :, idx_j] = feature_means[idx_j]

                # Perturb individually
                perturbed_i = data.copy()
                perturbed_i[0, :, idx_i] = feature_means[idx_i]

                perturbed_j = data.copy()
                perturbed_j[0, :, idx_j] = feature_means[idx_j]

                # Interaction = joint effect - sum of individual effects
                pred_both = model_wrapper(perturbed_both)[0, 0]
                pred_i = model_wrapper(perturbed_i)[0, 0]
                pred_j = model_wrapper(perturbed_j)[0, 0]

                effect_both = baseline_pred - pred_both
                effect_i = baseline_pred - pred_i
                effect_j = baseline_pred - pred_j

                interactions[i, j] = effect_both - effect_i - effect_j

    return pd.DataFrame(
        interactions, index=top_names, columns=top_names
    )


def compute_lag_importance(
    df: pd.DataFrame,
    target_col: str,
    max_lag: int = 30
) -> Dict[int, float]:
    """
    Calculate autocorrelation for each lag.

    Shows which past days are most correlated with current value.

    Args:
        df: DataFrame with target column
        target_col: Name of target column
        max_lag: Maximum lag to compute

    Returns:
        Dictionary mapping lag -> correlation
    """
    if target_col not in df.columns:
        return {}

    target = df[target_col].values
    lag_importance = {}

    for lag in range(1, min(max_lag + 1, len(target) // 2)):
        try:
            shifted = np.roll(target, lag)
            # Ignore first elements affected by roll
            corr = np.corrcoef(target[lag:], shifted[lag:])[0, 1]
            lag_importance[lag] = abs(corr) if not np.isnan(corr) else 0
        except Exception:
            lag_importance[lag] = 0

    return lag_importance


def compute_residual_analysis(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, Any]:
    """
    Complete residual analysis.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Dictionary with residual statistics
    """
    residuals = actual - predicted

    try:
        from scipy.stats import skew
        skewness = float(skew(residuals))
    except Exception:
        skewness = 0.0

    analysis = {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max_abs": float(np.max(np.abs(residuals))),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "skewness": skewness,
        "is_biased": abs(np.mean(residuals)) > np.std(residuals) * 0.5,
    }

    return analysis
