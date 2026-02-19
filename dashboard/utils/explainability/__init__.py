"""Explainability package for time series forecasting models.

This package provides comprehensive explainability tools for Darts time series models:
- Feature importance (correlation, permutation, SHAP)
- Gradient-based explanations (Captum: Saliency, Integrated Gradients)
- Attention visualization (TFT, Transformers)
- Decomposition analysis (STL, seasonality detection)
- Model-specific explainers (TFT, TSMixer, NHiTS, NBEATS)
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Base classes
from .base import (
    BaseExplainer,
    ExplainabilityResult,
    ModelType,
)

# Feature importance
from .feature_importance import (
    compute_correlation_importance,
    compute_permutation_importance,
    compute_permutation_importance as compute_permutation_importance_safe,
    compute_shap_importance,
    compute_shap_waterfall,
    compute_shap_interactions,
    compute_lag_importance,
    compute_residual_analysis,
)

# Gradient-based explanations
from .gradients import (
    compute_temporal_saliency,
    compute_integrated_gradients,
    compute_gradient_attributions,
    GradientExplainer,
    CAPTUM_AVAILABLE,
)

# Attention mechanisms
from .attention import (
    extract_tft_attention,
    TFTExplainer,
)

# Decomposition and seasonality
from .decomposition import (
    analyze_prediction_decomposition,
    detect_seasonality_patterns,
    compare_decompositions,
    DecompositionAnalyzer,
)

# Model-specific explainers
from .model_specific import (
    ModelExplainerFactory,
    GenericExplainer,
    TFTModelExplainer,
    TSMixerModelExplainer,
    NHiTSModelExplainer,
    NBEATSModelExplainer,
)

# Visualizations
from .visualizations import (
    plot_feature_importance_bar,
    plot_temporal_saliency_heatmap,
    plot_attention_heatmap,
    plot_shap_waterfall,
    plot_shap_force,
    plot_decomposition_comparison,
    plot_lag_importance,
    plot_residual_analysis,
    plot_seasonality_patterns,
    plot_prediction_vs_actual,
)


# =============================================================================
# LEGACY/COMPATIBILITY FUNCTIONS (from old explainability.py)
# =============================================================================

def compute_captum_attributions(
    model,
    input_tensor,
    method: str = 'integrated_gradients'
) -> Optional[np.ndarray]:
    """Legacy function - use compute_gradient_attributions instead."""
    warnings.warn(
        "compute_captum_attributions is deprecated, use compute_gradient_attributions instead",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        import torch
        from captum.attr import IntegratedGradients, Saliency, DeepLift

        model.eval()

        if method == 'integrated_gradients':
            attr_method = IntegratedGradients(model)
        elif method == 'saliency':
            attr_method = Saliency(model)
        elif method == 'deeplift':
            attr_method = DeepLift(model)
        else:
            attr_method = IntegratedGradients(model)

        attributions = attr_method.attribute(input_tensor)

        return attributions.detach().cpu().numpy()
    except Exception:
        return None


def plot_residual_histogram(residuals: np.ndarray):
    """Legacy function - use plot_residual_analysis for combined view."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color='rgba(31, 119, 180, 0.7)'))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=np.mean(residuals), line_dash="dot", line_color="green")
    fig.update_layout(title="Residual Distribution", height=300)
    return fig


def plot_residual_timeline(dates: pd.DatetimeIndex, residuals: np.ndarray):
    """Legacy function - use plot_residual_analysis for combined view."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode='lines', line=dict(color='#2E86AB')))
    std = np.std(residuals)
    fig.add_hrect(y0=-std, y1=std, fillcolor="rgba(0, 255, 0, 0.1)", layer="below", line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residuals over Time", height=300)
    return fig


def generate_explanation_summary(
    correlation_importance: Dict[str, float],
    lag_importance: Dict[int, float],
    residual_analysis: Dict[str, Any],
    input_chunk_length: int
) -> str:
    """Generate text summary of explainability analysis."""
    summary_parts = []

    if correlation_importance:
        top_features = sorted(correlation_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        top_names = ", ".join([f"**{name}** ({val:.1%})" for name, val in top_features])
        summary_parts.append(f"Most correlated features: {top_names}")

    if lag_importance:
        peak_lag = max(lag_importance.keys(), key=lambda k: lag_importance[k])
        peak_value = lag_importance[peak_lag]
        summary_parts.append(f"Most important lag: t-{peak_lag} (correlation: {peak_value:.2f})")

        recent_lags = [v for k, v in lag_importance.items() if k <= 7]
        distant_lags = [v for k, v in lag_importance.items() if k > 7]

        if recent_lags and distant_lags:
            if np.mean(recent_lags) > np.mean(distant_lags) * 1.3:
                summary_parts.append("The model relies mainly on **recent data**")
            elif np.mean(distant_lags) > np.mean(recent_lags) * 1.3:
                summary_parts.append("The model heavily uses **long-term history**")

    if residual_analysis:
        if residual_analysis.get('is_biased', False):
            bias = residual_analysis['mean']
            if bias > 0:
                summary_parts.append(f"**Bias detected**: under-estimates by {abs(bias):.3f}")
            else:
                summary_parts.append(f"**Bias detected**: over-estimates by {abs(bias):.3f}")
        else:
            summary_parts.append("**Balanced residuals** - no systematic bias")

    return "\n\n".join(summary_parts) if summary_parts else "Analysis not available."


def compute_feature_importance_permutation(
    model, series, past_covariates, covariate_cols: List[str], n_permutations: int = 5
) -> Dict[str, float]:
    """Compute permutation-based feature importance (wrapper for compatibility)."""
    return compute_permutation_importance(
        model=model, series=series, covariates=past_covariates,
        n_permutations=n_permutations,
        output_chunk_length=getattr(model, 'output_chunk_length', 7)
    )


def compute_temporal_importance(
    model, series, past_covariates, n_steps: int = 30
) -> Optional[np.ndarray]:
    """Compute temporal importance (wrapper for compatibility)."""
    result = compute_temporal_saliency(
        model=model, series=series, past_covariates=past_covariates,
        input_chunk_length=n_steps
    )
    if result is not None:
        return result

    # Fallback to lag importance
    try:
        target_df = series.to_dataframe()
        target_col = target_df.columns[0]
        lag_imp = compute_lag_importance(target_df, target_col, max_lag=n_steps)
        if lag_imp:
            return np.array([lag_imp.get(i, 0) for i in range(1, n_steps + 1)])
    except Exception:
        pass
    return None


def compute_local_explanation(
    model, series, past_covariates, selected_date, covariate_cols: List[str]
) -> Optional[Dict[str, float]]:
    """Compute local explanation for a specific date (wrapper for compatibility)."""
    try:
        explainer = ModelExplainerFactory.get_explainer(
            model,
            input_chunk_length=getattr(model, 'input_chunk_length', 30),
            output_chunk_length=getattr(model, 'output_chunk_length', 7)
        )
        result = explainer.explain_local(series, past_covariates)
        if result.success and result.feature_importance:
            return result.feature_importance
    except Exception:
        pass
    return None


def plot_feature_importance(importance: Dict[str, float]):
    """Alias for plot_feature_importance_bar."""
    return plot_feature_importance_bar(importance, title="Feature Importance")


def plot_temporal_importance(temporal_imp: np.ndarray):
    """Plot temporal importance array."""
    if temporal_imp is None:
        return None
    lag_dict = {i + 1: float(v) for i, v in enumerate(temporal_imp)}
    return plot_lag_importance(lag_dict, input_chunk_length=len(temporal_imp))


def plot_waterfall_explanation(local_exp: Dict[str, float], selected_date):
    """Alias for plot_shap_waterfall."""
    return plot_shap_waterfall(local_exp, title=f"Feature Contributions ({selected_date})")


def create_explanation_summary(importance: Dict[str, float]) -> str:
    """Create simple explanation summary from importance dict."""
    if not importance:
        return "No importance data available."
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    summary_parts = ["**Top Features:**"]
    for name, val in sorted_imp:
        summary_parts.append(f"- {name}: {val:.1%}")
    return "\n".join(summary_parts)


__all__ = [
    # Base
    "BaseExplainer",
    "ExplainabilityResult",
    "ModelType",
    # Feature importance
    "compute_correlation_importance",
    "compute_permutation_importance",
    "compute_permutation_importance_safe",
    "compute_shap_importance",
    "compute_shap_waterfall",
    "compute_shap_interactions",
    "compute_lag_importance",
    "compute_residual_analysis",
    # Gradients
    "compute_temporal_saliency",
    "compute_integrated_gradients",
    "compute_gradient_attributions",
    "compute_captum_attributions",
    "GradientExplainer",
    "CAPTUM_AVAILABLE",
    # Attention
    "extract_tft_attention",
    "TFTExplainer",
    # Decomposition
    "analyze_prediction_decomposition",
    "detect_seasonality_patterns",
    "compare_decompositions",
    "DecompositionAnalyzer",
    # Model-specific
    "ModelExplainerFactory",
    "GenericExplainer",
    "TFTModelExplainer",
    "TSMixerModelExplainer",
    "NHiTSModelExplainer",
    "NBEATSModelExplainer",
    # Visualizations
    "plot_feature_importance_bar",
    "plot_temporal_saliency_heatmap",
    "plot_attention_heatmap",
    "plot_shap_waterfall",
    "plot_shap_force",
    "plot_decomposition_comparison",
    "plot_lag_importance",
    "plot_residual_analysis",
    "plot_seasonality_patterns",
    "plot_prediction_vs_actual",
    # Legacy/Compatibility
    "plot_residual_histogram",
    "plot_residual_timeline",
    "generate_explanation_summary",
    "compute_feature_importance_permutation",
    "compute_temporal_importance",
    "compute_local_explanation",
    "plot_feature_importance",
    "plot_temporal_importance",
    "plot_waterfall_explanation",
    "create_explanation_summary",
]
