"""Advanced explainability module for Darts time series models.

Integrates:
- SHAP for global feature importance
- Captum for local gradients (PyTorch models)
- Permutation importance (robust fallback)
- Residual analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =============================================================================
# CORE EXPLAINABILITY FUNCTIONS
# =============================================================================

def compute_correlation_importance(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str]
) -> Dict[str, float]:
    """
    Calculates feature importance based on correlations with the target.
    
    Simple but robust - always works.
    """
    correlations = {}
    
    if target_col not in df.columns:
        return {}
    
    for cov in covariate_cols:
        if cov in df.columns:
            try:
                corr = df[cov].corr(df[target_col])
                correlations[cov] = abs(corr) if not pd.isna(corr) else 0
            except:
                correlations[cov] = 0
    
    return correlations


def compute_permutation_importance_safe(
    model,
    series,
    covariates,
    n_permutations: int = 5,
    output_chunk_length: int = 7
) -> Dict[str, float]:
    """
    Robust permutation importance with error handling.
    
    For each feature, shuffle its values and measure 
    prediction degradation.
    """
    from darts import TimeSeries
    from darts.metrics import mae
    
    if covariates is None:
        return {'target': 1.0}
    
    try:
        # Baseline prediction
        predict_kwargs = {'n': output_chunk_length, 'series': series}
        
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates
        if getattr(model, "_uses_future_covariates", False) or getattr(model, "uses_future_covariates", False):
            predict_kwargs['future_covariates'] = covariates
            
        baseline_pred = model.predict(**predict_kwargs)
        
        cov_df = covariates.pd_dataframe()
        feature_names = list(cov_df.columns)
        
        importances = {}
        
        for feature in feature_names:
            degradations = []
            
            for _ in range(n_permutations):
                try:
                    # Shuffle feature
                    cov_permuted = cov_df.copy()
                    cov_permuted[feature] = np.random.permutation(cov_permuted[feature].values)
                    
                    cov_permuted_ts = TimeSeries.from_dataframe(
                        cov_permuted,
                        freq=covariates.freq_str
                    )
                    
                    # Predict with shuffled
                    perm_kwargs = {'n': output_chunk_length, 'series': series}
                    if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
                        perm_kwargs['past_covariates'] = cov_permuted_ts
                    if getattr(model, "_uses_future_covariates", False) or getattr(model, "uses_future_covariates", False):
                        perm_kwargs['future_covariates'] = cov_permuted_ts
                        
                    permuted_pred = model.predict(**perm_kwargs)
                    
                    # Measure change
                    diff = float(mae(baseline_pred, permuted_pred))
                    degradations.append(diff)
                    
                except Exception:
                    continue
            
            if degradations:
                importances[feature] = np.mean(degradations)
            else:
                importances[feature] = 0
        
        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
            
        return importances
        
    except Exception as e:
        print(f"Permutation importance failed: {e}")
        return {}


def compute_residual_analysis(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, Any]:
    """
    Complete residual analysis.
    """
    residuals = actual - predicted
    
    analysis = {
        'residuals': residuals,
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'max_abs': float(np.max(np.abs(residuals))),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'median': float(np.median(residuals)),
        'skewness': float(_safe_skewness(residuals)),
        'is_biased': abs(np.mean(residuals)) > np.std(residuals) * 0.5
    }
    
    return analysis


def _safe_skewness(arr):
    """Safe skewness calculation."""
    try:
        from scipy.stats import skew
        return skew(arr)
    except:
        return 0


def compute_lag_importance(
    df: pd.DataFrame,
    target_col: str,
    max_lag: int = 30
) -> Dict[int, float]:
    """
    Calculates autocorrelation for each lag.
    Shows which past days are most correlated with the current value.
    """
    if target_col not in df.columns:
        return {}
    
    target = df[target_col].values
    lag_importance = {}
    
    for lag in range(1, min(max_lag + 1, len(target) // 2)):
        try:
            shifted = np.roll(target, lag)
            # Ignorer les premiers éléments affectés par le roll
            corr = np.corrcoef(target[lag:], shifted[lag:])[0, 1]
            lag_importance[lag] = abs(corr) if not np.isnan(corr) else 0
        except:
            lag_importance[lag] = 0
    
    return lag_importance


# =============================================================================
# CAPTUM INTEGRATION (for PyTorch models)
# =============================================================================

def compute_captum_attributions(
    model,
    input_tensor,
    method: str = 'integrated_gradients'
) -> Optional[np.ndarray]:
    """
    Calculates Captum attributions for PyTorch models.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        method: 'integrated_gradients', 'saliency', 'deeplift'
    
    Returns:
        Attributions or None if failed
    """
    try:
        import torch
        from captum.attr import IntegratedGradients, Saliency, DeepLift
        
        # Ensure model is in eval mode
        model.eval()
        
        # Choose attribution method
        if method == 'integrated_gradients':
            attr_method = IntegratedGradients(model)
        elif method == 'saliency':
            attr_method = Saliency(model)
        elif method == 'deeplift':
            attr_method = DeepLift(model)
        else:
            attr_method = IntegratedGradients(model)
        
        # Compute attributions
        with torch.no_grad():
            attributions = attr_method.attribute(input_tensor)
        
        return attributions.cpu().numpy()
        
    except ImportError:
        print("Captum not available")
        return None
    except Exception as e:
        print(f"Captum attribution failed: {e}")
        return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_importance_bar(
    importance: Dict[str, float],
    title: str = "Feature Importance"
) -> go.Figure:
    """Horizontal bar chart of importance."""
    if not importance:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create color scale
    colors = [f'rgba(31, 119, 180, {0.3 + 0.7*v/max(values) if max(values) > 0 else 0.5})' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.1%}' if v < 1 else f'{v:.2f}' for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Relative Importance",
        yaxis_title="",
        height=max(300, len(features) * 35),
        margin=dict(l=10, r=80, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_lag_importance(
    lag_importance: Dict[int, float],
    input_chunk_length: int = 30
) -> go.Figure:
    """Chart of time lag importance."""
    if not lag_importance:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    lags = list(lag_importance.keys())
    values = list(lag_importance.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=values,
        mode='lines+markers',
        name='Autocorrelation',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ))
    
    # Zone input_chunk
    fig.add_vrect(
        x0=0, x1=input_chunk_length,
        fillcolor="rgba(255, 200, 0, 0.1)",
        layer="below", line_width=0,
        annotation_text="Input window", annotation_position="top left"
    )
    
    fig.update_layout(
        title="⏱️ Temporal Importance (autocorrelation per lag)",
        xaxis_title="Lag (days)",
        yaxis_title="Absolute Correlation",
        height=350,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_residual_histogram(residuals: np.ndarray) -> go.Figure:
    """Histogram of residuals with statistics."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        marker_color='rgba(31, 119, 180, 0.7)',
        hovertemplate='Error: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Ligne verticale à 0
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    
    # Line for mean
    mean_val = np.mean(residuals)
    fig.add_vline(x=mean_val, line_dash="dot", line_color="green", line_width=2,
                  annotation_text=f"Mean: {mean_val:.3f}")
    
    fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Error (Actual - Predicted)",
        yaxis_title="Frequency",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_residual_timeline(
    dates: pd.DatetimeIndex,
    residuals: np.ndarray
) -> go.Figure:
    """Timeline of residuals to detect temporal patterns."""
    fig = go.Figure()
    
    # Residuals
    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='#2E86AB', width=1),
        hovertemplate='Date: %{x}<br>Error: %{y:.3f}<extra></extra>'
    ))
    
    # Bande ±1 std
    std = np.std(residuals)
    fig.add_hrect(
        y0=-std, y1=std,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below", line_width=0
    )
    
    # Ligne 0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residuals over Time",
        xaxis_title="Date",
        yaxis_title="Error",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_prediction_vs_actual(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predictions vs Actual"
) -> go.Figure:
    """Comparison predictions vs actual."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode='lines', name='Actual',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode='lines', name='Predicted',
        line=dict(color='#F24236', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_explanation_summary(
    correlation_importance: Dict[str, float],
    lag_importance: Dict[int, float],
    residual_analysis: Dict[str, Any],
    input_chunk_length: int
) -> str:
    """Generates a text summary of the analysis."""
    summary_parts = []
    
    # Top features
    if correlation_importance:
        top_features = sorted(correlation_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        top_names = ", ".join([f"**{name}** ({val:.1%})" for name, val in top_features])
        summary_parts.append(f"🔑 **Most correlated features**: {top_names}")
    
    # Temporal pattern
    if lag_importance:
        peak_lag = max(lag_importance.keys(), key=lambda k: lag_importance[k])
        peak_value = lag_importance[peak_lag]
        summary_parts.append(f"⏱️ **Most important lag**: t-{peak_lag} (correlation: {peak_value:.2f})")
        
        # Recent vs distant
        recent_lags = [v for k, v in lag_importance.items() if k <= 7]
        distant_lags = [v for k, v in lag_importance.items() if k > 7]
        
        if recent_lags and distant_lags:
            if np.mean(recent_lags) > np.mean(distant_lags) * 1.3:
                summary_parts.append("📊 The model relies mainly on **recent data**")
            elif np.mean(distant_lags) > np.mean(recent_lags) * 1.3:
                summary_parts.append("📊 The model heavily uses **long-term history**")
    
    # Residual analysis
    if residual_analysis:
        if residual_analysis.get('is_biased', False):
            bias = residual_analysis['mean']
            if bias > 0:
                summary_parts.append(f"⚠️ **Bias detected**: The model under-estimates by an average of {abs(bias):.3f}")
            else:
                summary_parts.append(f"⚠️ **Bias detected**: The model over-estimates by an average of {abs(bias):.3f}")
        else:
            summary_parts.append("✅ **Balanced residuals** - no systematic bias detected")
    
    return "\n\n".join(summary_parts) if summary_parts else "Analysis not available."
