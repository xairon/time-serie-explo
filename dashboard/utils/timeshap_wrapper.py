"""Wrapper for using TimeSHAP with Darts forecasting models.

Creates a bridge between Darts TimeSeries models and TimeSHAP explainability.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Import compatibility layer first
from dashboard.utils.timeshap_compat import import_timeshap


class DartsModelWrapper:
    """
    Wraps a Darts forecasting model to be compatible with TimeSHAP.
    
    TimeSHAP expects a function f(X) -> Y where:
    - X is a 3D numpy array: (n_sequences, sequence_length, n_features)
    - Y is a 2D numpy array: (n_sequences, 1) with predictions
    """
    
    def __init__(
        self,
        model,
        input_chunk_length: int = 30,
        forecast_horizon: int = 1
    ):
        self.model = model
        self.forecast_horizon = forecast_horizon
        self.input_chunk_length = input_chunk_length
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions compatible with TimeSHAP.
        
        Args:
            X: 3D array (n_sequences, seq_len, n_features)
        
        Returns:
            2D array (n_sequences, 1) with predictions
        """
        n_sequences = X.shape[0]
        predictions = []
        
        for i in range(n_sequences):
            try:
                pred = self._predict_single(X[i])
                predictions.append(pred)
            except:
                predictions.append(0.0)
        
        return np.array(predictions).reshape(-1, 1)
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Predict for a single sequence."""
        from darts import TimeSeries
        
        seq_len, n_features = x.shape
        
        # First column is target
        target_vals = x[:, 0:1]
        dates = pd.date_range(end='2020-12-31', periods=seq_len, freq='D')
        target_df = pd.DataFrame(target_vals, index=dates, columns=['target'])
        target_series = TimeSeries.from_dataframe(target_df)
        
        # Covariates
        covariates = None
        if n_features > 1:
            cov_vals = x[:, 1:]
            cov_df = pd.DataFrame(cov_vals, index=dates, 
                                   columns=[f'cov_{j}' for j in range(cov_vals.shape[1])])
            covariates = TimeSeries.from_dataframe(cov_df)
        
        # Predict
        pred_kwargs = {'n': self.forecast_horizon, 'series': target_series}
        
        if covariates is not None:
            if getattr(self.model, "uses_past_covariates", False):
                pred_kwargs['past_covariates'] = covariates
        
        pred = self.model.predict(**pred_kwargs)
        return float(pred.values()[0, 0])


def prepare_timeshap_data(
    df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    window_start: int,
    window_length: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare data for TimeSHAP analysis from DataFrame.
    
    Returns:
        - 3D array (1, window_length, n_features) 
        - List of feature names
    """
    # Extract window
    window_df = df.iloc[window_start:window_start + window_length]
    
    # Target first, then covariates
    feature_names = [target_col]
    cols = [target_col]
    
    for cov in covariate_cols:
        if cov in window_df.columns:
            feature_names.append(cov)
            cols.append(cov)
    
    data = window_df[cols].values
    
    # Reshape to 3D (1 sequence)
    data_3d = data.reshape(1, window_length, -1)
    
    return data_3d, feature_names


def compute_timeshap_simple(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Compute SHAP-like importance using simple perturbation method.
    
    This is a simpler fallback that doesn't require full TimeSHAP API.
    """
    seq_len = data.shape[1]
    n_features = data.shape[2]
    
    # Baseline prediction
    baseline_pred = model_wrapper(data)[0, 0]
    
    # Event-level: perturb each timestep
    event_importance = []
    baseline_mean = np.mean(data)
    
    for t in range(seq_len):
        perturbed = data.copy()
        perturbed[0, t, :] = baseline_mean  # Mask timestep t
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = abs(baseline_pred - perturbed_pred)
        event_importance.append(importance)
    
    # Feature-level: perturb each feature across all timesteps
    feature_importance = []
    for f in range(n_features):
        perturbed = data.copy()
        perturbed[0, :, f] = baseline_mean  # Mask feature f
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = baseline_pred - perturbed_pred  # Signed
        feature_importance.append(importance)
    
    # Create DataFrames
    event_df = pd.DataFrame({
        't': list(range(seq_len, 0, -1)),  # t-1, t-2, ...
        'Shapley Value': event_importance[::-1]  # Reverse to match t-n order
    })
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Shapley Value': feature_importance
    })
    
    return {
        'success': True,
        'event_data': event_df,
        'feat_data': feat_df,
        'feature_names': feature_names,
        'baseline_pred': baseline_pred
    }


def plot_timeshap_event(event_data: pd.DataFrame) -> Any:
    """Create interactive Plotly chart for event-level importance."""
    import plotly.graph_objects as go
    
    timesteps = event_data['t'].values
    shap_values = event_data['Shapley Value'].values
    
    # Normalize for color scale
    max_abs = max(abs(shap_values.min()), abs(shap_values.max()), 0.001)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f't-{int(t)}' for t in timesteps],
        y=shap_values,
        marker=dict(
            color=shap_values,
            colorscale='RdBu_r',
            cmin=-max_abs,
            cmax=max_abs,
            showscale=True,
            colorbar=dict(title="Impact")
        ),
        hovertemplate='<b>%{x}</b><br>Impact: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="⏱️ Importance of Each Timestep",
        xaxis_title="Timestep (t-n = n days ago)",
        yaxis_title="Impact on Prediction",
        height=350,
        hovermode='x unified'
    )
    
    return fig


def plot_timeshap_feature(feat_data: pd.DataFrame, feature_names: List[str]) -> Any:
    """Create interactive Plotly bar chart for feature-level importance."""
    import plotly.graph_objects as go
    
    features = feat_data['Feature'].values
    shap_values = feat_data['Shapley Value'].values
    
    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    values_sorted = shap_values[sorted_idx]
    
    # Colors based on positive/negative
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values_sorted]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features_sorted,
        x=values_sorted,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.4f}' for v in values_sorted],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🎯 Contribution of Each Feature",
        xaxis_title="Impact (positive = increases, negative = decreases)",
        yaxis_title="",
        height=max(300, len(features) * 35),
        margin=dict(l=10, r=80, t=50, b=30),
        yaxis=dict(autorange="reversed")
    )
    
    return fig
