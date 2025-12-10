"""Wrapper for using TimeSHAP with Darts forecasting models.

Creates a bridge between Darts TimeSeries models and TimeSHAP explainability.
Uses official TimeSHAP API functions for proper explanations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Callable
import warnings
import shap
import sys
import types

# --- SHAP COMPATIBILITY PATCH FOR TIMESHAP ---
# TimeSHAP requires `shap.explainers._kernel.Kernel`, which was moved/renamed in SHAP >=0.43.0.
# We manually alias it to `shap.KernelExplainer` to prevent ImportError.
try:
    if not hasattr(shap, 'explainers'):
        shap.explainers = types.ModuleType('shap.explainers')
    
    if not hasattr(shap.explainers, '_kernel'):
        # Create a fake _kernel module
        _kernel = types.ModuleType('shap.explainers._kernel')
        # Map Kernel to KernelExplainer (modern equivalent)
        from shap import KernelExplainer
        _kernel.Kernel = KernelExplainer
        
        # Inject into shap and sys.modules
        shap.explainers._kernel = _kernel
        sys.modules['shap.explainers._kernel'] = _kernel
except Exception as e:
    # If patching fails, we proceed and let the actual import fail if it must
    print(f"Warning: Failed to patch SHAP for TimeSHAP compatibility: {e}")
# ---------------------------------------------

warnings.filterwarnings('ignore')


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
        self._last_valid_pred = None  # Cache last valid prediction for fallback
        self._debug = False  # Set to True for debugging
        
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
                self._last_valid_pred = pred  # Cache valid prediction
                predictions.append(pred)
            except Exception as e:
                if self._debug:
                    print(f"[DartsWrapper] Prediction {i} failed: {e}")
                # Use last valid prediction as fallback, or 0 if none
                fallback = self._last_valid_pred if self._last_valid_pred is not None else 0.0
                predictions.append(fallback)
        
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
        
        # Covariates - Extend for prediction horizon
        covariates = None
        if n_features > 1:
            cov_vals = x[:, 1:]
            
            # Extend covariates for forecast horizon (repeat last values)
            extended_dates = pd.date_range(start=dates[0], periods=seq_len + self.forecast_horizon, freq='D')
            extended_cov_vals = np.vstack([cov_vals, np.tile(cov_vals[-1:], (self.forecast_horizon, 1))])
            
            cov_df = pd.DataFrame(
                extended_cov_vals, 
                index=extended_dates,
                columns=[f'cov_{j}' for j in range(cov_vals.shape[1])]
            )
            covariates = TimeSeries.from_dataframe(cov_df)
        
        # Predict with proper covariate handling
        pred_kwargs = {'n': self.forecast_horizon, 'series': target_series}
        
        if covariates is not None:
            if getattr(self.model, "_uses_past_covariates", False) or getattr(self.model, "uses_past_covariates", False):
                pred_kwargs['past_covariates'] = covariates
            if getattr(self.model, "_uses_future_covariates", False) or getattr(self.model, "uses_future_covariates", False):
                pred_kwargs['future_covariates'] = covariates
        
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
    
    data = window_df[cols].values.astype(np.float64)
    
    # Reshape to 3D (1 sequence)
    data_3d = data.reshape(1, window_length, -1)
    
    return data_3d, feature_names


def create_baseline(data_3d: np.ndarray) -> np.ndarray:
    """
    Create a baseline for SHAP by using the mean of the data.
    
    Args:
        data_3d: Input data (1, seq_len, n_features)
    
    Returns:
        Baseline array of same shape with mean values
    """
    baseline = np.zeros_like(data_3d)
    for f in range(data_3d.shape[2]):
        baseline[0, :, f] = np.mean(data_3d[0, :, f])
    return baseline


def compute_timeshap_local(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Compute local SHAP explanations using TimeSHAP official API.
    
    Args:
        model_wrapper: Wrapped model callable f(X) -> Y
        data: Input data array (1, seq_len, n_features)
        feature_names: List of feature names
        n_samples: Number of SHAP samples
    
    Returns:
        Dictionary with event_data, feat_data, cell_data DataFrames
    """
    seq_len = data.shape[1]
    n_features = data.shape[2]
    
    # Create baseline
    baseline = create_baseline(data)
    
    # Use official TimeSHAP local_report API
    try:
        from timeshap.explainer import local_report
        
        # Convert numpy array to DataFrame format expected by TimeSHAP
        seq_len = data.shape[1]
        n_features = data.shape[2]
        
        # Create DataFrame with proper format for TimeSHAP
        # TimeSHAP expects: (entity_uuid, timestamp, feature1, feature2, ...)
        rows = []
        for t in range(seq_len):
            row = {'entity': 'seq_0', 't': t}
            for f_idx, f_name in enumerate(feature_names):
                row[f_name] = data[0, t, f_idx]
            rows.append(row)
        
        data_df = pd.DataFrame(rows)
        
        # Baseline DataFrame
        baseline_rows = []
        for t in range(seq_len):
            row = {'entity': 'baseline', 't': t}
            for f_idx, f_name in enumerate(feature_names):
                row[f_name] = baseline[0, t, f_idx]
            baseline_rows.append(row)
        baseline_df = pd.DataFrame(baseline_rows)
        
        # Call local_report
        pruning_dict = {'tol': 0.025}
        event_dict = {'rs': 42, 'nsamples': n_samples}
        feature_dict = {'rs': 42, 'nsamples': n_samples}
        
        report = local_report(
            f=model_wrapper,
            data=data_df,
            entity_uuid='seq_0',
            entity_col='entity',
            time_col='t',
            model_features=feature_names,
            baseline=baseline_df,
            pruning_dict=pruning_dict,
            event_dict=event_dict,
            feature_dict=feature_dict
        )
        
        # Extract DataFrames from report
        # local_report returns tuple of (pruning_df, event_df, feat_df, cell_df)
        if isinstance(report, tuple) and len(report) >= 3:
            event_df = report[1] if len(report) > 1 else None
            feat_df = report[2] if len(report) > 2 else None
            cell_df = report[3] if len(report) > 3 else None
        else:
            event_df, feat_df, cell_df = None, None, None
        
        return {
            'success': True,
            'event_data': event_df,
            'feat_data': feat_df,
            'cell_data': cell_df,
            'feature_names': feature_names,
            'method': 'timeshap_local_report'
        }
        
    except Exception as e:
        print(f"[TimeSHAP] local_report failed: {e}, using fallback")
        import traceback
        traceback.print_exc()
        # Fall back to simple perturbation method
        return compute_shap_perturbation(model_wrapper, data, feature_names, n_samples)


def compute_shap_perturbation(
    model_wrapper: Callable,
    data: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Compute SHAP-like importance using simple perturbation method.
    
    This is a fallback when TimeSHAP official API fails.
    """
    seq_len = data.shape[1]
    n_features = data.shape[2]
    
    # Baseline prediction
    baseline_pred = model_wrapper(data)[0, 0]
    
    # Per-feature baseline means (more accurate than global mean)
    feature_means = np.mean(data[0], axis=0)  # Mean per feature across time
    
    # Event-level: perturb each timestep by replacing with feature means
    event_importance = []
    for t in range(seq_len):
        perturbed = data.copy()
        perturbed[0, t, :] = feature_means  # Replace all features at timestep t with their means
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = abs(baseline_pred - perturbed_pred)
        event_importance.append(importance)
    
    # Feature-level: perturb each feature across all timesteps
    feature_importance = []
    for f in range(n_features):
        perturbed = data.copy()
        perturbed[0, :, f] = feature_means[f]  # Replace feature f with its mean across all timesteps
        perturbed_pred = model_wrapper(perturbed)[0, 0]
        importance = baseline_pred - perturbed_pred  # Signed
        feature_importance.append(importance)
        
    # Cell-level: Skip by default - too slow. Only compute for first 5 features
    cell_rows = []
    top_5_features = min(5, n_features)  # Limit to top 5 features for speed
    if seq_len * top_5_features <= 200:  # Limit computation
        for f_idx, f_name in enumerate(feature_names):
            for t_idx in range(0, seq_len, max(1, seq_len // 10)):  # Sample 10 timesteps only
                t_label = seq_len - t_idx  # t-N days ago
                
                perturbed = data.copy()
                perturbed[0, t_idx, f_idx] = feature_means[f_idx]  # Replace cell with feature mean
                
                perturbed_pred = model_wrapper(perturbed)[0, 0]
                importance = baseline_pred - perturbed_pred
                
                cell_rows.append({
                    'Feature': f_name,
                    't': t_label, 
                    'Shapley Value': float(importance)
                })
    
    # Create DataFrames
    event_df = pd.DataFrame({
        't': list(range(seq_len, 0, -1)),
        'Shapley Value': [float(x) for x in event_importance[::-1]]
    })
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Shapley Value': [float(x) for x in feature_importance]
    })
    
    cell_df = pd.DataFrame(cell_rows) if cell_rows else None
    
    return {
        'success': True,
        'event_data': event_df,
        'feat_data': feat_df,
        'cell_data': cell_df,
        'feature_names': feature_names,
        'baseline_pred': float(baseline_pred),
        'method': 'perturbation_fallback'
    }


# =============================================================================
# PLOTTING FUNCTIONS - Using Plotly for reliability
# =============================================================================

def plot_event_importance(event_data: pd.DataFrame) -> Any:
    """Create Plotly bar chart for event/timestep importance."""
    import plotly.graph_objects as go
    
    if event_data is None or event_data.empty:
        return None
    
    # Handle different column names from TimeSHAP vs fallback
    t_col = 't' if 't' in event_data.columns else event_data.columns[0]
    val_col = 'Shapley Value' if 'Shapley Value' in event_data.columns else event_data.columns[-1]
    
    timesteps = event_data[t_col].values
    shap_values = event_data[val_col].values.astype(float)
    
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
        title="⏱️ Importance of Each Timestep (Days Ago)",
        xaxis_title="Timestep (t-n = n days ago)",
        yaxis_title="Impact on Prediction",
        height=350,
        hovermode='x unified'
    )
    
    return fig


def plot_feature_importance(feat_data: pd.DataFrame) -> Any:
    """Create Plotly horizontal bar chart for feature importance."""
    import plotly.graph_objects as go
    
    if feat_data is None or feat_data.empty:
        return None
    
    # Handle different column names
    feat_col = 'Feature' if 'Feature' in feat_data.columns else feat_data.columns[0]
    val_col = 'Shapley Value' if 'Shapley Value' in feat_data.columns else feat_data.columns[-1]
    
    features = feat_data[feat_col].values
    shap_values = feat_data[val_col].values.astype(float)
    
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
        title=" Feature Contributions",
        xaxis_title="Impact (positive = increases, negative = decreases)",
        yaxis_title="",
        height=max(300, len(features) * 35),
        margin=dict(l=10, r=80, t=50, b=30),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_cell_heatmap(cell_data: pd.DataFrame, top_x_feats: int = 10) -> Any:
    """Create Plotly heatmap for cell-level importance."""
    import plotly.graph_objects as go
    
    if cell_data is None or cell_data.empty or 't' not in cell_data.columns:
        return None
    
    # Pivot data
    pivot_df = cell_data.pivot(index='Feature', columns='t', values='Shapley Value')
    
    # Filter top features by total absolute importance
    feat_importance = cell_data.groupby('Feature')['Shapley Value'].apply(lambda x: x.abs().sum())
    top_feats = feat_importance.nlargest(top_x_feats).index
    pivot_df = pivot_df.loc[pivot_df.index.isin(top_feats)]
    
    timesteps = sorted(pivot_df.columns.tolist())
    z_data = pivot_df.values.astype(float)
    x_labels = [f"t-{int(t)}" for t in timesteps]
    y_labels = pivot_df.index.tolist()
    
    max_abs = max(abs(np.nanmin(z_data)), abs(np.nanmax(z_data)), 0.001)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Impact")
    ))
    
    fig.update_layout(
        title=" Detailed Impact (Feature × Time)",
        xaxis_title="Timestep",
        yaxis_title="Feature",
        height=max(400, len(y_labels) * 30)
    )
    
    return fig


# =============================================================================
# GLOBAL EXPLANATIONS
# =============================================================================

def compute_global_feature_importance(
    model_wrapper: Callable,
    test_df: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
    input_chunk_length: int,
    n_samples: int = 5
) -> pd.DataFrame:
    """
    Compute aggregated global feature importance using TimeSHAP global_report.
    """
    feature_names = [target_col] + [c for c in covariate_cols if c in test_df.columns]
    
    try:
        from timeshap.explainer import global_report
        
        # Prepare multiple sequences for global analysis
        max_start = len(test_df) - input_chunk_length - 1
        if max_start <= 0:
            return pd.DataFrame({'Feature': feature_names, 'Shapley Value': [0.0] * len(feature_names)})
        
        sample_positions = list(range(0, max_start, max(1, max_start // n_samples)))[:n_samples]
        
        # Build multi-sequence DataFrame for TimeSHAP
        all_rows = []
        for seq_idx, start_idx in enumerate(sample_positions):
            window_length = min(input_chunk_length, len(test_df) - start_idx)
            window_df = test_df.iloc[start_idx:start_idx + window_length]
            
            for t, (idx, row) in enumerate(window_df.iterrows()):
                data_row = {'entity': f'seq_{seq_idx}', 't': t}
                for f_name in feature_names:
                    if f_name in window_df.columns:
                        data_row[f_name] = row[f_name]
                all_rows.append(data_row)
        
        data_df = pd.DataFrame(all_rows)
        
        # Create baseline (mean across all data)
        baseline_rows = []
        for t in range(input_chunk_length):
            row = {'entity': 'baseline', 't': t}
            for f_name in feature_names:
                if f_name in test_df.columns:
                    row[f_name] = test_df[f_name].mean()
            baseline_rows.append(row)
        baseline_df = pd.DataFrame(baseline_rows)
        
        # Call global_report
        pruning_dict = {'tol': 0.025}
        event_dict = {'rs': 42, 'nsamples': 50}
        feature_dict = {'rs': 42, 'nsamples': 50}
        
        report = global_report(
            f=model_wrapper,
            data=data_df,
            entity_col='entity',
            time_col='t',
            model_features=feature_names,
            baseline=baseline_df,
            pruning_dict=pruning_dict,
            event_dict=event_dict,
            feature_dict=feature_dict
        )
        
        # Extract global feature importance from report
        # global_report returns (global_event_df, global_feat_df)
        if isinstance(report, tuple) and len(report) >= 2:
            global_feat_df = report[1]
            if global_feat_df is not None and not global_feat_df.empty:
                return global_feat_df
        
        # Fallback to aggregation
        raise Exception("global_report did not return expected format")
        
    except Exception as e:
        print(f"[TimeSHAP] global_report failed: {e}, using fallback aggregation")
        import traceback
        traceback.print_exc()
        
        # Fallback: Aggregate local SHAP values
        all_importances = {feat: [] for feat in feature_names}
        
        max_start = len(test_df) - input_chunk_length - 1
        if max_start <= 0:
            return pd.DataFrame({'Feature': feature_names, 'Shapley Value': [0.0] * len(feature_names)})
        
        sample_positions = list(range(0, max_start, max(1, max_start // n_samples)))[:n_samples]
        
        for start_idx in sample_positions:
            try:
                window_data, feat_names = prepare_timeshap_data(
                    test_df, target_col, covariate_cols,
                    window_start=start_idx,
                    window_length=min(input_chunk_length, len(test_df) - start_idx)
                )
                
                result = compute_shap_perturbation(model_wrapper, window_data, feat_names, n_samples=10)
                
                if result['success']:
                    feat_df = result['feat_data']
                    for _, row in feat_df.iterrows():
                        feat_name = row['Feature']
                        if feat_name in all_importances:
                            all_importances[feat_name].append(abs(float(row['Shapley Value'])))
            except:
                continue
        
        global_importance = {}
        for feat, values in all_importances.items():
            global_importance[feat] = float(np.mean(values)) if values else 0.0
        
        return pd.DataFrame({
            'Feature': list(global_importance.keys()),
            'Shapley Value': list(global_importance.values())
        }).sort_values('Shapley Value', ascending=False).reset_index(drop=True)


# =============================================================================
# LEGACY COMPATIBILITY ALIASES
# =============================================================================

def compute_timeshap_simple(*args, **kwargs):
    """Legacy alias for compute_timeshap_local."""
    return compute_timeshap_local(*args, **kwargs)

def plot_event_heatmap_timeshap(event_data, **kwargs):
    """Legacy alias for plot_event_importance."""
    return plot_event_importance(event_data)

def plot_feat_barplot_timeshap(feat_data, **kwargs):
    """Legacy alias for plot_feature_importance."""
    return plot_feature_importance(feat_data)

def plot_cell_level_timeshap(cell_data, **kwargs):
    """Legacy alias for plot_cell_heatmap."""
    return plot_cell_heatmap(cell_data, **kwargs)

def plot_global_feat_timeshap(feat_data, **kwargs):
    """Legacy alias for plot_feature_importance."""
    return plot_feature_importance(feat_data)

