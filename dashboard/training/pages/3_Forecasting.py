"""Forecasting Page - Sliding Window on TEST data.

This page allows users to:
1. Load a trained model.
2. Visualize one-step predictions on the test set.
3. Analyze model explainability locally and globally using TimeSHAP.
"""

import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import streamlit as st
import shap

# --- SHAP COMPATIBILITY PATCH FOR TIMESHAP ---
# TimeSHAP requires `shap.explainers._kernel.Kernel`, which was moved/renamed in SHAP >=0.43.0.
try:
    from shap import KernelExplainer
    import sys
    import types

    # 1. Try to get existing module or create new one
    try:
        from shap.explainers import _kernel
    except ImportError:
        _kernel = types.ModuleType("shap.explainers._kernel")
        sys.modules["shap.explainers._kernel"] = _kernel
        if hasattr(shap, "explainers"):
            shap.explainers._kernel = _kernel

    # 2. Inject Kernel attribute
    if not hasattr(_kernel, "Kernel"):
        _kernel.Kernel = KernelExplainer
        
    # 3. Double check sys.modules entry
    if "shap.explainers._kernel" not in sys.modules:
        sys.modules["shap.explainers._kernel"] = _kernel
        
except Exception as e:
    print(f"Warning: Failed to patch SHAP for TimeSHAP compatibility: {e}")
# ---------------------------------------------

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.model_registry import get_registry
from dashboard.utils.model_config import load_model_with_config, load_scalers
from dashboard.utils.forecasting import generate_single_window_forecast
from dashboard.utils.preprocessing import prepare_dataframe_for_darts
from darts.metrics import mae, rmse, mape, smape

# Nash-Sutcliffe Efficiency (NSE) - standard metric for hydrology
def nash_sutcliffe_efficiency(actual, predicted):
    """Calculate Nash-Sutcliffe Efficiency (NSE) - standard hydrology metric.
    
    NSE = 1 means perfect prediction
    NSE = 0 means prediction is as good as using the mean
    NSE < 0 means prediction is worse than using the mean
    """
    import numpy as np
    actual_vals = actual.values().flatten()
    pred_vals = predicted.values().flatten()
    
    min_len = min(len(actual_vals), len(pred_vals))
    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]
    
    mean_obs = np.mean(actual_vals)
    ss_res = np.sum((actual_vals - pred_vals) ** 2)
    ss_tot = np.sum((actual_vals - mean_obs) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    
    return 1 - (ss_res / ss_tot)


# Kling-Gupta Efficiency (KGE) - improved hydrology metric
def kling_gupta_efficiency(actual, predicted):
    """Calculate Kling-Gupta Efficiency (KGE) - improved hydrology metric.
    
    KGE decomposes the error into:
    - Correlation (r)
    - Bias ratio (beta)  
    - Variability ratio (gamma)
    
    KGE = 1 means perfect prediction
    KGE > -0.41 is considered useful (better than mean)
    """
    import numpy as np
    actual_vals = actual.values().flatten()
    pred_vals = predicted.values().flatten()
    
    min_len = min(len(actual_vals), len(pred_vals))
    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]
    
    # Correlation coefficient
    if np.std(actual_vals) == 0 or np.std(pred_vals) == 0:
        r = 0.0
    else:
        r = np.corrcoef(actual_vals, pred_vals)[0, 1]
    
    # Bias ratio (mean ratio)
    mean_obs = np.mean(actual_vals)
    mean_pred = np.mean(pred_vals)
    if mean_obs == 0:
        beta = 1.0 if mean_pred == 0 else np.inf
    else:
        beta = mean_pred / mean_obs
    
    # Variability ratio (std ratio)
    std_obs = np.std(actual_vals)
    std_pred = np.std(pred_vals)
    if std_obs == 0:
        gamma = 1.0 if std_pred == 0 else np.inf
    else:
        gamma = std_pred / std_obs
    
    # KGE formula
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return kge


st.set_page_config(layout="wide", page_title="Forecasting")


def load_model_data(model_entry):
    """Loads model, config, data, and scalers with caching."""
    cache_key = f"model_{model_entry.model_id}"
    if cache_key not in st.session_state:
        # Get full model path
        registry = get_registry(CHECKPOINTS_DIR.parent)  # checkpoints/
        model_dir = registry.checkpoints_dir / model_entry.path
        
        # Load fresh if not in cache
        model, config, data_dict = load_model_with_config(model_dir)
        scalers = load_scalers(model_dir)
        st.session_state[cache_key] = {
            'model': model,
            'config': config,
            'data_dict': data_dict,
            'scalers': scalers
        }
    return st.session_state[cache_key]


# =============================================================================
# SIDEBAR: MODEL SELECTION (Station → Dataset → Model)
# =============================================================================
st.sidebar.header("Model Selection")

# Get registry and scan for any unregistered models
registry = get_registry(CHECKPOINTS_DIR.parent)  # checkpoints/ is parent
registry.scan_existing_checkpoints()  # Auto-register legacy models

all_models = registry.list_all_models()

if not all_models:
    st.warning("No trained models available.")
    st.info("Please train a model first on the **Train Models** page.")
    st.stop()

# Helper function to detect if a station name is a real BSS code or just a filename
def is_real_station_code(station_name: str) -> bool:
    """
    Detect if the station name is a real BSS code (format like 00104X0054/P1)
    or just a derived filename (like 'piezo3' from 'piezo3.csv').
    """
    import re
    # BSS code pattern: digits + letter + digits + / + alphanumeric
    # Examples: 00104X0054/P1, 01234Y5678/F2
    bss_pattern = r'^\d{5}[A-Z]\d{4}/[A-Z0-9]+$'
    return bool(re.match(bss_pattern, station_name))

# 1. Select Data Source / Station
all_stations = registry.get_all_stations()

if not all_stations:
    st.warning("No stations found in registered models.")
    st.stop()

# Check if any station is a real BSS code
has_real_stations = any(is_real_station_code(s) for s in all_stations)

# Determine label based on whether we have real station codes
if has_real_stations:
    st.sidebar.markdown("### 1. Station")
    station_label = "Station"
else:
    st.sidebar.markdown("### 1. Source de données")
    station_label = "Source"

selected_station = st.sidebar.selectbox(
    station_label, 
    sorted(all_stations),
    label_visibility="collapsed"
)

# 2. Select Dataset (preprocessing config)
st.sidebar.markdown("### 2. Dataset")
datasets_dict = registry.get_datasets_for_station(selected_station)

if not datasets_dict:
    st.sidebar.warning(f"No datasets for station {selected_station}")
    st.stop()

# Simple dropdown with just dataset IDs
dataset_ids = list(datasets_dict.keys())
selected_dataset = st.sidebar.selectbox(
    "Dataset",
    dataset_ids,
    label_visibility="collapsed"
)

# Get dataset details from a model with this dataset
dataset_models = registry.get_models_for_station_dataset(selected_station, selected_dataset)
if dataset_models:
    sample_model = dataset_models[0]
    preproc = sample_model.preprocessing_config or {}
    columns = preproc.get('columns', {})
    
    # Show details expander (we'll populate with actual config later)
    with st.sidebar.expander("Dataset Details", expanded=False):
        st.markdown(f"**Source:** {sample_model.data_source or 'N/A'}")
        st.markdown(f"**Scaler:** {preproc.get('scaler_type', 'N/A')}")
        if columns.get('target'):
            st.markdown(f"**Target:** {columns.get('target')}")
        if columns.get('covariates'):
            st.markdown(f"**Covariates:** {', '.join(columns.get('covariates', []))}")

# 3. Select Model Type and Model
st.sidebar.markdown("### 3. Model")

# Get models for this station+dataset
global_models = registry.get_models_for_station_dataset(selected_station, selected_dataset, "global")
solo_models = registry.get_models_for_station_dataset(selected_station, selected_dataset, "single")

def get_model_label(m):
    """Generate label with metrics if available."""
    date_str = m.created_at[:10] if m.created_at else ""
    rmse_val = m.metrics.get('RMSE', m.metrics.get('rmse'))
    if rmse_val is not None:
        return f"{m.model_name} (RMSE={rmse_val:.4f}) - {date_str}"
    return f"{m.model_name} - {date_str}"

# Create combined model list with type prefix
all_station_models = []
model_labels = []

if global_models:
    for m in global_models:
        all_station_models.append(m)
        model_labels.append(f"[Global] {get_model_label(m)}")

if solo_models:
    for m in solo_models:
        all_station_models.append(m)
        model_labels.append(f"[Solo] {get_model_label(m)}")

if not all_station_models:
    source_label = "station" if has_real_stations else "source"
    st.sidebar.warning(f"No models for this {source_label} + dataset combination.")
    st.stop()

selected_model_idx = st.sidebar.selectbox(
    "Model",
    range(len(all_station_models)),
    format_func=lambda i: model_labels[i],
    label_visibility="collapsed"
)
selected_model_info = all_station_models[selected_model_idx]

# 4. For global models: select target station to predict
target_station = selected_station
if selected_model_info.model_type == "global":
    # Use appropriate label based on whether we have real station codes
    target_label = "Target Station" if has_real_stations else "Target Source"
    st.sidebar.markdown(f"### 4. {target_label}")
    target_station = st.sidebar.selectbox(
        target_label,
        selected_model_info.stations,
        index=selected_model_info.stations.index(selected_station) if selected_station in selected_model_info.stations else 0,
        label_visibility="collapsed"
    )

# Load selected model
with st.spinner("Loading model..."):
    try:
        loaded_data = load_model_data(selected_model_info)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = loaded_data['model']
config = loaded_data['config']
data_dict = loaded_data['data_dict']
scalers = loaded_data['scalers']

# Handle Global Models (Multi-station)
# Filter data and scalers for the target station if this is a global model
is_global_model = selected_model_info.model_type == "global"

if is_global_model and 'train' in data_dict and 'station' in data_dict['train'].columns:
    # Filter DataFrames for the target station
    data_dict_filtered = {}
    for k, v in data_dict.items():
        if isinstance(v, pd.DataFrame) and 'station' in v.columns:
            filtered = v[v['station'] == target_station].copy()
            # Drop station column after filtering to avoid issues
            filtered = filtered.drop(columns=['station'], errors='ignore')
            data_dict_filtered[k] = filtered
        else:
            data_dict_filtered[k] = v
    data_dict = data_dict_filtered
    
    # Debug info
    with st.sidebar.expander("Debug: Station Filtering", expanded=False):
        st.write(f"Target station: {target_station}")
        for k, v in data_dict.items():
            if isinstance(v, pd.DataFrame):
                st.write(f"{k}: {len(v)} rows")


    # Select correct scalers for this station (global models have per-station scalers)
    scalers_filtered = {}
    for scaler_name in ['target_preprocessor', 'cov_preprocessor']:
        s = scalers.get(scaler_name)
        if isinstance(s, dict):
            if target_station in s:
                scalers_filtered[scaler_name] = s[target_station]
            else:
                st.warning(f"No specific scaler for {target_station}, using default.")
                scalers_filtered[scaler_name] = list(s.values())[0] if s else None
        else:
            scalers_filtered[scaler_name] = s
    scalers = scalers_filtered

# Prepare dataframes - Correct data flow:
# - PROCESSED (train/val/test): for predictions - already normalized, no scaling needed!
# - RAW (train_raw/val_raw/test_raw): for display only
#
# The model was trained on PROCESSED data, so predictions must use PROCESSED data directly.
# DO NOT apply scalers again - data is already normalized.

# Full processed dataframe for predictions
# IMPORTANT: Always use train+val+test concatenation, NOT 'full' (full_data.csv)!
# full_data.csv contains RAW data, while train/val/test.csv contain NORMALIZED data.
full_df_processed = pd.concat([data_dict['train'], data_dict['val'], data_dict['test']])
full_df_processed = full_df_processed.sort_index()

# Test sets (processed)
test_df_processed = data_dict['test'].sort_index()

# Extract configuration FIRST (needed for raw data generation)
model_horizon = getattr(model, 'output_chunk_length', 30)
input_chunk = getattr(model, 'input_chunk_length', 30)
target_col = config.columns['target']
covariate_cols = config.columns.get('covariates', [])
use_covariates = config.use_covariates

# Helper function to generate raw data from processed data via inverse_transform
def generate_raw_from_processed(processed_df, scalers, target_col):
    """Generate raw (de-normalized) data from processed (normalized) data."""
    target_preprocessor = scalers.get('target_preprocessor')
    if target_preprocessor is not None:
        # Convert processed DataFrame to TimeSeries, inverse_transform, convert back
        processed_series, _ = prepare_dataframe_for_darts(
            processed_df, target_col, []
        )
        raw_series = target_preprocessor.inverse_transform(processed_series)
        raw_df = raw_series.pd_dataframe()
        raw_df.index = processed_df.index  # Ensure same index
        return raw_df
    else:
        # No scaler = data wasn't normalized, processed is raw
        return processed_df.copy()

# Get raw test data - CRITICAL for correct ground truth display
# If test_raw.csv wasn't saved, we need to generate it via inverse_transform
if 'test_raw' in data_dict:
    test_df_raw = data_dict['test_raw'].sort_index()
else:
    # Generate raw data via inverse_transform (same as predictions)
    test_df_raw = generate_raw_from_processed(test_df_processed, scalers, target_col)

# Ensure indices match
if not test_df_processed.index.equals(test_df_raw.index):
    if len(test_df_processed) == len(test_df_raw):
        test_df_raw = test_df_raw.copy()
        test_df_raw.index = test_df_processed.index

# Raw dataframe for display - also generate via inverse_transform if not available
if 'train_raw' in data_dict and 'val_raw' in data_dict and 'test_raw' in data_dict:
    full_df_raw = pd.concat([data_dict['train_raw'], data_dict['val_raw'], data_dict['test_raw']])
else:
    # Generate raw data via inverse_transform
    full_df_raw = generate_raw_from_processed(full_df_processed, scalers, target_col)
full_df_raw = full_df_raw.sort_index()


# =============================================================================
# INFO PANELS
# =============================================================================
col_data, col_model, col_metrics = st.columns(3)

with col_data:
    st.markdown("### Dataset Info")
    st.markdown(f"""
    | Split | Size |
    |-------|--------|
    | Train | {len(data_dict['train']):,} |
    | Val | {len(data_dict['val']):,} |
    | Test | {len(data_dict['test']):,} |
    """)

with col_model:
    st.markdown("### Model Info")
    st.markdown(f"""
    | Parameter | Value |
    |-----------|--------|
    | Type | **{config.model_name}** |
    | Input | {input_chunk} days |
    | Horizon | {model_horizon} days |
    | Covariates | {' ' + str(len(covariate_cols)) if use_covariates else ''} |
    """)

with col_metrics:
    st.markdown("### Validation Metrics")
    if config.metrics:
        cols = st.columns(2)
        # Show first 4 metrics
        for i, (name, value) in enumerate(list(config.metrics.items())[:6]):
             # Skip R2 if present (legacy models)
             if name.upper() in ['R2', 'R SQUARED']:
                 continue
                 
             with cols[i % 2]:
                if value is not None and not pd.isna(value):
                    st.metric(name, f"{value:.4f}")

with st.expander("Hyperparameters"):
    if hasattr(config, 'hyperparams') and config.hyperparams:
        hp_cols = st.columns(4)
        for i, (param, value) in enumerate(config.hyperparams.items()):
            with hp_cols[i % 4]:
                st.markdown(f"**{param}**: `{value}`")
    else:
        st.info("Hyperparameters not available")

st.markdown("---")


# =============================================================================
# SLIDING WINDOW - Test Set Only
# =============================================================================
st.markdown("### Sliding Window on Test Set")

# Calculate valid range over the TEST set only
test_len = len(test_df_processed)
valid_start = input_chunk  # Need at least input_chunk days before for context
valid_end = test_len - model_horizon  # Need room for prediction

if valid_start >= valid_end:
    st.error(f"Test set too small. Need at least {input_chunk + model_horizon} days, have {test_len}.")
    st.stop()

# Slider over test set
start_idx = st.slider(
    f"Slide window on test set ({model_horizon} days prediction)",
    min_value=valid_start,
    max_value=valid_end,
    value=valid_start,
    help=f"Input: {input_chunk} days of context | Prediction: {model_horizon} days ahead"
)

# Calculate window dates (relative to TEST data)
window_start_date = test_df_processed.index[start_idx]
window_end_date = test_df_processed.index[min(start_idx + model_horizon - 1, test_len - 1)]

st.caption(f"**Prediction window:** {window_start_date.strftime('%Y-%m-%d')} → {window_end_date.strftime('%Y-%m-%d')} ({model_horizon} days)")

st.markdown("---")


# =============================================================================
# NAVIGATION
# =============================================================================
if 'forecasting_tab' not in st.session_state:
    st.session_state.forecasting_tab = "Predictions"

selected_tab = st.radio(
    "Navigation",
    ["Predictions", "Explainability"],
    horizontal=True,
    key="forecasting_tab_radio",
    index=["Predictions", "Explainability"].index(st.session_state.forecasting_tab),
    label_visibility="collapsed"
)
st.session_state.forecasting_tab = selected_tab

# Cache key for window predictions
window_pred_key = f"window_pred_{selected_model_info.model_id}_{target_station}_{start_idx}"


# =============================================================================
# GENERATE WINDOW PREDICTION (Only when slider moves)
# =============================================================================
if window_pred_key not in st.session_state:
    with st.spinner("Generating prediction for window..."):
        try:
            # Generate forecast for the specific window using PROCESSED data
            results = generate_single_window_forecast(
                model=model,
                full_df=full_df_processed,
                target_col=target_col,
                covariate_cols=covariate_cols if use_covariates else None,
                preprocessing_config=config.preprocessing if hasattr(config, 'preprocessing') else {},
                scalers=scalers,
                start_date=window_start_date,
                use_covariates=use_covariates,
                already_processed=True,
                is_global_model=is_global_model
            )
            # results: pred_auto, pred_onestep, target, metrics_auto, metrics_onestep, horizon
            st.session_state[window_pred_key] = {
                'prediction': results[1],  # One-step prediction (RAW)
                'target': results[2],      # Ground truth (RAW)
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state[window_pred_key] = None


# =============================================================================
# TAB 1: PREDICTIONS - Test Set with Sliding Window
# =============================================================================
if selected_tab == "Predictions":
    
    cached_window = st.session_state.get(window_pred_key)
    
    # Prepare TEST set for display (RAW values)
    test_series_raw, _ = prepare_dataframe_for_darts(test_df_raw, target_col, [])
    
    fig = go.Figure()
    
    # 1. Highlight the input window (context)
    input_start_date = window_start_date - pd.Timedelta(days=input_chunk)
    fig.add_vrect(
        x0=input_start_date, x1=window_start_date,
        fillcolor="rgba(46, 134, 171, 0.15)",
        layer="below", line_width=1,
        line=dict(color="rgba(46, 134, 171, 0.4)"),
        annotation_text=f"Input ({input_chunk}d)", annotation_position="bottom left"
    )
    
    # 2. Highlight the prediction window
    fig.add_vrect(
        x0=window_start_date, x1=window_end_date,
        fillcolor="rgba(255, 200, 0, 0.25)",
        layer="below", line_width=1,
        line=dict(color="rgba(255, 200, 0, 0.6)"),
        annotation_text=f"Prediction ({model_horizon}d)", annotation_position="top right"
    )

    # 2. Ground Truth - Test set only
    fig.add_trace(go.Scatter(
        x=test_series_raw.time_index,
        y=test_series_raw.values().flatten(),
        mode='lines',
        name='Ground Truth',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # 3. Window Forecast (if available)
    if cached_window and cached_window.get('prediction') is not None:
        fig.add_trace(go.Scatter(
            x=cached_window['prediction'].time_index,
            y=cached_window['prediction'].values().flatten(),
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#E91E63', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"{target_col} - Test Set",
        xaxis_title="Date",
        yaxis_title=target_col,
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Display
    st.markdown("### Window Metrics")
    
    if cached_window and cached_window.get('prediction') is not None and cached_window.get('target') is not None:
        pred = cached_window['prediction']
        target = cached_window['target']
        
        min_len = min(len(pred), len(target))
        if min_len > 0:
            pred_aligned = pred[:min_len]
            target_aligned = target[:min_len]
            
            # Hydrology metrics
            metrics = {
                'MAE': float(mae(target_aligned, pred_aligned)),
                'RMSE': float(rmse(target_aligned, pred_aligned)),
                'NSE': nash_sutcliffe_efficiency(target_aligned, pred_aligned),
                'KGE': kling_gupta_efficiency(target_aligned, pred_aligned),
            }
            
            m_cols = st.columns(4)
            for i, (name, value) in enumerate(metrics.items()):
                with m_cols[i]:
                    st.metric(name, f"{value:.4f}")
    else:
        st.info("Déplacez le slider pour générer une prédiction.")
    
    # Export
    st.markdown("---")
    if cached_window and cached_window.get('prediction') is not None:
        pred = cached_window['prediction']
        target = cached_window['target']
        min_len = min(len(pred), len(target))
        
        export_df = pd.DataFrame({
            'date': pred.time_index[:min_len],
            'ground_truth': target.values().flatten()[:min_len],
            'prediction': pred.values().flatten()[:min_len]
        })
        
        st.download_button(
            label="📥 Export CSV",
            data=export_df.to_csv(index=False),
            file_name=f"prediction_{window_start_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 2: EXPLAINABILITY (TimeSHAP)
# =============================================================================
else:
    st.markdown("### Model Interpretation")
    
    if 'explain_subtab' not in st.session_state:
        st.session_state.explain_subtab = "Local Explanations"
        
    sub_tab = st.radio(
        "Sub-Navigation",
        ["Local Explanations", "Global Explanations"],
        horizontal=True,
        index=["Local Explanations", "Global Explanations"].index(st.session_state.explain_subtab),
        label_visibility="collapsed"
    )
    st.session_state.explain_subtab = sub_tab
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # SUB-TAB: LOCAL (Window)
    # -------------------------------------------------------------------------
    if sub_tab == "Local Explanations":
        st.markdown("#### Local SHAP Analysis")
        st.caption("Explains which features and past days influenced this specific prediction window.")
        
        cached_window = st.session_state.get(window_pred_key)
        
        # Prepare Context Data (Input Chunk)
        display_start = window_start_date - timedelta(days=input_chunk)
        display_end = window_end_date
        
        window_mask = (test_df_raw.index >= display_start) & (test_df_raw.index <= display_end)
        window_df = test_df_raw.loc[window_mask].copy()
        
        if len(window_df) > 0:
            # 1. Context Chart
            fig_target = go.Figure()
            
            # Highlight Prediction
            fig_target.add_vrect(
                x0=window_start_date, x1=window_end_date,
                fillcolor="rgba(255, 200, 0, 0.15)", line_color="orange",
                annotation_text="Prediction", annotation_position="top right"
            )
            fig_target.add_vline(x=window_start_date, line_dash="dash", line_color="orange")
            
            fig_target.add_trace(go.Scatter(
                x=window_df.index,
                y=window_df[target_col].values,
                mode='lines+markers', name='Ground Truth',
                line=dict(color='#2E86AB'), marker=dict(size=4)
            ))
            
            if cached_window and cached_window.get('pred_onestep') is not None:
                pred = cached_window['pred_onestep']
                fig_target.add_trace(go.Scatter(
                    x=pred.time_index, y=pred.values().flatten(),
                    mode='lines+markers', name='One-Step Prediction',
                    line=dict(color='#28A745'), marker=dict(size=6, symbol='diamond')
                ))
            
            fig_target.update_layout(
                title=f" Context ({input_chunk}d) + Prediction Window",
                height=300, margin=dict(l=10, r=10, t=40, b=20),
                hovermode='x unified'
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
            # 2. Covariates Display
            # Filter standard computed features to avoid clutter
            computed_patterns = ['day_of_week', 'month', 'sin', 'cos', 'weekday']
            base_covariates = [
                col for col in covariate_cols 
                if col in window_df.columns and not any(p in col.lower() for p in computed_patterns)
            ]
            
            if base_covariates:
                st.markdown(f"#####  Covariates ({len(base_covariates)})")
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                
                for i, feat_col in enumerate(base_covariates):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        fig_cov = go.Figure()
                        fig_cov.add_trace(go.Scatter(
                            x=window_df.index, y=window_df[feat_col].values,
                            mode='lines', name=feat_col,
                            line=dict(color=colors[i % len(colors)])
                        ))
                        fig_cov.update_layout(
                            title=dict(text=feat_col, font=dict(size=12)),
                            height=80, margin=dict(l=0, r=0, t=20, b=0),
                            showlegend=False, xaxis=dict(showticklabels=False)
                        )
                        st.plotly_chart(fig_cov, use_container_width=True)
                    with c2:
                        vals = window_df[feat_col].values
                        st.markdown(f"""
                        <div style='font-size:10px; padding-top:20px;'>
                        <b>Mean:</b> {vals.mean():.2f}<br>
                        <b>Min/Max:</b> {vals.min():.2f} / {vals.max():.2f}
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 3. Feature Importance (Correlation-based - no external dependencies)
        st.markdown("##### Feature Importance")
        st.caption("Correlation-based importance (correlation with target variable)")
        
        feature_cols = [target_col] + [c for c in covariate_cols if c in window_df.columns]
        
        if len(feature_cols) > 1:
            # Calculate correlations with target
            correlations = {}
            target_values = window_df[target_col].values
            
            for col in feature_cols:
                if col != target_col and col in window_df.columns:
                    col_values = window_df[col].values
                    if len(col_values) > 2:
                        corr = np.corrcoef(target_values, col_values)[0, 1]
                        if not np.isnan(corr):
                            correlations[col] = abs(corr)
            
            if correlations:
                # Sort by importance
                sorted_corr = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
                
                # Create bar chart
                importance_df = pd.DataFrame({
                    'Feature': list(sorted_corr.keys()),
                    'Importance': list(sorted_corr.values())
                })
                
                fig_importance = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#2E86AB'
                ))
                fig_importance.update_layout(
                    title="Feature Importance (|correlation| with target)",
                    xaxis_title="Absolute Correlation",
                    yaxis_title="Feature",
                    height=max(200, len(sorted_corr) * 30),
                    margin=dict(l=10, r=10, t=40, b=20),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Not enough data to compute feature importance.")
        else:
            st.info("No covariates available for importance analysis.")
    
    # -------------------------------------------------------------------------
    # SUB-TAB: GLOBAL (Aggregated)
    # -------------------------------------------------------------------------
    elif sub_tab == "Global Explanations":
        st.markdown("#### Global Feature Importance")
        st.caption("Correlation-based importance across the entire test set.")
        
        if st.button("Compute Global Importance"):
            with st.spinner("Computing..."):
                feature_cols = [c for c in covariate_cols if c in test_df_processed.columns]
                
                if feature_cols:
                    correlations = {}
                    target_values = test_df_processed[target_col].values
                    
                    for col in feature_cols:
                        col_values = test_df_processed[col].values
                        if len(col_values) > 2:
                            corr = np.corrcoef(target_values, col_values)[0, 1]
                            if not np.isnan(corr):
                                correlations[col] = abs(corr)
                    
                    if correlations:
                        sorted_corr = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
                        
                        importance_df = pd.DataFrame({
                            'Feature': list(sorted_corr.keys()),
                            'Importance': list(sorted_corr.values())
                        })
                        
                        chart = alt.Chart(importance_df).mark_bar().encode(
                            x=alt.X('Importance', title='|Correlation| with Target'),
                            y=alt.Y('Feature', sort='-x'),
                            tooltip=['Feature', 'Importance']
                        ).properties(
                            title='Global Feature Importance',
                            height=max(200, len(sorted_corr) * 25)
                        ).interactive()
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Show table
                        st.dataframe(importance_df.style.format({'Importance': '{:.3f}'}), use_container_width=True)
                    else:
                        st.info("Could not compute correlations.")
                else:
                    st.info("No covariates available.")

