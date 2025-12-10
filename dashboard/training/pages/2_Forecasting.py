"""Forecasting Page - Sliding Window on TEST data.

This page allows users to:
1. Load a trained model.
2. Visualize predictions on the test set (Autoregressive & One-Step).
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
from dashboard.utils.timeshap_wrapper import DartsModelWrapper
from dashboard.utils.preprocessing import prepare_dataframe_for_darts
from darts.metrics import mae, rmse, mape, smape, r2_score

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
# SIDEBAR: MODEL SELECTION
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

# Get all unique stations across all models
all_stations = registry.get_all_stations()

if not all_stations:
    st.warning("No stations found in registered models.")
    st.stop()

# Select station first
selected_station = st.sidebar.selectbox("Station", sorted(all_stations))

# Get models for this station (both single and global)
station_models = registry.get_models_for_station(selected_station)

if not station_models:
    st.warning(f"No models found for station {selected_station}")
    st.stop()

# Create model labels with type indicator
def get_model_label(m):
    type_icon = "🏠" if m.model_type == "single" else "🌍"
    date_str = m.created_at[:10] if m.created_at else "unknown"
    return f"{type_icon} {m.model_name} ({date_str})"

model_labels = [get_model_label(m) for m in station_models]

selected_model_idx = st.sidebar.selectbox(
    "Model", 
    range(len(station_models)), 
    format_func=lambda i: model_labels[i]
)
selected_model_info = station_models[selected_model_idx]

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
# If dataset has 'station' column, filter for specific station
is_global_model = False
if 'train' in data_dict and 'station' in data_dict['train'].columns:
    train_df = data_dict['train']
    available_stations = sorted(train_df['station'].unique())

    if len(available_stations) > 1:
        is_global_model = True
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌍 Global Model")
        target_station = st.sidebar.selectbox("Select Target Station", available_stations)

        # Filter DataFrames for the target station
        data_dict_filtered = {}
        for k, v in data_dict.items():
            if isinstance(v, pd.DataFrame) and 'station' in v.columns:
                data_dict_filtered[k] = v[v['station'] == target_station].copy()
            else:
                data_dict_filtered[k] = v

        data_dict = data_dict_filtered

        # Select correct scalers for this station
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
        st.sidebar.success(f"Predicting on: **{target_station}**")

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
    | Covariates | {'✅ ' + str(len(covariate_cols)) if use_covariates else '❌'} |
    """)

with col_metrics:
    st.markdown("### Validation Metrics")
    if config.metrics:
        # Show first 4 metrics
        cols = st.columns(2)
        for i, (name, value) in enumerate(list(config.metrics.items())[:4]):
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
# GLOBAL SLIDER - Analysis Window
# =============================================================================
st.markdown("### Analysis Window")

valid_start = input_chunk
valid_end = len(test_df_processed) - model_horizon

if valid_start >= valid_end:
    st.error(f"Not enough test data. Minimum required: {input_chunk + model_horizon} days")
    st.stop()

start_idx = st.slider(
    "Move slider to explore different windows (Predictions & Explanations)",
    min_value=valid_start,
    max_value=valid_end,
    value=valid_start,
    help=f"This window of {model_horizon} days is used for one-step predictions AND SHAP explanations"
)

window_start_date = test_df_processed.index[start_idx]
window_end_date = test_df_processed.index[min(start_idx + model_horizon - 1, len(test_df_processed) - 1)]
st.caption(f"Window: **{window_start_date.strftime('%Y-%m-%d')}** → **{window_end_date.strftime('%Y-%m-%d')}**")

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

# Cache keys
full_pred_key = f"full_pred_{selected_model_info.model_id}"
window_pred_key = f"window_pred_{selected_model_info.model_id}_{start_idx}"


# =============================================================================
# GENERATE PREDICTIONS (Compute Logic)
# =============================================================================

# 1. Autoregressive Predictions on Full Test Set (Once per model load)
if full_pred_key not in st.session_state:
    with st.spinner("Generating autoregressive predictions on test set..."):
        try:
            # Prepare series using Darts - using PROCESSED data (already normalized)
            full_series_processed, covariates_processed = prepare_dataframe_for_darts(
                full_df_processed, target_col, covariate_cols if use_covariates else []
            )
            
            # NO SCALING NEEDED - data is already normalized
            # Get scalers only for inverse_transform (to get RAW output for display)
            target_preprocessor = scalers.get('target_preprocessor')
            
            # Generate TRUE AUTOREGRESSIVE forecasts (multi-step ahead)
            # For global models, wrap single station data in lists
            if is_global_model:
                series_for_pred = [full_series_processed]
                cov_for_pred = [covariates_processed] if covariates_processed else None
            else:
                series_for_pred = full_series_processed
                cov_for_pred = covariates_processed

            forecast_kwargs = {
                'series': series_for_pred,  # Already normalized
                'start': test_df_processed.index[0],
                'forecast_horizon': model_horizon,
                'stride': model_horizon,
                'retrain': False,
                'last_points_only': False,
                'verbose': False
            }

            # Add covariates if available and model was TRAINED with them
            model_uses_past = getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False)
            model_uses_future = getattr(model, "_uses_future_covariates", False) or getattr(model, "uses_future_covariates", False)
            
            if cov_for_pred is not None and use_covariates:
                if model_uses_past:
                    forecast_kwargs['past_covariates'] = cov_for_pred
                if model_uses_future:
                    forecast_kwargs['future_covariates'] = cov_for_pred

            forecasts_list = model.historical_forecasts(**forecast_kwargs)
            
            # Handle results from global models (which return lists of lists)
            from darts import concatenate
            if is_global_model and isinstance(forecasts_list, list) and len(forecasts_list) > 0:
                # For global models, we get a list with one element (our station)
                if isinstance(forecasts_list[0], list):
                    # List of lists case
                    forecasts = concatenate(forecasts_list[0])
                else:
                    # Single list case
                    forecasts = forecasts_list[0]
            elif isinstance(forecasts_list, list) and len(forecasts_list) > 0:
                # Single model case - concatenate list of forecast windows
                forecasts = concatenate(forecasts_list)
            else:
                forecasts = forecasts_list
            
            # Inverse Transform to get RAW predictions for display
            # Metrics will be computed on NORMALIZED data (forecasts vs full_series_processed)
            if target_preprocessor:
                pred_auto_raw = target_preprocessor.inverse_transform(forecasts)
            else:
                pred_auto_raw = forecasts
            
            # Align with ground truth for metrics - use PROCESSED data for metrics
            test_series_processed = full_series_processed.slice(
                test_df_processed.index[0], test_df_processed.index[-1]
            )
            
            common_start = max(forecasts.start_time(), test_series_processed.start_time())
            common_end = min(forecasts.end_time(), test_series_processed.end_time())
            
            pred_aligned = forecasts.slice(common_start, common_end)
            test_aligned = test_series_processed.slice(common_start, common_end)
            
            # Metrics on NORMALIZED data (same scale as training)
            metrics_full = {
                'MAE': float(mae(test_aligned, pred_aligned)),
                'RMSE': float(rmse(test_aligned, pred_aligned)),
                'MAPE': float(mape(test_aligned, pred_aligned)),
                'R²': float(r2_score(test_aligned, pred_aligned)),
            }
            
            st.session_state[full_pred_key] = {
                'pred_auto': pred_auto_raw,  # RAW for display
                'pred_auto_processed': forecasts,  # PROCESSED for SHAP
                'target_raw': test_series_processed,  # Keep for backwards compat
                'target_processed': test_series_processed,  # PROCESSED for SHAP
                'metrics': metrics_full
            }
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state[full_pred_key] = None

# 2. Window Predictions (When slider moves)
if window_pred_key not in st.session_state:
    with st.spinner("Generating predictions for window..."):
        try:
            # Generate forecast for the specific window using PROCESSED data
            results = generate_single_window_forecast(
                model=model,
                full_df=full_df_processed,  # PROCESSED data
                target_col=target_col,
                covariate_cols=covariate_cols if use_covariates else None,
                preprocessing_config=config.preprocessing if hasattr(config, 'preprocessing') else {},
                scalers=scalers,
                start_date=window_start_date,
                use_covariates=use_covariates,
                already_processed=True,  # Flag: data is already normalized, don't scale
                is_global_model=is_global_model  # Pass global model flag
            )
            # Unpack results: pred_auto, pred_onestep, target, metrics_auto, metrics_onestep, horizon
            st.session_state[window_pred_key] = {
                'pred_auto_window': results[0],
                'pred_onestep': results[1],
                'target': results[2],
                'metrics_auto': results[3],
                'metrics_onestep': results[4]
            }
        except Exception as e:
            st.error(f"Window prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state[window_pred_key] = None


# =============================================================================
# TAB 1: PREDICTIONS
# =============================================================================
if selected_tab == "Predictions":
    
    st.markdown("### Full Test Set with Window Highlight")
    
    if st.session_state.get(full_pred_key):
        cached_full = st.session_state[full_pred_key]
        cached_window = st.session_state.get(window_pred_key)
        
        # 2. Ground Truth - Use RAW data directly for display
        # test_df_raw already contains the raw (non-normalized) values
        test_series_raw_display, _ = prepare_dataframe_for_darts(
            test_df_raw, target_col, []
        )
        
        fig = go.Figure()
        
        # 1. Highlight Window
        fig.add_vrect(
            x0=window_start_date, x1=window_end_date,
            fillcolor="rgba(255, 200, 0, 0.2)", 
            layer="below", line_width=0,
            annotation_text="Analysis Window", annotation_position="top left"
        )

        fig.add_trace(go.Scatter(
            x=test_series_raw_display.time_index,
            y=test_series_raw_display.values().flatten(),
            mode='lines',
            name='Ground Truth',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # 3. Full One-Step Forecasts
        pred_auto = cached_full['pred_auto']
        gt_start = test_series_raw_display.start_time()
        gt_end = test_series_raw_display.end_time()
        
        try:
            pred_auto_sliced = pred_auto.slice(gt_start, gt_end)
        except:
            pred_auto_sliced = pred_auto
        
        fig.add_trace(go.Scatter(
            x=pred_auto_sliced.time_index,
            y=pred_auto_sliced.values().flatten(),
            mode='lines',
            name='Historical Forecast (One-Step Rolling)',
            line=dict(color='#F24236', width=1, dash='dot')
        ))
        
        # 4. Window Specific Predictions: Only show One-Step Forecast for the selected window
        if cached_window:
            # One-Step Forecast (Window) - The main comparison metric
            if cached_window.get('pred_onestep') is not None:
                fig.add_trace(go.Scatter(
                    x=cached_window['pred_onestep'].time_index,
                    y=cached_window['pred_onestep'].values().flatten(),
                    mode='lines+markers',
                    name='One-Step Forecast (Window)',
                    line=dict(color='#28A745', width=3),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title=f"{target_col} - Test Set Analysis",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Display
        col_full, col_window = st.columns(2)
        
        with col_full:
            st.markdown("**Test Set Metrics (Autoregressive):**")
            m_cols = st.columns(3)
            for i, (name, value) in enumerate(cached_full['metrics'].items()):
                with m_cols[i % 3]:
                    if value is not None and not pd.isna(value):
                        st.metric(name, f"{value:.4f}")
        
        with col_window:
            if cached_window and cached_window.get('pred_onestep') is not None and cached_window.get('target') is not None:
                st.markdown("**Window Metrics (One-Step):**")
                try:
                    # Compute metrics directly from cached predictions
                    pred_os = cached_window['pred_onestep']
                    target_w = cached_window['target']
                    
                    # Align lengths
                    min_len = min(len(pred_os), len(target_w))
                    if min_len > 0:
                        pred_os_aligned = pred_os[:min_len]
                        target_w_aligned = target_w[:min_len]
                        
                        metrics_w = {
                            'MAE': float(mae(target_w_aligned, pred_os_aligned)),
                            'RMSE': float(rmse(target_w_aligned, pred_os_aligned)),
                            'MAPE': float(mape(target_w_aligned, pred_os_aligned)),
                            'R²': float(r2_score(target_w_aligned, pred_os_aligned)),
                        }
                        
                        m_cols = st.columns(4)
                        for i, (name, value) in enumerate(metrics_w.items()):
                            with m_cols[i % 4]:
                                st.metric(name, f"{value:.4f}")
                    else:
                        st.warning("No overlapping data for metrics")
                except Exception as e:
                    st.warning(f"Could not compute window metrics: {e}")
            else:
                st.info("Window predictions not yet computed")
    
    # Export Section
    st.markdown("---")
    st.markdown("### Export Predictions")

    if st.session_state.get(full_pred_key):
        pred_auto = st.session_state[full_pred_key]['pred_auto']  # This is RAW (for display)
        # Create ground truth from RAW data for export
        test_series_raw_export, _ = prepare_dataframe_for_darts(
            test_df_raw, target_col, []
        )
        target = test_series_raw_export

        common_start = max(pred_auto.start_time(), target.start_time())
        common_end = min(pred_auto.end_time(), target.end_time())
        pred_aligned = pred_auto.slice(common_start, common_end)
        target_aligned = target.slice(common_start, common_end)
        
        min_len = min(len(pred_aligned), len(target_aligned))
        
        if min_len > 0:
            export_df = pd.DataFrame({
                'date': pred_aligned.time_index[:min_len],
                'ground_truth': target_aligned.values().flatten()[:min_len],
                'prediction_autoregressive': pred_aligned.values().flatten()[:min_len]
            })
            
            col_csv, col_info = st.columns(2)
            with col_csv:
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=export_df.to_csv(index=False),
                    file_name=f"predictions_{selected_station}_{config.model_name}.csv",
                    mime="text/csv"
                )
            with col_info:
                st.success(f"{len(export_df)} points available for download.")

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
                title=f"📈 Context ({input_chunk}d) + Prediction Window",
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
                st.markdown(f"##### 📊 Covariates ({len(base_covariates)})")
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
        
        # 3. SHAP Computation
        shap_cache_key = f"shap_{selected_model_info.model_id}_{start_idx}"
        
        if shap_cache_key not in st.session_state:
            with st.spinner("Calculating local SHAP importance..."):
                try:
                    from timeshap.explainer import local_report
                    
                    # Wrapper
                    model_wrapper = DartsModelWrapper(
                        model=model,
                        input_chunk_length=input_chunk,
                        forecast_horizon=1
                    )
                    
                    # Prepare Data for TimeSHAP
                    feature_names = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                    
                    window_length = min(input_chunk, len(test_df_processed) - start_idx)
                    window_shap_df = test_df_processed.iloc[start_idx:start_idx + window_length]
                    
                    # Build rows explicitly
                    rows = []
                    actual_features = [f for f in feature_names if f in window_shap_df.columns]
                    for t, (_, row) in enumerate(window_shap_df.iterrows()):
                        data_row = {'entity': 'seq_0', 't': t}
                        for f_name in actual_features:
                            data_row[f_name] = float(row[f_name])
                        rows.append(data_row)
                    data_df = pd.DataFrame(rows)
                    
                    # Baseline (Mean)
                    baseline_rows = []
                    for t in range(window_length):
                        b_row = {f: float(test_df_processed[f].mean()) for f in actual_features}
                        baseline_rows.append(b_row)
                    baseline_df = pd.DataFrame(baseline_rows)
                    
                    # Config
                    pruning_dict = {'tol': 0.025}
                    event_dict = {'rs': 42, 'nsamples': 50}
                    feature_dict = {
                        'rs': 42, 
                        'nsamples': 50,
                        'plot_features': {f"Feature {i}": f for i, f in enumerate(actual_features)}
                    }
                    
                    # Run TimeSHAP
                    plot = local_report(
                        f=model_wrapper,
                        data=data_df,
                        entity_uuid='seq_0',
                        entity_col='entity',
                        time_col='t',
                        model_features=actual_features,
                        baseline=baseline_df,
                        pruning_dict=pruning_dict,
                        event_dict=event_dict,
                        feature_dict=feature_dict
                    )
                    
                    st.session_state[shap_cache_key] = {'success': True, 'plot': plot}
                    
                except Exception as e:
                    st.error(f"TimeSHAP Error: {e}")
                    st.session_state[shap_cache_key] = {'success': False, 'error': str(e)}

        # Display Result
        result = st.session_state.get(shap_cache_key)
        if result and result.get('success'):
            st.markdown("###  TimeSHAP Report")
            try:
                st.altair_chart(result['plot'], use_container_width=True)
            except Exception as e:
                st.warning(f"Error rendering chart: {e}")
        elif result:
            st.warning("SHAP calculation failed.")
    
    # -------------------------------------------------------------------------
    # SUB-TAB: GLOBAL (Aggregated)
    # -------------------------------------------------------------------------
    elif sub_tab == "Global Explanations":
        st.markdown("#### Global Feature Importance")
        st.caption("Aggregated SHAP values across multiple windows.")
        
        global_shap_key = f"global_shap_{selected_model_info.model_id}"
        
        if st.button("🔍 Compute Global SHAP") or global_shap_key in st.session_state:
             if global_shap_key not in st.session_state or st.session_state.get(global_shap_key, {}).get('success') is False:
                with st.spinner("Computing global importance (sub-sampling)..."):
                    try:
                        from timeshap.explainer.global_methods import calc_global_explanations
                        
                        model_wrapper = DartsModelWrapper(
                            model=model,
                            input_chunk_length=input_chunk,
                            forecast_horizon=1
                        )
                        
                        feature_names = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                        actual_features = [f for f in feature_names if f in test_df_processed.columns]
                        
                        # Sub-sample 5 windows for speed
                        n_samples = 5
                        max_start = len(test_df_processed) - input_chunk - 1
                        sample_positions = np.linspace(0, max_start, n_samples, dtype=int)
                        
                        # Build Data
                        all_rows = []
                        for seq_idx, s_idx in enumerate(sample_positions):
                            window_shap_df = test_df_processed.iloc[s_idx:s_idx + input_chunk]
                            for t, (_, row) in enumerate(window_shap_df.iterrows()):
                                data_row = {'entity': f'seq_{seq_idx}', 't': t}
                                for f in actual_features:
                                    data_row[f] = float(row[f])
                                all_rows.append(data_row)
                        data_df = pd.DataFrame(all_rows)
                        
                        # Baseline
                        baseline_rows = [{f: float(test_df_processed[f].mean()) for f in actual_features} for _ in range(input_chunk)]
                        baseline_df = pd.DataFrame(baseline_rows)
                        
                        # Calculate
                        _, _, feature_data = calc_global_explanations(
                            f=model_wrapper,
                            data=data_df,
                            entity_col='entity',
                            time_col='t',
                            model_features=actual_features,
                            baseline=baseline_df,
                            pruning_dict={'tol': 0.025},
                            event_dict={'rs': 42, 'nsamples': 50},
                            feature_dict={'rs': 42, 'nsamples': 50},
                            verbose=False
                        )
                        
                        # Aggregate
                        # Map generic Feature names back if needed, but assuming direct mapping for now
                        if 'Feature' in feature_data.columns and feature_data['Feature'].str.startswith("Feature ").all():
                             feature_map = {f"Feature {i}": name for i, name in enumerate(actual_features)}
                             feature_data['Feature'] = feature_data['Feature'].map(feature_map)
                        
                        global_importance = feature_data.groupby('Feature')['Shapley Value'].apply(lambda x: x.abs().mean()).reset_index()
                        
                        # Chart
                        chart = alt.Chart(global_importance).mark_bar().encode(
                            x=alt.X('Shapley Value', title='Mean |Shapley Value|'),
                            y=alt.Y('Feature', sort='-x'),
                            tooltip=['Feature', 'Shapley Value']
                        ).properties(title='Global Feature Importance').interactive()
                        
                        st.session_state[global_shap_key] = {'success': True, 'plot': chart}
                        
                    except Exception as e:
                         st.error(f"Global SHAP error: {e}")
                         st.session_state[global_shap_key] = {'success': False}

             # Display
             res = st.session_state.get(global_shap_key)
             if res and res.get('success'):
                 st.altair_chart(res['plot'], use_container_width=True)
