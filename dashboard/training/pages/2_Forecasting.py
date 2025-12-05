"""Forecasting Page - Sliding Window on TEST data."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.model_registry import get_registry
from dashboard.utils.model_config import load_model_with_config, load_scalers
from dashboard.utils.forecasting import generate_single_window_forecast
from dashboard.utils.export import add_download_button

# =============================================================================
st.sidebar.header("Model Selection")

registry = get_registry()
all_models = registry.scan_models()
models = [m for m in all_models if m.model_path.exists()]

if not models:
    st.warning("No trained models available.")
    st.info("Please train a model first on the **Train Models** page.")
    st.stop()

# Group by station
models_by_station = {}
for m in models:
    if m.station not in models_by_station:
        models_by_station[m.station] = []
    models_by_station[m.station].append(m)

selected_station = st.sidebar.selectbox("Station", sorted(models_by_station.keys()))
station_models = models_by_station[selected_station]
model_labels = [f"{m.model_type} ({m.creation_date})" for m in station_models]
selected_model_idx = st.sidebar.selectbox("Model", range(len(station_models)), format_func=lambda i: model_labels[i])
selected_model_info = station_models[selected_model_idx]

# Loading
cache_key = f"model_{selected_model_info.model_path}"
if cache_key not in st.session_state:
    with st.spinner("Loading..."):
        try:
            model, config, data_dict = load_model_with_config(selected_model_info.model_path.parent)
            scalers = load_scalers(selected_model_info.model_path.parent)
            st.session_state[cache_key] = {'model': model, 'config': config, 'data_dict': data_dict, 'scalers': scalers}
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

loaded_data = st.session_state[cache_key]
model = loaded_data['model']
config = loaded_data['config']
data_dict = loaded_data['data_dict']
scalers = loaded_data['scalers']

# Full DataFrame
if 'full' in data_dict:
    full_df = data_dict['full']
else:
    full_df = pd.concat([data_dict['train'], data_dict['val'], data_dict['test']])
full_df = full_df.sort_index()

test_df = data_dict['test'].sort_index()
test_df_raw = data_dict.get('test_raw', test_df).sort_index()

# Parameters
model_horizon = getattr(model, 'output_chunk_length', 30)
input_chunk = getattr(model, 'input_chunk_length', 30)
target_col = config.columns['target']
covariate_cols = config.columns.get('covariates', [])
use_covariates = config.use_covariates



# =============================================================================
# 2. INFO PANELS
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
        cols = st.columns(2)
        for i, (name, value) in enumerate(list(config.metrics.items())[:4]):
            with cols[i % 2]:
                if value is not None and not pd.isna(value):
                    st.metric(name, f"{value:.4f}")

# Hyperparameters
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
# 3. GLOBAL SLIDER - Window Position (works for all tabs)
# =============================================================================
st.markdown("### Analysis Window")

valid_start = input_chunk
valid_end = len(test_df) - model_horizon

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

window_start_date = test_df.index[start_idx]
window_end_date = test_df.index[min(start_idx + model_horizon - 1, len(test_df) - 1)]
st.caption(f"Window: **{window_start_date.strftime('%Y-%m-%d')}** → **{window_end_date.strftime('%Y-%m-%d')}**")

st.markdown("---")

# =============================================================================
# 4. PERSISTENT NAVIGATION
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

# Update state
st.session_state.forecasting_tab = selected_tab

# Cache keys for predictions (defined before navigation logic)
full_pred_key = f"full_pred_{selected_model_info.model_path}"
window_pred_key = f"window_pred_{selected_model_info.model_path}_{start_idx}"

# Imports needed for both tabs
from dashboard.utils.forecasting import generate_rolling_forecast
from darts import concatenate
from darts.metrics import mae, rmse, mape

# =============================================================================
# TAB: PREDICTIONS
# =============================================================================
# =========================================================================
# GENERATE PREDICTIONS (Runs on every reload/slider change)
# =========================================================================

# 1. Autoregressive predictions on full test set
if full_pred_key not in st.session_state:
    with st.spinner("Generating autoregressive predictions on test set..."):
        try:
            from dashboard.utils.preprocessing import prepare_dataframe_for_darts
            from darts.metrics import mae, rmse
            
            # Prepare series
            full_series, covariates = prepare_dataframe_for_darts(
                full_df, target_col, covariate_cols if use_covariates else []
            )
            
            # Scaling
            target_preprocessor = scalers.get('target_preprocessor')
            cov_preprocessor = scalers.get('cov_preprocessor')
            
            full_series_scaled = full_series
            covariates_scaled = covariates
            
            if target_preprocessor:
                full_series_scaled = target_preprocessor.transform(full_series)
            if cov_preprocessor and covariates is not None:
                covariates_scaled = cov_preprocessor.transform(covariates)
            
            # Historical forecasts with stride=1 for one-step on full test
            first_test_date = test_df.index[0]
            
            forecast_kwargs = {
                'series': full_series_scaled,
                'start': first_test_date,
                'forecast_horizon': 1,  # One-step each time
                'stride': 1,
                'retrain': False,
                'last_points_only': True,
                'verbose': False
            }
            
            if covariates_scaled is not None and use_covariates:
                if getattr(model, "uses_past_covariates", False):
                    forecast_kwargs['past_covariates'] = covariates_scaled
                if getattr(model, "uses_future_covariates", False):
                    forecast_kwargs['future_covariates'] = covariates_scaled
            
            forecasts_scaled = model.historical_forecasts(**forecast_kwargs)
            
            # Inverse scaling
            if target_preprocessor:
                pred_auto_full = target_preprocessor.inverse_transform(forecasts_scaled)
            else:
                pred_auto_full = forecasts_scaled
            
            # Real test series
            test_series = full_series.slice(test_df.index[0], test_df.index[-1])
            
            # Align for metrics
            common_start = max(pred_auto_full.start_time(), test_series.start_time())
            common_end = min(pred_auto_full.end_time(), test_series.end_time())
            
            pred_aligned = pred_auto_full.slice(common_start, common_end)
            test_aligned = test_series.slice(common_start, common_end)
            
            metrics_full = {
                'MAE': float(mae(test_aligned, pred_aligned)),
                'RMSE': float(rmse(test_aligned, pred_aligned)),
            }
            
            st.session_state[full_pred_key] = {
                'pred_auto': pred_auto_full, 
                'target': test_series, 
                'metrics': metrics_full
            }
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state[full_pred_key] = None

# 2. One-step predictions for selected window
if window_pred_key not in st.session_state:
    with st.spinner("Generating one-step predictions for window..."):
        try:
            _, pred_onestep, target_window, metrics_window, _ = generate_single_window_forecast(
                model=model, full_df=full_df, target_col=target_col,
                covariate_cols=covariate_cols if use_covariates else None,
                preprocessing_config=config.preprocessing if hasattr(config, 'preprocessing') else {},
                scalers=scalers, start_date=window_start_date, use_covariates=use_covariates
            )
            st.session_state[window_pred_key] = {'pred_onestep': pred_onestep, 'target': target_window, 'metrics': metrics_window}
        except Exception as e:
            st.error(f"Window prediction error: {e}")
            st.session_state[window_pred_key] = None

# =============================================================================
# TAB: PREDICTIONS
# =============================================================================
if selected_tab == "Predictions":
    
    # =========================================================================
    # MAIN CHART: Full Test Set + Highlighted Window
    # =========================================================================
    st.markdown("### Full Test Set with Window Highlight")
    
    if st.session_state.get(full_pred_key):
        cached_full = st.session_state[full_pred_key]
        cached_window = st.session_state.get(window_pred_key)
        
        fig = go.Figure()
        
        # 1. Highlight zone for selected window
        fig.add_vrect(
            x0=window_start_date, x1=window_end_date,
            fillcolor="rgba(255, 200, 0, 0.2)", 
            layer="below", line_width=0,
            annotation_text="One-Step Window", annotation_position="top left"
        )
        
        # 2. Real data (full test)
        fig.add_trace(go.Scatter(
            x=test_df_raw.index,
            y=test_df_raw[target_col].values,
            mode='lines',
            name='Ground Truth',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # 3. Autoregressive prediction (full test)
        fig.add_trace(go.Scatter(
            x=cached_full['pred_auto'].time_index,
            y=cached_full['pred_auto'].values().flatten(),
            mode='lines',
            name='Autoregressive Prediction',
            line=dict(color='#F24236', width=2)
        ))
        
        # 4. One-Step Prediction (window only)
        if cached_window and cached_window.get('pred_onestep') is not None:
            fig.add_trace(go.Scatter(
                x=cached_window['pred_onestep'].time_index,
                y=cached_window['pred_onestep'].values().flatten(),
                mode='lines+markers',
                name='One-Step Prediction (Window)',
                line=dict(color='#28A745', width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"{target_col} - Full Test Set",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # METRICS
        # =====================================================================
        col_full, col_window = st.columns(2)
        
        with col_full:
            st.markdown("**Full Test Metrics (Autoregressive):**")
            m_cols = st.columns(3)
            for i, (name, value) in enumerate(cached_full['metrics'].items()):
                with m_cols[i % 3]:
                    if value is not None and not pd.isna(value):
                        st.metric(name, f"{value:.4f}")
        
        with col_window:
            if cached_window:
                st.markdown("**Window Metrics (One-Step):**")
                m_cols = st.columns(3)
                for i, (name, value) in enumerate(cached_window['metrics'].items()):
                    with m_cols[i % 3]:
                        if value is not None and not pd.isna(value):
                            st.metric(name, f"{value:.4f}")
        
        # =====================================================================
        # EXPORT CSV
        # =====================================================================
        st.markdown("---")
        st.markdown("### Export Predictions")
        
        try:
            # Align series for export
            pred_auto = cached_full['pred_auto']
            target = cached_full['target']
            
            # Find temporal intersection
            common_start = max(pred_auto.start_time(), target.start_time())
            common_end = min(pred_auto.end_time(), target.end_time())
            
            pred_aligned = pred_auto.slice(common_start, common_end)
            target_aligned = target.slice(common_start, common_end)
            
            # Check lengths match
            pred_vals = pred_aligned.values().flatten()
            target_vals = target_aligned.values().flatten()
            dates = pred_aligned.time_index
            
            min_len = min(len(dates), len(pred_vals), len(target_vals))
            
            if min_len > 0:
                export_df = pd.DataFrame({
                    'date': dates[:min_len],
                    'ground_truth': target_vals[:min_len],
                    'prediction_autoregressive': pred_vals[:min_len]
                })
                
                col_csv, col_info = st.columns(2)
                with col_csv:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv_data,
                        file_name=f"predictions_{selected_station}_{config.model_name}.csv",
                        mime="text/csv",
                        key="download_csv_predictions"
                    )
                
                with col_info:
                    st.success(f"{len(export_df)} points available")
            else:
                st.warning("No data to export - series do not align.")
                
        except Exception as e:
            st.error(f"Export error: {e}")

else:  # selected_tab == "Explainability"
    st.markdown("### Model Interpretation")
    
    from dashboard.utils.explainability import (
        compute_correlation_importance,
        compute_lag_importance,
        compute_residual_analysis,
        plot_feature_importance_bar,
        plot_lag_importance,
        plot_residual_histogram,
        plot_residual_timeline
    )
    
    # Sub-tabs with persistent state
    if 'explain_subtab' not in st.session_state:
        st.session_state.explain_subtab = "Global Analysis"
        
    sub_tab = st.radio(
        "Sub-Navigation",
        ["Global Analysis", "TimeSHAP", "Quality"],
        horizontal=True,
        key="explain_subtab_radio",
        index=["Global Analysis", "TimeSHAP", "Quality"].index(st.session_state.explain_subtab),
        label_visibility="collapsed"
    )
    st.session_state.explain_subtab = sub_tab
    
    st.markdown("---")
    
    # =========================================================================
    # GLOBAL ANALYSIS
    # =========================================================================
    if sub_tab == "Global Analysis":
        # First show charts
        st.markdown("#### Predictions Overview")
        
        if st.session_state.get(full_pred_key):
            cached = st.session_state[full_pred_key]
            try:
                fig = go.Figure()
                
                # Ground Truth
                fig.add_trace(go.Scatter(
                    x=test_df_raw.index,
                    y=test_df_raw[target_col].values,
                    mode='lines', name='Ground Truth',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                # Predicted (autoregressive)
                pred_auto = cached['pred_auto']
                fig.add_trace(go.Scatter(
                    x=pred_auto.time_index,
                    y=pred_auto.values().flatten(),
                    mode='lines', name='Autoregressive',
                    line=dict(color='#F24236', width=2)
                ))
                
                fig.update_layout(
                    title="Autoregressive Predictions on Test Set",
                    height=350,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("Go back to Predictions tab to generate charts.")
        
        st.markdown("---")
        
        # Feature and Lag importance side-by-side
        col_feat, col_lag = st.columns(2)
        
        with col_feat:
            st.markdown("#### Feature Importance")
            st.caption("Higher bar means stronger correlation with target.")
            
            if use_covariates and covariate_cols:
                corr_importance = compute_correlation_importance(test_df_raw, target_col, covariate_cols)
                if corr_importance:
                    fig = plot_feature_importance_bar(corr_importance, "Correlation with Target")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No covariates in this model.")
        
        with col_lag:
            st.markdown("#### Lag Importance")
            st.caption("Shows autocorrelation: high value at t-7 implies weekly pattern.")
            
            lag_importance = compute_lag_importance(test_df_raw, target_col, max_lag=input_chunk)
            if lag_importance:
                fig = plot_lag_importance(lag_importance, input_chunk)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                peak_lag = max(lag_importance.keys(), key=lambda k: lag_importance[k])
                if peak_lag <= 3:
                    st.success(f"Model relies mostly on last **{peak_lag} days**")
                elif peak_lag == 7:
                    st.info("Weekly pattern detected (peak at t-7)")
                else:
                    st.info(f"Importance peak at **t-{peak_lag}**")
    
    # =========================================================================
    # TIMESHAP TAB
    # =========================================================================
    # =========================================================================
    # TIMESHAP TAB
    # =========================================================================
    elif sub_tab == "TimeSHAP":
        st.markdown("#### Local SHAP Analysis")
        st.caption("Use top slider to navigate between windows")
        
        # Full chart with yellow window AND one-step inside
        fig_full = go.Figure()
        
        # Ground Truth
        fig_full.add_trace(go.Scatter(
            x=test_df_raw.index,
            y=test_df_raw[target_col].values,
            mode='lines', name='Ground Truth',
            line=dict(color='#2E86AB', width=1.5)
        ))
        
        # One-step predictions in window (if available)
        window_pred_key_local = f"window_pred_{selected_model_info.model_path}_{start_idx}"
        cached_window = st.session_state.get(window_pred_key_local)
        if cached_window and cached_window.get('pred_onestep') is not None:
            try:
                pred_window = cached_window['pred_onestep']
                fig_full.add_trace(go.Scatter(
                    x=pred_window.time_index,
                    y=pred_window.values().flatten(),
                    mode='lines+markers', name='One-step',
                    line=dict(color='#2ca02c', width=2.5),
                    marker=dict(size=5)
                ))
            except:
                pass
        
        # Yellow window
        fig_full.add_vrect(
            x0=window_start_date, x1=window_end_date,
            fillcolor="rgba(255, 200, 0, 0.2)", 
            layer="below", line_width=2,
            line_color="orange"
        )
        
        fig_full.update_layout(
            title="Global View - Analyzed Window (Yellow) with One-Step Predictions (Green)",
            height=280,
            margin=dict(l=10, r=10, t=40, b=20),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_full, use_container_width=True)
        
        # Message if no one-step
        window_df = test_df_raw.loc[window_start_date:window_end_date]
        window_pred_key_check = f"window_pred_{selected_model_info.model_path}_{start_idx}"
        cached_check = st.session_state.get(window_pred_key_check)
        # Warning removed as predictions are now auto-generated
        
        st.markdown("---")
        
        st.info("""
        **SHAP Analysis**: Which features and days influenced this prediction?
        - Green = Positive contribution (increases prediction)
        - Red = Negative contribution (decreases prediction)
        """)
        
        # Auto-run SHAP analysis for the current window
        shap_cache_key = f"shap_{selected_model_info.model_path}_{start_idx}"
        
        if shap_cache_key not in st.session_state:
            with st.spinner("Calculating SHAP contributions (this may take a few seconds)..."):
                try:
                    from dashboard.utils.timeshap_wrapper import (
                        DartsModelWrapper,
                        prepare_timeshap_data,
                        compute_timeshap_simple,
                        plot_timeshap_event,
                        plot_timeshap_feature
                    )
                    
                    # Create wrapper
                    model_wrapper = DartsModelWrapper(
                        model=model,
                        input_chunk_length=input_chunk,
                        forecast_horizon=1
                    )
                    
                    # Prepare data
                    window_data, feature_names = prepare_timeshap_data(
                        test_df_raw,
                        target_col,
                        covariate_cols if use_covariates else [],
                        window_start=start_idx,
                        window_length=min(input_chunk, len(test_df_raw) - start_idx, 30)
                    )
                    
                    # Compute SHAP
                    result = compute_timeshap_simple(
                        model_wrapper, 
                        window_data, 
                        feature_names,
                        n_samples=50
                    )
                    
                    if result['success']:
                        st.session_state[shap_cache_key] = result
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown')}")
                        st.session_state[shap_cache_key] = None
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    with st.expander("Details"):
                        st.code(traceback.format_exc())
                    st.session_state[shap_cache_key] = None
        
        # Display results
        if st.session_state.get(shap_cache_key):
            result = st.session_state[shap_cache_key]
            
            if result.get('success'):
                from dashboard.utils.timeshap_wrapper import plot_timeshap_event, plot_timeshap_feature
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Important Past Days")
                    fig = plot_timeshap_event(result['event_data'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Influential Features")
                    fig = plot_timeshap_feature(result['feat_data'], result['feature_names'])
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("👆 Click **Analyze this window** to see SHAP contributions")
    
    # =========================================================================
    # PREDICTION QUALITY (RESIDUALS)
    # =========================================================================
    # =========================================================================
    # PREDICTION QUALITY (RESIDUALS)
    # =========================================================================
    elif sub_tab == "Quality":
        st.markdown("#### Prediction Quality")
        
        st.markdown("""
        > **Residuals** = Model Error = Ground Truth - Predicted
        > - Residuals > 0: Model under-estimates
        > - Residuals < 0: Model over-estimates
        > - Ideally, residuals are centered around 0 (no bias)
        """)
        
        if st.session_state.get(full_pred_key):
            cached = st.session_state[full_pred_key]
            
            try:
                pred = cached['pred_auto']
                target = cached['target']
                
                common_start = max(pred.start_time(), target.start_time())
                common_end = min(pred.end_time(), target.end_time())
                
                pred_slice = pred.slice(common_start, common_end)
                target_slice = target.slice(common_start, common_end)
                
                pred_vals = pred_slice.values().flatten()
                target_vals = target_slice.values().flatten()
                dates = pred_slice.time_index
                
                residuals = target_vals - pred_vals
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mean_res = np.mean(residuals)
                    st.metric("Mean Bias", f"{mean_res:+.4f}", 
                              delta="under-estimates" if mean_res > 0.01 else ("over-estimates" if mean_res < -0.01 else "OK"),
                              delta_color="inverse" if abs(mean_res) > 0.01 else "off")
                with col2:
                    st.metric("Std Dev", f"{np.std(residuals):.4f}")
                with col3:
                    st.metric("Max Error", f"{np.max(np.abs(residuals)):.4f}")
                with col4:
                    # Proportion within ±1 std
                    in_range = np.sum(np.abs(residuals) <= np.std(residuals)) / len(residuals) * 100
                    st.metric("In ±1σ", f"{in_range:.0f}%")
                
                st.markdown("---")
                
                # Charts
                col_hist, col_time = st.columns(2)
                
                with col_hist:
                    st.markdown("##### Error Distribution")
                    st.caption("Good models have errors centered around 0 (red line)")
                    fig = plot_residual_histogram(residuals)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_time:
                    st.markdown("##### Errors over Time")
                    st.caption("Check for patterns (model might be missing something)")
                    fig = plot_residual_timeline(dates, residuals)
                    st.plotly_chart(fig, use_container_width=True)
                

                    
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("Go back to **Predictions** tab to generate charts.")

# Footer
st.markdown("---")
st.caption("Use the top slider to explore different windows.")
