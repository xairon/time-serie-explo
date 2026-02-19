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
# This patch provides compatibility across SHAP versions.
import types
import logging

_shap_patch_logger = logging.getLogger(__name__)
_shap_version = getattr(shap, "__version__", "0.0")

def _version_gte(v1: str, v2: str) -> bool:
    """Compare version strings semantically."""
    try:
        return tuple(int(x) for x in v1.split('.')) >= tuple(int(x) for x in v2.split('.'))
    except (ValueError, AttributeError):
        return False

if _version_gte(_shap_version, "0.43"):
    # Apply patches for SHAP >= 0.43 compatibility
    def _apply_shap_patch():
        """Apply SHAP compatibility patch for TimeSHAP. Returns True if successful."""
        try:
            from shap import KernelExplainer

            # 1. Try to get existing module or create new one
            try:
                from shap.explainers import _kernel
            except ImportError:
                _kernel = types.ModuleType("shap.explainers._kernel")
                sys.modules["shap.explainers._kernel"] = _kernel
                if hasattr(shap, "explainers"):
                    shap.explainers._kernel = _kernel

            # 2. Inject Kernel attribute if missing
            if not hasattr(_kernel, "Kernel"):
                _kernel.Kernel = KernelExplainer

            # 3. Ensure sys.modules entry exists
            if "shap.explainers._kernel" not in sys.modules:
                sys.modules["shap.explainers._kernel"] = _kernel

            return True
        except ImportError as e:
            _shap_patch_logger.warning(f"SHAP not available for patching: {e}")
            return False
        except Exception as e:
            _shap_patch_logger.warning(f"Failed to patch SHAP for TimeSHAP compatibility: {e}")
            return False

    _SHAP_PATCHED = _apply_shap_patch()
else:
    logging.getLogger(__name__).warning(f"SHAP {_shap_version} non teste, patch non applique")
    _SHAP_PATCHED = False
# ---------------------------------------------

# Add project root to path
_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dashboard.config import CHECKPOINTS_DIR

# Infobulles métriques (échelle, calcul, interprétation) — défini ici pour ne pas dépendre de config
try:
    from dashboard.config import METRIC_TOOLTIPS
except ImportError:
    METRIC_TOOLTIPS = {
        'MAE': "Échelle : même unité que la cible (ex. m). Calcul : moyenne des |vrai − prédit|. Interprétation : plus c'est bas, mieux c'est ; erreur typique par pas.",
        'RMSE': "Échelle : même unité que la cible. Calcul : racine de la moyenne des carrés des erreurs. Interprétation : plus c'est bas, mieux c'est ; pénalise plus les grosses erreurs que le MAE.",
        'sMAPE': "Échelle : %. Calcul : moyenne de 2|prédit−vrai|/(|vrai|+|prédit|), en %. Interprétation : plus c'est bas, mieux c'est ; ≤10% bon, 10–20% moyen, >20% faible.",
        'WAPE': "Échelle : %. Calcul : somme des |erreurs| / somme des |vrais|, en %. Interprétation : plus c'est bas, mieux c'est ; plus stable que MAPE si la série passe près de 0.",
        'NRMSE': "Échelle : % (RMSE rapportée à l'amplitude). Calcul : RMSE / (max−min) × 100. Interprétation : plus c'est bas, mieux c'est ; ≤10% bon, 10–20% moyen, >20% faible.",
        'Dir_Acc': "Échelle : %. Calcul : part des pas où la direction (montée/descente) est correcte. Interprétation : plus c'est haut, mieux c'est ; 50% = hasard, >50% utile.",
        'NSE': "Échelle : sans unité (souvent entre −∞ et 1). Calcul : 1 − (variance des erreurs / variance des observations). Interprétation : 1 = parfait, 0 = comme prédire la moyenne, <0 = pire ; >0,75 bon, 0,5–0,75 moyen.",
        'KGE': "Échelle : sans unité. Calcul : combine corrélation, biais et variabilité. Interprétation : 1 = parfait, 0 = moyen, <0 = mauvais ; >0,75 bon, 0,5–0,75 moyen.",
    }

from dashboard.utils.model_registry import get_registry
# from dashboard.utils.model_config import load_model_with_config, load_scalers # Removed (Legacy)
from dashboard.utils.forecasting import generate_single_window_forecast
from dashboard.utils.preprocessing import prepare_dataframe_for_darts
from darts.metrics import mae, rmse, smape

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
        # Get registry
        registry = get_registry(CHECKPOINTS_DIR.parent)
        
        # Load via registry
        model = registry.load_model(model_entry)
        scalers = registry.load_scalers(model_entry)
        model_config = registry.load_model_config(model_entry)
        
        # Load datasets
        data_dict = {}
        for split in ['train', 'val', 'test']:
            data_dict[split] = registry.load_data(model_entry, split)
            # Also try to load raw if available, otherwise it will be generated later
            try:
                data_dict[f'{split}_raw'] = registry.load_data(model_entry, f'{split}_raw')
            except Exception:
                pass

        # Load covariates if available
        for split in ['train_cov', 'val_cov', 'test_cov']:
            try:
                data_dict[split] = registry.load_data(model_entry, split)
            except Exception:
                pass

        # Reconstruct config object for compatibility
        config = type('Config', (), {})()
        config.model_name = model_entry.model_name
        config.hyperparams = model_entry.hyperparams
        config.metrics = model_entry.metrics
        config.preprocessing = model_entry.preprocessing_config
        # Add columns info to config from preprocessing if available
        config.columns = model_entry.preprocessing_config.get('columns', {})
        config.use_covariates = bool(config.columns.get('covariates'))

        # Override from model_config.json if available (authoritative for metrics/columns)
        if model_config:
            config.metrics = model_config.get('metrics', config.metrics)
            config.metrics_sliding = model_config.get('metrics_sliding', getattr(config, 'metrics_sliding', {}))
            config.preprocessing = model_config.get('preprocessing', config.preprocessing)
            config.columns = model_config.get('columns', config.columns)
            config.use_covariates = bool(config.columns.get('covariates'))
        
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

# Get registry
registry = get_registry(CHECKPOINTS_DIR.parent)  # checkpoints/ is parent

# Scan for unregistered models - REMOVED (MLflow handles this automatically)
# newly_registered = registry.scan_existing_checkpoints()
# if newly_registered > 0:
#     st.sidebar.info(f"📦 Found and registered {newly_registered} new model(s)")

all_models = registry.list_all_models()

if not all_models:
    st.warning("No trained models available.")
    st.info("Please train a model first on the **Train Models** page.")
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
        st.write(f"CHECKPOINTS_DIR.parent: {CHECKPOINTS_DIR.parent}")
        st.write(f"Registry checkpoints_dir: {registry.checkpoints_dir}")
        
        # Check if directory exists
        if CHECKPOINTS_DIR.exists():
            st.write(f"✅ CHECKPOINTS_DIR exists")
            # List subdirectories
            subdirs = [d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()]
            st.write(f"Subdirectories: {subdirs[:10]}")  # Show first 10
        else:
            st.write(f"❌ CHECKPOINTS_DIR does not exist")
        
        # Check registry connection
        if registry.mlflow_manager.experiment:
             st.write(f"✅ MLflow Experiment: {registry.mlflow_manager.experiment_name}")
        else:
             st.write(f"❌ MLflow Experiment not found")
    
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
        scaler_label = preproc.get('scaler_type') or preproc.get('normalization') or "N/A"
        st.markdown(f"**Scaler:** {scaler_label}")
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

# Extract configuration FIRST (needed for raw data generation)
model_horizon = getattr(model, 'output_chunk_length', 30)
input_chunk = getattr(model, 'input_chunk_length', 30)

# Safely get target column with fallback
if not hasattr(config, 'columns') or not config.columns:
    st.error("Model configuration missing 'columns' info. Please retrain the model.")
    st.stop()

target_col = config.columns.get('target')
if not target_col:
    # Try to infer from data columns
    non_cov_cols = [c for c in data_dict['train'].columns if not c.startswith('lag_') and not c.endswith('_sin') and not c.endswith('_cos')]
    if non_cov_cols:
        target_col = non_cov_cols[0]
        st.warning(f"Target column not found in config, using '{target_col}'")
    else:
        st.error("Cannot determine target column from model configuration or data.")
        st.stop()

covariate_cols = config.columns.get('covariates', [])
use_covariates = getattr(config, 'use_covariates', bool(covariate_cols))

# Merge covariates into processed data if available
def _merge_covariates(data_dict_in: dict) -> dict:
    data_dict_out = data_dict_in.copy()
    for split in ['train', 'val', 'test']:
        cov_key = f"{split}_cov"
        base_df = data_dict_out.get(split)
        cov_df = data_dict_out.get(cov_key)
        if isinstance(base_df, pd.DataFrame) and isinstance(cov_df, pd.DataFrame):
            cov_df = cov_df.drop(columns=['station'], errors='ignore')
            data_dict_out[split] = base_df.join(cov_df, how='left')
    return data_dict_out

if use_covariates:
    data_dict = _merge_covariates(data_dict)

# Full processed dataframe for predictions
# IMPORTANT: Always use train+val+test concatenation, NOT 'full' (full_data.csv)!
# full_data.csv contains RAW data, while train/val/test.csv contain NORMALIZED data.
full_df_processed = pd.concat([data_dict['train'], data_dict['val'], data_dict['test']])
full_df_processed = full_df_processed.sort_index()

# Test sets (processed)
test_df_processed = data_dict['test'].sort_index()

# Validate covariates availability
missing_covariates = [c for c in covariate_cols if c not in full_df_processed.columns]
if use_covariates and missing_covariates:
    st.error(
        "Covariates missing in loaded artifacts. "
        f"Missing columns: {missing_covariates}. "
        "Please retrain the model so covariates are saved to MLflow."
    )
    st.stop()

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
        raw_df = raw_series.to_dataframe()
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
        if not test_df_processed.index.equals(test_df_raw.index):
            import logging
            logging.getLogger(__name__).warning(
                "Index mismatch between processed and raw data, keeping processed index"
            )

# Raw dataframe for display - also generate via inverse_transform if not available
if 'train_raw' in data_dict and 'val_raw' in data_dict and 'test_raw' in data_dict:
    full_df_raw = pd.concat([data_dict['train_raw'], data_dict['val_raw'], data_dict['test_raw']])
else:
    # Generate raw data via inverse_transform
    full_df_raw = generate_raw_from_processed(full_df_processed, scalers, target_col)
full_df_raw = full_df_raw.sort_index()

# Derive scale statistics for actionnable thresholds
target_series_raw_vals = test_df_raw[target_col].values.astype(float) if target_col in test_df_raw.columns else None
scale_stats = {}
if target_series_raw_vals is not None and len(target_series_raw_vals) > 0:
    q25 = float(np.percentile(target_series_raw_vals, 25))
    q75 = float(np.percentile(target_series_raw_vals, 75))
    iqr = float(q75 - q25)
    mean_abs = float(np.mean(np.abs(target_series_raw_vals)))
    std = float(np.std(target_series_raw_vals))
    scale_stats = {
        "mean_abs": mean_abs,
        "std": std,
        "iqr": iqr,
    }


# =============================================================================
# INFO PANELS
# =============================================================================
col_data, col_model = st.columns(2)

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

# Evict old cached predictions to prevent memory leak (keep last 10)
_pred_prefix = f"window_pred_{selected_model_info.model_id}_{target_station}_"
_cached_keys = [k for k in st.session_state if isinstance(k, str) and k.startswith(_pred_prefix)]
if len(_cached_keys) > 10:
    _sorted_keys = sorted(_cached_keys, key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else 0)
    for _old_key in _sorted_keys[:-10]:
        del st.session_state[_old_key]


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
# PREDICTION CHART - Always visible (both tabs)
# =============================================================================
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

# 3. Ground Truth - Test set only
fig.add_trace(go.Scatter(
    x=test_series_raw.time_index,
    y=test_series_raw.values().flatten(),
    mode='lines',
    name='Ground Truth',
    line=dict(color='#2E86AB', width=2)
))

# 4. Window Forecast (if available)
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
    height=400,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 1: PREDICTIONS - Metrics and Export
# =============================================================================
if selected_tab == "Predictions":

    # Metrics Display
    st.markdown("### Window Metrics")
    
    if cached_window and cached_window.get('prediction') is not None and cached_window.get('target') is not None:
        pred = cached_window['prediction']
        target = cached_window['target']
        
        min_len = min(len(pred), len(target))
        if min_len > 0:
            pred_aligned = pred[:min_len]
            target_aligned = target[:min_len]
            
            # Window metrics
            metrics = {
                'MAE': float(mae(target_aligned, pred_aligned)),
                'RMSE': float(rmse(target_aligned, pred_aligned)),
                'sMAPE': float(smape(target_aligned, pred_aligned)),
                'WAPE': float(np.sum(np.abs(target_aligned.values().flatten() - pred_aligned.values().flatten()))
                              / np.sum(np.abs(target_aligned.values().flatten())) * 100.0) if np.sum(np.abs(target_aligned.values().flatten())) != 0 else np.nan,
                'NRMSE': float(np.sqrt(np.mean((target_aligned.values().flatten() - pred_aligned.values().flatten()) ** 2))
                               / (np.max(target_aligned.values().flatten()) - np.min(target_aligned.values().flatten())) * 100.0)
                if (np.max(target_aligned.values().flatten()) - np.min(target_aligned.values().flatten())) != 0 else np.nan,
                'NSE': nash_sutcliffe_efficiency(target_aligned, pred_aligned),
                'KGE': kling_gupta_efficiency(target_aligned, pred_aligned),
            }

            display_order = ["MAE", "RMSE", "NRMSE", "sMAPE", "WAPE", "NSE", "KGE"]
            percent_metrics = {"sMAPE", "WAPE", "NRMSE"}
            m_cols = st.columns(4)
            for i, name in enumerate(display_order):
                value = metrics.get(name)
                with m_cols[i % 4]:
                    if value is not None and not pd.isna(value):
                        suffix = " %" if name in percent_metrics else ""
                        tip = METRIC_TOOLTIPS.get(name, "")
                        st.metric(name, f"{value:.4f}{suffix}", help=tip or None)

            if scale_stats:
                scale_ref = scale_stats["iqr"] if scale_stats["iqr"] > 0 else scale_stats["std"]
                if scale_ref == 0:
                    scale_ref = scale_stats["mean_abs"] if scale_stats["mean_abs"] > 0 else None
                if scale_ref:
                    rel_mae = float(metrics["MAE"] / scale_ref * 100.0) if metrics.get("MAE") is not None else None
                    rel_rmse = float(metrics["RMSE"] / scale_ref * 100.0) if metrics.get("RMSE") is not None else None
                    st.caption(
                        f"MAE ≈ {rel_mae:.1f}% et RMSE ≈ {rel_rmse:.1f}% de l’échelle (IQR/σ)."
                        if rel_mae is not None and rel_rmse is not None else
                        "MAE/RMSE relatifs non disponibles."
                    )
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
# TAB 2: EXPLAINABILITY (Enhanced with 6 tabs - State Managed)
# =============================================================================
else:
    st.markdown("### Model Interpretation")

    # Import explainability modules
    from dashboard.utils.explainability import (
        ModelExplainerFactory,
        ModelType,
        compute_correlation_importance,
        compute_permutation_importance_safe,
        compute_lag_importance,
        compute_residual_analysis,
        detect_seasonality_patterns,
        analyze_prediction_decomposition,
        plot_feature_importance_bar,
        plot_lag_importance,
        plot_temporal_saliency_heatmap,
        plot_shap_waterfall,
        plot_decomposition_comparison,
        plot_seasonality_patterns,
        plot_residual_analysis,
    )

    # Get model type and available methods
    model_type = ModelType.from_model(model)
    explainer = ModelExplainerFactory.get_explainer(model, input_chunk, model_horizon)
    available_methods = explainer.get_available_methods()

    # Model-specific info
    model_type_display = {
        ModelType.TFT: "TFT (Attention + Variable Selection)",
        ModelType.TSMIXER: "TSMixer (Gradient-based)",
        ModelType.NHITS: "NHiTS (Multi-scale)",
        ModelType.NBEATS: "NBEATS (Interpretable)",
        ModelType.LSTM: "LSTM/GRU (Temporal)",
        ModelType.TRANSFORMER: "Transformer (Attention)",
    }.get(model_type, "Generic")

    st.caption(f"**Model Type:** {model_type_display} | **Available methods:** {', '.join(available_methods)}")

    # State-managed tab navigation (persists across reruns)
    EXPLAIN_TABS = [
        "Overview",
        "Feature Importance",
        "Temporal Analysis",
        "Seasonality & Trends",
        "Model Internals",
        "Local Explanation"
    ]

    if 'explain_tab_index' not in st.session_state:
        st.session_state.explain_tab_index = 0

    # Tab navigation using columns of buttons for better UX
    tab_cols = st.columns(len(EXPLAIN_TABS))
    for i, (col, tab_name) in enumerate(zip(tab_cols, EXPLAIN_TABS)):
        with col:
            # Use button styling to show active tab
            button_type = "primary" if st.session_state.explain_tab_index == i else "secondary"
            if st.button(tab_name, key=f"explain_tab_{i}", type=button_type, use_container_width=True):
                st.session_state.explain_tab_index = i
                st.rerun()

    st.markdown("---")

    # Get current tab
    current_explain_tab = EXPLAIN_TABS[st.session_state.explain_tab_index]

    # Prepare common data
    cached_window = st.session_state.get(window_pred_key)
    display_start = window_start_date - timedelta(days=input_chunk)
    display_end = window_end_date
    window_mask = (test_df_raw.index >= display_start) & (test_df_raw.index <= display_end)
    window_df = test_df_raw.loc[window_mask].copy()

    # Filter computed features for cleaner display
    computed_patterns = ['day_of_week', 'month', 'sin', 'cos', 'weekday', '_lag_']
    base_covariates = [
        col for col in covariate_cols
        if col in window_df.columns and not any(p in col.lower() for p in computed_patterns)
    ]

    # =========================================================================
    # TAB 1: OVERVIEW - Summary + Top 5 Features
    # =========================================================================
    if current_explain_tab == "Overview":
        st.markdown("#### Explainability Overview")
        st.info("Quick summary of model explainability with top contributing features.")

        col1, col2 = st.columns([2, 1])

        correlations = {}

        with col1:
            # Top 5 features (correlation-based - always available)
            if len(window_df) > 0 and len(base_covariates) > 0:
                correlations = compute_correlation_importance(
                    window_df, target_col, base_covariates
                )

                if correlations:
                    top_5 = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5])
                    fig = plot_feature_importance_bar(top_5, title="Top 5 Features (Correlation)", top_k=5)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for feature importance.")
            else:
                st.info("No covariates available.")

        with col2:
            st.markdown("##### Quick Insights")

            # Summary statistics
            if correlations:
                top_feature = list(correlations.keys())[0] if correlations else "N/A"
                top_score = list(correlations.values())[0] if correlations else 0
                st.metric("Top Feature", top_feature, f"r={top_score:.2f}")

            # Lag analysis preview
            if len(window_df) > 5:
                lag_imp = compute_lag_importance(window_df, target_col, max_lag=min(14, len(window_df) // 2))
                if lag_imp:
                    peak_lag = max(lag_imp.keys(), key=lambda k: lag_imp[k])
                    st.metric("Most Important Lag", f"t-{peak_lag}", f"r={lag_imp[peak_lag]:.2f}")

            # Model capabilities
            st.markdown("##### Model Capabilities")
            caps = []
            if "attention" in available_methods:
                caps.append("Attention")
            if "integrated_gradients" in available_methods:
                caps.append("Gradients")
            if "shap" in available_methods:
                caps.append("SHAP")
            st.write(", ".join(caps) if caps else "Correlation only")

    # =========================================================================
    # TAB 2: FEATURE IMPORTANCE - Correlation/Permutation/SHAP
    # =========================================================================
    elif current_explain_tab == "Feature Importance":
        st.markdown("#### Feature Importance Analysis")

        method_tab = st.radio(
            "Method",
            ["Correlation", "Permutation", "SHAP"],
            horizontal=True,
            key="feat_imp_method"
        )

        if method_tab == "Correlation":
            st.caption("Fast correlation-based importance (no model inference required)")

            if len(base_covariates) > 0:
                # Local (window)
                st.markdown("##### Local (Current Window)")
                if len(window_df) > 0:
                    local_corr = compute_correlation_importance(window_df, target_col, base_covariates)
                    if local_corr:
                        fig = plot_feature_importance_bar(local_corr, title="Window Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)

                # Global (test set)
                st.markdown("##### Global (Test Set)")
                global_corr = compute_correlation_importance(test_df_processed, target_col,
                    [c for c in covariate_cols if c in test_df_processed.columns])
                if global_corr:
                    fig = plot_feature_importance_bar(global_corr, title="Global Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No covariates available for analysis.")

        elif method_tab == "Permutation":
            st.caption("Measures prediction degradation when shuffling feature values")

            if st.button("Compute Permutation Importance", key="perm_btn"):
                with st.spinner("Computing permutation importance (this may take a moment)..."):
                    try:
                        # Prepare series for permutation
                        target_series, cov_series = prepare_dataframe_for_darts(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns]
                        )

                        perm_imp = compute_permutation_importance_safe(
                            model, target_series, cov_series,
                            n_permutations=3,
                            output_chunk_length=model_horizon
                        )

                        if perm_imp and "_error" in perm_imp:
                            st.error(f"Permutation importance error: {perm_imp['_error']}")
                        elif perm_imp:
                            fig = plot_feature_importance_bar(perm_imp, title="Permutation Importance")
                            st.plotly_chart(fig, use_container_width=True)

                            # Table view
                            perm_df = pd.DataFrame({
                                'Feature': list(perm_imp.keys()),
                                'Importance': list(perm_imp.values())
                            }).sort_values('Importance', ascending=False)
                            st.dataframe(perm_df.style.format({'Importance': '{:.3%}'}), use_container_width=True)
                        else:
                            st.warning("Permutation importance computation failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Click button to compute permutation importance.")

        else:  # SHAP
            st.caption("SHAP values showing directional feature contributions")
            st.warning("SHAP computation can be slow. Consider using Correlation or Permutation for faster results.")

            if st.button("Compute SHAP Values", key="shap_btn"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        from dashboard.utils.timeshap_wrapper import (
                            DartsModelWrapper, prepare_timeshap_data, compute_shap_perturbation
                        )

                        # Prepare wrapper
                        wrapper = DartsModelWrapper(model, input_chunk, 1)

                        # Prepare data
                        feature_cols_shap = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                        window_start_idx = max(0, start_idx - input_chunk)
                        data_3d, feat_names = prepare_timeshap_data(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns],
                            window_start=window_start_idx,
                            window_length=min(input_chunk, len(test_df_processed) - window_start_idx)
                        )

                        result = compute_shap_perturbation(wrapper, data_3d, feat_names, n_samples=50)

                        if result.get('feat_data') is not None:
                            from dashboard.utils.timeshap_wrapper import plot_feature_importance
                            fig = plot_feature_importance(result['feat_data'])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("SHAP computation did not return feature data.")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")
            else:
                st.info("Click button to compute SHAP values.")

    # =========================================================================
    # TAB 3: TEMPORAL ANALYSIS - Lag importance + Saliency
    # =========================================================================
    elif current_explain_tab == "Temporal Analysis":
        st.markdown("#### Temporal Analysis")
        st.caption("Which past timesteps influence the prediction most?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Lag Importance (Autocorrelation)")
            if len(window_df) > 5:
                max_lag = min(input_chunk, len(window_df) // 2)
                lag_imp = compute_lag_importance(window_df, target_col, max_lag=max_lag)

                if lag_imp:
                    fig = plot_lag_importance(lag_imp, input_chunk_length=input_chunk)
                    st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    peak_lag = max(lag_imp.keys(), key=lambda k: lag_imp[k])
                    recent = [v for k, v in lag_imp.items() if k <= 7]
                    distant = [v for k, v in lag_imp.items() if k > 7]

                    if recent and distant:
                        if np.mean(recent) > np.mean(distant) * 1.3:
                            st.success("Recent days (t-1 to t-7) are most influential.")
                        elif np.mean(distant) > np.mean(recent) * 1.3:
                            st.info("Longer history (>7 days) is more influential.")
                        else:
                            st.info("Influence is distributed across the input window.")
                else:
                    st.info("Not enough data for lag analysis.")
            else:
                st.info("Window too small for lag analysis.")

        with col2:
            st.markdown("##### Gradient-Based Saliency")

            if "saliency" in available_methods or "integrated_gradients" in available_methods:
                if st.button("Compute Temporal Saliency", key="saliency_btn"):
                    with st.spinner("Computing gradients..."):
                        try:
                            target_series, cov_series = prepare_dataframe_for_darts(
                                test_df_processed, target_col,
                                [c for c in covariate_cols if c in test_df_processed.columns]
                            )

                            result = explainer.explain_local(target_series, cov_series)

                            if result.success and result.gradient_attributions is not None:
                                feature_names = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                                fig = plot_temporal_saliency_heatmap(
                                    result.gradient_attributions,
                                    feature_names[:result.gradient_attributions.shape[1]],
                                    title="Feature x Time Attribution",
                                    input_chunk_length=input_chunk
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            elif result.temporal_importance is not None:
                                # Fallback to temporal importance only
                                lag_dict = {i + 1: float(v) for i, v in enumerate(result.temporal_importance)}
                                fig = plot_lag_importance(lag_dict, input_chunk)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Gradient computation failed: {result.error_message}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.info("Click to compute gradient-based temporal importance.")
            else:
                st.info("Gradient methods not available for this model type.")

    # =========================================================================
    # TAB 4: SEASONALITY & TRENDS - STL Decomposition
    # =========================================================================
    elif current_explain_tab == "Seasonality & Trends":
        st.markdown("#### Seasonality & Trend Analysis")
        st.caption("Decompose time series into trend, seasonal, and residual components.")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Seasonality detection
            st.markdown("##### Detected Seasonality Patterns")

            if len(test_df_raw) > 60:
                target_series = test_df_raw[target_col]
                seasonality = detect_seasonality_patterns(target_series, periods=[7, 30, 365])

                fig = plot_seasonality_patterns(seasonality, title="Seasonality Detection")
                st.plotly_chart(fig, use_container_width=True)

                # Summary
                detected = [k for k, v in seasonality.items() if v.get('detected', False)]
                if detected:
                    st.success(f"Detected patterns: {', '.join(detected)}")
                else:
                    st.info("No significant seasonality detected.")
            else:
                st.info("Need at least 60 days of data for seasonality analysis.")

        with col2:
            st.markdown("##### Variance Decomposition")

            if st.button("Compute STL Decomposition", key="stl_btn"):
                with st.spinner("Computing STL decomposition..."):
                    try:
                        from dashboard.utils.statistics import stl_decomposition

                        # Use appropriate period
                        series_len = len(test_df_raw)
                        period = 7 if series_len < 365 else 365

                        decomp = stl_decomposition(test_df_raw[target_col], seasonal=period)

                        if decomp:
                            var = decomp.get('variance', {})
                            st.metric("Trend", f"{var.get('trend_pct', 0):.1f}%")
                            st.metric("Seasonal", f"{var.get('seasonal_pct', 0):.1f}%")
                            st.metric("Residual", f"{var.get('residual_pct', 0):.1f}%")
                    except Exception as e:
                        st.error(f"STL error: {e}")
            else:
                st.info("Click to compute decomposition.")

        # Decomposition comparison (if predictions available)
        if cached_window and cached_window.get('prediction') is not None:
            st.markdown("---")
            st.markdown("##### Actual vs Predicted Decomposition")

            if st.button("Compare Decompositions", key="decomp_compare_btn"):
                with st.spinner("Comparing decompositions..."):
                    try:
                        pred = cached_window['prediction']
                        target = cached_window['target']

                        # Get aligned values
                        common_idx = pred.time_index.intersection(target.time_index)
                        if len(common_idx) > 14:
                            actual_series = pd.Series(
                                target.values().flatten()[:len(common_idx)],
                                index=common_idx
                            )
                            pred_series = pd.Series(
                                pred.values().flatten()[:len(common_idx)],
                                index=common_idx
                            )

                            result = analyze_prediction_decomposition(
                                actual_series, pred_series, period=7
                            )

                            if result.get('success'):
                                st.write(f"**Trend correlation:** {result['comparison']['trend_correlation']:.3f}")
                                st.write(f"**Seasonal correlation:** {result['comparison']['seasonal_correlation']:.3f}")

                                # Interpretation
                                for key, interp in result.get('interpretation', {}).items():
                                    st.info(f"**{key.capitalize()}**: {interp}")
                            else:
                                st.warning(f"Decomposition comparison failed: {result.get('error')}")
                        else:
                            st.info("Not enough overlapping data for comparison.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # =========================================================================
    # TAB 5: MODEL INTERNALS - Model-specific explanations
    # =========================================================================
    elif current_explain_tab == "Model Internals":
        st.markdown("#### Model-Specific Internals")
        st.caption(f"Explanations specific to **{model_type_display}**")

        if model_type == ModelType.TFT:
            st.markdown("##### TFT Attention & Variable Selection")
            st.info("TFT provides built-in attention weights and variable selection network outputs.")

            if st.button("Extract TFT Attention", key="tft_attention_btn"):
                with st.spinner("Extracting TFT attention..."):
                    try:
                        from dashboard.utils.explainability.attention import TFTExplainer as TFTAttentionExplainer

                        target_series, cov_series = prepare_dataframe_for_darts(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns]
                        )

                        tft_explainer = TFTAttentionExplainer(model)
                        result = tft_explainer.explain(target_series, cov_series)

                        if result.get('success'):
                            # Encoder importance
                            if result.get('encoder_importance'):
                                st.markdown("**Encoder Variable Importance:**")
                                st.json(result['encoder_importance'])

                            # Attention heatmap
                            if result.get('attention') is not None:
                                from dashboard.utils.explainability.visualizations import plot_attention_heatmap
                                fig = plot_attention_heatmap(result['attention'], title="TFT Attention Weights")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"TFT extraction failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Click to extract TFT-specific explanations.")

        elif model_type == ModelType.TSMIXER:
            st.markdown("##### TSMixer Gradient Analysis")
            st.info("TSMixer doesn't have native attention. Using Integrated Gradients for feature attribution.")

            if st.button("Compute TSMixer Attributions", key="tsmixer_btn"):
                with st.spinner("Computing Integrated Gradients..."):
                    try:
                        target_series, cov_series = prepare_dataframe_for_darts(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns]
                        )

                        result = explainer.explain_local(target_series, cov_series)

                        if result.success:
                            if result.gradient_attributions is not None:
                                feature_names = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                                fig = plot_temporal_saliency_heatmap(
                                    result.gradient_attributions,
                                    feature_names[:result.gradient_attributions.shape[1]],
                                    title="TSMixer Feature x Time Attribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            if result.feature_importance:
                                st.markdown("**Feature Importance (from gradients):**")
                                fig = plot_feature_importance_bar(result.feature_importance)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"TSMixer attribution failed: {result.error_message}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Click to compute TSMixer-specific attributions.")

        elif model_type in (ModelType.NHITS, ModelType.NBEATS):
            st.markdown(f"##### {model_type.name} Stack Analysis")

            if model_type == ModelType.NHITS:
                st.info("NHiTS uses multi-scale hierarchical interpolation with multiple stacks.")
            else:
                st.info("NBEATS can use interpretable stacks (trend/seasonal) or generic stacks.")

            if st.button(f"Analyze {model_type.name} Structure", key="stack_btn"):
                with st.spinner("Analyzing model structure..."):
                    try:
                        target_series, cov_series = prepare_dataframe_for_darts(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns]
                        )

                        result = explainer.explain_local(target_series, cov_series)

                        if result.success:
                            if result.decomposition:
                                st.markdown("**Model Structure:**")
                                st.json(result.decomposition)

                            if result.gradient_attributions is not None:
                                feature_names = [target_col] + [c for c in covariate_cols if c in test_df_processed.columns]
                                fig = plot_temporal_saliency_heatmap(
                                    result.gradient_attributions,
                                    feature_names[:result.gradient_attributions.shape[1]],
                                    title=f"{model_type.name} Attribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Analysis failed: {result.error_message}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        else:
            st.markdown("##### Generic Model Analysis")
            st.info("Using gradient-based methods (if available) or correlation-based analysis.")

            if st.button("Run Generic Analysis", key="generic_btn"):
                with st.spinner("Running analysis..."):
                    try:
                        target_series, cov_series = prepare_dataframe_for_darts(
                            test_df_processed, target_col,
                            [c for c in covariate_cols if c in test_df_processed.columns]
                        )

                        result = explainer.explain_local(target_series, cov_series)

                        if result.success and result.feature_importance:
                            fig = plot_feature_importance_bar(result.feature_importance, title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Using correlation-based fallback.")
                            corr = compute_correlation_importance(
                                test_df_processed, target_col,
                                [c for c in covariate_cols if c in test_df_processed.columns]
                            )
                            if corr:
                                fig = plot_feature_importance_bar(corr)
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # =========================================================================
    # TAB 6: LOCAL EXPLANATION - Waterfall for single prediction
    # =========================================================================
    elif current_explain_tab == "Local Explanation":
        st.markdown("#### Local Explanation")
        st.caption("Deep-dive into a specific prediction window.")

        # Context visualization
        if len(window_df) > 0:
            st.markdown("##### Prediction Context")

            fig_context = go.Figure()
            fig_context.add_vrect(
                x0=window_start_date, x1=window_end_date,
                fillcolor="rgba(255, 200, 0, 0.15)", line_color="orange",
                annotation_text="Prediction", annotation_position="top right"
            )
            fig_context.add_vline(x=window_start_date, line_dash="dash", line_color="orange")

            fig_context.add_trace(go.Scatter(
                x=window_df.index, y=window_df[target_col].values,
                mode='lines+markers', name='Ground Truth',
                line=dict(color='#2E86AB'), marker=dict(size=4)
            ))

            if cached_window and cached_window.get('prediction') is not None:
                pred = cached_window['prediction']
                fig_context.add_trace(go.Scatter(
                    x=pred.time_index, y=pred.values().flatten(),
                    mode='lines+markers', name='Prediction',
                    line=dict(color='#28A745'), marker=dict(size=6, symbol='diamond')
                ))

            fig_context.update_layout(
                title=f"Context ({input_chunk}d) + Prediction ({model_horizon}d)",
                height=300, margin=dict(l=10, r=10, t=40, b=20),
                hovermode='x unified'
            )
            st.plotly_chart(fig_context, use_container_width=True)

        st.markdown("---")

        # Waterfall explanation
        st.markdown("##### Feature Contributions (Waterfall)")

        if st.button("Compute Local SHAP Waterfall", key="waterfall_btn"):
            with st.spinner("Computing local SHAP values..."):
                try:
                    from dashboard.utils.timeshap_wrapper import (
                        DartsModelWrapper, prepare_timeshap_data, compute_shap_perturbation
                    )

                    wrapper = DartsModelWrapper(model, input_chunk, 1)

                    window_start_idx = max(0, start_idx - input_chunk)
                    window_len = min(input_chunk, len(test_df_processed) - window_start_idx)

                    data_3d, feat_names = prepare_timeshap_data(
                        test_df_processed, target_col,
                        [c for c in covariate_cols if c in test_df_processed.columns],
                        window_start=window_start_idx,
                        window_length=window_len
                    )

                    result = compute_shap_perturbation(wrapper, data_3d, feat_names, n_samples=50)

                    if 'feature_importance' in result or 'feat_data' in result:
                        # Get feature importance dict
                        if 'feature_importance' in result:
                            feat_imp = result['feature_importance']
                        else:
                            feat_df = result['feat_data']
                            feat_imp = dict(zip(feat_df['Feature'], feat_df['Shapley Value']))

                        base_val = result.get('baseline_pred', 0)

                        fig = plot_shap_waterfall(feat_imp, base_value=base_val, title="Feature Contributions")
                        st.plotly_chart(fig, use_container_width=True)

                        # Top contributors table
                        st.markdown("**Top Contributors:**")
                        sorted_feat = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                        for feat, val in sorted_feat:
                            sign = "+" if val > 0 else ""
                            st.write(f"- **{feat}**: {sign}{val:.4f}")
                    else:
                        st.warning("Could not compute waterfall data.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Click to compute waterfall explanation for the current window.")

        # Residual analysis
        if cached_window and cached_window.get('prediction') is not None:
            st.markdown("---")
            st.markdown("##### Residual Analysis")

            pred = cached_window['prediction']
            target = cached_window['target']

            min_len = min(len(pred), len(target))
            actual_vals = target.values().flatten()[:min_len]
            pred_vals = pred.values().flatten()[:min_len]

            residual_stats = compute_residual_analysis(actual_vals, pred_vals)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Error", f"{residual_stats['mean']:.4f}")
            with col2:
                st.metric("Std Error", f"{residual_stats['std']:.4f}")
            with col3:
                bias_status = "Biased" if residual_stats['is_biased'] else "Balanced"
                st.metric("Bias Status", bias_status)

            fig = plot_residual_analysis(residual_stats['residuals'], pred.time_index[:min_len])
            st.plotly_chart(fig, use_container_width=True)

