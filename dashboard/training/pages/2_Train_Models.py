"""Model training page with extended Darts model support."""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.models_config import ALL_MODELS, MODEL_CATEGORIES, RECOMMENDED_MODELS, get_model_info, get_hyperparam_description
from dashboard.utils.preprocessing import (
    TimeSeriesPreprocessor,
    prepare_dataframe_for_darts,
    split_train_val_test,
    compute_data_statistics,
    get_preprocessing_summary
)
from dashboard.utils.model_factory import ModelFactory
from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.dataset_registry import get_dataset_registry
from dashboard.utils.training import run_training_pipeline
from dashboard.utils.training_monitor import TrainingMonitor, create_training_monitor_fragment
import threading
from dashboard.utils.export import add_download_button
from dashboard.components.live_log import LiveLogManager, create_progress_section, TrainingProgressTracker
import time

# Thread lock for training state updates to prevent race conditions
_training_lock = threading.Lock()


st.set_page_config(page_title="Train Models", page_icon="", layout="wide")

st.title(" Train Models")
st.markdown("Forecasting model training with configurable preprocessing.")
st.markdown("---")


# ============================================================================
# OPTION: Load a prepared dataset
# ============================================================================
dataset_registry = get_dataset_registry()
prepared_datasets = dataset_registry.scan_datasets()

if prepared_datasets:
    with st.expander("📦 Load a Prepared Dataset (optional)", expanded=not st.session_state.get('training_data_configured', False)):
        st.caption("Load a previously saved dataset configuration to skip manual setup.")
        
        dataset_options = ["-- Select a dataset --"] + [
            f"{d.name} ({d.n_rows:,} rows, {len(d.stations)} stations)" for d in prepared_datasets
        ]
        
        selected_idx = st.selectbox(
            "Available prepared datasets",
            range(len(dataset_options)),
            format_func=lambda i: dataset_options[i],
            key="prepared_dataset_selector"
        )
        
        if selected_idx > 0:
            selected_dataset = prepared_datasets[selected_idx - 1]
            
            col_info, col_load = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"""
                **Source**: `{selected_dataset.source_file}`  
                **Target**: `{selected_dataset.target_column}`  
                **Covariates**: {', '.join(selected_dataset.covariate_columns) or 'None'}  
                **Date range**: {selected_dataset.date_range[0][:10]} → {selected_dataset.date_range[1][:10]}
                """)
            
            with col_load:
                if st.button(" Load Dataset", type="primary"):
                    try:
                        df_loaded, config = dataset_registry.load_dataset(selected_dataset)
                        
                        # Populate session state
                        if selected_dataset.station_column:
                            st.session_state['training_data_raw'] = df_loaded
                            st.session_state['training_stations'] = selected_dataset.stations
                            st.session_state['training_date_col'] = df_loaded.index.name or 'date'
                            st.session_state['training_station_col'] = selected_dataset.station_column
                            st.session_state['training_is_multistation'] = True
                        else:
                            st.session_state['training_data'] = df_loaded
                            st.session_state['training_is_multistation'] = False
                        
                        st.session_state['training_variables'] = [selected_dataset.target_column] + selected_dataset.covariate_columns
                        st.session_state['training_target_var'] = selected_dataset.target_column
                        st.session_state['training_covariate_vars'] = selected_dataset.covariate_columns
                        st.session_state['training_filename'] = selected_dataset.source_file
                        st.session_state['training_dataset_name'] = selected_dataset.name
                        st.session_state['training_preprocessing'] = selected_dataset.preprocessing
                        st.session_state['training_data_configured'] = True
                        
                        st.success(f" Dataset '{selected_dataset.name}' loaded!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f" Error loading dataset: {e}")

# Check if data is loaded
if 'training_data_configured' not in st.session_state or not st.session_state['training_data_configured']:
    st.warning(" No data loaded")
    st.info(" Go back to **Dataset Preparation** to load a CSV file, or load a prepared dataset above")
    st.stop()

# Get data
is_multistation = st.session_state.get('training_is_multistation', False)
target_var = st.session_state['training_target_var']
covariate_vars = st.session_state['training_covariate_vars']
preprocessing_config = st.session_state['training_preprocessing']

st.success(f" Data loaded: **{st.session_state['training_filename']}**")

# Ensure dataset identifiers are set (avoid "unknown" in MLflow/Forecasting)
if not st.session_state.get('training_filename'):
    st.session_state['training_filename'] = st.session_state.get('training_dataset_name') or "session_dataset"
if not st.session_state.get('training_dataset_name'):
    st.session_state['training_dataset_name'] = st.session_state.get('training_filename')

# Show preprocessing summary
with st.expander(" Configured Preprocessing Summary"):
    st.markdown(get_preprocessing_summary(preprocessing_config))

st.markdown("---")


# ============================================================================
# DATA SELECTION
# ============================================================================
st.subheader(" Data Selection")

if is_multistation:
    df_raw = st.session_state['training_data_raw']
    date_col = st.session_state['training_date_col']
    station_col = st.session_state['training_station_col']
    stations = st.session_state['training_stations']

    col1, col2 = st.columns([3, 1])

    with col1:
        select_all = st.checkbox("Select all stations", value=False, key="select_all_stations")
        
        if select_all:
            selected_stations = stations
            st.info(f" All **{len(stations)} stations** selected for training")
        else:
            selected_stations = st.multiselect(
                "Select stations to train",
                options=stations,
                default=[stations[0]] if stations else [],
                help="Select one or more stations to train models on"
            )
            
            if not selected_stations:
                st.warning(" Please select at least one station")
                st.stop()

    with col2:
        st.metric("Selected", f"{len(selected_stations)} / {len(stations)}")
        
        # Training Strategy logic
        training_strategy = "Independent"
        if len(selected_stations) > 1:
            training_strategy = st.radio(
                "Training Strategy",
                ["Independent (One Model Per Station)", "Global Model (One Model For All)"],
                help="Independent: Trains a separate model for each station.\nGlobal: Trains a single model using data from all selected stations (larger dataset, potentially better generalization)."
            )
            
            if "Global" in training_strategy:
                st.info(" Training one Global Model on all selected stations.")
            else:
                st.caption("Models will be trained successively")
        else:
            st.caption("Training single model")

else:
    df_raw = st.session_state['training_data']
    filename = st.session_state.get('training_filename', 'station_data')
    station_name = Path(filename).stem
    selected_stations = [station_name]
    training_strategy = "Independent"  # Single station is always independent
    st.info(f"📍 Single-station data detected: {station_name}")

st.markdown("---")


# ============================================================================
# MODEL SELECTION
# ============================================================================
st.subheader(" Model Selection")

col1, col2 = st.columns([1, 2])

with col1:
    selected_category = st.selectbox(
        "Category",
        options=list(MODEL_CATEGORIES.keys()),
        help="Model type"
    )

    models_in_category = MODEL_CATEGORIES[selected_category]
    
    # Format display names
    display_names = []
    for model_name in models_in_category:
        if model_name in RECOMMENDED_MODELS:
            display_names.append(f"⭐ {model_name}")
        else:
            display_names.append(model_name)

    selected_model_display = st.selectbox(
        "Model",
        options=display_names,
        help="⭐ = Recommended to start"
    )

    selected_model = selected_model_display.replace("⭐ ", "")

with col2:
    model_info = get_model_info(selected_model)

    if model_info:
        st.markdown(f"### {model_info['name']}")
        st.info(model_info['description'])

        capabilities = []
        if model_info['multivariate']:
            capabilities.append(" Multivariate")
        if model_info['supports_past_covariates']:
            capabilities.append(" Past covariates")

        if capabilities:
            st.markdown("**Capabilities**: " + " | ".join(capabilities))
        else:
            st.markdown("**Capabilities**: Univariate only")

st.markdown("---")


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
st.subheader(" Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Forecast Horizon")
    input_chunk = st.number_input(
        "Input chunk (days)", min_value=7, max_value=365, value=30,
        help=get_hyperparam_description('input_chunk_length')
    )
    output_chunk = st.number_input(
        "Output chunk (days)", min_value=1, max_value=365, value=7,
        help=get_hyperparam_description('output_chunk_length')
    )

with col2:
    st.markdown("##### Data Splits")
    train_ratio = st.slider("Train (%)", 50, 90, 70) / 100
    val_ratio = st.slider("Validation (%)", 5, 30, 15) / 100
    test_ratio = 1.0 - train_ratio - val_ratio
    st.metric("Test (%)", f"{test_ratio*100:.0f}")

with col3:
    st.markdown("##### Common Hyperparameters")
    batch_size = st.number_input(
        "Batch size", 8, 128, 32,
        help=get_hyperparam_description('batch_size')
    )
    n_epochs = st.number_input(
        "Epochs", 10, 200, 50,
        help=get_hyperparam_description('n_epochs')
    )
    learning_rate = st.number_input(
        "Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f",
        help=get_hyperparam_description('learning_rate')
    )

# Model-Specific Hyperparameters
st.markdown("##### Model-Specific Hyperparameters")

with st.expander(f" Configure {selected_model} hyperparameters"):
    hyperparams = {
        'input_chunk_length': input_chunk,
        'output_chunk_length': output_chunk,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate
    }
    
    model_hyperparams_space = model_info['hyperparams']

    for param_name, param_space in model_hyperparams_space.items():
        description = get_hyperparam_description(param_name)

        if isinstance(param_space, tuple) and len(param_space) == 3:
            min_val, max_val, default_val = param_space
            if isinstance(min_val, int):
                hyperparams[param_name] = st.slider(
                    param_name, min_value=min_val, max_value=max_val, value=default_val, help=description
                )
            else:
                hyperparams[param_name] = st.slider(
                    param_name, min_value=float(min_val), max_value=float(max_val), value=float(default_val),
                    step=(max_val - min_val) / 100, help=description
                )
        elif isinstance(param_space, list):
            hyperparams[param_name] = st.selectbox(
                param_name, options=param_space, help=description
            )

# Loss function (PyTorch models)
st.markdown("##### Loss Function")

if ModelFactory.is_torch_model(selected_model):
    loss_function = st.selectbox(
        "Loss function",
        options=['MAE', 'MSE', 'Huber', 'Quantile', 'RMSE'],
        index=0,
        help="Select the loss function for training optimization."
    )

    if loss_function == 'Quantile':
        quantile_value = st.slider(
            "Quantile", 0.01, 0.99, 0.5, 0.01,
            help="Quantile to predict. 0.5 = median."
        )
        hyperparams['loss_quantile'] = quantile_value

    hyperparams['loss_fn'] = loss_function
else:
    st.info("ℹ️ Loss function selection is only available for Deep Learning models")

# Covariates
use_covariates_flag = False
total_covars = len(covariate_vars)

if total_covars > 0 and model_info['supports_past_covariates']:
    covar_desc = [c for c in covariate_vars]
    use_covariates_flag = st.checkbox(
        f"Use covariates ({total_covars} features)",
        value=True,
        help=f"Features: {', '.join(covar_desc)}"
    )

st.markdown("---")


# ============================================================================
# TRAINING MODE
# ============================================================================
st.subheader(" Training Mode")

col_mode, col_options = st.columns([1, 2])

with col_mode:
    training_mode = st.radio(
        "Mode",
        options=[" Manual", " Optuna"],
        help="**Manual**: You define hyperparameters. **Optuna**: Automatic optimization."
    )

with col_options:
    if training_mode == " Optuna":
        st.markdown("##### Optuna Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_trials = st.number_input("Trials", 5, 100, 20, help="Number of optimization trials")
        with c2:
            optuna_metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"], help="Metric to minimize")
        with c3:
            optuna_timeout = st.number_input("Timeout (min)", 5, 120, 30, help="Max time in minutes")
        
        st.info(" Optuna will test different hyperparameter combinations to find the best one.")
    
    # Early Stopping
    if ModelFactory.is_torch_model(selected_model):
        st.markdown("##### Early Stopping")
        c1, c2 = st.columns(2)
        with c1:
            use_early_stopping = st.checkbox("Enable Early Stopping", value=True)
        with c2:
            if use_early_stopping:
                early_stopping_patience = st.number_input("Patience (epochs)", 3, 30, 10)
            else:
                early_stopping_patience = 10
    else:
        use_early_stopping = False
        early_stopping_patience = 10

st.markdown("---")


# ============================================================================
# START TRAINING
# ============================================================================
st.subheader(" Start Training")

# Validation checks
if ModelFactory.is_torch_model(selected_model):
    normalization = preprocessing_config.get('normalization', 'None')
    if normalization == 'None':
        st.warning("""
         **Warning**: No normalization selected for Deep Learning model.
        This strongly increases the risk of convergence failure (NaN loss).
        Recommend: StandardScaler or MinMax.
        """)
        force_normalization = st.checkbox("I understand the risks, force training without normalization", value=False)
        if not force_normalization:
            st.stop()

button_label = " Start Optuna Optimization" if training_mode == " Optuna" else " Start Training"

# Initialiser le flag d'entraînement dans session_state
if 'training_started' not in st.session_state:
    st.session_state['training_started'] = False

# Si l'entraînement est en cours, afficher un bouton pour arrêter/réinitialiser
if st.session_state.get('training_started', False):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🔄 Training in progress... The page will auto-refresh to show progress.")
    with col2:
        if st.button("❌ Stop & Reset", use_container_width=True):
            # Réinitialiser tous les états d'entraînement
            st.session_state['training_started'] = False
            # Nettoyer les états de threads
            for key in list(st.session_state.keys()):
                if key.startswith('training_'):
                    del st.session_state[key]
            st.rerun()

# Si on clique sur le bouton, activer le flag et relancer
if not st.session_state.get('training_started', False):
    if st.button(button_label, type="primary", use_container_width=True):
        st.session_state['training_started'] = True
        st.rerun()

# Si l'entraînement a été démarré, exécuter le code d'entraînement
# Ce bloc reste actif même après st.rerun() grâce au flag dans session_state
if st.session_state.get('training_started', False):
    # Initialize Live Log Manager
    log_manager = LiveLogManager(max_entries=200, session_key="training_log")
    
    # Ne pas clear les logs si on est en train de monitorer (pour éviter de perdre l'historique)
    if 'training_initialized' not in st.session_state:
        log_manager.clear()  # Clear previous logs seulement au premier démarrage
        st.session_state['training_initialized'] = True
    
    # Create progress section
    progress_elements = create_progress_section()
    
    # Create log placeholder
    st.markdown("---")
    log_placeholder = st.empty()
    
    # Legacy placeholders for backwards compatibility with callback
    progress_bar = progress_elements['station_progress']
    status_text = progress_elements['current_operation']
    epoch_progress = progress_elements['epoch_progress']
    epoch_status = progress_elements['epoch_counter']
    epoch_metrics = progress_elements['metrics_row']
    loss_chart = progress_elements['loss_chart']

    try:
        results_all_stations = {}
        
        # Global Model lists
        is_global_mode = "Global" in training_strategy
        global_data = {
            'train': [], 'val': [], 'test': [],
            'train_cov': [], 'val_cov': [], 'test_cov': [], 'full_cov': []
        }
        global_metadata = {
            'station_data_map': {},
            'station_data_raw_map': {},
            'target_scalers': {},
            'cov_scalers': {}
        }
        
        # Log start (avoid duplicate logs during reruns while training is in progress)
        has_active_training = any(
            st.session_state.get(f"training_{s}", {}).get("in_progress", False)
            for s in selected_stations
        )
        if not has_active_training:
            log_manager.info(f"Starting training with {len(selected_stations)} station(s)")
            log_manager.info(f"Model: {selected_model} | Mode: {'Global' if is_global_mode else 'Independent'}")
            log_manager.render_inline(log_placeholder)

        for idx, station_name in enumerate(selected_stations):
            training_key = f'training_{station_name}'
            if training_key not in st.session_state:
                st.session_state[training_key] = {
                    'in_progress': False,
                    'result': None,
                    'thread': None,
                    'metrics_file': None,
                    'completed': False
                }
            training_state = st.session_state[training_key]

            # If training is already running, skip data prep/log spam and show monitor only
            if training_state.get('in_progress'):
                status_text.info(f"🧠 Training in progress for **{station_name}**...")
                metrics_file_path = training_state.get('metrics_file')
                if metrics_file_path:
                    metrics_file = Path(metrics_file_path)
                    monitor_key = f'monitor_{station_name}'
                    if monitor_key not in st.session_state:
                        monitor_fragment = create_training_monitor_fragment(
                            metrics_file=metrics_file,
                            progress_bar=epoch_progress,
                            status_text=epoch_status,
                            metrics_placeholder=epoch_metrics,
                            chart_placeholder=loss_chart,
                            rerun_on_complete=True
                        )
                        st.session_state[monitor_key] = monitor_fragment
                    st.session_state[monitor_key]()
                log_manager.render_inline(log_placeholder)
                st.stop()

            station_start_time = time.time()
            
            # Update station progress
            station_pct = (idx) / len(selected_stations)
            progress_elements['station_progress'].progress(station_pct, text=f"Preparing station {idx+1}/{len(selected_stations)}")
            progress_elements['station_counter'].metric("Stations", f"{idx+1}/{len(selected_stations)}")
            status_text.info(f"📍 Processing: **{station_name}**")
            
            log_manager.station(f"[{idx+1}/{len(selected_stations)}] Preparing station: {station_name}")
            log_manager.render_inline(log_placeholder)
            
            station_covariate_vars = list(covariate_vars)

            # 1. Prepare data
            if is_multistation:
                df_station = df_raw[df_raw[station_col] == station_name].copy()
                # Date may be a column (fresh load) or already the index (e.g. loaded from prepared dataset)
                if date_col in df_station.columns:
                    df_station = df_station.set_index(date_col).sort_index()
                else:
                    df_station = df_station.sort_index()
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]
            else:
                df_station = df_raw.copy()
                # Filter variables for single station too (index already set from load/prepared dataset)
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]

            initial_rows = len(df_station)
            log_manager.info(f"  → Loaded {initial_rows:,} rows")

            # 2. Missing Values - Handle comprehensively
            fill_method = preprocessing_config.get('fill_method', 'Drop rows')
            missing_before = df_station.isna().sum().sum()
            
            if fill_method == 'Drop rows':
                df_station = df_station.dropna()
            elif fill_method == 'Linear Interpolation':
                # Interpolate middle values, then fill edges with ffill/bfill
                df_station = df_station.interpolate(method='linear', limit_direction='both')
                df_station = df_station.ffill().bfill()  # Handle leading/trailing NaNs
            elif fill_method == 'Forward fill':
                df_station = df_station.ffill().bfill()  # bfill for leading NaNs
            elif fill_method == 'Backward fill':
                df_station = df_station.bfill().ffill()  # ffill for trailing NaNs
            
            # Final check: drop any remaining NaNs (safety net)
            remaining_nan = df_station.isna().sum().sum()
            if remaining_nan > 0:
                df_station = df_station.dropna()
                log_manager.warning(f"  → Dropped {remaining_nan} remaining NaN rows after {fill_method}")
            
            if missing_before > 0:
                log_manager.info(f"  → Handled {missing_before} missing values ({fill_method})")

            # 3. Duplicates
            if df_station.index.duplicated().any():
                n_dupes = df_station.index.duplicated().sum()
                log_manager.warning(f"  → {n_dupes} duplicates detected, aggregating by mean")
                df_station = df_station.groupby(df_station.index).mean()
            
            # 4. Darts Conversion - Ensure index is DatetimeIndex
            from darts import TimeSeries
            if not isinstance(df_station.index, pd.DatetimeIndex):
                df_station.index = pd.to_datetime(df_station.index)
            freq = pd.infer_freq(df_station.index) or 'D'
            target_series = TimeSeries.from_dataframe(df_station, value_cols=target_var, freq=freq, fill_missing_dates=True)

            covariates_series = None
            if use_covariates_flag and station_covariate_vars:
                covariates_series = TimeSeries.from_dataframe(df_station, value_cols=station_covariate_vars, freq=freq, fill_missing_dates=True)

            # Progress: 10-90% for data prep, reserve last 10% for training
            progress_pct = (idx + 1) / len(selected_stations)
            progress_elements['station_progress'].progress(progress_pct, text=f"Preparing station {idx+1}/{len(selected_stations)}")

            # 7. Split Train/Val/Test (Raw)
            train_raw, val_raw, test_raw = split_train_val_test(target_series, train_ratio, val_ratio, test_ratio)
            log_manager.info(f"  → Split: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")
            
            if covariates_series:
                train_cov_raw, val_cov_raw, test_cov_raw = split_train_val_test(covariates_series, train_ratio, val_ratio, test_ratio)
            else:
                train_cov_raw = val_cov_raw = test_cov_raw = None

            # 8. Normalization (Fit on Train)
            target_preprocessor = TimeSeriesPreprocessor(preprocessing_config)
            train = target_preprocessor.fit_transform(train_raw)
            val = target_preprocessor.transform(val_raw)
            test = target_preprocessor.transform(test_raw)

            covariates_scaled = None
            train_cov = val_cov = test_cov = None
            cov_preprocessor = None

            if covariates_series:
                cov_preprocessor = TimeSeriesPreprocessor(preprocessing_config)
                train_cov = cov_preprocessor.fit_transform(train_cov_raw)
                val_cov = cov_preprocessor.transform(val_cov_raw)
                test_cov = cov_preprocessor.transform(test_cov_raw)
                covariates_scaled = cov_preprocessor.transform(covariates_series)
            
            log_manager.info(f"  → Normalized (fit on train, transform val/test)")

            # Check for NaNs
            if np.isnan(train.values()).any() or np.isnan(val.values()).any():
                log_manager.error(f"  → NaN values detected after preprocessing!")
                st.error(" NaN values detected after preprocessing! Use a different missing value strategy.")
                st.stop()

            # Station preparation complete
            station_duration = time.time() - station_start_time
            log_manager.success(f"  ✓ Station prepared in {station_duration:.1f}s")
            log_manager.render_inline(log_placeholder)

            # 9. Global Accumulation or training
            col_mapping = {'target_var': target_var, 'covariate_vars': station_covariate_vars}

            if is_global_mode:
                global_data['train'].append(train)
                global_data['val'].append(val)
                global_data['test'].append(test)
                if train_cov:
                    global_data['train_cov'].append(train_cov)
                    global_data['val_cov'].append(val_cov)
                    global_data['test_cov'].append(test_cov)
                    global_data['full_cov'].append(covariates_scaled)

                global_metadata['station_data_map'][station_name] = df_station
                # Don't store df_station_raw as it contains added features - let inverse transform handle it
                global_metadata['station_data_raw_map'][station_name] = None
                global_metadata['target_scalers'][station_name] = target_preprocessor
                if cov_preprocessor:
                    global_metadata['cov_scalers'][station_name] = cov_preprocessor

                continue

            # 10. Independent Training
            log_manager.training(f"Starting independent training for {station_name}...")
            log_manager.render_inline(log_placeholder)
            
            # OPTUNA
            if training_mode == " Optuna":
                optuna_key = f"optuna_{station_name}"
                if optuna_key not in st.session_state:
                    st.session_state[optuna_key] = {
                        "in_progress": False,
                        "completed": False,
                        "best_params": None,
                    }
                optuna_state = st.session_state[optuna_key]

                if optuna_state.get("in_progress"):
                    status_text.info(f"🔄 Optuna en cours pour **{station_name}**...")
                    log_manager.render_inline(log_placeholder)
                    st.stop()

                if optuna_state.get("completed") and optuna_state.get("best_params"):
                    log_manager.info(f"  → Optuna déjà terminé, params réutilisés: {optuna_state['best_params']}")
                    hyperparams.update(optuna_state["best_params"])
                    study = optuna_state.get("study")
                else:
                    log_manager.info(f"Running Optuna optimization ({n_trials} trials)...")
                    log_manager.render_inline(log_placeholder)
                    optuna_state["in_progress"] = True
                    st.session_state[optuna_key] = optuna_state
                    
                    # Optuna Logic Block
                    import optuna
                    from optuna.visualization import plot_optimization_history, plot_param_importances
                    from dashboard.utils.optuna_training import create_optuna_objective
                    
                    optuna_status = st.empty()
                    
                    def optuna_callback(study, trial):
                        optuna_status.text(f"Trial {len(study.trials)}/{n_trials} - Best: {study.best_value:.4f}")
                        log_manager.progress(f"  Optuna trial {len(study.trials)}/{n_trials} - Best: {study.best_value:.4f}")
                        log_manager.render_inline(log_placeholder)

                    objective = create_optuna_objective(
                        model_name=selected_model,
                        train=train, val=val,
                        train_cov=train_cov, val_cov=val_cov, full_cov=covariates_scaled,
                        use_covariates=use_covariates_flag and (train_cov is not None),
                        metric=optuna_metric,
                        n_epochs=15, early_stopping=True, early_stopping_patience=5,
                        input_chunk_length=int(input_chunk),
                        output_chunk_length=int(output_chunk),
                    )

                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                    study.optimize(objective, n_trials=n_trials, timeout=optuna_timeout*60, callbacks=[optuna_callback])
                    
                    log_manager.success(f"Best params found: {study.best_params}")
                    st.success(f"Best params found: {study.best_params}")
                    hyperparams.update(study.best_params)
                    optuna_state["best_params"] = study.best_params
                    optuna_state["completed"] = True
                    optuna_state["in_progress"] = False
                    optuna_state["study"] = study
                    st.session_state[optuna_key] = optuna_state
                
                # Graphiques Optuna (historique + importance des paramètres)
                if study is not None:
                    from optuna.visualization import plot_optimization_history, plot_param_importances
                    with st.expander("📊 Optuna – Historique et importance des paramètres", expanded=True):
                        col_hist, col_imp = st.columns(2)
                        with col_hist:
                            try:
                                fig_hist = plot_optimization_history(study)
                                fig_hist.update_layout(template='plotly_white', height=350, margin=dict(l=50, r=50, t=40, b=50))
                                st.plotly_chart(fig_hist, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Impossible de tracer l'historique : {e}")
                        with col_imp:
                            try:
                                fig_imp = plot_param_importances(study)
                                if fig_imp is not None:
                                    fig_imp.update_layout(template='plotly_white', height=350, margin=dict(l=50, r=50, t=40, b=50))
                                    st.plotly_chart(fig_imp, use_container_width=True)
                                else:
                                    st.info("Importance des paramètres non disponible (trop peu de trials).")
                            except Exception as e:
                                st.warning(f"Impossible de tracer l'importance : {e}")
                        with st.expander("Voir le détail des trials"):
                            st.dataframe(study.trials_dataframe(), use_container_width=True, hide_index=True)
                else:
                    st.info("Historique Optuna indisponible (session expirée).")
            
            # FINAL TRAINING
            log_manager.training(f"Training final model ({n_epochs} epochs)...")
            log_manager.render_inline(log_placeholder)
            status_text.success(f"🧠 **Training model for {station_name}...**")

            # MONITORING EN TEMPS RÉEL avec thread
            
            # Si déjà complété (succès ou erreur), skip (évite les boucles de rerun)
            if training_state.get('completed', False):
                if training_state.get('result'):
                    log_manager.info(f"  → Training already completed for {station_name}, skipping...")
                    results_all_stations[station_name] = training_state['result']
                continue
            
            # NOUVEAU SYSTÈME: Utiliser un fichier JSON persistant pour les métriques
            import tempfile
            import json
            metrics_file_path = training_state.get('metrics_file')
            if metrics_file_path:
                metrics_file = Path(metrics_file_path)
            else:
                safe_station_name = station_name.replace('/', '_').replace('\\', '_')
                metrics_file = Path(tempfile.gettempdir()) / f"training_metrics_{safe_station_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                # Nettoyer le fichier s'il existe déjà
                if metrics_file.exists():
                    metrics_file.unlink()
                # Créer le fichier JSON initial pour que le monitoring puisse le lire immédiatement
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                initial_metrics = {
                    'status': 'initializing',
                    'start_time': None,
                    'current_epoch': 0,
                    'total_epochs': n_epochs,
                    'train_losses': [],
                    'val_losses': [],
                    'epochs': [],
                    'last_update': None
                }
                with open(metrics_file, 'w') as f:
                    json.dump(initial_metrics, f, indent=2)
                training_state['metrics_file'] = str(metrics_file)
                log_manager.info(f"  → Metrics file: {metrics_file}")
            
            # Si l'entraînement n'a pas encore démarré, le lancer dans un thread
            # Use lock to prevent race conditions with st.rerun()
            dataset_name = (
                st.session_state.get('training_dataset_name')
                or st.session_state.get('training_filename')
                or station_name
            )
            original_filename = st.session_state.get('training_filename') or dataset_name
            should_start_thread = False
            with _training_lock:
                if not training_state['in_progress'] and training_state['thread'] is None and not training_state.get('completed', False):
                    training_state['in_progress'] = True
                    training_state['result'] = None
                    should_start_thread = True

            if should_start_thread:
                def train_in_thread():
                    """Lance l'entraînement dans un thread séparé."""
                    try:
                        # Mettre à jour le fichier de métriques pour signaler le démarrage
                        try:
                            metrics_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(metrics_file, 'r') as f:
                                metrics_payload = json.load(f)
                        except Exception:
                            metrics_payload = {}
                        metrics_payload.update({
                            'status': 'training',
                            'start_time': datetime.now().isoformat(),
                            'last_update': datetime.now().isoformat()
                        })
                        try:
                            with open(metrics_file, 'w') as f:
                                json.dump(metrics_payload, f, indent=2)
                        except Exception:
                            pass

                        result = run_training_pipeline(
                            model_name=selected_model,
                            hyperparams=hyperparams,
                            train=train, val=val, test=test,
                            train_cov=train_cov, val_cov=val_cov, test_cov=test_cov, full_cov=covariates_scaled,
                            use_covariates=use_covariates_flag and (train_cov is not None),
                            save_dir=CHECKPOINTS_DIR,
                            station_name=station_name,
                            verbose=False,
                            metrics_file=metrics_file,
                            n_epochs=n_epochs,
                            early_stopping_patience=early_stopping_patience if use_early_stopping else None,
                            station_data_df=df_station,
                            station_data_df_raw=None,
                            column_mapping=col_mapping,
                            target_preprocessor=target_preprocessor,
                            cov_preprocessor=cov_preprocessor,
                            original_filename=original_filename,
                            dataset_name=dataset_name,
                            preprocessing_config=preprocessing_config,
                            all_stations=[station_name]
                        )
                        training_state['result'] = result
                    except Exception as e:
                        training_state['result'] = {'status': 'error', 'error': str(e)}
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_payload = json.load(f)
                        except Exception:
                            metrics_payload = {}
                        metrics_payload.update({
                            'status': 'error',
                            'error': str(e),
                            'last_update': datetime.now().isoformat()
                        })
                        try:
                            with open(metrics_file, 'w') as f:
                                json.dump(metrics_payload, f, indent=2)
                        except Exception:
                            pass
                    finally:
                        # IMPORTANT: Use lock and set completed=True BEFORE in_progress=False
                        # This prevents race conditions with st.rerun()
                        with _training_lock:
                            training_state['completed'] = True
                            training_state['in_progress'] = False

                # Lancer le thread
                thread = threading.Thread(target=train_in_thread, daemon=True)
                thread.start()
                training_state['thread'] = thread
            
            # Fragment de monitoring (affiché uniquement pendant l'entraînement)
            monitor_key = f'monitor_{station_name}'
            
            if training_state['in_progress']:
                # En cours : créer le fragment si besoin, l'appeler, puis st.stop().
                # Le fragment fait st.rerun() à la fin → on reviendra pour traiter les résultats.
                if monitor_key not in st.session_state:
                    monitor_fragment = create_training_monitor_fragment(
                        metrics_file=metrics_file,
                        progress_bar=epoch_progress,
                        status_text=epoch_status,
                        metrics_placeholder=epoch_metrics,
                        chart_placeholder=loss_chart,
                        rerun_on_complete=True
                    )
                    st.session_state[monitor_key] = monitor_fragment
                st.session_state[monitor_key]()
                log_manager.render_inline(log_placeholder)
                st.stop()
            
            # Entraînement terminé : traiter les résultats (sans appeler le fragment)
            training_results = training_state['result']
            training_state['thread'] = None
            
            if metrics_file.exists():
                TrainingMonitor(metrics_file).display_progress(
                    progress_bar=epoch_progress,
                    status_text=epoch_status,
                    metrics_placeholder=epoch_metrics,
                    chart_placeholder=loss_chart,
                    rerun_on_complete=False
                )
                try:
                    metrics_file.unlink()
                except Exception:
                    pass
                training_state['metrics_file'] = None
                if monitor_key in st.session_state:
                    del st.session_state[monitor_key]
            
            if training_results.get('status') == 'success':
                metrics = training_results.get('metrics', {})
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None and not isinstance(v, list))
                log_manager.success(f"Training complete for {station_name}: {metrics_str}")
                if training_results.get('saved_path'):
                    log_manager.info(f"Model saved to: {training_results['saved_path']}")
                # Mark as completed to prevent re-training on rerun
                training_state['completed'] = True
            else:
                log_manager.error(f"Training failed for {station_name}: {training_results.get('error', 'Unknown error')}")
                # Also mark as completed (with error) to prevent infinite retry loops
                training_state['completed'] = True
            log_manager.render_inline(log_placeholder)
            
            results_all_stations[station_name] = training_results
        
        # End Loop

        # 11. Global Training Execution
        if is_global_mode and global_data['train']:
            # Mark station preparation complete
            progress_elements['station_progress'].progress(1.0, text="✅ All stations prepared")
            
            log_manager.info(f"")
            log_manager.training(f"═══ STARTING GLOBAL MODEL TRAINING ═══")
            log_manager.info(f"Training global model on {len(selected_stations)} stations")
            total_train_samples = sum(len(t) for t in global_data['train'])
            total_val_samples = sum(len(v) for v in global_data['val'])
            log_manager.info(f"Total samples: train={total_train_samples:,}, val={total_val_samples:,}")
            log_manager.render_inline(log_placeholder)
            
            status_text.success(f"🚀 **Training Global Model on {len(selected_stations)} stations...**")
            
            # NOUVEAU SYSTÈME: Utiliser un fichier JSON pour les métriques
            import tempfile
            import json
            metrics_file = Path(tempfile.gettempdir()) / f"training_metrics_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Nettoyer le fichier s'il existe déjà
            if metrics_file.exists():
                metrics_file.unlink()
            
            # Créer le fichier JSON initial pour que le monitoring puisse le lire immédiatement
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            initial_metrics = {
                'status': 'initializing',
                'start_time': None,
                'current_epoch': 0,
                'total_epochs': n_epochs,
                'train_losses': [],
                'val_losses': [],
                'epochs': [],
                'last_update': None
            }
            with open(metrics_file, 'w') as f:
                json.dump(initial_metrics, f, indent=2)
            
            # MONITORING EN TEMPS RÉEL avec thread pour modèle global
            training_key_global = 'training_global'
            if training_key_global not in st.session_state:
                st.session_state[training_key_global] = {
                    'in_progress': False,
                    'result': None,
                    'thread': None,
                    'completed': False  # Flag pour éviter de relancer après complétion
                }
            
            training_state_global = st.session_state[training_key_global]
            
            # Si déjà complété, skip (évite les boucles de rerun)
            if training_state_global.get('completed', False) and training_state_global.get('result'):
                log_manager.info(f"  → Global training already completed, skipping...")
                results_all_stations["Global_Model"] = training_state_global['result']
            else:
                # Si l'entraînement n'a pas encore démarré, le lancer dans un thread
                # Use lock to prevent race conditions with st.rerun()
                dataset_name = (
                    st.session_state.get('training_dataset_name')
                    or st.session_state.get('training_filename')
                    or "global_dataset"
                )
                should_start_global_thread = False
                with _training_lock:
                    if not training_state_global['in_progress'] and training_state_global['thread'] is None and not training_state_global.get('completed', False):
                        training_state_global['in_progress'] = True
                        training_state_global['result'] = None
                        should_start_global_thread = True

                if should_start_global_thread:
                    def train_global_in_thread():
                        """Lance l'entraînement global dans un thread séparé."""
                        try:
                            result = run_training_pipeline(
                                model_name=selected_model,
                                hyperparams=hyperparams,
                                train=global_data['train'],
                                val=global_data['val'],
                                test=global_data['test'],
                                train_cov=global_data['train_cov'] if global_data['train_cov'] else None,
                                val_cov=global_data['val_cov'] if global_data['val_cov'] else None,
                                test_cov=global_data['test_cov'] if global_data['test_cov'] else None,
                                full_cov=global_data['full_cov'] if global_data['full_cov'] else None,
                                use_covariates=use_covariates_flag and (len(global_data['train_cov']) > 0),
                                save_dir=CHECKPOINTS_DIR,
                                station_name=f"Global_All_{len(selected_stations)}st",
                                verbose=False,
                                metrics_file=metrics_file,
                                n_epochs=n_epochs,
                                early_stopping_patience=early_stopping_patience if use_early_stopping else None,
                                station_data_df=global_metadata['station_data_map'],
                                station_data_df_raw=None,
                                column_mapping=col_mapping,
                                target_preprocessor=global_metadata['target_scalers'],
                                cov_preprocessor=global_metadata['cov_scalers'] if global_metadata['cov_scalers'] else None,
                                original_filename="Multi-Station Global",
                                dataset_name=dataset_name,
                                preprocessing_config=preprocessing_config,
                                all_stations=selected_stations
                            )
                            training_state_global['result'] = result
                        except Exception as e:
                            training_state_global['result'] = {'status': 'error', 'error': str(e)}
                        finally:
                            # IMPORTANT: Use lock and set completed=True BEFORE in_progress=False
                            with _training_lock:
                                training_state_global['completed'] = True
                                training_state_global['in_progress'] = False

                    # Lancer le thread
                    thread = threading.Thread(target=train_global_in_thread, daemon=True)
                    thread.start()
                    training_state_global['thread'] = thread
                
                # Fragment de monitoring global (affiché uniquement pendant l'entraînement)
                monitor_key_global = 'monitor_global'
                
                if training_state_global['in_progress']:
                    if monitor_key_global not in st.session_state:
                        monitor_fragment_global = create_training_monitor_fragment(
                            metrics_file=metrics_file,
                            progress_bar=epoch_progress,
                            status_text=epoch_status,
                            metrics_placeholder=epoch_metrics,
                            chart_placeholder=loss_chart,
                            rerun_on_complete=True
                        )
                        st.session_state[monitor_key_global] = monitor_fragment_global
                    st.session_state[monitor_key_global]()
                    log_manager.render_inline(log_placeholder)
                    st.stop()
                
                # Entraînement global terminé : traiter les résultats
                training_results = training_state_global['result']
                training_state_global['thread'] = None
                
                if metrics_file.exists():
                    TrainingMonitor(metrics_file).display_progress(
                        progress_bar=epoch_progress,
                        status_text=epoch_status,
                        metrics_placeholder=epoch_metrics,
                        chart_placeholder=loss_chart,
                        rerun_on_complete=False
                    )
                    try:
                        metrics_file.unlink()
                    except Exception:
                        pass
                
                if training_results and training_results.get('status') == 'success':
                    log_manager.success(f"Global model training complete!")
                    log_manager.info(f"Saved to: {training_results.get('saved_path', 'N/A')}")
                    training_state_global['completed'] = True
                elif training_results:
                    log_manager.error(f"Global training failed: {training_results.get('error', 'Unknown error')}")
                    training_state_global['completed'] = True
                log_manager.render_inline(log_placeholder)
                
                if training_results:
                    results_all_stations["Global_Model"] = training_results

        # Vérifier si tous les entraînements sont terminés avant d'afficher les résultats
        all_training_complete = True
        for station_name in selected_stations:
            training_key = f'training_{station_name}'
            if training_key in st.session_state:
                training_state = st.session_state[training_key]
                if training_state.get('in_progress', False):
                    all_training_complete = False
                    break
        
        if is_global_mode:
            training_key_global = 'training_global'
            if training_key_global in st.session_state:
                training_state_global = st.session_state[training_key_global]
                if training_state_global.get('in_progress', False):
                    all_training_complete = False
        
        # Si l'entraînement est en cours, ne pas afficher les résultats finaux
        if not all_training_complete:
            # Mettre à jour le log et attendre
            log_manager.render_inline(log_placeholder)
            st.stop()

        # Final status - seulement quand tout est terminé
        progress_elements['epoch_progress'].progress(1.0, text="✅ Training complete!")
        status_text.success(" **Training sequence completed!**")
        st.success("✅ All training tasks finished.")

        # Display Results
        st.markdown("### 📊 Results Summary")
        
        def format_metric(value, fmt=".4f", suffix=""):
            """Safely format a metric that might be a list or scalar."""
            import numpy as np
            if value is None:
                return '-'
            if isinstance(value, (list, tuple)):
                # For global models, take mean of per-station metrics
                value = np.mean([v for v in value if v is not None and not np.isnan(v)])
            if isinstance(value, float) and np.isnan(value):
                return '-'
            return f"{value:{fmt}}{suffix}"
        
        summary_data = []
        for station, res in results_all_stations.items():
            metrics = res.get('metrics', {})
            status = ' Success' if res['status'] == 'success' else ' Error'
            summary_data.append({
                'Station': station,
                'Status': status,
                'MAE': format_metric(metrics.get('MAE')),
                'RMSE': format_metric(metrics.get('RMSE')),
                'MAPE': format_metric(metrics.get('MAPE'), ".2f", "%"),
                'sMAPE': format_metric(metrics.get('sMAPE'), ".2f", "%"),
                'Saved': '✓' if res.get('saved_path') else '-'
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        # Show errors if any
        for station, res in results_all_stations.items():
            if res['status'] == 'error' and 'error' in res:
                with st.expander(f" Error details for {station}", expanded=True):
                    st.error(res.get('error', 'Unknown error'))
                    if 'traceback' in res:
                        st.code(res['traceback'])
        
        # Download buttons
        for station, res in results_all_stations.items():
            if res['status'] == 'success' and 'saved_path' in res:
                path = Path(res['saved_path'])
                if path.exists():
                    st.markdown(f"**📦 {station}**")
                    add_download_button(path.parent, key_suffix=f"dl_{station}")

        # Final log render with expander (une seule fois, pas en double)
        # Le log inline est déjà affiché dans log_placeholder, on ajoute juste l'expander en bas
        with st.expander("📋 Complete Training Log", expanded=False):
            log_text = log_manager.get_formatted_logs()
            if log_text:
                st.code(log_text, language=None)
            else:
                st.info("No logs available")
        
        # Réinitialiser le flag quand tout est terminé
        if all_training_complete:
            st.session_state['training_started'] = False
            if 'training_initialized' in st.session_state:
                del st.session_state['training_initialized']
            # Nettoyer les fragments de monitoring
            for key in list(st.session_state.keys()):
                if key.startswith('monitor_'):
                    del st.session_state[key]
            
            # Rafraîchir le registry pour que les modèles apparaissent dans Forecasting
            try:
                from dashboard.utils.model_registry import get_registry
                registry = get_registry(CHECKPOINTS_DIR.parent)
                if hasattr(registry, "scan_existing_checkpoints"):
                    registry.scan_existing_checkpoints()  # Re-scan pour détecter les nouveaux modèles
                log_manager.info(f"✅ Registry refreshed. {len(registry.list_all_models())} model(s) available.")
                log_manager.render_inline(log_placeholder)
            except Exception as e:
                log_manager.warning(f"Could not refresh registry: {e}")
                log_manager.render_inline(log_placeholder)

    except Exception as e:
        # Log the error
        log_manager.error(f"CRITICAL ERROR: {e}")
        log_manager.render_inline(log_placeholder)
        
        st.error(f" Critical Error: {e}")
        st.session_state['training_started'] = False
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")

