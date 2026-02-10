"""Model training page with extended Darts model support."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.models_config import MODEL_CATEGORIES, RECOMMENDED_MODELS, get_model_info, get_hyperparam_description
from dashboard.utils.preprocessing import (
    TimeSeriesPreprocessor,
    split_train_val_test,
    get_preprocessing_summary
)
from dashboard.utils.model_factory import ModelFactory
from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.dataset_registry import get_dataset_registry
from dashboard.utils.training import run_training_pipeline
from dashboard.utils.export import add_download_button


# =============================================================================
# TRAINING STATE MANAGEMENT
# =============================================================================

class TrainingPhase(Enum):
    """Clear training phases to avoid ambiguous states."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"


def get_training_state() -> Dict[str, Any]:
    """Get or initialize the global training state."""
    if 'training_state' not in st.session_state:
        st.session_state['training_state'] = {
            'phase': TrainingPhase.IDLE.value,
            'current_station_idx': 0,
            'total_stations': 0,
            'metrics_file': None,
            'thread': None,
            'results': {},
            'error': None,
            'logs': [],
            'start_time': None,
        }
    return st.session_state['training_state']


def reset_training_state():
    """Completely reset training state for a new training session."""
    # Clean up any existing metrics file
    state = get_training_state()
    if state.get('metrics_file'):
        try:
            metrics_path = Path(state['metrics_file'])
            if metrics_path.exists():
                metrics_path.unlink()
        except Exception:
            pass

    # Reset the state
    st.session_state['training_state'] = {
        'phase': TrainingPhase.IDLE.value,
        'current_station_idx': 0,
        'total_stations': 0,
        'metrics_file': None,
        'thread': None,
        'results': {},
        'error': None,
        'logs': [],
        'start_time': None,
    }


def add_log(message: str, level: str = "info"):
    """Add a log message to the training state."""
    state = get_training_state()
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji_map = {
        "info": "i",
        "success": "v",
        "warning": "!",
        "error": "x",
        "training": "*",
        "station": ">",
    }
    emoji = emoji_map.get(level, "i")
    state['logs'].append(f"[{timestamp}] [{emoji}] {message}")
    # Keep only last 200 logs
    if len(state['logs']) > 200:
        state['logs'] = state['logs'][-200:]


def get_logs_text() -> str:
    """Get formatted logs as text."""
    state = get_training_state()
    return "\n".join(state.get('logs', []))


def _write_log_to_state(state_dict: Dict[str, Any], message: str, level: str = "info"):
    """Thread-safe log writing directly to state dictionary."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji_map = {
        "info": "i",
        "success": "v",
        "warning": "!",
        "error": "x",
        "training": "*",
        "station": ">",
    }
    emoji = emoji_map.get(level, "i")
    log_entry = f"[{timestamp}] [{emoji}] {message}"

    if 'logs' not in state_dict:
        state_dict['logs'] = []
    state_dict['logs'].append(log_entry)
    # Keep only last 200 logs
    if len(state_dict['logs']) > 200:
        state_dict['logs'] = state_dict['logs'][-200:]


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(page_title="Train Models", page_icon="", layout="wide")

st.title(" Train Models")
st.markdown("Forecasting model training with configurable preprocessing.")
st.markdown("---")


# =============================================================================
# LOAD PREPARED DATASET
# =============================================================================

dataset_registry = get_dataset_registry()
prepared_datasets = dataset_registry.scan_datasets()

if prepared_datasets:
    with st.expander("Load a Prepared Dataset (optional)", expanded=not st.session_state.get('training_data_configured', False)):
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
                **Date range**: {selected_dataset.date_range[0][:10]} -> {selected_dataset.date_range[1][:10]}
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

_display_ds_name = st.session_state.get('training_dataset_name') or st.session_state['training_filename']
st.success(f" Data loaded: **{_display_ds_name}**")

# Ensure dataset identifiers are set
if not st.session_state.get('training_filename'):
    st.session_state['training_filename'] = st.session_state.get('training_dataset_name') or "session_dataset"
if not st.session_state.get('training_dataset_name'):
    st.session_state['training_dataset_name'] = st.session_state.get('training_filename')

# Show preprocessing summary
with st.expander(" Configured Preprocessing Summary"):
    st.markdown(get_preprocessing_summary(preprocessing_config))

st.markdown("---")


# =============================================================================
# DATA SELECTION
# =============================================================================

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
                help="Independent: Trains a separate model for each station.\nGlobal: Trains a single model using data from all selected stations."
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
    # Utiliser le nom du dataset donné par l'utilisateur en priorité (ex: "09994X0521/P4B_prepared")
    # plutôt que le nom de la source brute (ex: "db_hubeau_daily_chroniques")
    dataset_display_name = st.session_state.get('training_dataset_name') or filename
    station_name = Path(dataset_display_name).stem if '/' not in dataset_display_name else dataset_display_name
    selected_stations = [station_name]
    training_strategy = "Independent"
    st.info(f" Single-station data detected: {station_name}")

st.markdown("---")


# =============================================================================
# MODEL SELECTION
# =============================================================================

st.subheader(" Model Selection")

col1, col2 = st.columns([1, 2])

with col1:
    selected_category = st.selectbox(
        "Category",
        options=list(MODEL_CATEGORIES.keys()),
        help="Model type"
    )

    models_in_category = MODEL_CATEGORIES[selected_category]

    display_names = []
    for model_name in models_in_category:
        if model_name in RECOMMENDED_MODELS:
            display_names.append(f"* {model_name}")
        else:
            display_names.append(model_name)

    selected_model_display = st.selectbox(
        "Model",
        options=display_names,
        help="* = Recommended to start"
    )

    selected_model = selected_model_display.replace("* ", "")

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


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

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
    st.info("Loss function selection is only available for Deep Learning models")

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


# =============================================================================
# TRAINING MODE
# =============================================================================

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
            optuna_metric = st.selectbox("Metric", ["MAE", "RMSE"], help="Metric to minimize")
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
                early_stopping_patience = None
    else:
        use_early_stopping = False
        early_stopping_patience = None

st.markdown("---")


# =============================================================================
# START TRAINING SECTION
# =============================================================================

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

# Get current training state
training_state = get_training_state()
current_phase = training_state['phase']


# =============================================================================
# TRAINING CONTROL BUTTONS
# =============================================================================

if current_phase == TrainingPhase.IDLE.value:
    # Show start button
    if st.button(button_label, type="primary", use_container_width=True):
        reset_training_state()
        state = get_training_state()
        state['phase'] = TrainingPhase.PREPARING.value
        state['total_stations'] = len(selected_stations)
        state['start_time'] = time.time()
        add_log(f"Starting training with {len(selected_stations)} station(s)")
        add_log(f"Model: {selected_model} | Mode: {'Global' if 'Global' in training_strategy else 'Independent'}")
        st.rerun()

elif current_phase in [TrainingPhase.PREPARING.value, TrainingPhase.TRAINING.value]:
    # Show stop button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Training in progress... Phase: {current_phase}")
    with col2:
        if st.button(" Stop & Reset", type="secondary", use_container_width=True):
            reset_training_state()
            st.rerun()

elif current_phase in [TrainingPhase.COMPLETED.value, TrainingPhase.ERROR.value]:
    # Show new training button
    if st.button(" Start New Training", type="primary", use_container_width=True):
        reset_training_state()
        st.rerun()


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

if current_phase == TrainingPhase.PREPARING.value:
    # Create progress UI
    st.markdown("### Training Progress")

    progress_bar = st.progress(0, text="Preparing data...")
    status_text = st.empty()

    st.markdown("---")

    epoch_progress = st.progress(0, text="Waiting for training to start...")
    metrics_row = st.empty()
    loss_chart = st.empty()

    st.markdown("---")
    log_placeholder = st.empty()

    try:
        is_global_mode = "Global" in training_strategy

        # Data structures for global model
        global_data = {
            'train': [], 'val': [], 'test': [],
            'train_cov': [], 'val_cov': [], 'test_cov': [], 'full_cov': []
        }
        global_metadata = {
            'station_data_map': {},
            'target_scalers': {},
            'cov_scalers': {}
        }

        # Prepare data for all stations
        prepared_stations = []

        for idx, station_name in enumerate(selected_stations):
            progress_pct = (idx + 1) / len(selected_stations)
            progress_bar.progress(progress_pct, text=f"Preparing station {idx+1}/{len(selected_stations)}: {station_name}")
            status_text.info(f" Processing: **{station_name}**")
            add_log(f"[{idx+1}/{len(selected_stations)}] Preparing station: {station_name}", "station")

            station_covariate_vars = list(covariate_vars)

            # 1. Prepare data
            if is_multistation:
                df_station = df_raw[df_raw[station_col] == station_name].copy()
                if date_col in df_station.columns:
                    df_station = df_station.set_index(date_col).sort_index()
                else:
                    df_station = df_station.sort_index()
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]
            else:
                df_station = df_raw.copy()
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]

            initial_rows = len(df_station)
            add_log(f"  -> Loaded {initial_rows:,} rows")

            # 2. Missing Values
            fill_method = preprocessing_config.get('fill_method', 'Drop rows')
            missing_before = df_station.isna().sum().sum()

            if fill_method == 'Drop rows':
                df_station = df_station.dropna()
            elif fill_method == 'Linear Interpolation':
                df_station = df_station.interpolate(method='linear', limit_direction='both')
                df_station = df_station.ffill().bfill()
            elif fill_method == 'Forward fill':
                df_station = df_station.ffill().bfill()
            elif fill_method == 'Backward fill':
                df_station = df_station.bfill().ffill()

            # Final check: drop any remaining NaNs
            remaining_nan = df_station.isna().sum().sum()
            if remaining_nan > 0:
                df_station = df_station.dropna()
                add_log(f"  -> Dropped {remaining_nan} remaining NaN rows after {fill_method}", "warning")

            if missing_before > 0:
                add_log(f"  -> Handled {missing_before} missing values ({fill_method})")

            # 3. Duplicates
            if df_station.index.duplicated().any():
                n_dupes = df_station.index.duplicated().sum()
                add_log(f"  -> {n_dupes} duplicates detected, aggregating by mean", "warning")
                df_station = df_station.groupby(df_station.index).mean()

            # 4. Darts Conversion
            from darts import TimeSeries
            if not isinstance(df_station.index, pd.DatetimeIndex):
                df_station.index = pd.to_datetime(df_station.index)
            freq = pd.infer_freq(df_station.index) or 'D'
            target_series = TimeSeries.from_dataframe(df_station, value_cols=target_var, freq=freq, fill_missing_dates=True)

            covariates_series = None
            if use_covariates_flag and station_covariate_vars:
                covariates_series = TimeSeries.from_dataframe(df_station, value_cols=station_covariate_vars, freq=freq, fill_missing_dates=True)

            # 5. Split Train/Val/Test
            train_raw, val_raw, test_raw = split_train_val_test(target_series, train_ratio, val_ratio, test_ratio)
            add_log(f"  -> Split: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

            if covariates_series:
                train_cov_raw, val_cov_raw, test_cov_raw = split_train_val_test(covariates_series, train_ratio, val_ratio, test_ratio)
            else:
                train_cov_raw = val_cov_raw = test_cov_raw = None

            # 6. Normalization (Fit on Train)
            target_preprocessor = TimeSeriesPreprocessor(preprocessing_config)
            train = target_preprocessor.fit_transform(train_raw)
            val = target_preprocessor.transform(val_raw)
            test = target_preprocessor.transform(test_raw)

            train_cov = val_cov = test_cov = None
            covariates_scaled = None
            cov_preprocessor = None

            if covariates_series:
                cov_preprocessor = TimeSeriesPreprocessor(preprocessing_config)
                train_cov = cov_preprocessor.fit_transform(train_cov_raw)
                val_cov = cov_preprocessor.transform(val_cov_raw)
                test_cov = cov_preprocessor.transform(test_cov_raw)
                covariates_scaled = cov_preprocessor.transform(covariates_series)

            add_log(f"  -> Normalized (fit on train, transform val/test)")

            # Check for NaNs
            if np.isnan(train.values()).any() or np.isnan(val.values()).any():
                add_log(f"  -> NaN values detected after preprocessing!", "error")
                raise ValueError("NaN values detected after preprocessing!")

            add_log(f"  [OK] Station prepared", "success")

            # Store prepared data
            col_mapping = {'target_var': target_var, 'covariate_vars': station_covariate_vars}

            prepared_data = {
                'station_name': station_name,
                'train': train,
                'val': val,
                'test': test,
                'train_cov': train_cov,
                'val_cov': val_cov,
                'test_cov': test_cov,
                'full_cov': covariates_scaled,
                'df_station': df_station,
                'target_preprocessor': target_preprocessor,
                'cov_preprocessor': cov_preprocessor,
                'col_mapping': col_mapping,
            }
            prepared_stations.append(prepared_data)

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
                global_metadata['target_scalers'][station_name] = target_preprocessor
                if cov_preprocessor:
                    global_metadata['cov_scalers'][station_name] = cov_preprocessor

            # Update log display
            log_placeholder.code(get_logs_text(), language=None)

        # Store prepared data in state
        training_state['prepared_stations'] = prepared_stations
        training_state['global_data'] = global_data
        training_state['global_metadata'] = global_metadata
        training_state['is_global_mode'] = is_global_mode

        # Store training config
        training_state['config'] = {
            'model_name': selected_model,
            'hyperparams': hyperparams,
            'use_covariates': use_covariates_flag,
            'n_epochs': n_epochs,
            'early_stopping_patience': early_stopping_patience,
            'training_mode': training_mode,
            'selected_stations': selected_stations,
        }

        if training_mode == " Optuna":
            training_state['optuna_config'] = {
                'n_trials': n_trials,
                'metric': optuna_metric,
                'timeout': optuna_timeout,
            }

        # Move to training phase
        training_state['phase'] = TrainingPhase.TRAINING.value
        add_log("Data preparation complete. Starting training...", "success")
        st.rerun()

    except Exception as e:
        add_log(f"CRITICAL ERROR: {e}", "error")
        training_state['phase'] = TrainingPhase.ERROR.value
        training_state['error'] = str(e)
        st.error(f" Critical Error: {e}")
        import traceback
        st.code(traceback.format_exc())


elif current_phase == TrainingPhase.TRAINING.value:
    # Training execution
    st.markdown("### Training Progress")

    progress_bar = st.progress(1.0, text="Data preparation complete")
    status_text = st.empty()

    st.markdown("---")

    epoch_progress = st.progress(0, text="Training in progress...")
    epoch_status = st.empty()
    metrics_row = st.empty()
    loss_chart = st.empty()

    st.markdown("---")
    log_placeholder = st.empty()
    log_placeholder.code(get_logs_text(), language=None)

    # Get stored data
    prepared_stations = training_state.get('prepared_stations', [])
    global_data = training_state.get('global_data', {})
    global_metadata = training_state.get('global_metadata', {})
    is_global_mode = training_state.get('is_global_mode', False)
    config = training_state.get('config', {})

    model_name = config.get('model_name', selected_model)
    hyperparams_training = config.get('hyperparams', hyperparams)
    use_cov = config.get('use_covariates', use_covariates_flag)
    epochs = config.get('n_epochs', n_epochs)
    es_patience = config.get('early_stopping_patience', early_stopping_patience)
    t_mode = config.get('training_mode', training_mode)
    sel_stations = config.get('selected_stations', selected_stations)

    # Check if training thread exists and is still running
    thread = training_state.get('thread')
    metrics_file_path = training_state.get('metrics_file')

    if thread is None:
        # Start training in a new thread
        import tempfile
        safe_name = "training_" + datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = Path(tempfile.gettempdir()) / f"{safe_name}_metrics.json"

        # Initialize metrics file
        initial_metrics = {
            'status': 'initializing',
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': epochs,
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'last_update': None
        }
        with open(metrics_file, 'w') as f:
            json.dump(initial_metrics, f, indent=2)

        training_state['metrics_file'] = str(metrics_file)

        # Capture values needed in the thread (avoid accessing session_state from thread)
        _dataset_name = st.session_state.get('training_dataset_name') or st.session_state.get('training_filename') or "dataset"
        _original_filename = st.session_state.get('training_filename') or _dataset_name
        _preproc_config = preprocessing_config.copy() if preprocessing_config else {}
        _start_time = training_state.get('start_time', time.time())

        # Reference to state dict (safe since dict is mutable and shared)
        _state_ref = training_state

        # Define training function
        def run_training():
            results = {}

            try:
                if is_global_mode and global_data.get('train'):
                    # Global model training
                    _write_log_to_state(_state_ref, "Starting GLOBAL MODEL training", "training")

                    col_mapping = prepared_stations[0]['col_mapping'] if prepared_stations else {'target_var': target_var, 'covariate_vars': list(covariate_vars)}

                    result = run_training_pipeline(
                        model_name=model_name,
                        hyperparams=hyperparams_training,
                        train=global_data['train'],
                        val=global_data['val'],
                        test=global_data['test'],
                        train_cov=global_data['train_cov'] if global_data['train_cov'] else None,
                        val_cov=global_data['val_cov'] if global_data['val_cov'] else None,
                        test_cov=global_data['test_cov'] if global_data['test_cov'] else None,
                        full_cov=global_data['full_cov'] if global_data['full_cov'] else None,
                        use_covariates=use_cov and bool(global_data.get('train_cov')),
                        save_dir=CHECKPOINTS_DIR,
                        station_name=f"Global_All_{len(sel_stations)}st",
                        verbose=False,
                        metrics_file=metrics_file,
                        n_epochs=epochs,
                        early_stopping_patience=es_patience,
                        station_data_df=global_metadata['station_data_map'],
                        station_data_df_raw=None,
                        column_mapping=col_mapping,
                        target_preprocessor=global_metadata['target_scalers'],
                        cov_preprocessor=global_metadata['cov_scalers'] if global_metadata['cov_scalers'] else None,
                        original_filename="Multi-Station Global",
                        dataset_name=_dataset_name,
                        preprocessing_config=_preproc_config,
                        all_stations=sel_stations
                    )
                    results["Global_Model"] = result

                else:
                    # Independent model training
                    for idx, data in enumerate(prepared_stations):
                        station = data['station_name']
                        _write_log_to_state(_state_ref, f"Training model for {station} ({idx+1}/{len(prepared_stations)})", "training")

                        # Update metrics file for this station
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                            metrics_data['current_station'] = station
                            metrics_data['station_idx'] = idx + 1
                            metrics_data['total_stations'] = len(prepared_stations)
                            # Reset epoch tracking for new station
                            metrics_data['current_epoch'] = 0
                            metrics_data['train_losses'] = []
                            metrics_data['val_losses'] = []
                            metrics_data['epochs'] = []
                            metrics_data['status'] = 'training'
                            with open(metrics_file, 'w') as f:
                                json.dump(metrics_data, f, indent=2)
                        except Exception:
                            pass

                        result = run_training_pipeline(
                            model_name=model_name,
                            hyperparams=hyperparams_training,
                            train=data['train'],
                            val=data['val'],
                            test=data['test'],
                            train_cov=data['train_cov'],
                            val_cov=data['val_cov'],
                            test_cov=data['test_cov'],
                            full_cov=data['full_cov'],
                            use_covariates=use_cov and data['train_cov'] is not None,
                            save_dir=CHECKPOINTS_DIR,
                            station_name=station,
                            verbose=False,
                            metrics_file=metrics_file,
                            n_epochs=epochs,
                            early_stopping_patience=es_patience,
                            station_data_df=data['df_station'],
                            station_data_df_raw=None,
                            column_mapping=data['col_mapping'],
                            target_preprocessor=data['target_preprocessor'],
                            cov_preprocessor=data['cov_preprocessor'],
                            original_filename=_original_filename,
                            dataset_name=_dataset_name,
                            preprocessing_config=_preproc_config,
                            all_stations=[station]
                        )
                        results[station] = result

                        if result.get('status') == 'success':
                            _write_log_to_state(_state_ref, f"Training complete for {station}", "success")
                        else:
                            _write_log_to_state(_state_ref, f"Training failed for {station}: {result.get('error', 'Unknown')}", "error")

                        # Clean up GPU memory between stations to avoid accumulation
                        if len(prepared_stations) > 1:
                            try:
                                from dashboard.utils.xpu_support import cleanup_gpu_memory
                                cleanup_gpu_memory(model=result.get('model'))
                            except Exception:
                                pass

                # Mark as completed
                _state_ref['results'] = results
                _state_ref['phase'] = TrainingPhase.COMPLETED.value

                # Update metrics file to completed
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    metrics_data['status'] = 'completed'
                    metrics_data['total_time_seconds'] = time.time() - _start_time
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics_data, f, indent=2)
                except Exception:
                    pass

            except Exception as e:
                _state_ref['error'] = str(e)
                _state_ref['phase'] = TrainingPhase.ERROR.value
                _write_log_to_state(_state_ref, f"Training error: {e}", "error")

                # Update metrics file to error
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    metrics_data['status'] = 'error'
                    metrics_data['error'] = str(e)
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics_data, f, indent=2)
                except Exception:
                    pass

            finally:
                # Always clean up GPU memory after training
                try:
                    from dashboard.utils.xpu_support import cleanup_gpu_memory
                    cleanup_gpu_memory()
                    _write_log_to_state(_state_ref, "GPU memory released", "info")
                except Exception as cleanup_err:
                    _write_log_to_state(_state_ref, f"GPU cleanup warning: {cleanup_err}", "warning")

        # Start training thread
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        training_state['thread'] = thread
        add_log("Training thread started", "info")

    # Monitor training progress
    if metrics_file_path:
        metrics_file = Path(metrics_file_path)

        # Read current metrics
        metrics_data = None
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        if metrics_data:
            status = metrics_data.get('status', 'unknown')
            current_epoch = metrics_data.get('current_epoch', 0)
            total_epochs_display = metrics_data.get('total_epochs', epochs)
            train_losses = metrics_data.get('train_losses', [])
            val_losses = metrics_data.get('val_losses', [])
            epochs_list = metrics_data.get('epochs', [])
            eta = metrics_data.get('eta_seconds')

            # Update progress and status based on training phase
            if status == 'training':
                # Active training
                if total_epochs_display and total_epochs_display > 0:
                    progress = current_epoch / total_epochs_display
                    epoch_progress.progress(min(progress, 1.0), text=f"Epoch {current_epoch}/{total_epochs_display}")

                status_msg = f"Training - Epoch {current_epoch}"
                if total_epochs_display:
                    status_msg += f"/{total_epochs_display}"
                if train_losses and train_losses[-1] is not None:
                    status_msg += f" | Train: {train_losses[-1]:.4f}"
                if val_losses and val_losses[-1] is not None:
                    status_msg += f" | Val: {val_losses[-1]:.4f}"
                if eta:
                    status_msg += f" | ETA: {int(eta)}s"
                epoch_status.text(status_msg)

            elif status == 'finalizing':
                # Model training finished, now evaluating and saving
                epoch_progress.progress(1.0, text="Training finished, evaluating...")
                epoch_status.text("Evaluating model and saving artifacts...")

            # Update metrics display (for both training and finalizing)
            if status in ['training', 'finalizing']:
                with metrics_row.container():
                    cols = st.columns(4)
                    cols[0].metric("Epoch", f"{current_epoch}/{total_epochs_display or '?'}")
                    if train_losses and train_losses[-1] is not None:
                        cols[1].metric("Train Loss", f"{train_losses[-1]:.4f}")
                    else:
                        cols[1].metric("Train Loss", "N/A")
                    if val_losses and val_losses[-1] is not None:
                        cols[2].metric("Val Loss", f"{val_losses[-1]:.4f}")
                    else:
                        cols[2].metric("Val Loss", "N/A")
                    if eta and status == 'training':
                        mins, secs = divmod(int(eta), 60)
                        cols[3].metric("ETA", f"{mins}m {secs}s")
                    elif status == 'finalizing':
                        cols[3].metric("Status", "Saving...")

                # Update loss chart
                if epochs_list and (train_losses or val_losses):
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    if train_losses:
                        valid_train = [(e, l) for e, l in zip(epochs_list, train_losses) if l is not None]
                        if valid_train:
                            epochs_train, losses_train = zip(*valid_train)
                            fig.add_trace(go.Scatter(
                                x=list(epochs_train), y=list(losses_train),
                                mode='lines+markers', name='Train Loss',
                                line=dict(color='#FF4B4B', width=2)
                            ))

                    if val_losses:
                        valid_val = [(e, l) for e, l in zip(epochs_list, val_losses) if l is not None]
                        if valid_val:
                            epochs_val, losses_val = zip(*valid_val)
                            fig.add_trace(go.Scatter(
                                x=list(epochs_val), y=list(losses_val),
                                mode='lines+markers', name='Val Loss',
                                line=dict(color='#0068C9', width=2)
                            ))

                    fig.update_layout(
                        title="Loss Evolution",
                        xaxis_title="Epoch", yaxis_title="Loss",
                        template="plotly_white", height=300,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    loss_chart.plotly_chart(fig, use_container_width=True)

            elif status in ['completed', 'error']:
                # Check if the training thread has finished updating the state
                # The thread should have already set phase and results
                thread_phase = training_state.get('phase', '')
                has_results = bool(training_state.get('results'))

                if thread_phase in [TrainingPhase.COMPLETED.value, TrainingPhase.ERROR.value] or has_results:
                    # Thread has finished, safe to transition
                    st.rerun()
                else:
                    # Thread hasn't finished yet, wait a bit more
                    # This handles the case where the callback wrote 'completed' but
                    # the thread hasn't finished updating the state dict
                    epoch_status.text("Finalizing results...")
                    epoch_progress.progress(1.0, text="Training finished, collecting results...")

    # Update logs
    log_placeholder.code(get_logs_text(), language=None)

    # Auto-refresh while training
    if current_phase == TrainingPhase.TRAINING.value:
        time.sleep(1)
        st.rerun()


elif current_phase == TrainingPhase.COMPLETED.value:
    # Show results
    st.markdown("### Training Complete!")

    results = training_state.get('results', {})

    if not results:
        st.warning("Training completed but no results were recorded.")
        st.info("This may happen if the training was interrupted or if there was an issue during result collection.")

        # Show logs anyway
        with st.expander("Training Log", expanded=True):
            log_text = get_logs_text()
            if log_text:
                st.code(log_text, language=None)
            else:
                st.info("No logs available")
    else:
        st.success(f"All training tasks finished. {len(results)} model(s) trained.")

        # Summary table
        st.markdown("### Results Summary")

        def format_metric(value, fmt=".4f", suffix=""):
            if value is None:
                return '-'
            if isinstance(value, (list, tuple)):
                value = np.mean([v for v in value if v is not None and not np.isnan(v)])
            if isinstance(value, float) and np.isnan(value):
                return '-'
            return f"{value:{fmt}}{suffix}"

        summary_data = []
        for station, res in results.items():
            metrics = res.get('metrics', {})
            status = ' Success' if res.get('status') == 'success' else ' Error'
            summary_data.append({
                'Station': station,
                'Status': status,
                'MAE': format_metric(metrics.get('MAE')),
                'RMSE': format_metric(metrics.get('RMSE')),
                'sMAPE': format_metric(metrics.get('sMAPE'), ".2f", "%"),
                'Saved': '[OK]' if res.get('saved_path') else '-'
            })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Show errors if any
        for station, res in results.items():
            if res.get('status') == 'error' and 'error' in res:
                with st.expander(f" Error details for {station}", expanded=True):
                    st.error(res.get('error', 'Unknown error'))
                    if 'traceback' in res:
                        st.code(res['traceback'])

        # Download buttons
        st.markdown("### Download Models")
        for station, res in results.items():
            if res.get('status') == 'success' and 'saved_path' in res:
                path = Path(res['saved_path'])
                if path.exists():
                    st.markdown(f"**{station}**")
                    add_download_button(path.parent, key_suffix=f"dl_{station}")

        # Refresh registry
        try:
            from dashboard.utils.model_registry import get_registry
            registry = get_registry(CHECKPOINTS_DIR.parent)
            if hasattr(registry, "scan_existing_checkpoints"):
                registry.scan_existing_checkpoints()
            st.info(f" Registry refreshed. {len(registry.list_all_models())} model(s) available.")
        except Exception as e:
            st.warning(f"Could not refresh registry: {e}")

    # Show logs
    with st.expander(" Complete Training Log", expanded=False):
        log_text = get_logs_text()
        if log_text:
            st.code(log_text, language=None)
        else:
            st.info("No logs available")


elif current_phase == TrainingPhase.ERROR.value:
    # Show error
    st.error(f" Training Error: {training_state.get('error', 'Unknown error')}")

    # Show logs
    with st.expander(" Training Log", expanded=True):
        log_text = get_logs_text()
        if log_text:
            st.code(log_text, language=None)


# Footer
st.markdown("---")
