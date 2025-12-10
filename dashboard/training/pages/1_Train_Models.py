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
    add_datetime_features,
    add_lag_features,
    split_train_val_test,
    compute_data_statistics,
    get_preprocessing_summary
)
from dashboard.utils.model_factory import ModelFactory
from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.dataset_registry import get_dataset_registry
from dashboard.utils.training import run_training_pipeline
from dashboard.utils.callbacks import StreamlitProgressCallback
from dashboard.utils.export import add_download_button


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
    st.info(f" Single-station data detected: {station_name}")

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
        if model_info['supports_future_covariates']:
            capabilities.append(" Future covariates")

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
        "Output chunk (days)", min_value=1, max_value=90, value=7,
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
# Covariates
use_covariates_flag = False
has_datetime = preprocessing_config.get('datetime_features', False)
has_lags = len(preprocessing_config.get('lags', [])) > 0
total_covars = len(covariate_vars) + (6 if has_datetime else 0) + len(preprocessing_config.get('lags', []))

if total_covars > 0 and (model_info['supports_past_covariates'] or model_info['supports_future_covariates']):
    covar_desc = [c for c in covariate_vars]
    if has_datetime: covar_desc.append("Time Features")
    if has_lags: covar_desc.append("Lags")
    
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

if st.button(button_label, type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.markdown("###  Training Progress")
    epoch_progress = st.progress(0)
    epoch_status = st.empty()
    epoch_metrics = st.empty()
    loss_chart = st.empty()

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

        for idx, station_name in enumerate(selected_stations):
            status_text.text(f" Station {idx+1}/{len(selected_stations)}: {station_name}")
            
            station_covariate_vars = list(covariate_vars)

            # 1. Prepare data
            if is_multistation:
                df_station = df_raw[df_raw[station_col] == station_name].copy()
                df_station = df_station.set_index(date_col).sort_index()
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]
            else:
                df_station = df_raw.copy()
                # Filter variables for single station too
                available_cols = [target_var] + [c for c in covariate_vars if c in df_station.columns]
                df_station = df_station[available_cols]

            # 2. Missing Values
            fill_method = preprocessing_config.get('fill_method', 'Drop rows')
            if fill_method == 'Drop rows':
                df_station = df_station.dropna()
            elif fill_method == 'Linear Interpolation':
                df_station = df_station.interpolate(method='linear')
            elif fill_method == 'Forward fill':
                df_station = df_station.ffill()
            elif fill_method == 'Backward fill':
                df_station = df_station.bfill()

            # 3. Duplicates
            if df_station.index.duplicated().any():
                n_dupes = df_station.index.duplicated().sum()
                status_text.text(f" {n_dupes} duplicates detected: aggregating by mean...")
                df_station = df_station.groupby(df_station.index).mean()
            
            # 4. Time Features
            if preprocessing_config.get('datetime_features', False):
                if not isinstance(df_station.index, pd.DatetimeIndex):
                    df_station.index = pd.to_datetime(df_station.index)
                
                df_station['day_of_week'] = df_station.index.dayofweek
                df_station['month'] = df_station.index.month
                df_station['day_sin'] = np.sin(2 * np.pi * df_station.index.dayofyear / 365.25)
                df_station['day_cos'] = np.cos(2 * np.pi * df_station.index.dayofyear / 365.25)
                df_station['month_sin'] = np.sin(2 * np.pi * df_station.index.month / 12)
                df_station['month_cos'] = np.cos(2 * np.pi * df_station.index.month / 12)
                
                datetime_cols = ['day_of_week', 'month', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
                station_covariate_vars += datetime_cols
                status_text.text(f" Time features added ({len(datetime_cols)} cols)")
            
            # 5. Lags
            lags = preprocessing_config.get('lags', [])
            if lags:
                for lag in lags:
                    df_station[f'{target_var}_lag_{lag}'] = df_station[target_var].shift(lag)
                df_station = df_station.dropna()
                station_covariate_vars += [f'{target_var}_lag_{lag}' for lag in lags]
                status_text.text(f" Lags added: {lags}")
            
            # 6. Darts Conversion - Ensure index is DatetimeIndex
            from darts import TimeSeries
            if not isinstance(df_station.index, pd.DatetimeIndex):
                df_station.index = pd.to_datetime(df_station.index)
            freq = pd.infer_freq(df_station.index) or 'D'
            target_series = TimeSeries.from_dataframe(df_station, value_cols=target_var, freq=freq, fill_missing_dates=True)

            covariates_series = None
            if use_covariates_flag and station_covariate_vars:
                covariates_series = TimeSeries.from_dataframe(df_station, value_cols=station_covariate_vars, freq=freq, fill_missing_dates=True)

            progress_bar.progress(10 + idx * 10)

            # 7. Split Train/Val/Test (Raw)
            status_text.text(f"✂️ Splitting data...")
            train_raw, val_raw, test_raw = split_train_val_test(target_series, train_ratio, val_ratio, test_ratio)
            
            if covariates_series:
                train_cov_raw, val_cov_raw, test_cov_raw = split_train_val_test(covariates_series, train_ratio, val_ratio, test_ratio)
            else:
                train_cov_raw = val_cov_raw = test_cov_raw = None

            # 8. Normalization (Fit on Train)
            status_text.text(f" Normalization...")
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

            # Check for NaNs
            if np.isnan(train.values()).any() or np.isnan(val.values()).any():
                st.error(" NaN values detected after preprocessing! Use a different missing value strategy.")
                st.stop()

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

                status_text.text(f"📦 Prepared {station_name} for Global Model")
                continue

            # 10. Independent Training
            callbacks_list = []
            pl_trainer_kwargs = None
            if ModelFactory.is_torch_model(selected_model):
                cb = StreamlitProgressCallback(n_epochs, epoch_progress, epoch_status, epoch_metrics, loss_chart)
                callbacks_list.append(cb)
                if use_early_stopping:
                    from pytorch_lightning.callbacks import EarlyStopping
                    callbacks_list.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min'))
                pl_trainer_kwargs = {'callbacks': callbacks_list}

            # OPTUNA
            if training_mode == " Optuna":
                status_text.text(f" Optuna optimization for {station_name}...")
                
                # Optuna Logic Block
                import optuna
                from dashboard.utils.optuna_training import create_optuna_objective
                
                optuna_status = st.empty()
                
                def optuna_callback(study, trial):
                    optuna_status.text(f"Trial {len(study.trials)}/{n_trials} - Best: {study.best_value:.4f}")

                objective = create_optuna_objective(
                    model_name=selected_model,
                    train=train, val=val,
                    train_cov=train_cov, val_cov=val_cov, full_cov=covariates_scaled,
                    use_covariates=use_covariates_flag and (train_cov is not None),
                    metric=optuna_metric,
                    n_epochs=15, early_stopping=True, early_stopping_patience=5
                )

                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_trials, timeout=optuna_timeout*60, callbacks=[optuna_callback])
                
                st.success(f"Best params found: {study.best_params}")
                hyperparams.update(study.best_params)
            
            # FINAL TRAINING
            status_text.text(f"🧠 Training final model for {station_name}...")

            training_results = run_training_pipeline(
                model_name=selected_model,
                hyperparams=hyperparams,
                train=train, val=val, test=test,
                train_cov=train_cov, val_cov=val_cov, test_cov=test_cov, full_cov=covariates_scaled,
                use_covariates=use_covariates_flag and (train_cov is not None),
                save_dir=CHECKPOINTS_DIR,
                station_name=station_name,
                verbose=False,
                pl_trainer_kwargs=pl_trainer_kwargs,
                station_data_df=df_station,
                station_data_df_raw=None,  # Let run_training_pipeline generate raw data via inverse transform
                column_mapping=col_mapping,
                target_preprocessor=target_preprocessor,
                cov_preprocessor=cov_preprocessor,
                original_filename=st.session_state.get('training_filename', 'unknown'),
                preprocessing_config=preprocessing_config,
                all_stations=[station_name]  # Single station model
            )
            
            results_all_stations[station_name] = training_results
        
        # End Loop

        # 11. Global Training Execution
        if is_global_mode and global_data['train']:
            status_text.text(f" Training Global Model on {len(selected_stations)} stations...")
            
            callbacks_list = []
            pl_trainer_kwargs = None
            if ModelFactory.is_torch_model(selected_model):
                cb = StreamlitProgressCallback(n_epochs, epoch_progress, epoch_status, epoch_metrics, loss_chart)
                callbacks_list.append(cb)
                if use_early_stopping:
                    from pytorch_lightning.callbacks import EarlyStopping
                    callbacks_list.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min'))
                pl_trainer_kwargs = {'callbacks': callbacks_list}
            
            training_results = run_training_pipeline(
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
                pl_trainer_kwargs=pl_trainer_kwargs,
                station_data_df=global_metadata['station_data_map'],
                station_data_df_raw=None,  # Let run_training_pipeline generate raw data via inverse transform
                column_mapping=col_mapping, # Uses last col_mapping (assumes homogeneous)
                target_preprocessor=global_metadata['target_scalers'],
                cov_preprocessor=global_metadata['cov_scalers'] if global_metadata['cov_scalers'] else None,
                original_filename="Multi-Station Global",
                preprocessing_config=preprocessing_config,
                all_stations=selected_stations  # All stations for global model
            )
            results_all_stations["Global_Model"] = training_results

        progress_bar.progress(100)
        status_text.text(" Training sequence completed!")
        st.success(" All training tasks finished.")

        # Display Results
        st.markdown("###  Results Summary")
        
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
                'R²': format_metric(metrics.get('R2')),
                'MAPE': format_metric(metrics.get('MAPE'), ".2f", "%"),
                'sMAPE': format_metric(metrics.get('sMAPE'), ".2f", "%"),
                'Saved': '' if 'saved_path' in res else ''
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

    except Exception as e:
        st.error(f" Critical Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")

