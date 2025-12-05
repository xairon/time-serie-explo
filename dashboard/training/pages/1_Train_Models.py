"""Model training page with extended Darts model support."""

import streamlit as st
import pandas as pd
import torch
import sys
from pathlib import Path
from datetime import datetime

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

st.set_page_config(page_title="Train Models", page_icon="🎯", layout="wide")

st.title("🎯 Train Models")
st.markdown("Forecasting model training with configurable preprocessing.")
st.markdown("---")

# Check if data is loaded
if 'training_data_configured' not in st.session_state or not st.session_state['training_data_configured']:
    st.warning("⚠️ No data loaded")
    st.info("👉 Go back to the home page to load a CSV file")
    st.stop()

# Get data
is_multistation = st.session_state.get('training_is_multistation', False)
target_var = st.session_state['training_target_var']
covariate_vars = st.session_state['training_covariate_vars']
preprocessing_config = st.session_state['training_preprocessing']

st.success(f"✅ Data loaded: **{st.session_state['training_filename']}**")

# Show preprocessing summary
with st.expander("📋 Configured Preprocessing Summary"):
    st.markdown(get_preprocessing_summary(preprocessing_config))

st.markdown("---")

# ============================================================================
# DATA SELECTION
# ============================================================================
st.subheader("1️⃣ Data Selection")

if is_multistation:
    df_raw = st.session_state['training_data_raw']
    date_col = st.session_state['training_date_col']
    station_col = st.session_state['training_station_col']
    stations = st.session_state['training_stations']

    col1, col2 = st.columns(2)

    with col1:
        train_mode = st.radio(
            "Training Mode",
            options=["Specific Station", "All Stations (Loop)"],
            help="Train on a single station or all successively"
        )

    with col2:
        if train_mode == "Specific Station":
            selected_station = st.selectbox(
                "Station",
                options=stations,
                help="Station to use for training"
            )
            selected_stations = [selected_station]
        else:
            selected_stations = stations
            st.info(f"📍 **{len(selected_stations)} stations** will be trained successively")

else:
    df_raw = st.session_state['training_data']
    # Use filename without extension as station name
    filename = st.session_state.get('training_filename', 'station_data')
    station_name = Path(filename).stem
    selected_stations = [station_name]
    st.info(f"📊 Single-station data detected: {station_name}")

st.markdown("---")

# ============================================================================
# MODEL SELECTION
# ============================================================================
st.subheader("2️⃣ Model Selection")

col1, col2 = st.columns([1, 2])

with col1:
    # Category
    selected_category = st.selectbox(
        "Category",
        options=list(MODEL_CATEGORIES.keys()),
        help="Model type"
    )

    # Models in category
    models_in_category = MODEL_CATEGORIES[selected_category]

    # Highlight recommended ones
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

    # Extraire le nom réel (enlever ⭐)
    selected_model = selected_model_display.replace("⭐ ", "")

with col2:
    # Model Info
    model_info = get_model_info(selected_model)

    if model_info:
        st.markdown(f"### {model_info['name']}")
        st.info(model_info['description'])

        # Capabilities
        capabilities = []
        if model_info['multivariate']:
            capabilities.append("✅ Multivariate")
        if model_info['supports_past_covariates']:
            capabilities.append("✅ Past covariates")
        if model_info['supports_future_covariates']:
            capabilities.append("✅ Future covariates")

        if capabilities:
            st.markdown("**Capabilities**: " + " | ".join(capabilities))
        else:
            st.markdown("**Capabilities**: Univariate only")

st.markdown("---")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
st.subheader("3️⃣ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Forecast Horizon")
    input_chunk = st.number_input(
        "Input chunk (days)",
        min_value=7,
        max_value=365,
        value=30,
        help=get_hyperparam_description('input_chunk_length')
    )
    output_chunk = st.number_input(
        "Output chunk (days)",
        min_value=1,
        max_value=90,
        value=7,
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
        "Batch size",
        8, 128, 32,
        help=get_hyperparam_description('batch_size')
    )
    n_epochs = st.number_input(
        "Epochs",
        10, 200, 50,
        help=get_hyperparam_description('n_epochs')
    )
    learning_rate = st.number_input(
        "Learning rate",
        min_value=1e-5,
        max_value=1e-2,
        value=1e-3,
        format="%.5f",
        help=get_hyperparam_description('learning_rate')
    )

# Model-Specific Hyperparameters
st.markdown("##### Model-Specific Hyperparameters")

with st.expander(f"⚙️ Configure {selected_model} hyperparameters"):
    hyperparams = {
        'input_chunk_length': input_chunk,
        'output_chunk_length': output_chunk,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate
    }

    # Get model search space
    model_hyperparams_space = model_info['hyperparams']

    # Create inputs based on type
    for param_name, param_space in model_hyperparams_space.items():
        # Récupérer la description
        description = get_hyperparam_description(param_name)

        if isinstance(param_space, tuple) and len(param_space) == 3:
            # (min, max, default)
            min_val, max_val, default_val = param_space

            if isinstance(min_val, int):
                hyperparams[param_name] = st.slider(
                    param_name,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    help=description
                )
            else:
                hyperparams[param_name] = st.slider(
                    param_name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=(max_val - min_val) / 100,
                    help=description
                )

        elif isinstance(param_space, list):
            # Choice from a list
            hyperparams[param_name] = st.selectbox(
                param_name,
                options=param_space,
                help=description
            )

# Loss function selection (for PyTorch models)
st.markdown("##### Loss Function")
from dashboard.utils.model_factory import ModelFactory

if ModelFactory.is_torch_model(selected_model):
    loss_function = st.selectbox(
        "Loss function",
        options=['MAE', 'MSE', 'Huber', 'Quantile', 'RMSE'],
        index=0,  # MAE default
        help="""
        **MAE (Mean Absolute Error)**: Robust to outliers, penalizes errors uniformly.

        **MSE (Mean Squared Error)**: Strongly penalizes large errors.

        **Huber Loss**: Compromise between MAE and MSE. Quadratic for small errors, linear for large ones.

        **Quantile Loss**: For probabilistic predictions with confidence intervals.

        **RMSE (Root Mean Squared Error)**: Root of MSE. Same scale as original data.
        """
    )

    # Special config for Quantile Loss
    if loss_function == 'Quantile':
        quantile_value = st.slider(
            "Quantile",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help="Quantile to predict. 0.5 = median, 0.1 = 10th percentile, 0.9 = 90th percentile"
        )
        hyperparams['loss_quantile'] = quantile_value

    # Add loss to hyperparams
    hyperparams['loss_fn'] = loss_function
else:
    st.info("ℹ️ Loss function selection is only available for Deep Learning models")

# Covariates option
use_covariates = False
if covariate_vars and (model_info['supports_past_covariates'] or model_info['supports_future_covariates']):
    use_covariates = st.checkbox(
        f"Use covariates ({len(covariate_vars)} available)",
        value=True,
        help=f"Covariates: {', '.join(covariate_vars)}"
    )

st.markdown("---")

# ============================================================================
# TRAINING MODE
# ============================================================================
st.subheader("4️⃣ Training Mode")

col_mode, col_options = st.columns([1, 2])

with col_mode:
    training_mode = st.radio(
        "Mode",
        options=["🎯 Manual", "🧪 Optuna"],
        help="**Manual**: You define hyperparameters. **Optuna**: Automatic optimization."
    )

with col_options:
    if training_mode == "🧪 Optuna":
        st.markdown("##### Optuna Configuration")
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            n_trials = st.number_input("Trials", 5, 100, 20, help="Number of optimization trials")
        with col_o2:
            optuna_metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"], help="Metric to minimize")
        with col_o3:
            optuna_timeout = st.number_input("Timeout (min)", 5, 120, 30, help="Max time in minutes")
        
        st.info("💡 Optuna will test different hyperparameter combinations to find the best one.")
    
    # Early Stopping (for PyTorch models)
    if ModelFactory.is_torch_model(selected_model):
        st.markdown("##### Early Stopping")
        col_es1, col_es2 = st.columns(2)
        with col_es1:
            use_early_stopping = st.checkbox(
                "Enable Early Stopping", 
                value=True,
                help="Stops training if model stops improving"
            )
        with col_es2:
            if use_early_stopping:
                early_stopping_patience = st.number_input(
                    "Patience (epochs)", 
                    3, 30, 10,
                    help="Number of epochs without improvement before stopping"
                )
            else:
                early_stopping_patience = 10
    else:
        use_early_stopping = False
        early_stopping_patience = 10

st.markdown("---")

# ============================================================================
# START TRAINING
# ============================================================================
st.subheader("5️⃣ Start Training")

## ⚠️ Checks before training
# Check if normalization is enabled for deep learning models
if ModelFactory.is_torch_model(selected_model):
    normalization = preprocessing_config.get('normalization', 'None')
    if normalization == 'None':
        st.warning("""
        ⚠️ **Warning**: You haven't selected normalization, but you are using a deep learning model.
        
        Deep learning models work **much better** with normalized data.
        
        **Risks without normalization**:
        - `nan` losses (gradient explosion)
        - Non-convergence
        - Poor performance
        
        👉 **Recommendation**: Go back to home page and select **StandardScaler (z-score)** or **MinMax (0-1)**.
        """)
        
        force_normalization = st.checkbox(
            "I understand the risks, force training without normalization",
            value=False
        )
        
        if not force_normalization:
            st.stop()

# Training Button
button_label = "🧪 Start Optuna Optimization" if training_mode == "🧪 Optuna" else "🚀 Start Training"

if st.button(button_label, type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Loop over stations
        results_all_stations = {}

        for idx, station_name in enumerate(selected_stations):
            status_text.text(f"📊 Station {idx+1}/{len(selected_stations)}: {station_name}")

            # 1. Prepare data for this station
            if is_multistation:
                df_station = df_raw[df_raw[station_col] == station_name].copy()
                df_station = df_station.set_index(date_col).sort_index()
                df_station = df_station[[target_var] + covariate_vars]
            else:
                df_station = df_raw.copy()
            
            # 🆕 Save RAW data (BEFORE preprocessing) for display
            df_station_raw = df_station.copy()

            # 1.5 Handle missing values according to config BEFORE Darts conversion
            fill_method = preprocessing_config.get('fill_method', 'Drop rows')
            if fill_method == 'Drop rows':
                df_station = df_station.dropna()
            elif fill_method == 'Linear Interpolation':
                df_station = df_station.interpolate(method='linear')
            elif fill_method == 'Forward fill':
                df_station = df_station.ffill()
            elif fill_method == 'Backward fill':
                df_station = df_station.bfill()

            # 1.8 Handle index duplicates (CRITICAL for Darts)
            if df_station.index.duplicated().any():
                n_dupes = df_station.index.duplicated().sum()
                status_text.text(f"⚠️ {n_dupes} temporal duplicates detected: aggregating by mean...")
                # Faire la moyenne des valeurs pour les dates en double
                df_station = df_station.groupby(df_station.index).mean()
            
            # 1.9 🆕 Add time features if enabled
            if preprocessing_config.get('datetime_features', False):
                df_station['day_of_week'] = df_station.index.dayofweek
                df_station['month'] = df_station.index.month
                df_station['day_sin'] = np.sin(2 * np.pi * df_station.index.dayofyear / 365.25)
                df_station['day_cos'] = np.cos(2 * np.pi * df_station.index.dayofyear / 365.25)
                df_station['month_sin'] = np.sin(2 * np.pi * df_station.index.month / 12)
                df_station['month_cos'] = np.cos(2 * np.pi * df_station.index.month / 12)
                
                # Add to covariates
                datetime_cols = ['day_of_week', 'month', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
                covariate_vars = covariate_vars + datetime_cols
                status_text.text(f"📅 Time features added ({len(datetime_cols)} columns)")
            
            # 1.10 🆕 Add lags if enabled
            lags = preprocessing_config.get('lags', [])
            if lags:
                for lag in lags:
                    df_station[f'{target_var}_lag_{lag}'] = df_station[target_var].shift(lag)
                df_station = df_station.dropna()  # Drop NaNs created by shifts
                
                lag_cols = [f'{target_var}_lag_{lag}' for lag in lags]
                covariate_vars = covariate_vars + lag_cols
                status_text.text(f"📊 Lags added: {lags}")
            
            # 2. Convertir en TimeSeries Darts
            from darts import TimeSeries
            
            # Détecter la fréquence automatiquement si possible, sinon 'D'
            inferred_freq = pd.infer_freq(df_station.index)
            freq = inferred_freq if inferred_freq else 'D'

            target_series = TimeSeries.from_dataframe(
                df_station,
                value_cols=target_var,
                freq=freq
            )

            covariates_series = None
            if use_covariates and covariate_vars:
                covariates_series = TimeSeries.from_dataframe(
                    df_station,
                    value_cols=covariate_vars,
                    freq='D'
                )

            progress_bar.progress(10 + idx * 10)

            # 3. Split train/val/test on RAW DATA (before normalization!)
            # ⚠️ IMPORTANT: Split BEFORE normalization to avoid data leakage
            status_text.text(f"✂️ Splitting data (before normalization)...")

            train_raw, val_raw, test_raw = split_train_val_test(
                target_series,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )

            if covariates_series is not None:
                train_cov_raw, val_cov_raw, test_cov_raw = split_train_val_test(
                    covariates_series,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
            else:
                train_cov_raw = val_cov_raw = test_cov_raw = None

            progress_bar.progress(20 + idx * 10)

            # 4. Apply preprocessing (FIT on TRAIN only!)
            # ⚠️ CRITICAL: Fit scaler ONLY on train to avoid data leakage
            status_text.text(f"⚙️ Normalization (fit on train only)...")

            # Target preprocessing
            target_preprocessor = TimeSeriesPreprocessor(preprocessing_config)

            # FIT + TRANSFORM sur train
            train = target_preprocessor.fit_transform(train_raw)

            # TRANSFORM SEULEMENT sur val et test (sans refitter!)
            val = target_preprocessor.transform(val_raw)
            test = target_preprocessor.transform(test_raw)

            # Covariates preprocessing (scaler séparé)
            if covariates_series is not None:
                cov_preprocessor = TimeSeriesPreprocessor(preprocessing_config)

                # FIT + TRANSFORM sur train
                train_cov = cov_preprocessor.fit_transform(train_cov_raw)

                # TRANSFORM SEULEMENT sur val et test (sans refitter!)
                val_cov = cov_preprocessor.transform(val_cov_raw)
                test_cov = cov_preprocessor.transform(test_cov_raw)

                # Créer les covariates complètes (pour prédiction) en transformant toutes les données
                # avec le scaler fitté sur train
                covariates_scaled = cov_preprocessor.transform(covariates_series)
            else:
                train_cov = val_cov = test_cov = None
                covariates_scaled = None

            st.success(f"✅ Data prepared: Train={len(train)}, Val={len(val)}, Test={len(test)}")
            st.info("ℹ️ Scaler fitted on train only (no data leakage)")
            
            # ⚠️ Vérification critique : NaN dans les données après preprocessing
            # Utiliser .values() pour accéder au tableau numpy sous-jacent (plus robuste)
            import numpy as np
            train_has_nan = np.isnan(train.values()).any()
            val_has_nan = np.isnan(val.values()).any()
            test_has_nan = np.isnan(test.values()).any()
            
            if train_has_nan or val_has_nan or test_has_nan:
                st.error("""
                ❌ **ERROR**: NaN values detected in data after preprocessing!
                
                This will cause `nan` losses during training.
                
                **Solution**: Go back to home page and select a missing value handling method
                (e.g., **Linear Interpolation** instead of **Drop rows**).
                """)
                st.stop()
            
            # Check for extreme values (unnormalized)
            train_values = train.values()
            if train_values.max() > 1000 or train_values.min() < -1000:
                st.warning(f"""
                ⚠️ **Extreme values detected**: min={train_values.min():.2f}, max={train_values.max():.2f}
                
                This may cause training issues. Ensure normalization is applied.
                """)

            progress_bar.progress(30 + idx * 10)

            # 5. Créer et entraîner le modèle
            from dashboard.utils.training import run_training_pipeline
            from dashboard.utils.callbacks import StreamlitProgressCallback
            from dashboard.config import CHECKPOINTS_DIR
            from dashboard.utils.model_factory import ModelFactory

            # =====================================================================
            # OPTUNA MODE
            # =====================================================================
            if training_mode == "🧪 Optuna":
                status_text.text(f"🧪 Optuna optimization for {station_name}...")
                
                st.markdown("---")
                st.markdown("### 🧪 Optuna Optimization")
                
                optuna_progress = st.progress(0)
                optuna_status = st.empty()
                optuna_best = st.empty()
                optuna_logs = st.empty()
                
                from dashboard.utils.optuna_training import create_optuna_objective, run_optuna_study
                import optuna
                
                # Liste pour stocker les logs des trials
                trial_logs = []
                
                # Callback de progression avec logs
                def optuna_callback(study, trial):
                    n_done = len(study.trials)
                    optuna_progress.progress(min(n_done / n_trials, 1.0))
                    
                    # Status du trial
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        status_emoji = "✅"
                        score = f"{trial.value:.4f}"
                    elif trial.state == optuna.trial.TrialState.PRUNED:
                        status_emoji = "⏭️"
                        score = "Pruned"
                    else:
                        status_emoji = "❌"
                        score = "Failed"
                    
                    # Extraire quelques params clés
                    params_str = ", ".join([f"{k}={v}" for k, v in list(trial.params.items())[:3]])
                    
                    trial_logs.append({
                        'Trial': trial.number,
                        'Status': status_emoji,
                        f'{optuna_metric}': score,
                        'Params': params_str + "..."
                    })
                    
                    # Afficher les 5 derniers trials
                    optuna_logs.dataframe(
                        pd.DataFrame(trial_logs[-5:]),
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Show best so far
                    if study.best_trial:
                        optuna_best.success(f"🏆 **Best so far**: {optuna_metric} = {study.best_value:.4f}")
                    
                    optuna_status.text(f"Trial {n_done}/{n_trials}")
                
                # Créer l'objective
                objective = create_optuna_objective(
                    model_name=selected_model,
                    train=train,
                    val=val,
                    train_cov=train_cov,
                    val_cov=val_cov,
                    full_cov=covariates_scaled,
                    use_covariates=use_covariates and (train_cov is not None),
                    metric=optuna_metric,
                    n_epochs=15,  # Moins d'epochs pour Optuna (plus rapide)
                    early_stopping=True,
                    early_stopping_patience=5
                )
                
                # Créer l'étude manuellement pour avoir le callback complet
                study = optuna.create_study(
                    study_name=f"{selected_model}_{station_name}",
                    direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                
                # Optimiser avec callback
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=optuna_timeout * 60,
                    callbacks=[optuna_callback],
                    show_progress_bar=False
                )
                
                optuna_progress.progress(100)
                
                # Résumé des trials
                n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                
                # Stop reason
                if len(study.trials) < n_trials:
                    stop_reason = f"⏱️ Timeout reached ({optuna_timeout} min)"
                else:
                    stop_reason = f"✅ {n_trials} trials completed"
                
                st.info(f"**Summary**: {n_complete} ✅ | {n_failed} ❌ | {n_pruned} ⏭️ — {stop_reason}")
                
                # Show final results
                best_params = study.best_params
                best_value = study.best_value
                
                if best_value == float('inf'):
                    st.error("⚠️ All trials failed! Check model configuration.")
                else:
                    st.success(f"✅ Optuna finished! Best {optuna_metric}: **{best_value:.4f}**")
                
                col_best1, col_best2 = st.columns(2)
                with col_best1:
                    st.markdown("##### 🏆 Best Hyperparameters")
                    st.json(best_params)
                
                with col_best2:
                    st.markdown("##### 📈 History")
                    try:
                        from optuna.visualization import plot_optimization_history
                        fig = plot_optimization_history(study)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Chart not available")
                
                # Table of all trials
                with st.expander("📋 All trials"):
                    trials_df = study.trials_dataframe()
                    st.dataframe(trials_df, use_container_width=True)
                
                # Update hyperparams with best found
                hyperparams.update(best_params)
                
                st.markdown("---")
                status_text.text(f"🧠 Final training with best hyperparameters...")
            
            # =====================================================================
            # COMMON PREPARATION (Manual and after Optuna)
            # =====================================================================
            st.markdown("### 📈 Training Progress")

            epoch_progress = st.progress(0)
            epoch_status = st.empty()
            epoch_metrics = st.empty()
            loss_chart = st.empty()

            # Créer le callback pour PyTorch Lightning
            callback = None
            pl_trainer_kwargs = None

            if ModelFactory.is_torch_model(selected_model):
                callbacks_list = []
                
                # Callback de progression Streamlit
                callback = StreamlitProgressCallback(
                    total_epochs=n_epochs,
                    progress_bar=epoch_progress,
                    status_text=epoch_status,
                    metrics_placeholder=epoch_metrics,
                    chart_placeholder=loss_chart
                )
                callbacks_list.append(callback)
                
                # Early Stopping si activé
                if use_early_stopping:
                    from pytorch_lightning.callbacks import EarlyStopping
                    early_stop_callback = EarlyStopping(
                        monitor='val_loss',
                        patience=early_stopping_patience,
                        mode='min',
                        verbose=True
                    )
                    callbacks_list.append(early_stop_callback)

                pl_trainer_kwargs = {
                    'callbacks': callbacks_list
                }

            # Préparer le mapping des colonnes pour sauvegarder avec noms standardisés
            col_mapping = {
                'target_var': target_var,
                'covariate_vars': covariate_vars
            }

            # Lancer l'entraînement final
            training_results = run_training_pipeline(
                model_name=selected_model,
                hyperparams=hyperparams,
                train=train,
                val=val,
                test=test,
                train_cov=train_cov,
                val_cov=val_cov,
                test_cov=test_cov,
                full_cov=covariates_scaled,
                use_covariates=use_covariates and (train_cov is not None),
                save_dir=CHECKPOINTS_DIR,
                station_name=station_name,
                verbose=False,
                pl_trainer_kwargs=pl_trainer_kwargs,
                station_data_df=df_station,
                station_data_df_raw=df_station_raw,
                column_mapping=col_mapping,
                target_preprocessor=target_preprocessor,
                cov_preprocessor=cov_preprocessor if covariates_series is not None else None,
                original_filename=st.session_state.get('training_filename', 'unknown'),
                preprocessing_config=preprocessing_config
            )

            progress_bar.progress(90 + idx * 5)

            # Stocker les résultats
            results_all_stations[station_name] = training_results

        progress_bar.progress(100)
        status_text.text("✅ Training complete!")

        st.success(f"🎉 Training complete for {len(selected_stations)} station(s)!")

        # Show results
        st.markdown("### 📊 Results")

        # Tableau récapitulatif
        summary_data = []
        for station_name, result in results_all_stations.items():
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                summary_data.append({
                    'Station': station_name,
                    'Status': '✅ Success',
                    'MAE': f"{metrics.get('MAE', 0):.4f}" if metrics.get('MAE') else 'N/A',
                    'RMSE': f"{metrics.get('RMSE', 0):.4f}" if metrics.get('RMSE') else 'N/A',
                    'R²': f"{metrics.get('R2', 0):.4f}" if metrics.get('R2') else 'N/A',
                    'Saved': '✅' if 'saved_path' in result else '❌'
                })
            else:
                summary_data.append({
                    'Station': station_name,
                    'Status': '❌ Error',
                    'MAE': 'N/A',
                    'RMSE': 'N/A',
                    'R²': 'N/A',
                    'Saved': '❌'
                })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Détails par station
        for station_name, result in results_all_stations.items():
            with st.expander(f"📍 Détails - {station_name}"):
                if result['status'] == 'success':
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Métriques")
                        metrics = result.get('metrics', {})
                        for metric_name, metric_value in metrics.items():
                            if not pd.isna(metric_value):
                                st.metric(metric_name, f"{metric_value:.4f}")

                    with col2:
                        st.markdown("#### Info modèle")
                        st.info(f"""
                        - Modèle : {result['model_name']}
                        - Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}
                        - Checkpoint : {result.get('saved_path', 'Non sauvegardé')}
                        """)

                    # Afficher les hyperparamètres
                    with st.expander("⚙️ Hyperparamètres utilisés"):
                        st.json(hyperparams)

                else:
                    st.error(f"Erreur : {result.get('error', 'Inconnue')}")
                    if 'traceback' in result:
                        st.code(result['traceback'])

        st.markdown("---")
        st.success("""
        ✅ **Entraînement terminé avec succès !**

        Les modèles ont été entraînés et sauvegardés dans `checkpoints/darts/`.

        **Prochaines étapes** :
        - Utilisez la page **Forecasting** pour faire des prédictions
        - Téléchargez l'archive ci-dessous pour sauvegarder le modèle
        """)
        
        # Bouton téléchargement pour chaque modèle entraîné
        from dashboard.utils.export import add_download_button
        for station_name, result in results_all_stations.items():
            if result['status'] == 'success' and 'saved_path' in result:
                saved_path = Path(result['saved_path'])
                if saved_path.exists():
                    st.markdown(f"**📦 {station_name}**")
                    add_download_button(saved_path.parent, key_suffix=f"train_{station_name}")

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")

