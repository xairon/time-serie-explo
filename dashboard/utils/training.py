"""Training module for Darts models.

Protocol d'entraînement (standard type article de recherche):

- Split temporel strict train/val/test (aucun shuffle).
- Scalers (normalisation) fit uniquement sur train, transform sur val/test.
- Entraînement sur train; validation sur val (early stopping uniquement).
- Test jamais utilisé avant l'évaluation finale.
- Covariables: past_covariates uniquement (pas de future → pas de fuite).
- Métriques calculées sur l'échelle originale (inverse transform).

Checklist (aucun effet de bord, pas de fuite):
- split_train_val_test: tranches contiguës, pas de shuffle.
- fit_transform sur train uniquement; transform sur val/test (preprocessing).
- model.fit(series=train, val_series=val, past_covariates only).
- full_train = concatenate([train, val]) : nouvelle série, train/val inchangés.
- evaluate_model : inverse_transform crée de nouvelles séries; test du callant non modifié.
- calculate_metrics : slice_intersect renvoie de nouveaux objets; actual/predicted inchangés.
- horizon : min(output_chunk, len(test)) single, min sur len(ts) pour list (global).
- Préconditions : train_ratio, val_ratio, test_ratio > 0 ; test non vide ; output_chunk_length >= 1.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union, Sequence
from pathlib import Path
from darts import TimeSeries, concatenate
from darts.metrics import mae, rmse, mape, smape
from darts.models.forecasting.forecasting_model import ForecastingModel

from dashboard.utils.model_factory import ModelFactory


def train_model(
    model: ForecastingModel,
    train_series: Union[TimeSeries, Sequence[TimeSeries]],
    val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    train_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    verbose: bool = True
) -> ForecastingModel:
    """
    Trains a Darts model.

    Args:
        model: Darts model instance
        train_series: Training series
        val_series: Validation series (optional)
        train_past_covariates: Past covariates for training
        val_past_covariates: Past covariates for validation
        verbose: Whether to print logs

    Returns:
        Trained model
    """
    # Prepare fit arguments
    fit_kwargs = {
        'series': train_series,
        'verbose': verbose
    }

    # Add validation if provided
    if val_series is not None:
        fit_kwargs['val_series'] = val_series

    # Add past covariates for training (no future covariates to avoid bias)
    if train_past_covariates is not None:
        fit_kwargs['past_covariates'] = train_past_covariates

        # If validation is provided, validation covariates are required
        if val_series is not None and val_past_covariates is not None:
            fit_kwargs['val_past_covariates'] = val_past_covariates

    # Train
    model.fit(**fit_kwargs)

    return model


def evaluate_model(
    model: ForecastingModel,
    train_series: Union[TimeSeries, Sequence[TimeSeries]],
    test_series: Union[TimeSeries, Sequence[TimeSeries]],
    horizon: int,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    num_samples: int = 1,
    target_scaler: Optional[Any] = None
) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Dict[str, float]]:
    """
    Evaluates a model on the test set.

    Ne modifie pas test_series du callant : inverse_transform renvoie de nouvelles
    séries ; on ne fait que rebind des variables locales.

    Args:
        model: Trained model
        train_series: Training series (required for some models)
        test_series: Test series
        horizon: Forecast horizon
        past_covariates: Past covariates (FULL dataset)
        num_samples: Number of samples for probabilistic prediction
        target_scaler: Scaler(s) to inverse transform predictions (optional)

    Returns:
        Tuple (predictions, metrics)
    """
    # Determine horizon and target series for prediction
    is_list = isinstance(test_series, list) or isinstance(test_series, tuple)
    
    if is_list:
        horizon_chk = len(test_series[0])
    else:
        horizon_chk = len(test_series)
        
    pred_kwargs = {
        'n': min(horizon, horizon_chk),
        'series': train_series,
        'num_samples': num_samples
    }

    # Only use past covariates to avoid prediction bias
    if past_covariates is not None:
        pred_kwargs['past_covariates'] = past_covariates

    predictions = model.predict(**pred_kwargs)
    
    # INVERSE TRANSFORM (Metric Calculation on Original Scale)
    if target_scaler:
        if is_list:
            # Global/Multi-series: target_scaler should be a list or a single scaler
            if isinstance(target_scaler, list):
                # Apply each scaler to corresponding series
                predictions = [scaler.inverse_transform(p) for scaler, p in zip(target_scaler, predictions)]
                test_series = [scaler.inverse_transform(t) for scaler, t in zip(target_scaler, test_series)]
            else:
                # Apply single scaler to all
                predictions = target_scaler.inverse_transform(predictions)
                test_series = target_scaler.inverse_transform(test_series)
        else:
            # Single series
            predictions = target_scaler.inverse_transform(predictions)
            test_series = target_scaler.inverse_transform(test_series)

    # Calculate metrics
    metrics = calculate_metrics(test_series, predictions)

    return predictions, metrics


def evaluate_model_sliding(
    model: ForecastingModel,
    full_train: TimeSeries,
    test: TimeSeries,
    horizon: int,
    past_covariates: Optional[TimeSeries] = None,
    target_scaler: Optional[Any] = None,
    stride: Optional[int] = None,
) -> Dict[str, float]:
    """
    Évalue le modèle sur le test via fenêtrage glissant (sliding window).
    Standard pour séries temporelles : plusieurs fenêtres non chevauchantes,
    métriques agrégées (moyenne MAE, RMSE, etc.).

    Utilise historical_forecasts sur full_series = train+val+test, start = début test,
    stride = horizon (fenêtres non chevauchantes). Uniquement série unique (pas global).

    Returns:
        Métriques agrégées (moyenne sur les fenêtres), ou dict vide si échec.
    """
    stride = stride or horizon
    full_series = concatenate([full_train, test], axis=0)
    start = test.start_time()

    pred_kwargs = {
        'series': full_series,
        'start': start,
        'forecast_horizon': horizon,
        'stride': stride,
        'last_points_only': False,
        'verbose': False,
    }
    if past_covariates is not None and getattr(model, 'supports_past_covariates', False):
        pred_kwargs['past_covariates'] = past_covariates

    try:
        raw = model.historical_forecasts(**pred_kwargs)
    except Exception:
        return {}

    if isinstance(raw, TimeSeries):
        raw = [raw]

    mlist = ['MAE', 'RMSE', 'MAPE', 'sMAPE']
    agg: Dict[str, List[float]] = {k: [] for k in mlist}
    dir_accs: List[float] = []

    for fc in raw:
        actual_slice = test.slice_intersect(fc)
        if len(actual_slice) == 0:
            continue
        pred_slice = fc.slice_intersect(actual_slice)
        if len(pred_slice) == 0:
            continue
        if target_scaler is not None:
            actual_slice = target_scaler.inverse_transform(actual_slice)
            pred_slice = target_scaler.inverse_transform(pred_slice)
        m = calculate_metrics(actual_slice, pred_slice, metrics_list=mlist + ['Dir_Acc'])
        for k in mlist:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(float(v))
        da = m.get('Dir_Acc')
        if da is not None and not (isinstance(da, float) and np.isnan(da)):
            dir_accs.append(float(da))

    out: Dict[str, float] = {}
    for k in mlist:
        if agg[k]:
            out[k] = float(np.nanmean(agg[k]))
    if dir_accs:
        out['Dir_Acc'] = float(np.nanmean(dir_accs))
    return out


def calculate_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
    metrics_list: Optional[list] = None
) -> Dict[str, float]:
    """
    Calculates evaluation metrics.

    Args:
        actual: Actual series
        predicted: Predicted series
        metrics_list: List of metrics to calculate

    Returns:
        Dictionary with metrics
    """
    if metrics_list is None:
        metrics_list = ['MAE', 'RMSE', 'MAPE', 'sMAPE']

    results = {}

    is_list = isinstance(actual, list) or isinstance(actual, tuple)
    
    if is_list:
        actual_aligned = []
        predicted_aligned = []
        for act, pred in zip(actual, predicted):
            actual_aligned.append(act.slice_intersect(pred))
            predicted_aligned.append(pred.slice_intersect(act))
    else:
        actual_aligned = actual.slice_intersect(predicted)
        predicted_aligned = predicted.slice_intersect(actual)

    try:
        if 'MAE' in metrics_list:
            results['MAE'] = mae(actual_aligned, predicted_aligned)
    except Exception:
        results['MAE'] = np.nan

    try:
        if 'RMSE' in metrics_list:
            results['RMSE'] = rmse(actual_aligned, predicted_aligned)
    except Exception:
        results['RMSE'] = np.nan

    try:
        if 'MAPE' in metrics_list:
            results['MAPE'] = mape(actual_aligned, predicted_aligned)
    except Exception:
        results['MAPE'] = np.nan



    try:
        if 'sMAPE' in metrics_list:
            results['sMAPE'] = smape(actual_aligned, predicted_aligned)
    except Exception:
        results['sMAPE'] = np.nan

    # Direction accuracy
    try:
        if is_list:
             accs = []
             for act, pred in zip(actual_aligned, predicted_aligned):
                 actual_diff = np.diff(act.values().flatten())
                 pred_diff = np.diff(pred.values().flatten())
                 if len(actual_diff) > 0:
                     correct = np.sum((actual_diff > 0) == (pred_diff > 0))
                     accs.append(correct / len(actual_diff) * 100)
             results['Dir_Acc'] = np.mean(accs) if accs else np.nan
        else:
            actual_diff = np.diff(actual_aligned.values().flatten())
            pred_diff = np.diff(predicted_aligned.values().flatten())
            correct_direction = np.sum((actual_diff > 0) == (pred_diff > 0))
            results['Dir_Acc'] = correct_direction / len(actual_diff) * 100
    except Exception:
        results['Dir_Acc'] = np.nan

    return results


def save_model(
    model: ForecastingModel,
    save_dir: Path,
    model_name: str,
    station: str,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Saves a trained model (deprecated method).

    DEPRECATED: Use run_training_pipeline with save_dir instead.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{station}.pkl"
    filepath = save_dir / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(filepath))

    if metadata:
        import json
        metadata_path = save_dir / f"{model_name}_{station}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_model(filepath: Path) -> ForecastingModel:
    """
    Loads a saved model.

    Args:
        filepath: Path to the model file

    Returns:
        Loaded model
    """
    from darts.models.forecasting.forecasting_model import ForecastingModel
    model = ForecastingModel.load(str(filepath))
    return model


def run_training_pipeline(
    model_name: str,
    hyperparams: Dict[str, Any],
    train: Union[TimeSeries, Sequence[TimeSeries]],
    val: Union[TimeSeries, Sequence[TimeSeries]],
    test: Union[TimeSeries, Sequence[TimeSeries]],
    train_cov: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    val_cov: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    test_cov: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    full_cov: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    use_covariates: bool = True,
    save_dir: Optional[Path] = None,
    station_name: str = 'default',
    verbose: bool = True,
    progress_callback: Optional[Any] = None,  # DEPRECATED: Use metrics_file instead
    pl_trainer_kwargs: Optional[Dict[str, Any]] = None,  # DEPRECATED: Use metrics_file and early_stopping_patience instead
    station_data_df: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    station_data_df_raw: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    target_preprocessor: Optional[Any] = None,
    cov_preprocessor: Optional[Any] = None,
    original_filename: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    all_stations: Optional[List[str]] = None,  # For global models: list of all station IDs
    early_stopping_patience: Optional[int] = None,
    metrics_file: Optional[Path] = None,  # NEW: Path to JSON file for metrics (replaces Streamlit callbacks)
    n_epochs: Optional[int] = None  # NEW: Total epochs for metrics callback
) -> Dict[str, Any]:
    """
    Complete training pipeline with split saving.

    Protocol (standard type article): split temporel → fit scalers on train only →
    train on train, early-stop on val → evaluate on test only; past_covariates only.

    Args:
        model_name: Model name
        hyperparams: Hyperparameters
        train, val, test: Train/Val/Test Darts TimeSeries (split temporel strict)
        train_cov, val_cov, test_cov: Split covariates
        full_cov: Full covariates (train+val+test) for prediction
        use_covariates: Whether to use covariates
        save_dir: Directory to save results
        station_name: Station name
        verbose: Verbose logging
        progress_callback: Progress callback
        pl_trainer_kwargs: PyTorch Lightning Trainer kwargs
        station_data_df: Full PROCESSED station DataFrame (required for saving)
        station_data_df_raw: Full RAW station DataFrame (for display)
        column_mapping: Column mapping
        target_preprocessor: Fitted scaler for target
        cov_preprocessor: Fitted scaler for covariates
        original_filename: Original filename
        preprocessing_config: Preprocessing configuration

    Returns:
        Dictionary with results (model, metrics, predictions, saved_path)
    """
    results = {
        'model_name': model_name,
        'station': station_name,
        'status': 'success'
    }

    try:
        # 0. Reproducibilité (graines fixes)
        import torch
        import pytorch_lightning as pl
        try:
            from dashboard.config import RANDOM_SEED
        except Exception:
            RANDOM_SEED = 42
        pl.seed_everything(RANDOM_SEED, workers=True)

        # 1. Configure Trainer (GPU & Callbacks)
        from core.callbacks import create_training_callbacks

        trainer_kwargs = {}
        
        # Merge avec pl_trainer_kwargs si fourni (pour compatibilité)
        if pl_trainer_kwargs:
            trainer_kwargs.update(pl_trainer_kwargs)
        
        # Auto-detect GPU
        if torch.cuda.is_available():
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = 1
            if verbose:
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            trainer_kwargs['accelerator'] = 'cpu'
            if verbose:
                print("⚠️ GPU not available, using CPU")

        # Configure Callbacks - NOUVELLE APPROCHE STANDARD
        # On utilise uniquement des callbacks standards qui n'ont pas de dépendances Streamlit
        callbacks = create_training_callbacks(
            metrics_file=metrics_file,
            total_epochs=n_epochs or hyperparams.get('n_epochs'),
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor="val_loss",
            early_stopping_mode="min"
        )
        
        # Si pl_trainer_kwargs contient des callbacks (ancien code), on les ignore
        if 'callbacks' in trainer_kwargs:
            if verbose:
                print("⚠️ Warning: Callbacks from pl_trainer_kwargs are ignored. Use metrics_file instead.")
            trainer_kwargs['callbacks'] = callbacks
        else:
            trainer_kwargs['callbacks'] = callbacks
        
        if verbose and metrics_file:
            print(f"📊 Metrics will be written to: {metrics_file}")

        # 2. Create model
        model = ModelFactory.create_model(
            model_name,
            hyperparams,
            pl_trainer_kwargs_override=trainer_kwargs
        )

        # 3. Prepare covariates (past_covariates only — no future, no leakage)
        train_past_cov = None
        val_past_cov = None

        if use_covariates and train_cov is not None:
            supports_past = getattr(model, "supports_past_covariates", False)
            
            if supports_past:
                train_past_cov = train_cov
                val_past_cov = val_cov

        # 4. Train (train only; val for early stopping only)
        model = train_model(
            model=model,
            train_series=train,
            val_series=val,
            train_past_covariates=train_past_cov,
            val_past_covariates=val_past_cov,
            verbose=verbose
        )

        results['model'] = model

        # 5. Evaluate on test only (original scale via inverse transform)
        output_chunk = hyperparams.get('output_chunk_length', 7)

        # Historique = train + val (sans muter train/val; concatenate retourne une nouvelle série)
        if isinstance(train, list) and isinstance(val, list):
            full_train = [concatenate([t, v], axis=0) for t, v in zip(train, val)]
        else:
            full_train = concatenate([train, val], axis=0)

        # Longueur test: single → n_timesteps; global (list) → n_timesteps de chaque série
        if isinstance(test, (list, tuple)):
            _test_len = min(len(ts) for ts in test)
        else:
            _test_len = len(test)
        _horizon = min(output_chunk, _test_len)
        if _horizon <= 0:
            raise ValueError(
                "Test set too short or output_chunk_length too small for evaluation. "
                "Ensure test_ratio > 0 and output_chunk_length >= 1."
            )

        # Full covariates for prediction (only past_covariates)
        full_past_cov = None
        if use_covariates and full_cov is not None:
            if getattr(model, "supports_past_covariates", False):
                full_past_cov = full_cov

        # Prepare scalers for evaluation
        eval_scaler = None
        if target_preprocessor:
            if isinstance(train, list): # Global model
                if isinstance(target_preprocessor, dict):
                    # Map station index to scaler
                    # all_stations is required here to map index -> station_name -> scaler
                    if all_stations:
                        eval_scaler = [target_preprocessor.get(s) for s in all_stations]
                    else:
                        # Fallback: repeat first scaler if no station list (risky but better than crash)
                        first_scaler = list(target_preprocessor.values())[0]
                        eval_scaler = [first_scaler] * len(train)
                else:
                    # Single scaler for all series (unlikely for global but possible)
                    eval_scaler = target_preprocessor
            else: # Single model
                eval_scaler = target_preprocessor

        predictions, metrics = evaluate_model(
            model=model,
            train_series=full_train,
            test_series=test,
            horizon=_horizon,
            past_covariates=full_past_cov,
            num_samples=1,
            target_scaler=eval_scaler
        )

        results['predictions'] = predictions
        results['metrics'] = metrics

        # 5b. Sliding-window evaluation on test (single-series only; standard pratique)
        metrics_sliding: Dict[str, float] = {}
        if not isinstance(train, (list, tuple)):
            _full_cov = full_past_cov if isinstance(full_past_cov, TimeSeries) else None
            metrics_sliding = evaluate_model_sliding(
                model=model,
                full_train=full_train,
                test=test,
                horizon=_horizon,
                past_covariates=_full_cov,
                target_scaler=eval_scaler,
                stride=_horizon,
            )
        results['metrics_sliding'] = metrics_sliding

        # 6. Save if requested (model + config + splits)
        if save_dir and station_data_df is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            clean_station_name = station_name.split('/')[-1] if '/' in station_name else station_name
            is_global = isinstance(train, list) or isinstance(train, tuple)
            
            # Initialize sizes with defaults to avoid UnboundLocalError
            train_size = 0
            val_size = 0
            test_size = 0
            
            if is_global:
                # Reconstruct DataFrames WITH station column using all_stations mapping
                # train, val, test are lists of SCALED TimeSeries
                # train_cov, val_cov, test_cov are lists of SCALED TimeSeries (if used)
                station_list = all_stations if all_stations else []
                
                if station_list and len(train) == len(station_list):
                    train_dfs = []
                    val_dfs = []
                    test_dfs = []
                    
                    for i, station in enumerate(station_list):
                        # Train
                        df_t = train[i].to_dataframe()
                        if train_cov and isinstance(train_cov, list) and i < len(train_cov):
                            df_t = pd.concat([df_t, train_cov[i].to_dataframe()], axis=1)
                        df_t['station'] = station
                        train_dfs.append(df_t)
                        
                        # Val
                        if i < len(val):
                            df_v = val[i].to_dataframe()
                            if val_cov and isinstance(val_cov, list) and i < len(val_cov):
                                df_v = pd.concat([df_v, val_cov[i].to_dataframe()], axis=1)
                            df_v['station'] = station
                            val_dfs.append(df_v)
                            
                        # Test
                        if i < len(test):
                            df_ts = test[i].to_dataframe()
                            if test_cov and isinstance(test_cov, list) and i < len(test_cov):
                                df_ts = pd.concat([df_ts, test_cov[i].to_dataframe()], axis=1)
                            df_ts['station'] = station
                            test_dfs.append(df_ts)
                    
                    train_df = pd.concat(train_dfs)
                    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame()
                    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame()
                else:
                    # Fallback (loses station info, but better than crash)
                    train_df = pd.concat([t.to_dataframe() for t in train])
                    val_df = pd.concat([t.to_dataframe() for t in val])
                    test_df = pd.concat([t.to_dataframe() for t in test])
                
                full_df = pd.concat([train_df, val_df, test_df])
                
                # Calculate sizes for global models
                train_size = len(train_df)
                val_size = len(val_df)
                test_size = len(test_df)
                
                # Raw data reconstruction via inverse transform
                # For global models, generate raw data from normalized data using station-specific scalers
                train_raw_list = []
                val_raw_list = []
                test_raw_list = []

                # Check if we have station-specific scalers
                has_dict_scalers = isinstance(target_preprocessor, dict)

                for i, station in enumerate(station_list):
                    # Get the appropriate scaler for this station
                    if has_dict_scalers and station in target_preprocessor:
                        station_target_scaler = target_preprocessor[station]
                        station_cov_scaler = cov_preprocessor.get(station) if isinstance(cov_preprocessor, dict) else cov_preprocessor
                    elif has_dict_scalers:
                        # Fallback to first available scaler
                        station_target_scaler = list(target_preprocessor.values())[0] if target_preprocessor else None
                        station_cov_scaler = list(cov_preprocessor.values())[0] if isinstance(cov_preprocessor, dict) and cov_preprocessor else cov_preprocessor
                    else:
                        # Single scaler for all
                        station_target_scaler = target_preprocessor
                        station_cov_scaler = cov_preprocessor

                    # Inverse transform train
                    if i < len(train) and station_target_scaler:
                        train_raw = station_target_scaler.inverse_transform(train[i])
                        df_t_raw = train_raw.to_dataframe()
                        if train_cov and isinstance(train_cov, list) and i < len(train_cov) and station_cov_scaler:
                            cov_raw = station_cov_scaler.inverse_transform(train_cov[i])
                            df_t_raw = pd.concat([df_t_raw, cov_raw.to_dataframe()], axis=1)
                        df_t_raw['station'] = station
                        train_raw_list.append(df_t_raw)

                    # Inverse transform val
                    if i < len(val) and station_target_scaler:
                        val_raw = station_target_scaler.inverse_transform(val[i])
                        df_v_raw = val_raw.to_dataframe()
                        if val_cov and isinstance(val_cov, list) and i < len(val_cov) and station_cov_scaler:
                            cov_raw = station_cov_scaler.inverse_transform(val_cov[i])
                            df_v_raw = pd.concat([df_v_raw, cov_raw.to_dataframe()], axis=1)
                        df_v_raw['station'] = station
                        val_raw_list.append(df_v_raw)

                    # Inverse transform test
                    if i < len(test) and station_target_scaler:
                        test_raw = station_target_scaler.inverse_transform(test[i])
                        df_ts_raw = test_raw.to_dataframe()
                        if test_cov and isinstance(test_cov, list) and i < len(test_cov) and station_cov_scaler:
                            cov_raw = station_cov_scaler.inverse_transform(test_cov[i])
                            df_ts_raw = pd.concat([df_ts_raw, cov_raw.to_dataframe()], axis=1)
                        df_ts_raw['station'] = station
                        test_raw_list.append(df_ts_raw)

                train_df_raw = pd.concat(train_raw_list) if train_raw_list else None
                val_df_raw = pd.concat(val_raw_list) if val_raw_list else None
                test_df_raw = pd.concat(test_raw_list) if test_raw_list else None
                    
                station_data_df = full_df

            else:
                # SINGLE MODE
                # Use train/val/test (SCALED TimeSeries) directly + merge covariates
                
                # Train (SCALED)
                train_df = train.to_dataframe()
                if train_cov:
                    train_df = pd.concat([train_df, train_cov.to_dataframe()], axis=1)
                
                # Val (SCALED)
                val_df = val.to_dataframe()
                if val_cov:
                    val_df = pd.concat([val_df, val_cov.to_dataframe()], axis=1)
                    
                # Test (SCALED)
                test_df = test.to_dataframe()
                if test_cov:
                    test_df = pd.concat([test_df, test_cov.to_dataframe()], axis=1)

                train_size = len(train_df)
                val_size = len(val_df)
                test_size = len(test_df)
                
                full_df = pd.concat([train_df, val_df, test_df])
                
                # Generate RAW data via inverse_transform (guaranteed alignment!)
                train_df_raw = None
                val_df_raw = None
                test_df_raw = None
                
                if target_preprocessor is not None:
                    # Inverse transform target for each split
                    train_raw_target = target_preprocessor.inverse_transform(train)
                    val_raw_target = target_preprocessor.inverse_transform(val)
                    test_raw_target = target_preprocessor.inverse_transform(test)
                    
                    train_df_raw = train_raw_target.to_dataframe()
                    val_df_raw = val_raw_target.to_dataframe()
                    test_df_raw = test_raw_target.to_dataframe()
                    
                    # Add inverse-transformed covariates if available
                    if cov_preprocessor is not None and train_cov is not None:
                        train_df_raw = pd.concat([train_df_raw, cov_preprocessor.inverse_transform(train_cov).to_dataframe()], axis=1)
                        val_df_raw = pd.concat([val_df_raw, cov_preprocessor.inverse_transform(val_cov).to_dataframe()], axis=1)
                        test_df_raw = pd.concat([test_df_raw, cov_preprocessor.inverse_transform(test_cov).to_dataframe()], axis=1)

            if column_mapping:
                columns_config = {
                    'date': 'date',
                    'target': column_mapping['target_var'],
                    'covariates': column_mapping['covariate_vars']
                }
            else:
                columns_config = {
                    'date': 'date',
                    'target': 'level',
                    'covariates': ['PRELIQ_Q', 'T_Q', 'ETP_Q']
                }

            # Fallback pour Source/Dataset "unknown": utiliser target + covariables
            _src = (original_filename or '').strip()
            if not _src or _src.lower() == 'unknown':
                _t = columns_config.get('target', 'target')
                _c = columns_config.get('covariates') or []
                _cov = '+'.join(str(x) for x in _c[:5])
                _src = f"{_t}" + (f"+{_cov}" if _cov else "")

            if preprocessing_config:
                preproc = {
                    'fill_method': preprocessing_config.get('fill_method', 'Unknown'),
                    'scaler_type': preprocessing_config.get('scaler_type', 'StandardScaler'),
                    'columns': columns_config  # Include target and covariates for dataset identification
                }
            else:
                preproc = {
                    'fill_method': 'Unknown',
                    'scaler_type': 'StandardScaler',
                    'columns': columns_config
                }

            from dashboard.utils.model_config import ModelConfig, save_model_with_data

            # Helper to safely handle metrics that might be arrays (global models)
            def _safe_metric(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = np.nanmean(val)
                try:
                    vf = float(val)
                    return None if np.isnan(vf) else vf
                except:
                    return None

            _sliding = results.get('metrics_sliding') or {}
            _sliding_safe = {k: _safe_metric(v) for k, v in _sliding.items() if _safe_metric(v) is not None}

            config = ModelConfig(
                model_name=model_name,
                station=clean_station_name,
                original_station_id=station_name,
                columns=columns_config,
                data_source={
                    'type': 'embedded',
                    'original_file': _src
                },
                splits={
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size
                },
                preprocessing=preproc,
                hyperparams={k: str(v) for k, v in hyperparams.items()},
                metrics={k: _safe_metric(v) for k, v in metrics.items()},
                use_covariates=use_covariates,
                metrics_sliding=_sliding_safe if _sliding_safe else None
            )

            # Determine model type and stations list
            model_type = "global" if is_global else "single"
            stations_list = all_stations if all_stations else [station_name]
            
            model_dir = save_model_with_data(
                model=model,
                save_dir=save_dir,
                model_name=model_name,
                station_name=station_name,
                config=config,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                full_df=station_data_df,
                target_preprocessor=target_preprocessor,
                cov_preprocessor=cov_preprocessor,
                train_df_raw=train_df_raw,
                val_df_raw=val_df_raw,
                test_df_raw=test_df_raw,
                model_type=model_type,
                stations=stations_list
            )

            results['saved_path'] = str(model_dir)
            results['scalers_saved'] = str(model_dir / "scalers.pkl")

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results
