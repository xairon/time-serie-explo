"""Training module for Darts models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union, Sequence
from pathlib import Path
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, r2_score, smape
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
    num_samples: int = 1
) -> Tuple[Union[TimeSeries, Sequence[TimeSeries]], Dict[str, float]]:
    """
    Evaluates a model on the test set.

    Args:
        model: Trained model
        train_series: Training series (required for some models)
        test_series: Test series
        horizon: Forecast horizon
        past_covariates: Past covariates (FULL dataset)
        num_samples: Number of samples for probabilistic prediction

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

    # Calculate metrics
    metrics = calculate_metrics(test_series, predictions)

    return predictions, metrics


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
        metrics_list = ['MAE', 'RMSE', 'MAPE', 'R2', 'sMAPE']

    results = {}

    is_list = isinstance(actual, list) or isinstance(actual, tuple)
    
    if is_list:
        actual_aligned = []
        predicted_aligned = []
        for act, pred in zip(actual, predicted):
             inter = act.slice_intersect(pred)
             actual_aligned.append(inter)
             predicted_aligned.append(pred)
    else:
        actual_aligned = actual.slice_intersect(predicted)
        predicted_aligned = predicted

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
        if 'R2' in metrics_list:
            results['R2'] = r2_score(actual_aligned, predicted_aligned)
    except Exception:
        results['R2'] = np.nan

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
    progress_callback: Optional[Any] = None,
    pl_trainer_kwargs: Optional[Dict[str, Any]] = None,
    station_data_df: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    station_data_df_raw: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    target_preprocessor: Optional[Any] = None,
    cov_preprocessor: Optional[Any] = None,
    original_filename: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    all_stations: Optional[List[str]] = None  # For global models: list of all station IDs
) -> Dict[str, Any]:
    """
    Complete training pipeline with split saving.

    Args:
        model_name: Model name
        hyperparams: Hyperparameters
        train, val, test: Train/Val/Test Darts TimeSeries
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
        # 1. Create model
        model = ModelFactory.create_model(
            model_name,
            hyperparams,
            pl_trainer_kwargs_override=pl_trainer_kwargs
        )

        # 2. Prepare covariates based on model support (only past_covariates to avoid bias)
        train_past_cov = None
        val_past_cov = None

        if use_covariates and train_cov is not None:
            supports_past = getattr(model, "supports_past_covariates", False)
            
            if supports_past:
                train_past_cov = train_cov
                val_past_cov = val_cov

        # 3. Train
        model = train_model(
            model=model,
            train_series=train,
            val_series=val,
            train_past_covariates=train_past_cov,
            val_past_covariates=val_past_cov,
            verbose=verbose
        )

        results['model'] = model

        # 4. Evaluate on test
        output_chunk = hyperparams.get('output_chunk_length', 7)

        # For prediction, we need the series up containing the test set
        if isinstance(train, list) and isinstance(val, list):
             full_train = [t.append(v) for t, v in zip(train, val)]
        else:
             full_train = train.append(val)

        # Full covariates for prediction (only past_covariates)
        full_past_cov = None
        if use_covariates and full_cov is not None:
            if getattr(model, "supports_past_covariates", False):
                full_past_cov = full_cov

        predictions, metrics = evaluate_model(
            model=model,
            train_series=full_train,
            test_series=test,
            horizon=min(output_chunk, len(test)),
            past_covariates=full_past_cov,
            num_samples=1
        )

        results['predictions'] = predictions
        results['metrics'] = metrics

        # 5. Save if requested (NEW SYSTEM with splits)
        if save_dir and station_data_df is not None:
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

            config = ModelConfig(
                model_name=model_name,
                station=clean_station_name,
                original_station_id=station_name,
                columns=columns_config,
                data_source={
                    'type': 'embedded',
                    'original_file': original_filename or 'unknown'
                },
                splits={
                    'train_size': train_size,
                    'val_size': val_size,
                    'test_size': test_size
                },
                preprocessing=preproc,
                hyperparams={k: str(v) for k, v in hyperparams.items()},
                metrics={k: _safe_metric(v) for k, v in metrics.items()},
                use_covariates=use_covariates
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
