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
    train_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        train_future_covariates: Future covariates for training
        val_future_covariates: Future covariates for validation
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

    # Add covariates for training
    if train_past_covariates is not None:
        fit_kwargs['past_covariates'] = train_past_covariates

        # If validation is provided, validation covariates are required
        if val_series is not None and val_past_covariates is not None:
            fit_kwargs['val_past_covariates'] = val_past_covariates

    if train_future_covariates is not None:
        fit_kwargs['future_covariates'] = train_future_covariates

        # If validation is provided, validation covariates are required
        if val_series is not None and val_future_covariates is not None:
            fit_kwargs['val_future_covariates'] = val_future_covariates

    # Train
    model.fit(**fit_kwargs)

    return model


def evaluate_model(
    model: ForecastingModel,
    train_series: Union[TimeSeries, Sequence[TimeSeries]],
    test_series: Union[TimeSeries, Sequence[TimeSeries]],
    horizon: int,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        future_covariates: Future covariates (FULL dataset)
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

    # Covariates must cover the FULL range (train + test)
    if past_covariates is not None:
        pred_kwargs['past_covariates'] = past_covariates

    if future_covariates is not None:
        pred_kwargs['future_covariates'] = future_covariates

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
    preprocessing_config: Optional[Dict[str, Any]] = None
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

        # 2. Prepare covariates based on model support
        train_past_cov = None
        val_past_cov = None
        train_future_cov = None
        val_future_cov = None

        if use_covariates and train_cov is not None:
            supports_past = getattr(model, "supports_past_covariates", False)
            supports_future = getattr(model, "supports_future_covariates", False)
            
            if supports_past:
                train_past_cov = train_cov
                val_past_cov = val_cov
            elif supports_future:
                train_future_cov = train_cov
                val_future_cov = val_cov

        # 3. Train
        model = train_model(
            model=model,
            train_series=train,
            val_series=val,
            train_past_covariates=train_past_cov,
            val_past_covariates=val_past_cov,
            train_future_covariates=train_future_cov,
            val_future_covariates=val_future_cov,
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

        # Full covariates for prediction
        full_past_cov = None
        full_future_cov = None
        if use_covariates and full_cov is not None:
            if getattr(model, "supports_past_covariates", False):
                full_past_cov = full_cov
            elif getattr(model, "supports_future_covariates", False):
                full_future_cov = full_cov

        predictions, metrics = evaluate_model(
            model=model,
            train_series=full_train,
            test_series=test,
            horizon=min(output_chunk, len(test)),
            past_covariates=full_past_cov,
            future_covariates=full_future_cov,
            num_samples=1
        )

        results['predictions'] = predictions
        results['metrics'] = metrics

        # 5. Save if requested (NEW SYSTEM with splits)
        if save_dir and station_data_df is not None:
            clean_station_name = station_name.split('/')[-1] if '/' in station_name else station_name
            is_global = isinstance(train, list) or isinstance(train, tuple)

            if is_global:
                train_df = pd.concat([t.to_dataframe() for t in train])
                val_df = pd.concat([t.to_dataframe() for t in val])
                test_df = pd.concat([t.to_dataframe() for t in test])
                
                train_size = len(train_df)
                val_size = len(val_df)
                test_size = len(test_df)
                
                full_df = pd.concat([train_df, val_df, test_df])
                
                train_df_raw = None
                val_df_raw = None
                test_df_raw = None
                
                if isinstance(station_data_df_raw, dict):
                    raw_list = []
                    for s, df in station_data_df_raw.items():
                        d = df.copy()
                        d['station'] = s
                        raw_list.append(d)
                    
                    if raw_list:
                        station_data_df_raw = pd.concat(raw_list)
                    else:
                        station_data_df_raw = None
                
                station_data_df = full_df

            else:
                train_size = len(train)
                val_size = len(val)
                test_size = len(test)

                train_df = station_data_df.iloc[:train_size].copy()
                val_df = station_data_df.iloc[train_size:train_size + val_size].copy()
                test_df = station_data_df.iloc[train_size + val_size:].copy()
                
                train_df_raw = None
                val_df_raw = None
                test_df_raw = None
                
                if station_data_df_raw is not None and not isinstance(station_data_df_raw, dict):
                    train_df_raw = station_data_df_raw.iloc[:train_size].copy()
                    val_df_raw = station_data_df_raw.iloc[train_size:train_size + val_size].copy()
                    test_df_raw = station_data_df_raw.iloc[train_size + val_size:].copy()

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
                    'scaler_type': preprocessing_config.get('scaler_type', 'StandardScaler')
                }
            else:
                preproc = {
                    'fill_method': 'Unknown',
                    'scaler_type': 'StandardScaler'
                }

            from dashboard.utils.model_config import ModelConfig, save_model_with_data

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
                metrics={k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
                use_covariates=use_covariates
            )

            model_dir = save_model_with_data(
                model=model,
                save_dir=save_dir,
                model_name=model_name,
                station_name=clean_station_name,
                config=config,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                full_df=station_data_df,
                target_preprocessor=target_preprocessor,
                cov_preprocessor=cov_preprocessor,
                train_df_raw=train_df_raw,
                val_df_raw=val_df_raw,
                test_df_raw=test_df_raw
            )

            results['saved_path'] = str(model_dir)
            results['scalers_saved'] = str(model_dir / "scalers.pkl")

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results
