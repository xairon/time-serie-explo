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

MLflow Integration:
- Tracing enabled for key functions via @trace_training_step
- Dataset lineage tracking
- System metrics (CPU/GPU/memory) during training
"""

import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union, Sequence
from pathlib import Path
from darts import TimeSeries, concatenate
from darts.metrics import mae, rmse, smape
from darts.models.forecasting.forecasting_model import ForecastingModel

from dashboard.utils.model_factory import ModelFactory
from dashboard.utils.mlflow_client import trace_training_step

logger = logging.getLogger(__name__)


@trace_training_step("model_fit")
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

    # Safety: never pass future covariates during training
    if getattr(model, "supports_future_covariates", False):
        logger.info("Future covariates are not used in training to avoid leakage.")

    # Train
    model.fit(**fit_kwargs)

    return model


@trace_training_step("model_evaluation")
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
    except Exception as e:
        logger.warning(f"Sliding window evaluation failed: {e}")
        return {}

    if isinstance(raw, TimeSeries):
        raw = [raw]

    mlist = ['MAE', 'RMSE', 'sMAPE', 'WAPE', 'NRMSE', 'NSE', 'KGE']
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


def _to_numpy(series: TimeSeries) -> np.ndarray:
    return series.values().flatten()


def _wape(actual_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    denom = np.sum(np.abs(actual_vals))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual_vals - pred_vals)) / denom * 100.0)


def _nrmse(actual_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    denom = np.max(actual_vals) - np.min(actual_vals)
    if denom == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual_vals - pred_vals) ** 2)) / denom * 100.0)


def _nse(actual_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    mean_obs = np.mean(actual_vals)
    ss_res = np.sum((actual_vals - pred_vals) ** 2)
    ss_tot = np.sum((actual_vals - mean_obs) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    return float(1 - (ss_res / ss_tot))


def _kge(actual_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    if np.std(actual_vals) == 0 or np.std(pred_vals) == 0:
        r = 0.0
    else:
        r = np.corrcoef(actual_vals, pred_vals)[0, 1]
    mean_obs = np.mean(actual_vals)
    mean_pred = np.mean(pred_vals)
    if mean_obs == 0:
        beta = 1.0 if mean_pred == 0 else np.inf
    else:
        beta = mean_pred / mean_obs
    std_obs = np.std(actual_vals)
    std_pred = np.std(pred_vals)
    if std_obs == 0:
        gamma = 1.0 if std_pred == 0 else np.inf
    else:
        gamma = std_pred / std_obs
    return float(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


@trace_training_step("calculate_metrics")
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
        metrics_list = ['MAE', 'RMSE', 'sMAPE', 'WAPE', 'NRMSE', 'NSE', 'KGE']

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

    def _compute_scalar_metrics(act, pred):
        a = _to_numpy(act)
        p = _to_numpy(pred)
        out = {}
        if 'WAPE' in metrics_list:
            out['WAPE'] = _wape(a, p)
        if 'NRMSE' in metrics_list:
            out['NRMSE'] = _nrmse(a, p)
        if 'NSE' in metrics_list:
            out['NSE'] = _nse(a, p)
        if 'KGE' in metrics_list:
            out['KGE'] = _kge(a, p)
        return out

    try:
        if 'MAE' in metrics_list:
            results['MAE'] = mae(actual_aligned, predicted_aligned)
    except Exception as e:
        logger.warning(f"MAE calculation failed: {e}")
        results['MAE'] = np.nan

    try:
        if 'RMSE' in metrics_list:
            results['RMSE'] = rmse(actual_aligned, predicted_aligned)
    except Exception as e:
        logger.warning(f"RMSE calculation failed: {e}")
        results['RMSE'] = np.nan

    try:
        if 'sMAPE' in metrics_list:
            results['sMAPE'] = smape(actual_aligned, predicted_aligned)
    except Exception as e:
        logger.warning(f"sMAPE calculation failed: {e}")
        results['sMAPE'] = np.nan

    try:
        if is_list:
            agg = {k: [] for k in ['WAPE', 'NRMSE', 'NSE', 'KGE']}
            for act, pred in zip(actual_aligned, predicted_aligned):
                scalar_metrics = _compute_scalar_metrics(act, pred)
                for k, v in scalar_metrics.items():
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        agg[k].append(v)
            for k, vals in agg.items():
                if vals:
                    results[k] = float(np.nanmean(vals))
        else:
            results.update(_compute_scalar_metrics(actual_aligned, predicted_aligned))
    except Exception as e:
        logger.warning(f"Scalar metrics calculation failed: {e}")

    # Direction accuracy
    try:
        if 'Dir_Acc' in metrics_list:
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
                if len(actual_diff) > 0:
                    correct_direction = np.sum((actual_diff > 0) == (pred_diff > 0))
                    results['Dir_Acc'] = correct_direction / len(actual_diff) * 100
                else:
                    results['Dir_Acc'] = np.nan
    except Exception as e:
        logger.warning(f"Dir_Acc calculation failed: {e}")
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
    dataset_name: Optional[str] = None,
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

    # MLflow Setup
    from dashboard.utils.mlflow_client import get_mlflow_manager
    mlflow_manager = get_mlflow_manager()
    
    # Normalize dataset name (for UI/registry)
    dataset_name = dataset_name or original_filename or "unknown"

    # Generate run name and tags
    run_name = f"{model_name}_{station_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tags = {
        'station': station_name,
        'model': model_name,
        'type': 'global' if (isinstance(train, list) or isinstance(train, tuple)) else 'single'
    }
    
    # Determine model type for tag
    if isinstance(train, list) or isinstance(train, tuple):
        tags['model_type'] = 'global'
        tags['station_count'] = str(len(train))
        # Log all station names for global models
        if all_stations:
            tags['stations'] = json.dumps(all_stations)
    else:
        tags['model_type'] = 'single'
        tags['station_count'] = "1"
        tags['stations'] = json.dumps([station_name])

    # START MLFLOW RUN (avec system metrics: CPU/GPU/memory)
    with mlflow_manager.start_run(run_name=run_name, tags=tags, log_system_metrics=True) as run:
        fallback_run = None
        import mlflow
        if mlflow.active_run() is None:
            logger.error("MLflow run missing after start_run. Starting fallback run for logging.")
            fallback_run = mlflow_manager.start_run(
                run_name=f"{run_name}_fallback",
                tags={**tags, "run_status": "fallback_log"}
            )
            fallback_run.__enter__()
            try:
                mlflow.set_tag("fallback_used", "true")
            except Exception as e:
                logger.warning(f"Failed to set fallback_used tag: {e}")

        try:
            # Log Hyperparams & Config
            mlflow_manager.log_params(hyperparams, prefix="hp_")
            if preprocessing_config:
                mlflow_manager.log_params(preprocessing_config, prefix="preproc_")
            
            # Compute dataset sizes
            train_size = len(train) if not isinstance(train, list) else sum(len(t) for t in train)
            val_size = len(val) if not isinstance(val, list) else sum(len(v) for v in val)
            test_size = len(test) if not isinstance(test, list) else sum(len(t) for t in test)
            
            mlflow_manager.log_params({
                'original_filename': original_filename or 'unknown',
                'dataset_name': dataset_name,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
            })

            # Log column mapping for Forecasting page compatibility
            if column_mapping:
                mlflow_manager.log_params({
                    'columns_target': column_mapping.get('target_var', ''),
                    'columns_covariates': json.dumps(column_mapping.get('covariate_vars', [])),
                })

            # Log input datasets for lineage tracking
            if station_data_df is not None:
                try:
                    if isinstance(station_data_df, dict):
                        # Global model: log all stations
                        mlflow_manager.log_input_dataset(
                            station_data_df,
                            name="input_data",
                            context="training"
                        )
                    elif hasattr(station_data_df, 'head'):
                        mlflow_manager.log_input_dataset(
                            station_data_df,
                            name=f"input_{station_name}",
                            context="training"
                        )
                except Exception as e:
                    logger.warning(f"Failed to log input dataset: {e}")

            # 0. Reproducibilité (graines fixes)
            import torch
            import pytorch_lightning as pl
            try:
                from dashboard.config import RANDOM_SEED
            except Exception:
                RANDOM_SEED = 42
            pl.seed_everything(RANDOM_SEED, workers=True)

            # 1. Configure Trainer (GPU & Callbacks)
            from dashboard.utils.callbacks import create_training_callbacks

            trainer_kwargs = {}
            if pl_trainer_kwargs:
                trainer_kwargs.update(pl_trainer_kwargs)

            # Device detection: XPU > CUDA > CPU
            from dashboard.utils.xpu_support import is_xpu_available, get_xpu_trainer_kwargs, get_xpu_device_name

            if is_xpu_available():
                # Intel XPU (Arc GPU) - use custom accelerator/strategy
                xpu_kwargs = get_xpu_trainer_kwargs()
                trainer_kwargs.update(xpu_kwargs)
                if verbose:
                    device_name = get_xpu_device_name(0)
                    print(f"🚀 Using Intel XPU: {device_name}")
            elif torch.cuda.is_available():
                trainer_kwargs['accelerator'] = 'gpu'
                trainer_kwargs['devices'] = 1
                if verbose:
                    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                trainer_kwargs['accelerator'] = 'cpu'
                if verbose:
                    print(f"⚠️ GPU not available, using CPU")

            # Create callbacks with MLflow enabled
            # IMPORTANT: Monitor val_loss for early stopping (not train_loss!)
            # This prevents overfitting by stopping when validation loss stops improving
            callbacks = create_training_callbacks(
                metrics_file=metrics_file,
                total_epochs=n_epochs or hyperparams.get('n_epochs'),
                early_stopping_patience=early_stopping_patience,
                early_stopping_monitor="val_loss",  # Must be val_loss, not train_loss
                early_stopping_mode="min",
                use_mlflow=True,  # Enable MLflow callback
                enable_lr_monitor=False
            )
            
            if 'callbacks' in trainer_kwargs:
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

            # 3. Prepare covariates
            train_past_cov = None
            val_past_cov = None

            if use_covariates and train_cov is not None:
                supports_past = getattr(model, "supports_past_covariates", False)
                if supports_past:
                    train_past_cov = train_cov
                    val_past_cov = val_cov

            # 4. DEBUG: Verify train/val are different (helps detect data leakage)
            def _debug_series_stats(series, name):
                """Log series statistics for debugging."""
                if isinstance(series, (list, tuple)):
                    for i, s in enumerate(series):
                        vals = s.values().flatten()
                        logger.info(f"  {name}[{i}]: len={len(s)}, mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
                else:
                    vals = series.values().flatten()
                    logger.info(f"  {name}: len={len(series)}, mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

            logger.info("=== DATA VERIFICATION (train vs val) ===")
            _debug_series_stats(train, "train")
            _debug_series_stats(val, "val")

            # Check for low variance (potential issue for loss calculation)
            if not isinstance(train, (list, tuple)):
                train_vals = train.values().flatten()
                val_vals = val.values().flatten()
                if train_vals.std() < 1e-6:
                    logger.warning(f"LOW VARIANCE WARNING: train has std={train_vals.std():.6f} - model may learn constant!")
                if val_vals.std() < 1e-6:
                    logger.warning(f"LOW VARIANCE WARNING: val has std={val_vals.std():.6f} - val_loss may be artificially low!")

                # Check if val is essentially the same as train (hash check)
                train_hash = hash(tuple(train_vals[:100].round(6)))
                val_hash = hash(tuple(val_vals[:min(100, len(val_vals))].round(6)))
                if train_hash == val_hash:
                    logger.error("DATA LEAKAGE DETECTED: train and val have identical values!")

            # Verify train and val don't overlap
            if not isinstance(train, (list, tuple)):
                train_end = train.time_index[-1]
                val_start = val.time_index[0]
                if train_end >= val_start:
                    logger.warning(f"POTENTIAL DATA LEAKAGE: train ends at {train_end}, val starts at {val_start}")
                else:
                    logger.info(f"  Split OK: train ends at {train_end}, val starts at {val_start}")

            logger.info("=========================================")

            # 4. Train
            model = train_model(
                model=model,
                train_series=train,
                val_series=val,
                train_past_covariates=train_past_cov,
                val_past_covariates=val_past_cov,
                verbose=verbose
            )

            results['model'] = model

            # 4b. Log train/val loss curves to MLflow (per-epoch)
            if metrics_file and Path(metrics_file).exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics_payload = json.load(f)
                    epochs = metrics_payload.get("epochs", [])
                    train_losses = metrics_payload.get("train_losses", [])
                    val_losses = metrics_payload.get("val_losses", [])
                    for idx, epoch in enumerate(epochs):
                        if idx < len(train_losses) and train_losses[idx] is not None:
                            mlflow.log_metric("train_loss", float(train_losses[idx]), step=int(epoch))
                        if idx < len(val_losses) and val_losses[idx] is not None:
                            mlflow.log_metric("val_loss", float(val_losses[idx]), step=int(epoch))
                except Exception as e:
                    logger.warning(f"Failed to log loss curves to MLflow: {e}")

            # 5. Evaluate
            output_chunk = hyperparams.get('output_chunk_length', 7)
            
            if isinstance(train, list) and isinstance(val, list):
                full_train = [concatenate([t, v], axis=0) for t, v in zip(train, val)]
            else:
                full_train = concatenate([train, val], axis=0)

            if isinstance(test, (list, tuple)):
                _test_len = min(len(ts) for ts in test)
            else:
                _test_len = len(test)
            _horizon = min(output_chunk, _test_len)

            if _horizon <= 0:
                raise ValueError(
                    f"Invalid horizon ({_horizon}): output_chunk_length={output_chunk}, "
                    f"test_length={_test_len}. Ensure test set is non-empty and output_chunk_length >= 1."
                )

            full_past_cov = None
            if use_covariates and full_cov is not None:
                if getattr(model, "supports_past_covariates", False):
                    full_past_cov = full_cov

            eval_scaler = None
            if target_preprocessor:
                if isinstance(train, list):  # Global
                    if isinstance(target_preprocessor, dict):
                        if all_stations:
                            # Validate that we have a scaler for each station
                            missing_scalers = [s for s in all_stations if s not in target_preprocessor]
                            if missing_scalers:
                                logger.warning(f"Missing scalers for stations: {missing_scalers}. Using first scaler as fallback.")
                            eval_scaler = [target_preprocessor.get(s, list(target_preprocessor.values())[0]) for s in all_stations]
                        else:
                            # No station list provided - use scalers in order they were added
                            logger.warning("No all_stations list provided for global model. Using scalers in dict order.")
                            eval_scaler = list(target_preprocessor.values())
                            # Ensure we have enough scalers
                            if len(eval_scaler) < len(train):
                                logger.error(f"Not enough scalers ({len(eval_scaler)}) for stations ({len(train)})")
                                raise ValueError(f"Scaler count ({len(eval_scaler)}) doesn't match station count ({len(train)})")
                    else:
                        # Single scaler for all - this is valid if all stations share same scale
                        eval_scaler = target_preprocessor
                else:  # Single
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
            
            # Log metrics to MLflow
            mlflow_manager.log_metrics(metrics)

            # 5b. Sliding
            metrics_sliding = {}
            if not isinstance(train, (list, tuple)):
                try:
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
                    # Log sliding metrics
                    sliding_logs = {f"sliding_{k}": v for k,v in metrics_sliding.items()}
                    mlflow_manager.log_metrics(sliding_logs)
                except Exception as e:
                    print(f"Warning: Sliding eval failed: {e}")
            
            results['metrics_sliding'] = metrics_sliding

            # 6. Save Artifacts for MLflow (scalers, config, model)
            # Create a dictionary of artifacts to log
            custom_artifacts = {}
            
            # Save scalers
            if target_preprocessor or cov_preprocessor:
                scalers = {'target': target_preprocessor, 'covariates': cov_preprocessor}
                custom_artifacts['scalers.pkl'] = scalers

            # Save datasets (Train/Val/Test)
            import pandas as pd
            
            # Helper to convert TS/List[TS] to DF
            def _to_df(ts_data, station_names=None):
                if isinstance(ts_data, (list, tuple)):
                    dfs = []
                    for i, ts in enumerate(ts_data):
                        df = ts.to_dataframe()
                        if station_names and i < len(station_names):
                            df['station'] = station_names[i]
                        dfs.append(df)
                    return pd.concat(dfs) if dfs else pd.DataFrame()
                else:
                    return ts_data.to_dataframe()

            # Prepare station list for Global models
            _stations = all_stations if (isinstance(train, list) and all_stations) else None
            
            custom_artifacts['train.csv'] = _to_df(train, _stations)
            custom_artifacts['val.csv'] = _to_df(val, _stations)
            custom_artifacts['test.csv'] = _to_df(test, _stations)

            # Save covariates (processed) when available
            if train_cov is not None:
                custom_artifacts['train_cov.csv'] = _to_df(train_cov, _stations)
            if val_cov is not None:
                custom_artifacts['val_cov.csv'] = _to_df(val_cov, _stations)
            if test_cov is not None:
                custom_artifacts['test_cov.csv'] = _to_df(test_cov, _stations)
            
            if full_cov: # Save covariates if available? Usually redundant if embedded in TS but good for reference
                 pass # Skip for now to save space, assuming reproducible from source + code
            
            # Save config/metadata
            # Include columns info in preprocessing for Forecasting page compatibility
            preprocessing_with_columns = preprocessing_config.copy() if preprocessing_config else {}
            if column_mapping:
                preprocessing_with_columns['columns'] = {
                    'target': column_mapping.get('target_var'),
                    'covariates': column_mapping.get('covariate_vars', [])
                }

            config_dict = {
                'model_name': model_name,
                'station': station_name,
                'hyperparams': hyperparams,
                'metrics': metrics,
                'metrics_sliding': metrics_sliding,
                'original_filename': original_filename,
                'dataset_name': dataset_name,
                'preprocessing': preprocessing_with_columns,
                'columns': {
                    'target': column_mapping.get('target_var') if column_mapping else None,
                    'covariates': column_mapping.get('covariate_vars', []) if column_mapping else []
                },
                'type': 'global' if isinstance(train, list) else 'single'
            }
            custom_artifacts['model_config.json'] = config_dict

            # 7. Compute and save IPS reference stats (for Counterfactual Analysis page)
            # IPS must be computed on RAW physical values (m NGF), not normalized.
            # We use inverse_transform to recover raw training data.
            try:
                from dashboard.utils.counterfactual.ips import (
                    compute_ips_reference,
                    extract_scaler_params,
                    ref_stats_to_json,
                    validate_ips_data,
                )
                # Extract real scaler params
                if target_preprocessor:
                    _scalers_dict = {'target': target_preprocessor, 'covariates': cov_preprocessor}
                    _mu, _sigma, _cov_params = extract_scaler_params(_scalers_dict)
                    if _mu is not None and _sigma is not None:
                        # Denormalize training target to raw values
                        train_target_df = _to_df(train, _stations)
                        target_var_name = column_mapping.get('target_var') if column_mapping else None
                        if target_var_name and target_var_name in train_target_df.columns:
                            gwl_norm = train_target_df[target_var_name]
                        else:
                            gwl_norm = train_target_df.iloc[:, 0]
                        gwl_raw = gwl_norm * _sigma + _mu

                        # Validate and compute IPS reference
                        _validation = validate_ips_data(gwl_raw)
                        if _validation["valid"]:
                            _ref_stats = compute_ips_reference(gwl_raw, aggregate_to_monthly=True)
                            ips_meta = {
                                "ref_stats": {str(k): list(v) for k, v in _ref_stats.items()},
                                "mu_target": _mu,
                                "sigma_target": _sigma,
                                "covariate_params": _cov_params,
                                "n_years": _validation["n_years"],
                                "n_monthly_values": _validation.get("n_monthly_values", 0),
                                "validation": {
                                    "valid": _validation["valid"],
                                    "warnings": _validation["warnings"],
                                },
                            }
                            custom_artifacts['ips_reference.json'] = ips_meta
                            logger.info(f"IPS reference computed and saved ({_validation['n_years']} years)")
                        else:
                            logger.warning(f"IPS reference not computed: {_validation['errors']}")
            except Exception as e:
                logger.warning(f"Could not compute IPS reference stats: {e}")

            # Log Model + Artifacts (with model signature/info)
            # Robustness: ensure there's an active MLflow run before logging
            import mlflow
            if mlflow.active_run() is None:
                logger.error("No active MLflow run before model logging. Starting fallback run.")
                with mlflow_manager.start_run(
                    run_name=f"{run_name}_fallback",
                    tags={**tags, "run_status": "fallback_log"}
                ):
                    mlflow_manager.log_model_with_signature(
                        model,
                        artifact_path="model",
                        custom_artifacts=custom_artifacts
                    )
            else:
                mlflow_manager.log_model_with_signature(
                    model,
                    artifact_path="model",
                    custom_artifacts=custom_artifacts
                )

            # NOTE: We intentionally do NOT register models in MLflow Registry.
            # Runs are the source of truth; promotion to Registry is manual.
            
            # Legacy save logic removed (replaced by MLflow)

        except Exception as e:
            # Log error
            # mlflow.end_run(status='FAILED') # handled by context manager if exception propagates
            raise e
        finally:
            if fallback_run is not None:
                fallback_run.__exit__(None, None, None)

            # Clean up GPU memory after training
            try:
                from dashboard.utils.xpu_support import cleanup_gpu_memory
                # Pass model to move it to CPU before clearing GPU cache
                _model = results.get('model') if results else None
                cleanup_gpu_memory(model=_model)
            except Exception:
                pass  # Don't fail if cleanup fails

    return results


