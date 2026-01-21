"""Module for generating forecasts using Darts models."""

import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import mae, rmse, mape, smape
from darts import concatenate
from dashboard.utils.preprocessing import prepare_dataframe_for_darts


def generate_single_window_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    use_covariates: bool = True,
    already_processed: bool = False,
    is_global_model: bool = False
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, Dict[str, float], Dict[str, float], int]:
    """
    Generates a forecast for a single window starting from a given date.

    Returns both the autoregressive prediction AND the one-step prediction, 
    along with metrics for BOTH.

    Args:
        model: Trained Darts model.
        full_df: Complete DataFrame (train + val + test).
        target_col: Name of the target column.
        covariate_cols: List of covariate columns.
        preprocessing_config: Preprocessing configuration.
        scalers: Dictionary of scalers.
        start_date: Start date for the prediction.
        use_covariates: If True, uses covariates.
        already_processed: If True, data is already normalized (skip scaling).

    Returns:
        A tuple containing:
            - pred_autoregressive: The autoregressive prediction (RAW scale for display).
            - pred_onestep: The one-step prediction (RAW scale for display).
            - target_series: The actual target series (PROCESSED for metrics).
            - metrics_auto: Metrics for autoregressive (computed on PROCESSED).
            - metrics_onestep: Metrics for one-step (computed on PROCESSED).
            - horizon: The forecast horizon used.
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # Get model horizon
    horizon = getattr(model, 'output_chunk_length', 30)
    
    # Prepare full series
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Get scalers for inverse transform (output to RAW for display)
    target_preprocessor = scalers.get('target_preprocessor')
    
    # Scaling logic - skip if already processed
    if already_processed:
        # Data is already normalized - use directly
        full_series_for_model = full_series
        covariates_for_model = covariates
    else:
        # Data is RAW - need to scale
        cov_preprocessor = scalers.get('cov_preprocessor')
        full_series_for_model = full_series
        covariates_for_model = covariates
        
        if target_preprocessor:
            full_series_for_model = target_preprocessor.transform(full_series)
        if cov_preprocessor and covariates is not None:
            covariates_for_model = cov_preprocessor.transform(covariates)
        
    # Cut history just before start_date
    history_cutoff = start_date - pd.Timedelta(days=1)
    history_series = full_series_for_model.split_after(history_cutoff)[0]
    
    # =========================================================================
    # 1. AUTOREGRESSIVE PREDICTION (model predicts on its own predictions)
    # =========================================================================
    # For global models, wrap single station data in lists
    if is_global_model:
        series_for_pred = [history_series]
        cov_for_pred = [covariates_for_model] if covariates_for_model else None
    else:
        series_for_pred = history_series
        cov_for_pred = covariates_for_model

    predict_kwargs = {
        'n': horizon,
        'series': series_for_pred
    }

    if cov_for_pred is not None:
        # Only use past_covariates to avoid prediction bias
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = cov_for_pred

    try:
        pred_auto_processed = model.predict(**predict_kwargs)
        # Handle global model results (list with one element)
        if is_global_model and isinstance(pred_auto_processed, list):
            pred_auto_processed = pred_auto_processed[0]
    except Exception as e:
        raise ValueError(f"Autoregressive prediction failed: {e}")
    
    # =========================================================================
    # 2. ONE-STEP PREDICTION (each prediction uses real past values)
    # =========================================================================
    end_date = start_date + pd.Timedelta(days=horizon - 1)

    try:
        # Use historical_forecasts with forecast_horizon=1 for one-step
        # For global models, wrap data in lists
        if is_global_model:
            series_for_onestep = [full_series_for_model]
            cov_for_onestep = [covariates_for_model] if covariates_for_model else None
        else:
            series_for_onestep = full_series_for_model
            cov_for_onestep = covariates_for_model

        onestep_kwargs = {
            'series': series_for_onestep,
            'start': start_date,
            'forecast_horizon': 1,
            'stride': 1,
            'retrain': False,
            'verbose': False
        }

        if cov_for_onestep is not None:
            # Only use past_covariates to avoid prediction bias
            if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
                onestep_kwargs['past_covariates'] = cov_for_onestep

        onestep_forecasts = model.historical_forecasts(**onestep_kwargs)

        # Handle global model results
        if is_global_model and isinstance(onestep_forecasts, list) and len(onestep_forecasts) > 0:
            # For global models, we get a list with one element
            if isinstance(onestep_forecasts[0], list):
                pred_onestep_processed = concatenate(onestep_forecasts[0][:horizon])
            else:
                pred_onestep_processed = onestep_forecasts[0][:horizon]
        elif isinstance(onestep_forecasts, list):
            pred_onestep_processed = concatenate(onestep_forecasts[:horizon])
        else:
            pred_onestep_processed = onestep_forecasts[:horizon]

    except Exception as e:
        print(f"One-step prediction failed: {e}")
        pred_onestep_processed = pred_auto_processed
    
    # Inverse scaling - convert predictions to RAW for display
    if target_preprocessor:
        pred_auto_raw = target_preprocessor.inverse_transform(pred_auto_processed)
        pred_onestep_raw = target_preprocessor.inverse_transform(pred_onestep_processed)
    else:
        pred_auto_raw = pred_auto_processed
        pred_onestep_raw = pred_onestep_processed
        
    # Extract corresponding real slice (PROCESSED for metrics)
    target_series_processed = full_series_for_model.slice(start_date, end_date)

    # Get RAW target for display (inverse transform if scaler exists)
    if target_preprocessor:
        target_series_raw = target_preprocessor.inverse_transform(target_series_processed)
    else:
        target_series_raw = target_series_processed

    # Align lengths
    min_len = min(len(pred_auto_processed), len(target_series_processed))
    pred_auto_processed_aligned = pred_auto_processed[:min_len]
    pred_onestep_processed_aligned = pred_onestep_processed[:min_len] if len(pred_onestep_processed) >= min_len else pred_onestep_processed
    target_processed_aligned = target_series_processed[:min_len]

    # Also align RAW outputs
    pred_auto_raw = pred_auto_raw[:min_len]
    pred_onestep_raw = pred_onestep_raw[:min_len] if len(pred_onestep_raw) >= min_len else pred_onestep_raw
    target_series_raw = target_series_raw[:min_len]

    # Calculate metrics on PROCESSED data (same scale as model training)
    def compute_metrics(target, pred):
        return {
            'MAE': float(mae(target, pred)),
            'RMSE': float(rmse(target, pred)),
            'MAPE': float(mape(target, pred)),
            'sMAPE': float(smape(target, pred))
        }

    metrics_auto = compute_metrics(target_processed_aligned, pred_auto_processed_aligned)
    metrics_onestep = compute_metrics(target_processed_aligned, pred_onestep_processed_aligned)

    # Return RAW predictions and target for display, metrics computed on PROCESSED
    return pred_auto_raw, pred_onestep_raw, target_series_raw, metrics_auto, metrics_onestep, horizon


def generate_global_forecast(
    model: ForecastingModel,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    use_covariates: bool = True
) -> Tuple[TimeSeries, TimeSeries, Dict[str, float]]:
    """
    Generates a global forecast on the entire test set.
    
    Args:
        model: Trained Darts model.
        history_df: History DataFrame (train + val).
        target_df: Target DataFrame (test).
        target_col: Name of the target column.
        covariate_cols: List of covariate columns.
        preprocessing_config: Preprocessing configuration (for fill_method).
        scalers: Dictionary of scalers (target_preprocessor, cov_preprocessor).
        use_covariates: If True, uses covariates.
        
    Returns:
        A tuple containing:
            - pred_series: Predicted series.
            - target_series: Original target series (for metrics).
            - metrics: Dictionary of evaluation metrics.
    """
    # 1. Get fill method from config
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # 2. Prepare Darts series
    # History (Context)
    history_series, _ = prepare_dataframe_for_darts(
        history_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Target (Ground Truth for metrics)
    target_series, _ = prepare_dataframe_for_darts(
        target_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # Covariates for prediction (must cover the future)
    covariates_future = None
    if use_covariates and covariate_cols:
        # We need covariates on history + target
        full_cov_df = pd.concat([history_df, target_df])
        _, covariates_future = prepare_dataframe_for_darts(
            full_cov_df,
            target_col=target_col,
            covariate_cols=covariate_cols,
            freq='D',
            fill_method=fill_method
        )
    
    # 3. Apply scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    history_series_scaled = history_series
    target_series_scaled = target_series
    covariates_scaled = covariates_future
    
    if target_preprocessor:
        history_series_scaled = target_preprocessor.transform(history_series)
        target_series_scaled = target_preprocessor.transform(target_series)
        
    if cov_preprocessor and covariates_future is not None:
        covariates_scaled = cov_preprocessor.transform(covariates_future)
        
    # 4. Predict
    n_pred = len(target_series)
    predict_kwargs = {
        'n': n_pred,
        'series': history_series_scaled
    }
    
    if covariates_scaled is not None:
        # Only use past_covariates to avoid prediction bias
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates_scaled

    pred_series_scaled = model.predict(**predict_kwargs)
    
    # 5. Inverse scaling
    if target_preprocessor:
        pred_series = target_preprocessor.inverse_transform(pred_series_scaled)
    else:
        pred_series = pred_series_scaled
        
    # 6. Calculate metrics
    # Align lengths
    min_len = min(len(pred_series), len(target_series))
    pred_series = pred_series[:min_len]
    target_series = target_series[:min_len]
    
    metrics = {
        'MAE': float(mae(target_series, pred_series)),
        'RMSE': float(rmse(target_series, pred_series)),
        'MAPE': float(mape(target_series, pred_series)),
        'sMAPE': float(smape(target_series, pred_series))
    }
    
    return pred_series, target_series, metrics


def generate_rolling_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    forecast_horizon: int,
    stride: int,
    use_covariates: bool = True
) -> Tuple[List[TimeSeries], TimeSeries]:
    """
    Generates rolling forecasts (historical forecasts).
    
    Args:
        model: Trained Darts model.
        full_df: Complete DataFrame (train + val + test).
        target_col: Name of the target column.
        covariate_cols: List of covariate columns.
        preprocessing_config: Preprocessing configuration.
        scalers: Dictionary of scalers.
        start_date: Start date for the forecasts (usually start of test set).
        forecast_horizon: Forecast horizon at each step.
        stride: Window shift step.
        use_covariates: If True, uses covariates.
        
    Returns:
        Tuple (list of forecast windows, full target series).
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # Prepare full series
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
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
        
    # Historical Forecasts
    forecast_kwargs = {
        'series': full_series_scaled,
        'start': start_date,
        'forecast_horizon': forecast_horizon,
        'stride': stride,
        'retrain': False,
        'last_points_only': False, # We want the full trajectory
        'verbose': False
    }
    
    if covariates_scaled is not None:
        # Only use past_covariates to avoid prediction bias
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            forecast_kwargs['past_covariates'] = covariates_scaled
            
    forecasts_scaled = model.historical_forecasts(**forecast_kwargs)
    
    # Inverse scaling for each window
    forecasts = []
    if isinstance(forecasts_scaled, TimeSeries):
        forecasts_scaled = [forecasts_scaled]
        
    for ts in forecasts_scaled:
        if target_preprocessor:
            forecasts.append(target_preprocessor.inverse_transform(ts))
        else:
            forecasts.append(ts)
            
    return forecasts, full_series


def generate_comparison_forecast(
    model: ForecastingModel,
    full_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]],
    preprocessing_config: Dict[str, Any],
    scalers: Dict[str, Any],
    start_date: pd.Timestamp,
    forecast_horizon: int,
    use_covariates: bool = True
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, Dict[str, float], Dict[str, float]]:
    """
    Generates a comparison between autoregressive forecast and exact window (teacher forcing).
    
    Args:
        model: Trained Darts model.
        full_df: Complete DataFrame.
        target_col: Name of the target column.
        covariate_cols: List of covariate columns.
        preprocessing_config: Preprocessing configuration.
        scalers: Dictionary of scalers.
        start_date: Start date of the comparison window.
        forecast_horizon: Duration of the comparison window.
        use_covariates: If True, uses covariates.
        
    Returns:
        Tuple (target_slice, autoregressive_forecast, exact_window_forecast, metrics_auto, metrics_exact).
    """
    fill_method = preprocessing_config.get('fill_method', 'Interpolation linéaire')
    
    # 1. Prepare data
    full_series, covariates = prepare_dataframe_for_darts(
        full_df,
        target_col=target_col,
        covariate_cols=covariate_cols if use_covariates else None,
        freq='D',
        fill_method=fill_method
    )
    
    # 2. Scaling
    target_preprocessor = scalers.get('target_preprocessor')
    cov_preprocessor = scalers.get('cov_preprocessor')
    
    full_series_scaled = full_series
    covariates_scaled = covariates
    
    if target_preprocessor:
        full_series_scaled = target_preprocessor.transform(full_series)
        
    if cov_preprocessor and covariates is not None:
        covariates_scaled = cov_preprocessor.transform(covariates)
        
    # 3. Autoregressive Forecast (Drift)
    # Cut history just before start_date
    history_cutoff = start_date - pd.Timedelta(days=1)
    history_series_scaled = full_series_scaled.split_after(history_cutoff)[0]
    
    predict_kwargs = {
        'n': forecast_horizon,
        'series': history_series_scaled
    }
    
    if covariates_scaled is not None:
        # Only use past_covariates to avoid prediction bias
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            predict_kwargs['past_covariates'] = covariates_scaled
            
    autoregressive_scaled = model.predict(**predict_kwargs)
    
    # 4. Exact Forecast (Historical Forecast / Teacher Forcing)
    # We ask for a forecast starting exactly at start_date
    # historical_forecasts uses real past values for each point if horizon=1
    
    hist_kwargs = {
        'series': full_series_scaled,
        'start': start_date,
        'forecast_horizon': 1, # Horizon 1 = exact step-by-step prediction
        'stride': 1,
        'retrain': False,
        'last_points_only': True,
        'verbose': False
    }
    
    if covariates_scaled is not None:
        # Only use past_covariates to avoid prediction bias
        if getattr(model, "_uses_past_covariates", False) or getattr(model, "uses_past_covariates", False):
            hist_kwargs['past_covariates'] = covariates_scaled
            
    # Note: historical_forecasts will generate until the end of the series if not limited
    exact_window_scaled = model.historical_forecasts(**hist_kwargs)
    
    # Slice to keep only the requested window
    end_date = start_date + pd.Timedelta(days=forecast_horizon - 1)
    exact_window_scaled = exact_window_scaled.slice(start_date, end_date)
    
    # 5. Inverse Scaling
    if target_preprocessor:
        autoregressive = target_preprocessor.inverse_transform(autoregressive_scaled)
        exact_window = target_preprocessor.inverse_transform(exact_window_scaled)
        target_slice = target_preprocessor.inverse_transform(full_series_scaled.slice(start_date, end_date))
    else:
        autoregressive = autoregressive_scaled
        exact_window = exact_window_scaled
        target_slice = full_series_scaled.slice(start_date, end_date)
        
    # 6. Metrics
    def calc_metrics(true, pred):
        return {
            'MAE': float(mae(true, pred)),
            'RMSE': float(rmse(true, pred)),
            'MAPE': float(mape(true, pred)),

        }
        
    metrics_auto = calc_metrics(target_slice, autoregressive)
    metrics_exact = calc_metrics(target_slice, exact_window)
    
    return target_slice, autoregressive, exact_window, metrics_auto, metrics_exact
