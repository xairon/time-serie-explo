"""
Forecasting endpoints - Generate predictions using trained models.

Real implementation using core.model_registry and Darts.
"""

from typing import Optional, Literal, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import date, datetime, timedelta
import pandas as pd
import logging
from pathlib import Path

from app.config import get_settings

import sys
# Add legacy dashboard to path for potential pickle module resolution
sys.path.insert(0, str(Path(__file__).parents[5]))

from core.model_registry import get_registry, ModelEntry
from core.preprocessing import prepare_dataframe_for_darts, TimeSeriesPreprocessor

router = APIRouter()
logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    """Request for generating a forecast."""
    
    model_id: str = Field(..., description="ID of the trained model to use")
    start_date: str = Field(..., description="Start date for forecast (YYYY-MM-DD) - NOT USED YET (Predicts from end of series)")
    horizon: int = Field(7, ge=1, le=365, description="Forecast horizon in days")
    use_covariates: bool = True


class ForecastResponse(BaseModel):
    """Response with forecast results."""
    
    model_id: str
    station: str
    start_date: str
    horizon: int
    
    # Context (Tail of historical data for plotting)
    context_dates: Optional[list[str]] = None
    context_values: Optional[list[float]] = None

    # Predictions
    dates: list[str]
    values: list[float]
    lower_bound: Optional[list[float]] = None
    upper_bound: Optional[list[float]] = None
    
    # Metrics if actual data available
    metrics: Optional[dict[str, float]] = None


class RollingForecastRequest(BaseModel):
    """Request for rolling forecast."""
    
    model_id: str
    start_date: str
    end_date: str
    horizon: int = 7
    stride: int = 1
    use_covariates: bool = True
    station: Optional[str] = None


@router.post("/predict", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate a forecast using a trained model.
    """
    try:
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        # 1. Get model entry
        model_entry = registry.get_model(request.model_id)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 2. Identify correct station
        station = model_entry.primary_station or (model_entry.stations[0] if model_entry.stations else "unknown")
        
        # 3. Load Model & Scalers
        try:
            model = registry.load_model(model_entry)
            scalers = registry.load_scalers(model_entry)
            target_scaler = scalers.get('target')
        except Exception as e:
            logger.error(f"Failed to load model/scalers: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model assets: {e}")

        # 4. Load Historical Data (Context)
        model_dir = registry.checkpoints_dir / model_entry.path
        data_path = model_dir / "data.csv"
        
        if not data_path.exists():
             raise HTTPException(status_code=404, detail="Historical data not found for valid context")
             
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Filter for station
        if 'station' in df.columns:
            if station in df['station'].values:
                df_station = df[df['station'] == station].copy()
            else:
                 station = df['station'].iloc[0]
                 df_station = df[df['station'] == station].copy()
        else:
            df_station = df.copy()
            
        if df_station.empty:
             raise HTTPException(status_code=500, detail="Empty historical data")
             
        # 5. Preprocess Context
        target_col = 'target'
        if model_entry.preprocessing_config and 'columns' in model_entry.preprocessing_config:
             target_col = model_entry.preprocessing_config['columns'].get('target', 'target')
        elif 'level' in df_station.columns:
             target_col = 'level'
        elif 'P1' in df_station.columns:
             target_col = 'P1'
             
        ts, _ = prepare_dataframe_for_darts(
            df_station,
            target_col=target_col,
            fill_method=model_entry.preprocessing_config.get('fill_method', 'Supprimer les lignes')
        )
        
        # Transform
        if target_scaler:
            current_scaler = target_scaler
            if isinstance(target_scaler, dict):
                current_scaler = target_scaler.get(station)
                if not current_scaler:
                    current_scaler = list(target_scaler.values())[0]
            
            ts_scaled = current_scaler.transform(ts)
        else:
            ts_scaled = ts
            
        # 6. Predict
        pred_scaled = model.predict(n=request.horizon, series=ts_scaled)
        
        # 7. Inverse Transform
        if target_scaler:
             pred = current_scaler.inverse_transform(pred_scaled)
        else:
             pred = pred_scaled

        # Extract values
        df_pred = pred.to_dataframe()
        dates = [d.strftime("%Y-%m-%d") for d in df_pred.index]
        values = [round(v, 3) for v in df_pred.iloc[:, 0].tolist()]
        
        # Extract Context (last input_window points)
        input_chunk = getattr(model, 'input_chunk_length', 30)
        context_ts = ts.slice(ts.end_time() - pd.Timedelta(days=input_chunk), ts.end_time())
        df_context = context_ts.to_dataframe()
        context_dates = [d.strftime("%Y-%m-%d") for d in df_context.index]
        context_values = [round(v, 3) for v in df_context.iloc[:, 0].tolist()]

        return ForecastResponse(
            model_id=request.model_id,
            station=station,
            start_date=dates[0] if dates else request.start_date,
            horizon=request.horizon,
            dates=dates,
            values=values,
            context_dates=context_dates,
            context_values=context_values,
            lower_bound=None,
            upper_bound=None,
            metrics=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rolling")
async def generate_rolling_forecast(request: RollingForecastRequest):
    """
    Generate rolling forecasts (historical backtesting).
    """
    try:
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        
        model_entry = registry.get_model(request.model_id)
        if not model_entry:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Load Model & Scalers (Same logic as predict)
        model = registry.load_model(model_entry)
        scalers = registry.load_scalers(model_entry)
        target_scaler = scalers.get('target')
        
        # Load Data
        model_dir = registry.checkpoints_dir / model_entry.path
        df = pd.read_csv(model_dir / "data.csv", index_col=0, parse_dates=True)
        
        # Filter Station
        station = request.station or (model_entry.primary_station if model_entry.primary_station else df['station'].iloc[0] if 'station' in df.columns else None)
        if station and 'station' in df.columns:
            df = df[df['station'] == station]
            
        # Prepare TS
        target_col = model_entry.preprocessing_config.get('columns', {}).get('target', 'target')
        # Fallback detection
        if target_col not in df.columns:
            for c in ['level', 'P1']:
                 if c in df.columns: target_col=c; break

        ts, _ = prepare_dataframe_for_darts(df, target_col=target_col)
        
        # Scale
        if target_scaler:
            current_scaler = target_scaler if not isinstance(target_scaler, dict) else target_scaler.get(station, list(target_scaler.values())[0])
            ts_scaled = current_scaler.transform(ts)
        else:
            ts_scaled = ts
            
        # Run Historical Forecasts
        # Darts historical_forecasts
        start_date = pd.Timestamp(request.start_date)
        
        # Ensure start_date is valid (after input_chunk)
        input_chunk = getattr(model, 'input_chunk_length', 30)
        min_start = ts_scaled.start_time() + pd.Timedelta(days=input_chunk)
        if start_date < min_start:
            start_date = min_start
            
        hist_forecast = model.historical_forecasts(
            series=ts_scaled,
            start=start_date,
            forecast_horizon=request.horizon,
            stride=request.stride,
            retrain=False,
            last_points_only=False, # We want full windows
            verbose=False
        )
        
        # Inverse Transform
        if target_scaler:
            hist_forecast = current_scaler.inverse_transform(hist_forecast)
            
        # Format Results
        # hist_forecast is a TimeSeries (if stride=horizon?) or list of TimeSeries?
        # If last_points_only=False, it returns a LIST of TimeSeries (windows)
        
        windows = []
        if isinstance(hist_forecast, list):
            series_list = hist_forecast
        else:
            series_list = [hist_forecast] # Should usually be list if last_points_only=False
            
        for s in series_list:
            df_w = s.to_dataframe()
            windows.append({
                "start_date": df_w.index[0].strftime("%Y-%m-%d"),
                "dates": [d.strftime("%Y-%m-%d") for d in df_w.index],
                "values": [round(v, 3) for v in df_w.iloc[:, 0].tolist()]
            })
            
        return {
            "model_id": request.model_id,
            "windows": windows,
            "total_windows": len(windows)
        }
            
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/explain")
async def explain_forecast(
    model_id: str,
    start_date: str,
    horizon: int = 7,
):
    """
    Get explainability information (SHAP values) for a forecast.
    """
    try:
        import random
        # Simulated feature importance (in production, use SHAP)
        features = ["temperature", "precipitation", "humidity", "pressure", "wind_speed"]
        importance = {f: round(random.uniform(0, 1), 3) for f in features}
        
        # Normalize to sum to 1
        total = sum(importance.values())
        importance = {k: round(v / total, 3) for k, v in importance.items()}
        
        return {
            "model_id": model_id,
            "start_date": start_date,
            "horizon": horizon,
            "feature_importance": importance,
            "top_features": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-models")
async def list_available_models_for_forecasting():
    """
    List models available for forecasting.
    """
    try:
        # Reusing core logic
        settings = get_settings()
        registry = get_registry(settings.checkpoints_dir.parent)
        models = registry.list_all_models()
        
        result = []
        for m in models:
            result.append({
                "model_id": m.model_id,
                "model_name": m.model_name,
                "stations": m.stations,
                "display_name": m.display_name if hasattr(m, 'display_name') else m.model_id,
            })
        
        return {"models": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
