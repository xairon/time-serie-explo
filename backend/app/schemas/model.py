"""
Pydantic schemas for models.
"""

from datetime import datetime
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a trained model."""
    
    model_id: str
    model_name: str  # Architecture (TFT, NHiTS, etc.)
    model_type: Literal["single", "global"]
    
    # Station info
    stations: list[str]
    primary_station: Optional[str] = None
    
    # Data info
    dataset_id: Optional[str] = None
    data_source: Optional[str] = None
    
    # Configuration
    hyperparams: dict[str, Any]
    preprocessing_config: Optional[dict[str, Any]] = None
    input_chunk_length: int
    output_chunk_length: int
    
    # Metrics
    metrics: Optional[dict[str, float]] = None
    
    # Metadata
    created_at: datetime
    path: str
    
    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """Response with list of models."""
    
    models: list[ModelInfo]
    total: int


class ModelMetrics(BaseModel):
    """Evaluation metrics for a model."""
    
    model_id: str
    station: str
    
    # Regression metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    r2: Optional[float] = None
    
    # Additional info
    evaluation_date: Optional[datetime] = None
    test_period: Optional[tuple[str, str]] = None
