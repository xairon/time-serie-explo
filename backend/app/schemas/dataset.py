"""
Pydantic schemas for datasets.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for a dataset."""
    
    fill_method: str = "interpolate"
    scaler_type: str = "minmax"
    add_datetime_features: bool = True
    lags: Optional[list[int]] = None


class DatasetCreate(BaseModel):
    """Request to create a new dataset."""
    
    name: str = Field(..., description="Dataset name")
    source_type: str = Field(..., description="csv or database")
    source_file: Optional[str] = None
    source_table: Optional[str] = None
    source_schema: str = "public"
    
    date_column: str
    target_column: str
    covariate_columns: Optional[list[str]] = None
    station_column: Optional[str] = None
    
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class DatasetInfo(BaseModel):
    """Dataset metadata."""
    
    id: str
    name: str
    source_type: str
    source_file: Optional[str] = None
    source_table: Optional[str] = None
    
    date_column: str
    target_column: str
    covariate_columns: list[str]
    station_column: Optional[str] = None
    stations: Optional[list[str]] = None
    
    date_range: tuple[str, str]
    row_count: int
    
    preprocessing: PreprocessingConfig
    created_at: datetime
    
    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Response with list of datasets."""
    
    datasets: list[DatasetInfo]
    total: int


class DatasetStatistics(BaseModel):
    """Statistics for a dataset."""
    
    dataset_id: str
    row_count: int
    date_range: tuple[str, str]
    target_stats: dict[str, float]
    missing_values: dict[str, int]
    stations: Optional[list[str]] = None
