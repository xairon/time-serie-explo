"""Schemas package."""

from app.schemas.source import (
    DatabaseConnectionRequest,
    DatabaseConnectionResponse,
    TableInfo,
    ColumnInfo,
    TableSchemaResponse,
    DataQueryRequest,
    DataQueryRequest,
    DataPreviewResponse,
    StationSummaryRequest,
    StationSummaryResponse,
)
from app.schemas.dataset import (
    PreprocessingConfig,
    DatasetCreate,
    DatasetInfo,
    DatasetListResponse,
    DatasetStatistics,
)
from app.schemas.training import (
    TrainingConfig,
    TrainingRequest,
    TrainingJobStatus,
    TrainingJobResponse,
    TrainingProgressUpdate,
)
from app.schemas.model import (
    ModelInfo,
    ModelListResponse,
    ModelMetrics,
)

__all__ = [
    # Source schemas
    "DatabaseConnectionRequest",
    "DatabaseConnectionResponse",
    "TableInfo",
    "ColumnInfo",
    "TableSchemaResponse",
    "DataQueryRequest",
    "DataPreviewResponse",
    "StationSummaryRequest",
    "StationSummaryResponse",
    # Dataset schemas
    "PreprocessingConfig",
    "DatasetCreate",
    "DatasetInfo",
    "DatasetListResponse",
    "DatasetStatistics",
    # Training schemas
    "TrainingConfig",
    "TrainingRequest",
    "TrainingJobStatus",
    "TrainingJobResponse",
    "TrainingProgressUpdate",
    # Model schemas
    "ModelInfo",
    "ModelListResponse",
    "ModelMetrics",
]
