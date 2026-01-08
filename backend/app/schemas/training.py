"""
Pydantic schemas for training.
"""

from datetime import datetime
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    # Model selection
    model_name: str = Field(..., description="Model architecture (TFT, NHiTS, etc.)")
    
    # Data splits
    train_ratio: float = Field(0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(0.15, ge=0.05, le=0.3)
    
    # Sequence parameters
    input_chunk_length: int = Field(30, ge=1, le=365)
    output_chunk_length: int = Field(7, ge=1, le=90)
    
    # Training parameters
    epochs: int = Field(100, ge=1, le=500)
    batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1)
    
    # Advanced Config
    early_stopping_patience: int = Field(5, ge=0, le=50, description="Patience for early stopping (0 to disable)")
    optimize_hyperparameters: bool = Field(False, description="Enable Optuna optimization")
    
    # Model-specific hyperparameters
    model_hyperparams: Optional[dict[str, Any]] = None


class TrainingRequest(BaseModel):
    """Request to start a training job."""
    
    dataset_id: str
    stations: list[str] = Field(..., min_length=1)
    training_strategy: Literal["independent", "global"] = "independent"
    config: TrainingConfig
    
    use_covariates: bool = True
    save_model: bool = True


class TrainingJobStatus(BaseModel):
    """Status of a training job."""
    
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    
    # Metrics during training
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    model_id: Optional[str] = None
    error_message: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """Response when starting a training job."""
    
    job_id: str
    status: str
    message: str


class TrainingProgressUpdate(BaseModel):
    """WebSocket message for training progress."""
    
    job_id: str
    event_type: Literal["progress", "metric", "log", "completed", "error"]
    
    # Progress info
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress: Optional[float] = None
    
    # Metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # Log message
    message: Optional[str] = None
    
    # Completion info
    model_id: Optional[str] = None
    error: Optional[str] = None
