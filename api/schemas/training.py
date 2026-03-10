"""Schemas for the training API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    """Request to start a training run."""

    model_name: str
    dataset_id: str
    station_name: str = "default"
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    use_covariates: bool = True
    early_stopping: bool = True
    early_stopping_patience: Optional[int] = 10
    n_epochs: Optional[int] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    loss_function: str = "MAE"


class TrainingStatus(BaseModel):
    """Status of a training task."""

    task_id: str
    status: str
    task_type: str = "training"
    config: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: float = 0.0


class TrainingResult(BaseModel):
    """Result of a completed training task."""

    task_id: str
    status: str
    metrics: Optional[dict[str, Any]] = None
    metrics_sliding: Optional[dict[str, Any]] = None
    model_name: Optional[str] = None
    station: Optional[str] = None
    error: Optional[str] = None
