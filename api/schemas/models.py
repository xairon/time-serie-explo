"""Schemas for the models API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    """Summary of a trained model."""

    model_id: str
    model_name: str
    model_type: str  # "single" or "global"
    stations: list[str] = Field(default_factory=list)
    primary_station: Optional[str] = None
    created_at: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    data_source: Optional[str] = None


class ModelDetail(ModelSummary):
    """Full model details."""

    run_id: str
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    display_name: str = ""


class AvailableModel(BaseModel):
    """Description of an available model architecture."""

    name: str
    is_torch: bool
    description: str = ""
    category: str = ""
    default_hyperparams: dict[str, Any] = Field(default_factory=dict)
