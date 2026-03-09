"""Schemas for the forecasting API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request for a single-window forecast."""

    model_id: str
    start_date: str
    use_covariates: bool = True
    freq: str = "D"


class RollingForecastRequest(BaseModel):
    """Request for rolling (historical) forecasts."""

    model_id: str
    start_date: str
    forecast_horizon: int
    stride: int = 1
    use_covariates: bool = True
    freq: str = "D"


class ComparisonForecastRequest(BaseModel):
    """Request for comparison forecast (autoregressive vs teacher forcing)."""

    model_id: str
    start_date: str
    forecast_horizon: int
    use_covariates: bool = True
    freq: str = "D"


class GlobalForecastRequest(BaseModel):
    """Request for a global forecast over the full test set."""

    model_id: str
    use_covariates: bool = True
    freq: str = "D"


class ForecastResult(BaseModel):
    """Result of a forecast operation."""

    predictions: list[dict[str, Any]] = Field(default_factory=list)
    target: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    horizon: Optional[int] = None
    # Additional fields for comparison
    predictions_onestep: Optional[list[dict[str, Any]]] = None
    metrics_onestep: Optional[dict[str, float]] = None
    predictions_exact: Optional[list[dict[str, Any]]] = None
    metrics_exact: Optional[dict[str, float]] = None
    # Rolling forecasts
    forecast_windows: Optional[list[list[dict[str, Any]]]] = None
