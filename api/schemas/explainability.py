"""Schemas for the explainability API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    """Request for an explainability computation."""

    model_id: str
    method: str = "correlation"  # correlation, permutation, saliency, integrated_gradients, deeplift
    target_step: int = 0
    n_samples: int = 100
    n_permutations: int = 5
    n_steps: int = 50  # For integrated gradients


class ExplainResult(BaseModel):
    """Result of an explainability computation."""

    method: str
    success: bool = True
    error_message: Optional[str] = None
    feature_importance: Optional[dict[str, float | None]] = None
    temporal_importance: Optional[list[float]] = None
    gradient_attributions: Optional[list[list[float]]] = None
    attention_weights: Optional[list[list[float]]] = None
    encoder_importance: Optional[dict[str, float]] = None
    decoder_importance: Optional[dict[str, float]] = None
    shap_values: Optional[list[list[float]]] = None
    feature_names: list[str] = Field(default_factory=list)
    model_type: Optional[str] = None


class LagImportanceResult(BaseModel):
    """Result of lag importance (autocorrelation) analysis."""

    lags: list[int]
    autocorrelations: list[float]
    partial_autocorrelations: Optional[list[float]] = None
    significant_lags: Optional[list[int]] = None


class ResidualAnalysisResult(BaseModel):
    """Result of residual analysis."""

    mean_error: float
    std_error: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    normality_pvalue: Optional[float] = None
    acf_lag1: Optional[float] = None
    bias_status: str = "unknown"
    residuals: Optional[list[float]] = None
    dates: Optional[list[str]] = None


class SeasonalityResult(BaseModel):
    """Result of seasonality detection."""

    detected_periods: list[int] = Field(default_factory=list)
    period_strengths: Optional[dict[str, float]] = None
    decomposition: Optional[dict[str, list[float]]] = None
    decomposition_dates: Optional[list[str]] = None
    variance_trend: Optional[float] = None
    variance_seasonal: Optional[float] = None
    variance_residual: Optional[float] = None
