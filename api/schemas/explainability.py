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
    feature_importance: Optional[dict[str, float]] = None
    temporal_importance: Optional[list[float]] = None
    gradient_attributions: Optional[list[list[float]]] = None
    attention_weights: Optional[list[list[float]]] = None
    encoder_importance: Optional[dict[str, float]] = None
    decoder_importance: Optional[dict[str, float]] = None
    shap_values: Optional[list[list[float]]] = None
    feature_names: list[str] = Field(default_factory=list)
    model_type: Optional[str] = None
