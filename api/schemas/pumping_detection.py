"""Pydantic schemas for pumping detection API."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class PastasConfig(BaseModel):
    response_function: str = "Gamma"
    noise_model: bool = True


class ChangepointConfig(BaseModel):
    method: str = "pelt"  # pelt, beast, both
    min_segment_length: int = 90


class MLConfig(BaseModel):
    model_type: str = "TFTModel"
    input_chunk_length: int = 365
    output_chunk_length: int = 30
    max_epochs: int = 100
    clean_residual_threshold: Union[float, Literal["auto"]] = "auto"


class XAIConfig(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["integrated_gradients"])
    window_size: int = 90
    stride: int = 30


class EmbeddingsConfig(BaseModel):
    encoder: str = "softclt"
    window_size: int = 365
    n_twins: int = 5


class FusionConfig(BaseModel):
    js_divergence_threshold: float = 0.3
    spearman_threshold: float = 0.5
    embedding_drift_threshold: float = 2.0
    acf_significance: float = 0.05
    min_layers_for_high: str = "all"
    merge_gap_days: int = 30


class PumpingDetectionConfig(BaseModel):
    pastas: PastasConfig = Field(default_factory=PastasConfig)
    changepoint: ChangepointConfig = Field(default_factory=ChangepointConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    xai: XAIConfig = Field(default_factory=XAIConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class PumpingDetectionRequest(BaseModel):
    dataset_id: str
    config: PumpingDetectionConfig = Field(default_factory=PumpingDetectionConfig)


class SuspectWindow(BaseModel):
    start: str
    end: str
    confidence: str
    duration_months: int
    layers: list[str]
    max_concordance: float


class PumpingDetectionResult(BaseModel):
    global_score: float
    suspect_windows: list[SuspectWindow]
    pastas: dict[str, Any] = Field(default_factory=dict)
    ml_xai: dict[str, Any] = Field(default_factory=dict)
    embeddings: dict[str, Any] = Field(default_factory=dict)
    clean_periods: dict[str, Any] = Field(default_factory=dict)
    changepoints: dict[str, Any] = Field(default_factory=dict)
