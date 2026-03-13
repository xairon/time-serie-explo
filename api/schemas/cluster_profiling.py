# api/schemas/cluster_profiling.py
"""Pydantic models for cluster profiling responses."""
from __future__ import annotations

from pydantic import BaseModel


class MetadataDistribution(BaseModel):
    key: str
    clusters: dict[str, dict[str, int]]  # cluster_id (str) → {value: count}


class ConcordanceMetric(BaseModel):
    key: str
    ari: float
    nmi: float
    cramers_v: float


class ClusterPrototype(BaseModel):
    cluster_id: int
    medoid_id: str
    dates: list[str]
    medoid_values: list[float]
    p10: list[float]
    p90: list[float]


class FeatureFingerprint(BaseModel):
    cluster_id: int
    features: dict[str, float]      # normalized [0,1]
    features_raw: dict[str, float]  # original values


class ShapExplanation(BaseModel):
    feature_importance: dict[str, float]
    shap_per_cluster: dict[str, dict[str, float]]  # cluster_id as string key
    proxy_accuracy: float
    warning: str | None = None


class ProfilingResponse(BaseModel):
    domain: str
    n_stations: int
    n_clusters: int
    distributions: list[MetadataDistribution]
    concordance: list[ConcordanceMetric]
    prototypes: list[ClusterPrototype]
    fingerprints: list[FeatureFingerprint]
    shap: ShapExplanation
    warnings: list[str] = []
