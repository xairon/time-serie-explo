"""Pydantic schemas for outlier diagnostics."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ClimateContext(BaseModel):
    precip_mm: Optional[float] = None
    precip_zscore: Optional[float] = None
    temp_c: Optional[float] = None
    temp_zscore: Optional[float] = None
    etp_mm: Optional[float] = None
    etp_zscore: Optional[float] = None
    spli: Optional[float] = None
    spli_class: Optional[str] = None
    spi: Optional[float] = None
    spi_class: Optional[str] = None


class DataQuality(BaseModel):
    gap_days: int
    coverage_pct: float
    nearest_gap_distance_days: Optional[int] = None


class NeighborZscore(BaseModel):
    code_bss: str
    zscore: float


class NeighborContext(BaseModel):
    total: int
    anomalous: int
    neighbor_zscores: list[NeighborZscore]


class OutlierDiagnostic(BaseModel):
    date: str
    residual: float
    residual_zscore: float
    severity: float
    category: str
    category_label: str
    secondary_tags: list[str]
    explanation: str
    climate: ClimateContext
    contributions: dict[str, float]
    observed: float
    simulated: float
    data_quality: DataQuality
    neighbors: NeighborContext


class OutlierSummary(BaseModel):
    by_category: dict[str, int]
    seasonal_pattern: dict[str, int]
    median_severity: float


class OutlierDiagnosticsResponse(BaseModel):
    run_id: str
    code_bss: str
    sigma: float
    threshold: float
    n_residuals: int
    n_outliers: int
    outliers: list[OutlierDiagnostic]
    summary: OutlierSummary
