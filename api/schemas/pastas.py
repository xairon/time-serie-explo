"""Pydantic schemas for Pastas API."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------- Comparison ----------

class CompareRequest(BaseModel):
    run_ids: list[str]

class CompareModelResult(BaseModel):
    run_id: str
    name: str
    code_bss: str
    params: dict[str, Any]
    metrics: dict[str, float]
    observed: "TimeSeriesData"
    simulated: "TimeSeriesData"

class CompareResponse(BaseModel):
    models: list[CompareModelResult]


# ---------- Fit ----------

class RechargeConfig(BaseModel):
    type: Literal["Linear", "FlexModel"] = "Linear"
    kwargs: dict[str, Any] = {}

class ResponseConfig(BaseModel):
    type: Literal["Gamma", "Exponential", "Hantush"] = "Gamma"
    kwargs: dict[str, Any] = {}

class NoiseConfig(BaseModel):
    type: Literal["ArNoiseModel", "none"] = "ArNoiseModel"

class SolverConfig(BaseModel):
    type: Literal["LeastSquares", "Lmfit"] = "LeastSquares"
    kwargs: dict[str, Any] = {}

class AdditionalStressRow(BaseModel):
    date: date
    value: float

class AdditionalStress(BaseModel):
    type: Literal["well", "river", "custom"]
    name: str
    rfunc: str = "Exponential"
    csv_rows: list[AdditionalStressRow]

class FitRequest(BaseModel):
    code_bss: str
    tmin: Optional[date] = None
    tmax: Optional[date] = None
    recharge: RechargeConfig = RechargeConfig()
    response: ResponseConfig = ResponseConfig()
    noise: NoiseConfig = NoiseConfig()
    solver: SolverConfig = SolverConfig()
    name: Optional[str] = None
    val_split: Optional[float] = None  # fraction for validation (e.g. 0.3 = last 30%)
    additional_stresses: list[AdditionalStress] = []

class TimeSeriesData(BaseModel):
    index: list[str]
    values: list[float]

class FitParameter(BaseModel):
    name: str
    optimal: float
    stderr: Optional[float]
    initial: float
    pmin: Optional[float]
    pmax: Optional[float]
    vary: bool

class FitResponse(BaseModel):
    run_id: str
    code_bss: str = ""
    metrics: dict[str, float]
    parameters: list[FitParameter]
    observed: TimeSeriesData
    simulated: TimeSeriesData
    residuals: TimeSeriesData
    contributions: dict[str, TimeSeriesData]
    step_response: TimeSeriesData
    block_response: TimeSeriesData
    acf: dict[str, Any]
    warnings: list[str] = []
    pastas_version: str
    validation_metrics: Optional[dict[str, float]] = None
    cal_period: Optional[list[str]] = None   # [tmin_cal, tmax_cal]
    val_period: Optional[list[str]] = None   # [tmin_val, tmax_val]

class PastasModelSummary(BaseModel):
    run_id: str
    name: str
    code_bss: str
    recharge_type: str
    response_type: str
    evp: Optional[float] = None
    rmse: Optional[float] = None
    created_at: str
    pastas_version: str

# ---------- Scenario ----------

class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"] = "pumping_synthetic"
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float = Field(ge=0)
    start: date
    end: date
    distance_m: float = Field(gt=0)
    screen_depth_m: Optional[float] = None
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"
    period_days: int = 365

class PumpingRow(BaseModel):
    date: date
    Q_m3d: float

class PumpingUpload(BaseModel):
    type: Literal["pumping_upload"] = "pumping_upload"
    csv_rows: list[PumpingRow]
    distance_m: float = Field(gt=0)
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"

class LinearTrendMod(BaseModel):
    type: Literal["linear_trend"] = "linear_trend"
    start: date
    end: date
    slope_m_per_year: float

class ScaleStressMod(BaseModel):
    type: Literal["scale_stress"] = "scale_stress"
    stress: Literal["precip", "evap"]
    factor: float = Field(gt=0)
    start: date
    end: date

Modification = Annotated[
    PumpingSynthetic | PumpingUpload | LinearTrendMod | ScaleStressMod,
    Field(discriminator="type"),
]

class ScenarioRequest(BaseModel):
    run_id: str
    tmin: date
    tmax: date
    modifications: list[Modification]

class ScenarioResponse(BaseModel):
    baseline: TimeSeriesData
    scenario: TimeSeriesData
    delta: TimeSeriesData
    contributions_baseline: dict[str, TimeSeriesData]
    contributions_scenario: dict[str, TimeSeriesData]
    warnings: list[str] = []

# ---------- Preview ----------

class StationPreview(BaseModel):
    code_bss: str
    metadata: dict[str, Any]
    piezo: TimeSeriesData
    precip: TimeSeriesData
    evap: TimeSeriesData
    stats: dict[str, Any]
