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
    type: Literal["Linear", "FlexModel", "Berendrecht", "Peterson"] = "Linear"
    kwargs: dict[str, Any] = {}

class ResponseConfig(BaseModel):
    type: Literal["Gamma", "Exponential", "Hantush", "DoubleExponential", "FourParam"] = "Gamma"
    kwargs: dict[str, Any] = {}

class NoiseConfig(BaseModel):
    type: Literal["ArNoiseModel", "ArmaNoiseModel", "none"] = "ArNoiseModel"

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
    val_split: Optional[float] = None
    include_temp: bool = False

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
    noise_type: str = "unknown"
    solver_type: str = "unknown"
    evp: Optional[float] = None
    rmse: Optional[float] = None
    nse: Optional[float] = None
    aic: Optional[float] = None
    val_nse: Optional[float] = None
    val_evp: Optional[float] = None
    has_validation: bool = False
    include_temp: bool = False
    created_at: str
    pastas_version: str

# ---------- Scenario ----------

class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"] = "pumping_synthetic"
    usage: Optional[Literal["aep", "irrigation", "industrial"]] = None
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float = Field(ge=0)
    start: date
    end: date
    distance_m: float = Field(gt=0)
    screen_depth_m: Optional[float] = None
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"
    period_days: int = 365
    season_months: Optional[list[int]] = None
    peak_months: Optional[list[int]] = None
    peak_factor: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    pulse_duration_days: int = Field(default=30, ge=1)

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

# ---------- Scenario Presets ----------

AquiferFamilyType = Literal["alluvial", "sedimentary", "karst", "fractured", "volcanic"]

class PumpingRangeSchema(BaseModel):
    default: float
    typical_min: float
    typical_max: float
    hard_min: float
    hard_max: float

class PumpingProfileSchema(BaseModel):
    rate_m3d: PumpingRangeSchema
    distance_m: PumpingRangeSchema
    pattern: str
    active_months: list[int]
    peak_months: list[int]
    peak_factor: float
    typical_duration_days: int
    rfunc: str

class PresetScenarioSchema(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    modifications: list[dict[str, Any]]

class NonPumpingLimitsSchema(BaseModel):
    scale_stress: PumpingRangeSchema
    linear_trend: PumpingRangeSchema

class ScenarioPresetsResponse(BaseModel):
    aquifer_families: dict[str, str]
    pumping_profiles: dict[str, dict[str, PumpingProfileSchema]]
    non_pumping_limits: NonPumpingLimitsSchema
    presets: list[PresetScenarioSchema]
    detected_family: Optional[str] = None

class ValidateModificationsRequest(BaseModel):
    modifications: list[Modification]
    aquifer_family: Optional[AquiferFamilyType] = None

class ValidateModificationsResponse(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]

class SaveScenarioRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    modifications: list[Modification]
    tmin: Optional[date] = None
    tmax: Optional[date] = None

class SavedScenario(BaseModel):
    name: str
    description: str = ""
    created_at: str = ""
    aquifer_family: Optional[str] = None
    tmin: Optional[str] = None
    tmax: Optional[str] = None
    modifications: list[dict[str, Any]] = []

# --- Pre-fit Diagnostics ---

class DiagnoseRequest(BaseModel):
    code_bss: str

class DiagnosticIndicator(BaseModel):
    value: Any = None
    status: str
    detail: Optional[str] = None

class DiagnosticRecommendation(BaseModel):
    type: str
    message: str
    action: str
    params: dict[str, Any] = {}

class DiagnoseResponse(BaseModel):
    coverage: DiagnosticIndicator
    gaps: DiagnosticIndicator
    trend: DiagnosticIndicator
    breakpoints: DiagnosticIndicator
    seasonality: DiagnosticIndicator
    record_length: DiagnosticIndicator
    recommended_tmin: Optional[str] = None
    recommended_tmax: Optional[str] = None
    recommendations: list[DiagnosticRecommendation] = []

# --- STOWA ---

class StowaResultSchema(BaseModel):
    evp_pass: bool
    evp_value: float
    autocorrelation_pass: bool
    runs_test_pvalue: float
    t95_pass: bool
    t95_days: float
    t95_threshold: float
    gain_pass: bool
    gain_significance: float
    overall_pass: bool
    suggestions: list[str] = []

# --- Auto-fit ---

class AutoFitRequest(BaseModel):
    code_bss: str
    warm_up_years: int = 2
    val_split: float = 0.2
    include_temp: bool = False
    add_trend: Optional[bool] = None

# --- Compare AI ---

class CompareAIRequest(BaseModel):
    pastas_run_id: str
    ai_model_id: str


# ---------- Preview ----------

class StationPreview(BaseModel):
    code_bss: str
    metadata: dict[str, Any]
    piezo: TimeSeriesData
    precip: TimeSeriesData
    evap: TimeSeriesData
    stats: dict[str, Any]
    preset: dict[str, str] = {}
    diagnostics: dict[str, Any] | None = None
