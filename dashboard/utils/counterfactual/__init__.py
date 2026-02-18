"""PhysCF - Physics-Informed Counterfactual Explanations for Groundwater Forecasting.

Integrated into time-serie-explo for use with Darts models (TFT, GRU, etc.).
"""

from .perturbation import PerturbationLayer
from .darts_adapter import DartsModelAdapter
from .physcf_optim import generate_counterfactual
from .optuna_optim import generate_counterfactual_optuna
from .comet_hydro import generate_counterfactual_comet
from .ips import (
    IPS_CLASSES,
    IPS_ORDER,
    IPS_WINDOWS,
    BRGM_MIN_YEARS,
    AQUIFER_IPS_WINDOW,
    AQUIFER_CATEGORY,
    compute_ips_reference,
    compute_ips_reference_n,
    compute_all_ips_references,
    compute_ips_series,
    compute_ips_series_n,
    compute_rolling_monthly_mean,
    validate_ips_data,
    get_aquifer_ips_info,
    daily_to_monthly_mean,
    gwl_to_ips_class,
    gwl_to_ips_zscore,
    ips_class_to_gwl_bounds,
    extract_scaler_params,
    ref_stats_to_json,
    ref_stats_from_json,
)
from .metrics import (
    validity_ratio,
    proximity_theta,
    cc_compliance,
    param_count,
)

__all__ = [
    "PerturbationLayer",
    "DartsModelAdapter",
    "generate_counterfactual",
    "generate_counterfactual_optuna",
    "generate_counterfactual_comet",
    "IPS_CLASSES",
    "IPS_ORDER",
    "IPS_WINDOWS",
    "BRGM_MIN_YEARS",
    "AQUIFER_IPS_WINDOW",
    "AQUIFER_CATEGORY",
    "compute_ips_reference",
    "compute_ips_reference_n",
    "compute_all_ips_references",
    "compute_ips_series",
    "compute_ips_series_n",
    "compute_rolling_monthly_mean",
    "validate_ips_data",
    "get_aquifer_ips_info",
    "daily_to_monthly_mean",
    "gwl_to_ips_class",
    "gwl_to_ips_zscore",
    "ips_class_to_gwl_bounds",
    "extract_scaler_params",
    "ref_stats_to_json",
    "ref_stats_from_json",
    "validity_ratio",
    "proximity_theta",
    "cc_compliance",
    "param_count",
]
