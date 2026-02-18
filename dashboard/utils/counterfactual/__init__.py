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
    compute_ips_reference,
    gwl_to_ips_class,
    ips_class_to_gwl_bounds,
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
    "compute_ips_reference",
    "gwl_to_ips_class",
    "ips_class_to_gwl_bounds",
    "validity_ratio",
    "proximity_theta",
    "cc_compliance",
    "param_count",
]
