"""Build additional Pastas stress models (wells, river, custom)."""
from __future__ import annotations

import pandas as pd
import pastas as ps

from dashboard.utils.pastas.config import RFUNC_REGISTRY


def build_well_stress(
    q_series: pd.Series,
    name: str,
    rfunc_type: str = "Exponential",
) -> ps.StressModel:
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Exponential)
    return ps.StressModel(q_series, rfunc=rfunc_cls(), name=name, settings="well")


def build_river_stress(
    river_series: pd.Series,
    name: str,
    rfunc_type: str = "Exponential",
) -> ps.StressModel:
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Exponential)
    return ps.StressModel(river_series, rfunc=rfunc_cls(), name=name, settings="waterlevel")


def build_custom_stress(
    series: pd.Series,
    name: str,
    rfunc_type: str = "Gamma",
) -> ps.StressModel:
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Gamma)
    return ps.StressModel(series, rfunc=rfunc_cls(), name=name, settings="prec")
