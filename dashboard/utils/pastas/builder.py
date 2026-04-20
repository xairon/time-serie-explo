"""Build a Pastas TFN model from configuration parameters."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import pastas as ps

from dashboard.utils.pastas.config import (
    RECHARGE_REGISTRY,
    RFUNC_REGISTRY,
    NOISE_REGISTRY,
)


class ValidationError(Exception):
    """Raised when input series fail validation."""


def _validate_series(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
) -> None:
    if len(gwl) < 365:
        raise ValidationError(
            f"Piezometric series has {len(gwl)} observations, needs at least 365."
        )

    nan_ratio = gwl.isna().mean()
    if nan_ratio > 0.20:
        raise ValidationError(
            f"Piezometric series has {nan_ratio:.0%} NaN values (max 20%)."
        )

    overlap = gwl.index.intersection(precip.index).intersection(evap.index)
    if len(overlap) < 365:
        raise ValidationError(
            f"Observation and stress series have only {len(overlap)} days of overlap "
            f"(need at least 365)."
        )


def build_model(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    recharge_type: str,
    response_type: str,
    noise_type: str,
    tmin: Optional[str],
    tmax: Optional[str],
) -> tuple[ps.Model, Optional[str], Optional[str]]:
    """Build an unsolved Pastas model.

    Returns:
        (model, tmin, tmax) — model is ready for .solve().
    """
    _validate_series(gwl, precip, evap)

    recharge_cls = RECHARGE_REGISTRY[recharge_type]
    rfunc_cls = RFUNC_REGISTRY[response_type]

    model = ps.Model(gwl, name="gwl")

    rm = ps.RechargeModel(
        precip,
        evap,
        rfunc=rfunc_cls(),
        recharge=recharge_cls(),
        name="recharge",
    )
    model.add_stressmodel(rm)

    if noise_type != "none":
        noise_cls = NOISE_REGISTRY[noise_type]
        model.add_noisemodel(noise_cls())

    return model, tmin, tmax
