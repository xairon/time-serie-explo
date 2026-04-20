"""Shared fixtures for Pastas tests."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_station():
    """Generate synthetic piezometric, precipitation, and evapotranspiration series.

    Uses known Gamma response + Linear recharge so that a Pastas fit should
    recover EVP > 80%.
    """
    rng = np.random.default_rng(42)
    n = 365 * 5
    dates = pd.date_range("2015-01-01", periods=n, freq="D")

    precip = rng.exponential(3.0, n)
    evap = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365)

    recharge = np.maximum(precip - 0.8 * evap, 0)
    gwl_base = 10.0
    gwl = np.full(n, gwl_base, dtype=float)
    alpha = 0.98
    for i in range(1, n):
        gwl[i] = alpha * gwl[i - 1] + 0.002 * recharge[i]
    gwl += rng.normal(0, 0.02, n)

    return {
        "piezo": pd.Series(gwl, index=dates, name="gwl"),
        "precip": pd.Series(precip, index=dates, name="precip"),
        "evap": pd.Series(evap, index=dates, name="evap"),
        "dates": dates,
    }


@pytest.fixture
def series_hash(synthetic_station):
    """SHA-256 hash of the synthetic station series."""
    parts = []
    for key in ("piezo", "precip", "evap"):
        s = synthetic_station[key]
        parts.append(s.index.astype(str).str.cat())
        parts.append(np.array2string(s.values, separator=","))
    return hashlib.sha256("".join(parts).encode()).hexdigest()
