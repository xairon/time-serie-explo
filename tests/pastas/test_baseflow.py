"""Tests for baseflow separation."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from dashboard.utils.pastas.baseflow import compute_baseflow


def test_bfi_in_valid_range():
    model = MagicMock()
    dates = pd.date_range("2018-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    obs = pd.Series(10 + 0.5 * np.sin(2 * np.pi * np.arange(365) / 365) + rng.normal(0, 0.05, 365), index=dates)
    model.observations.return_value = obs

    result = compute_baseflow(model, "2018-01-01", "2018-12-31")
    assert result["bfi"] is not None
    assert 0 <= result["bfi"] <= 1
    assert len(result["baseflow"]) == 365
    assert len(result["quickflow"]) == 365


def test_baseflow_plus_quickflow_equals_observed():
    model = MagicMock()
    dates = pd.date_range("2018-01-01", periods=365, freq="D")
    obs = pd.Series(10 + 0.3 * np.sin(2 * np.pi * np.arange(365) / 365), index=dates)
    model.observations.return_value = obs

    result = compute_baseflow(model, "2018-01-01", "2018-12-31")
    total = np.array(result["baseflow"]) + np.array(result["quickflow"])
    np.testing.assert_allclose(total, result["observed"], atol=1e-10)
