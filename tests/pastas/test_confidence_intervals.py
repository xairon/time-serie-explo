"""Tests for confidence intervals."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from dashboard.utils.pastas.confidence_intervals import compute_confidence_bands


def test_returns_correct_structure():
    model = MagicMock()
    dates = pd.date_range("2018-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    sim = pd.Series(10 + 0.5 * np.sin(2 * np.pi * np.arange(365) / 365), index=dates)
    res = pd.Series(rng.normal(0, 0.1, 365), index=dates)
    model.simulate.return_value = sim
    model.residuals.return_value = res

    result = compute_confidence_bands(model, "2018-01-01", "2018-12-31")
    assert "p5" in result
    assert "p95" in result
    assert len(result["p5"]) == 365
    assert all(p5 <= p95 for p5, p95 in zip(result["p5"], result["p95"]))


def test_bands_contain_simulation():
    model = MagicMock()
    dates = pd.date_range("2018-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    sim = pd.Series(np.full(365, 10.0), index=dates)
    res = pd.Series(rng.normal(0, 0.1, 365), index=dates)
    model.simulate.return_value = sim
    model.residuals.return_value = res

    result = compute_confidence_bands(model, "2018-01-01", "2018-12-31")
    for i in range(365):
        assert result["p5"][i] <= result["simulation"][i] <= result["p95"][i]
