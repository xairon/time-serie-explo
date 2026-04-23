"""Tests for recession analysis."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from dashboard.utils.pastas.recession import compute_recession_analysis


def test_detects_recession_segments():
    model = MagicMock()
    dates = pd.date_range("2015-01-01", periods=365*3, freq="D")
    # Create a series with clear recession: rise then 60-day exponential decay
    values = np.full(365*3, 10.0)
    # Inject a recession from day 100 to 200
    for i in range(100, 200):
        values[i] = 12.0 * np.exp(-(i - 100) / 40.0)
    # Inject another from day 500 to 600
    for i in range(500, 600):
        values[i] = 11.5 * np.exp(-(i - 500) / 50.0)

    model.observations.return_value = pd.Series(values, index=dates)

    result = compute_recession_analysis(model, "2015-01-01", "2017-12-31", min_duration_days=20)
    assert result["n_segments"] >= 1
    assert all("T_days" in s for s in result["segments"])


def test_empty_series():
    model = MagicMock()
    model.observations.return_value = pd.Series(dtype=float)
    result = compute_recession_analysis(model, "2020-01-01", "2020-12-31")
    assert result["n_segments"] == 0
    assert result["segments"] == []
