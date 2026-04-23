"""Tests for STL decomposition."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from dashboard.utils.pastas.signal_decomposition import compute_stl_decomposition


def test_decomposes_seasonal_signal():
    model = MagicMock()
    dates = pd.date_range("2010-01-01", periods=365*5, freq="D")
    signal = 10 + 0.01 * np.arange(365*5) / 365 + np.sin(2 * np.pi * np.arange(365*5) / 365)
    model.observations.return_value = pd.Series(signal, index=dates)

    result = compute_stl_decomposition(model, "2010-01-01", "2014-12-31")
    assert len(result["trend"]) > 0
    assert result["seasonal_strength"] is not None
    assert result["seasonal_strength"] > 0.5


def test_short_series_returns_empty():
    model = MagicMock()
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    model.observations.return_value = pd.Series(np.ones(30), index=dates)
    result = compute_stl_decomposition(model, "2020-01-01", "2020-01-30")
    assert result["index"] == []
