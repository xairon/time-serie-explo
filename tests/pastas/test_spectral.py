"""Tests for spectral analysis."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from dashboard.utils.pastas.spectral import compute_spectral_analysis


def test_returns_frequencies_and_psd():
    model = MagicMock()
    dates = pd.date_range("2015-01-01", periods=365*3, freq="D")
    rng = np.random.default_rng(42)
    signal = 10 + np.sin(2 * np.pi * np.arange(365*3) / 365) + rng.normal(0, 0.1, 365*3)
    obs = pd.Series(signal, index=dates)
    sim = pd.Series(signal + rng.normal(0, 0.05, 365*3), index=dates)
    model.observations.return_value = obs
    model.simulate.return_value = sim

    result = compute_spectral_analysis(model, "2015-01-01", "2017-12-31")
    assert len(result["frequencies"]) > 0
    assert len(result["psd_observed"]) == len(result["frequencies"])
    assert len(result["psd_simulated"]) == len(result["frequencies"])
    assert all(p > 0 for p in result["psd_observed"])


def test_short_series_returns_empty():
    model = MagicMock()
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    model.observations.return_value = pd.Series(np.ones(100), index=dates)
    model.simulate.return_value = pd.Series(np.ones(100), index=dates)

    result = compute_spectral_analysis(model, "2020-01-01", "2020-04-10")
    assert result["frequencies"] == []
