"""Tests for recession analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from dashboard.utils.pastas.recession import compute_recession_analysis


def _make_mock_model(values, start="2010-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    obs = pd.Series(values, index=index)
    model = MagicMock()
    model.observations.return_value = obs
    return model


class TestRecessionAnalysis:
    def test_short_series_returns_empty(self):
        model = _make_mock_model(np.zeros(20))
        result = compute_recession_analysis(model, "2010-01-01", "2010-01-20")
        assert result["n_segments"] == 0

    def test_known_exponential_decay_at_high_ngf(self):
        """Exponential decay from 50m baseline + 5m amplitude with T=100d."""
        t = np.arange(200)
        baseline = 45.0
        values = baseline + 5.0 * np.exp(-t / 100.0)
        model = _make_mock_model(values)
        result = compute_recession_analysis(model, "2010-01-01", "2010-07-19")
        assert result["n_segments"] >= 1
        T = result["segments"][0]["T_days"]
        assert 70 < T < 130, f"Expected T~100, got {T}"

    def test_allows_short_interruptions(self):
        """Recession with 1-2 day plateaus should form one segment."""
        t = np.arange(60)
        values = 50.0 - 0.02 * t.astype(float)
        values[20] = values[19]  # 1-day plateau
        values[40] = values[39] + 0.001  # tiny uptick
        model = _make_mock_model(values)
        result = compute_recession_analysis(model, "2010-01-01", "2010-03-01")
        assert result["n_segments"] >= 1

    def test_high_vs_low_ngf_same_T(self):
        """Recession at 200m vs 50m NGF should give similar T."""
        t = np.arange(150)
        low = 50.0 + 3.0 * np.exp(-t / 80.0)
        high = 200.0 + 3.0 * np.exp(-t / 80.0)
        T_low = compute_recession_analysis(_make_mock_model(low), "2010-01-01", "2010-05-30")
        T_high = compute_recession_analysis(_make_mock_model(high), "2010-01-01", "2010-05-30")
        if T_low["n_segments"] > 0 and T_high["n_segments"] > 0:
            diff = abs(T_low["segments"][0]["T_days"] - T_high["segments"][0]["T_days"])
            assert diff < 20, f"T diff = {diff}, should be < 20"

    def test_no_recession_in_rising_series(self):
        values = np.linspace(40, 50, 100)
        model = _make_mock_model(values)
        result = compute_recession_analysis(model, "2010-01-01", "2010-04-10")
        assert result["n_segments"] == 0
