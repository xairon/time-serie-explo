"""Tests for baseflow separation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from dashboard.utils.pastas.baseflow import compute_baseflow, _lyne_hollick_3pass


def _make_mock_model(values, start="2010-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    obs = pd.Series(values, index=index)
    model = MagicMock()
    model.observations.return_value = obs
    return model


class TestLyneHollick3Pass:
    def test_three_pass_differs_from_one_pass(self):
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(200) * 0.1)
        qf_3pass = _lyne_hollick_3pass(signal, alpha=0.925)
        # Single forward pass for comparison
        n = len(signal)
        qf_1pass = np.zeros(n)
        for t in range(1, n):
            qf_1pass[t] = 0.925 * qf_1pass[t-1] + (1.925)/2 * (signal[t] - signal[t-1])
        qf_1pass = np.maximum(qf_1pass, 0)
        assert not np.allclose(qf_3pass, qf_1pass), "3-pass should differ from 1-pass"

    def test_zero_signal_returns_zero_quickflow(self):
        signal = np.zeros(100)
        qf = _lyne_hollick_3pass(signal)
        assert np.allclose(qf, 0)

    def test_quickflow_is_nonnegative(self):
        np.random.seed(123)
        signal = np.cumsum(np.random.randn(300) * 0.5)
        qf = _lyne_hollick_3pass(signal)
        assert np.all(qf >= -1e-10)


class TestComputeBaseflow:
    def test_short_series_returns_none(self):
        model = _make_mock_model(np.zeros(10))
        result = compute_baseflow(model, "2010-01-01", "2010-01-10")
        assert result["bfi"] is None

    def test_constant_level_bfi_is_one(self):
        model = _make_mock_model(np.full(100, 50.0))
        result = compute_baseflow(model, "2010-01-01", "2010-04-10")
        assert result["bfi"] == 1.0

    def test_sine_wave_has_high_bfi(self):
        t = np.arange(365)
        values = 50.0 + 2.0 * np.sin(2 * np.pi * t / 365)
        model = _make_mock_model(values)
        result = compute_baseflow(model, "2010-01-01", "2010-12-31")
        assert result["bfi"] is not None
        assert 0.3 < result["bfi"] < 1.0

    def test_noisy_signal_has_lower_bfi(self):
        np.random.seed(42)
        smooth = 50.0 + np.cumsum(np.random.randn(200) * 0.01)
        noisy = 50.0 + np.cumsum(np.random.randn(200) * 0.5)
        bfi_smooth = compute_baseflow(_make_mock_model(smooth), "2010-01-01", "2010-07-19")["bfi"]
        bfi_noisy = compute_baseflow(_make_mock_model(noisy), "2010-01-01", "2010-07-19")["bfi"]
        assert bfi_smooth > bfi_noisy

    def test_output_arrays_match_length(self):
        values = 50.0 + np.cumsum(np.random.randn(100) * 0.1)
        model = _make_mock_model(values)
        result = compute_baseflow(model, "2010-01-01", "2010-04-10")
        assert len(result["observed"]) == len(result["baseflow"]) == len(result["quickflow"])
