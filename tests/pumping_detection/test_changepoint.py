import numpy as np
import pandas as pd
import pytest


def _make_residuals_with_changepoints():
    """Residuals with known mean shift at day 200 and 400."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=600, freq="D")
    values = rng.normal(0, 0.3, 600)
    values[200:400] -= 1.0
    return pd.Series(values, index=dates, name="residuals")


class TestChangepointDetector:
    def test_pelt_detects_known_changepoints(self):
        from dashboard.utils.pumping_detection.changepoint import ChangepointDetector

        residuals = _make_residuals_with_changepoints()
        detector = ChangepointDetector(method="pelt", min_segment_length=60)
        result = detector.detect(residuals)

        assert "changepoints" in result
        assert len(result["changepoints"]) >= 2
        cp_indices = [cp["index"] for cp in result["changepoints"]]
        assert any(abs(cp - 200) < 30 for cp in cp_indices)
        assert any(abs(cp - 400) < 30 for cp in cp_indices)

    def test_returns_empty_on_flat_signal(self):
        from dashboard.utils.pumping_detection.changepoint import ChangepointDetector

        rng = np.random.default_rng(42)
        dates = pd.date_range("2015-01-01", periods=600, freq="D")
        flat = pd.Series(rng.normal(0, 0.3, 600), index=dates)
        detector = ChangepointDetector(method="pelt", min_segment_length=60)
        result = detector.detect(flat)

        assert len(result["changepoints"]) <= 1
