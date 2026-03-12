import numpy as np
import pandas as pd
import pytest


def _make_residuals_clean_and_dirty():
    """Residuals: clean Jan-Jun, dirty Jul-Dec (high values + autocorrelation)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=730, freq="D")
    values = rng.normal(0, 0.2, 730)
    for start in [181, 546]:
        end = min(start + 184, 730)
        values[start:end] = np.cumsum(rng.normal(-0.05, 0.3, end - start))
    return pd.Series(values, index=dates, name="residuals")


class TestCleanPeriodSelector:
    def test_select_returns_boolean_mask(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        assert "mask" in result
        assert len(result["mask"]) == len(residuals)
        assert result["mask"].dtype == bool

    def test_clean_periods_mostly_in_first_half(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        mask = result["mask"]
        first_half_clean = mask[:365].sum()
        second_half_clean = mask[365:].sum()
        assert first_half_clean > second_half_clean

    def test_stats_include_total_clean_days(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        assert "n_clean_days" in result
        assert "pct_clean" in result
        assert result["n_clean_days"] > 0
        assert 0 < result["pct_clean"] <= 100
