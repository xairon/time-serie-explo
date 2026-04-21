from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.utils.pastas.diagnostics_prefit import PreFitDiagnostic, run_prefit_diagnostics


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_series(
    n_days: int = 5000,
    gap_start: int = 1000,
    gap_length: int = 0,
    trend_slope: float = 0.0,
    breakpoint_at: int | None = None,
) -> pd.Series:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = rng.standard_normal(n_days) * 0.3
    trend = trend_slope * np.arange(n_days) / 365.25
    values = 50.0 + seasonal + noise + trend
    if breakpoint_at and breakpoint_at < n_days:
        values[breakpoint_at:] += 3.0
    series = pd.Series(values, index=dates, name="piezo")
    if gap_length > 0:
        mask = (np.arange(n_days) >= gap_start) & (np.arange(n_days) < gap_start + gap_length)
        series[mask] = np.nan
    return series.dropna()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanLongSeries:
    """Clean 7000-day series (~19 years): expect all green, good coverage."""

    def setup_method(self):
        self.series = _make_series(n_days=7000)
        self.result = run_prefit_diagnostics(self.series)

    def test_returns_prefit_diagnostic(self):
        assert isinstance(self.result, PreFitDiagnostic)

    def test_coverage_above_95(self):
        assert self.result.coverage_pct > 95.0

    def test_coverage_status_green(self):
        assert self.result.coverage_status == "green"

    def test_gaps_status_green(self):
        assert self.result.gaps_status == "green"

    def test_no_gap_periods(self):
        assert self.result.gap_periods == []

    def test_record_status_green(self):
        assert self.result.record_status == "green"

    def test_record_years_above_15(self):
        assert self.result.record_years > 15.0

    def test_seasonality_status_green(self):
        # Strong seasonal signal should be green (ACF at lag 12 > 0.3)
        assert self.result.seasonality_status == "green"

    def test_seasonality_strength_reasonable(self):
        assert self.result.seasonality_strength > 0.3

    def test_recommended_period_exists(self):
        # Should always produce a recommended period
        assert self.result.recommended_tmin is not None
        assert self.result.recommended_tmax is not None


class TestSeriesWithLargeGap:
    """Series with a 200-day gap: expect gaps orange/red, gap_periods populated."""

    def setup_method(self):
        self.series = _make_series(n_days=5000, gap_start=1000, gap_length=200)
        self.result = run_prefit_diagnostics(self.series)

    def test_gap_detected(self):
        assert self.result.max_gap_days >= 200

    def test_gap_periods_not_empty(self):
        assert len(self.result.gap_periods) > 0

    def test_gap_period_has_required_keys(self):
        for gp in self.result.gap_periods:
            assert "start" in gp
            assert "end" in gp
            assert "days" in gp

    def test_gap_status_not_green(self):
        assert self.result.gaps_status in ("orange", "red")

    def test_coverage_reduced(self):
        # 200 missing out of 5000 → ~96% but gap is large
        assert self.result.coverage_pct < 100.0


class TestSeriesWithStrongTrend:
    """Series with slope -0.5 m/yr: trend should be detected."""

    def setup_method(self):
        # 5000 days with strong downward trend
        self.series = _make_series(n_days=5000, trend_slope=-0.5)
        self.result = run_prefit_diagnostics(self.series)

    def test_trend_detected(self):
        assert self.result.trend_detected is True

    def test_trend_pvalue_significant(self):
        assert self.result.trend_pvalue is not None
        assert self.result.trend_pvalue < 0.05

    def test_trend_slope_negative(self):
        assert self.result.trend_slope is not None
        assert self.result.trend_slope < 0.0

    def test_trend_status_not_green(self):
        assert self.result.trend_status in ("orange", "red")

    def test_trend_recommendation_present(self):
        types = [r["type"] for r in self.result.recommendations]
        assert "trend" in types


class TestSeriesWithBreakpoint:
    """Series with a +3m step at day 2500: breakpoint should be detected."""

    def setup_method(self):
        self.series = _make_series(n_days=5000, breakpoint_at=2500)
        self.result = run_prefit_diagnostics(self.series)

    def test_breakpoint_detected(self):
        assert self.result.breakpoint_detected is True

    def test_breakpoint_date_not_none(self):
        assert self.result.breakpoint_date is not None

    def test_breakpoint_status_orange(self):
        assert self.result.breakpoint_status == "orange"

    def test_breakpoint_recommendation_present(self):
        types = [r["type"] for r in self.result.recommendations]
        assert "breakpoint" in types


class TestShortSeries:
    """500-day series (~1.4 years): record should be red."""

    def setup_method(self):
        self.series = _make_series(n_days=500)
        self.result = run_prefit_diagnostics(self.series)

    def test_record_status_red(self):
        assert self.result.record_status == "red"

    def test_record_years_below_5(self):
        assert self.result.record_years < 5.0


class TestSeriesWithGapAndTrend:
    """Series with both a gap and a trend: both recommendations present."""

    def setup_method(self):
        self.series = _make_series(
            n_days=5000, gap_start=800, gap_length=200, trend_slope=-0.5
        )
        self.result = run_prefit_diagnostics(self.series)

    def test_trend_detected(self):
        assert self.result.trend_detected is True

    def test_gap_detected(self):
        assert self.result.max_gap_days >= 200

    def test_gap_recommendation_present(self):
        types = [r["type"] for r in self.result.recommendations]
        assert "gap" in types

    def test_trend_recommendation_present(self):
        types = [r["type"] for r in self.result.recommendations]
        assert "trend" in types

    def test_recommendations_have_required_keys(self):
        for rec in self.result.recommendations:
            assert "type" in rec
            assert "message" in rec
            assert "action" in rec
            assert "params" in rec


class TestRecommendedPeriod:
    """Recommended period should always be present and parseable."""

    @pytest.mark.parametrize("n_days,gap_length,trend_slope,bp", [
        (7000, 0, 0.0, None),
        (5000, 200, 0.0, None),
        (5000, 0, -0.5, None),
        (5000, 0, 0.0, 2500),
        (5000, 200, -0.5, None),
    ])
    def test_recommended_period_parseable(self, n_days, gap_length, trend_slope, bp):
        series = _make_series(
            n_days=n_days, gap_length=gap_length, trend_slope=trend_slope, breakpoint_at=bp
        )
        result = run_prefit_diagnostics(series)
        assert result.recommended_tmin is not None
        assert result.recommended_tmax is not None
        # Must be parseable date strings
        pd.Timestamp(result.recommended_tmin)
        pd.Timestamp(result.recommended_tmax)


class TestDataclassFields:
    """Verify all required fields are present and typed correctly."""

    def setup_method(self):
        self.result = run_prefit_diagnostics(_make_series(n_days=3000))

    def test_coverage_pct_is_float(self):
        assert isinstance(self.result.coverage_pct, float)

    def test_coverage_status_valid(self):
        assert self.result.coverage_status in ("green", "orange", "red")

    def test_max_gap_days_is_int(self):
        assert isinstance(self.result.max_gap_days, int)

    def test_gaps_status_valid(self):
        assert self.result.gaps_status in ("green", "orange", "red")

    def test_trend_status_valid(self):
        assert self.result.trend_status in ("green", "orange", "red")

    def test_breakpoint_status_valid(self):
        assert self.result.breakpoint_status in ("green", "orange")

    def test_seasonality_status_valid(self):
        assert self.result.seasonality_status in ("green", "orange", "red")

    def test_record_status_valid(self):
        assert self.result.record_status in ("green", "orange", "red")

    def test_recommendations_is_list(self):
        assert isinstance(self.result.recommendations, list)
