"""Tests for STOWA 4-criteria TFN model quality assessment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from dashboard.utils.pastas.stowa import (
    StowaResult,
    _runs_test_pvalue,
    _compute_t95,
    assess_stowa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    evp_value: float = 80.0,
    noise_series: pd.Series | None = None,
    residuals_series: pd.Series | None = None,
    t95_value: float = 100.0,
    gain_optimal: float = 1.0,
    gain_stderr: float = 0.3,
) -> MagicMock:
    """Build a mock Pastas model with the given characteristics."""
    n = 365
    dates = pd.date_range("2015-01-01", periods=n, freq="D")

    if noise_series is None:
        # White noise → runs test should pass (p > 0.05)
        rng = np.random.default_rng(0)
        noise_series = pd.Series(rng.normal(0, 1, n), index=dates)

    if residuals_series is None:
        rng = np.random.default_rng(1)
        residuals_series = pd.Series(rng.normal(0, 1, n), index=dates)

    # Build step response: rises from 0 to final_value over t95_value days
    step_index = pd.RangeIndex(0, 600)
    final = 1.0
    step_values = np.where(step_index < t95_value, step_index / t95_value * final, final)
    step_series = pd.Series(step_values, index=step_index)

    # Parameter mock
    param = MagicMock()
    param.optimal = gain_optimal
    param.pstderr = gain_stderr

    params_df = pd.DataFrame(
        {"optimal": [gain_optimal], "pstderr": [gain_stderr]},
        index=["recharge_A"],
    )

    model = MagicMock()
    model.stats.evp.return_value = evp_value
    model.noise.return_value = noise_series
    model.residuals.return_value = residuals_series
    model.get_step_response.return_value = step_series
    model.parameters = params_df

    return model


# ---------------------------------------------------------------------------
# _runs_test_pvalue
# ---------------------------------------------------------------------------

class TestRunsTestPvalue:
    def test_white_noise_has_high_pvalue(self):
        """White noise should not systematically reject randomness.

        We run the test across 10 different seeds and verify that the vast
        majority (>= 8/10) produce p > 0.05, consistent with a Type I error
        rate of 5% under the null hypothesis.
        """
        rejections = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            s = pd.Series(rng.normal(0, 1, 1000))
            p = _runs_test_pvalue(s)
            assert 0.0 <= p <= 1.0
            if p <= 0.05:
                rejections += 1
        # At alpha=0.05, we expect ~0-1 false rejections out of 10
        assert rejections <= 2, (
            f"Runs test rejected too many white-noise series ({rejections}/10). "
            "Expected at most 2 rejections by chance."
        )

    def test_alternating_series_has_low_pvalue(self):
        """A perfectly alternating series (many runs) is non-random → low p-value."""
        s = pd.Series([1.0, -1.0] * 100)
        p = _runs_test_pvalue(s)
        assert p < 0.05, f"Alternating series should have p < 0.05, got {p:.4f}"

    def test_constant_series_returns_float(self):
        """A constant series (1 run) should return a float without crashing."""
        s = pd.Series([1.0] * 50)
        p = _runs_test_pvalue(s)
        assert isinstance(p, float)

    def test_returns_float_in_range(self):
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0, 1, 100))
        p = _runs_test_pvalue(s)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# _compute_t95
# ---------------------------------------------------------------------------

class TestComputeT95:
    def test_step_reaches_95pct_at_correct_index(self):
        """t95 should match the first index where |step| >= 0.95 * |final|."""
        # Step rises linearly from 0 to 1 over 200 steps
        index = pd.RangeIndex(0, 300)
        values = np.minimum(np.arange(300) / 200.0, 1.0)
        step = pd.Series(values, index=index)

        model = MagicMock()
        model.get_step_response.return_value = step

        t95 = _compute_t95(model, "recharge")
        # 0.95 reached at step index 190 (190/200 = 0.95)
        assert abs(t95 - 190.0) <= 2.0, f"Expected t95 ≈ 190, got {t95}"

    def test_already_at_95_from_start(self):
        """If step immediately reaches 95% of final, t95 should be the first index."""
        index = pd.RangeIndex(0, 10)
        values = np.ones(10) * 1.0
        step = pd.Series(values, index=index)

        model = MagicMock()
        model.get_step_response.return_value = step

        t95 = _compute_t95(model, "recharge")
        assert t95 == pytest.approx(0.0, abs=1.0)

    def test_negative_final_value(self):
        """Works correctly when the step response final value is negative."""
        index = pd.RangeIndex(0, 200)
        values = -np.minimum(np.arange(200) / 150.0, 1.0)
        step = pd.Series(values, index=index)

        model = MagicMock()
        model.get_step_response.return_value = step

        t95 = _compute_t95(model, "recharge")
        # Reaches -0.95 at index 142 (142/150 = 0.947, 143/150 = 0.953)
        assert 140.0 <= t95 <= 150.0, f"Expected t95 between 140 and 150, got {t95}"


# ---------------------------------------------------------------------------
# assess_stowa — all passing
# ---------------------------------------------------------------------------

class TestAssessStowaAllPass:
    def test_all_criteria_pass(self):
        """With good EVP, random noise, small t95, and significant gain → all pass."""
        model = _make_model(
            evp_value=85.0,
            t95_value=100.0,   # << 50% of 730 days (365 days threshold)
            gain_optimal=2.0,
            gain_stderr=0.5,   # significance = 4.0 > 1.96
        )
        result = assess_stowa(
            model=model,
            tmin="2015-01-01",
            tmax="2016-12-31",
            cal_period_days=730,
        )
        assert isinstance(result, StowaResult)
        assert result.evp_pass, f"EVP should pass (value={result.evp_value})"
        assert result.autocorrelation_pass, "Autocorrelation should pass for white noise"
        assert result.t95_pass, f"t95 should pass (t95={result.t95_days}, threshold={result.t95_threshold})"
        assert result.gain_pass, f"Gain should pass (significance={result.gain_significance})"
        assert result.overall_pass

    def test_result_fields_populated(self):
        """All numeric fields are populated after a successful assessment."""
        model = _make_model(evp_value=75.0, t95_value=50.0, gain_optimal=1.0, gain_stderr=0.2)
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)

        assert isinstance(result.evp_value, float)
        assert isinstance(result.runs_test_pvalue, float)
        assert isinstance(result.t95_days, float)
        assert isinstance(result.t95_threshold, float)
        assert isinstance(result.gain_significance, float)
        assert isinstance(result.suggestions, list)


# ---------------------------------------------------------------------------
# assess_stowa — individual criterion failures
# ---------------------------------------------------------------------------

class TestAssessStowaEvpFail:
    def test_evp_below_threshold_fails(self):
        """EVP below 70% → evp_pass = False, overall_pass = False."""
        model = _make_model(
            evp_value=50.0,
            t95_value=50.0,
            gain_optimal=2.0,
            gain_stderr=0.3,
        )
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)

        assert not result.evp_pass
        assert not result.overall_pass
        assert result.evp_value == pytest.approx(50.0)
        assert any("EVP" in s or "evp" in s.lower() for s in result.suggestions)

    def test_evp_custom_threshold(self):
        """EVP = 65% fails with default threshold 70%, passes with threshold 60%."""
        model = _make_model(evp_value=65.0, t95_value=50.0, gain_optimal=2.0, gain_stderr=0.3)

        result_fail = assess_stowa(
            model=model, tmin=None, tmax=None, cal_period_days=365, evp_threshold=70.0
        )
        assert not result_fail.evp_pass

        result_pass = assess_stowa(
            model=model, tmin=None, tmax=None, cal_period_days=365, evp_threshold=60.0
        )
        assert result_pass.evp_pass


class TestAssessStowaT95Fail:
    def test_t95_exceeds_half_cal_period_fails(self):
        """t95 > 50% of cal_period_days → t95_pass = False."""
        # cal_period_days = 200, threshold = 100 days; t95_value = 150 > 100
        model = _make_model(
            evp_value=80.0,
            t95_value=150.0,
            gain_optimal=2.0,
            gain_stderr=0.3,
        )
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=200)

        assert not result.t95_pass
        assert not result.overall_pass
        assert result.t95_threshold == pytest.approx(100.0)
        assert any("t95" in s.lower() or "step" in s.lower() for s in result.suggestions)

    def test_t95_exactly_at_threshold_passes(self):
        """t95 just under threshold should pass."""
        model = _make_model(evp_value=80.0, t95_value=99.0, gain_optimal=2.0, gain_stderr=0.3)
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=200)
        assert result.t95_pass


class TestAssessStowaGainFail:
    def test_gain_insignificant_fails(self):
        """Gain |optimal/stderr| < 1.96 → gain_pass = False."""
        model = _make_model(
            evp_value=80.0,
            t95_value=50.0,
            gain_optimal=0.1,
            gain_stderr=0.5,  # significance = 0.2 < 1.96
        )
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)

        assert not result.gain_pass
        assert not result.overall_pass
        assert result.gain_significance == pytest.approx(0.2)
        assert any("gain" in s.lower() or "A" in s for s in result.suggestions)

    def test_gain_significance_borderline(self):
        """|optimal/stderr| = 2.0 > 1.96 → gain_pass = True."""
        model = _make_model(
            evp_value=80.0,
            t95_value=50.0,
            gain_optimal=2.0,
            gain_stderr=1.0,  # significance = 2.0
        )
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)
        assert result.gain_pass


class TestAssessStowaAutocorrelationFail:
    def test_autocorrelated_noise_fails(self):
        """Strongly autocorrelated noise → autocorrelation_pass = False."""
        # Create a series with very strong autocorrelation (AR with high coefficient)
        n = 500
        rng = np.random.default_rng(99)
        x = np.zeros(n)
        # AR(1) with coefficient 0.99 → highly autocorrelated, few runs
        for i in range(1, n):
            x[i] = 0.99 * x[i - 1] + rng.normal(0, 0.01)
        dates = pd.date_range("2015-01-01", periods=n, freq="D")
        noise = pd.Series(x, index=dates)

        model = _make_model(
            evp_value=80.0,
            noise_series=noise,
            t95_value=50.0,
            gain_optimal=2.0,
            gain_stderr=0.3,
        )
        result = assess_stowa(
            model=model, tmin=None, tmax=None, cal_period_days=365, runs_alpha=0.05
        )
        assert not result.autocorrelation_pass
        assert any("noise" in s.lower() or "autocorr" in s.lower() or "residual" in s.lower()
                   for s in result.suggestions)

    def test_noise_fallback_to_residuals(self):
        """If model.noise() raises, falls back to model.residuals()."""
        rng = np.random.default_rng(42)
        n = 300
        dates = pd.date_range("2015-01-01", periods=n, freq="D")
        residuals = pd.Series(rng.normal(0, 1, n), index=dates)

        model = _make_model(evp_value=80.0, t95_value=50.0, gain_optimal=2.0, gain_stderr=0.3)
        model.noise.side_effect = Exception("no noise model")
        model.residuals.return_value = residuals

        # Should not raise
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)
        assert isinstance(result, StowaResult)
        assert isinstance(result.runs_test_pvalue, float)


# ---------------------------------------------------------------------------
# overall_pass logic
# ---------------------------------------------------------------------------

class TestOverallPass:
    def test_overall_false_when_any_criterion_fails(self):
        """overall_pass is False when any single criterion fails."""
        # Only EVP fails
        model = _make_model(evp_value=50.0, t95_value=50.0, gain_optimal=2.0, gain_stderr=0.3)
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=365)
        assert result.evp_pass is False
        assert result.overall_pass is False

    def test_overall_true_requires_all_four(self):
        """overall_pass is True only when all 4 criteria pass."""
        model = _make_model(evp_value=90.0, t95_value=50.0, gain_optimal=5.0, gain_stderr=0.5)
        result = assess_stowa(model=model, tmin=None, tmax=None, cal_period_days=730)
        # White noise model should pass autocorrelation
        # overall_pass = evp AND autocorr AND t95 AND gain
        assert result.overall_pass == (
            result.evp_pass and result.autocorrelation_pass
            and result.t95_pass and result.gain_pass
        )
