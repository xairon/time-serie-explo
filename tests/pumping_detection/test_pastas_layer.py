import numpy as np
import pandas as pd
import pytest


def _make_synthetic_series(n_days=1000, seed=42):
    """Create synthetic piézo + precip + ETP with a known 'pumping' dip."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    natural = 50.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.3, n_days)
    pumping = np.zeros(n_days)
    pumping[400:600] = -1.5
    piezo = pd.Series(natural + pumping, index=dates, name="piezo")
    precip = pd.Series(3.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1, n_days),
                       index=dates, name="precip").clip(lower=0)
    etp = pd.Series(2.0 + 1.5 * np.sin(2 * np.pi * (t - 90) / 365.25) + rng.normal(0, 0.3, n_days),
                    index=dates, name="etp").clip(lower=0)
    return piezo, precip, etp


class TestPastasAnalyzer:
    def test_analyze_returns_expected_keys(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        assert "residuals" in result
        assert "acf_stats" in result
        assert "pastas_fit_quality" in result
        assert isinstance(result["residuals"], pd.Series)
        assert len(result["residuals"]) > 0

    def test_acf_stats_contain_ljung_box(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        acf = result["acf_stats"]
        assert "acf_values" in acf
        assert "pacf_values" in acf
        assert "ljung_box_pvalue" in acf

    def test_fit_quality_contains_evp(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        quality = result["pastas_fit_quality"]
        assert "evp" in quality
        assert 0 <= quality["evp"] <= 100
