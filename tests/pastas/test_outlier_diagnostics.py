"""Tests for outlier diagnostics module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from dashboard.utils.pastas.outlier_diagnostics import (
    _detect_outliers,
    _build_climate_context,
    _build_data_quality,
    _build_neighbor_context,
    _classify_outlier,
    _generate_explanation,
    compute_outlier_diagnostics,
    CATEGORY_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def residuals_with_outliers():
    """Monthly residuals with 2 clear outliers."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    values = rng.normal(0, 0.1, 60)
    # Inject outliers at index 10 (Nov 2018) and 30 (Jul 2020)
    values[10] = 0.5
    values[30] = -0.45
    return pd.Series(values, index=dates)


@pytest.fixture
def climate_df():
    """Historical monthly climate data for a station."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "mois": dates,
        "precipitation_totale": 60 + 20 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 10, 120),
        "temperature_moyenne": 12 + 8 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 1, 120),
        "evaporation_moyenne": 2 + 1.5 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 0.3, 120),
    })


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetectOutliers:
    def test_identifies_outliers_above_threshold(self, residuals_with_outliers):
        outliers, sigma = _detect_outliers(residuals_with_outliers)
        assert len(outliers) == 2
        assert sigma > 0

    def test_returns_dates_and_values(self, residuals_with_outliers):
        outliers, sigma = _detect_outliers(residuals_with_outliers)
        for o in outliers:
            assert "date" in o
            assert "residual" in o
            assert "residual_zscore" in o
            assert abs(o["residual"]) > 2 * sigma

    def test_no_outliers_when_all_normal(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        values = np.full(60, 0.05)
        residuals = pd.Series(values, index=dates)
        outliers, _ = _detect_outliers(residuals)
        assert len(outliers) == 0


# ---------------------------------------------------------------------------
# Climate context
# ---------------------------------------------------------------------------

class TestBuildClimateContext:
    def test_computes_zscore_for_calendar_month(self, climate_df):
        # Pick a March date
        target_date = pd.Timestamp("2019-03-01")
        ctx = _build_climate_context(target_date, climate_df, spli_lookup={}, spi_lookup={})
        assert ctx["precip_mm"] is not None
        assert ctx["precip_zscore"] is not None
        assert ctx["temp_c"] is not None
        assert ctx["temp_zscore"] is not None

    def test_includes_spli_spi_when_available(self, climate_df):
        target_date = pd.Timestamp("2019-03-01")
        spli_lookup = {"2019-03-01": {"spli": 1.5, "classification": "TRES_HAUT"}}
        spi_lookup = {"2019-03-01": {"spi": 2.0, "classification": "TRES_HAUT"}}
        ctx = _build_climate_context(target_date, climate_df, spli_lookup=spli_lookup, spi_lookup=spi_lookup)
        assert ctx["spli"] == 1.5
        assert ctx["spli_class"] == "TRES_HAUT"
        assert ctx["spi"] == 2.0

    def test_handles_missing_month_gracefully(self, climate_df):
        target_date = pd.Timestamp("2030-06-01")
        ctx = _build_climate_context(target_date, climate_df, spli_lookup={}, spi_lookup={})
        assert ctx["precip_mm"] is None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyOutlier:
    def test_data_gap_wins_over_climate(self):
        outlier = {"residual": 0.5, "residual_zscore": 3.0}
        # gap_days must clear the DATA_GAP threshold (>=7) so the rule fires; the
        # point of this test is that DATA_GAP outranks CLIMATE_EXTREME.
        data_quality = {"gap_days": 7, "coverage_pct": 80.0, "nearest_gap_distance_days": 3}
        climate = {"precip_zscore": 2.5, "temp_zscore": 0.1, "etp_zscore": -0.5,
                   "precip_mm": 140, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 5, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.3, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "DATA_GAP"
        assert "CLIMATE_EXTREME" in tags

    def test_climate_extreme_detected(self):
        outlier = {"residual": 0.4, "residual_zscore": 2.5}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 2.8, "temp_zscore": 0.5, "etp_zscore": -0.3,
                   "precip_mm": 150, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 3, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.2, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "CLIMATE_EXTREME"

    def test_regional_signal_detected(self):
        outlier = {"residual": 0.3, "residual_zscore": 2.1}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 1.0, "temp_zscore": 0.5, "etp_zscore": -0.3,
                   "precip_mm": 80, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 4, "anomalous": 3, "neighbor_zscores": [
            {"code_bss": "A", "zscore": 2.0}, {"code_bss": "B", "zscore": 1.8},
            {"code_bss": "C", "zscore": 1.6}, {"code_bss": "D", "zscore": 0.3},
        ]}
        contributions = {"recharge": 0.15, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "REGIONAL_SIGNAL"

    def test_dominant_contribution_detected(self):
        outlier = {"residual": 0.35, "residual_zscore": 2.3}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 0.5, "temp_zscore": 0.2, "etp_zscore": -0.1,
                   "precip_mm": 65, "temp_c": 12, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 2, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.95, "evap": 0.01, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "DOMINANT_CONTRIBUTION"

    def test_unknown_when_no_rule_matches(self):
        outlier = {"residual": 0.25, "residual_zscore": 2.05}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 0.3, "temp_zscore": 0.1, "etp_zscore": 0.0,
                   "precip_mm": 60, "temp_c": 12, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 2, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.1, "evap": 0.08, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "UNKNOWN"


class TestGenerateExplanation:
    def test_data_gap_explanation(self):
        explanation = _generate_explanation(
            "DATA_GAP", [],
            climate={"precip_mm": 60, "precip_zscore": 0.3},
            data_quality={"gap_days": 12, "coverage_pct": 80},
            neighbors={"total": 3, "anomalous": 1},
            contributions={"recharge": 0.2},
            residual_zscore=2.5,
        )
        assert "12" in explanation
        assert "gap" in explanation.lower()

    def test_multiple_tags_concatenate(self):
        explanation = _generate_explanation(
            "CLIMATE_EXTREME", ["REGIONAL_SIGNAL"],
            climate={"precip_mm": 140, "precip_zscore": 2.5, "temp_c": 10, "temp_zscore": 0.3,
                     "etp_mm": 2, "etp_zscore": -0.5},
            data_quality={"gap_days": 0, "coverage_pct": 100},
            neighbors={"total": 4, "anomalous": 3},
            contributions={"recharge": 0.3},
            residual_zscore=2.8,
        )
        assert "precipitation" in explanation.lower() or "precip" in explanation.lower()
        assert "neighbor" in explanation.lower()


# ---------------------------------------------------------------------------
# Seasonal bias (pass 2)
# ---------------------------------------------------------------------------

class TestSeasonalBiasPass2:
    def test_seasonal_bias_applied_to_unclassified(self):
        """3+ outliers of the same sign in one quarter → SEASONAL_BIAS on UNKNOWN ones."""
        from dashboard.utils.pastas.outlier_diagnostics import _apply_seasonal_bias

        outliers = [
            {"date": pd.Timestamp("2019-01-01"), "residual": 0.3, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-02-01"), "residual": 0.25, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-03-01"), "residual": 0.28, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2020-07-01"), "residual": -0.4, "category": "CLIMATE_EXTREME", "secondary_tags": []},
        ]
        _apply_seasonal_bias(outliers)

        # First three should become SEASONAL_BIAS (they were UNKNOWN)
        assert outliers[0]["category"] == "SEASONAL_BIAS"
        assert outliers[1]["category"] == "SEASONAL_BIAS"
        assert outliers[2]["category"] == "SEASONAL_BIAS"
        # Fourth keeps its category but gets a tag if applicable
        assert outliers[3]["category"] == "CLIMATE_EXTREME"

    def test_no_seasonal_bias_with_fewer_than_3(self):
        from dashboard.utils.pastas.outlier_diagnostics import _apply_seasonal_bias

        outliers = [
            {"date": pd.Timestamp("2019-01-01"), "residual": 0.3, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-02-01"), "residual": 0.25, "category": "UNKNOWN", "secondary_tags": []},
        ]
        _apply_seasonal_bias(outliers)
        assert outliers[0]["category"] == "UNKNOWN"
        assert outliers[1]["category"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Full pipeline (mocked DB)
# ---------------------------------------------------------------------------

class TestComputeOutlierDiagnosticsMocked:
    def test_returns_correct_structure(self):
        """Test with mocked model and DB engine."""
        import pastas as ps

        # Create a minimal mock model
        model = MagicMock(spec=ps.Model)
        dates = pd.date_range("2018-01-01", periods=60, freq="MS")
        rng = np.random.default_rng(42)
        residual_values = rng.normal(0, 0.1, 60)
        residual_values[10] = 0.5  # outlier
        model.residuals.return_value = pd.Series(residual_values, index=dates)
        model.simulate.return_value = pd.Series(10 + residual_values * 0.1, index=dates)
        model.observations.return_value = pd.Series(10 + residual_values * 0.1 + residual_values, index=dates)
        model.stressmodels = {"recharge": MagicMock()}

        contrib_series = pd.Series(np.full(60, 0.5), index=dates)
        model.get_contribution.return_value = contrib_series

        # Mock engine
        engine = MagicMock(spec=Engine)

        with patch("dashboard.utils.pastas.outlier_diagnostics._fetch_climate_data") as mock_climate, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_daily_data") as mock_daily, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_sibling_codes") as mock_siblings, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_neighbor_monthly") as mock_neighbor, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_drought_indices") as mock_drought:

            mock_climate.return_value = pd.DataFrame({
                "mois": dates,
                "precipitation_totale": rng.normal(60, 10, 60),
                "temperature_moyenne": rng.normal(12, 2, 60),
                "evaporation_moyenne": rng.normal(2, 0.5, 60),
            })
            mock_daily.return_value = pd.DataFrame({
                "date": pd.date_range("2017-12-01", periods=365 * 6, freq="D"),
                "niveau_nappe_eau": rng.normal(10, 0.5, 365 * 6),
            })
            mock_siblings.return_value = []
            mock_neighbor.return_value = pd.DataFrame()
            mock_drought.return_value = ({}, {})

            result = compute_outlier_diagnostics(
                model=model,
                code_bss="TEST/001",
                cal_tmin="2018-01-01",
                cal_tmax="2022-12-01",
                engine=engine,
            )

        assert "sigma" in result
        assert "threshold" in result
        assert "n_outliers" in result
        assert "outliers" in result
        assert "summary" in result
        assert isinstance(result["outliers"], list)
        if result["n_outliers"] > 0:
            o = result["outliers"][0]
            assert "category" in o
            assert "climate" in o
            assert "data_quality" in o
            assert "neighbors" in o
            assert "contributions" in o
            assert "explanation" in o
            assert "severity" in o
