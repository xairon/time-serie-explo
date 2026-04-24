"""Tests for scenario presets referential and validation."""
from __future__ import annotations

import pytest

from dashboard.utils.pastas.scenario_presets import (
    AquiferFamily,
    detect_aquifer_family,
)


class TestDetectAquiferFamily:
    def test_alluvial_nature_3(self):
        assert detect_aquifer_family("3", None) == "alluvial"

    def test_alluvial_milieu_5(self):
        assert detect_aquifer_family(None, "5") == "alluvial"

    def test_sedimentary_porous(self):
        assert detect_aquifer_family("5", "1") == "sedimentary"

    def test_sedimentary_double_porosity(self):
        assert detect_aquifer_family("5", "4") == "sedimentary"

    def test_karst_nature_4(self):
        assert detect_aquifer_family("4", None) == "karst"

    def test_karst_milieu_3(self):
        assert detect_aquifer_family(None, "3") == "karst"

    def test_fractured(self):
        assert detect_aquifer_family("5", "2") == "fractured"

    def test_bedrock(self):
        assert detect_aquifer_family("0", None) == "fractured"

    def test_volcanic(self):
        assert detect_aquifer_family("6", None) == "volcanic"

    def test_mountain(self):
        assert detect_aquifer_family("7", None) == "volcanic"

    def test_composite_milieu_8(self):
        assert detect_aquifer_family(None, "8") == "sedimentary"

    def test_none_fallback(self):
        assert detect_aquifer_family(None, None) == "sedimentary"

    def test_unknown_code_fallback(self):
        assert detect_aquifer_family("99", "99") == "sedimentary"


from dashboard.utils.pastas.scenario_presets import (
    get_pumping_profile,
    get_all_profiles,
    SCALE_STRESS_LIMITS,
    LINEAR_TREND_LIMITS,
    PumpingProfile,
    Range,
    validate_modifications,
    ValidationResult,
)

FAMILIES = ["alluvial", "sedimentary", "karst", "fractured", "volcanic"]
USAGES = ["aep", "irrigation", "industrial"]


class TestReferentialIntegrity:
    @pytest.mark.parametrize("usage", USAGES)
    @pytest.mark.parametrize("family", FAMILIES)
    def test_typical_within_hard(self, usage, family):
        p = get_pumping_profile(usage, family)
        assert p.rate_m3d.typical_min >= p.rate_m3d.hard_min
        assert p.rate_m3d.typical_max <= p.rate_m3d.hard_max
        assert p.rate_m3d.default >= p.rate_m3d.typical_min
        assert p.rate_m3d.default <= p.rate_m3d.typical_max
        assert p.distance_m.typical_min >= p.distance_m.hard_min
        assert p.distance_m.typical_max <= p.distance_m.hard_max

    @pytest.mark.parametrize("usage", USAGES)
    @pytest.mark.parametrize("family", FAMILIES)
    def test_profile_has_valid_months(self, usage, family):
        p = get_pumping_profile(usage, family)
        assert all(1 <= m <= 12 for m in p.active_months)
        assert all(m in p.active_months for m in p.peak_months)

    def test_irrigation_seasonal(self):
        for family in FAMILIES:
            p = get_pumping_profile("irrigation", family)
            assert p.pattern == "seasonal"
            assert set(p.active_months).issubset(range(1, 13))

    def test_industrial_constant(self):
        for family in FAMILIES:
            p = get_pumping_profile("industrial", family)
            assert p.pattern == "constant"
            assert p.peak_factor == 1.0

    def test_get_all_profiles_complete(self):
        profiles = get_all_profiles()
        assert set(profiles.keys()) == set(USAGES)
        for usage in USAGES:
            assert set(profiles[usage].keys()) == set(FAMILIES)

    def test_scale_stress_limits_consistent(self):
        assert SCALE_STRESS_LIMITS.hard_min < SCALE_STRESS_LIMITS.typical_min
        assert SCALE_STRESS_LIMITS.typical_max < SCALE_STRESS_LIMITS.hard_max

    def test_linear_trend_limits_consistent(self):
        assert LINEAR_TREND_LIMITS.hard_min < LINEAR_TREND_LIMITS.typical_min
        assert LINEAR_TREND_LIMITS.typical_max < LINEAR_TREND_LIMITS.hard_max


class TestHardValidation:
    def test_reject_rate_above_hard_max(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 99999, "distance_m": 500,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"}]
        result = validate_modifications(mods, "fractured")
        assert not result.valid
        assert any("m³/j" in e for e in result.errors)

    def test_reject_negative_rate(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": -10, "distance_m": 500,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"}]
        result = validate_modifications(mods, "alluvial")
        assert not result.valid

    def test_reject_distance_below_hard_min(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 100, "distance_m": 1,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"}]
        result = validate_modifications(mods, "alluvial")
        assert not result.valid

    def test_reject_factor_out_of_range(self):
        mods = [{"type": "scale_stress", "stress": "precip", "factor": 10.0,
                 "start": "2020-01-01", "end": "2021-01-01"}]
        result = validate_modifications(mods, "alluvial")
        assert not result.valid

    def test_reject_slope_out_of_range(self):
        mods = [{"type": "linear_trend", "slope_m_per_year": 5.0,
                 "start": "2020-01-01", "end": "2021-01-01"}]
        result = validate_modifications(mods, "alluvial")
        assert not result.valid

    def test_reject_end_before_start(self):
        mods = [{"type": "scale_stress", "stress": "precip", "factor": 0.8,
                 "start": "2022-01-01", "end": "2020-01-01"}]
        result = validate_modifications(mods, "alluvial")
        assert not result.valid

    def test_reject_cumulative_pumping_above_2x_hard_max(self):
        mods = [
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 500,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 800,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 1000,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
        ]
        result = validate_modifications(mods, "fractured")
        assert not result.valid
        assert any("cumulé" in e for e in result.errors)

    def test_accept_valid_modifications(self):
        mods = [
            {"type": "pumping_synthetic", "rate_m3d": 100, "distance_m": 500,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
            {"type": "scale_stress", "stress": "precip", "factor": 0.8,
             "start": "2020-01-01", "end": "2021-01-01"},
        ]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert len(result.errors) == 0


class TestSoftWarnings:
    def test_warn_rate_above_typical(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 2000, "distance_m": 500,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant",
                 "usage": "aep"}]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert any("inhabituel" in w for w in result.warnings)

    def test_warn_close_distance(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 100, "distance_m": 30,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"}]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert any("Distance" in w for w in result.warnings)

    def test_warn_severe_precip_reduction(self):
        mods = [{"type": "scale_stress", "stress": "precip", "factor": 0.3,
                 "start": "2020-01-01", "end": "2021-01-01"}]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert any("sévère" in w for w in result.warnings)

    def test_warn_irrigation_off_season(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 100, "distance_m": 500,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "seasonal",
                 "usage": "irrigation", "season_months": [11, 12, 1, 2]}]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert any("végétative" in w for w in result.warnings)

    def test_no_warnings_for_typical_values(self):
        mods = [{"type": "pumping_synthetic", "rate_m3d": 300, "distance_m": 500,
                 "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant",
                 "usage": "aep"}]
        result = validate_modifications(mods, "alluvial")
        assert result.valid
        assert len(result.warnings) == 0
