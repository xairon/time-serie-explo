# Realistic Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add physically plausible scenario presets, double-layer validation (hard limits + soft warnings), usage-based pumping profiles, and scenario persistence to the Pastas scenario system.

**Architecture:** New `scenario_presets.py` module holds the referential (usage x aquifer family), validation logic, aquifer detection, and preset builders. Backend exposes it via new API endpoints; frontend caches the referential on page mount and uses it for instant validation + contextual presets. Scenarios are saved as MLflow artifacts.

**Tech Stack:** Python dataclasses, FastAPI, Pydantic, MLflow artifacts API, React, TanStack React Query, TypeScript

---

### Task 1: Scenario Presets Referential (Backend)

**Files:**
- Create: `dashboard/utils/pastas/scenario_presets.py`
- Test: `tests/pastas/test_scenario_presets.py`

- [ ] **Step 1: Write failing tests for aquifer family detection**

Create `tests/pastas/test_scenario_presets.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py -v`
Expected: `ModuleNotFoundError: No module named 'dashboard.utils.pastas.scenario_presets'`

- [ ] **Step 3: Implement scenario_presets.py — data structures and detection**

Create `dashboard/utils/pastas/scenario_presets.py`:

```python
"""Referential of realistic pumping profiles per usage type and aquifer family."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AquiferFamily = Literal["alluvial", "sedimentary", "karst", "fractured", "volcanic"]
PumpingUsage = Literal["aep", "irrigation", "industrial"]

AQUIFER_FAMILY_LABELS: dict[AquiferFamily, str] = {
    "alluvial": "Alluvial",
    "sedimentary": "Sédimentaire poreux",
    "karst": "Karstique",
    "fractured": "Socle / fracturé",
    "volcanic": "Volcanique",
}

USAGE_LABELS: dict[PumpingUsage, str] = {
    "aep": "AEP (eau potable)",
    "irrigation": "Irrigation",
    "industrial": "Industriel",
}


def detect_aquifer_family(
    nature_eh: str | None,
    milieu_eh: str | None,
) -> AquiferFamily:
    """Map BDLISA nature_eh / milieu_eh codes to an aquifer family."""
    if nature_eh == "3":
        return "alluvial"
    if nature_eh == "4":
        return "karst"
    if nature_eh == "6":
        return "volcanic"
    if nature_eh == "7":
        return "volcanic"
    if nature_eh == "0":
        return "fractured"
    if nature_eh == "5" and milieu_eh:
        if milieu_eh == "2":
            return "fractured"
        return "sedimentary"
    if milieu_eh == "5":
        return "alluvial"
    if milieu_eh == "3":
        return "karst"
    return "sedimentary"


@dataclass(frozen=True)
class Range:
    """Numeric range with default, typical bounds, and hard limits."""
    default: float
    typical_min: float
    typical_max: float
    hard_max: float
    hard_min: float = 0.0


@dataclass(frozen=True)
class PumpingProfile:
    """Realistic pumping parameters for a usage x aquifer_family combination."""
    rate_m3d: Range
    distance_m: Range
    pattern: str
    active_months: tuple[int, ...] = tuple(range(1, 13))
    peak_months: tuple[int, ...] = ()
    peak_factor: float = 1.0
    typical_duration_days: int = 365
    rfunc: str = "Exponential"


# --- Distance ranges per aquifer family (shared across usages) ---

_DISTANCE: dict[AquiferFamily, Range] = {
    "alluvial":    Range(default=500,  typical_min=200,  typical_max=2000, hard_min=10, hard_max=50000),
    "sedimentary": Range(default=1000, typical_min=300,  typical_max=5000, hard_min=10, hard_max=50000),
    "karst":       Range(default=1000, typical_min=100,  typical_max=5000, hard_min=10, hard_max=50000),
    "fractured":   Range(default=300,  typical_min=100,  typical_max=1000, hard_min=10, hard_max=20000),
    "volcanic":    Range(default=500,  typical_min=150,  typical_max=2000, hard_min=10, hard_max=30000),
}

# --- Rate ranges: PUMPING_RATES[usage][family] ---

_RATES: dict[PumpingUsage, dict[AquiferFamily, Range]] = {
    "aep": {
        "alluvial":    Range(300, 100, 800,  5000),
        "sedimentary": Range(200, 50,  500,  3000),
        "karst":       Range(400, 50,  1000, 8000),
        "fractured":   Range(30,  10,  80,   500),
        "volcanic":    Range(80,  20,  200,  1500),
    },
    "irrigation": {
        "alluvial":    Range(500, 100, 1500, 5000),
        "sedimentary": Range(300, 50,  800,  3000),
        "karst":       Range(600, 100, 2000, 8000),
        "fractured":   Range(20,  5,   50,   300),
        "volcanic":    Range(60,  10,  150,  1000),
    },
    "industrial": {
        "alluvial":    Range(200, 50,  500,  3000),
        "sedimentary": Range(150, 30,  400,  2000),
        "karst":       Range(300, 50,  800,  5000),
        "fractured":   Range(15,  5,   40,   200),
        "volcanic":    Range(50,  10,  120,  800),
    },
}

# --- Non-pumping hard limits ---

SCALE_STRESS_LIMITS = Range(default=1.0, typical_min=0.5, typical_max=2.0, hard_min=0.1, hard_max=5.0)
LINEAR_TREND_LIMITS = Range(default=-0.01, typical_min=-0.1, typical_max=0.1, hard_min=-1.0, hard_max=1.0)


def get_pumping_profile(
    usage: PumpingUsage,
    family: AquiferFamily,
) -> PumpingProfile:
    """Build a complete pumping profile for the given usage x aquifer combination."""
    rate = _RATES[usage][family]
    distance = _DISTANCE[family]

    if usage == "aep":
        return PumpingProfile(
            rate_m3d=rate,
            distance_m=distance,
            pattern="constant",
            active_months=tuple(range(1, 13)),
            peak_months=(6, 7, 8),
            peak_factor=1.25,
        )
    elif usage == "irrigation":
        return PumpingProfile(
            rate_m3d=rate,
            distance_m=distance,
            pattern="seasonal",
            active_months=(4, 5, 6, 7, 8, 9),
            peak_months=(6, 7, 8),
            peak_factor=1.3,
        )
    else:  # industrial
        return PumpingProfile(
            rate_m3d=rate,
            distance_m=distance,
            pattern="constant",
            active_months=tuple(range(1, 13)),
            peak_months=(),
            peak_factor=1.0,
        )


def get_all_profiles() -> dict[str, dict[str, dict]]:
    """Return the full referential as nested dicts for JSON serialization."""
    result: dict[str, dict[str, dict]] = {}
    for usage in ("aep", "irrigation", "industrial"):
        result[usage] = {}
        for family in ("alluvial", "sedimentary", "karst", "fractured", "volcanic"):
            p = get_pumping_profile(usage, family)
            result[usage][family] = {
                "rate_m3d": _range_to_dict(p.rate_m3d),
                "distance_m": _range_to_dict(p.distance_m),
                "pattern": p.pattern,
                "active_months": list(p.active_months),
                "peak_months": list(p.peak_months),
                "peak_factor": p.peak_factor,
                "typical_duration_days": p.typical_duration_days,
                "rfunc": p.rfunc,
            }
    return result


def _range_to_dict(r: Range) -> dict:
    return {
        "default": r.default,
        "typical_min": r.typical_min,
        "typical_max": r.typical_max,
        "hard_min": r.hard_min,
        "hard_max": r.hard_max,
    }
```

- [ ] **Step 4: Run tests to verify detection passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py::TestDetectAquiferFamily -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Write failing tests for referential integrity**

Append to `tests/pastas/test_scenario_presets.py`:

```python
from dashboard.utils.pastas.scenario_presets import (
    get_pumping_profile,
    get_all_profiles,
    SCALE_STRESS_LIMITS,
    LINEAR_TREND_LIMITS,
    PumpingProfile,
    Range,
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
```

- [ ] **Step 6: Run full test suite**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add dashboard/utils/pastas/scenario_presets.py tests/pastas/test_scenario_presets.py
git commit -m "feat(pastas): add scenario presets referential with aquifer detection"
```

---

### Task 2: Validation System (Backend)

**Files:**
- Modify: `dashboard/utils/pastas/scenario_presets.py`
- Test: `tests/pastas/test_scenario_presets.py`

- [ ] **Step 1: Write failing tests for validation**

Append to `tests/pastas/test_scenario_presets.py`:

```python
from dashboard.utils.pastas.scenario_presets import (
    validate_modifications,
    ValidationResult,
)


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
        ]
        # fractured hard_max for any usage is max 500, so 2x = 1000. 800 < 1000 => valid
        # But let's set individual rate to 250 each to be under individual max,
        # and use fractured where hard_max = 500, 2x = 1000. Total 800 passes.
        # Need total > 1000 for fractured:
        mods2 = [
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 500,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 800,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
            {"type": "pumping_synthetic", "rate_m3d": 400, "distance_m": 1000,
             "start": "2020-01-01", "end": "2021-01-01", "pattern": "constant"},
        ]
        result = validate_modifications(mods2, "fractured")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py::TestHardValidation -v`
Expected: `ImportError: cannot import name 'validate_modifications'`

- [ ] **Step 3: Implement validation functions**

Append to `dashboard/utils/pastas/scenario_presets.py`:

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _get_max_hard_rate(family: AquiferFamily) -> float:
    """Return the highest hard_max rate across all usages for a given family."""
    return max(r[family].hard_max for r in _RATES.values())


def validate_modifications(
    modifications: list[dict],
    aquifer_family: AquiferFamily,
) -> ValidationResult:
    """Validate a list of modifications against physical plausibility bounds."""
    errors: list[str] = []
    warnings: list[str] = []
    total_pumping_rate = 0.0
    family_label = AQUIFER_FAMILY_LABELS.get(aquifer_family, aquifer_family)

    for i, mod in enumerate(modifications):
        mod_type = mod.get("type", "")
        start = mod.get("start")
        end = mod.get("end")
        prefix = f"Modification #{i + 1}"

        if start and end and str(end) <= str(start):
            errors.append(f"{prefix} : date de fin antérieure ou égale à la date de début")

        if mod_type in ("pumping_synthetic", "pumping_upload"):
            rate = float(mod.get("rate_m3d", 0))
            distance = float(mod.get("distance_m", 0))
            usage = mod.get("usage")
            hard_max_rate = _get_max_hard_rate(aquifer_family)
            dist_range = _DISTANCE[aquifer_family]

            if rate < 0:
                errors.append(f"{prefix} : débit négatif ({rate} m³/j)")
            elif rate > hard_max_rate:
                errors.append(
                    f"{prefix} : débit de {rate} m³/j dépasse le maximum "
                    f"pour nappe {family_label} ({hard_max_rate} m³/j)"
                )

            if distance < dist_range.hard_min:
                errors.append(
                    f"{prefix} : distance {distance}m inférieure au minimum ({dist_range.hard_min}m)"
                )
            elif distance > dist_range.hard_max:
                errors.append(
                    f"{prefix} : distance {distance}m dépasse le maximum ({dist_range.hard_max}m)"
                )

            total_pumping_rate += rate

            if usage and usage in _RATES:
                profile_rate = _RATES[usage][aquifer_family]
                usage_label = USAGE_LABELS.get(usage, usage)
                if rate > 0 and (rate < profile_rate.typical_min or rate > profile_rate.typical_max):
                    warnings.append(
                        f"Débit de {rate} m³/j inhabituel pour un pompage {usage_label} "
                        f"sur nappe {family_label} — plage typique : "
                        f"{profile_rate.typical_min}-{profile_rate.typical_max} m³/j"
                    )

            if 0 < distance < 50:
                warnings.append(
                    f"Distance très faible ({distance}m), l'impact piézométrique "
                    f"pourrait être surestimé"
                )

            if usage == "irrigation":
                season = mod.get("season_months", [])
                if season and not any(m in range(4, 10) for m in season):
                    warnings.append("Pompage d'irrigation hors période végétative")

        elif mod_type == "scale_stress":
            factor = float(mod.get("factor", 1.0))
            if factor < SCALE_STRESS_LIMITS.hard_min or factor > SCALE_STRESS_LIMITS.hard_max:
                errors.append(
                    f"{prefix} : facteur {factor} hors limites "
                    f"[{SCALE_STRESS_LIMITS.hard_min}, {SCALE_STRESS_LIMITS.hard_max}]"
                )
            elif factor < 0.5:
                stress_name = "précipitation" if mod.get("stress") == "precip" else "évapotranspiration"
                pct = round((1 - factor) * 100)
                warnings.append(
                    f"Réduction de {stress_name} de {pct}% — scénario très sévère"
                )

        elif mod_type == "linear_trend":
            slope = float(mod.get("slope_m_per_year", mod.get("slope", 0)))
            if slope < LINEAR_TREND_LIMITS.hard_min or slope > LINEAR_TREND_LIMITS.hard_max:
                errors.append(
                    f"{prefix} : pente {slope} m/an hors limites "
                    f"[{LINEAR_TREND_LIMITS.hard_min}, {LINEAR_TREND_LIMITS.hard_max}]"
                )

    cumulative_hard_max = 2 * _get_max_hard_rate(aquifer_family)
    if total_pumping_rate > cumulative_hard_max:
        errors.append(
            f"Débit de pompage cumulé ({total_pumping_rate} m³/j) dépasse "
            f"2× le maximum pour nappe {family_label} ({cumulative_hard_max} m³/j)"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run all validation tests**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/scenario_presets.py tests/pastas/test_scenario_presets.py
git commit -m "feat(pastas): add double-layer validation for scenario modifications"
```

---

### Task 3: Preset Scenario Builders (Backend)

**Files:**
- Modify: `dashboard/utils/pastas/scenario_presets.py`
- Test: `tests/pastas/test_scenario_presets.py`

- [ ] **Step 1: Write failing tests for preset builders**

Append to `tests/pastas/test_scenario_presets.py`:

```python
from dashboard.utils.pastas.scenario_presets import (
    build_preset_scenarios,
)


class TestPresetScenarios:
    def test_returns_6_presets(self):
        presets = build_preset_scenarios("alluvial", "2020-01-01", "2023-12-31")
        assert len(presets) == 6

    def test_preset_ids_unique(self):
        presets = build_preset_scenarios("alluvial", "2020-01-01", "2023-12-31")
        ids = [p["id"] for p in presets]
        assert len(ids) == len(set(ids))

    def test_aep_well_preset_uses_referential(self):
        presets = build_preset_scenarios("fractured", "2020-01-01", "2023-12-31")
        aep = next(p for p in presets if p["id"] == "aep_well")
        mods = aep["modifications"]
        assert len(mods) == 1
        assert mods[0]["type"] == "pumping_synthetic"
        assert mods[0]["usage"] == "aep"
        assert mods[0]["rate_m3d"] == 30  # fractured AEP default

    def test_irrigation_preset_seasonal(self):
        presets = build_preset_scenarios("alluvial", "2020-01-01", "2023-12-31")
        irr = next(p for p in presets if p["id"] == "irrigation")
        mod = irr["modifications"][0]
        assert mod["pattern"] == "seasonal"
        assert mod["season_months"] == [4, 5, 6, 7, 8, 9]

    def test_drought_preset(self):
        presets = build_preset_scenarios("alluvial", "2020-01-01", "2023-12-31")
        drought = next(p for p in presets if p["id"] == "summer_drought")
        mod = drought["modifications"][0]
        assert mod["type"] == "scale_stress"
        assert mod["stress"] == "precip"
        assert mod["factor"] == 0.7

    def test_all_presets_pass_validation(self):
        for family in FAMILIES:
            presets = build_preset_scenarios(family, "2020-01-01", "2023-12-31")
            for preset in presets:
                result = validate_modifications(preset["modifications"], family)
                assert result.valid, f"Preset {preset['id']} fails validation on {family}: {result.errors}"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py::TestPresetScenarios -v`
Expected: `ImportError: cannot import name 'build_preset_scenarios'`

- [ ] **Step 3: Implement preset builders**

Append to `dashboard/utils/pastas/scenario_presets.py`:

```python
def build_preset_scenarios(
    family: AquiferFamily,
    tmin: str,
    tmax: str,
) -> list[dict]:
    """Build the 6 contextual preset scenarios with concrete values."""
    aep = get_pumping_profile("aep", family)
    irr = get_pumping_profile("irrigation", family)
    ind = get_pumping_profile("industrial", family)

    return [
        {
            "id": "aep_well",
            "name": "Nouveau forage AEP",
            "description": f"Pompage eau potable {aep.rate_m3d.default} m³/j",
            "icon": "🚰",
            "modifications": [{
                "type": "pumping_synthetic",
                "usage": "aep",
                "pattern": aep.pattern,
                "rate_m3d": aep.rate_m3d.default,
                "distance_m": aep.distance_m.default,
                "start": tmin,
                "end": tmax,
                "rfunc": aep.rfunc,
                "season_months": list(aep.active_months),
                "peak_months": list(aep.peak_months),
                "peak_factor": aep.peak_factor,
            }],
        },
        {
            "id": "irrigation",
            "name": "Irrigation saisonnière",
            "description": f"Pompage agricole {irr.rate_m3d.default} m³/j (avr-sep)",
            "icon": "🌾",
            "modifications": [{
                "type": "pumping_synthetic",
                "usage": "irrigation",
                "pattern": "seasonal",
                "rate_m3d": irr.rate_m3d.default,
                "distance_m": irr.distance_m.default,
                "start": tmin,
                "end": tmax,
                "rfunc": irr.rfunc,
                "season_months": list(irr.active_months),
                "peak_months": list(irr.peak_months),
                "peak_factor": irr.peak_factor,
            }],
        },
        {
            "id": "industrial",
            "name": "Prélèvement industriel",
            "description": f"Pompage constant {ind.rate_m3d.default} m³/j",
            "icon": "🏭",
            "modifications": [{
                "type": "pumping_synthetic",
                "usage": "industrial",
                "pattern": "constant",
                "rate_m3d": ind.rate_m3d.default,
                "distance_m": ind.distance_m.default,
                "start": tmin,
                "end": tmax,
                "rfunc": ind.rfunc,
                "season_months": list(ind.active_months),
            }],
        },
        {
            "id": "summer_drought",
            "name": "Sécheresse estivale",
            "description": "−30% précipitations juin-septembre",
            "icon": "☀️",
            "modifications": [{
                "type": "scale_stress",
                "stress": "precip",
                "factor": 0.7,
                "start": tmin,
                "end": tmax,
            }],
        },
        {
            "id": "prolonged_drought",
            "name": "Sécheresse prolongée",
            "description": "−20% précip + +10% ETP sur 2 ans",
            "icon": "🏜️",
            "modifications": [
                {"type": "scale_stress", "stress": "precip", "factor": 0.8,
                 "start": tmin, "end": tmax},
                {"type": "scale_stress", "stress": "evap", "factor": 1.1,
                 "start": tmin, "end": tmax},
            ],
        },
        {
            "id": "climate_trend",
            "name": "Tendance climatique",
            "description": "Baisse −2 cm/an + hausse ETP +5%",
            "icon": "📉",
            "modifications": [
                {"type": "linear_trend", "start": tmin, "end": tmax,
                 "slope_m_per_year": -0.02},
                {"type": "scale_stress", "stress": "evap", "factor": 1.05,
                 "start": tmin, "end": tmax},
            ],
        },
    ]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/scenario_presets.py tests/pastas/test_scenario_presets.py
git commit -m "feat(pastas): add contextual preset scenario builders"
```

---

### Task 4: Integrate Validation into Scenario Simulation (Backend)

**Files:**
- Modify: `dashboard/utils/pastas/scenario.py`
- Modify: `dashboard/utils/pastas/fit_service.py` (add BDLISA tags)

- [ ] **Step 1: Add BDLISA metadata to MLflow tags during fit**

In `dashboard/utils/pastas/fit_service.py`, after line 409 (`"station_id": station_id,`), the `tags_dict` needs `nature_eh` and `milieu_eh`. But these aren't currently passed to `run_fit`. We need to add an optional `metadata` parameter.

Find the `run_fit` function signature in `fit_service.py` and add `station_metadata`:

Modify `dashboard/utils/pastas/fit_service.py` — find the function signature of `run_fit` and add `station_metadata: dict[str, str] | None = None` parameter. Then add to `tags_dict`:

```python
if station_metadata:
    if station_metadata.get("nature_eh"):
        tags_dict["nature_eh"] = station_metadata["nature_eh"]
    if station_metadata.get("milieu_eh"):
        tags_dict["milieu_eh"] = station_metadata["milieu_eh"]
```

- [ ] **Step 2: Wire station_metadata through the fit API endpoint**

In `api/routers/pastas.py`, the fit endpoint already has access to station metadata via `load_station_series`. Pass it to `run_fit`:

Find where `run_fit(` is called in `api/routers/pastas.py` and add `station_metadata=station.metadata` (or equivalent dict with nature_eh/milieu_eh).

- [ ] **Step 3: Add aquifer_family resolver to scenario.py**

Add a helper function to `dashboard/utils/pastas/scenario.py`:

```python
from dashboard.utils.pastas.scenario_presets import (
    detect_aquifer_family,
    validate_modifications,
    AquiferFamily,
)

def _resolve_aquifer_family(run_id: str) -> AquiferFamily:
    """Resolve aquifer family from MLflow run tags."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags
    return detect_aquifer_family(tags.get("nature_eh"), tags.get("milieu_eh"))
```

Add `import mlflow` at the top of `scenario.py`.

- [ ] **Step 4: Integrate validation into simulate_scenario**

In `simulate_scenario()`, add validation after loading the model and before applying modifications:

```python
# After loading original model, before baseline simulation:
aquifer_family = _resolve_aquifer_family(run_id)
validation = validate_modifications(modifications, aquifer_family)
if not validation.valid:
    raise ValueError(
        "Modifications invalides : " + " ; ".join(validation.errors)
    )
warnings.extend(validation.warnings)
```

- [ ] **Step 5: Run existing scenario tests to verify backward compatibility**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario.py -v`
Expected: All 6 existing tests PASS (they use `monkeypatch` which may need MLflow tags adjustment — if they fail due to missing tags, the `detect_aquifer_family` returns `"sedimentary"` by default, so they should still pass)

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/pastas/scenario.py dashboard/utils/pastas/fit_service.py api/routers/pastas.py
git commit -m "feat(pastas): integrate validation into scenario simulation pipeline"
```

---

### Task 5: API Endpoints for Presets and Validation

**Files:**
- Modify: `api/schemas/pastas.py`
- Modify: `api/routers/pastas.py`

- [ ] **Step 1: Add new Pydantic schemas**

In `api/schemas/pastas.py`, add after the existing `ScenarioResponse` class (around line 172):

```python
# ---------- Scenario Presets ----------

AquiferFamily = Literal["alluvial", "sedimentary", "karst", "fractured", "volcanic"]
PumpingUsage = Literal["aep", "irrigation", "industrial"]

class PumpingRangeSchema(BaseModel):
    default: float
    typical_min: float
    typical_max: float
    hard_min: float
    hard_max: float

class PumpingProfileSchema(BaseModel):
    rate_m3d: PumpingRangeSchema
    distance_m: PumpingRangeSchema
    pattern: str
    active_months: list[int]
    peak_months: list[int]
    peak_factor: float
    typical_duration_days: int
    rfunc: str

class PresetScenarioSchema(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    modifications: list[dict[str, Any]]

class NonPumpingLimitsSchema(BaseModel):
    scale_stress: PumpingRangeSchema
    linear_trend: PumpingRangeSchema

class ScenarioPresetsResponse(BaseModel):
    aquifer_families: dict[str, str]
    pumping_profiles: dict[str, dict[str, PumpingProfileSchema]]
    non_pumping_limits: NonPumpingLimitsSchema
    presets: list[PresetScenarioSchema]
    detected_family: Optional[str] = None

class ValidateModificationsRequest(BaseModel):
    modifications: list[Modification]
    aquifer_family: Optional[AquiferFamily] = None

class ValidateModificationsResponse(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
```

Also update `PumpingSynthetic` to add the new optional fields:

```python
class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"] = "pumping_synthetic"
    usage: Optional[Literal["aep", "irrigation", "industrial"]] = None
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float = Field(ge=0)
    start: date
    end: date
    distance_m: float = Field(gt=0)
    screen_depth_m: Optional[float] = None
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"
    period_days: int = 365
    season_months: Optional[list[int]] = None
    peak_months: Optional[list[int]] = None
    peak_factor: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    pulse_duration_days: int = Field(default=30, ge=1)
```

- [ ] **Step 2: Add the GET /scenario-presets endpoint**

In `api/routers/pastas.py`, add a new endpoint after the simulate endpoint:

```python
@router.get("/scenario-presets", response_model=ScenarioPresetsResponse)
def scenario_presets(
    aquifer_family: Optional[str] = Query(None),
    tmin: Optional[str] = Query(None),
    tmax: Optional[str] = Query(None),
) -> ScenarioPresetsResponse:
    """Return the full scenario referential for frontend cache."""
    from dashboard.utils.pastas.scenario_presets import (
        AQUIFER_FAMILY_LABELS,
        SCALE_STRESS_LIMITS,
        LINEAR_TREND_LIMITS,
        get_all_profiles,
        build_preset_scenarios,
        _range_to_dict,
    )

    family = aquifer_family or "sedimentary"
    t0 = tmin or "2020-01-01"
    t1 = tmax or "2024-12-31"

    return ScenarioPresetsResponse(
        aquifer_families=AQUIFER_FAMILY_LABELS,
        pumping_profiles=get_all_profiles(),
        non_pumping_limits={
            "scale_stress": _range_to_dict(SCALE_STRESS_LIMITS),
            "linear_trend": _range_to_dict(LINEAR_TREND_LIMITS),
        },
        presets=build_preset_scenarios(family, t0, t1),
        detected_family=family,
    )
```

- [ ] **Step 3: Add the POST /validate-modifications endpoint**

```python
@router.post("/validate-modifications", response_model=ValidateModificationsResponse)
def validate_modifications_endpoint(
    req: ValidateModificationsRequest,
) -> ValidateModificationsResponse:
    """Pre-validate modifications without running a simulation."""
    from dashboard.utils.pastas.scenario_presets import validate_modifications

    family = req.aquifer_family or "sedimentary"
    mods = [m.model_dump() for m in req.modifications]
    result = validate_modifications(mods, family)

    return ValidateModificationsResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
    )
```

- [ ] **Step 4: Add schema imports**

Add the new schemas to the import block at the top of `api/routers/pastas.py`:

```python
from api.schemas.pastas import (
    ...,
    ScenarioPresetsResponse,
    ValidateModificationsRequest,
    ValidateModificationsResponse,
)
```

And add `from typing import Optional` and `from fastapi import Query` if not already imported.

- [ ] **Step 5: Test endpoints manually**

Run: `docker compose up -d --build backend`

Then test:
```bash
rtk proxy curl -s http://localhost:49513/api/v1/pastas/scenario-presets | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['presets']), 'presets,', len(d['pumping_profiles']), 'usage types')"
```
Expected: `6 presets, 3 usage types`

```bash
rtk proxy curl -s -X POST http://localhost:49513/api/v1/pastas/validate-modifications -H 'Content-Type: application/json' -d '{"modifications":[{"type":"pumping_synthetic","pattern":"constant","rate_m3d":99999,"start":"2020-01-01","end":"2021-01-01","distance_m":500}],"aquifer_family":"fractured"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('valid:', d['valid'], 'errors:', d['errors'])"
```
Expected: `valid: False errors: [...]`

- [ ] **Step 6: Commit**

```bash
git add api/schemas/pastas.py api/routers/pastas.py
git commit -m "feat(pastas): add scenario-presets and validate-modifications API endpoints"
```

---

### Task 6: Scenario Persistence (Backend)

**Files:**
- Modify: `dashboard/utils/pastas/scenario_presets.py`
- Modify: `api/routers/pastas.py`
- Modify: `api/schemas/pastas.py`

- [ ] **Step 1: Write failing tests for persistence**

Append to `tests/pastas/test_scenario_presets.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dashboard.utils.pastas.scenario_presets import (
    save_scenario,
    list_scenarios,
    delete_scenario,
    load_scenario,
)


class TestScenarioPersistence:
    def test_save_and_list(self, tmp_path):
        run_id = "test_run_123"
        mods = [{"type": "scale_stress", "stress": "precip", "factor": 0.8,
                 "start": "2020-01-01", "end": "2021-01-01"}]

        with patch("dashboard.utils.pastas.scenario_presets.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            # Mock log_artifact to write file to tmp_path
            def fake_log_artifact(rid, local_path, artifact_path=None):
                dest = tmp_path / "artifacts" / (artifact_path or "") / Path(local_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(local_path, dest)
            mock_client.log_artifact.side_effect = fake_log_artifact

            save_scenario(run_id, "my_scenario", mods, "Test scenario",
                          aquifer_family="alluvial", tmin="2020-01-01", tmax="2021-01-01")

            mock_client.log_artifact.assert_called_once()
            call_args = mock_client.log_artifact.call_args
            assert call_args[0][0] == run_id
            assert call_args[1].get("artifact_path") == "scenarios" or call_args[0][2] == "scenarios"

    def test_load_scenario(self, tmp_path):
        scenario_data = {
            "name": "test",
            "description": "",
            "created_at": "2026-04-24T14:00:00",
            "aquifer_family": "alluvial",
            "tmin": "2020-01-01",
            "tmax": "2021-01-01",
            "modifications": [{"type": "scale_stress", "stress": "precip", "factor": 0.8,
                               "start": "2020-01-01", "end": "2021-01-01"}],
        }
        scenario_path = tmp_path / "test.json"
        scenario_path.write_text(json.dumps(scenario_data))

        with patch("dashboard.utils.pastas.scenario_presets.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_client.download_artifacts.return_value = str(scenario_path)

            result = load_scenario("run_123", "test")
            assert result["name"] == "test"
            assert len(result["modifications"]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py::TestScenarioPersistence -v`
Expected: `ImportError: cannot import name 'save_scenario'`

- [ ] **Step 3: Implement persistence functions**

Append to `dashboard/utils/pastas/scenario_presets.py`:

```python
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

SCENARIOS_ARTIFACT_PATH = "scenarios"


def save_scenario(
    run_id: str,
    name: str,
    modifications: list[dict],
    description: str = "",
    aquifer_family: str | None = None,
    tmin: str | None = None,
    tmax: str | None = None,
) -> None:
    """Save a named scenario as an MLflow artifact."""
    data = {
        "name": name,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "aquifer_family": aquifer_family,
        "tmin": tmin,
        "tmax": tmax,
        "modifications": modifications,
    }
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = Path(tmpdir) / f"{safe_name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        client.log_artifact(run_id, str(path), SCENARIOS_ARTIFACT_PATH)


def list_scenarios(run_id: str) -> list[dict]:
    """List saved scenarios for a model run."""
    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id, SCENARIOS_ARTIFACT_PATH)
    except Exception:
        return []

    scenarios = []
    for art in artifacts:
        if art.path.endswith(".json"):
            try:
                local = client.download_artifacts(run_id, art.path)
                data = json.loads(Path(local).read_text())
                scenarios.append(data)
            except Exception as exc:
                logger.warning("Failed to load scenario %s: %s", art.path, exc)
    return scenarios


def load_scenario(run_id: str, name: str) -> dict:
    """Load a specific saved scenario by name."""
    client = mlflow.tracking.MlflowClient()
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    artifact_path = f"{SCENARIOS_ARTIFACT_PATH}/{safe_name}.json"
    local = client.download_artifacts(run_id, artifact_path)
    return json.loads(Path(local).read_text())


def delete_scenario(run_id: str, name: str) -> None:
    """Delete a saved scenario. MLflow doesn't support artifact deletion natively,
    so we overwrite with an empty marker file."""
    client = mlflow.tracking.MlflowClient()
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / f"{safe_name}.json"
        marker.write_text(json.dumps({"_deleted": True}))
        client.log_artifact(run_id, str(marker), SCENARIOS_ARTIFACT_PATH)
```

- [ ] **Step 4: Add API schemas for persistence**

Append to `api/schemas/pastas.py`:

```python
class SaveScenarioRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    modifications: list[Modification]
    tmin: Optional[date] = None
    tmax: Optional[date] = None

class SavedScenario(BaseModel):
    name: str
    description: str = ""
    created_at: str
    aquifer_family: Optional[str] = None
    tmin: Optional[str] = None
    tmax: Optional[str] = None
    modifications: list[dict[str, Any]]

class ApplyScenarioRequest(BaseModel):
    target_run_id: str
```

- [ ] **Step 5: Add CRUD endpoints**

In `api/routers/pastas.py`, add:

```python
@router.get("/models/{run_id}/scenarios", response_model=list[SavedScenario])
def get_scenarios(run_id: str) -> list[SavedScenario]:
    """List saved scenarios for a model."""
    from dashboard.utils.pastas.scenario_presets import list_scenarios
    return [s for s in list_scenarios(run_id) if not s.get("_deleted")]


@router.post("/models/{run_id}/scenarios", status_code=201)
def create_scenario(run_id: str, req: SaveScenarioRequest) -> dict:
    """Save a named scenario."""
    from dashboard.utils.pastas.scenario_presets import save_scenario, detect_aquifer_family
    from dashboard.utils.pastas.scenario import _resolve_aquifer_family

    family = _resolve_aquifer_family(run_id)
    mods = [m.model_dump() for m in req.modifications]
    save_scenario(
        run_id=run_id,
        name=req.name,
        modifications=mods,
        description=req.description,
        aquifer_family=family,
        tmin=str(req.tmin) if req.tmin else None,
        tmax=str(req.tmax) if req.tmax else None,
    )
    return {"status": "saved", "name": req.name}


@router.delete("/models/{run_id}/scenarios/{name}")
def remove_scenario(run_id: str, name: str) -> dict:
    """Delete a saved scenario."""
    from dashboard.utils.pastas.scenario_presets import delete_scenario
    delete_scenario(run_id, name)
    return {"status": "deleted", "name": name}


@router.post("/models/{run_id}/scenarios/{name}/apply")
def apply_scenario(run_id: str, name: str, req: ApplyScenarioRequest) -> SavedScenario:
    """Load a saved scenario, optionally adjusting for cross-model reuse."""
    from dashboard.utils.pastas.scenario_presets import load_scenario, validate_modifications
    from dashboard.utils.pastas.scenario import _resolve_aquifer_family
    from dashboard.utils.pastas.io import load_model

    scenario = load_scenario(run_id, name)
    target_family = _resolve_aquifer_family(req.target_run_id)
    source_family = scenario.get("aquifer_family")

    warnings = []
    if source_family and source_family != target_family:
        warnings.append(
            f"Scénario calibré sur nappe {source_family}, appliqué sur nappe {target_family} "
            f"— vérifiez les ordres de grandeur"
        )

    # Clamp dates to target model period
    target_model = load_model(req.target_run_id)
    model_tmin = str(target_model.get_tmin(use_oseries=True, use_stresses=True).date())
    model_tmax = str(target_model.get_tmax(use_oseries=True, use_stresses=True).date())

    for mod in scenario["modifications"]:
        if "start" in mod and mod["start"] < model_tmin:
            mod["start"] = model_tmin
        if "end" in mod and mod["end"] > model_tmax:
            mod["end"] = model_tmax

    validation = validate_modifications(scenario["modifications"], target_family)
    warnings.extend(validation.warnings)

    scenario["_warnings"] = warnings
    return scenario
```

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/pastas/scenario_presets.py api/routers/pastas.py api/schemas/pastas.py tests/pastas/test_scenario_presets.py
git commit -m "feat(pastas): add scenario persistence via MLflow artifacts"
```

---

### Task 7: Frontend — Types, API Client, and Hooks

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useScenarioPresets.ts`
- Create: `frontend/src/hooks/useSavedScenarios.ts`

- [ ] **Step 1: Add TypeScript types**

In `frontend/src/lib/types.ts`, add after `PastasScenarioResponse`:

```typescript
export interface PumpingRange {
  default: number
  typical_min: number
  typical_max: number
  hard_min: number
  hard_max: number
}

export interface PumpingProfileData {
  rate_m3d: PumpingRange
  distance_m: PumpingRange
  pattern: string
  active_months: number[]
  peak_months: number[]
  peak_factor: number
  typical_duration_days: number
  rfunc: string
}

export interface PresetScenario {
  id: string
  name: string
  description: string
  icon: string
  modifications: Record<string, unknown>[]
}

export interface ScenarioPresetsData {
  aquifer_families: Record<string, string>
  pumping_profiles: Record<string, Record<string, PumpingProfileData>>
  non_pumping_limits: {
    scale_stress: PumpingRange
    linear_trend: PumpingRange
  }
  presets: PresetScenario[]
  detected_family: string | null
}

export interface SavedScenario {
  name: string
  description: string
  created_at: string
  aquifer_family: string | null
  tmin: string | null
  tmax: string | null
  modifications: Record<string, unknown>[]
}

export type AquiferFamily = 'alluvial' | 'sedimentary' | 'karst' | 'fractured' | 'volcanic'
export type PumpingUsage = 'aep' | 'irrigation' | 'industrial'
```

- [ ] **Step 2: Add API client methods**

In `frontend/src/lib/api.ts`, add to the `pastas` object:

```typescript
scenarioPresets: (params?: { aquifer_family?: string; tmin?: string; tmax?: string }) => {
  const qs = new URLSearchParams()
  if (params?.aquifer_family) qs.set('aquifer_family', params.aquifer_family)
  if (params?.tmin) qs.set('tmin', params.tmin)
  if (params?.tmax) qs.set('tmax', params.tmax)
  const query = qs.toString()
  return fetchJson<ScenarioPresetsData>(`/pastas/scenario-presets${query ? `?${query}` : ''}`)
},
savedScenarios: (runId: string) =>
  fetchJson<SavedScenario[]>(`/pastas/models/${runId}/scenarios`),
saveScenario: (runId: string, body: { name: string; description?: string; modifications: Array<Record<string, unknown>>; tmin?: string; tmax?: string }) =>
  postJson<{ status: string; name: string }>(`/pastas/models/${runId}/scenarios`, body),
deleteScenario: (runId: string, name: string) =>
  deleteJson(`/pastas/models/${runId}/scenarios/${encodeURIComponent(name)}`),
```

Add `ScenarioPresetsData` and `SavedScenario` to the import block at the top of `api.ts`.

- [ ] **Step 3: Create useScenarioPresets hook**

Create `frontend/src/hooks/useScenarioPresets.ts`:

```typescript
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useScenarioPresets(params?: { aquifer_family?: string; tmin?: string; tmax?: string }) {
  return useQuery({
    queryKey: ['pastas', 'scenario-presets', params?.aquifer_family, params?.tmin, params?.tmax],
    queryFn: () => api.pastas.scenarioPresets(params),
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  })
}
```

- [ ] **Step 4: Create useSavedScenarios hook**

Create `frontend/src/hooks/useSavedScenarios.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useSavedScenarios(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'saved-scenarios', runId],
    queryFn: () => api.pastas.savedScenarios(runId!),
    enabled: !!runId,
  })
}

export function useSaveScenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runId, body }: {
      runId: string
      body: { name: string; description?: string; modifications: Array<Record<string, unknown>>; tmin?: string; tmax?: string }
    }) => api.pastas.saveScenario(runId, body),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ['pastas', 'saved-scenarios', variables.runId] })
    },
  })
}

export function useDeleteScenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runId, name }: { runId: string; name: string }) =>
      api.pastas.deleteScenario(runId, name),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ['pastas', 'saved-scenarios', variables.runId] })
    },
  })
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/hooks/useScenarioPresets.ts frontend/src/hooks/useSavedScenarios.ts
git commit -m "feat(pastas): add frontend types, API client, and hooks for scenario presets"
```

---

### Task 8: Frontend — PumpingSyntheticEditor with Usage Selector and Warnings

**Files:**
- Modify: `frontend/src/components/pastas/modifications/PumpingSyntheticEditor.tsx`

- [ ] **Step 1: Add usage selector and warning logic**

Replace the content of `PumpingSyntheticEditor.tsx`. Key changes:
- Add a `usage` selector (AEP / Irrigation / Industriel / Personnalisé)
- Accept `aquiferFamily` and `presetsData` props
- Show inline warnings when values are outside typical range
- Set `min`/`max` on inputs from hard limits

```typescript
import { useMemo } from 'react'
import type { PumpingProfileData, PumpingRange } from '@/lib/types'

interface PumpingSyntheticData {
  type: 'pumping_synthetic'
  usage?: 'aep' | 'irrigation' | 'industrial'
  pattern: 'constant' | 'seasonal' | 'pulse'
  rate_m3d: number
  distance_m: number
  start: string
  end: string
  rfunc: 'Exponential' | 'Hantush'
  season_months?: number[]
  peak_months?: number[]
  peak_factor?: number
  pulse_duration_days?: number
}

interface PumpingSyntheticEditorProps {
  data: PumpingSyntheticData
  onChange: (data: PumpingSyntheticData) => void
  profile?: PumpingProfileData | null
}

const USAGES = [
  { value: 'aep' as const, label: 'AEP (eau potable)', icon: '🚰' },
  { value: 'irrigation' as const, label: 'Irrigation', icon: '🌾' },
  { value: 'industrial' as const, label: 'Industriel', icon: '🏭' },
] as const

// ... (keep existing PATTERNS, SEASON_PRESETS, MONTH_LABELS, RFUNCS, inputClass, generatePreview)

function RangeWarning({ value, range, label }: { value: number; range?: PumpingRange; label: string }) {
  if (!range || value <= 0) return null
  if (value < range.typical_min || value > range.typical_max) {
    return (
      <p className="text-[10px] mt-1 text-yellow-400">
        ⚠ {label} inhabituel — plage typique : {range.typical_min}–{range.typical_max}
      </p>
    )
  }
  return null
}

export function PumpingSyntheticEditor({ data, onChange, profile }: PumpingSyntheticEditorProps) {
  function update(patch: Partial<PumpingSyntheticData>) {
    onChange({ ...data, ...patch })
  }

  function selectUsage(usage: 'aep' | 'irrigation' | 'industrial' | undefined) {
    const patch: Partial<PumpingSyntheticData> = { usage: usage || undefined }
    if (usage && profile) {
      // pre-fill from profile
      patch.rate_m3d = profile.rate_m3d.default
      patch.distance_m = profile.distance_m.default
      patch.pattern = profile.pattern as 'constant' | 'seasonal' | 'pulse'
      patch.season_months = profile.active_months
      patch.peak_months = profile.peak_months
      patch.peak_factor = profile.peak_factor
      patch.rfunc = profile.rfunc as 'Exponential' | 'Hantush'
    }
    onChange({ ...data, ...patch })
  }

  // ... (keep existing preview, toggleMonth, etc.)

  const rateRange = profile?.rate_m3d
  const distRange = profile?.distance_m

  return (
    <div className="space-y-3">
      {/* Usage selector */}
      <div>
        <label className="block text-xs text-text-muted mb-1.5">Type de pompage</label>
        <div className="grid grid-cols-4 gap-1">
          {USAGES.map(u => (
            <button
              key={u.value}
              onClick={() => selectUsage(u.value)}
              className={`px-2 py-1.5 rounded-lg text-[10px] font-medium transition-colors ${
                data.usage === u.value
                  ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
              }`}
            >
              {u.icon} {u.label}
            </button>
          ))}
          <button
            onClick={() => selectUsage(undefined)}
            className={`px-2 py-1.5 rounded-lg text-[10px] font-medium transition-colors ${
              !data.usage
                ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
            }`}
          >
            Personnalisé
          </button>
        </div>
      </div>

      {/* ... existing pattern selector, season months, pulse duration, preview ... */}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Débit (m³/j)</label>
          <input
            type="number"
            value={data.rate_m3d}
            onChange={(e) => update({ rate_m3d: parseFloat(e.target.value) || 0 })}
            className={inputClass}
            step="10"
            min={rateRange?.hard_min ?? 0}
            max={rateRange?.hard_max}
          />
          <RangeWarning value={data.rate_m3d} range={rateRange} label="Débit" />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Distance (m)</label>
          <input
            type="number"
            value={data.distance_m}
            onChange={(e) => update({ distance_m: parseFloat(e.target.value) || 0 })}
            className={inputClass}
            step="100"
            min={distRange?.hard_min ?? 1}
            max={distRange?.hard_max}
          />
          <RangeWarning value={data.distance_m} range={distRange} label="Distance" />
        </div>
      </div>

      {/* ... existing date and rfunc fields ... */}
    </div>
  )
}

export type { PumpingSyntheticData }
```

Note: The full component keeps ALL existing UI (patterns, season months, pulse duration, preview chart, dates, rfunc) and adds the usage selector at the top plus `RangeWarning` under the rate/distance inputs. The `profile` prop is optional for backward compatibility.

- [ ] **Step 2: Update ModificationCard to pass profile**

In `frontend/src/components/pastas/ModificationCard.tsx`, add `profile` to the props interface and pass it to `PumpingSyntheticEditor`:

```typescript
interface ModificationCardProps {
  index: number
  data: ModificationData
  onChange: (data: ModificationData) => void
  onDelete: () => void
  profile?: PumpingProfileData | null
}

// In the render:
{data.type === 'pumping_synthetic' && (
  <PumpingSyntheticEditor
    data={data}
    onChange={(d) => onChange(d)}
    profile={profile}
  />
)}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/pastas/modifications/PumpingSyntheticEditor.tsx frontend/src/components/pastas/ModificationCard.tsx
git commit -m "feat(pastas): add usage selector and range warnings to pumping editor"
```

---

### Task 9: Frontend — ScaleStressEditor and LinearTrendEditor Hard Limits

**Files:**
- Modify: `frontend/src/components/pastas/modifications/ScaleStressEditor.tsx`
- Modify: `frontend/src/components/pastas/modifications/LinearTrendEditor.tsx`

- [ ] **Step 1: Add hard limits to ScaleStressEditor**

In `ScaleStressEditor.tsx`, add a `limits` prop and apply bounds:

```typescript
import type { PumpingRange } from '@/lib/types'

interface ScaleStressEditorProps {
  data: ScaleStressData
  onChange: (data: ScaleStressData) => void
  limits?: PumpingRange | null
}

// In the input:
<input
  type="number"
  value={data.factor}
  onChange={(e) => update({ factor: parseFloat(e.target.value) || 1 })}
  className={inputClass}
  step="0.05"
  min={limits?.hard_min ?? 0}
  max={limits?.hard_max}
/>
{limits && (data.factor < (limits.typical_min) || data.factor > (limits.typical_max)) && data.factor !== 1 && (
  <p className="text-[10px] mt-1 text-yellow-400">
    ⚠ Facteur inhabituel — plage typique : {limits.typical_min}–{limits.typical_max}
  </p>
)}
```

- [ ] **Step 2: Add hard limits to LinearTrendEditor**

In `LinearTrendEditor.tsx`, add a `limits` prop and apply bounds:

```typescript
import type { PumpingRange } from '@/lib/types'

interface LinearTrendEditorProps {
  data: LinearTrendData
  onChange: (data: LinearTrendData) => void
  limits?: PumpingRange | null
}

// In the input:
<input
  type="number"
  value={data.slope_m_per_year}
  onChange={(e) => update({ slope_m_per_year: parseFloat(e.target.value) || 0 })}
  className={inputClass}
  step="0.001"
  min={limits?.hard_min}
  max={limits?.hard_max}
/>
{limits && (data.slope_m_per_year < (limits.typical_min) || data.slope_m_per_year > (limits.typical_max)) && (
  <p className="text-[10px] mt-1 text-yellow-400">
    ⚠ Pente inhabituelle — plage typique : {limits.typical_min} à {limits.typical_max} m/an
  </p>
)}
```

- [ ] **Step 3: Update ModificationCard to pass limits**

In `ModificationCard.tsx`, add `scaleStressLimits` and `linearTrendLimits` props:

```typescript
interface ModificationCardProps {
  index: number
  data: ModificationData
  onChange: (data: ModificationData) => void
  onDelete: () => void
  profile?: PumpingProfileData | null
  scaleStressLimits?: PumpingRange | null
  linearTrendLimits?: PumpingRange | null
}

// Pass them:
{data.type === 'scale_stress' && (
  <ScaleStressEditor data={data} onChange={(d) => onChange(d)} limits={scaleStressLimits} />
)}
{data.type === 'linear_trend' && (
  <LinearTrendEditor data={data} onChange={(d) => onChange(d)} limits={linearTrendLimits} />
)}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/modifications/ScaleStressEditor.tsx frontend/src/components/pastas/modifications/LinearTrendEditor.tsx frontend/src/components/pastas/ModificationCard.tsx
git commit -m "feat(pastas): add hard limits and range warnings to stress/trend editors"
```

---

### Task 10: Frontend — ScenariosPage Redesign

**Files:**
- Modify: `frontend/src/pages/pastas/ScenariosPage.tsx`
- Modify: `frontend/src/components/pastas/ScenarioComposer.tsx`

- [ ] **Step 1: Update ScenarioComposer to accept and pass presets data**

In `ScenarioComposer.tsx`, add props for presets data and pass them through:

```typescript
import type { PumpingProfileData, PumpingRange } from '@/lib/types'

interface ScenarioComposerProps {
  modifications: ModificationData[]
  onChange: (mods: ModificationData[]) => void
  tmin: string
  tmax: string
  pumpingProfile?: PumpingProfileData | null
  scaleStressLimits?: PumpingRange | null
  linearTrendLimits?: PumpingRange | null
}

// Pass to ModificationCard:
<ModificationCard
  key={i}
  index={i}
  data={mod}
  onChange={(d) => updateModification(i, d)}
  onDelete={() => deleteModification(i)}
  profile={pumpingProfile}
  scaleStressLimits={scaleStressLimits}
  linearTrendLimits={linearTrendLimits}
/>
```

- [ ] **Step 2: Redesign ScenariosPage with contextual presets and saved scenarios**

Replace the content of `ScenariosPage.tsx`. Key changes:
1. Load presets via `useScenarioPresets()`
2. Detect aquifer family from station metadata
3. Replace static PRESETS with dynamic contextual presets
4. Add "Mes scénarios" section with save/load/delete
5. Pass `profile` and `limits` to ScenarioComposer

```typescript
import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Loader2, Play, Info, Save, Trash2, FolderOpen } from 'lucide-react'
import { usePastasModels, usePastasSimulate, usePastasModel } from '@/hooks/usePastas'
import { useScenarioPresets } from '@/hooks/useScenarioPresets'
import { useSavedScenarios, useSaveScenario, useDeleteScenario } from '@/hooks/useSavedScenarios'
import { ScenarioComposer } from '@/components/pastas/ScenarioComposer'
import { ScenarioResultsPanel } from '@/components/pastas/ScenarioResultsPanel'
import type { PastasScenarioResponse, AquiferFamily } from '@/lib/types'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import type { ModificationData } from '@/components/pastas/ModificationCard'

function modificationToPayload(mod: ModificationData): Record<string, unknown> {
  if (mod.type === 'pumping_upload') {
    return { type: mod.type, csv_rows: mod.rows, distance_m: mod.distance_m, rfunc: mod.rfunc }
  }
  return mod as unknown as Record<string, unknown>
}

export default function ScenariosPage() {
  const [searchParams] = useSearchParams()
  const { data: models = [] } = usePastasModels()
  const simulateMutation = usePastasSimulate()

  const [runId, setRunId] = useState(searchParams.get('model') ?? '')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modifications, setModifications] = useState<ModificationData[]>([])
  const [simResult, setSimResult] = useState<PastasScenarioResponse | null>(null)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [saveName, setSaveName] = useState('')

  const { data: selectedModel } = usePastasModel(runId || null)

  // Detect aquifer family from station preview metadata
  const [aquiferFamily, setAquiferFamily] = useState<AquiferFamily>('sedimentary')

  // Load presets referential
  const { data: presetsData } = useScenarioPresets({
    aquifer_family: aquiferFamily,
    tmin: tmin || undefined,
    tmax: tmax || undefined,
  })

  // Load saved scenarios
  const { data: savedScenarios = [] } = useSavedScenarios(runId || null)
  const saveScenarioMutation = useSaveScenario()
  const deleteScenarioMutation = useDeleteScenario()

  useEffect(() => {
    if (selectedModel && !tmin && !tmax) {
      const obs = selectedModel.observed
      if (obs?.index?.length > 0) {
        setTmin(obs.index[0].slice(0, 10))
        setTmax(obs.index[obs.index.length - 1].slice(0, 10))
      }
    }
  }, [selectedModel])

  const selected = models.find(m => m.run_id === runId)
  const canSimulate = !!runId && !!tmin && !!tmax

  // Get current pumping profile for the selected usage x aquifer
  const currentProfile = useMemo(() => {
    if (!presetsData) return null
    const pumpingMod = modifications.find(m => m.type === 'pumping_synthetic') as { usage?: string } | undefined
    const usage = pumpingMod?.usage || 'aep'
    return presetsData.pumping_profiles[usage]?.[aquiferFamily] ?? null
  }, [presetsData, modifications, aquiferFamily])

  async function handleSimulate() {
    if (!canSimulate) return
    try {
      const result = await simulateMutation.mutateAsync({
        run_id: runId,
        tmin,
        tmax,
        modifications: modifications.map(modificationToPayload),
      })
      setSimResult(result)
    } catch { /* Error handled by mutation state */ }
  }

  function applyPreset(preset: { modifications: Record<string, unknown>[] }) {
    const mods = preset.modifications.map(m => {
      const mod = { ...m } as Record<string, unknown>
      if ('start' in mod && !mod.start && tmin) mod.start = tmin
      if ('end' in mod && !mod.end && tmax) mod.end = tmax
      return mod as unknown as ModificationData
    })
    setModifications(mods)
  }

  function loadSavedScenario(scenario: { modifications: Record<string, unknown>[] }) {
    applyPreset(scenario)
  }

  async function handleSave() {
    if (!runId || !saveName.trim()) return
    await saveScenarioMutation.mutateAsync({
      runId,
      body: {
        name: saveName.trim(),
        modifications: modifications.map(modificationToPayload),
        tmin: tmin || undefined,
        tmax: tmax || undefined,
      },
    })
    setSaveDialogOpen(false)
    setSaveName('')
  }

  const inputClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column */}
      <div className="w-96 shrink-0 space-y-4">
        <OnboardingBanner
          id="scenarios"
          title="Simulate what-if scenarios"
          description="Start from a calibrated model and modify conditions: add pumping, a climate trend, or change precipitation. Compare the result to the baseline scenario."
          steps={[
            'Select a calibrated model',
            'Choose a preset or create your own modifications',
            'Run the simulation — baseline vs scenario are displayed with contributions',
          ]}
        />

        {/* Model picker (unchanged) */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Calibrated model</h2>
          <select
            value={runId}
            onChange={(e) => { setRunId(e.target.value); setSimResult(null) }}
            className={inputClass}
          >
            <option value="">-- Select a model --</option>
            {models.map((m) => (
              <option key={m.run_id} value={m.run_id}>
                {m.name || m.run_id.slice(0, 8)} — {m.code_bss}
              </option>
            ))}
          </select>
          {selected && (
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-accent-cyan">{selected.code_bss}</span>
                {aquiferFamily !== 'sedimentary' && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-white/10 text-text-muted">
                    {presetsData?.aquifer_families[aquiferFamily] ?? aquiferFamily}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Simulation window (unchanged) */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Simulation window</h2>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-text-muted mb-1">Start</label>
              <input type="date" value={tmin} onChange={(e) => setTmin(e.target.value)} className={inputClass} />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">End</label>
              <input type="date" value={tmax} onChange={(e) => setTmax(e.target.value)} className={inputClass} />
            </div>
          </div>
        </div>

        {/* Contextual presets */}
        {runId && presetsData && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <h2 className="text-sm font-semibold text-text-primary mb-3">Scénarios prêts à l'emploi</h2>
            <div className="grid grid-cols-2 gap-1.5">
              {presetsData.presets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => applyPreset(p)}
                  className="text-left px-3 py-2 rounded-lg border border-white/5 hover:border-accent-cyan/20 hover:bg-accent-cyan/5 transition-colors group"
                >
                  <div className="text-sm mb-0.5">{p.icon}</div>
                  <div className="text-[10px] font-medium text-text-secondary group-hover:text-text-primary">{p.name}</div>
                  <div className="text-[9px] text-text-muted leading-tight">{p.description}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Custom modifications */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Modifications</h2>
          <ScenarioComposer
            modifications={modifications}
            onChange={setModifications}
            tmin={tmin}
            tmax={tmax}
            pumpingProfile={currentProfile}
            scaleStressLimits={presetsData?.non_pumping_limits?.scale_stress ?? null}
            linearTrendLimits={presetsData?.non_pumping_limits?.linear_trend ?? null}
          />
        </div>

        {/* Simulate + Save buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleSimulate}
            disabled={!canSimulate || simulateMutation.isPending}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {simulateMutation.isPending ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Simulating…</>
            ) : (
              <><Play className="w-4 h-4" /> Simulate</>
            )}
          </button>
          {modifications.length > 0 && runId && (
            <button
              onClick={() => setSaveDialogOpen(true)}
              className="flex items-center gap-1.5 px-3 py-2.5 rounded-lg bg-white/5 text-text-secondary text-sm border border-white/10 hover:border-white/20 transition-colors"
              title="Save scenario"
            >
              <Save className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Save dialog */}
        {saveDialogOpen && (
          <div className="bg-bg-card border border-white/10 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-semibold text-text-primary">Save scenario</h3>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Scenario name..."
              className={inputClass}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            />
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={!saveName.trim() || saveScenarioMutation.isPending}
                className="flex-1 px-3 py-1.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-xs font-medium border border-accent-cyan/30 disabled:opacity-40"
              >
                {saveScenarioMutation.isPending ? 'Saving…' : 'Save'}
              </button>
              <button
                onClick={() => setSaveDialogOpen(false)}
                className="px-3 py-1.5 rounded-lg text-text-muted text-xs border border-white/10"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {simulateMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {simulateMutation.error instanceof Error ? simulateMutation.error.message : 'Simulation failed.'}
            </p>
          </div>
        )}

        {/* Saved scenarios */}
        {runId && savedScenarios.length > 0 && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <h2 className="text-sm font-semibold text-text-primary mb-3">Mes scénarios</h2>
            <div className="space-y-1.5">
              {savedScenarios.map((s) => (
                <div
                  key={s.name}
                  className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/5 hover:border-white/10 group"
                >
                  <button
                    onClick={() => loadSavedScenario(s)}
                    className="flex-1 text-left"
                  >
                    <div className="text-xs font-medium text-text-secondary group-hover:text-text-primary flex items-center gap-1.5">
                      <FolderOpen className="w-3 h-3" />
                      {s.name}
                    </div>
                    <div className="text-[10px] text-text-muted">
                      {s.modifications.length} modification{s.modifications.length > 1 ? 's' : ''}
                      {s.created_at && ` — ${s.created_at.slice(0, 10)}`}
                    </div>
                  </button>
                  <button
                    onClick={() => deleteScenarioMutation.mutate({ runId, name: s.name })}
                    className="p-1 text-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right column (unchanged) */}
      <div className="flex-1 min-w-0">
        {simResult ? (
          <ScenarioResultsPanel result={simResult} modifications={modifications} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-3">
              <Info className="w-8 h-8 mx-auto text-text-muted/50" />
              <div>
                <p className="text-text-secondary">No simulation</p>
                <p className="mt-1">Select a model, configure modifications, then run the simulation.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Build frontend and test**

Run: `docker compose up -d --build frontend`

Test in browser:
1. Navigate to Scenarios page
2. Select a model — verify presets grid appears with 6 contextual presets
3. Click a preset — verify modifications pre-fill
4. Add a pumping modification — verify usage selector appears
5. Set a rate above typical range — verify yellow warning appears
6. Set a factor > 5.0 on scale_stress — verify hard limit prevents it
7. Run a simulation — verify results display
8. Save a scenario — verify it appears in "Mes scénarios"
9. Load a saved scenario — verify modifications fill in

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/pastas/ScenariosPage.tsx frontend/src/components/pastas/ScenarioComposer.tsx
git commit -m "feat(pastas): redesign scenarios page with contextual presets and persistence"
```

---

### Task 11: Aquifer Family Detection from Model

**Files:**
- Modify: `frontend/src/pages/pastas/ScenariosPage.tsx`

- [ ] **Step 1: Resolve aquifer family from station preview**

The station preview endpoint already returns `preset` dict with BDLISA info. We need to fetch it when a model is selected and derive the aquifer family.

In `ScenariosPage.tsx`, add a call to the preview endpoint when `code_bss` changes:

```typescript
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

// Inside the component:
const codeBss = selected?.code_bss
const { data: stationPreview } = useQuery({
  queryKey: ['pastas', 'preview', codeBss],
  queryFn: () => api.pastas.preview(codeBss!),
  enabled: !!codeBss,
  staleTime: 30 * 60 * 1000,
})

// Derive aquifer family from station metadata
useEffect(() => {
  if (stationPreview?.metadata) {
    const meta = stationPreview.metadata as Record<string, string>
    // The backend detect_aquifer_family logic, simplified for frontend:
    const nature = meta.nature_eh
    const milieu = meta.milieu_eh
    let family: AquiferFamily = 'sedimentary'
    if (nature === '3') family = 'alluvial'
    else if (nature === '4') family = 'karst'
    else if (nature === '6' || nature === '7') family = 'volcanic'
    else if (nature === '0') family = 'fractured'
    else if (nature === '5' && milieu === '2') family = 'fractured'
    else if (milieu === '5') family = 'alluvial'
    else if (milieu === '3') family = 'karst'
    setAquiferFamily(family)
  }
}, [stationPreview])
```

- [ ] **Step 2: Build and test**

Run: `docker compose up -d --build frontend`
Test: select a model, verify the aquifer family badge appears and presets adapt.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/pastas/ScenariosPage.tsx
git commit -m "feat(pastas): auto-detect aquifer family from station metadata"
```

---

### Task 12: Final Integration Test and Cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run backend tests**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_scenario_presets.py tests/pastas/test_scenario.py -v`
Expected: All tests PASS

- [ ] **Step 2: Build and test full stack**

Run: `docker compose up -d --build`

Manual test checklist:
1. [ ] Scenarios page loads, model picker works
2. [ ] Contextual presets show 6 cards with correct values for detected aquifer
3. [ ] Clicking a preset fills modifications with realistic values
4. [ ] Usage selector on pumping editor pre-fills profile values
5. [ ] Yellow warning appears when rate is outside typical range
6. [ ] Hard limits prevent absurd values in inputs
7. [ ] Simulation runs successfully and shows results
8. [ ] Save button opens dialog, saves scenario
9. [ ] Saved scenario appears in "Mes scénarios" list
10. [ ] Loading a saved scenario fills modifications
11. [ ] Deleting a saved scenario removes it from list
12. [ ] Existing scenarios (no usage field) still work

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat(pastas): complete realistic scenarios with presets, validation, and persistence"
```
