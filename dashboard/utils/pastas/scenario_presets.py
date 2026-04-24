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


_DISTANCE: dict[AquiferFamily, Range] = {
    "alluvial":    Range(default=500,  typical_min=200,  typical_max=2000, hard_min=10, hard_max=50000),
    "sedimentary": Range(default=1000, typical_min=300,  typical_max=5000, hard_min=10, hard_max=50000),
    "karst":       Range(default=1000, typical_min=100,  typical_max=5000, hard_min=10, hard_max=50000),
    "fractured":   Range(default=300,  typical_min=100,  typical_max=1000, hard_min=10, hard_max=20000),
    "volcanic":    Range(default=500,  typical_min=150,  typical_max=2000, hard_min=10, hard_max=30000),
}

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
