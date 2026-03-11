"""Shared constants for the Counterfactual Analysis module.

Reusable across the CF page, visualization, and export.
"""

from __future__ import annotations

from typing import Any, Optional

from .perturbation import PerturbationLayer

# IPS class colors for charts (BRGM-inspired color scale)
IPS_COLORS: dict[str, str] = {
    "very_low":        "#8B0000",
    "low":             "#DC143C",
    "moderately_low":  "#FF8C00",
    "normal":          "#228B22",
    "moderately_high": "#4169E1",
    "high":            "#0000CD",
    "very_high":       "#00008B",
}

# French labels for IPS classes
IPS_LABELS: dict[str, str] = {
    "very_low":        "Tres bas",
    "low":             "Bas",
    "moderately_low":  "Moderement bas",
    "normal":          "Normal",
    "moderately_high": "Moderement haut",
    "high":            "Haut",
    "very_high":       "Tres haut",
}

# CF method display colors
CF_METHOD_COLORS: dict[str, str] = {
    "PhysCF (gradient)": "#FF6B35",
    "PhysCF (Optuna)":   "#9B59B6",
    "CoMTE":             "#2ECC71",
}

# Month -> season name mapping
MONTH_TO_SEASON_NAME: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

SEASON_NAMES_FR: dict[str, str] = {
    "DJF": "Hiver",
    "MAM": "Printemps",
    "JJA": "Ete",
    "SON": "Automne",
}

# PhysCF parameter UI metadata (labels, explanations)
_PARAM_UI: dict[str, dict[str, str]] = {
    "s_P_DJF": {
        "label": "Precipitation Hiver (Dec-Fev)",
        "label_short": "P Hiver",
        "explanation": (
            "Facteur multiplicatif sur les precipitations hivernales. "
            "s_P=0.7 signifie -30% de pluie en hiver. La recharge hivernale "
            "est critique pour les nappes inertielles (craie, calcaire)."
        ),
    },
    "s_P_MAM": {
        "label": "Precipitation Printemps (Mar-Mai)",
        "label_short": "P Printemps",
        "explanation": (
            "Facteur sur les precipitations printanieres. Le printemps assure "
            "la transition entre hautes et basses eaux."
        ),
    },
    "s_P_JJA": {
        "label": "Precipitation Ete (Jun-Aout)",
        "label_short": "P Ete",
        "explanation": (
            "Facteur sur les precipitations estivales. Moins de pluie d'ete "
            "accelere le rabattement car l'ETP est maximale."
        ),
    },
    "s_P_SON": {
        "label": "Precipitation Automne (Sep-Nov)",
        "label_short": "P Automne",
        "explanation": (
            "Facteur sur les precipitations automnales. L'automne amorce "
            "le cycle annuel de recharge."
        ),
    },
    "delta_T": {
        "label": "Offset Temperature",
        "label_short": "Temperature",
        "explanation": (
            "Offset global de temperature. +2C simule un rechauffement modere. "
            "Impacte l'ETP via Clausius-Clapeyron: +7% d'evaporation par degC."
        ),
    },
    "delta_s": {
        "label": "Decalage Temporel",
        "label_short": "Decalage",
        "explanation": (
            "Decale conjointement P et T dans le temps. +15j simule un retard "
            "saisonnier. L'ETP n'est PAS decalee (suit via CC)."
        ),
    },
    "delta_etp": {
        "label": "Residuel ETP",
        "label_short": "Residuel ETP",
        "explanation": (
            "Perturbation additive de l'ETP au-dela du couplage CC. "
            "Volontairement petit pour forcer la conformite physique."
        ),
    },
}

# Build PHYSCF_PARAM_INFO from the single source of truth (PerturbationLayer.PARAM_RANGES)
PHYSCF_PARAM_INFO: dict[str, dict[str, Any]] = {}
for _key, _ranges in PerturbationLayer.PARAM_RANGES.items():
    PHYSCF_PARAM_INFO[_key] = {
        **_PARAM_UI[_key],
        "range_min": _ranges["min"],
        "range_max": _ranges["max"],
        "identity": _ranges["identity"],
        "unit": _ranges["unit"],
    }

STRESS_COLUMNS: list[str] = PerturbationLayer.STRESS_COLUMNS
CONVERGENCE_THRESHOLD: float = PerturbationLayer.CONVERGENCE_THRESHOLD
IPS_REFERENCE_PERIOD: tuple[str, str] = ("1981", "2010")

# Preset scenarios for quick exploration
PRESET_SCENARIOS: dict[str, Optional[dict[str, str]]] = {
    "Personnalise": None,
    "Secheresse moderee (P-20%, T+1.5C)": {
        "description": "Episode sec decennal avec rechauffement leger",
        "suggested_from": "normal",
        "suggested_to": "moderately_low",
    },
    "Secheresse severe (P-40%, T+3C)": {
        "description": "Secheresse exceptionnelle multi-saisons",
        "suggested_from": "normal",
        "suggested_to": "low",
    },
    "Recharge hivernale abondante (P_DJF+50%)": {
        "description": "Hiver tres pluvieux rechargeant la nappe",
        "suggested_from": "moderately_low",
        "suggested_to": "normal",
    },
    "Changement climatique 2050 (T+2C, P-10%)": {
        "description": "Scenario inspire RCP4.5 France metropolitaine mi-siecle",
        "suggested_from": "normal",
        "suggested_to": "moderately_low",
    },
}
