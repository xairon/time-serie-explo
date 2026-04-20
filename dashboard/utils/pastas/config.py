"""Registry of Pastas components, options, and BDLISA-based presets."""
from __future__ import annotations

import pastas as ps

RECHARGE_REGISTRY: dict[str, type] = {
    "Linear": ps.rch.Linear,
    "FlexModel": ps.rch.FlexModel,
    "Berendrecht": ps.rch.Berendrecht,
    "Peterson": ps.rch.Peterson,
}

RFUNC_REGISTRY: dict[str, type] = {
    "Gamma": ps.Gamma,
    "Exponential": ps.Exponential,
    "Hantush": ps.Hantush,
    "DoubleExponential": ps.DoubleExponential,
    "FourParam": ps.FourParam,
    "One": ps.One,
}

NOISE_REGISTRY: dict[str, type] = {
    "ArNoiseModel": ps.ArNoiseModel,
    "ArmaNoiseModel": ps.ArmaNoiseModel,
}

SOLVER_REGISTRY: dict[str, type] = {
    "LeastSquares": ps.LeastSquares,
    "Lmfit": ps.LmfitSolve,
}


def get_options() -> dict[str, list[str]]:
    """Return all available options for UI dropdowns."""
    return {
        "recharge": list(RECHARGE_REGISTRY.keys()),
        "response": [k for k in RFUNC_REGISTRY.keys() if k != "One"],
        "noise": list(NOISE_REGISTRY.keys()) + ["none"],
        "solver": list(SOLVER_REGISTRY.keys()),
    }


# --- BDLISA-based presets ---

BDLISA_PRESETS: dict[str, dict[str, str]] = {
    # Key: "{nature_eh}_{milieu_eh}" or "{nature_eh}" as fallback
    # milieu_eh: 1=poreux, 2=fracturé, 3=karst, 4=double porosité, 5=alluvial, 8=composite

    # Alluvial (nature=3 or milieu=5)
    "3": {"recharge": "Linear", "response": "Exponential", "noise": "ArNoiseModel",
           "label": "Alluvial", "description": "Réponse rapide, système simple"},
    "milieu_5": {"recharge": "Linear", "response": "Exponential", "noise": "ArNoiseModel",
                  "label": "Alluvionnaire", "description": "Réponse rapide, temps de réponse court"},

    # Sédimentaire poreux (nature=5, milieu=1) — sable, grès
    "5_1": {"recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel",
            "label": "Sédimentaire poreux (sable/grès)", "description": "Réponse intermédiaire, Gamma standard"},

    # Sédimentaire double porosité (nature=5, milieu=4) — craie
    "5_4": {"recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel",
            "label": "Sédimentaire double porosité (craie)", "description": "Forte inertie, temps de réponse long"},

    # Sédimentaire fracturé (nature=5, milieu=2)
    "5_2": {"recharge": "FlexModel", "response": "Gamma", "noise": "ArNoiseModel",
            "label": "Sédimentaire fracturé", "description": "Recharge complexe, FlexModel recommandé"},

    # Karstique (nature=4 or milieu=3)
    "4": {"recharge": "FlexModel", "response": "DoubleExponential", "noise": "ArmaNoiseModel",
          "label": "Karstique", "description": "Double porosité conduits/matrice, double exponentielle"},
    "milieu_3": {"recharge": "FlexModel", "response": "DoubleExponential", "noise": "ArmaNoiseModel",
                  "label": "Karstique", "description": "Deux temps de réponse (conduits + matrice)"},

    # Volcanique (nature=6)
    "6": {"recharge": "FlexModel", "response": "Gamma", "noise": "ArNoiseModel",
          "label": "Volcanique", "description": "Système hétérogène, FlexModel recommandé"},

    # Socle (nature=0) / Montagne (nature=7)
    "0": {"recharge": "FlexModel", "response": "Gamma", "noise": "ArNoiseModel",
          "label": "Socle", "description": "Fracturé, recharge non-linéaire"},
    "7": {"recharge": "FlexModel", "response": "Gamma", "noise": "ArNoiseModel",
          "label": "Montagne", "description": "Système complexe, recharge variable"},

    # Composite (milieu=8)
    "milieu_8": {"recharge": "FlexModel", "response": "FourParam", "noise": "ArmaNoiseModel",
                  "label": "Composite", "description": "Milieu hétérogène, modèle flexible"},
}

# Default when no BDLISA info
DEFAULT_PRESET = {
    "recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel",
    "label": "Standard", "description": "Configuration par défaut, adaptée à la majorité des nappes",
}


def get_preset(nature_eh: str | None, milieu_eh: str | None) -> dict[str, str]:
    """Get recommended Pastas config based on BDLISA classification."""
    if nature_eh and milieu_eh:
        key = f"{nature_eh}_{milieu_eh}"
        if key in BDLISA_PRESETS:
            return BDLISA_PRESETS[key]

    if nature_eh and nature_eh in BDLISA_PRESETS:
        return BDLISA_PRESETS[nature_eh]

    if milieu_eh:
        key = f"milieu_{milieu_eh}"
        if key in BDLISA_PRESETS:
            return BDLISA_PRESETS[key]

    return DEFAULT_PRESET
