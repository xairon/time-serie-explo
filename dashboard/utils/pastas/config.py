"""Registry of Pastas components and P1 options."""
from __future__ import annotations

import pastas as ps

RECHARGE_REGISTRY: dict[str, type] = {
    "Linear": ps.rch.Linear,
    "FlexModel": ps.rch.FlexModel,
}

RFUNC_REGISTRY: dict[str, type] = {
    "Gamma": ps.Gamma,
    "Exponential": ps.Exponential,
    "Hantush": ps.Hantush,
    "One": ps.One,
}

NOISE_REGISTRY: dict[str, type] = {
    "ArNoiseModel": ps.ArNoiseModel,
}

SOLVER_REGISTRY: dict[str, type] = {
    "LeastSquares": ps.LeastSquares,
    "Lmfit": ps.LmfitSolve,
}


def get_p1_options() -> dict[str, list[str]]:
    """Return the P1-scope options for UI dropdowns."""
    return {
        "recharge": list(RECHARGE_REGISTRY.keys()),
        "response": [k for k in RFUNC_REGISTRY.keys() if k != "One"],
        "noise": list(NOISE_REGISTRY.keys()) + ["none"],
        "solver": list(SOLVER_REGISTRY.keys()),
    }
