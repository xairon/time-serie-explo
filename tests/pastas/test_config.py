"""Tests for Pastas config registries."""
import pytest


def test_recharge_registry_keys():
    from dashboard.utils.pastas.config import RECHARGE_REGISTRY
    assert "Linear" in RECHARGE_REGISTRY
    assert "FlexModel" in RECHARGE_REGISTRY


def test_rfunc_registry_keys():
    from dashboard.utils.pastas.config import RFUNC_REGISTRY
    for name in ("Gamma", "Exponential", "Hantush", "One"):
        assert name in RFUNC_REGISTRY, f"{name} missing from RFUNC_REGISTRY"


def test_noise_registry_keys():
    from dashboard.utils.pastas.config import NOISE_REGISTRY
    assert "ArNoiseModel" in NOISE_REGISTRY


def test_solver_registry_keys():
    from dashboard.utils.pastas.config import SOLVER_REGISTRY
    assert "LeastSquares" in SOLVER_REGISTRY
    assert "Lmfit" in SOLVER_REGISTRY


def test_registry_values_are_callable():
    from dashboard.utils.pastas.config import (
        RECHARGE_REGISTRY, RFUNC_REGISTRY, NOISE_REGISTRY, SOLVER_REGISTRY,
    )
    for name, cls in RECHARGE_REGISTRY.items():
        assert callable(cls), f"RECHARGE_REGISTRY[{name}] is not callable"
    for name, cls in RFUNC_REGISTRY.items():
        assert callable(cls), f"RFUNC_REGISTRY[{name}] is not callable"
    for name, cls in NOISE_REGISTRY.items():
        assert callable(cls), f"NOISE_REGISTRY[{name}] is not callable"
    for name, cls in SOLVER_REGISTRY.items():
        assert callable(cls), f"SOLVER_REGISTRY[{name}] is not callable"


def test_p1_options():
    """P1 scope: only these options are returned."""
    from dashboard.utils.pastas.config import get_p1_options
    opts = get_p1_options()
    assert set(opts["recharge"]) == {"Linear", "FlexModel"}
    assert set(opts["response"]) == {"Gamma", "Exponential", "Hantush"}
    assert set(opts["noise"]) == {"ArNoiseModel", "none"}
    assert set(opts["solver"]) == {"LeastSquares", "Lmfit"}
