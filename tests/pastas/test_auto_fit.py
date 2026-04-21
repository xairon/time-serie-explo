"""Tests for the auto-fit grid search engine (build_config_grid only)."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# build_config_grid tests
# ---------------------------------------------------------------------------

def test_default_grid_has_at_least_four_entries():
    """Default grid: 2 response functions × 2 noise options = 4 entries minimum."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid()
    assert len(grid) >= 4, f"Expected >= 4 entries, got {len(grid)}"


def test_default_grid_config_keys():
    """Each config dict has exactly the required keys."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    required_keys = {"recharge", "response", "noise", "add_trend"}
    grid = build_config_grid()
    for i, cfg in enumerate(grid):
        assert set(cfg.keys()) == required_keys, (
            f"Config #{i} has unexpected keys: {set(cfg.keys())}"
        )


def test_default_grid_no_duplicates():
    """No duplicate configs in the default grid."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid()
    seen = []
    for cfg in grid:
        key = (cfg["recharge"], cfg["response"], cfg["noise"], cfg["add_trend"])
        assert key not in seen, f"Duplicate config: {cfg}"
        seen.append(key)


def test_default_grid_covers_baseline():
    """Default grid always includes Linear + Gamma + ArNoiseModel and Linear + Exponential + ArNoiseModel."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid()
    configs = [
        (c["recharge"], c["response"], c["noise"]) for c in grid if not c["add_trend"]
    ]
    assert ("Linear", "Gamma", "ArNoiseModel") in configs, (
        "Missing baseline: Linear + Gamma + ArNoiseModel"
    )
    assert ("Linear", "Exponential", "ArNoiseModel") in configs, (
        "Missing baseline: Linear + Exponential + ArNoiseModel"
    )


def test_default_grid_covers_no_noise():
    """Default grid always includes at least one no-noise variant."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid()
    no_noise = [c for c in grid if c["noise"] == "none" and not c["add_trend"]]
    assert len(no_noise) >= 1, "Expected at least one no-noise config"


def test_add_trend_doubles_entries():
    """add_trend=True doubles the entries (each config exists with and without trend)."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid_no_trend = build_config_grid(add_trend=False)
    grid_with_trend = build_config_grid(add_trend=True)

    # Each no-trend config should have a matching with-trend counterpart
    assert len(grid_with_trend) == 2 * len(grid_no_trend), (
        f"Expected {2 * len(grid_no_trend)} entries with add_trend=True, "
        f"got {len(grid_with_trend)}"
    )

    # Check that both add_trend=False and add_trend=True variants exist for each base
    no_trend_configs = {
        (c["recharge"], c["response"], c["noise"])
        for c in grid_with_trend if not c["add_trend"]
    }
    with_trend_configs = {
        (c["recharge"], c["response"], c["noise"])
        for c in grid_with_trend if c["add_trend"]
    }
    assert no_trend_configs == with_trend_configs, (
        "Mismatch between no-trend and with-trend config sets"
    )


def test_add_trend_no_duplicates():
    """No duplicates in grid with add_trend=True."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid(add_trend=True)
    seen = []
    for cfg in grid:
        key = (cfg["recharge"], cfg["response"], cfg["noise"], cfg["add_trend"])
        assert key not in seen, f"Duplicate config with trend: {cfg}"
        seen.append(key)


def test_bdlisa_preset_adds_recommended_config():
    """BDLISA preset adds its recommended recharge/response/noise if different from defaults."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    from dashboard.utils.pastas.config import BDLISA_PRESETS

    # Use a preset with non-default settings (karstique: FlexModel + DoubleExponential + ArmaNoiseModel)
    karst_preset = BDLISA_PRESETS["4"]
    grid = build_config_grid(bdlisa_preset=karst_preset)

    # The BDLISA-recommended combo should appear in the grid
    bdlisa_configs = [
        (c["recharge"], c["response"], c["noise"])
        for c in grid if not c["add_trend"]
    ]
    assert (karst_preset["recharge"], karst_preset["response"], karst_preset["noise"]) in bdlisa_configs, (
        f"BDLISA preset config not found in grid: {karst_preset}"
    )


def test_bdlisa_default_preset_no_extra_entries():
    """BDLISA preset that matches defaults does not add duplicate entries."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    from dashboard.utils.pastas.config import DEFAULT_PRESET

    # DEFAULT_PRESET is Linear + Gamma + ArNoiseModel — already in the grid
    grid_no_preset = build_config_grid()
    grid_with_preset = build_config_grid(bdlisa_preset=DEFAULT_PRESET)

    assert len(grid_no_preset) == len(grid_with_preset), (
        "Passing the default preset should not add extra entries "
        f"(no_preset={len(grid_no_preset)}, with_preset={len(grid_with_preset)})"
    )


def test_grid_with_bdlisa_and_trend():
    """Combined: BDLISA preset + add_trend=True doubles all entries."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    from dashboard.utils.pastas.config import BDLISA_PRESETS

    karst_preset = BDLISA_PRESETS["4"]
    grid_base = build_config_grid(bdlisa_preset=karst_preset, add_trend=False)
    grid_trend = build_config_grid(bdlisa_preset=karst_preset, add_trend=True)

    assert len(grid_trend) == 2 * len(grid_base), (
        f"Expected {2 * len(grid_base)} entries, got {len(grid_trend)}"
    )


def test_grid_values_are_valid_strings():
    """All config values are non-empty strings; add_trend is a bool."""
    from dashboard.utils.pastas.auto_fit import build_config_grid
    grid = build_config_grid(add_trend=True)
    for cfg in grid:
        assert isinstance(cfg["recharge"], str) and cfg["recharge"], "recharge must be a non-empty str"
        assert isinstance(cfg["response"], str) and cfg["response"], "response must be a non-empty str"
        assert isinstance(cfg["noise"], str) and cfg["noise"], "noise must be a non-empty str"
        assert isinstance(cfg["add_trend"], bool), "add_trend must be a bool"
