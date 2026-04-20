"""Apply modifications to a calibrated Pastas model and simulate what-if scenarios."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
import pastas as ps

from dashboard.utils.pastas.io import load_model

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Output of a scenario simulation."""

    baseline: pd.Series
    scenario: pd.Series
    delta: pd.Series
    contributions_baseline: dict[str, pd.Series]
    contributions_scenario: dict[str, pd.Series]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Series generators
# ---------------------------------------------------------------------------

def _generate_pumping_series(
    start: str,
    end: str,
    pattern: str,
    rate_m3d: float,
    period_days: int = 365,
) -> pd.Series:
    """Generate a pumping rate time series Q(t) in m³/d.

    Parameters
    ----------
    start, end:
        ISO date strings for the stress period.
    pattern:
        One of "constant", "seasonal", "pulse".
    rate_m3d:
        Peak (or mean) pumping rate in m³/d.
    period_days:
        Period for seasonal pattern (default 365 days).
    """
    idx = pd.date_range(start=start, end=end, freq="D")
    t = np.arange(len(idx))

    if pattern == "seasonal":
        values = rate_m3d * 0.5 * (1 - np.cos(2 * np.pi * t / period_days))
    elif pattern == "pulse":
        # Single pulse in the middle of the period
        mid = len(idx) // 2
        values = np.zeros(len(idx))
        if len(idx) > 0:
            values[mid] = rate_m3d
    else:  # constant
        values = np.full(len(idx), rate_m3d)

    return pd.Series(values, index=idx, name="pumping")


# ---------------------------------------------------------------------------
# Modification handlers
# ---------------------------------------------------------------------------

def _apply_pumping_synthetic(model: ps.Model, mod: dict[str, Any]) -> None:
    """Add a synthetic pumping stress model.

    Expected mod keys: name, start, end, pattern, rate_m3d, rfunc (optional),
    period_days (optional).
    """
    name = mod.get("name", "pumping")
    start = str(mod["start"])
    end = str(mod["end"])
    pattern = mod.get("pattern", "constant")
    rate_m3d = float(mod["rate_m3d"])
    period_days = int(mod.get("period_days", 365))

    rfunc_name = mod.get("rfunc", "Gamma")
    rfunc_registry = {
        "Gamma": ps.Gamma,
        "Exponential": ps.Exponential,
        "Hantush": ps.Hantush,
        "One": ps.One,
    }
    rfunc_cls = rfunc_registry.get(rfunc_name, ps.Gamma)

    q_series = _generate_pumping_series(start, end, pattern, rate_m3d, period_days)

    # Extend the pumping series with zeros to cover the model's full time range.
    # Pastas raises ValueError if the stress series doesn't cover the simulation window.
    model_tmin = model.get_tmin(use_oseries=True, use_stresses=True)
    model_tmax = model.get_tmax(use_oseries=True, use_stresses=True)
    full_idx = pd.date_range(start=model_tmin, end=model_tmax, freq="D")
    q_full = pd.Series(0.0, index=full_idx, name="pumping")
    # Fill the pumping window
    common = q_series.index.intersection(full_idx)
    q_full.loc[common] = q_series.loc[common]

    # Pastas computes gain_scale_factor from series.std(); if the series is all
    # zeros (rate=0), std=0 and the initial amplitude becomes -inf, causing NaN
    # in simulation. Use gain_scale_factor=1.0 as a safe fallback.
    gsf = float(q_full.std()) if float(q_full.std()) > 0 else 1.0

    sm = ps.StressModel(
        q_full,
        rfunc=rfunc_cls(),
        name=name,
        up=False,  # pumping lowers GWL
        settings="well",
        gain_scale_factor=gsf,
    )
    model.add_stressmodel(sm)


def _apply_pumping_upload(model: ps.Model, mod: dict[str, Any]) -> None:
    """Add a pumping stress model from user-provided CSV rows.

    Expected mod keys: name, csv_rows ([{date, Q_m3d}, ...]), rfunc (optional).
    """
    name = mod.get("name", "pumping_upload")
    csv_rows = mod["csv_rows"]
    rfunc_name = mod.get("rfunc", "Gamma")
    rfunc_registry = {
        "Gamma": ps.Gamma,
        "Exponential": ps.Exponential,
        "Hantush": ps.Hantush,
        "One": ps.One,
    }
    rfunc_cls = rfunc_registry.get(rfunc_name, ps.Gamma)

    dates = pd.to_datetime([row["date"] for row in csv_rows])
    values = [float(row["Q_m3d"]) for row in csv_rows]
    q_series = pd.Series(values, index=dates, name=name).sort_index()
    q_series.index.freq = pd.infer_freq(q_series.index)

    gsf = float(q_series.std()) if float(q_series.std()) > 0 else 1.0
    sm = ps.StressModel(
        q_series,
        rfunc=rfunc_cls(),
        name=name,
        up=False,
        settings="well",
        gain_scale_factor=gsf,
    )
    model.add_stressmodel(sm)


def _apply_linear_trend(model: ps.Model, mod: dict[str, Any]) -> None:
    """Add a LinearTrend stressmodel.

    Expected mod keys: start, end, name (optional).
    """
    start = str(mod["start"])
    end = str(mod["end"])
    name = mod.get("name", "trend")

    lt = ps.LinearTrend(start=start, end=end, name=name)
    model.add_stressmodel(lt)

    # Set initial slope if provided.
    # LinearTrend parameter is named "{name}_a" (m/day), so convert m/year → m/day.
    slope = mod.get("slope")
    if slope is not None:
        param_name = f"{name}_a"
        slope_per_day = float(slope) / 365.25
        if param_name in model.parameters.index:
            model.set_parameter(param_name, initial=slope_per_day)


def _apply_scale_stress(model: ps.Model, mod: dict[str, Any]) -> None:
    """Scale precip or evap series in a RechargeModel.

    Expected mod keys: stress_index (0=precip, 1=evap), factor, start (optional), end (optional).
    """
    stress_index = int(mod.get("stress_index", 0))
    factor = float(mod["factor"])
    start = mod.get("start")
    end = mod.get("end")

    # Find the recharge stress model
    rm_name = mod.get("stressmodel_name", "recharge")
    if rm_name not in model.stressmodels:
        # Fall back to first RechargeModel
        for name, sm in model.stressmodels.items():
            if isinstance(sm, ps.RechargeModel):
                rm_name = name
                break
        else:
            raise ValueError("No RechargeModel found in model stressmodels.")

    rm: ps.RechargeModel = model.stressmodels[rm_name]

    # Extract calibrated parameter values before deleting
    params_before = model.parameters.copy()

    # Get the two stress series
    series_list = [ts.series.copy() for ts in rm.stress]

    # Scale the target series within the window
    target = series_list[stress_index].copy()
    if start is not None or end is not None:
        mask = pd.Series(True, index=target.index)
        if start is not None:
            mask = mask & (target.index >= pd.Timestamp(start))
        if end is not None:
            mask = mask & (target.index <= pd.Timestamp(end))
        target[mask] *= factor
    else:
        target *= factor
    series_list[stress_index] = target

    # Preserve rfunc class and recharge class
    rfunc_cls = type(rm.rfunc)
    recharge_cls = type(rm.recharge)

    # Delete old recharge model and add new one with scaled series
    model.del_stressmodel(rm_name)

    new_rm = ps.RechargeModel(
        series_list[0],
        series_list[1],
        rfunc=rfunc_cls(),
        recharge=recharge_cls(),
        name=rm_name,
    )
    model.add_stressmodel(new_rm)

    # Restore calibrated parameter values
    for param_name in model.parameters.index:
        if param_name in params_before.index:
            row = params_before.loc[param_name]
            optimal = row.get("optimal")
            if pd.notna(optimal):
                try:
                    model.set_parameter(param_name, initial=float(optimal), optimal=float(optimal))
                except Exception:
                    try:
                        model.set_parameter(param_name, initial=float(optimal))
                    except Exception:
                        logger.debug("Could not restore parameter %s", param_name)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    "pumping_synthetic": _apply_pumping_synthetic,
    "pumping_upload": _apply_pumping_upload,
    "linear_trend": _apply_linear_trend,
    "scale_stress": _apply_scale_stress,
}


def apply_modification(model: ps.Model, mod: dict[str, Any]) -> None:
    """Apply a single modification to a Pastas model in-place.

    Parameters
    ----------
    model:
        A calibrated Pastas model (will be mutated).
    mod:
        Modification dict with at least a "type" key.
    """
    mod_type = mod["type"]
    handler = _HANDLERS.get(mod_type)
    if handler is None:
        raise ValueError(
            f"Unknown modification type '{mod_type}'. "
            f"Available: {list(_HANDLERS.keys())}"
        )
    handler(model, mod)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def simulate_scenario(
    run_id: str,
    tmin: Optional[str],
    tmax: Optional[str],
    modifications: list[dict[str, Any]],
) -> ScenarioResult:
    """Load a calibrated model, apply modifications, and simulate.

    Parameters
    ----------
    run_id:
        MLflow run ID of the calibrated Pastas model.
    tmin, tmax:
        Simulation window (ISO date strings or None for model defaults).
    modifications:
        Ordered list of modification dicts. Each must have a "type" key.

    Returns
    -------
    ScenarioResult
        Contains baseline, scenario, delta, and per-stress contributions.
    """
    warnings: list[str] = []

    # Load original model (cached)
    original = load_model(run_id)

    # --- Baseline simulation ---
    try:
        baseline = original.simulate(tmin=tmin, tmax=tmax)
    except Exception as exc:
        warnings.append(f"Baseline simulation warning: {exc}")
        baseline = original.simulate()

    contributions_baseline: dict[str, pd.Series] = {}
    for sm_name in original.stressmodels:
        try:
            contributions_baseline[sm_name] = original.get_contribution(
                sm_name, tmin=tmin, tmax=tmax
            )
        except Exception as exc:
            logger.debug("Could not compute baseline contribution for %s: %s", sm_name, exc)

    # --- Scenario: deep-copy and apply modifications ---
    scenario_model = copy.deepcopy(original)

    for i, mod in enumerate(modifications):
        try:
            apply_modification(scenario_model, mod)
        except Exception as exc:
            warnings.append(f"Modification #{i} ({mod.get('type', '?')}) failed: {exc}")
            logger.warning("Modification %d failed: %s", i, exc, exc_info=True)

    try:
        scenario = scenario_model.simulate(tmin=tmin, tmax=tmax)
    except Exception as exc:
        warnings.append(f"Scenario simulation warning: {exc}")
        scenario = scenario_model.simulate()

    contributions_scenario: dict[str, pd.Series] = {}
    for sm_name in scenario_model.stressmodels:
        try:
            contributions_scenario[sm_name] = scenario_model.get_contribution(
                sm_name, tmin=tmin, tmax=tmax
            )
        except Exception as exc:
            logger.debug("Could not compute scenario contribution for %s: %s", sm_name, exc)

    # Align index for delta computation
    common_idx = baseline.index.intersection(scenario.index)
    delta = scenario.loc[common_idx] - baseline.loc[common_idx]

    return ScenarioResult(
        baseline=baseline,
        scenario=scenario,
        delta=delta,
        contributions_baseline=contributions_baseline,
        contributions_scenario=contributions_scenario,
        warnings=warnings,
    )
