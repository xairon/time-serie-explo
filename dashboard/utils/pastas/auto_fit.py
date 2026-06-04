"""Auto-fit grid search engine for Pastas TFN models.

Builds a configuration grid (recharge × response × noise), fits each candidate
via ``run_fit``, assesses quality with STOWA criteria, and ranks results.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

# Note: run_fit, FitResult, assess_stowa, StowaResult are imported lazily inside
# run_auto_fit() so that build_config_grid() can be used without the full ML stack.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline grid: always evaluated regardless of BDLISA preset
# ---------------------------------------------------------------------------

_BASELINE_CONFIGS: list[dict[str, Any]] = [
    {"recharge": "Linear",    "response": "Gamma",       "noise": "ArNoiseModel"},
    {"recharge": "Linear",    "response": "Gamma",       "noise": "none"},
    {"recharge": "Linear",    "response": "Exponential", "noise": "ArNoiseModel"},
    {"recharge": "Linear",    "response": "Exponential", "noise": "none"},
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AutoFitCandidate:
    """Outcome of fitting a single configuration candidate."""

    config: dict[str, Any]
    fit_result: Optional[FitResult] = None
    stowa: Optional[StowaResult] = None
    aic: Optional[float] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0


@dataclass
class AutoFitResult:
    """Aggregated result of the auto-fit grid search."""

    candidates: list[AutoFitCandidate] = field(default_factory=list)
    best: Optional[AutoFitCandidate] = None
    total_elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_config_grid(
    bdlisa_preset: Optional[dict[str, str]] = None,
    add_trend: bool = False,
) -> list[dict[str, Any]]:
    """Build the list of candidate configurations to evaluate.

    Parameters
    ----------
    bdlisa_preset:
        Optional BDLISA-recommended preset dict with keys ``recharge``,
        ``response``, and ``noise``.  If its combination differs from any
        already in the baseline, it is appended.
    add_trend:
        When ``True``, each base config is duplicated with ``add_trend=True``
        *and* ``add_trend=False``, effectively doubling the grid.

    Returns
    -------
    list[dict]
        Each element has keys: ``recharge``, ``response``, ``noise``,
        ``add_trend``.
    """
    # Start from a copy of the baseline (without the add_trend key yet)
    base_configs: list[dict[str, Any]] = [dict(c) for c in _BASELINE_CONFIGS]

    # Optionally add the BDLISA-recommended combo
    if bdlisa_preset is not None:
        preset_key = (
            bdlisa_preset.get("recharge"),
            bdlisa_preset.get("response"),
            bdlisa_preset.get("noise"),
        )
        existing_keys = {
            (c["recharge"], c["response"], c["noise"]) for c in base_configs
        }
        if preset_key not in existing_keys:
            base_configs.append({
                "recharge": bdlisa_preset["recharge"],
                "response": bdlisa_preset["response"],
                "noise": bdlisa_preset["noise"],
            })

    # Expand with add_trend flag
    if add_trend:
        expanded: list[dict[str, Any]] = []
        for cfg in base_configs:
            expanded.append({**cfg, "add_trend": False})
            expanded.append({**cfg, "add_trend": True})
        result = expanded
    else:
        result = [{**cfg, "add_trend": False} for cfg in base_configs]

    # Deduplicate (preserve order)
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for cfg in result:
        key = (cfg["recharge"], cfg["response"], cfg["noise"], cfg["add_trend"])
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)

    return deduped


# ---------------------------------------------------------------------------
# Auto-fit runner
# ---------------------------------------------------------------------------

def run_auto_fit(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    temp: Optional[pd.Series],
    code_bss: str,
    db_url: str,
    bdlisa_preset: Optional[dict[str, str]] = None,
    warm_up_years: int = 2,
    add_trend: bool = False,
    val_split: float = 0.2,
    include_temp: bool = False,
    on_progress: Optional[Callable[[int, int, str, str], None]] = None,
    owner_id: Optional[str] = None,
) -> AutoFitResult:
    """Run the auto-fit grid search.

    For each configuration in ``build_config_grid()``:
    1. Fits the model via :func:`run_fit`.
    2. Loads the Pastas model from MLflow and runs :func:`assess_stowa`.
    3. Records AIC, STOWA result, and timing.

    Ranking:
    - STOWA-passing models sorted by AIC ascending (best fit first).
    - If none pass STOWA, fallback to lowest AIC overall.

    Parameters
    ----------
    gwl:
        Groundwater level time series.
    precip:
        Precipitation stress time series.
    evap:
        Evapotranspiration stress time series.
    temp:
        Optional temperature time series (used when ``include_temp=True``).
    code_bss:
        BSS station code (used as dataset_id for MLflow).
    db_url:
        PostgreSQL connection URL (not used directly here, reserved for
        future stress loading).
    bdlisa_preset:
        Optional BDLISA preset dict to expand the grid.
    warm_up_years:
        Years of warm-up period excluded from metrics.
    add_trend:
        Whether to include trend variants in the grid.
    val_split:
        Fraction of observations reserved for validation (0 = no split).
    include_temp:
        Whether to include temperature as an additional stress.
    on_progress:
        Optional callback ``(current, total, label, status_message)`` called
        after each candidate attempt.

    Returns
    -------
    AutoFitResult
    """
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.fit_service import run_fit, FitResult  # noqa: F401
    from dashboard.utils.pastas.stowa import assess_stowa

    grid = build_config_grid(bdlisa_preset=bdlisa_preset, add_trend=add_trend)
    total = len(grid)
    candidates: list[AutoFitCandidate] = []
    global_start = time.perf_counter()

    # Early overlap / data validation — fail fast instead of repeating
    # the same error for every config in the grid.
    from dashboard.utils.pastas.builder import _validate_series, ValidationError
    try:
        _validate_series(gwl, precip, evap)
    except ValidationError as exc:
        msg = str(exc)
        logger.warning("auto_fit pre-validation failed: %s", msg)
        return AutoFitResult(
            candidates=[AutoFitCandidate(config={}, error=msg, elapsed_s=0.0)],
            best=None,
            total_elapsed_s=time.perf_counter() - global_start,
        )

    for idx, cfg in enumerate(grid):
        recharge = cfg["recharge"]
        response = cfg["response"]
        noise = cfg["noise"]
        trend = cfg["add_trend"]
        label = f"{recharge}/{response}/{noise}" + ("/trend" if trend else "")

        logger.info("auto_fit [%d/%d] %s", idx + 1, total, label)

        if on_progress:
            try:
                on_progress(idx, total, label, "fitting…")
            except Exception:
                pass

        t0 = time.perf_counter()
        candidate = AutoFitCandidate(config=cfg)

        try:
            # Build additional stresses list
            additional_stresses: list[dict[str, Any]] = []
            if include_temp and temp is not None:
                additional_stresses.append({
                    "type": "custom",
                    "name": "temperature",
                    "rfunc": "Gamma",
                    "series": temp,
                })

            fit_result = run_fit(
                gwl=gwl,
                precip=precip,
                evap=evap,
                recharge_type=recharge,
                response_type=response,
                noise_type=noise,
                solver_type="LeastSquares",
                solver_kwargs=None,
                tmin=None,
                tmax=None,
                dataset_id=code_bss,
                name=f"{code_bss}_{recharge}_{response}_{noise}",
                val_split=val_split if val_split > 0 else None,
                additional_stresses=additional_stresses or None,
                warm_up_years=warm_up_years,
                two_pass=(noise != "none"),
                add_trend=trend,
                owner_id=owner_id,
            )
            candidate.fit_result = fit_result

            # Extract AIC from metrics (may be NaN)
            aic = fit_result.metrics.get("aic")
            if aic is not None and not (isinstance(aic, float) and aic != aic):
                candidate.aic = float(aic)

            # Assess STOWA: need the live Pastas model
            try:
                pastas_model = load_model(fit_result.run_id)

                # Compute calibration period length in days
                obs_index = gwl.dropna().index
                if fit_result.cal_period:
                    tmin_cal = pd.Timestamp(fit_result.cal_period[0])
                    tmax_cal = pd.Timestamp(fit_result.cal_period[1])
                else:
                    tmin_cal = obs_index.min()
                    tmax_cal = obs_index.max()
                cal_period_days = float((tmax_cal - tmin_cal).days)

                # Effective tmin skips warm-up
                if warm_up_years > 0:
                    effective_tmin = str(
                        (tmin_cal + pd.DateOffset(years=warm_up_years)).date()
                    )
                else:
                    effective_tmin = str(tmin_cal.date())

                stowa_result = assess_stowa(
                    model=pastas_model,
                    tmin=effective_tmin,
                    tmax=str(tmax_cal.date()),
                    cal_period_days=cal_period_days,
                )
                candidate.stowa = stowa_result
            except Exception as exc:
                logger.warning("STOWA assessment failed for %s: %s", label, exc)
                # stowa stays None — not a fatal error

        except Exception as exc:
            logger.error("Fit failed for %s: %s", label, exc)
            candidate.error = str(exc)

        candidate.elapsed_s = time.perf_counter() - t0
        candidates.append(candidate)

        if on_progress:
            try:
                status = "done" if candidate.error is None else f"error: {candidate.error}"
                on_progress(idx + 1, total, label, status)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    successful = [c for c in candidates if c.error is None and c.fit_result is not None]
    stowa_pass = [c for c in successful if c.stowa is not None and c.stowa.overall_pass]

    def _nse_key(c: AutoFitCandidate) -> float:
        nse = c.fit_result.metrics.get("nse") if c.fit_result else None
        return -(nse if nse is not None else -float("inf"))

    if stowa_pass:
        ranked = sorted(stowa_pass, key=_nse_key)
    else:
        ranked = sorted(successful, key=_nse_key)

    best = ranked[0] if ranked else None

    total_elapsed = time.perf_counter() - global_start
    return AutoFitResult(
        candidates=candidates,
        best=best,
        total_elapsed_s=total_elapsed,
    )
