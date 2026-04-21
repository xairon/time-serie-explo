"""STOWA 4-criteria quality assessment for Pastas TFN models.

Reference: STOWA (2012) - Beoordeling van tijdreeksmodellen voor
grondwaterbeheer. Dutch standard for TFN model acceptance.

The four criteria are:
1. EVP (Explained Variance Percentage) >= threshold (default 70%)
2. Randomness of noise/residuals (Wald-Wolfowitz runs test, p > alpha)
3. Step response time t95 < 50% of the calibration period
4. Gain parameter is statistically significant (|optimal/stderr| > 1.96)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StowaResult:
    """Assessment outcome for the four STOWA TFN quality criteria."""

    # Criterion 1: Explained Variance Percentage
    evp_pass: bool
    evp_value: float

    # Criterion 2: Autocorrelation / randomness of noise
    autocorrelation_pass: bool
    runs_test_pvalue: float

    # Criterion 3: Step-response time t95
    t95_pass: bool
    t95_days: float
    t95_threshold: float

    # Criterion 4: Gain parameter significance
    gain_pass: bool
    gain_significance: float  # |optimal / stderr|

    # Overall verdict (all 4 must pass)
    overall_pass: bool

    # Fix suggestions for failed criteria
    suggestions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _runs_test_pvalue(series: pd.Series) -> float:
    """Wald-Wolfowitz runs test for randomness.

    Tests whether the sequence of values above/below the median is random.
    A low p-value indicates systematic structure (autocorrelation, trend, etc.).

    Parameters
    ----------
    series:
        1-D time series (may contain NaN values, which are dropped).

    Returns
    -------
    float
        Two-sided p-value from the normal approximation.
    """
    clean = series.dropna().values
    n = len(clean)

    if n < 10:
        # Not enough data for a meaningful test; return 1.0 (do not reject)
        return 1.0

    median = np.median(clean)
    # Encode above median as 1, below as 0; values equal to median are excluded
    above = clean > median
    below = clean < median
    mask = above | below
    coded = above[mask].astype(int)
    n_eff = len(coded)

    if n_eff < 2:
        return 1.0

    n1 = int(coded.sum())       # count above median
    n2 = n_eff - n1             # count below median

    if n1 == 0 or n2 == 0:
        # All values on one side → only 1 run, clearly non-random, return ~0
        return 0.0

    # Count runs: a run is a consecutive subsequence of identical values
    runs = 1 + int(np.abs(np.diff(coded)).sum())

    # Expected number of runs and variance under H0
    expected_runs = (2 * n1 * n2) / n_eff + 1
    variance_runs = (
        2 * n1 * n2 * (2 * n1 * n2 - n_eff)
        / (n_eff ** 2 * (n_eff - 1))
    )

    if variance_runs <= 0:
        return 1.0

    z = (runs - expected_runs) / np.sqrt(variance_runs)
    # Two-sided p-value
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return p_value


def _compute_t95(model, sm_name: str) -> float:
    """Compute the number of days for the step response to reach 95% of final.

    Parameters
    ----------
    model:
        Fitted Pastas model.
    sm_name:
        Name of the stress model (stressmodel) to use.

    Returns
    -------
    float
        Index value (days) where |step| first reaches 95% of |final_value|.
        If the threshold is never reached, returns the last index value.
    """
    step = model.get_step_response(sm_name)
    values = step.values
    index = step.index

    if len(values) == 0:
        return 0.0

    final_value = values[-1]
    if final_value == 0:
        return 0.0

    threshold_95 = 0.95 * abs(final_value)
    abs_values = np.abs(values)

    # Find first index where absolute step reaches 95% of final
    mask = abs_values >= threshold_95
    if not mask.any():
        return float(index[-1])

    first_idx = int(np.argmax(mask))
    return float(index[first_idx])


def _find_gain_param(parameters: pd.DataFrame) -> tuple[float, float] | None:
    """Extract the gain parameter (ending in '_A') from the parameters DataFrame.

    Parameters
    ----------
    parameters:
        Pastas model.parameters DataFrame with columns 'optimal' and 'pstderr'.

    Returns
    -------
    tuple[float, float] or None
        (optimal, stderr) for the first gain parameter found, or None.
    """
    for param_name in parameters.index:
        if str(param_name).endswith("_A"):
            optimal = float(parameters.loc[param_name, "optimal"])
            stderr = parameters.loc[param_name, "pstderr"]
            if stderr is None or (isinstance(stderr, float) and np.isnan(stderr)):
                return None
            return optimal, float(stderr)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_stowa(
    model,
    tmin,
    tmax,
    cal_period_days: float,
    evp_threshold: float = 70.0,
    runs_alpha: float = 0.05,
) -> StowaResult:
    """Assess a fitted Pastas model against the four STOWA quality criteria.

    Parameters
    ----------
    model:
        Fitted Pastas ``Model`` instance.
    tmin:
        Start of the evaluation window (passed to ``model.stats.evp``).
    tmax:
        End of the evaluation window (passed to ``model.stats.evp``).
    cal_period_days:
        Length of the calibration period in days.
    evp_threshold:
        Minimum acceptable EVP (default 70%).
    runs_alpha:
        Significance level for the runs test (default 0.05).

    Returns
    -------
    StowaResult
        Per-criterion pass/fail flags and suggestions for any failures.
    """
    suggestions: list[str] = []

    # ------------------------------------------------------------------
    # Criterion 1: EVP
    # ------------------------------------------------------------------
    evp_value = float(model.stats.evp(tmin, tmax))
    evp_pass = evp_value >= evp_threshold
    if not evp_pass:
        suggestions.append(
            f"EVP is {evp_value:.1f}% (threshold {evp_threshold:.1f}%). "
            "Consider adding stresses, extending the calibration period, or "
            "checking data quality."
        )

    # ------------------------------------------------------------------
    # Criterion 2: Randomness of noise (Wald-Wolfowitz runs test)
    # ------------------------------------------------------------------
    try:
        noise_series = model.noise()
    except Exception:
        noise_series = None

    if noise_series is None or len(noise_series.dropna()) < 10:
        # Fallback to residuals if no noise model
        noise_series = model.residuals()

    runs_pvalue = _runs_test_pvalue(noise_series)
    autocorrelation_pass = runs_pvalue > runs_alpha
    if not autocorrelation_pass:
        suggestions.append(
            f"Runs test p-value is {runs_pvalue:.4f} (threshold {runs_alpha}). "
            "Noise/residuals show systematic structure. Consider adding an "
            "ArNoiseModel or additional stress components to capture autocorrelation."
        )

    # ------------------------------------------------------------------
    # Criterion 3: Step-response time t95 < 50% of calibration period
    # ------------------------------------------------------------------
    t95_threshold = cal_period_days / 2.0

    # Find the first stress model name for step response
    try:
        sm_names = list(model.stressmodels.keys())
        sm_name = sm_names[0] if sm_names else "recharge"
    except Exception:
        sm_name = "recharge"

    t95_days = _compute_t95(model, sm_name)
    t95_pass = t95_days < t95_threshold
    if not t95_pass:
        suggestions.append(
            f"Step-response t95 is {t95_days:.1f} days (threshold {t95_threshold:.1f} days = "
            f"50% of calibration period {cal_period_days:.0f} days). "
            "Extend the calibration period or use a response function with faster decay."
        )

    # ------------------------------------------------------------------
    # Criterion 4: Gain parameter statistical significance
    # ------------------------------------------------------------------
    gain_result = _find_gain_param(model.parameters)

    if gain_result is None:
        # No gain parameter found or stderr unavailable — cannot assess
        gain_significance = float("nan")
        gain_pass = False
        suggestions.append(
            "Could not assess gain significance: no parameter ending in '_A' found, "
            "or standard error is unavailable. Ensure the model has been solved."
        )
    else:
        optimal, stderr = gain_result
        if stderr == 0.0:
            gain_significance = float("inf")
            gain_pass = True
        else:
            gain_significance = abs(optimal / stderr)
            gain_pass = gain_significance > 1.96

        if not gain_pass:
            suggestions.append(
                f"Gain parameter significance is {gain_significance:.2f} (threshold 1.96). "
                "The recharge gain (_A) is not statistically significant. "
                "Check the stress data or consider a different recharge formulation."
            )

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    overall_pass = evp_pass and autocorrelation_pass and t95_pass and gain_pass

    return StowaResult(
        evp_pass=evp_pass,
        evp_value=evp_value,
        autocorrelation_pass=autocorrelation_pass,
        runs_test_pvalue=runs_pvalue,
        t95_pass=t95_pass,
        t95_days=t95_days,
        t95_threshold=t95_threshold,
        gain_pass=gain_pass,
        gain_significance=gain_significance,
        overall_pass=overall_pass,
        suggestions=suggestions,
    )
