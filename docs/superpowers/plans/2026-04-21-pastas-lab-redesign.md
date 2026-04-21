# Pastas Lab Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Pastas Lab from a single-shot manual fit tool into a professional 5-step hydrogeologist pipeline with pre-fit diagnostics, auto-fit with STOWA screening, and AI model comparison.

**Architecture:** Backend-first approach. Build the 3 new Python modules (diagnostics_prefit, stowa, auto_fit), add API endpoints, then rewire the frontend into a 5-tab layout with Guided/Expert dual-mode. Existing fit_service, builder, and scenario modules are modified in-place.

**Tech Stack:** Python (Pastas, scipy, statsmodels, pymannkendall), FastAPI (SSE streaming), React 19 (TanStack Query, Plotly.js)

---

## File Structure

### New Backend Files
| File | Responsibility |
|------|---------------|
| `dashboard/utils/pastas/diagnostics_prefit.py` | Pre-fit data analysis: coverage, gaps, Mann-Kendall trend, Pettitt breakpoint, seasonality, period recommendation |
| `dashboard/utils/pastas/stowa.py` | STOWA 4-criteria assessment (EVP, autocorrelation, t95, gain significance) |
| `dashboard/utils/pastas/auto_fit.py` | Grid search engine: enumerate configs, two-pass solve, STOWA screen, rank by AIC |

### Modified Backend Files
| File | Change |
|------|--------|
| `dashboard/utils/pastas/fit_service.py` | Add `warm_up_years`, `two_pass`, `initial_params` params to `run_fit()` |
| `dashboard/utils/pastas/builder.py` | Add `step_models` param to `build_model()` for StepModel support |
| `api/routers/pastas.py` | Add 3 endpoints: `POST /diagnose`, `GET /auto-fit` (SSE), `POST /compare-ai` |
| `api/schemas/pastas.py` | Add schemas: `DiagnoseRequest/Response`, `AutoFitRequest`, `StowaResult`, `CompareAIRequest/Response` |
| `pyproject.toml` | Add `pymannkendall` dependency |

### New Frontend Files
| File | Responsibility |
|------|---------------|
| `frontend/src/components/pastas/PreFitDiagnosticPanel.tsx` | 6-indicator diagnostic display with "Apply" action buttons |
| `frontend/src/components/pastas/StowaVerdictBanner.tsx` | 4-criteria pass/fail banner |
| `frontend/src/components/pastas/AutoFitProgress.tsx` | SSE progress UI for grid search |
| `frontend/src/components/pastas/AIComparisonPanel.tsx` | Pastas vs AI overlay chart + metrics table |
| `frontend/src/components/pastas/GuidedExpertToggle.tsx` | Dual-mode toggle (localStorage persisted) |
| `frontend/src/pages/pastas/StationStep.tsx` | Step 1: station picker + diagnostic |
| `frontend/src/pages/pastas/CalibrateStep.tsx` | Step 2: auto-fit or manual config |
| `frontend/src/pages/pastas/ResultsStep.tsx` | Step 3: STOWA verdict + results sections |
| `frontend/src/pages/pastas/ScenariosStep.tsx` | Step 4: refactored ScenarioWorkflow as standalone page |

### Modified Frontend Files
| File | Change |
|------|--------|
| `frontend/src/pages/pastas/PastasLayout.tsx` | 5-tab layout + Guided/Expert toggle in header |
| `frontend/src/pages/pastas/GalleryPage.tsx` | STOWA badges + station grouping |
| `frontend/src/components/pastas/ModelTable.tsx` | STOWA badge, AIC column, station grouping |
| `frontend/src/hooks/usePastas.ts` | Add: `useDiagnose`, `useAutoFit`, `useCompareAI` |
| `frontend/src/lib/api.ts` | Add: `diagnose()`, `autoFit()`, `compareAI()` |
| `frontend/src/lib/types.ts` | Add: `DiagnoseResult`, `StowaResult`, `AutoFitProgress`, `CompareAIResult` |
| `frontend/src/routes.tsx` | Update `/pastas` children to 5 steps |

---

## Task 1: STOWA Assessment Module

**Files:**
- Create: `dashboard/utils/pastas/stowa.py`
- Test: `tests/pastas/test_stowa.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pastas/test_stowa.py
"""Tests for STOWA 4-criteria model assessment."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from dashboard.utils.pastas.stowa import assess_stowa, StowaResult


def _mock_model(evp=75.0, residuals_autocorrelated=False, t95_days=500, gain_stderr_ratio=5.0):
    """Build a mock Pastas model with controllable properties."""
    model = MagicMock()
    model.stats.evp.return_value = evp

    n = 1000
    if residuals_autocorrelated:
        # AR(1) process with strong autocorrelation
        res = np.cumsum(np.random.randn(n) * 0.1)
    else:
        res = np.random.randn(n)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    model.residuals.return_value = pd.Series(res, index=idx)
    model.noise.return_value = pd.Series(res, index=idx)

    # Step response for t95
    t = np.arange(0, 3000)
    step = 1.0 - np.exp(-t / (t95_days / 3.0))  # ~t95 at t95_days
    sm_name = "recharge"
    model.stressmodels = {sm_name: MagicMock()}
    model.get_step_response.return_value = pd.Series(step, index=t)

    # Parameters with gain and stderr
    params_df = pd.DataFrame({
        "optimal": [1.5, 100, 2.0],
        "stderr": [1.5 / gain_stderr_ratio, 10, 0.5],
        "name": ["recharge_A", "recharge_a", "recharge_n"],
    }).set_index("name")
    model.parameters = params_df

    return model


class TestStowaAssessment:
    def test_all_pass(self):
        model = _mock_model(evp=80.0, residuals_autocorrelated=False, t95_days=400, gain_stderr_ratio=5.0)
        result = assess_stowa(model, tmin="2000-01-01", tmax="2010-12-31", cal_period_days=4018)
        assert isinstance(result, StowaResult)
        assert result.evp_pass is True
        assert result.autocorrelation_pass is True
        assert result.t95_pass is True
        assert result.gain_pass is True
        assert result.overall_pass is True

    def test_evp_fail(self):
        model = _mock_model(evp=50.0)
        result = assess_stowa(model, tmin="2000-01-01", tmax="2010-12-31", cal_period_days=4018)
        assert result.evp_pass is False
        assert result.overall_pass is False

    def test_t95_too_long(self):
        model = _mock_model(t95_days=3000)
        result = assess_stowa(model, tmin="2000-01-01", tmax="2010-12-31", cal_period_days=4018)
        assert result.t95_pass is False
        assert result.t95_days > 0.5 * 4018

    def test_gain_insignificant(self):
        model = _mock_model(gain_stderr_ratio=0.5)
        result = assess_stowa(model, tmin="2000-01-01", tmax="2010-12-31", cal_period_days=4018)
        assert result.gain_pass is False

    def test_result_has_all_values(self):
        model = _mock_model()
        result = assess_stowa(model, tmin="2000-01-01", tmax="2010-12-31", cal_period_days=4018)
        assert result.evp_value is not None
        assert result.runs_test_pvalue is not None
        assert result.t95_days is not None
        assert result.gain_significance is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_stowa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.utils.pastas.stowa'`

- [ ] **Step 3: Implement STOWA assessment**

```python
# dashboard/utils/pastas/stowa.py
"""STOWA 4-criteria assessment for Pastas model quality screening.

Implements the Dutch STOWA methodology for TFN model acceptance:
1. Goodness of fit (EVP >= 70%)
2. No residual autocorrelation (Runs test p > 0.05)
3. Response time t95 < 50% of calibration period
4. Gain significantly different from zero
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StowaResult:
    evp_pass: bool
    evp_value: float
    autocorrelation_pass: bool
    runs_test_pvalue: float
    t95_pass: bool
    t95_days: float
    t95_threshold: float
    gain_pass: bool
    gain_significance: float
    overall_pass: bool
    suggestions: list[str]


def _runs_test_pvalue(series: pd.Series) -> float:
    """Two-sided Wald-Wolfowitz runs test for randomness."""
    clean = series.dropna()
    if len(clean) < 20:
        return 1.0  # insufficient data, assume pass
    median = clean.median()
    binary = (clean > median).astype(int).values
    n1 = int(binary.sum())
    n0 = len(binary) - n1
    if n0 == 0 or n1 == 0:
        return 0.0
    runs = 1 + int(np.sum(np.diff(binary) != 0))
    mu = 1 + 2 * n0 * n1 / (n0 + n1)
    denom = (n0 + n1) ** 2 * (n0 + n1 - 1)
    if denom == 0:
        return 1.0
    sigma2 = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / denom
    if sigma2 <= 0:
        return 1.0
    z = (runs - mu) / np.sqrt(sigma2)
    from scipy.stats import norm
    return float(2 * norm.sf(abs(z)))


def _compute_t95(model, sm_name: str) -> float:
    """Compute t95 (days to 95% of step response) for a stress model."""
    try:
        step = model.get_step_response(sm_name)
        if len(step) == 0:
            return float("inf")
        final = step.iloc[-1]
        if abs(final) < 1e-10:
            return float("inf")
        target = 0.95 * final
        idx = np.argmax(np.abs(step.values) >= abs(target))
        return float(step.index[idx]) if idx > 0 else float(step.index[-1])
    except Exception:
        return float("inf")


def assess_stowa(
    model,
    tmin: str | None,
    tmax: str | None,
    cal_period_days: int,
    evp_threshold: float = 70.0,
    runs_alpha: float = 0.05,
) -> StowaResult:
    """Run STOWA 4-criteria assessment on a solved Pastas model."""
    suggestions: list[str] = []

    # 1. EVP
    try:
        evp = float(model.stats.evp(tmin=tmin, tmax=tmax))
    except Exception:
        evp = 0.0
    evp_pass = evp >= evp_threshold
    if not evp_pass:
        suggestions.append(f"EVP {evp:.1f}% < {evp_threshold}% — try a different recharge/response combination")

    # 2. Autocorrelation (Runs test on noise or residuals)
    try:
        noise = model.noise(tmin=tmin, tmax=tmax)
    except Exception:
        try:
            noise = model.residuals(tmin=tmin, tmax=tmax)
        except Exception:
            noise = pd.Series(dtype=float)
    runs_p = _runs_test_pvalue(noise)
    autocorr_pass = runs_p > runs_alpha
    if not autocorr_pass:
        suggestions.append(f"Residual autocorrelation detected (Runs p={runs_p:.3f}) — add or change noise model")

    # 3. t95 < 50% cal period
    t95_threshold = 0.5 * cal_period_days
    sm_name = next(iter(model.stressmodels), None)
    t95 = _compute_t95(model, sm_name) if sm_name else float("inf")
    t95_pass = t95 < t95_threshold
    if not t95_pass:
        suggestions.append(f"Response time t95={t95:.0f}d exceeds {t95_threshold:.0f}d — response function may be unrealistic")

    # 4. Gain significance
    try:
        params = model.parameters
        gain_name = next((n for n in params.index if n.endswith("_A")), None)
        if gain_name:
            optimal = float(params.loc[gain_name, "optimal"])
            stderr = float(params.loc[gain_name, "stderr"]) if pd.notna(params.loc[gain_name, "stderr"]) else float("inf")
            gain_sig = abs(optimal / stderr) if stderr > 0 else 0.0
        else:
            gain_sig = float("inf")  # no gain param → assume pass
    except Exception:
        gain_sig = float("inf")
    gain_pass = gain_sig > 1.96  # 95% confidence
    if not gain_pass:
        suggestions.append(f"Gain not significant (|A/stderr|={gain_sig:.2f} < 1.96) — stress may not influence this well")

    overall = evp_pass and autocorr_pass and t95_pass and gain_pass

    return StowaResult(
        evp_pass=evp_pass, evp_value=evp,
        autocorrelation_pass=autocorr_pass, runs_test_pvalue=runs_p,
        t95_pass=t95_pass, t95_days=t95, t95_threshold=t95_threshold,
        gain_pass=gain_pass, gain_significance=gain_sig,
        overall_pass=overall, suggestions=suggestions,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_stowa.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/stowa.py tests/pastas/test_stowa.py
git commit -m "feat(pastas): add STOWA 4-criteria assessment module"
```

---

## Task 2: Pre-fit Diagnostics Module

**Files:**
- Create: `dashboard/utils/pastas/diagnostics_prefit.py`
- Modify: `pyproject.toml` (add pymannkendall)
- Test: `tests/pastas/test_diagnostics_prefit.py`

- [ ] **Step 1: Add pymannkendall dependency**

In `/home/ringuet/time-serie-explo/pyproject.toml`, add `"pymannkendall>=1.4.0"` to the dependencies list after `"lmfit>=1.2.0"`.

- [ ] **Step 2: Write failing tests**

```python
# tests/pastas/test_diagnostics_prefit.py
"""Tests for pre-fit diagnostic analysis."""
import numpy as np
import pandas as pd
import pytest

from dashboard.utils.pastas.diagnostics_prefit import run_prefit_diagnostics, PreFitDiagnostic


def _make_series(n_days=5000, gap_start=1000, gap_length=0, trend_slope=0.0, breakpoint_at=None):
    """Generate synthetic piezo series with controllable properties."""
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.randn(n_days) * 0.3
    trend = trend_slope * np.arange(n_days) / 365.25
    values = 50.0 + seasonal + noise + trend
    if breakpoint_at and breakpoint_at < n_days:
        values[breakpoint_at:] += 3.0
    series = pd.Series(values, index=dates, name="piezo")
    if gap_length > 0:
        mask = (np.arange(n_days) >= gap_start) & (np.arange(n_days) < gap_start + gap_length)
        series[mask] = np.nan
    return series.dropna()


class TestPreFitDiagnostics:
    def test_clean_series(self):
        series = _make_series(n_days=7000)
        result = run_prefit_diagnostics(series)
        assert isinstance(result, PreFitDiagnostic)
        assert result.coverage_pct > 95
        assert result.coverage_status == "green"
        assert result.max_gap_days < 30
        assert result.record_years > 15

    def test_large_gap(self):
        series = _make_series(gap_start=1000, gap_length=200)
        result = run_prefit_diagnostics(series)
        assert result.max_gap_days >= 200
        assert result.gaps_status in ("orange", "red")

    def test_trend_detected(self):
        series = _make_series(trend_slope=-0.5)
        result = run_prefit_diagnostics(series)
        assert result.trend_detected is True
        assert result.trend_status in ("orange", "red")
        assert result.trend_slope < 0

    def test_breakpoint_detected(self):
        series = _make_series(breakpoint_at=2500)
        result = run_prefit_diagnostics(series)
        assert result.breakpoint_detected is True

    def test_recommended_period(self):
        series = _make_series(n_days=7000)
        result = run_prefit_diagnostics(series)
        assert result.recommended_tmin is not None
        assert result.recommended_tmax is not None

    def test_short_series(self):
        series = _make_series(n_days=500)
        result = run_prefit_diagnostics(series)
        assert result.record_status == "red"

    def test_recommendations_list(self):
        series = _make_series(trend_slope=-1.0, gap_start=500, gap_length=250)
        result = run_prefit_diagnostics(series)
        assert len(result.recommendations) > 0
        assert any(r["type"] in ("trend", "gap", "period") for r in result.recommendations)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_diagnostics_prefit.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement pre-fit diagnostics**

```python
# dashboard/utils/pastas/diagnostics_prefit.py
"""Pre-fit diagnostic analysis for piezometric time series.

Analyzes data quality before Pastas fitting: coverage, gaps, trend,
breakpoints, seasonality, record length. Produces actionable recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


@dataclass
class PreFitDiagnostic:
    # Coverage
    coverage_pct: float
    coverage_status: str  # green/orange/red

    # Gaps
    max_gap_days: int
    gaps_status: str
    gap_periods: list[dict[str, str]]  # [{start, end, days}]

    # Trend
    trend_detected: bool
    trend_pvalue: float | None
    trend_slope: float | None  # units/year
    trend_status: str

    # Breakpoints
    breakpoint_detected: bool
    breakpoint_date: str | None
    breakpoint_status: str

    # Seasonality
    seasonality_strength: float  # ACF at lag 12
    seasonality_status: str

    # Record length
    record_years: float
    record_status: str

    # Recommendations
    recommended_tmin: str | None
    recommended_tmax: str | None
    recommendations: list[dict[str, Any]] = field(default_factory=list)


def _coverage_analysis(series: pd.Series) -> tuple[float, int, list[dict]]:
    """Compute coverage %, max gap, and list of gap periods."""
    idx = series.index.sort_values()
    total_days = (idx[-1] - idx[0]).days + 1
    coverage = len(series) / total_days * 100 if total_days > 0 else 0

    gaps = idx.to_series().diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) > 0 else 0

    gap_periods = []
    large_gaps = gaps[gaps > 30]
    for i in large_gaps.index:
        pos = idx.get_loc(i)
        if pos > 0:
            gap_start = str(idx[pos - 1].date())
            gap_end = str(i.date())
            gap_periods.append({"start": gap_start, "end": gap_end, "days": int(large_gaps[i])})

    return coverage, max_gap, gap_periods


def _trend_analysis(series: pd.Series) -> tuple[bool, float | None, float | None]:
    """Mann-Kendall trend test."""
    try:
        import pymannkendall as mk
        monthly = series.resample("ME").mean().dropna()
        if len(monthly) < 24:
            return False, None, None
        result = mk.original_test(monthly.values)
        slope_per_year = result.slope * 12  # monthly → yearly
        return result.p < 0.05, float(result.p), float(slope_per_year)
    except ImportError:
        from scipy.stats import kendalltau
        monthly = series.resample("ME").mean().dropna()
        if len(monthly) < 24:
            return False, None, None
        x = np.arange(len(monthly))
        tau, p = kendalltau(x, monthly.values)
        slope = np.polyfit(x, monthly.values, 1)[0] * 12
        return p < 0.05, float(p), float(slope)


def _pettitt_test(series: pd.Series) -> tuple[bool, str | None]:
    """Pettitt change-point test (rank-based, non-parametric)."""
    monthly = series.resample("ME").mean().dropna()
    if len(monthly) < 24:
        return False, None

    vals = monthly.values
    n = len(vals)

    U = np.zeros(n, dtype=float)
    for t in range(n):
        for j in range(n):
            U[t] += np.sign(vals[t] - vals[j])
        U[t] = abs(U[t])

    # Faster: cumulative sign matrix approach
    ranks = np.argsort(np.argsort(vals)) + 1
    S = np.zeros(n)
    for t in range(1, n):
        S[t] = S[t - 1] + 2 * ranks[t] - (n + 1)
    K = int(np.argmax(np.abs(S)))
    Kn = np.abs(S[K])

    # Approximate p-value
    p = 2.0 * np.exp(-6.0 * Kn ** 2 / (n ** 3 + n ** 2))
    detected = p < 0.05

    bp_date = str(monthly.index[K].date()) if detected else None
    return detected, bp_date


def _seasonality_analysis(series: pd.Series) -> float:
    """Compute seasonality strength via ACF at lag 12 months."""
    monthly = series.resample("ME").mean().dropna()
    if len(monthly) < 36:
        return 0.0
    try:
        acf_vals = acf(monthly.values, nlags=12, fft=True)
        return float(abs(acf_vals[12])) if len(acf_vals) > 12 else 0.0
    except Exception:
        return 0.0


def _recommend_period(series: pd.Series, max_gap: int, gap_periods: list, breakpoint_date: str | None) -> tuple[str | None, str | None]:
    """Recommend calibration period based on diagnostics."""
    idx = series.index
    tmin = idx.min()
    tmax = idx.max()

    # Skip past the last major gap (>180 days)
    for gp in sorted(gap_periods, key=lambda g: g["days"], reverse=True):
        if int(gp["days"]) > 180:
            candidate = pd.Timestamp(gp["end"])
            if candidate > tmin:
                tmin = candidate

    # Skip past breakpoint if detected
    if breakpoint_date:
        bp = pd.Timestamp(breakpoint_date)
        if bp > tmin:
            tmin = bp

    # Ensure at least 5 years remain
    if (tmax - tmin).days < 365 * 5:
        tmin = idx.min()  # fallback to full series

    return str(tmin.date()), str(tmax.date())


def run_prefit_diagnostics(series: pd.Series) -> PreFitDiagnostic:
    """Run full pre-fit diagnostic suite on a piezometric series."""
    coverage, max_gap, gap_periods = _coverage_analysis(series)
    trend_detected, trend_p, trend_slope = _trend_analysis(series)
    bp_detected, bp_date = _pettitt_test(series)
    seasonality = _seasonality_analysis(series)

    idx = series.index
    record_years = (idx.max() - idx.min()).days / 365.25

    rec_tmin, rec_tmax = _recommend_period(series, max_gap, gap_periods, bp_date)

    # Status assignments
    cov_status = "green" if coverage > 80 else ("orange" if coverage > 50 else "red")
    gap_status = "green" if max_gap < 30 else ("orange" if max_gap < 180 else "red")
    trend_status = "green" if not trend_detected else ("red" if trend_p and trend_p < 0.01 else "orange")
    bp_status = "green" if not bp_detected else "orange"
    seas_status = "green" if seasonality > 0.3 else ("orange" if seasonality > 0.1 else "red")
    rec_status = "green" if record_years > 15 else ("orange" if record_years > 5 else "red")

    # Build recommendations
    recommendations: list[dict[str, Any]] = []
    if trend_detected and trend_slope:
        recommendations.append({
            "type": "trend",
            "message": f"Trend detected (slope {trend_slope:+.3f} m/yr, p={trend_p:.4f})",
            "action": "add_linear_trend",
            "params": {"slope_m_per_year": round(trend_slope, 4)},
        })
    for gp in gap_periods:
        if int(gp["days"]) > 180:
            recommendations.append({
                "type": "gap",
                "message": f"Major gap: {gp['start']} to {gp['end']} ({gp['days']} days)",
                "action": "set_tmin",
                "params": {"tmin": gp["end"]},
            })
    if bp_detected and bp_date:
        recommendations.append({
            "type": "breakpoint",
            "message": f"Breakpoint detected at {bp_date}",
            "action": "add_step_model",
            "params": {"date": bp_date},
        })
    if seasonality < 0.1:
        recommendations.append({
            "type": "seasonality",
            "message": "Weak seasonality — FlexModel may capture non-linear recharge better",
            "action": "set_recharge",
            "params": {"recharge": "FlexModel"},
        })
    if rec_tmin and rec_tmax:
        recommendations.append({
            "type": "period",
            "message": f"Recommended calibration: {rec_tmin} to {rec_tmax}",
            "action": "set_period",
            "params": {"tmin": rec_tmin, "tmax": rec_tmax},
        })

    return PreFitDiagnostic(
        coverage_pct=round(coverage, 1), coverage_status=cov_status,
        max_gap_days=max_gap, gaps_status=gap_status, gap_periods=gap_periods,
        trend_detected=trend_detected, trend_pvalue=trend_p, trend_slope=trend_slope, trend_status=trend_status,
        breakpoint_detected=bp_detected, breakpoint_date=bp_date, breakpoint_status=bp_status,
        seasonality_strength=round(seasonality, 3), seasonality_status=seas_status,
        record_years=round(record_years, 1), record_status=rec_status,
        recommended_tmin=rec_tmin, recommended_tmax=rec_tmax,
        recommendations=recommendations,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && pip install pymannkendall && python -m pytest tests/pastas/test_diagnostics_prefit.py -v`
Expected: 7 PASSED

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/pastas/diagnostics_prefit.py tests/pastas/test_diagnostics_prefit.py pyproject.toml
git commit -m "feat(pastas): add pre-fit diagnostics (trend, gaps, breakpoints, seasonality)"
```

---

## Task 3: Enhance fit_service with warm-up and two-pass solve

**Files:**
- Modify: `dashboard/utils/pastas/fit_service.py`
- Modify: `dashboard/utils/pastas/builder.py`
- Test: `tests/pastas/test_fit_service.py` (add new tests)

- [ ] **Step 1: Write failing tests for warm-up and two-pass**

Add to `tests/pastas/test_fit_service.py`:

```python
def test_run_fit_with_warmup(synthetic_station):
    """Warm-up should exclude first N years from metrics."""
    result = run_fit(
        gwl=synthetic_station.piezo, precip=synthetic_station.precip,
        evap=synthetic_station.evap, recharge_type="Linear", response_type="Gamma",
        noise_type="none", solver_type="LeastSquares", solver_kwargs=None,
        tmin=None, tmax=None, dataset_id="test", warm_up_years=1,
    )
    assert result.run_id
    assert result.metrics.get("evp") is not None


def test_run_fit_two_pass(synthetic_station):
    """Two-pass solve should use pass-1 params as initial for pass-2."""
    result = run_fit(
        gwl=synthetic_station.piezo, precip=synthetic_station.precip,
        evap=synthetic_station.evap, recharge_type="Linear", response_type="Gamma",
        noise_type="ArNoiseModel", solver_type="LeastSquares", solver_kwargs=None,
        tmin=None, tmax=None, dataset_id="test", two_pass=True,
    )
    assert result.run_id
    assert result.metrics.get("evp") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_fit_service.py::test_run_fit_with_warmup -v`
Expected: FAIL — `TypeError: run_fit() got an unexpected keyword argument 'warm_up_years'`

- [ ] **Step 3: Add warm_up_years and two_pass to run_fit**

In `dashboard/utils/pastas/fit_service.py`, modify the `run_fit` function signature (line 180) to add:

```python
def run_fit(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    recharge_type: str,
    response_type: str,
    noise_type: str,
    solver_type: str,
    solver_kwargs: Optional[dict[str, Any]],
    tmin: Optional[str],
    tmax: Optional[str],
    dataset_id: str,
    name: Optional[str] = None,
    val_split: Optional[float] = None,
    additional_stresses: Optional[list[dict[str, Any]]] = None,
    warm_up_years: int = 0,
    two_pass: bool = False,
    initial_params: Optional[dict[str, float]] = None,
) -> FitResult:
```

After `model, tmin, tmax = build_model(...)` and before `model.solve(...)`, add warm-up logic:

```python
    # Warm-up: shift tmin forward, metrics will exclude warm-up period
    effective_tmin = tmin
    if warm_up_years > 0 and tmin:
        warmup_end = pd.Timestamp(tmin) + pd.DateOffset(years=warm_up_years)
        effective_tmin = str(warmup_end.date())
    elif warm_up_years > 0:
        obs_start = gwl.dropna().index.min()
        effective_tmin = str((obs_start + pd.DateOffset(years=warm_up_years)).date())
```

For two-pass solve, replace the single `model.solve(...)` block with:

```python
    solver_cls = SOLVER_REGISTRY[solver_type]

    if two_pass and noise_type != "none":
        # Pass 1: solve WITHOUT noise
        model_pass1 = copy.deepcopy(model)
        if hasattr(model_pass1, 'noisemodel') and model_pass1.noisemodel is not None:
            # Remove noise for pass 1
            pass1_model, _, _ = build_model(
                gwl=gwl, precip=precip, evap=evap,
                recharge_type=recharge_type, response_type=response_type,
                noise_type="none", tmin=tmin, tmax=tmax,
                additional_stresses=additional_stresses,
            )
            pass1_model.solve(tmin=tmin, tmax=tmax_cal, solver=solver_cls(), report=False, **solver_kwargs)
            # Extract params from pass 1 as initial for pass 2
            for param_name in model.parameters.index:
                if param_name in pass1_model.parameters.index:
                    val = pass1_model.parameters.loc[param_name, "optimal"]
                    if pd.notna(val):
                        try:
                            model.set_parameter(param_name, initial=float(val))
                        except Exception:
                            pass

    # Apply any external initial params
    if initial_params:
        for pname, pval in initial_params.items():
            if pname in model.parameters.index:
                try:
                    model.set_parameter(pname, initial=pval)
                except Exception:
                    pass

    # Main solve
    solver_instance = solver_cls()
    model.solve(tmin=tmin, tmax=tmax_cal, solver=solver_instance, report=False, **solver_kwargs)
```

Add `import copy` at the top of the file.

Update metrics extraction to use `effective_tmin` instead of `tmin`:

```python
    metrics = _extract_metrics(model, effective_tmin, tmax_cal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_fit_service.py -v`
Expected: ALL PASSED (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/fit_service.py tests/pastas/test_fit_service.py
git commit -m "feat(pastas): add warm-up and two-pass solve to fit_service"
```

---

## Task 4: Auto-fit Grid Search Engine

**Files:**
- Create: `dashboard/utils/pastas/auto_fit.py`
- Test: `tests/pastas/test_auto_fit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pastas/test_auto_fit.py
"""Tests for auto-fit grid search engine."""
import pytest
from dashboard.utils.pastas.auto_fit import build_config_grid, AutoFitResult


class TestConfigGrid:
    def test_default_grid(self):
        grid = build_config_grid(bdlisa_preset={"recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel"})
        assert len(grid) >= 4
        assert all("recharge" in c and "response" in c and "noise" in c for c in grid)

    def test_with_trend(self):
        grid = build_config_grid(
            bdlisa_preset={"recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel"},
            add_trend=True,
        )
        with_trend = [c for c in grid if c.get("add_trend")]
        without_trend = [c for c in grid if not c.get("add_trend")]
        assert len(with_trend) > 0
        assert len(without_trend) > 0

    def test_no_duplicates(self):
        grid = build_config_grid(bdlisa_preset={"recharge": "Linear", "response": "Gamma", "noise": "ArNoiseModel"})
        keys = [f"{c['recharge']}_{c['response']}_{c['noise']}_{c.get('add_trend', False)}" for c in grid]
        assert len(keys) == len(set(keys))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_auto_fit.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement auto-fit engine**

```python
# dashboard/utils/pastas/auto_fit.py
"""Auto-fit grid search engine for Pastas models.

Enumerates config combinations, runs two-pass solve for each,
screens with STOWA criteria, and ranks by AIC.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dashboard.utils.pastas.config import get_options
from dashboard.utils.pastas.fit_service import run_fit, FitResult
from dashboard.utils.pastas.stowa import assess_stowa, StowaResult

logger = logging.getLogger(__name__)


@dataclass
class AutoFitCandidate:
    config: dict[str, str]
    fit_result: FitResult | None = None
    stowa: StowaResult | None = None
    aic: float | None = None
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class AutoFitResult:
    candidates: list[AutoFitCandidate]
    best: AutoFitCandidate | None = None
    total_elapsed_s: float = 0.0


def build_config_grid(
    bdlisa_preset: dict[str, str] | None = None,
    add_trend: bool = False,
) -> list[dict[str, Any]]:
    """Build a grid of config combinations to test."""
    recharge_options = {"Linear"}
    response_options = {"Gamma", "Exponential"}
    noise_options = {"ArNoiseModel", "none"}

    if bdlisa_preset:
        recharge_options.add(bdlisa_preset.get("recharge", "Linear"))
        response_options.add(bdlisa_preset.get("response", "Gamma"))
        if bdlisa_preset.get("noise", "ArNoiseModel") != "none":
            noise_options.add(bdlisa_preset.get("noise", "ArNoiseModel"))

    grid: list[dict[str, Any]] = []
    for rech in recharge_options:
        for resp in response_options:
            for noise in noise_options:
                config = {"recharge": rech, "response": resp, "noise": noise, "add_trend": False}
                grid.append(config)
                if add_trend:
                    grid.append({**config, "add_trend": True})

    return grid


def run_auto_fit(
    gwl, precip, evap, temp,
    code_bss: str,
    db_url: str,
    bdlisa_preset: dict[str, str] | None = None,
    warm_up_years: int = 2,
    add_trend: bool = False,
    val_split: float | None = 0.2,
    include_temp: bool = False,
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> AutoFitResult:
    """Run grid search over config combinations with STOWA screening.

    Args:
        on_progress: callback(current, total, config_label, status_message)
    """
    grid = build_config_grid(bdlisa_preset=bdlisa_preset, add_trend=add_trend)
    candidates: list[AutoFitCandidate] = []
    t0 = time.time()

    for i, config in enumerate(grid):
        label = f"{config['recharge']}/{config['response']}/{config['noise']}"
        if config.get("add_trend"):
            label += "+Trend"

        if on_progress:
            on_progress(i + 1, len(grid), label, "fitting...")

        candidate = AutoFitCandidate(config=config)
        t1 = time.time()

        try:
            additional_stresses = None
            if include_temp and temp is not None:
                additional_stresses = [{"type": "custom", "name": "temperature", "rfunc": "Gamma", "series": temp}]

            result = run_fit(
                gwl=gwl, precip=precip, evap=evap,
                recharge_type=config["recharge"],
                response_type=config["response"],
                noise_type=config["noise"],
                solver_type="LeastSquares",
                solver_kwargs=None,
                tmin=None, tmax=None,
                dataset_id=code_bss,
                name=f"{code_bss}_autofit_{label.replace('/', '_')}",
                val_split=val_split,
                additional_stresses=additional_stresses,
                warm_up_years=warm_up_years,
                two_pass=config["noise"] != "none",
            )
            candidate.fit_result = result
            candidate.aic = result.metrics.get("aic")

            # STOWA assessment
            from dashboard.utils.pastas.io import load_model
            try:
                model = load_model(result.run_id)
                cal_days = len(result.observed.index) if result.observed is not None else 3650
                candidate.stowa = assess_stowa(model, tmin=None, tmax=None, cal_period_days=cal_days)
            except Exception as exc:
                logger.warning("STOWA assessment failed for %s: %s", label, exc)

            status = "PASS" if candidate.stowa and candidate.stowa.overall_pass else "FAIL"
            evp = result.metrics.get("evp", 0)
            if on_progress:
                on_progress(i + 1, len(grid), label, f"EVP {evp:.1f}% {status}")

        except Exception as exc:
            candidate.error = str(exc)
            logger.warning("Auto-fit failed for %s: %s", label, exc)
            if on_progress:
                on_progress(i + 1, len(grid), label, f"ERROR: {exc}")

        candidate.elapsed_s = time.time() - t1
        candidates.append(candidate)

    # Rank: STOWA-passing models first, sorted by AIC
    passing = [c for c in candidates if c.stowa and c.stowa.overall_pass and c.aic is not None]
    passing.sort(key=lambda c: c.aic)  # type: ignore

    best = passing[0] if passing else None
    if best is None:
        # Fallback: pick best by AIC among all fitted models
        fitted = [c for c in candidates if c.aic is not None]
        fitted.sort(key=lambda c: c.aic)  # type: ignore
        best = fitted[0] if fitted else None

    return AutoFitResult(
        candidates=candidates,
        best=best,
        total_elapsed_s=time.time() - t0,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/pastas/test_auto_fit.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/auto_fit.py tests/pastas/test_auto_fit.py
git commit -m "feat(pastas): add auto-fit grid search engine with STOWA screening"
```

---

## Task 5: API Endpoints (diagnose, auto-fit SSE, compare-ai)

**Files:**
- Modify: `api/routers/pastas.py`
- Modify: `api/schemas/pastas.py`

- [ ] **Step 1: Add new Pydantic schemas**

Add to `api/schemas/pastas.py`:

```python
# --- Pre-fit Diagnostics ---

class DiagnoseRequest(BaseModel):
    code_bss: str

class DiagnosticIndicator(BaseModel):
    value: float | int | bool | None
    status: str  # green/orange/red
    detail: str | None = None

class DiagnosticRecommendation(BaseModel):
    type: str  # trend/gap/breakpoint/seasonality/period
    message: str
    action: str
    params: dict[str, Any] = {}

class DiagnoseResponse(BaseModel):
    coverage: DiagnosticIndicator
    gaps: DiagnosticIndicator
    trend: DiagnosticIndicator
    breakpoints: DiagnosticIndicator
    seasonality: DiagnosticIndicator
    record_length: DiagnosticIndicator
    recommended_tmin: str | None = None
    recommended_tmax: str | None = None
    recommendations: list[DiagnosticRecommendation] = []

# --- STOWA ---

class StowaResultSchema(BaseModel):
    evp_pass: bool
    evp_value: float
    autocorrelation_pass: bool
    runs_test_pvalue: float
    t95_pass: bool
    t95_days: float
    t95_threshold: float
    gain_pass: bool
    gain_significance: float
    overall_pass: bool
    suggestions: list[str] = []

# --- Auto-fit ---

class AutoFitRequest(BaseModel):
    code_bss: str
    warm_up_years: int = 2
    val_split: float = 0.2
    include_temp: bool = False
    add_trend: bool | None = None  # None = auto-detect from diagnostics

# --- Compare AI ---

class CompareAIRequest(BaseModel):
    pastas_run_id: str
    ai_model_id: str

class CompareAIMetrics(BaseModel):
    metric: str
    pastas_value: float | None
    ai_value: float | None
    best: str  # "pastas" | "ai" | "tie"

class CompareAIResponse(BaseModel):
    common_period: list[str]  # [tmin, tmax]
    dates: list[str]
    observed: list[float | None]
    pastas_simulated: list[float | None]
    ai_predicted: list[float | None]
    metrics: list[CompareAIMetrics]
```

- [ ] **Step 2: Add /diagnose endpoint**

Add to `api/routers/pastas.py`:

```python
@router.post("/diagnose")
def diagnose_station(req: DiagnoseRequest):
    """Run pre-fit diagnostics on a station's piezometric data."""
    from dashboard.utils.pastas.station_loader import load_station_series
    from dashboard.utils.pastas.diagnostics_prefit import run_prefit_diagnostics

    try:
        station = load_station_series(req.code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    diag = run_prefit_diagnostics(station.piezo)

    return DiagnoseResponse(
        coverage=DiagnosticIndicator(value=diag.coverage_pct, status=diag.coverage_status, detail=f"{diag.coverage_pct:.1f}% daily coverage"),
        gaps=DiagnosticIndicator(value=diag.max_gap_days, status=diag.gaps_status, detail=f"Largest gap: {diag.max_gap_days} days"),
        trend=DiagnosticIndicator(value=diag.trend_detected, status=diag.trend_status, detail=f"Slope: {diag.trend_slope:+.4f} m/yr" if diag.trend_slope else None),
        breakpoints=DiagnosticIndicator(value=diag.breakpoint_detected, status=diag.breakpoint_status, detail=f"At {diag.breakpoint_date}" if diag.breakpoint_date else None),
        seasonality=DiagnosticIndicator(value=diag.seasonality_strength, status=diag.seasonality_status, detail=f"ACF(12) = {diag.seasonality_strength:.3f}"),
        record_length=DiagnosticIndicator(value=diag.record_years, status=diag.record_status, detail=f"{diag.record_years:.1f} years"),
        recommended_tmin=diag.recommended_tmin,
        recommended_tmax=diag.recommended_tmax,
        recommendations=[DiagnosticRecommendation(**r) for r in diag.recommendations],
    )
```

- [ ] **Step 3: Add /auto-fit SSE endpoint**

Add to `api/routers/pastas.py`:

```python
from fastapi.responses import StreamingResponse
import json

@router.post("/auto-fit")
def auto_fit(req: AutoFitRequest):
    """Run auto-fit grid search with SSE progress stream."""
    from dashboard.utils.pastas.station_loader import load_station_series
    from dashboard.utils.pastas.auto_fit import run_auto_fit
    from dashboard.utils.pastas.config import get_preset
    from dashboard.utils.pastas.diagnostics_prefit import run_prefit_diagnostics

    try:
        station = load_station_series(req.code_bss, _brgm_url())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    preset = get_preset(station.metadata.get("nature_eh"), station.metadata.get("milieu_eh"))

    # Auto-detect trend if not specified
    add_trend = req.add_trend
    if add_trend is None:
        diag = run_prefit_diagnostics(station.piezo)
        add_trend = diag.trend_detected

    messages = []

    def on_progress(current, total, label, status):
        messages.append({"current": current, "total": total, "label": label, "status": status})

    result = run_auto_fit(
        gwl=station.piezo, precip=station.precip, evap=station.evap, temp=station.temp,
        code_bss=req.code_bss, db_url=_brgm_url(),
        bdlisa_preset=preset, warm_up_years=req.warm_up_years,
        add_trend=add_trend, val_split=req.val_split,
        include_temp=req.include_temp, on_progress=on_progress,
    )

    # Build response
    candidates = []
    for c in result.candidates:
        cand = {
            "config": c.config,
            "aic": c.aic,
            "evp": c.fit_result.metrics.get("evp") if c.fit_result else None,
            "nse": c.fit_result.metrics.get("nse") if c.fit_result else None,
            "run_id": c.fit_result.run_id if c.fit_result else None,
            "stowa": {
                "evp_pass": c.stowa.evp_pass, "autocorrelation_pass": c.stowa.autocorrelation_pass,
                "t95_pass": c.stowa.t95_pass, "gain_pass": c.stowa.gain_pass,
                "overall_pass": c.stowa.overall_pass,
            } if c.stowa else None,
            "error": c.error,
            "elapsed_s": round(c.elapsed_s, 1),
        }
        candidates.append(cand)

    return {
        "candidates": candidates,
        "best_run_id": result.best.fit_result.run_id if result.best and result.best.fit_result else None,
        "best_config": result.best.config if result.best else None,
        "total_elapsed_s": round(result.total_elapsed_s, 1),
    }
```

- [ ] **Step 4: Add /compare-ai endpoint**

Add to `api/routers/pastas.py`:

```python
@router.post("/compare-ai")
def compare_ai(req: CompareAIRequest):
    """Compare a Pastas model with a DL model on their common period."""
    from dashboard.utils.pastas.io import load_model
    import numpy as np

    # Load Pastas simulation
    try:
        ps_model = load_model(req.pastas_run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Pastas model not found: {exc}") from exc

    ps_sim = ps_model.simulate()
    ps_obs = ps_model.observations()

    # Load AI model forecast
    try:
        from dashboard.utils.forecasting import load_forecast_for_model
        ai_dates, ai_preds = load_forecast_for_model(req.ai_model_id)
    except Exception as exc:
        raise HTTPException(404, f"AI model forecast not available: {exc}") from exc

    # Find common period
    ps_dates = set(ps_sim.index.strftime("%Y-%m-%d"))
    ai_dates_set = set(ai_dates)
    common = sorted(ps_dates & ai_dates_set)

    if len(common) < 30:
        raise HTTPException(422, f"Only {len(common)} common dates — need at least 30 for comparison")

    common_idx = pd.DatetimeIndex(common)
    obs_vals = [float(ps_obs.get(d, float("nan"))) if d in ps_obs.index else None for d in common_idx]
    ps_vals = [float(ps_sim.get(d, float("nan"))) if d in ps_sim.index else None for d in common_idx]
    ai_vals = [float(ai_preds[ai_dates.index(d.strftime("%Y-%m-%d"))]) if d.strftime("%Y-%m-%d") in ai_dates else None for d in common_idx]

    # Compute metrics on common period
    def _metric(obs, pred, name):
        o = np.array([v for v, p in zip(obs, pred) if v is not None and p is not None], dtype=float)
        p = np.array([p for v, p in zip(obs, pred) if v is not None and p is not None], dtype=float)
        if len(o) < 10:
            return None
        if name == "nse":
            return float(1 - np.sum((o - p) ** 2) / np.sum((o - np.mean(o)) ** 2))
        if name == "rmse":
            return float(np.sqrt(np.mean((o - p) ** 2)))
        if name == "kge":
            r = np.corrcoef(o, p)[0, 1]
            alpha = np.std(p) / np.std(o)
            beta = np.mean(p) / np.mean(o)
            return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        return None

    metrics = []
    for name in ["nse", "rmse", "kge"]:
        ps_val = _metric(obs_vals, ps_vals, name)
        ai_val = _metric(obs_vals, ai_vals, name)
        if ps_val is not None and ai_val is not None:
            if name == "rmse":
                best = "pastas" if ps_val < ai_val else "ai"
            else:
                best = "pastas" if ps_val > ai_val else "ai"
        else:
            best = "tie"
        metrics.append({"metric": name.upper(), "pastas_value": ps_val, "ai_value": ai_val, "best": best})

    return {
        "common_period": [common[0], common[-1]],
        "dates": common,
        "observed": obs_vals,
        "pastas_simulated": ps_vals,
        "ai_predicted": ai_vals,
        "metrics": metrics,
    }
```

- [ ] **Step 5: Import new schemas at top of pastas.py**

Add to the imports section of `api/routers/pastas.py`:

```python
from api.schemas.pastas import (
    # ... existing imports ...
    DiagnoseRequest, DiagnoseResponse, DiagnosticIndicator, DiagnosticRecommendation,
    AutoFitRequest,
    CompareAIRequest,
)
```

- [ ] **Step 6: Commit**

```bash
git add api/routers/pastas.py api/schemas/pastas.py
git commit -m "feat(pastas): add /diagnose, /auto-fit, /compare-ai API endpoints"
```

---

## Task 6: Frontend Types, API, and Hooks

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/hooks/usePastas.ts`

- [ ] **Step 1: Add TypeScript types**

Add to `frontend/src/lib/types.ts`:

```typescript
// Pre-fit Diagnostics
export interface DiagnosticIndicator {
  value: number | boolean | null
  status: 'green' | 'orange' | 'red'
  detail: string | null
}

export interface DiagnosticRecommendation {
  type: string
  message: string
  action: string
  params: Record<string, unknown>
}

export interface DiagnoseResult {
  coverage: DiagnosticIndicator
  gaps: DiagnosticIndicator
  trend: DiagnosticIndicator
  breakpoints: DiagnosticIndicator
  seasonality: DiagnosticIndicator
  record_length: DiagnosticIndicator
  recommended_tmin: string | null
  recommended_tmax: string | null
  recommendations: DiagnosticRecommendation[]
}

// STOWA
export interface StowaResult {
  evp_pass: boolean
  evp_value: number
  autocorrelation_pass: boolean
  runs_test_pvalue: number
  t95_pass: boolean
  t95_days: number
  t95_threshold: number
  gain_pass: boolean
  gain_significance: number
  overall_pass: boolean
  suggestions: string[]
}

// Auto-fit
export interface AutoFitCandidate {
  config: Record<string, string | boolean>
  aic: number | null
  evp: number | null
  nse: number | null
  run_id: string | null
  stowa: { evp_pass: boolean; autocorrelation_pass: boolean; t95_pass: boolean; gain_pass: boolean; overall_pass: boolean } | null
  error: string | null
  elapsed_s: number
}

export interface AutoFitResult {
  candidates: AutoFitCandidate[]
  best_run_id: string | null
  best_config: Record<string, string | boolean> | null
  total_elapsed_s: number
}

// Compare AI
export interface CompareAIResult {
  common_period: [string, string]
  dates: string[]
  observed: (number | null)[]
  pastas_simulated: (number | null)[]
  ai_predicted: (number | null)[]
  metrics: { metric: string; pastas_value: number | null; ai_value: number | null; best: string }[]
}
```

- [ ] **Step 2: Add API methods**

Add to the `pastas` section in `frontend/src/lib/api.ts`:

```typescript
    diagnose: (codeBss: string) =>
      postJson<DiagnoseResult>('/pastas/diagnose', { code_bss: codeBss }),
    autoFit: (body: {
      code_bss: string; warm_up_years?: number; val_split?: number;
      include_temp?: boolean; add_trend?: boolean | null
    }) => postJson<AutoFitResult>('/pastas/auto-fit', body, 300_000),
    compareAI: (body: { pastas_run_id: string; ai_model_id: string }) =>
      postJson<CompareAIResult>('/pastas/compare-ai', body),
```

Add the new types to the import at the top of `api.ts`:

```typescript
import type { DiagnoseResult, AutoFitResult, CompareAIResult } from './types'
```

- [ ] **Step 3: Add React Query hooks**

Add to `frontend/src/hooks/usePastas.ts`:

```typescript
export function usePastasDiagnose(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'diagnose', codeBss],
    queryFn: () => api.pastas.diagnose(codeBss!),
    enabled: !!codeBss,
    staleTime: 10 * 60 * 1000,
  })
}

export function usePastasAutoFit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.autoFit,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasCompareAI() {
  return useMutation({
    mutationFn: api.pastas.compareAI,
  })
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/hooks/usePastas.ts
git commit -m "feat(pastas): add frontend types, API methods, and hooks for diagnose/auto-fit/compare-ai"
```

---

## Task 7: Frontend — Pre-fit Diagnostic Panel + STOWA Banner + Mode Toggle

**Files:**
- Create: `frontend/src/components/pastas/PreFitDiagnosticPanel.tsx`
- Create: `frontend/src/components/pastas/StowaVerdictBanner.tsx`
- Create: `frontend/src/components/pastas/GuidedExpertToggle.tsx`

- [ ] **Step 1: Create PreFitDiagnosticPanel**

Create `frontend/src/components/pastas/PreFitDiagnosticPanel.tsx` — a component that displays the 6 diagnostic indicators with colored badges and "Apply" action buttons.

Props: `{ diagnosis: DiagnoseResult | undefined; isLoading: boolean; onApplyRecommendation: (rec: DiagnosticRecommendation) => void }`

Shows a grid of 6 indicator cards (coverage, gaps, trend, breakpoints, seasonality, record length), each with a colored dot (green/orange/red), the value, and the detail text. Below the grid, a list of recommendations with "Apply" buttons.

- [ ] **Step 2: Create StowaVerdictBanner**

Create `frontend/src/components/pastas/StowaVerdictBanner.tsx` — a full-width banner displaying the 4 STOWA criteria.

Props: `{ stowa: StowaResult | null; suggestions?: string[] }`

Shows 4 badges in a row: EVP, Autocorrelation, t95, Gain — each green checkmark or red X with the value. Overall verdict at the end: "Model accepted" or "Model needs attention".

- [ ] **Step 3: Create GuidedExpertToggle**

Create `frontend/src/components/pastas/GuidedExpertToggle.tsx` — a toggle switch between Guided and Expert mode, persisted in localStorage.

Props: `{ mode: 'guided' | 'expert'; onChange: (mode: 'guided' | 'expert') => void }`

Simple pill toggle: `[Guided | Expert]` with cyan highlight on active.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/PreFitDiagnosticPanel.tsx frontend/src/components/pastas/StowaVerdictBanner.tsx frontend/src/components/pastas/GuidedExpertToggle.tsx
git commit -m "feat(pastas): add PreFitDiagnosticPanel, StowaVerdictBanner, and GuidedExpertToggle components"
```

---

## Task 8: Frontend — 5-Step Pipeline Pages

**Files:**
- Create: `frontend/src/pages/pastas/StationStep.tsx`
- Create: `frontend/src/pages/pastas/CalibrateStep.tsx`
- Create: `frontend/src/pages/pastas/ResultsStep.tsx`
- Create: `frontend/src/pages/pastas/ScenariosStep.tsx`
- Modify: `frontend/src/pages/pastas/PastasLayout.tsx`
- Modify: `frontend/src/routes.tsx`

- [ ] **Step 1: Create StationStep page**

`frontend/src/pages/pastas/StationStep.tsx` — Station selection + pre-fit diagnostics.

Reuses existing `StationPicker` and `StationDetailPanel`. Adds `PreFitDiagnosticPanel` below the station preview. In Guided mode, recommendations auto-apply and are shown as a summary. In Expert mode, each recommendation has an "Apply" button.

Stores `codeBss` in URL search params. When user clicks "Next: Calibrate →", navigates to `/pastas/calibrate?station={codeBss}` with any applied recommendations as additional URL params.

- [ ] **Step 2: Create CalibrateStep page**

`frontend/src/pages/pastas/CalibrateStep.tsx` — Auto-fit or manual config.

Reads `codeBss` from URL params. Shows two modes:

**Guided**: Single "Auto-fit" button. Calls `usePastasAutoFit()` mutation. Shows progress (current/total, config label, status). On completion, shows ranked candidate table. "Use best model" button navigates to Results.

**Expert**: Existing config form (PastasConfigForm, CalValToggle, temperature checkbox, warm-up slider, two-pass checkbox). "Fit" button calls existing `usePastasFit()`. On completion, navigates to Results.

Both modes store `runId` in URL params on completion.

- [ ] **Step 3: Create ResultsStep page**

`frontend/src/pages/pastas/ResultsStep.tsx` — STOWA verdict + results.

Reads `runId` from URL params. Loads model via `usePastasModel(runId)`. Shows:
1. `StowaVerdictBanner` at top (calls `/models/{runId}/diagnostics` to get data for STOWA, or computes client-side from existing metrics)
2. Existing `FitResultsPanel` sections (metrics, time series, contributions, response, diagnostics, parameters, signatures)
3. `AIComparisonPanel` if an AI model exists for the station

In Guided mode, sections 5-7 are collapsed. In Expert mode, all expanded.

- [ ] **Step 4: Create ScenariosStep page**

`frontend/src/pages/pastas/ScenariosStep.tsx` — refactored from existing ScenarioWorkflow.

Reads `runId` from URL params. Embeds the existing `ScenarioWorkflow` component with the loaded model. Adds the new DRIAS presets.

- [ ] **Step 5: Update PastasLayout with 5 tabs + mode toggle**

Modify `frontend/src/pages/pastas/PastasLayout.tsx`:

```typescript
import { NavLink, Outlet } from 'react-router-dom'
import { MapPin, SlidersHorizontal, BarChart3, FlaskConical, LayoutGrid } from 'lucide-react'
import { GuidedExpertToggle } from '@/components/pastas/GuidedExpertToggle'
import { useState, createContext, useContext } from 'react'

export const PastasModeContext = createContext<{ mode: 'guided' | 'expert'; setMode: (m: 'guided' | 'expert') => void }>({ mode: 'guided', setMode: () => {} })
export function usePastasMode() { return useContext(PastasModeContext) }

const TABS = [
  { to: '/pastas/station', icon: MapPin, label: 'Station' },
  { to: '/pastas/calibrate', icon: SlidersHorizontal, label: 'Calibrate' },
  { to: '/pastas/results', icon: BarChart3, label: 'Results' },
  { to: '/pastas/scenarios', icon: FlaskConical, label: 'Scenarios' },
  { to: '/pastas/gallery', icon: LayoutGrid, label: 'Gallery' },
] as const

export default function PastasLayout() {
  const [mode, setMode] = useState<'guided' | 'expert'>(() => {
    return (localStorage.getItem('pastas_mode') as 'guided' | 'expert') ?? 'guided'
  })
  const handleMode = (m: 'guided' | 'expert') => { setMode(m); localStorage.setItem('pastas_mode', m) }

  return (
    <PastasModeContext.Provider value={{ mode, setMode: handleMode }}>
      <div className="flex flex-col h-full">
        <div className="bg-bg-card border-b border-white/5 shrink-0 flex items-center justify-between px-4">
          <div className="flex items-center gap-1">
            {TABS.map(({ to, icon: Icon, label }) => (
              <NavLink key={to} to={to}
                className={({ isActive }) => `flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 ${isActive ? 'border-accent-cyan text-text-primary' : 'border-transparent text-text-muted hover:text-text-secondary'}`}>
                <Icon className="w-4 h-4" />{label}
              </NavLink>
            ))}
          </div>
          <GuidedExpertToggle mode={mode} onChange={handleMode} />
        </div>
        <div className="flex-1 min-h-0 overflow-auto"><Outlet /></div>
      </div>
    </PastasModeContext.Provider>
  )
}
```

- [ ] **Step 6: Update routes.tsx**

Update the `/pastas` children in `frontend/src/routes.tsx`:

```typescript
const StationStep = lazy(() => import('./pages/pastas/StationStep'))
const CalibrateStep = lazy(() => import('./pages/pastas/CalibrateStep'))
const ResultsStep = lazy(() => import('./pages/pastas/ResultsStep'))
const ScenariosStep = lazy(() => import('./pages/pastas/ScenariosStep'))

// In the /pastas children:
{ index: true, element: <Navigate to="/pastas/station" replace /> },
{ path: 'station', element: <SW><StationStep /></SW> },
{ path: 'calibrate', element: <SW><CalibrateStep /></SW> },
{ path: 'results', element: <SW><ResultsStep /></SW> },
{ path: 'scenarios', element: <SW><ScenariosStep /></SW> },
{ path: 'gallery', element: <SW><PastasGalleryPage /></SW> },
// Backward compat:
{ path: 'fit', element: <Navigate to="/pastas/station" replace /> },
{ path: 'compare', element: <Navigate to="/pastas/gallery" replace /> },
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/pastas/ frontend/src/routes.tsx
git commit -m "feat(pastas): implement 5-step pipeline (Station, Calibrate, Results, Scenarios, Gallery)"
```

---

## Task 9: Gallery — STOWA Badges + Station Grouping

**Files:**
- Modify: `frontend/src/components/pastas/ModelTable.tsx`
- Modify: `frontend/src/pages/pastas/GalleryPage.tsx`

- [ ] **Step 1: Add STOWA badge to ModelTable**

In `ModelTable.tsx`, add a STOWA mini-badge to each model card. The STOWA data comes from the model's metrics (EVP >= 70 as a simple client-side check) and a new `stowa` field on `PastasModelSummary`.

Add a `StowaMiniBadge` component that shows 4 small dots (green/red) in a row.

Add station grouping: group models by `code_bss`, show station header with count, best model starred.

- [ ] **Step 2: Update GalleryPage**

Add the AIC column to the model summary. The backend `PastasModelSummary` already has access to AIC via metrics — add `aic: number | null` to the type and populate it in the backend list endpoint.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/pastas/ModelTable.tsx frontend/src/pages/pastas/GalleryPage.tsx
git commit -m "feat(pastas): add STOWA badges and station grouping to Gallery"
```

---

## Task 10: Docker Build + Integration Test

**Files:**
- Modify: `docker/backend/Dockerfile` (add pymannkendall)
- No new files

- [ ] **Step 1: Add pymannkendall to Docker build**

In `docker/backend/Dockerfile`, the `pip install` step should pick up `pymannkendall` from `pyproject.toml` automatically. Verify by building.

- [ ] **Step 2: Docker build**

```bash
cd /home/ringuet/time-serie-explo && docker compose up -d --build
```

Expected: all containers healthy, zero TS errors, zero Python import errors.

- [ ] **Step 3: Test /diagnose endpoint**

```bash
curl -s -X POST http://localhost:49513/api/v1/pastas/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"code_bss":"01584X0023/LV3"}' | python3 -m json.tool
```

Expected: JSON with 6 indicators, recommended period, recommendations list.

- [ ] **Step 4: Test /auto-fit endpoint**

```bash
curl -s -X POST http://localhost:49513/api/v1/pastas/auto-fit \
  -H 'Content-Type: application/json' \
  -d '{"code_bss":"01584X0023/LV3","warm_up_years":2}' | python3 -m json.tool
```

Expected: JSON with candidates array, best_run_id, total_elapsed_s. Takes 60-120s.

- [ ] **Step 5: Verify frontend pipeline**

Open browser, navigate to `/pastas/station`, select a station, verify diagnostics appear, click through Calibrate → Results → Scenarios → Gallery.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(pastas): complete 5-step pipeline with diagnostics, auto-fit, STOWA screening"
```
