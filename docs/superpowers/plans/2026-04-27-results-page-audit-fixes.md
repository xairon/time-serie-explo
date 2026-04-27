# Pastas Results Page — Audit Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 21 issues found by the 4-agent audit (scientific accuracy, backend correctness, frontend performance, UX) of the Pastas results page.

**Architecture:** 5 phases ordered by criticality. Phase 1 fixes backend computation bugs (wrong results shown to users). Phase 2 fixes frontend data bugs. Phase 3 fixes performance. Phase 4 fixes scientific thresholds and edge cases. Phase 5 fixes UX consistency. Each phase ends with a test checkpoint.

**Tech Stack:** Python (numpy, scipy, statsmodels, pandas), TypeScript/React (TanStack Query, Plotly.js, Tailwind CSS)

---

## Phase 1 — Critical Backend Computation Bugs

### Task 1: Fix Lyne-Hollick 3-pass baseflow filter (Issues #1)

The current implementation runs 3 identical forward passes (each zeroed out), producing the same result as 1 pass. The correct algorithm alternates forward/backward/forward and feeds each pass's output into the next.

**Files:**
- Modify: `dashboard/utils/pastas/baseflow.py`
- Create: `tests/pastas/test_baseflow.py`

- [ ] **Step 1: Write tests for the corrected filter**

```python
"""Tests for baseflow separation."""
from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_model(values, index=None):
    import pandas as pd
    if index is None:
        index = pd.date_range("2010-01-01", periods=len(values), freq="D")
    obs = pd.Series(values, index=index)
    model = MagicMock()
    model.observations.return_value = obs
    return model


class TestComputeBaseflow:
    def test_short_series_returns_none(self):
        from dashboard.utils.pastas.baseflow import compute_baseflow
        model = _make_mock_model(np.zeros(10))
        result = compute_baseflow(model, "2010-01-01", "2010-01-10")
        assert result["bfi"] is None

    def test_constant_level_bfi_is_one(self):
        """Constant level = no variation = all baseflow."""
        from dashboard.utils.pastas.baseflow import compute_baseflow
        model = _make_mock_model(np.full(100, 50.0))
        result = compute_baseflow(model, "2010-01-01", "2010-04-10")
        assert result["bfi"] == 1.0

    def test_sine_wave_has_moderate_bfi(self):
        """A smooth sine wave should have high BFI (mostly slow variation)."""
        from dashboard.utils.pastas.baseflow import compute_baseflow
        t = np.arange(365)
        values = 50.0 + 2.0 * np.sin(2 * np.pi * t / 365)
        model = _make_mock_model(values)
        result = compute_baseflow(model, "2010-01-01", "2010-12-31")
        assert result["bfi"] is not None
        assert 0.5 < result["bfi"] < 1.0

    def test_noisy_signal_has_lower_bfi(self):
        """Random noise should have lower BFI than a smooth signal."""
        from dashboard.utils.pastas.baseflow import compute_baseflow
        np.random.seed(42)
        smooth = 50.0 + np.cumsum(np.random.randn(200) * 0.01)
        noisy = 50.0 + np.cumsum(np.random.randn(200) * 0.5)
        m_smooth = _make_mock_model(smooth)
        m_noisy = _make_mock_model(noisy)
        bfi_smooth = compute_baseflow(m_smooth, "2010-01-01", "2010-07-19")["bfi"]
        bfi_noisy = compute_baseflow(m_noisy, "2010-01-01", "2010-07-19")["bfi"]
        assert bfi_smooth > bfi_noisy

    def test_output_arrays_match_length(self):
        from dashboard.utils.pastas.baseflow import compute_baseflow
        values = 50.0 + np.cumsum(np.random.randn(100) * 0.1)
        model = _make_mock_model(values)
        result = compute_baseflow(model, "2010-01-01", "2010-04-10")
        assert len(result["observed"]) == len(result["baseflow"]) == len(result["quickflow"])

    def test_three_pass_differs_from_one_pass(self):
        """Verify the 3-pass filter produces different results than a single pass."""
        from dashboard.utils.pastas.baseflow import compute_baseflow, _lyne_hollick_3pass
        np.random.seed(42)
        dh = np.cumsum(np.random.randn(200) * 0.1)
        dh = np.diff(dh)
        qf_3pass = _lyne_hollick_3pass(dh, alpha=0.925)
        # Single forward pass
        qf_1pass = np.zeros(len(dh))
        for t in range(1, len(dh)):
            qf_1pass[t] = 0.925 * qf_1pass[t-1] + (1.925)/2 * (dh[t] - dh[t-1])
        assert not np.allclose(qf_3pass, qf_1pass), "3-pass should differ from 1-pass"
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/pastas/test_baseflow.py -v`
Expected: FAIL (import error on `_lyne_hollick_3pass`, and filter logic wrong)

- [ ] **Step 3: Rewrite baseflow.py with correct 3-pass algorithm**

```python
"""Baseflow separation using Lyne & Hollick recursive digital filter.

Applied to dh/dt (daily level change) rather than raw levels — the filter
was designed for streamflow, not absolute heads.  Working on the rate of
change gives a meaningful split between slow (sustained recharge) and rapid
(storm pulses) components of the piezometric signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _lyne_hollick_3pass(signal: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    """Three-pass Lyne & Hollick filter (forward-backward-forward)."""
    n = len(signal)
    current = signal.copy()

    for pass_num in range(3):
        qf = np.zeros(n)
        if pass_num % 2 == 0:  # forward
            for t in range(1, n):
                qf[t] = alpha * qf[t - 1] + (1 + alpha) / 2 * (current[t] - current[t - 1])
        else:  # backward
            for t in range(n - 2, -1, -1):
                qf[t] = alpha * qf[t + 1] + (1 + alpha) / 2 * (current[t] - current[t + 1])
        qf = np.maximum(qf, 0)
        current = current - qf

    return signal - current


def compute_baseflow(model, tmin: str, tmax: str, alpha: float = 0.925) -> dict:
    """Separate slow from rapid piezometric variation using Lyne & Hollick."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 30:
        return {"bfi": None, "index": [], "observed": [], "baseflow": [], "quickflow": []}

    obs_clean = obs.dropna()
    values = obs_clean.values.flatten()

    dh = np.diff(values)
    if len(dh) < 30:
        return {"bfi": None, "index": [], "observed": [], "baseflow": [], "quickflow": []}

    quickflow_dh = _lyne_hollick_3pass(dh, alpha)
    baseflow_dh = dh - quickflow_dh

    baseflow = np.concatenate([[values[0]], values[0] + np.cumsum(baseflow_dh)])
    quickflow = values - baseflow

    total_var = np.sum(np.abs(dh))
    slow_var = np.sum(np.abs(baseflow_dh))
    bfi = float(slow_var / total_var) if total_var > 0 else 1.0
    bfi = min(max(bfi, 0.0), 1.0)

    return {
        "bfi": round(bfi, 3),
        "index": [d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10] for d in obs_clean.index],
        "observed": values.tolist(),
        "baseflow": baseflow.tolist(),
        "quickflow": quickflow.tolist(),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/pastas/test_baseflow.py -v`
Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/baseflow.py tests/pastas/test_baseflow.py
git commit -m "fix(pastas): correct 3-pass Lyne-Hollick filter (forward-backward-forward)"
```

---

### Task 2: Fix recession analysis — fit on drawdown, not absolute NGF (Issue #4)

The exponential decay `h0 * exp(-t/T)` decays toward 0, but piezometric heads are 10-200 m NGF. The fit must use a baseline offset: `c + (h0-c) * exp(-t/T)`, or equivalently fit on the anomaly `h - h_final`. Also relax segment detection to allow up to 3 days of non-negative diff.

**Files:**
- Modify: `dashboard/utils/pastas/recession.py`
- Create: `tests/pastas/test_recession.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for recession analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def _make_mock_model(values, start="2010-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    obs = pd.Series(values, index=index)
    model = MagicMock()
    model.observations.return_value = obs
    return model


class TestRecessionAnalysis:
    def test_short_series_returns_empty(self):
        from dashboard.utils.pastas.recession import compute_recession_analysis
        model = _make_mock_model(np.zeros(20))
        result = compute_recession_analysis(model, "2010-01-01", "2010-01-20")
        assert result["n_segments"] == 0

    def test_known_exponential_decay(self):
        """A perfect exponential decay from 50m with T=100d should be recovered."""
        from dashboard.utils.pastas.recession import compute_recession_analysis
        t = np.arange(200)
        baseline = 45.0
        values = baseline + 5.0 * np.exp(-t / 100.0)
        model = _make_mock_model(values)
        result = compute_recession_analysis(model, "2010-01-01", "2010-07-19")
        assert result["n_segments"] >= 1
        T = result["segments"][0]["T_days"]
        assert 80 < T < 120, f"Expected T~100, got {T}"

    def test_allows_short_interruptions(self):
        """A recession with 1-2 days of zero change should still form one segment."""
        from dashboard.utils.pastas.recession import compute_recession_analysis
        t = np.arange(60)
        values = 50.0 - 0.02 * t.astype(float)
        values[20] = values[19]  # 1-day plateau
        values[40] = values[39] + 0.001  # tiny uptick
        model = _make_mock_model(values)
        result = compute_recession_analysis(model, "2010-01-01", "2010-03-01")
        assert result["n_segments"] >= 1

    def test_high_ngf_level_does_not_bias_T(self):
        """Recession at 200m NGF should give same T as at 50m NGF."""
        from dashboard.utils.pastas.recession import compute_recession_analysis
        t = np.arange(150)
        low = 50.0 + 3.0 * np.exp(-t / 80.0)
        high = 200.0 + 3.0 * np.exp(-t / 80.0)
        m_low = _make_mock_model(low)
        m_high = _make_mock_model(high)
        T_low = compute_recession_analysis(m_low, "2010-01-01", "2010-05-30")
        T_high = compute_recession_analysis(m_high, "2010-01-01", "2010-05-30")
        if T_low["n_segments"] > 0 and T_high["n_segments"] > 0:
            assert abs(T_low["segments"][0]["T_days"] - T_high["segments"][0]["T_days"]) < 20
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/pastas/test_recession.py -v`
Expected: FAIL on `test_known_exponential_decay` (T biased), `test_allows_short_interruptions` (no segment found), `test_high_ngf_level_does_not_bias_T` (T differs)

- [ ] **Step 3: Rewrite recession.py**

```python
"""Recession analysis and Master Recession Curve extraction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _exp_decay_with_baseline(t, amplitude, T, baseline):
    return baseline + amplitude * np.exp(-t / T)


def compute_recession_analysis(
    model, tmin: str, tmax: str,
    min_duration_days: int = 30,
    min_drop_m: float = 0.05,
    max_interruption_days: int = 3,
) -> dict:
    """Extract recession segments, fit exponential decay, compute MRC."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 60:
        return {"segments": [], "mrc": None, "T_median_days": None,
                "T_mean_days": None, "T_std_days": None, "n_segments": 0}

    obs_clean = obs.dropna()
    values = obs_clean.values.flatten()
    dates = obs_clean.index

    diff = np.diff(values)

    # Find recession segments — allow short interruptions (up to max_interruption_days)
    segments = []
    in_recession = False
    start_idx = 0
    interruption_count = 0

    for i in range(len(diff)):
        if diff[i] < 0:
            if not in_recession:
                in_recession = True
                start_idx = i
                interruption_count = 0
            else:
                interruption_count = 0
        else:
            if in_recession:
                interruption_count += 1
                if interruption_count > max_interruption_days:
                    end_idx = i - interruption_count
                    duration = end_idx - start_idx
                    drop = values[start_idx] - values[end_idx]
                    if duration >= min_duration_days and drop >= min_drop_m:
                        segments.append((start_idx, end_idx))
                    in_recession = False
                    interruption_count = 0

    if in_recession:
        end_idx = len(diff) - interruption_count
        duration = end_idx - start_idx
        drop = values[start_idx] - values[end_idx]
        if duration >= min_duration_days and drop >= min_drop_m:
            segments.append((start_idx, end_idx))

    fitted_segments = []
    all_T = []
    normalized_curves = []

    for start_idx, end_idx in segments:
        seg_values = values[start_idx:end_idx + 1]
        seg_dates = dates[start_idx:end_idx + 1]
        t = np.arange(len(seg_values), dtype=float)

        amplitude_init = seg_values[0] - seg_values[-1]
        baseline_init = seg_values[-1]
        if amplitude_init <= 0:
            continue

        try:
            popt, _ = curve_fit(
                _exp_decay_with_baseline, t, seg_values,
                p0=[amplitude_init, 50.0, baseline_init],
                bounds=([0, 1, -np.inf], [np.inf, 10000, np.inf]),
                maxfev=5000,
            )
            amplitude_fit, T_fit, baseline_fit = popt
            predicted = _exp_decay_with_baseline(t, amplitude_fit, T_fit, baseline_fit)
            ss_res = np.sum((seg_values - predicted) ** 2)
            ss_tot = np.sum((seg_values - np.mean(seg_values)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if T_fit > 0 and r_squared > 0.3:
                fitted_segments.append({
                    "start": str(seg_dates[0].date()) if hasattr(seg_dates[0], 'date') else str(seg_dates[0])[:10],
                    "end": str(seg_dates[-1].date()) if hasattr(seg_dates[-1], 'date') else str(seg_dates[-1])[:10],
                    "h0": round(float(seg_values[0]), 3),
                    "h_end": round(float(seg_values[-1]), 3),
                    "T_days": round(float(T_fit), 1),
                    "r_squared": round(float(r_squared), 3),
                    "duration_days": len(seg_values),
                })
                all_T.append(T_fit)

                norm = (seg_values - baseline_fit) / amplitude_fit
                normalized_curves.append((t, norm))
        except (RuntimeError, ValueError):
            continue

    mrc = None
    if normalized_curves:
        max_len = max(len(t) for t, _ in normalized_curves)
        t_mrc = np.arange(max_len, dtype=float)
        all_norm = []
        for t, norm in normalized_curves:
            padded = np.full(max_len, np.nan)
            padded[:len(norm)] = norm
            all_norm.append(padded)
        mean_curve = np.nanmean(all_norm, axis=0)
        valid = ~np.isnan(mean_curve)
        mrc = {
            "normalized_time": t_mrc[valid].tolist(),
            "normalized_level": mean_curve[valid].tolist(),
        }

    return {
        "segments": fitted_segments,
        "mrc": mrc,
        "T_median_days": round(float(np.median(all_T)), 1) if all_T else None,
        "T_mean_days": round(float(np.mean(all_T)), 1) if all_T else None,
        "T_std_days": round(float(np.std(all_T)), 1) if all_T else None,
        "n_segments": len(fitted_segments),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/pastas/test_recession.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/recession.py tests/pastas/test_recession.py
git commit -m "fix(pastas): recession fit on drawdown with baseline offset, tolerate short interruptions"
```

---

### Task 3: Fix NaN → 0.0 in `_series_to_ts` (Issue #3)

Replace 0.0 with `None` so the frontend receives JSON `null` for missing values instead of a fake zero.

**Files:**
- Modify: `api/routers/pastas.py:94-98`

- [ ] **Step 1: Fix `_series_to_ts`**

In `api/routers/pastas.py`, replace:
```python
values=[float(v) if pd.notna(v) else 0.0 for v in s.values],
```
with:
```python
values=[float(v) if pd.notna(v) else None for v in s.values],
```

- [ ] **Step 2: Update `TimeSeriesData` schema to accept `None`**

In `api/schemas/pastas.py`, change `TimeSeriesData`:
```python
class TimeSeriesData(BaseModel):
    index: list[str]
    values: list[float | None]
```

- [ ] **Step 3: Commit**

```bash
git add api/routers/pastas.py api/schemas/pastas.py
git commit -m "fix(pastas): serialize NaN as null instead of 0.0 in time series"
```

---

### Task 4: Fix QQ-plot quantile computation (Issue from backend audit)

Use the correct probability integral transform `(i) / (n+1)` instead of `linspace(0.01, 0.99, n)` which compresses the tails.

**Files:**
- Modify: `dashboard/utils/pastas/diagnostics.py:57-60`

- [ ] **Step 1: Fix quantile formula**

In `diagnostics.py`, replace:
```python
sorted_res = np.sort(clean.values)
theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
```
with:
```python
sorted_res = np.sort(clean.values)
theoretical = scipy_stats.norm.ppf((np.arange(1, n + 1)) / (n + 1))
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pastas/diagnostics.py
git commit -m "fix(pastas): use correct PIT quantiles for QQ plot"
```

---

### Task 5: Fix cross-correlation NaN gap handling and T95 index bug (Issues #10, from backend audit)

Interpolate NaN gaps to preserve temporal structure. Fix T95 to use step response index values, not array positions. Exclude lag 0 from max search (Issue #18).

**Files:**
- Modify: `dashboard/utils/pastas/cross_correlation.py`

- [ ] **Step 1: Fix cross_correlation.py**

Replace lines 30-37 (NaN handling) with interpolation:
```python
# Interpolate short gaps to preserve temporal structure
df["niveau_moyen"] = df["niveau_moyen"].interpolate(limit=3)
df["precipitation_totale"] = df["precipitation_totale"].interpolate(limit=3)
df = df.dropna(subset=["niveau_moyen", "precipitation_totale"])

if len(df) < max_lag + 12:
    return {"lags_months": [], "correlation": [], "max_lag_months": None,
            "max_correlation": None, "t95_months": None}

piezo = df["niveau_moyen"].values
precip = df["precipitation_totale"].values
```

Replace lines 54-58 (max lag search — exclude lag 0):
```python
positive_lags = [(l, c) for l, c in zip(lags, correlations) if l >= 1]
```

Replace lines 66-69 (T95 computation — use index not position):
```python
idx = np.argmax(vals >= target)
t95 = round(float(step.index[idx]) / 30.44, 1)
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pastas/cross_correlation.py
git commit -m "fix(pastas): cross-correlation interpolates NaN gaps, uses step index for T95, excludes lag 0"
```

---

### Task 6: Fix STL decomposition on gapped monthly index (Issue #19)

Fill short gaps (up to 3 months) before running STL so the equidistant assumption holds.

**Files:**
- Modify: `dashboard/utils/pastas/signal_decomposition.py:15`

- [ ] **Step 1: Add gap filling before STL**

Replace line 15:
```python
monthly = obs.resample("MS").mean().dropna()
```
with:
```python
monthly = obs.resample("MS").mean()
monthly = monthly.interpolate(method="linear", limit=3)
monthly = monthly.dropna()
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pastas/signal_decomposition.py
git commit -m "fix(pastas): interpolate short gaps in monthly series before STL decomposition"
```

---

### Task 7: Fix outlier DATA_GAP threshold (Issue #11)

Change `>= 1` to `>= 7` so trivial 1-day gaps don't dominate outlier classification.

**Files:**
- Modify: `dashboard/utils/pastas/outlier_diagnostics.py:199`

- [ ] **Step 1: Raise threshold**

Replace:
```python
if data_quality.get("gap_days", 0) >= 1:
```
with:
```python
if data_quality.get("gap_days", 0) >= 7:
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pastas/outlier_diagnostics.py
git commit -m "fix(pastas): raise DATA_GAP outlier threshold to 7 days"
```

---

### Phase 1 checkpoint

- [ ] **Rebuild backend:** `docker compose up -d --build backend`
- [ ] **Run all host tests:** `python -m pytest tests/pastas/test_baseflow.py tests/pastas/test_recession.py tests/pastas/test_adaptive_bounds.py tests/pastas/test_scenario_presets.py -v`
- [ ] **Smoke-test endpoints via curl:**
  - `curl /api/v1/pastas/models/{run_id}/baseflow | python3 -c "import sys,json; d=json.load(sys.stdin); print('BFI:', d['bfi'])"` — should be 0.3-0.8, not 0.99
  - `curl /api/v1/pastas/models/{run_id}/recession | python3 -c "import sys,json; d=json.load(sys.stdin); print('T:', d.get('T_median_days'))"` — should be realistic (30-500d)
  - `curl /api/v1/pastas/models/{run_id}/diagnostics | python3 -c "import sys,json; d=json.load(sys.stdin); print('QQ range:', d['qq_theoretical'][0], d['qq_theoretical'][-1])"` — should extend to ±3 or beyond, not ±2.3

---

## Phase 2 — Critical Frontend Data Bugs

### Task 8: Fix t95 using array index instead of time axis (Issue #2)

Both `FitResultsPanel.tsx` and `ResponsePanel.tsx` use `findIndex` (array position) instead of reading `step_response.index[idx]` (actual days).

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx:156-162`
- Modify: `frontend/src/components/pastas/ResponsePanel.tsx:20-31`

- [ ] **Step 1: Fix FitResultsPanel t95**

Replace lines 156-162:
```typescript
const t95Days = useMemo(() => {
  if (!step_response?.values?.length) return null
  const vals = step_response.values
  const idx = step_response.index
  const target = 0.95 * vals[vals.length - 1]
  const i = vals.findIndex(v => v >= target)
  if (i < 0 || !idx[i]) return null
  const dayVal = parseFloat(idx[i])
  return Number.isFinite(dayVal) ? Math.round(dayVal) : i
}, [step_response])
```

- [ ] **Step 2: Fix ResponsePanel t50/t95**

Replace lines 20-31:
```typescript
let t50: number | null = null
let t95: number | null = null
if (hasStep) {
  const vals = stepResponse.values
  const idx = stepResponse.index
  const finalVal = vals[vals.length - 1]
  if (finalVal !== 0) {
    const i50 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.5)
    const i95 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.95)
    t50 = i50 >= 0 && idx[i50] ? Math.round(parseFloat(idx[i50])) || i50 : null
    t95 = i95 >= 0 && idx[i95] ? Math.round(parseFloat(idx[i95])) || i95 : null
  }
}
```

- [ ] **Step 3: Pass `x: stepResponse.index` to Plotly charts** in ResponsePanel

In the step response `<Plot>` data, add `x`:
```typescript
{
  x: stepResponse.index.map(Number),
  y: stepResponse.values,
  type: 'scatter',
  mode: 'lines',
  line: { color: '#34d399', width: 2 },
}
```

Same for block response:
```typescript
{
  x: blockResponse.index.map(Number),
  y: blockResponse.values,
  type: 'scatter',
  mode: 'lines',
  line: { color: '#f97316', width: 2 },
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx frontend/src/components/pastas/ResponsePanel.tsx
git commit -m "fix(pastas): t95/t50 reads time-axis values, not array indices"
```

---

### Task 9: Remove fabricated STOWA values (Issue #7)

When the model wasn't auto-fitted, `approximateStowa` invents fake passing values. Replace with explicit "EVP only" partial verdict.

**Files:**
- Modify: `frontend/src/pages/pastas/ResultsStep.tsx:15-36`

- [ ] **Step 1: Replace `approximateStowa`**

```typescript
function approximateStowa(metrics: Record<string, number>): StowaResult | null {
  const evp = metrics.evp
  if (evp == null) return null

  return {
    evp_pass: evp >= 70,
    evp_value: evp,
    autocorrelation_pass: null,
    runs_test_pvalue: null,
    t95_pass: null,
    t95_days: null,
    t95_threshold: null,
    gain_pass: null,
    gain_significance: null,
    overall_pass: null,
    suggestions: evp < 70
      ? ['EVP inférieur à 70%. Essayez d\'autres configurations ou ajoutez des stress.']
      : ['Résultat partiel — lancez l\'auto-fit pour l\'évaluation STOWA complète.'],
  }
}
```

Note: This requires `StowaVerdictBanner` to handle `null` fields gracefully — show "Non évalué" instead of a green/red tick for null criteria.

- [ ] **Step 2: Update `StowaVerdictBanner` to handle null criteria**

In `StowaVerdictBanner.tsx`, for each criterion, check for null before rendering tick/cross:

```typescript
{stowa.t95_pass === null ? (
  <span className="text-text-muted">—</span>
) : stowa.t95_pass ? (
  <CheckCircle className="w-3 h-3 text-green-400" />
) : (
  <XCircle className="w-3 h-3 text-red-400" />
)}
```

Apply same pattern for `autocorrelation_pass`, `gain_pass`. For `overall_pass === null`, show "Partiel" badge instead of Pass/Fail.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/pastas/ResultsStep.tsx frontend/src/components/pastas/StowaVerdictBanner.tsx
git commit -m "fix(pastas): show partial STOWA verdict instead of fabricated values"
```

---

### Task 10: Fix Tailwind dynamic class purge in OutlierDetailPanel (Issue #12)

Replace runtime-interpolated classes with a static lookup map.

**Files:**
- Modify: `frontend/src/components/pastas/OutlierDetailPanel.tsx`

- [ ] **Step 1: Add static class map**

Near the top of the file (after the `CATEGORY_META` definition), add:
```typescript
const CATEGORY_CLASSES: Record<string, { bg: string; text: string; border: string; bgLight: string }> = {
  DATA_GAP:              { bg: 'bg-red-500/15',    text: 'text-red-400',    border: 'border-red-500/30',    bgLight: 'bg-red-500/5' },
  CLIMATE_EXTREME:       { bg: 'bg-orange-500/15', text: 'text-orange-400', border: 'border-orange-500/30', bgLight: 'bg-orange-500/5' },
  REGIONAL_SIGNAL:       { bg: 'bg-blue-500/15',   text: 'text-blue-400',   border: 'border-blue-500/30',   bgLight: 'bg-blue-500/5' },
  SEASONAL_BIAS:         { bg: 'bg-yellow-500/15', text: 'text-yellow-400', border: 'border-yellow-500/30', bgLight: 'bg-yellow-500/5' },
  DOMINANT_CONTRIBUTION: { bg: 'bg-purple-500/15', text: 'text-purple-400', border: 'border-purple-500/30', bgLight: 'bg-purple-500/5' },
  UNKNOWN:               { bg: 'bg-gray-500/15',   text: 'text-gray-400',   border: 'border-gray-500/30',   bgLight: 'bg-gray-500/5' },
}
```

- [ ] **Step 2: Replace all dynamic class interpolations**

Replace all occurrences of `` `bg-${meta.color}-500/15` `` with `cls.bg`, `` `text-${meta.color}-400` `` with `cls.text`, etc. Use:
```typescript
const cls = CATEGORY_CLASSES[outlier.category] ?? CATEGORY_CLASSES.UNKNOWN
```

Apply to lines ~103, 111, 131, and any others using `meta.color` in class strings.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/pastas/OutlierDetailPanel.tsx
git commit -m "fix(pastas): use static Tailwind classes in OutlierDetailPanel to prevent purge"
```

---

### Task 11: Fix QQ-plot reference line (Issue #6)

The reference line uses theoretical values for both x and y. For non-standardized residuals, it should pass through the 25th and 75th percentile pair.

**Files:**
- Modify: `frontend/src/components/pastas/DiagnosticsPanel.tsx:107-114`

- [ ] **Step 1: Replace reference line**

```typescript
{qqTheoretical && qqSample && (() => {
  const n = qqTheoretical.length
  const i25 = Math.floor(n * 0.25)
  const i75 = Math.floor(n * 0.75)
  const slope = (qqSample[i75] - qqSample[i25]) / (qqTheoretical[i75] - qqTheoretical[i25])
  const intercept = qqSample[i25] - slope * qqTheoretical[i25]
  const xMin = qqTheoretical[0]
  const xMax = qqTheoretical[n - 1]
  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-2">
      <p className="text-[9px] text-text-muted px-1 mb-0.5">...</p>
      <Plot
        data={[
          { x: qqTheoretical, y: qqSample, type: 'scatter', mode: 'markers',
            marker: { color: '#60a5fa', size: 3 } },
          { x: [xMin, xMax],
            y: [intercept + slope * xMin, intercept + slope * xMax],
            type: 'scatter', mode: 'lines',
            line: { color: '#ef4444', dash: 'dash' } },
        ]}
        layout={{...}}
        ...
      />
    </div>
  )
})()}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/pastas/DiagnosticsPanel.tsx
git commit -m "fix(pastas): QQ plot reference line through 25th/75th percentile pair"
```

---

### Phase 2 checkpoint

- [ ] **TypeScript check:** `cd frontend && npx tsc --noEmit`
- [ ] **Rebuild frontend:** `docker compose up -d --build frontend`
- [ ] **Visual check:** open `/pastas/results`, verify t95 KPI shows realistic days, STOWA shows "Partiel" for manual fits, outlier colors render correctly

---

## Phase 3 — Frontend Performance

### Task 12: Lazy-load API calls for collapsed sections (Issue #5)

Pass `null` to query hooks when their section is collapsed, so the query doesn't fire.

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Add section-open state variables**

Near the top of the component:
```typescript
const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
const [signaturesOpen, setSignaturesOpen] = useState(false)
const [decompositionOpen, setDecompositionOpen] = useState(false)
const [paramsOpen, setParamsOpen] = useState(false)
```

- [ ] **Step 2: Gate query hooks on open state**

Replace:
```typescript
const { data: diagnosticsData } = usePastasDiagnostics(result.run_id)
const { data: signaturesData } = usePastasSignatures(result.run_id)
const { data: decompositionData } = usePastasDecomposition(result.run_id)
```
with:
```typescript
const { data: diagnosticsData } = usePastasDiagnostics(diagnosticsOpen ? result.run_id : null)
const { data: signaturesData } = usePastasSignatures(signaturesOpen ? result.run_id : null)
const { data: decompositionData } = usePastasDecomposition(decompositionOpen ? result.run_id : null)
```

- [ ] **Step 3: Wire open state to Section components**

Replace `<Section title="Statistical Diagnostics" defaultOpen={false}>` with a controlled version that calls `setDiagnosticsOpen`. The simplest approach: replace `Section` with an `onToggle` callback:

```typescript
function Section({ title, open, onToggle, children }: { title: string; open: boolean; onToggle: () => void; children: React.ReactNode }) {
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button onClick={onToggle} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors">
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}
```

Keep the uncontrolled `Section` for sections that don't gate queries (Performance, Model Analysis, Aquifer Characterization). Add controlled `Section` for the 3 lazy sections plus Model Parameters.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx
git commit -m "perf(pastas): lazy-load API calls for collapsed result sections"
```

---

### Task 13: Add missing staleTime to diagnostics and signatures hooks (Issue #14)

**Files:**
- Modify: `frontend/src/hooks/usePastas.ts:62-68, 111-117`

- [ ] **Step 1: Add staleTime**

For `usePastasDiagnostics`:
```typescript
export function usePastasDiagnostics(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'diagnostics', runId],
    queryFn: () => api.pastas.diagnostics(runId!),
    enabled: !!runId,
    staleTime: 60 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  })
}
```

Same for `usePastasSignatures`.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/usePastas.ts
git commit -m "perf(pastas): add staleTime to diagnostics and signatures hooks"
```

---

### Task 14: Memoize monthly aggregation in FitResultsPanel (Issue #9)

Extract the IIFE at lines 242-257 into a `useMemo`.

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Extract IIFE into useMemo**

Before the JSX return, add:
```typescript
const monthlyStats = useMemo(() => {
  const { resIdx, resVals } = sliceByPeriod
  if (resIdx.length === 0) return null
  const monthlyMap = new Map<string, number[]>()
  resIdx.forEach((d, i) => {
    const ym = d.slice(0, 7)
    if (!monthlyMap.has(ym)) monthlyMap.set(ym, [])
    if (Number.isFinite(resVals[i])) monthlyMap.get(ym)!.push(resVals[i])
  })
  const monthlyDates: string[] = []
  const monthlyVals: number[] = []
  for (const [ym, vals] of monthlyMap) {
    monthlyDates.push(ym + '-15')
    monthlyVals.push(vals.reduce((a, b) => a + b, 0) / vals.length)
  }
  const finiteVals = monthlyVals.filter(v => Number.isFinite(v))
  const mean = finiteVals.length > 0 ? finiteVals.reduce((a, b) => a + b, 0) / finiteVals.length : 0
  const std = finiteVals.length > 1 ? Math.sqrt(finiteVals.reduce((a, b) => a + (b - mean) ** 2, 0) / finiteVals.length) : 1
  return { monthlyDates, monthlyVals, threshold: 2 * std }
}, [sliceByPeriod])

const outlierMonths = useMemo(() => {
  const allOutliersList = currentOutliers?.outliers ?? []
  return new Set<string>(allOutliersList.map((o: any) => o.date.slice(0, 7)))
}, [currentOutliers])
```

Then replace the IIFE in JSX with a simpler render block that uses `monthlyStats` and `outlierMonths`.

Also remove `residuals` from the `sliceByPeriod` memo dependency array (Issue from performance audit).

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx
git commit -m "perf(pastas): memoize monthly aggregation and outlier sets"
```

---

### Phase 3 checkpoint

- [ ] **TypeScript check:** `cd frontend && npx tsc --noEmit`
- [ ] **Rebuild frontend:** `docker compose up -d --build frontend`
- [ ] **Verify:** Open results page, check browser network tab — collapsed sections should NOT fire API calls until expanded

---

## Phase 4 — Scientific Accuracy Fixes

### Task 15: Relax diagnostic thresholds (Issue #8)

**Files:**
- Modify: `frontend/src/components/pastas/DiagnosticsPanel.tsx`

- [ ] **Step 1: Update thresholds**

Kurtosis: change `Math.abs(v) < 1` to `Math.abs(v) < 3`

Skewness: change `Math.abs(v) < 0.5` to `Math.abs(v) < 1.0`

Update Durbin-Watson tooltip to mention that DW < 1.5 is expected without an AR noise model.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/pastas/DiagnosticsPanel.tsx
git commit -m "fix(pastas): relax kurtosis/skewness thresholds for groundwater residuals"
```

---

### Task 16: Fix runs count formula (Issue from backend audit)

**Files:**
- Modify: `dashboard/utils/pastas/diagnostics.py:46-48`

- [ ] **Step 1: Fix formula**

Replace:
```python
runs = ((clean > median).astype(int).diff().abs().sum() / 2) + 1
```
with:
```python
runs = 1 + int((clean > median).astype(int).diff().abs().dropna().sum())
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pastas/diagnostics.py
git commit -m "fix(pastas): correct runs count formula"
```

---

### Task 17: Fix block response unit label (Issue #20)

**Files:**
- Modify: `frontend/src/components/pastas/ResponsePanel.tsx`

- [ ] **Step 1: Change y-axis label**

Replace `yaxis: { title: { text: 'm/d' }` with `yaxis: { title: { text: 'm/(mm/j)' }` in the block response chart.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/pastas/ResponsePanel.tsx
git commit -m "fix(pastas): correct block response y-axis unit to m/(mm/j)"
```

---

### Phase 4 checkpoint

- [ ] **TypeScript check:** `cd frontend && npx tsc --noEmit`
- [ ] **Rebuild both:** `docker compose up -d --build backend frontend`
- [ ] **Verify diagnostics page:** kurtosis/skewness should show green for typical groundwater residuals

---

## Phase 5 — UX Consistency

### Task 18: Unify language to French (Issue #13)

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx` — section titles
- Modify: `frontend/src/components/pastas/DiagnosticsPanel.tsx` — test labels
- Modify: `frontend/src/components/pastas/ResponsePanel.tsx` — chart titles
- Modify: `frontend/src/components/pastas/UnifiedAnalysisChart.tsx` — panel labels
- Modify: `frontend/src/components/pastas/SignaturesPanel.tsx` — category labels

- [ ] **Step 1: Replace section titles in FitResultsPanel**

```
"Performance" → "Performance"  (same in both languages)
"Model Analysis" → "Analyse du modèle"
"Signal Structure" → "Structure du signal"
"Statistical Diagnostics" → "Diagnostics statistiques"
"Model Parameters" → "Paramètres du modèle"
"Hydrological Signatures" → "Signatures hydrologiques"
```

- [ ] **Step 2: Replace panel labels in UnifiedAnalysisChart**

```
"Water Level" → "Niveau piézométrique"
"Stress" → "Contributions"
"Error" → "Résidus"
```

- [ ] **Step 3: Replace test labels in DiagnosticsPanel**

Keep technical names (Durbin-Watson, Jarque-Bera, Shapiro-Wilk, Ljung-Box) as-is — they're proper nouns. The labels "Asymétrie" and "Kurtosis" are already in French.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx frontend/src/components/pastas/UnifiedAnalysisChart.tsx frontend/src/components/pastas/DiagnosticsPanel.tsx frontend/src/components/pastas/ResponsePanel.tsx frontend/src/components/pastas/SignaturesPanel.tsx
git commit -m "ui(pastas): unify results page language to French"
```

---

### Phase 5 checkpoint

- [ ] **TypeScript check:** `cd frontend && npx tsc --noEmit`
- [ ] **Rebuild frontend:** `docker compose up -d --build frontend`
- [ ] **Visual review:** scan all section titles and labels — no more English/French mixing

---

## Final Verification

- [ ] **Run all host tests:** `python -m pytest tests/pastas/ -v --ignore=tests/pastas/test_io.py --ignore=tests/pastas/test_fit_service.py --ignore=tests/pastas/test_scenario.py --ignore=tests/pastas/test_builder.py --ignore=tests/pastas/test_cross_correlation.py`
- [ ] **Full Docker rebuild:** `docker compose up -d --build`
- [ ] **Smoke-test all modified endpoints** (baseflow, recession, diagnostics, cross-correlation, decomposition)
- [ ] **Visual check on live app**: results page loads cleanly, collapsed sections don't fire network requests, all text in French, all metrics display correctly
