# Pumping Detection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Pumping Detection" page to Junon that runs a 3-layer unsupervised pipeline (Pastas physics + ML/XAI + embeddings) to detect hidden groundwater pumping from piezometric data.

**Architecture:** Layer 1 (Pastas residuals + BEAST/PELT change points) runs first, producing clean period masks. Layer 2 (TFT trained on clean periods + XAI attribution drift) and Layer 3 (SoftCLT/TS2Vec embedding drift + twin stations) run in parallel. A fusion engine combines per-month scores from all available layers into a concordance-based suspicion score. Layer 3 is optional (encoder not yet available).

**Tech Stack:** Python 3.12, pastas, ruptures, Rbeast (optional), statsmodels, Darts/PyTorch, Captum, FastAPI + SSE, React 19 + React Query 5 + Plotly.js

**Spec:** `docs/superpowers/specs/2026-03-12-pumping-detection-design.md`

---

## File Structure

### New files to create

**Backend utils** (`dashboard/utils/pumping_detection/`):
- `__init__.py` — Exports all layer classes
- `pastas_layer.py` — `PastasAnalyzer`: calibration, residuals, ACF/PACF. Extends existing `PastasWrapper`.
- `changepoint.py` — `ChangepointDetector`: BEAST (if available) + PELT via ruptures.
- `clean_period.py` — `CleanPeriodSelector`: identifies clean windows from Pastas residuals + Ljung-Box.
- `ml_layer.py` — `MLAnalyzer`: train transient TFT on clean periods, predict full series.
- `xai_layer.py` — `XAIDriftAnalyzer`: compute IG/SHAP/attention per window, compute drift metrics (JS, Spearman, FA).
- `embedding_layer.py` — `EmbeddingAnalyzer`: SoftCLT drift scores + twin station search. Stub for Phase 1.
- `fusion.py` — `FusionEngine`: monthly grid alignment, concordance scoring, window merging.
- `bnpe_client.py` — `BNPEClient`: Hub'Eau Prélèvements API with caching.
- `pipeline.py` — `PumpingDetectionPipeline`: orchestrates all layers with SSE events + cancellation.

**API** (`api/`):
- `routers/pumping_detection.py` — FastAPI router with SSE streaming.
- `schemas/pumping_detection.py` — Pydantic request/response models.

**Frontend** (`frontend/src/`):
- `hooks/usePumpingDetection.ts` — React Query mutation + SSE hook.
- `pages/PumpingDetectionPage.tsx` — Main page layout.
- `components/pumping/AnnotatedChroniquePlot.tsx` — Plotly chart with suspect window overlays.
- `components/pumping/PastasPanel.tsx` — Residuals + ACF/PACF + changepoints.
- `components/pumping/XAIDriftPanel.tsx` — Attribution heatmap + divergence curve.
- `components/pumping/EmbeddingPanel.tsx` — UMAP trajectory + twin stations (or "not available" state).
- `components/pumping/VerdictPanel.tsx` — Fusion score + suspect period summary.

**Tests** (`tests/pumping_detection/`):
- `test_pastas_layer.py`
- `test_changepoint.py`
- `test_clean_period.py`
- `test_xai_layer.py`
- `test_fusion.py`
- `test_bnpe_client.py`
- `test_pipeline.py`

### Files to modify
- `frontend/src/lib/api.ts` — Add `pumpingDetection` namespace.
- `frontend/src/App.tsx` (or router config) — Add `/pumping-detection` route.
- `api/routers/__init__.py` — Register new router.
- `pyproject.toml` — Add `ruptures` dependency.

---

## Chunk 1: Dependencies & Layer 1 — Physics (Pastas + Change Points)

### Task 1: Validate and add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Test dependency installation in Docker**

```bash
docker compose exec backend pip install ruptures
docker compose exec backend python -c "import ruptures; print(ruptures.__version__)"
docker compose exec backend pip install Rbeast
docker compose exec backend python -c "import Rbeast; print('BEAST OK')"
```

If Rbeast fails, note it — we'll proceed with PELT only.

- [ ] **Step 2: Add `ruptures` to pyproject.toml**

Add `ruptures` to the dependencies list in `pyproject.toml`. Only add `Rbeast` if step 1 succeeded.

- [ ] **Step 3: Rebuild Docker image**

```bash
docker compose up -d --build
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(pumping): add ruptures dependency for change point detection"
```

### Task 2: PastasAnalyzer — residual extraction + ACF

**Files:**
- Create: `dashboard/utils/pumping_detection/__init__.py`
- Create: `dashboard/utils/pumping_detection/pastas_layer.py`
- Test: `tests/pumping_detection/test_pastas_layer.py`

**Reference:** Existing `PastasWrapper` at `dashboard/utils/counterfactual/pastas_validation.py:39`. Reuse its `fit()` and `predict()` methods.

- [ ] **Step 1: Write failing test for PastasAnalyzer**

```python
# tests/pumping_detection/test_pastas_layer.py
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_series(n_days=1000, seed=42):
    """Create synthetic piézo + precip + ETP with a known 'pumping' dip."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_days, freq="D")
    # Natural signal: seasonal + trend + noise
    t = np.arange(n_days)
    natural = 50.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.3, n_days)
    # Inject pumping dip: days 400-600
    pumping = np.zeros(n_days)
    pumping[400:600] = -1.5  # 1.5m drawdown
    piezo = pd.Series(natural + pumping, index=dates, name="piezo")
    precip = pd.Series(3.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1, n_days),
                       index=dates, name="precip").clip(lower=0)
    etp = pd.Series(2.0 + 1.5 * np.sin(2 * np.pi * (t - 90) / 365.25) + rng.normal(0, 0.3, n_days),
                    index=dates, name="etp").clip(lower=0)
    return piezo, precip, etp


class TestPastasAnalyzer:
    def test_analyze_returns_expected_keys(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        assert "residuals" in result
        assert "acf_stats" in result
        assert "pastas_fit_quality" in result
        assert isinstance(result["residuals"], pd.Series)
        assert len(result["residuals"]) > 0

    def test_acf_stats_contain_ljung_box(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        acf = result["acf_stats"]
        assert "acf_values" in acf
        assert "pacf_values" in acf
        assert "ljung_box_pvalue" in acf

    def test_fit_quality_contains_evp(self):
        from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer

        piezo, precip, etp = _make_synthetic_series()
        analyzer = PastasAnalyzer()
        result = analyzer.analyze(piezo, precip, etp)

        quality = result["pastas_fit_quality"]
        assert "evp" in quality
        assert 0 <= quality["evp"] <= 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/pumping_detection/test_pastas_layer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.utils.pumping_detection'`

- [ ] **Step 3: Implement PastasAnalyzer**

```python
# dashboard/utils/pumping_detection/__init__.py
"""Pumping detection pipeline — unsupervised 3-layer hybrid detection."""

# dashboard/utils/pumping_detection/pastas_layer.py
"""Layer 1: Pastas TFN residual analysis + ACF/PACF diagnostics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from dashboard.utils.counterfactual.pastas_validation import PastasWrapper

logger = logging.getLogger(__name__)


class PastasAnalyzer:
    """Calibrate Pastas recharge-only model and extract residual diagnostics."""

    def __init__(self, response_function: str = "Gamma", noise_model: bool = True):
        self.response_function = response_function
        self.noise_model = noise_model
        self._wrapper = PastasWrapper()

    def analyze(
        self,
        piezo: pd.Series,
        precip: pd.Series,
        etp: pd.Series,
        max_acf_lag: int = 30,
    ) -> dict[str, Any]:
        """Run full Pastas analysis: fit, extract residuals, compute ACF.

        Returns dict with keys: residuals, acf_stats, pastas_fit_quality, modeled.
        """
        tmin = piezo.index.min()
        tmax = piezo.index.max()

        # Fit Pastas
        self._wrapper.fit(piezo, precip, etp, tmin=tmin, tmax=tmax)

        # Get modeled values and residuals
        modeled = self._wrapper.predict(tmin=tmin, tmax=tmax)
        # Align on common index
        common_idx = piezo.index.intersection(modeled.index)
        residuals = piezo.loc[common_idx] - modeled.loc[common_idx]
        residuals.name = "residuals"

        # Fit quality
        ss_res = (residuals ** 2).sum()
        ss_tot = ((piezo.loc[common_idx] - piezo.loc[common_idx].mean()) ** 2).sum()
        evp = (1 - ss_res / ss_tot) * 100 if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / len(residuals)))

        # ACF / PACF
        nlags = min(max_acf_lag, len(residuals) // 2 - 1)
        acf_values = acf(residuals.dropna(), nlags=nlags, fft=True)
        pacf_values = pacf(residuals.dropna(), nlags=nlags)

        # Ljung-Box test
        lb_result = acorr_ljungbox(residuals.dropna(), lags=[nlags], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])

        return {
            "residuals": residuals,
            "modeled": modeled.loc[common_idx],
            "acf_stats": {
                "acf_values": acf_values.tolist(),
                "pacf_values": pacf_values.tolist(),
                "ljung_box_pvalue": lb_pvalue,
                "nlags": nlags,
            },
            "pastas_fit_quality": {
                "evp": float(evp),
                "rmse": rmse,
                "n_observations": len(common_idx),
            },
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/pumping_detection/test_pastas_layer.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pumping_detection/ tests/pumping_detection/
git commit -m "feat(pumping): add PastasAnalyzer with residual extraction and ACF diagnostics"
```

### Task 3: ChangepointDetector — PELT + optional BEAST

**Files:**
- Create: `dashboard/utils/pumping_detection/changepoint.py`
- Test: `tests/pumping_detection/test_changepoint.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pumping_detection/test_changepoint.py
import numpy as np
import pandas as pd
import pytest


def _make_residuals_with_changepoints():
    """Residuals with known mean shift at day 200 and 400."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=600, freq="D")
    values = rng.normal(0, 0.3, 600)
    values[200:400] -= 1.0  # Mean shift = pumping signal in residuals
    return pd.Series(values, index=dates, name="residuals")


class TestChangepointDetector:
    def test_pelt_detects_known_changepoints(self):
        from dashboard.utils.pumping_detection.changepoint import ChangepointDetector

        residuals = _make_residuals_with_changepoints()
        detector = ChangepointDetector(method="pelt", min_segment_length=60)
        result = detector.detect(residuals)

        assert "changepoints" in result
        assert len(result["changepoints"]) >= 2
        # Changepoints should be near day 200 and 400
        cp_indices = [cp["index"] for cp in result["changepoints"]]
        assert any(abs(cp - 200) < 30 for cp in cp_indices)
        assert any(abs(cp - 400) < 30 for cp in cp_indices)

    def test_returns_empty_on_flat_signal(self):
        from dashboard.utils.pumping_detection.changepoint import ChangepointDetector

        rng = np.random.default_rng(42)
        dates = pd.date_range("2015-01-01", periods=600, freq="D")
        flat = pd.Series(rng.normal(0, 0.3, 600), index=dates)
        detector = ChangepointDetector(method="pelt", min_segment_length=60)
        result = detector.detect(flat)

        assert len(result["changepoints"]) <= 1  # At most the end point
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/pumping_detection/test_changepoint.py -v
```

- [ ] **Step 3: Implement ChangepointDetector**

```python
# dashboard/utils/pumping_detection/changepoint.py
"""Change point detection on Pastas residuals via PELT and optional BEAST."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import Rbeast
    BEAST_AVAILABLE = True
except ImportError:
    BEAST_AVAILABLE = False


class ChangepointDetector:
    """Detect change points in residual time series."""

    def __init__(self, method: str = "pelt", min_segment_length: int = 90):
        self.method = method
        self.min_segment_length = min_segment_length

    def detect(self, residuals: pd.Series) -> dict[str, Any]:
        """Detect changepoints. Returns dict with 'changepoints' list and 'method_used'."""
        results: dict[str, Any] = {"changepoints": [], "method_used": []}

        if self.method in ("pelt", "both"):
            pelt_cps = self._run_pelt(residuals)
            results["changepoints"].extend(pelt_cps)
            results["method_used"].append("pelt")

        if self.method in ("beast", "both") and BEAST_AVAILABLE:
            beast_cps = self._run_beast(residuals)
            results["changepoints"].extend(beast_cps)
            results["method_used"].append("beast")
        elif self.method == "beast" and not BEAST_AVAILABLE:
            logger.warning("BEAST not available, falling back to PELT")
            pelt_cps = self._run_pelt(residuals)
            results["changepoints"].extend(pelt_cps)
            results["method_used"].append("pelt_fallback")

        return results

    def _run_pelt(self, residuals: pd.Series) -> list[dict[str, Any]]:
        """PELT change point detection via ruptures."""
        import ruptures

        signal = residuals.dropna().values
        algo = ruptures.Pelt(model="rbf", min_size=self.min_segment_length).fit(signal)
        # pen=3 is a reasonable default for this application
        breakpoints = algo.predict(pen=3)

        changepoints = []
        for bp in breakpoints[:-1]:  # Last one is always len(signal)
            date = residuals.dropna().index[min(bp, len(signal) - 1)]
            changepoints.append({
                "index": int(bp),
                "date": str(date.date()),
                "method": "pelt",
                "confidence": None,  # PELT doesn't provide confidence
            })
        return changepoints

    def _run_beast(self, residuals: pd.Series) -> list[dict[str, Any]]:
        """BEAST Bayesian change point detection."""
        signal = residuals.dropna().values
        result = Rbeast.beast(signal, season="none")

        changepoints = []
        if hasattr(result, "trend") and hasattr(result.trend, "cp"):
            for i, cp_idx in enumerate(result.trend.cp):
                if np.isnan(cp_idx):
                    continue
                idx = int(cp_idx)
                date = residuals.dropna().index[min(idx, len(signal) - 1)]
                prob = float(result.trend.cpPr[i]) if hasattr(result.trend, "cpPr") else None
                changepoints.append({
                    "index": idx,
                    "date": str(date.date()),
                    "method": "beast",
                    "confidence": prob,
                })
        return changepoints
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pumping_detection/test_changepoint.py -v
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pumping_detection/changepoint.py tests/pumping_detection/test_changepoint.py
git commit -m "feat(pumping): add ChangepointDetector with PELT and optional BEAST"
```

### Task 4: CleanPeriodSelector

**Files:**
- Create: `dashboard/utils/pumping_detection/clean_period.py`
- Test: `tests/pumping_detection/test_clean_period.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pumping_detection/test_clean_period.py
import numpy as np
import pandas as pd
import pytest


def _make_residuals_clean_and_dirty():
    """Residuals: clean Jan-Jun, dirty Jul-Dec (high values + autocorrelation)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=730, freq="D")  # 2 years
    values = rng.normal(0, 0.2, 730)
    # Make Jul-Dec dirty each year (days 181-365, 546-730)
    for start in [181, 546]:
        end = min(start + 184, 730)
        values[start:end] = np.cumsum(rng.normal(-0.05, 0.3, end - start))
    return pd.Series(values, index=dates, name="residuals")


class TestCleanPeriodSelector:
    def test_select_returns_boolean_mask(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        assert "mask" in result
        assert len(result["mask"]) == len(residuals)
        assert result["mask"].dtype == bool

    def test_clean_periods_mostly_in_first_half(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        mask = result["mask"]
        first_half_clean = mask[:365].sum()
        second_half_clean = mask[365:].sum()
        # First half should have more clean days
        assert first_half_clean > second_half_clean

    def test_stats_include_total_clean_days(self):
        from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector

        residuals = _make_residuals_clean_and_dirty()
        selector = CleanPeriodSelector()
        result = selector.select(residuals)

        assert "n_clean_days" in result
        assert "pct_clean" in result
        assert result["n_clean_days"] > 0
        assert 0 < result["pct_clean"] <= 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/pumping_detection/test_clean_period.py -v
```

- [ ] **Step 3: Implement CleanPeriodSelector**

```python
# dashboard/utils/pumping_detection/clean_period.py
"""Clean period identification from Pastas residuals.

Algorithm (from spec):
1. Compute amplitude threshold T = n_sigma * std(residuals).
2. Rolling Ljung-Box test (180-day window, max lag 30, alpha 0.05).
3. Day is clean if |r(t)| < T AND Ljung-Box p-value > alpha.
4. Merge contiguous clean days, discard windows < min_window_days.
5. If total < min_total_days, relax threshold iteratively.
6. Fallback: seasonal heuristic (Nov 1 - Mar 31).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


class CleanPeriodSelector:
    """Select temporal windows where Pastas explains the data well."""

    def __init__(
        self,
        n_sigma: float = 2.0,
        rolling_window: int = 180,
        max_lag: int = 30,
        alpha: float = 0.05,
        min_window_days: int = 90,
        min_total_days: int = 365,
    ):
        self.n_sigma = n_sigma
        self.rolling_window = rolling_window
        self.max_lag = max_lag
        self.alpha = alpha
        self.min_window_days = min_window_days
        self.min_total_days = min_total_days

    def select(self, residuals: pd.Series) -> dict[str, Any]:
        """Identify clean periods in residuals.

        Returns dict with: mask (bool Series), n_clean_days, pct_clean,
        windows (list of (start, end) tuples), method.
        """
        residuals_clean = residuals.dropna()

        # Try with increasing sigma thresholds
        for sigma_mult in [self.n_sigma, 3.0, 4.0]:
            mask = self._compute_mask(residuals_clean, sigma_mult)
            windows = self._merge_windows(mask)
            total_clean = mask.sum()
            if total_clean >= self.min_total_days:
                return self._build_result(residuals, mask, windows, f"auto_{sigma_mult}sigma")

        # Fallback: seasonal heuristic (Nov-Mar = clean)
        mask = self._seasonal_heuristic(residuals)
        windows = self._merge_windows(mask)
        return self._build_result(residuals, mask, windows, "seasonal_heuristic")

    def _compute_mask(self, residuals: pd.Series, sigma_mult: float) -> pd.Series:
        """Compute clean mask: amplitude + Ljung-Box criteria."""
        threshold = sigma_mult * residuals.std()

        # Amplitude criterion
        amp_clean = residuals.abs() < threshold

        # Rolling Ljung-Box criterion
        lb_clean = pd.Series(False, index=residuals.index)
        half_win = self.rolling_window // 2

        for i in range(half_win, len(residuals) - half_win, 30):  # Step by 30 for speed
            window = residuals.iloc[max(0, i - half_win):i + half_win]
            if len(window) < self.max_lag + 1:
                continue
            try:
                lb = acorr_ljungbox(window, lags=[self.max_lag], return_df=True)
                pval = lb["lb_pvalue"].iloc[0]
                if pval > self.alpha:
                    # Mark the whole step range as clean
                    start_idx = max(0, i - 15)
                    end_idx = min(len(residuals), i + 15)
                    lb_clean.iloc[start_idx:end_idx] = True
            except Exception:
                continue

        return amp_clean & lb_clean

    def _seasonal_heuristic(self, residuals: pd.Series) -> pd.Series:
        """Nov 1 - Mar 31 presumed clean (no agricultural pumping)."""
        months = residuals.index.month
        return pd.Series((months >= 11) | (months <= 3), index=residuals.index)

    def _merge_windows(self, mask: pd.Series) -> list[tuple[str, str]]:
        """Merge contiguous clean days into windows, discard < min_window_days."""
        windows = []
        in_window = False
        start = None

        for i, (date, is_clean) in enumerate(mask.items()):
            if is_clean and not in_window:
                start = date
                in_window = True
            elif not is_clean and in_window:
                duration = (date - start).days
                if duration >= self.min_window_days:
                    windows.append((str(start.date()), str(date.date())))
                in_window = False

        # Close last window
        if in_window and start is not None:
            duration = (mask.index[-1] - start).days
            if duration >= self.min_window_days:
                windows.append((str(start.date()), str(mask.index[-1].date())))

        return windows

    def _build_result(
        self, residuals: pd.Series, mask: pd.Series, windows: list, method: str
    ) -> dict[str, Any]:
        # Reindex mask to full residuals index (including NaN positions)
        full_mask = pd.Series(False, index=residuals.index)
        full_mask.loc[mask.index] = mask
        n_clean = int(full_mask.sum())
        return {
            "mask": full_mask,
            "n_clean_days": n_clean,
            "pct_clean": round(100 * n_clean / len(residuals), 1),
            "windows": windows,
            "method": method,
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pumping_detection/test_clean_period.py -v
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pumping_detection/clean_period.py tests/pumping_detection/test_clean_period.py
git commit -m "feat(pumping): add CleanPeriodSelector with Ljung-Box + seasonal fallback"
```

---

## Chunk 2: Layer 2 — ML + XAI Drift Detection

### Task 5: XAIDriftAnalyzer — attribution computation + drift metrics

**Files:**
- Create: `dashboard/utils/pumping_detection/xai_layer.py`
- Test: `tests/pumping_detection/test_xai_layer.py`

**Reference:** Existing explainability modules at `dashboard/utils/explainability/gradients.py` (Captum IG), `dashboard/utils/explainability/feature_importance.py` (SHAP). Reuse their computation functions.

- [ ] **Step 1: Write failing test for drift metrics (pure math, no model needed)**

```python
# tests/pumping_detection/test_xai_layer.py
import numpy as np
import pytest


class TestDriftMetrics:
    def test_js_divergence_identical_distributions(self):
        from dashboard.utils.pumping_detection.xai_layer import js_divergence

        p = np.array([0.5, 0.3, 0.2])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_js_divergence_different_distributions(self):
        from dashboard.utils.pumping_detection.xai_layer import js_divergence

        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        jsd = js_divergence(p, q)
        assert 0 < jsd <= np.log(2)  # JS is bounded by ln(2)

    def test_feature_agreement_perfect_overlap(self):
        from dashboard.utils.pumping_detection.xai_layer import feature_agreement

        ranking_a = [0, 1, 2, 3, 4]
        ranking_b = [0, 1, 2, 3, 4]
        assert feature_agreement(ranking_a, ranking_b, k=3) == 1.0

    def test_feature_agreement_no_overlap(self):
        from dashboard.utils.pumping_detection.xai_layer import feature_agreement

        ranking_a = [0, 1, 2, 3, 4]
        ranking_b = [4, 3, 2, 1, 0]
        fa = feature_agreement(ranking_a, ranking_b, k=2)
        assert fa == 0.0  # Top-2 of a = {0,1}, top-2 of b = {4,3}

    def test_compute_window_drift(self):
        from dashboard.utils.pumping_detection.xai_layer import compute_window_drift

        # Simulate attributions: 3 features, reference vs test
        ref_attrs = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2]])
        test_attrs = np.array([[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.2, 0.1, 0.7]])
        drift = compute_window_drift(ref_attrs, test_attrs)

        assert "js_divergence" in drift
        assert "spearman_corr" in drift
        assert "feature_agreement" in drift
        assert drift["js_divergence"] > 0.1  # Distributions are quite different
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/pumping_detection/test_xai_layer.py -v
```

- [ ] **Step 3: Implement drift metrics**

```python
# dashboard/utils/pumping_detection/xai_layer.py
"""Layer 2: XAI attribution drift analysis.

Computes per-window feature attributions (IG, SHAP, attention) and measures
drift from clean-period baseline using JS divergence, Spearman correlation,
and Feature Agreement.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    # Ensure valid probability distributions
    p = np.abs(p) + 1e-12
    q = np.abs(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)  # scipy returns sqrt(JSD)


def feature_agreement(ranking_a: list[int], ranking_b: list[int], k: int = 3) -> float:
    """Fraction of top-K features that overlap between two rankings."""
    top_a = set(ranking_a[:k])
    top_b = set(ranking_b[:k])
    return len(top_a & top_b) / k


def compute_window_drift(
    ref_attributions: np.ndarray,
    test_attributions: np.ndarray,
    k: int = 3,
) -> dict[str, float]:
    """Compute drift metrics between reference and test attribution matrices.

    Args:
        ref_attributions: (n_ref_samples, n_features) — attributions from clean windows.
        test_attributions: (n_test_samples, n_features) — attributions from test window.
        k: top-K for feature agreement.

    Returns:
        Dict with js_divergence, spearman_corr, feature_agreement.
    """
    # Average absolute attributions per feature
    ref_importance = np.abs(ref_attributions).mean(axis=0)
    test_importance = np.abs(test_attributions).mean(axis=0)

    # JS divergence
    jsd = js_divergence(ref_importance, test_importance)

    # Spearman rank correlation
    corr, _ = spearmanr(ref_importance, test_importance)

    # Feature agreement
    ref_ranking = np.argsort(-ref_importance).tolist()
    test_ranking = np.argsort(-test_importance).tolist()
    k_actual = min(k, len(ref_importance))
    fa = feature_agreement(ref_ranking, test_ranking, k=k_actual)

    return {
        "js_divergence": float(jsd),
        "spearman_corr": float(corr) if not np.isnan(corr) else 0.0,
        "feature_agreement": float(fa),
    }


class XAIDriftAnalyzer:
    """Compute XAI attributions and drift metrics across time windows.

    Uses existing explainability modules from dashboard/utils/explainability/.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        window_size: int = 90,
        stride: int = 30,
    ):
        self.methods = methods or ["integrated_gradients"]
        self.window_size = window_size
        self.stride = stride

    def analyze(
        self,
        model: Any,
        series: Any,  # Darts TimeSeries
        covariates: Any,  # Darts TimeSeries
        clean_mask: Any,  # pd.Series[bool]
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Compute attributions on all windows, then drift from clean baseline.

        Returns dict with: attributions, drift_metrics, feature_names.
        """
        from dashboard.utils.explainability.gradients import compute_integrated_gradients

        # Compute attributions per sliding window
        all_attributions = []
        window_dates = []
        n_steps = len(series)

        for start in range(0, n_steps - self.window_size, self.stride):
            end = start + self.window_size
            try:
                # Slice series to window, then compute IG
                window_series = series[start:end]
                window_cov = covariates[start:end] if covariates is not None else None
                attrs = compute_integrated_gradients(
                    model, window_series,
                    past_covariates=window_cov,
                    input_chunk_length=min(self.window_size, 30),
                )
                all_attributions.append(attrs)
                mid_date = series.time_index[start + self.window_size // 2]
                window_dates.append(str(mid_date.date()))
            except Exception as e:
                logger.warning(f"IG failed for window {start}-{end}: {e}")
                continue

        if not all_attributions:
            return {"attributions": [], "drift_metrics": [], "feature_names": feature_names}

        attributions = np.array(all_attributions)

        # Separate clean vs all windows using mask
        clean_indices = []
        for i, date_str in enumerate(window_dates):
            # Check if majority of window is clean
            window_start = i * self.stride
            window_end = window_start + self.window_size
            if window_end <= len(clean_mask):
                pct_clean = clean_mask.iloc[window_start:window_end].mean()
                if pct_clean > 0.7:
                    clean_indices.append(i)

        if not clean_indices:
            logger.warning("No clean windows found for XAI baseline")
            return {
                "attributions": attributions.tolist(),
                "drift_metrics": [],
                "feature_names": feature_names,
                "window_dates": window_dates,
            }

        ref_attrs = attributions[clean_indices]

        # Compute drift for each window
        drift_metrics = []
        for i in range(len(attributions)):
            drift = compute_window_drift(ref_attrs, attributions[i:i+1])
            drift["window_date"] = window_dates[i]
            drift["is_clean"] = i in clean_indices
            drift_metrics.append(drift)

        return {
            "attributions": attributions.tolist(),
            "drift_metrics": drift_metrics,
            "feature_names": feature_names,
            "window_dates": window_dates,
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pumping_detection/test_xai_layer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pumping_detection/xai_layer.py tests/pumping_detection/test_xai_layer.py
git commit -m "feat(pumping): add XAIDriftAnalyzer with JS divergence, Spearman, and Feature Agreement"
```

### Task 6: MLAnalyzer — transient TFT training on clean periods

**Files:**
- Create: `dashboard/utils/pumping_detection/ml_layer.py`

**Reference:** `dashboard/utils/model_factory.py` for `ModelFactory`, `dashboard/utils/training.py` for training patterns.

- [ ] **Step 1: Write failing test**

```python
# tests/pumping_detection/test_ml_layer.py
import numpy as np
import pandas as pd
import pytest

from dashboard.utils.pumping_detection.ml_layer import MLAnalyzer


class TestFilterToClean:
    """Unit tests for _filter_to_clean (no GPU needed)."""

    def test_longest_clean_segment(self):
        """Should return the longest contiguous True segment."""
        analyzer = MLAnalyzer()
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        mask = pd.Series(False, index=dates)
        # Two clean segments: 20 days and 50 days
        mask.iloc[10:30] = True   # 20 days
        mask.iloc[40:90] = True   # 50 days (longest)

        # Mock a simple TimeSeries-like with slice
        from unittest.mock import MagicMock
        ts = MagicMock()
        result_ts = MagicMock()
        ts.slice.return_value = result_ts

        result = analyzer._filter_to_clean(ts, mask)
        assert result is result_ts
        # Verify slice called with correct dates
        call_args = ts.slice.call_args[0]
        assert call_args[0] == pd.Timestamp("2020-02-10")  # mask.iloc[40]
        assert call_args[1] == pd.Timestamp("2020-03-31")  # mask.iloc[89]

    def test_no_clean_segment_returns_none(self):
        """Should return None when mask is all False."""
        analyzer = MLAnalyzer()
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        mask = pd.Series(False, index=dates)
        result = analyzer._filter_to_clean(None, mask)
        assert result is None


class TestMLAnalyzerIntegration:
    @pytest.mark.slow
    def test_train_transient_model_returns_model(self):
        """Integration test — needs actual Darts + PyTorch."""
        pytest.skip("Integration test — run manually with GPU")
```

Tests the core `_filter_to_clean` logic without GPU. The full training integration test remains as a manual slow test.

- [ ] **Step 2: Implement MLAnalyzer**

```python
# dashboard/utils/pumping_detection/ml_layer.py
"""Layer 2: ML model training on clean periods + full-series prediction.

Trains a transient (disposable) TFT model using the existing ModelFactory,
without MLflow logging or model registry persistence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLAnalyzer:
    """Train a TFT model on clean periods and predict on the full series."""

    def __init__(
        self,
        model_type: str = "TFTModel",
        input_chunk_length: int = 365,
        output_chunk_length: int = 30,
        max_epochs: int = 100,
    ):
        self.model_type = model_type
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.max_epochs = max_epochs

    def train_and_predict(
        self,
        target: Any,  # Darts TimeSeries (full)
        covariates: Any,  # Darts TimeSeries (full)
        clean_mask: pd.Series,
        stop_event: Any = None,
    ) -> dict[str, Any]:
        """Train TFT on clean periods, predict on full series.

        Returns dict with: predictions, ml_residuals, training_metrics, model.
        """
        from darts import TimeSeries
        from darts.metrics import mae, rmse

        from dashboard.utils.model_factory import ModelFactory
        from dashboard.utils.preprocessing import split_data, scale_data

        # Filter to clean periods
        clean_target = self._filter_to_clean(target, clean_mask)

        if clean_target is None or len(clean_target) < self.input_chunk_length + self.output_chunk_length + 100:
            return {"error": "Insufficient clean data for training", "predictions": None}

        clean_covariates = self._filter_to_clean(covariates, clean_mask)

        # Split clean data: 80% train, 20% val
        split_point = int(len(clean_target) * 0.8)
        train_target = clean_target[:split_point]
        val_target = clean_target[split_point:]
        train_cov = clean_covariates[:split_point] if clean_covariates else None
        val_cov = clean_covariates[split_point:] if clean_covariates else None

        # Create model via factory (no MLflow)
        model = ModelFactory.create(
            model_type=self.model_type,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.max_epochs,
        )

        # Train
        model.fit(
            train_target,
            past_covariates=train_cov,
            val_series=val_target,
            val_past_covariates=val_cov,
        )

        # Predict on full series (sliding window)
        predictions = model.historical_forecasts(
            series=target,
            past_covariates=covariates,
            start=self.input_chunk_length,
            forecast_horizon=self.output_chunk_length,
            stride=self.output_chunk_length,
            retrain=False,
            last_points_only=True,
        )

        # Compute residuals
        pred_values = predictions.pd_series()
        actual_values = target.pd_series().loc[pred_values.index]
        ml_residuals = actual_values - pred_values

        # Metrics
        training_metrics = {
            "mae": float(mae(val_target, model.predict(len(val_target), past_covariates=val_cov))),
            "n_clean_train": split_point,
            "n_clean_val": len(clean_target) - split_point,
        }

        return {
            "predictions": predictions,
            "ml_residuals": ml_residuals,
            "training_metrics": training_metrics,
            "model": model,
        }

    def _filter_to_clean(self, ts: Any, mask: pd.Series) -> Any | None:
        """Extract the longest contiguous clean segment from a TimeSeries."""
        from darts import TimeSeries

        # Find longest contiguous True segment in mask
        groups = mask.astype(int).diff().ne(0).cumsum()
        # Count True values per contiguous group
        clean_lengths = mask.groupby(groups).sum()
        clean_lengths = clean_lengths[clean_lengths > 0]

        if clean_lengths.empty:
            return None

        # Get the longest clean segment
        longest = clean_lengths.idxmax()
        segment_mask = (groups == longest) & mask
        start_date = segment_mask.index[segment_mask].min()
        end_date = segment_mask.index[segment_mask].max()

        return ts.slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/utils/pumping_detection/ml_layer.py tests/pumping_detection/test_ml_layer.py
git commit -m "feat(pumping): add MLAnalyzer for transient TFT training on clean periods"
```

---

## Chunk 3: Fusion + Embedding Stub + Pipeline Orchestration

### Task 7: FusionEngine — concordance scoring

**Files:**
- Create: `dashboard/utils/pumping_detection/fusion.py`
- Test: `tests/pumping_detection/test_fusion.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pumping_detection/test_fusion.py
import pandas as pd
import pytest


class TestFusionEngine:
    def test_two_layer_concordance(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine

        months = pd.date_range("2015-01-01", periods=24, freq="MS")
        layer1_flags = pd.Series([False]*6 + [True]*6 + [False]*12, index=months)
        layer2_flags = pd.Series([False]*6 + [True]*4 + [False]*14, index=months)

        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags, "xai": layer2_flags})

        assert "suspect_windows" in result
        assert "global_score" in result
        assert 0 <= result["global_score"] <= 1
        # Months 7-10 should be HIGH (2/2), months 11-12 should be MEDIUM (1/2)
        high_windows = [w for w in result["suspect_windows"] if w["confidence"] == "high"]
        assert len(high_windows) >= 1

    def test_all_clean_returns_zero_score(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine

        months = pd.date_range("2015-01-01", periods=12, freq="MS")
        layer1_flags = pd.Series([False]*12, index=months)
        layer2_flags = pd.Series([False]*12, index=months)

        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags, "xai": layer2_flags})

        assert result["global_score"] == 0.0
        assert len(result["suspect_windows"]) == 0

    def test_adapts_to_single_layer(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine

        months = pd.date_range("2015-01-01", periods=12, freq="MS")
        layer1_flags = pd.Series([False]*3 + [True]*6 + [False]*3, index=months)

        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags})

        assert result["global_score"] > 0
        # With 1 layer, any flagged month should appear
        assert len(result["suspect_windows"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/pumping_detection/test_fusion.py -v
```

- [ ] **Step 3: Implement FusionEngine**

```python
# dashboard/utils/pumping_detection/fusion.py
"""Fusion engine: concordance-based scoring across detection layers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FusionEngine:
    """Combine per-month flags from multiple layers into a suspicion score."""

    def __init__(self, merge_gap_days: int = 30):
        self.merge_gap_days = merge_gap_days

    def fuse(self, layer_flags: dict[str, pd.Series]) -> dict[str, Any]:
        """Fuse per-month boolean flags from available layers.

        Args:
            layer_flags: Dict of {layer_name: pd.Series[bool]} aligned on monthly index.

        Returns:
            Dict with suspect_windows, global_score, per_month_details.
        """
        if not layer_flags:
            return {"suspect_windows": [], "global_score": 0.0, "per_month_details": []}

        n_layers = len(layer_flags)
        # Align all on the same index (union of all months)
        all_months = sorted(set().union(*(s.index for s in layer_flags.values())))
        index = pd.DatetimeIndex(all_months)

        # Count concordance per month
        per_month = []
        for month in index:
            flagged_by = []
            for name, flags in layer_flags.items():
                if month in flags.index and flags.loc[month]:
                    flagged_by.append(name)

            n_flagged = len(flagged_by)
            if n_flagged == n_layers:
                confidence = "high"
            elif n_flagged > 0 and n_flagged >= n_layers / 2:
                confidence = "medium"
            elif n_flagged > 0:
                confidence = "low"
            else:
                confidence = "clean"

            per_month.append({
                "month": str(month.date()),
                "confidence": confidence,
                "flagged_by": flagged_by,
                "concordance": n_flagged / n_layers if n_layers > 0 else 0,
            })

        # Merge adjacent suspect months into windows
        suspect_windows = self._merge_windows(per_month)

        # Global score: mean concordance across all months
        concordances = [m["concordance"] for m in per_month]
        global_score = sum(concordances) / len(concordances) if concordances else 0.0

        return {
            "suspect_windows": suspect_windows,
            "global_score": round(float(global_score), 3),
            "per_month_details": per_month,
        }

    def _merge_windows(self, per_month: list[dict]) -> list[dict]:
        """Merge adjacent suspect months into contiguous windows."""
        windows = []
        current_start = None
        current_months = []

        for entry in per_month:
            if entry["confidence"] != "clean":
                if current_start is None:
                    current_start = entry["month"]
                current_months.append(entry)
            else:
                if current_start is not None:
                    windows.append(self._build_window(current_start, current_months))
                    current_start = None
                    current_months = []

        if current_start is not None:
            windows.append(self._build_window(current_start, current_months))

        return windows

    def _build_window(self, start: str, months: list[dict]) -> dict:
        """Build a suspect window from a list of consecutive suspect months."""
        confidences = [m["confidence"] for m in months]
        # Window confidence = highest confidence in the window
        if "high" in confidences:
            confidence = "high"
        elif "medium" in confidences:
            confidence = "medium"
        else:
            confidence = "low"

        all_layers = set()
        for m in months:
            all_layers.update(m["flagged_by"])

        return {
            "start": start,
            "end": months[-1]["month"],
            "confidence": confidence,
            "duration_months": len(months),
            "layers": sorted(all_layers),
            "max_concordance": max(m["concordance"] for m in months),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pumping_detection/test_fusion.py -v
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pumping_detection/fusion.py tests/pumping_detection/test_fusion.py
git commit -m "feat(pumping): add FusionEngine with concordance scoring and window merging"
```

### Task 8: EmbeddingAnalyzer — stub for Phase 1

**Files:**
- Create: `dashboard/utils/pumping_detection/embedding_layer.py`

- [ ] **Step 1: Implement stub**

```python
# dashboard/utils/pumping_detection/embedding_layer.py
"""Layer 3: Embedding drift analysis (stub — SoftCLT/TS2Vec not yet available).

When the encoder becomes available, this module will:
1. Compute temporal embeddings in sliding windows
2. Track embedding trajectory drift (Mahalanobis distance)
3. Find twin stations via cosine similarity
4. Produce UMAP projections for visualization
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ENCODER_AVAILABLE = False

try:
    # Will be replaced with actual import when SoftCLT/TS2Vec is ready
    # from dashboard.utils.embeddings import SoftCLTEncoder
    pass
except ImportError:
    pass


class EmbeddingAnalyzer:
    """Embedding drift analysis for pumping detection (stub)."""

    def __init__(self, encoder: str = "softclt", window_size: int = 365, n_twins: int = 5):
        self.encoder = encoder
        self.window_size = window_size
        self.n_twins = n_twins

    @property
    def available(self) -> bool:
        return ENCODER_AVAILABLE

    def analyze(self, piezo: Any, **kwargs: Any) -> dict[str, Any]:
        """Run embedding analysis. Returns empty result if encoder not available."""
        if not self.available:
            return {
                "available": False,
                "message": "SoftCLT/TS2Vec encoder not yet available. Layer 3 skipped.",
                "embedding_trajectory": None,
                "drift_scores": None,
                "twin_stations": None,
                "umap_projection": None,
            }

        # TODO: implement when encoder is ready
        raise NotImplementedError("Encoder integration pending")
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pumping_detection/embedding_layer.py
git commit -m "feat(pumping): add EmbeddingAnalyzer stub (Layer 3 optional, encoder pending)"
```

### Task 9: BNPEClient — Hub'Eau Prélèvements API

**Files:**
- Create: `dashboard/utils/pumping_detection/bnpe_client.py`
- Test: `tests/pumping_detection/test_bnpe_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pumping_detection/test_bnpe_client.py
import pytest
from unittest.mock import patch, MagicMock


class TestBNPEClient:
    def test_fetch_nearby_returns_list(self):
        from dashboard.utils.pumping_detection.bnpe_client import BNPEClient

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"code_ouvrage": "OPR001", "nom_ouvrage": "Forage A",
                 "latitude": 48.0, "longitude": 2.0}
            ],
            "count": 1,
        }

        client = BNPEClient()
        with patch("requests.get", return_value=mock_response):
            result = client.fetch_nearby(lat=48.0, lon=2.0, radius_km=5)

        assert result["bnpe_available"] is True
        assert len(result["ouvrages"]) == 1
        assert result["ouvrages"][0]["code_ouvrage"] == "OPR001"

    def test_fetch_nearby_timeout(self):
        from dashboard.utils.pumping_detection.bnpe_client import BNPEClient
        import requests

        client = BNPEClient(timeout=1)
        with patch("requests.get", side_effect=requests.Timeout):
            result = client.fetch_nearby(lat=48.0, lon=2.0, radius_km=5)

        assert result["bnpe_available"] is False
        assert result["ouvrages"] == []
```

- [ ] **Step 2: Implement BNPEClient**

```python
# dashboard/utils/pumping_detection/bnpe_client.py
"""Hub'Eau Prélèvements API client with caching."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

HUBEAU_PRELEVEMENTS_URL = "https://hubeau.eaufrance.fr/api/v1/prelevements"

# Simple in-memory cache with TTL
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 24 * 3600  # 24 hours


class BNPEClient:
    """Fetch nearby BNPE declared pumping facilities."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_nearby(self, lat: float, lon: float, radius_km: float = 5) -> dict[str, Any]:
        """Fetch ouvrages near a location. Returns cached results if available."""
        cache_key = f"{round(lat, 2)}_{round(lon, 2)}_{radius_km}"

        # Check cache
        if cache_key in _cache:
            ts, data = _cache[cache_key]
            if time.time() - ts < _CACHE_TTL:
                return data

        try:
            # Hub'Eau uses bounding box, approximate from radius
            delta = radius_km / 111.0  # ~1 degree ≈ 111 km
            params = {
                "latitude": f"[{lat - delta},{lat + delta}]",
                "longitude": f"[{lon - delta},{lon + delta}]",
                "fields": "code_ouvrage,nom_ouvrage,latitude,longitude,code_commune_insee,nom_commune",
                "format": "json",
                "size": 100,
            }
            resp = requests.get(
                f"{HUBEAU_PRELEVEMENTS_URL}/ouvrages",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            result = {
                "bnpe_available": True,
                "ouvrages": data.get("data", []),
                "count": data.get("count", 0),
            }
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"BNPE API unavailable: {e}")
            result = {"bnpe_available": False, "ouvrages": [], "count": 0}
        except Exception as e:
            logger.error(f"BNPE API error: {e}")
            result = {"bnpe_available": False, "ouvrages": [], "count": 0}

        _cache[cache_key] = (time.time(), result)
        return result
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/pumping_detection/test_bnpe_client.py -v
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/utils/pumping_detection/bnpe_client.py tests/pumping_detection/test_bnpe_client.py
git commit -m "feat(pumping): add BNPEClient for Hub'Eau Prélèvements API with caching"
```

### Task 10: PumpingDetectionPipeline — orchestrator

**Files:**
- Create: `dashboard/utils/pumping_detection/pipeline.py`

- [ ] **Step 1: Implement pipeline orchestrator**

```python
# dashboard/utils/pumping_detection/pipeline.py
"""Pumping detection pipeline orchestrator.

Runs Layer 1 → (Layer 2 + Layer 3 in parallel) → Fusion.
Emits SSE-compatible progress events via a callback.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import pandas as pd

from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer
from dashboard.utils.pumping_detection.changepoint import ChangepointDetector
from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector
from dashboard.utils.pumping_detection.ml_layer import MLAnalyzer
from dashboard.utils.pumping_detection.xai_layer import XAIDriftAnalyzer
from dashboard.utils.pumping_detection.embedding_layer import EmbeddingAnalyzer
from dashboard.utils.pumping_detection.fusion import FusionEngine

logger = logging.getLogger(__name__)


class PumpingDetectionPipeline:
    """Orchestrate the 3-layer pumping detection pipeline."""

    def __init__(self, config: dict[str, Any], emit: Callable | None = None):
        self.config = config
        self.emit = emit or (lambda *a, **kw: None)

    def run(
        self,
        piezo: pd.Series,
        precip: pd.Series,
        etp: pd.Series,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline. Returns complete results dict."""
        results: dict[str, Any] = {}
        stop = stop_event or threading.Event()

        # --- Layer 1: Physics ---
        self.emit("progress", {"stage": "pastas", "pct": 0.10, "message": "Calibrating Pastas..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        pastas_cfg = self.config.get("pastas", {})
        analyzer = PastasAnalyzer(**pastas_cfg)
        try:
            pastas_result = analyzer.analyze(piezo, precip, etp)
            results["pastas"] = pastas_result
            self.emit("metrics", {"stage": "pastas", "partial_result": {
                "evp": pastas_result["pastas_fit_quality"]["evp"],
                "rmse": pastas_result["pastas_fit_quality"]["rmse"],
            }})
        except Exception as e:
            logger.error(f"Layer 1 (Pastas) failed: {e}")
            results["pastas"] = {"error": str(e)}

        # Change points
        self.emit("progress", {"stage": "changepoint", "pct": 0.20, "message": "Detecting change points..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        cp_cfg = self.config.get("changepoint", {})
        try:
            detector = ChangepointDetector(**cp_cfg)
            residuals = results.get("pastas", {}).get("residuals", pd.Series(dtype=float))
            cp_result = detector.detect(residuals)
            results["changepoints"] = cp_result
            self.emit("metrics", {"stage": "changepoint", "partial_result": {
                "n_changepoints": len(cp_result.get("changepoints", [])),
            }})
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            results["changepoints"] = {"error": str(e)}

        # Clean period selection
        self.emit("progress", {"stage": "clean", "pct": 0.30, "message": "Selecting clean periods..."})
        clean_cfg = self.config.get("ml", {})
        selector = CleanPeriodSelector(
            n_sigma=clean_cfg.get("clean_residual_threshold", 2.0)
            if isinstance(clean_cfg.get("clean_residual_threshold"), (int, float))
            else 2.0
        )
        residuals = results.get("pastas", {}).get("residuals", pd.Series(dtype=float))
        clean_result = selector.select(residuals) if len(residuals) > 0 else {"mask": pd.Series(dtype=bool), "n_clean_days": 0, "pct_clean": 0}
        results["clean_periods"] = {k: v for k, v in clean_result.items() if k != "mask"}
        clean_mask = clean_result.get("mask", pd.Series(dtype=bool))
        self.emit("metrics", {"stage": "clean", "partial_result": {
            "n_clean_days": clean_result.get("n_clean_days", 0),
            "pct_clean": clean_result.get("pct_clean", 0),
        }})

        # --- Layer 2 + 3 in parallel ---
        layer2_result: dict[str, Any] = {}
        layer3_result: dict[str, Any] = {}

        def run_layer2():
            nonlocal layer2_result
            if stop.is_set():
                return
            self.emit("progress", {"stage": "ml_train", "pct": 0.45, "message": "Training TFT on clean data..."})
            try:
                ml_cfg = self.config.get("ml", {})
                ml_analyzer = MLAnalyzer(
                    model_type=ml_cfg.get("model_type", "TFTModel"),
                    input_chunk_length=ml_cfg.get("input_chunk_length", 365),
                    output_chunk_length=ml_cfg.get("output_chunk_length", 30),
                    max_epochs=ml_cfg.get("max_epochs", 100),
                )
                # Convert to Darts TimeSeries for ML
                from darts import TimeSeries
                target_ts = TimeSeries.from_series(piezo)
                cov_df = pd.DataFrame({"precip": precip, "temp": etp}, index=piezo.index)
                cov_ts = TimeSeries.from_dataframe(cov_df)

                ml_result = ml_analyzer.train_and_predict(target_ts, cov_ts, clean_mask, stop)
                layer2_result = ml_result

                if "error" not in ml_result:
                    self.emit("progress", {"stage": "xai", "pct": 0.65, "message": "Computing XAI attributions..."})
                    xai_cfg = self.config.get("xai", {})
                    xai_analyzer = XAIDriftAnalyzer(**xai_cfg)
                    xai_result = xai_analyzer.analyze(
                        model=ml_result.get("model"),
                        series=target_ts,
                        covariates=cov_ts,
                        clean_mask=clean_mask,
                        feature_names=list(cov_df.columns),
                    )
                    layer2_result["xai"] = xai_result
            except Exception as e:
                logger.error(f"Layer 2 (ML+XAI) failed: {e}")
                layer2_result = {"error": str(e)}

        def run_layer3():
            nonlocal layer3_result
            if stop.is_set():
                return
            self.emit("progress", {"stage": "embedding", "pct": 0.80, "message": "Analyzing embeddings..."})
            emb_cfg = self.config.get("embeddings", {})
            emb_analyzer = EmbeddingAnalyzer(**emb_cfg)
            layer3_result = emb_analyzer.analyze(piezo)

        t2 = threading.Thread(target=run_layer2)
        t3 = threading.Thread(target=run_layer3)
        t2.start()
        t3.start()
        t2.join()
        t3.join()

        results["ml_xai"] = layer2_result
        results["embeddings"] = layer3_result

        # --- Fusion ---
        self.emit("progress", {"stage": "fusion", "pct": 0.90, "message": "Computing fusion scores..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        layer_flags = self._build_monthly_flags(results, piezo.index)
        fusion_cfg = self.config.get("fusion", {})
        engine = FusionEngine(merge_gap_days=fusion_cfg.get("merge_gap_days", 30))
        fusion_result = engine.fuse(layer_flags)
        results["fusion"] = fusion_result

        return results

    def _build_monthly_flags(self, results: dict, date_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """Convert layer results to per-month boolean flags for fusion."""
        months = pd.date_range(date_index.min(), date_index.max(), freq="MS")
        flags: dict[str, pd.Series] = {}

        # Layer 1: Pastas — flag months with significant ACF
        pastas = results.get("pastas", {})
        if "acf_stats" in pastas:
            lb_pval = pastas["acf_stats"].get("ljung_box_pvalue", 1.0)
            acf_sig = self.config.get("fusion", {}).get("acf_significance", 0.05)
            # If global ACF is significant, use changepoints to locate windows
            if lb_pval < acf_sig:
                pastas_flags = pd.Series(False, index=months)
                for cp in results.get("changepoints", {}).get("changepoints", []):
                    cp_date = pd.Timestamp(cp["date"])
                    # Flag surrounding months
                    for m in months:
                        if abs((m - cp_date).days) < 90:
                            pastas_flags.loc[m] = True
                flags["pastas"] = pastas_flags

        # Layer 2: XAI — flag months with high JS divergence
        xai = results.get("ml_xai", {}).get("xai", {})
        if xai.get("drift_metrics"):
            js_thresh = self.config.get("fusion", {}).get("js_divergence_threshold", 0.3)
            xai_flags = pd.Series(False, index=months)
            for dm in xai["drift_metrics"]:
                if dm.get("js_divergence", 0) > js_thresh:
                    date = pd.Timestamp(dm["window_date"])
                    nearest_month = months[months.get_indexer([date], method="nearest")[0]]
                    xai_flags.loc[nearest_month] = True
            flags["xai"] = xai_flags

        # Layer 3: Embeddings — skip if not available
        emb = results.get("embeddings", {})
        if emb.get("available") and emb.get("drift_scores") is not None:
            emb_thresh = self.config.get("fusion", {}).get("embedding_drift_threshold", 2.0)
            emb_flags = pd.Series(False, index=months)
            # TODO: implement when encoder is ready
            flags["embeddings"] = emb_flags

        return flags
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pumping_detection/pipeline.py
git commit -m "feat(pumping): add PumpingDetectionPipeline orchestrator with SSE events"
```

### Task 11: Update `__init__.py` exports

- [ ] **Step 1: Update exports**

```python
# dashboard/utils/pumping_detection/__init__.py
"""Pumping detection pipeline — unsupervised 3-layer hybrid detection."""

from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer
from dashboard.utils.pumping_detection.changepoint import ChangepointDetector
from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector
from dashboard.utils.pumping_detection.ml_layer import MLAnalyzer
from dashboard.utils.pumping_detection.xai_layer import XAIDriftAnalyzer
from dashboard.utils.pumping_detection.embedding_layer import EmbeddingAnalyzer
from dashboard.utils.pumping_detection.fusion import FusionEngine
from dashboard.utils.pumping_detection.pipeline import PumpingDetectionPipeline
from dashboard.utils.pumping_detection.bnpe_client import BNPEClient

__all__ = [
    "PastasAnalyzer",
    "ChangepointDetector",
    "CleanPeriodSelector",
    "MLAnalyzer",
    "XAIDriftAnalyzer",
    "EmbeddingAnalyzer",
    "FusionEngine",
    "PumpingDetectionPipeline",
    "BNPEClient",
]
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/pumping_detection/__init__.py
git commit -m "feat(pumping): export all pumping detection modules"
```

---

## Chunk 4: API Layer (FastAPI Router + Schemas)

### Task 12: Pydantic schemas

**Files:**
- Create: `api/schemas/pumping_detection.py`

- [ ] **Step 1: Implement schemas**

```python
# api/schemas/pumping_detection.py
"""Pydantic schemas for pumping detection API."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class PastasConfig(BaseModel):
    response_function: str = "Gamma"
    noise_model: bool = True


class ChangepointConfig(BaseModel):
    method: str = "pelt"  # pelt, beast, both
    min_segment_length: int = 90


class MLConfig(BaseModel):
    model_type: str = "TFTModel"
    input_chunk_length: int = 365
    output_chunk_length: int = 30
    max_epochs: int = 100
    clean_residual_threshold: Union[float, Literal["auto"]] = "auto"


class XAIConfig(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["integrated_gradients"])
    window_size: int = 90
    stride: int = 30


class EmbeddingsConfig(BaseModel):
    encoder: str = "softclt"
    window_size: int = 365
    n_twins: int = 5


class FusionConfig(BaseModel):
    js_divergence_threshold: float = 0.3
    spearman_threshold: float = 0.5
    embedding_drift_threshold: float = 2.0
    acf_significance: float = 0.05
    min_layers_for_high: str = "all"
    merge_gap_days: int = 30


class PumpingDetectionConfig(BaseModel):
    pastas: PastasConfig = Field(default_factory=PastasConfig)
    changepoint: ChangepointConfig = Field(default_factory=ChangepointConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    xai: XAIConfig = Field(default_factory=XAIConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class PumpingDetectionRequest(BaseModel):
    dataset_id: str
    config: PumpingDetectionConfig = Field(default_factory=PumpingDetectionConfig)


class SuspectWindow(BaseModel):
    start: str
    end: str
    confidence: str
    duration_months: int
    layers: list[str]
    max_concordance: float


class PumpingDetectionResult(BaseModel):
    global_score: float
    suspect_windows: list[SuspectWindow]
    pastas: dict[str, Any] = Field(default_factory=dict)
    ml_xai: dict[str, Any] = Field(default_factory=dict)
    embeddings: dict[str, Any] = Field(default_factory=dict)
    clean_periods: dict[str, Any] = Field(default_factory=dict)
    changepoints: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 2: Commit**

```bash
git add api/schemas/pumping_detection.py
git commit -m "feat(pumping): add Pydantic schemas for pumping detection API"
```

### Task 13: FastAPI router with SSE

**Files:**
- Create: `api/routers/pumping_detection.py`
- Modify: `api/routers/__init__.py`

**Reference:** Follow the exact pattern from `api/routers/counterfactual.py` — background thread + SSE stream via `task_manager`.

- [ ] **Step 1: Implement router**

```python
# api/routers/pumping_detection.py
"""Pumping detection API router with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.serializers import clean_nans
from api.task_manager import TaskStatus, task_manager
from api.schemas.pumping_detection import PumpingDetectionRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pumping-detection", tags=["pumping-detection"])


def _run_pipeline_thread(task_id: str, req: PumpingDetectionRequest) -> None:
    """Background thread running the pumping detection pipeline."""
    task = task_manager.get(task_id)
    if task is None:
        return

    with task.lock:
        task.status = TaskStatus.RUNNING

    metrics_file = Path(settings.results_dir) / f"pd_metrics_{task_id}.json"
    task.metrics_file = str(metrics_file)

    def emit(event_type: str, data: dict):
        """Write SSE event to metrics file (overwrite with latest state, matching CF pattern)."""
        try:
            with open(metrics_file, "w") as f:
                json.dump({"event": event_type, **clean_nans(data)}, f)
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")

    try:
        import pandas as pd
        from dashboard.utils.dataset_registry import DatasetRegistry
        from dashboard.utils.pumping_detection.pipeline import PumpingDetectionPipeline

        # Load dataset (follows api/routers/datasets.py pattern)
        registry = DatasetRegistry(datasets_dir=Path(settings.data_dir) / "prepared")
        datasets = registry.scan_datasets()
        ds = next((d for d in datasets if d.path.name == req.dataset_id), None)
        if ds is None:
            raise ValueError(f"Dataset not found: {req.dataset_id}")

        df, config = registry.load_dataset(ds)
        target_col = ds.target_column or df.columns[0]
        covariate_cols = ds.covariate_columns or []
        piezo = df[target_col]
        precip = df[covariate_cols[0]] if len(covariate_cols) > 0 else pd.Series(dtype=float)
        etp = df[covariate_cols[1]] if len(covariate_cols) > 1 else pd.Series(dtype=float)

        # Run pipeline
        config = req.config.model_dump()
        pipeline = PumpingDetectionPipeline(config=config, emit=emit)
        result = pipeline.run(piezo, precip, etp, stop_event=task.stop_event)

        with task.lock:
            task.result = clean_nans(result)
            task.status = TaskStatus.COMPLETED

        emit("done", {"status": "completed"})

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        with task.lock:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        emit("error", {"stage": "pipeline", "error_message": str(e), "recoverable": False})


@router.post("/analyze")
def start_analysis(req: PumpingDetectionRequest):
    """Start a pumping detection analysis."""
    task = task_manager.create("pumping_detection", config=req.model_dump())
    thread = threading.Thread(target=_run_pipeline_thread, args=(task.task_id, req), daemon=True)
    task.thread = thread
    thread.start()
    return {"task_id": task.task_id}


@router.get("/{task_id}/stream")
async def stream_progress(task_id: str):
    """SSE stream of analysis progress."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        metrics_file = Path(task.metrics_file) if task.metrics_file else None
        terminal_states = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

        while True:
            if metrics_file and metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    yield {"event": "progress", "data": json.dumps(clean_nans(data))}
                except (json.JSONDecodeError, OSError):
                    pass

            current_status = task.status
            if current_status in terminal_states:
                final = {
                    "status": current_status.value,
                    "error": task.error,
                }
                if task.result:
                    final["result"] = task.result
                yield {"event": "done", "data": json.dumps(clean_nans(final))}
                return

            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.get("/{task_id}/results")
def get_results(task_id: str):
    """Get full results after completion."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(400, f"Task status: {task.status.value}")
    return task.result


@router.get("/{task_id}/layer/{layer_name}")
def get_layer_result(task_id: str, layer_name: str):
    """Get partial result for a specific layer (enables progressive rendering)."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    valid_layers = {"pastas", "changepoints", "clean_periods", "ml_xai", "embeddings", "fusion"}
    if layer_name not in valid_layers:
        raise HTTPException(400, f"Invalid layer: {layer_name}. Valid: {valid_layers}")
    if not task.result or layer_name not in task.result:
        raise HTTPException(404, f"Layer '{layer_name}' not yet available")
    return task.result[layer_name]


@router.post("/{task_id}/cancel")
def cancel_analysis(task_id: str):
    """Cancel a running analysis."""
    if not task_manager.cancel(task_id):
        raise HTTPException(404, "Task not found or already finished")
    task = task_manager.get(task_id)
    return {"status": "cancelled", "partial_results": task.result if task else None}


@router.get("/bnpe-context")
def get_bnpe_context(lat: float, lon: float, radius_km: float = 5):
    """Fetch nearby BNPE declared pumping facilities."""
    from dashboard.utils.pumping_detection.bnpe_client import BNPEClient
    client = BNPEClient()
    return client.fetch_nearby(lat=lat, lon=lon, radius_km=radius_km)
```

- [ ] **Step 2: Register router in `__init__.py`**

Read `api/routers/__init__.py` and add the pumping_detection router to the list.

- [ ] **Step 3: Commit**

```bash
git add api/routers/pumping_detection.py api/schemas/pumping_detection.py api/routers/__init__.py
git commit -m "feat(pumping): add FastAPI router with SSE streaming and BNPE endpoint"
```

---

## Chunk 5: Frontend (React Page + Components)

### Task 14: API client extension

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add pumpingDetection namespace to API client**

Add after the `counterfactual` section in `api.ts`:

```typescript
pumpingDetection: {
  analyze: (body: { dataset_id: string; config?: Record<string, unknown> }) =>
    postJson<{ task_id: string }>('/pumping-detection/analyze', body),
  stream: (taskId: string) =>
    new EventSource(`${API_BASE}/pumping-detection/${taskId}/stream`),
  results: (taskId: string) =>
    fetchJson<PumpingDetectionResult>(`/pumping-detection/${taskId}/results`),
  cancel: (taskId: string) =>
    postJson<{ status: string }>(`/pumping-detection/${taskId}/cancel`, {}),
  bnpeContext: (lat: number, lon: number, radiusKm: number = 5) =>
    fetchJson<BNPEContextResult>(`/pumping-detection/bnpe-context?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`),
},
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(pumping): add pumping detection endpoints to API client"
```

### Task 15: React hook — usePumpingDetection

**Files:**
- Create: `frontend/src/hooks/usePumpingDetection.ts`

- [ ] **Step 1: Implement hook**

```typescript
// frontend/src/hooks/usePumpingDetection.ts
import { useMutation, useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'

interface PumpingStage {
  stage: string
  pct: number
  message: string
}

interface PumpingDetectionState {
  taskId: string | null
  stages: PumpingStage[]
  currentStage: PumpingStage | null
  partialResults: Record<string, unknown>
  status: 'idle' | 'running' | 'done' | 'error' | 'cancelled'
  error: string | null
}

export function usePumpingDetection() {
  const [state, setState] = useState<PumpingDetectionState>({
    taskId: null,
    stages: [],
    currentStage: null,
    partialResults: {},
    status: 'idle',
    error: null,
  })
  const esRef = useRef<EventSource | null>(null)

  const analyzeMutation = useMutation({
    mutationFn: (body: { dataset_id: string; config?: Record<string, unknown> }) =>
      api.pumpingDetection.analyze(body),
    onSuccess: (data) => {
      setState(prev => ({ ...prev, taskId: data.task_id, status: 'running', stages: [] }))
    },
  })

  // SSE connection
  useEffect(() => {
    if (!state.taskId || state.status !== 'running') return

    const es = api.pumpingDetection.stream(state.taskId)
    esRef.current = es

    es.addEventListener('progress', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as PumpingStage
      setState(prev => ({
        ...prev,
        currentStage: data,
        stages: [...prev.stages, data],
      }))
    })

    es.addEventListener('metrics', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as { stage: string; partial_result: unknown }
      setState(prev => ({
        ...prev,
        partialResults: { ...prev.partialResults, [data.stage]: data.partial_result },
      }))
    })

    es.addEventListener('error', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data)
        if (!data.recoverable) {
          setState(prev => ({ ...prev, status: 'error', error: data.error_message }))
        }
      } catch {
        setState(prev => ({ ...prev, status: 'error', error: 'Connexion SSE perdue' }))
      }
    })

    es.addEventListener('done', () => {
      setState(prev => ({ ...prev, status: 'done' }))
      es.close()
    })

    return () => { es.close(); esRef.current = null }
  }, [state.taskId, state.status])

  const cancel = useCallback(() => {
    if (state.taskId) {
      api.pumpingDetection.cancel(state.taskId)
      setState(prev => ({ ...prev, status: 'cancelled' }))
      esRef.current?.close()
    }
  }, [state.taskId])

  const reset = useCallback(() => {
    esRef.current?.close()
    setState({ taskId: null, stages: [], currentStage: null, partialResults: {}, status: 'idle', error: null })
  }, [])

  return {
    analyze: analyzeMutation.mutate,
    cancel,
    reset,
    ...state,
    isAnalyzing: analyzeMutation.isPending || state.status === 'running',
  }
}

export function usePumpingResults(taskId: string | null) {
  return useQuery({
    queryKey: ['pumping-results', taskId],
    queryFn: () => api.pumpingDetection.results(taskId!),
    enabled: !!taskId,
  })
}

export function useBNPEContext(lat: number | null, lon: number | null) {
  return useQuery({
    queryKey: ['bnpe-context', lat, lon],
    queryFn: () => api.pumpingDetection.bnpeContext(lat!, lon!),
    enabled: lat != null && lon != null,
  })
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/usePumpingDetection.ts
git commit -m "feat(pumping): add usePumpingDetection React hook with SSE streaming"
```

### Task 16: React page + components

**Files:**
- Create: `frontend/src/pages/PumpingDetectionPage.tsx`
- Create: `frontend/src/components/pumping/AnnotatedChroniquePlot.tsx`
- Create: `frontend/src/components/pumping/PastasPanel.tsx`
- Create: `frontend/src/components/pumping/XAIDriftPanel.tsx`
- Create: `frontend/src/components/pumping/EmbeddingPanel.tsx`
- Create: `frontend/src/components/pumping/VerdictPanel.tsx`
- Modify: Router config (add `/pumping-detection` route)

**Reference:** Follow the layout/pattern of `ForecastingPage.tsx` and `CounterfactualPage.tsx`. Use @frontend-design:frontend-design skill for implementation. Use @scientific-skills:plotly for chart components.

This task is large — it will be broken into sub-steps by the implementing agent. The key requirement is:
1. Page with dataset selector + config panel + analyze button
2. Main Plotly chart showing piezo series with colored suspect window overlays
3. Three diagnostic panels (Pastas / XAI / Embeddings) as tabs or columns
4. Verdict panel showing fusion score and suspect periods
5. Progressive rendering: show each panel as its layer completes via SSE

- [ ] **Step 1: Create page skeleton with dataset selector and analyze button**
- [ ] **Step 2: Add AnnotatedChroniquePlot component (Plotly)**
- [ ] **Step 3: Add PastasPanel (residuals + ACF + changepoints)**
- [ ] **Step 4: Add XAIDriftPanel (heatmap + divergence curve)**
- [ ] **Step 5: Add EmbeddingPanel (placeholder "not available" state)**
- [ ] **Step 6: Add VerdictPanel (score + suspect periods)**
- [ ] **Step 7: Wire up SSE streaming for progressive rendering**
- [ ] **Step 8: Add route to router config**
- [ ] **Step 9: Commit**

```bash
git add frontend/src/pages/PumpingDetectionPage.tsx frontend/src/components/pumping/
git commit -m "feat(pumping): add PumpingDetectionPage with all visualization components"
```

---

## Chunk 6: Integration Test + Final Wiring

### Task 17: Integration test

**Files:**
- Create: `tests/pumping_detection/test_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
# tests/pumping_detection/test_pipeline.py
"""Integration test for the pumping detection pipeline.

Uses synthetic data with known pumping to validate end-to-end behavior.
"""
import numpy as np
import pandas as pd
import pytest


def _make_full_synthetic_dataset(n_years=5, seed=42):
    """Create a multi-year synthetic dataset with pumping in summers of years 3-4."""
    rng = np.random.default_rng(seed)
    n_days = n_years * 365
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # Natural piezo: seasonal + noise
    piezo = 50.0 + 3.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.3, n_days)

    # Inject pumping: summers of year 3 and 4 (days 730-915, 1095-1280)
    for start in [730 + 150, 1095 + 150]:  # ~June each year
        end = min(start + 120, n_days)  # ~4 months
        piezo[start:end] -= 1.5  # 1.5m drawdown

    piezo = pd.Series(piezo, index=dates, name="piezo")
    precip = pd.Series(
        3.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1, n_days),
        index=dates, name="precip"
    ).clip(lower=0)
    etp = pd.Series(
        2.0 + 1.5 * np.sin(2 * np.pi * (t - 90) / 365.25) + rng.normal(0, 0.3, n_days),
        index=dates, name="etp"
    ).clip(lower=0)

    return piezo, precip, etp


class TestPipelineE2E:
    @pytest.mark.slow
    def test_pipeline_runs_end_to_end(self):
        from dashboard.utils.pumping_detection.pipeline import PumpingDetectionPipeline

        piezo, precip, etp = _make_full_synthetic_dataset()
        events = []

        def capture_emit(event_type, data):
            events.append((event_type, data))

        config = {
            "pastas": {"response_function": "Gamma"},
            "changepoint": {"method": "pelt", "min_segment_length": 60},
            "ml": {"model_type": "TFTModel", "max_epochs": 5, "input_chunk_length": 180, "output_chunk_length": 14},
            "xai": {"methods": ["integrated_gradients"], "window_size": 90, "stride": 30},
            "fusion": {"js_divergence_threshold": 0.3, "merge_gap_days": 30},
        }

        pipeline = PumpingDetectionPipeline(config=config, emit=capture_emit)
        result = pipeline.run(piezo, precip, etp)

        # Basic structure checks
        assert "pastas" in result
        assert "fusion" in result
        assert "global_score" in result["fusion"]
        assert 0 <= result["fusion"]["global_score"] <= 1

        # Should have detected some suspect windows
        suspect = result["fusion"]["suspect_windows"]
        assert len(suspect) >= 1

        # Events should have been emitted
        assert len(events) > 0
        stages = [e[1].get("stage") for e in events if e[0] == "progress"]
        assert "pastas" in stages
```

- [ ] **Step 2: Run integration test (may be slow)**

```bash
pytest tests/pumping_detection/test_pipeline.py -v -m slow --timeout=600
```

- [ ] **Step 3: Commit**

```bash
git add tests/pumping_detection/test_pipeline.py
git commit -m "test(pumping): add end-to-end integration test with synthetic pumping data"
```

### Task 18: Final wiring and smoke test

- [ ] **Step 1: Rebuild Docker**

```bash
docker compose up -d --build
```

- [ ] **Step 2: Verify API endpoint is accessible**

```bash
curl -s http://localhost:49513/api/v1/pumping-detection/bnpe-context?lat=48.0&lon=2.0&radius_km=5 | head -c 200
```

- [ ] **Step 3: Verify frontend builds**

```bash
cd frontend && npm run build
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(pumping): complete Phase 1 pumping detection feature"
```
