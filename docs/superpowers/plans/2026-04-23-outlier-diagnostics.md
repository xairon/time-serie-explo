# Outlier Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a residual outlier investigation module to the Pastas pipeline — backend classifies each outlier by cross-referencing climate/data-quality/neighbors, frontend makes red bars clickable with an expandable diagnostic panel.

**Architecture:** One new backend module (`outlier_diagnostics.py`) computes all outlier diagnostics in batch via a single SQL round-trip per data source. One new API endpoint exposes the results. One new React component (`OutlierDetailPanel`) renders the expandable detail. The existing residuals bar chart in `FitResultsPanel` becomes interactive.

**Tech Stack:** Python (Pastas, pandas, numpy, scipy, SQLAlchemy), FastAPI + Pydantic, React 19 + TanStack Query + Plotly.js, PostgreSQL (BRGM data warehouse)

**Spec:** `docs/superpowers/specs/2026-04-23-outlier-diagnostics-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `dashboard/utils/pastas/outlier_diagnostics.py` | Core computation: detect outliers, gather context, classify, explain |
| Create | `api/schemas/pastas_outlier.py` | Pydantic response models |
| Create | `frontend/src/components/pastas/OutlierDetailPanel.tsx` | Expandable detail panel component |
| Create | `tests/pastas/test_outlier_diagnostics.py` | Backend unit tests |
| Modify | `api/routers/pastas.py` (after line 460) | New endpoint |
| Modify | `frontend/src/lib/api.ts` (line ~327) | API client method |
| Modify | `frontend/src/hooks/usePastas.ts` (after line 68) | React Query hook |
| Modify | `frontend/src/components/pastas/FitResultsPanel.tsx` (lines 257-286) | Clickable bars + panel integration |

---

### Task 1: Pydantic Schemas

**Files:**
- Create: `api/schemas/pastas_outlier.py`

- [ ] **Step 1: Create schema file**

```python
# api/schemas/pastas_outlier.py
"""Pydantic schemas for outlier diagnostics."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ClimateContext(BaseModel):
    precip_mm: Optional[float] = None
    precip_zscore: Optional[float] = None
    temp_c: Optional[float] = None
    temp_zscore: Optional[float] = None
    etp_mm: Optional[float] = None
    etp_zscore: Optional[float] = None
    spli: Optional[float] = None
    spli_class: Optional[str] = None
    spi: Optional[float] = None
    spi_class: Optional[str] = None


class DataQuality(BaseModel):
    gap_days: int
    coverage_pct: float
    nearest_gap_distance_days: Optional[int] = None


class NeighborZscore(BaseModel):
    code_bss: str
    zscore: float


class NeighborContext(BaseModel):
    total: int
    anomalous: int
    neighbor_zscores: list[NeighborZscore]


class OutlierDiagnostic(BaseModel):
    date: str
    residual: float
    residual_zscore: float
    severity: float
    category: str
    category_label: str
    secondary_tags: list[str]
    explanation: str
    climate: ClimateContext
    contributions: dict[str, float]
    observed: float
    simulated: float
    data_quality: DataQuality
    neighbors: NeighborContext


class OutlierSummary(BaseModel):
    by_category: dict[str, int]
    seasonal_pattern: dict[str, int]
    median_severity: float


class OutlierDiagnosticsResponse(BaseModel):
    run_id: str
    code_bss: str
    sigma: float
    threshold: float
    n_residuals: int
    n_outliers: int
    outliers: list[OutlierDiagnostic]
    summary: OutlierSummary
```

- [ ] **Step 2: Commit**

```bash
git add api/schemas/pastas_outlier.py
git commit -m "feat(pastas): add Pydantic schemas for outlier diagnostics"
```

---

### Task 2: Core Backend — Outlier Detection & Context Gathering

**Files:**
- Create: `dashboard/utils/pastas/outlier_diagnostics.py`
- Test: `tests/pastas/test_outlier_diagnostics.py`

- [ ] **Step 1: Write the test file with detection + climate context tests**

```python
# tests/pastas/test_outlier_diagnostics.py
"""Tests for outlier diagnostics module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dashboard.utils.pastas.outlier_diagnostics import (
    _detect_outliers,
    _build_climate_context,
    _build_data_quality,
    _build_neighbor_context,
    _classify_outlier,
    _generate_explanation,
    compute_outlier_diagnostics,
    CATEGORY_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def residuals_with_outliers():
    """Monthly residuals with 2 clear outliers."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    values = rng.normal(0, 0.1, 60)
    # Inject outliers at index 10 (Nov 2018) and 30 (Jul 2020)
    values[10] = 0.5
    values[30] = -0.45
    return pd.Series(values, index=dates)


@pytest.fixture
def climate_df():
    """Historical monthly climate data for a station."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "mois": dates,
        "precipitation_totale": 60 + 20 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 10, 120),
        "temperature_moyenne": 12 + 8 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 1, 120),
        "evaporation_moyenne": 2 + 1.5 * np.sin(2 * np.pi * np.arange(120) / 12) + rng.normal(0, 0.3, 120),
    })


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetectOutliers:
    def test_identifies_outliers_above_threshold(self, residuals_with_outliers):
        outliers, sigma = _detect_outliers(residuals_with_outliers)
        assert len(outliers) == 2
        assert sigma > 0

    def test_returns_dates_and_values(self, residuals_with_outliers):
        outliers, sigma = _detect_outliers(residuals_with_outliers)
        for o in outliers:
            assert "date" in o
            assert "residual" in o
            assert "residual_zscore" in o
            assert abs(o["residual"]) > 2 * sigma

    def test_no_outliers_when_all_normal(self):
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        values = np.full(60, 0.05)
        residuals = pd.Series(values, index=dates)
        outliers, _ = _detect_outliers(residuals)
        assert len(outliers) == 0


# ---------------------------------------------------------------------------
# Climate context
# ---------------------------------------------------------------------------

class TestBuildClimateContext:
    def test_computes_zscore_for_calendar_month(self, climate_df):
        # Pick a March date
        target_date = pd.Timestamp("2019-03-01")
        ctx = _build_climate_context(target_date, climate_df, spli_lookup={}, spi_lookup={})
        assert ctx["precip_mm"] is not None
        assert ctx["precip_zscore"] is not None
        assert ctx["temp_c"] is not None
        assert ctx["temp_zscore"] is not None

    def test_includes_spli_spi_when_available(self, climate_df):
        target_date = pd.Timestamp("2019-03-01")
        spli_lookup = {"2019-03-01": {"spli": 1.5, "classification": "TRES_HAUT"}}
        spi_lookup = {"2019-03-01": {"spi": 2.0, "classification": "TRES_HAUT"}}
        ctx = _build_climate_context(target_date, climate_df, spli_lookup=spli_lookup, spi_lookup=spi_lookup)
        assert ctx["spli"] == 1.5
        assert ctx["spli_class"] == "TRES_HAUT"
        assert ctx["spi"] == 2.0

    def test_handles_missing_month_gracefully(self, climate_df):
        target_date = pd.Timestamp("2030-06-01")
        ctx = _build_climate_context(target_date, climate_df, spli_lookup={}, spi_lookup={})
        assert ctx["precip_mm"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pastas/test_outlier_diagnostics.py -v --no-header -x 2>&1 | head -20`
Expected: FAIL — module not found

- [ ] **Step 3: Implement outlier detection and context gathering**

```python
# dashboard/utils/pastas/outlier_diagnostics.py
"""Compute outlier diagnostics for Pastas model residuals."""
from __future__ import annotations

import logging
from statistics import median
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "DATA_GAP": "Data gap",
    "CLIMATE_EXTREME": "Extreme climate event",
    "REGIONAL_SIGNAL": "Regional signal",
    "DOMINANT_CONTRIBUTION": "Dominant contribution",
    "SEASONAL_BIAS": "Seasonal bias",
    "UNKNOWN": "Undetermined",
}


def _detect_outliers(residuals: pd.Series) -> tuple[list[dict], float]:
    """Find residuals exceeding 2σ. Returns (outlier_dicts, sigma)."""
    clean = residuals.dropna()
    if len(clean) < 10:
        return [], 0.0
    sigma = float(clean.std())
    if sigma == 0:
        return [], 0.0
    threshold = 2 * sigma
    outliers = []
    for date, value in clean.items():
        if abs(value) > threshold:
            outliers.append({
                "date": pd.Timestamp(date),
                "residual": float(value),
                "residual_zscore": float(abs(value) / sigma),
            })
    return outliers, sigma


def _build_climate_context(
    target_date: pd.Timestamp,
    climate_df: pd.DataFrame,
    spli_lookup: dict[str, dict],
    spi_lookup: dict[str, dict],
) -> dict[str, Any]:
    """Build climate context for one outlier month."""
    result: dict[str, Any] = {
        "precip_mm": None, "precip_zscore": None,
        "temp_c": None, "temp_zscore": None,
        "etp_mm": None, "etp_zscore": None,
        "spli": None, "spli_class": None,
        "spi": None, "spi_class": None,
    }
    if climate_df.empty:
        return result

    month_col = pd.to_datetime(climate_df["mois"])
    cal_month = target_date.month
    same_month_mask = month_col.dt.month == cal_month
    same_month = climate_df.loc[same_month_mask]

    target_row = climate_df.loc[month_col == target_date]
    if target_row.empty or same_month.empty:
        # Try lookup by SPLI/SPI even without climate row
        date_key = target_date.strftime("%Y-%m-%d")
        spli_entry = spli_lookup.get(date_key, {})
        spi_entry = spi_lookup.get(date_key, {})
        result["spli"] = spli_entry.get("spli")
        result["spli_class"] = spli_entry.get("classification")
        result["spi"] = spi_entry.get("spi")
        result["spi_class"] = spi_entry.get("classification")
        return result

    row = target_row.iloc[0]

    for col, key_val, key_z in [
        ("precipitation_totale", "precip_mm", "precip_zscore"),
        ("temperature_moyenne", "temp_c", "temp_zscore"),
        ("evaporation_moyenne", "etp_mm", "etp_zscore"),
    ]:
        if col in same_month.columns:
            vals = same_month[col].dropna()
            val = row.get(col)
            if val is not None and not pd.isna(val) and len(vals) >= 3:
                mean = float(vals.mean())
                std = float(vals.std())
                result[key_val] = float(val)
                result[key_z] = float((val - mean) / std) if std > 0 else 0.0

    date_key = target_date.strftime("%Y-%m-%d")
    spli_entry = spli_lookup.get(date_key, {})
    spi_entry = spi_lookup.get(date_key, {})
    result["spli"] = spli_entry.get("spli")
    result["spli_class"] = spli_entry.get("classification")
    result["spi"] = spi_entry.get("spi")
    result["spi_class"] = spi_entry.get("classification")

    return result


def _build_data_quality(
    target_date: pd.Timestamp,
    daily_df: pd.DataFrame,
) -> dict[str, Any]:
    """Check data quality around an outlier (±30 days)."""
    result = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
    if daily_df.empty:
        return result

    date_col = pd.to_datetime(daily_df["date"])
    window_start = target_date - pd.Timedelta(days=30)
    window_end = target_date + pd.Timedelta(days=30)
    mask = (date_col >= window_start) & (date_col <= window_end)
    window = daily_df.loc[mask]

    total_days = 61
    if window.empty:
        result["gap_days"] = total_days
        result["coverage_pct"] = 0.0
        return result

    if "niveau_nappe_eau" in window.columns:
        non_null = window["niveau_nappe_eau"].notna().sum()
    else:
        non_null = len(window)

    result["gap_days"] = total_days - int(non_null)
    result["coverage_pct"] = round(float(non_null) / total_days * 100, 1)

    # Find nearest gap
    all_dates = date_col.sort_values()
    gaps = all_dates.diff().dt.days
    gap_mask = gaps > 1
    if gap_mask.any():
        gap_dates = all_dates[gap_mask]
        distances = (gap_dates - target_date).abs().dt.days
        result["nearest_gap_distance_days"] = int(distances.min())

    return result


def _build_neighbor_context(
    target_date: pd.Timestamp,
    sibling_codes: list[str],
    monthly_neighbors_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute z-scores for BDLISA siblings at the target month."""
    result: dict[str, Any] = {"total": len(sibling_codes), "anomalous": 0, "neighbor_zscores": []}
    if not sibling_codes or monthly_neighbors_df.empty:
        return result

    cal_month = target_date.month
    month_col = pd.to_datetime(monthly_neighbors_df["mois"])

    for code in sibling_codes:
        sib_mask = monthly_neighbors_df["code_bss"] == code
        sib_data = monthly_neighbors_df.loc[sib_mask]
        sib_months = pd.to_datetime(sib_data["mois"])

        same_cal = sib_data.loc[sib_months.dt.month == cal_month]
        target_row = sib_data.loc[sib_months == target_date]

        if same_cal.empty or target_row.empty or "niveau_moyen" not in same_cal.columns:
            continue

        vals = same_cal["niveau_moyen"].dropna()
        target_val = target_row.iloc[0].get("niveau_moyen")
        if target_val is None or pd.isna(target_val) or len(vals) < 3:
            continue

        mean = float(vals.mean())
        std = float(vals.std())
        if std == 0:
            continue
        zscore = float((target_val - mean) / std)
        result["neighbor_zscores"].append({"code_bss": code, "zscore": round(zscore, 2)})
        if abs(zscore) > 1.5:
            result["anomalous"] += 1

    return result
```

- [ ] **Step 4: Run tests to verify detection & context pass**

Run: `pytest tests/pastas/test_outlier_diagnostics.py -v --no-header -x 2>&1 | tail -15`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/outlier_diagnostics.py tests/pastas/test_outlier_diagnostics.py
git commit -m "feat(pastas): add outlier detection and context gathering"
```

---

### Task 3: Classification & Explanation Engine

**Files:**
- Modify: `dashboard/utils/pastas/outlier_diagnostics.py`
- Modify: `tests/pastas/test_outlier_diagnostics.py`

- [ ] **Step 1: Add classification tests**

Append to `tests/pastas/test_outlier_diagnostics.py`:

```python
# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyOutlier:
    def test_data_gap_wins_over_climate(self):
        outlier = {"residual": 0.5, "residual_zscore": 3.0}
        data_quality = {"gap_days": 5, "coverage_pct": 80.0, "nearest_gap_distance_days": 3}
        climate = {"precip_zscore": 2.5, "temp_zscore": 0.1, "etp_zscore": -0.5,
                   "precip_mm": 140, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 5, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.3, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "DATA_GAP"
        assert "CLIMATE_EXTREME" in tags

    def test_climate_extreme_detected(self):
        outlier = {"residual": 0.4, "residual_zscore": 2.5}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 2.8, "temp_zscore": 0.5, "etp_zscore": -0.3,
                   "precip_mm": 150, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 3, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.2, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "CLIMATE_EXTREME"

    def test_regional_signal_detected(self):
        outlier = {"residual": 0.3, "residual_zscore": 2.1}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 1.0, "temp_zscore": 0.5, "etp_zscore": -0.3,
                   "precip_mm": 80, "temp_c": 10, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 4, "anomalous": 3, "neighbor_zscores": [
            {"code_bss": "A", "zscore": 2.0}, {"code_bss": "B", "zscore": 1.8},
            {"code_bss": "C", "zscore": 1.6}, {"code_bss": "D", "zscore": 0.3},
        ]}
        contributions = {"recharge": 0.15, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "REGIONAL_SIGNAL"

    def test_dominant_contribution_detected(self):
        outlier = {"residual": 0.35, "residual_zscore": 2.3}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 0.5, "temp_zscore": 0.2, "etp_zscore": -0.1,
                   "precip_mm": 65, "temp_c": 12, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 2, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.95, "evap": 0.01, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "DOMINANT_CONTRIBUTION"

    def test_unknown_when_no_rule_matches(self):
        outlier = {"residual": 0.25, "residual_zscore": 2.05}
        data_quality = {"gap_days": 0, "coverage_pct": 100.0, "nearest_gap_distance_days": None}
        climate = {"precip_zscore": 0.3, "temp_zscore": 0.1, "etp_zscore": 0.0,
                   "precip_mm": 60, "temp_c": 12, "etp_mm": 2,
                   "spli": None, "spli_class": None, "spi": None, "spi_class": None}
        neighbors = {"total": 2, "anomalous": 0, "neighbor_zscores": []}
        contributions = {"recharge": 0.1, "evap": 0.08, "constant_d": 12.0}

        cat, tags = _classify_outlier(outlier, climate, data_quality, neighbors, contributions)
        assert cat == "UNKNOWN"


class TestGenerateExplanation:
    def test_data_gap_explanation(self):
        explanation = _generate_explanation(
            "DATA_GAP", [],
            climate={"precip_mm": 60, "precip_zscore": 0.3},
            data_quality={"gap_days": 12, "coverage_pct": 80},
            neighbors={"total": 3, "anomalous": 1},
            contributions={"recharge": 0.2},
            residual_zscore=2.5,
        )
        assert "12" in explanation
        assert "gap" in explanation.lower()

    def test_multiple_tags_concatenate(self):
        explanation = _generate_explanation(
            "CLIMATE_EXTREME", ["REGIONAL_SIGNAL"],
            climate={"precip_mm": 140, "precip_zscore": 2.5, "temp_c": 10, "temp_zscore": 0.3,
                     "etp_mm": 2, "etp_zscore": -0.5},
            data_quality={"gap_days": 0, "coverage_pct": 100},
            neighbors={"total": 4, "anomalous": 3},
            contributions={"recharge": 0.3},
            residual_zscore=2.8,
        )
        assert "precipitation" in explanation.lower() or "precip" in explanation.lower()
        assert "neighbor" in explanation.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/pastas/test_outlier_diagnostics.py::TestClassifyOutlier -v --no-header -x 2>&1 | head -10`
Expected: FAIL — `_classify_outlier` not importable

- [ ] **Step 3: Implement classification and explanation**

Append to `dashboard/utils/pastas/outlier_diagnostics.py`:

```python
def _classify_outlier(
    outlier: dict,
    climate: dict,
    data_quality: dict,
    neighbors: dict,
    contributions: dict,
) -> tuple[str, list[str]]:
    """Classify an outlier. Returns (primary_category, secondary_tags)."""
    matched: list[str] = []

    # Rule 1: DATA_GAP
    if data_quality.get("gap_days", 0) >= 1:
        matched.append("DATA_GAP")

    # Rule 2: CLIMATE_EXTREME
    for key in ("precip_zscore", "temp_zscore", "etp_zscore"):
        z = climate.get(key)
        if z is not None and abs(z) > 2.0:
            if "CLIMATE_EXTREME" not in matched:
                matched.append("CLIMATE_EXTREME")
            break

    # Rule 3: REGIONAL_SIGNAL
    total = neighbors.get("total", 0)
    anomalous = neighbors.get("anomalous", 0)
    if total > 0 and anomalous / total >= 0.5:
        matched.append("REGIONAL_SIGNAL")

    # Rule 4: DOMINANT_CONTRIBUTION
    # Exclude constant/baseline terms
    stress_contribs = {k: abs(v) for k, v in contributions.items()
                       if not k.startswith("constant") and not k.startswith("Constant")}
    total_contrib = sum(stress_contribs.values())
    if total_contrib > 0:
        max_contrib = max(stress_contribs.values())
        if max_contrib / total_contrib > 0.8:
            matched.append("DOMINANT_CONTRIBUTION")

    if not matched:
        return "UNKNOWN", []

    primary = matched[0]
    secondary = matched[1:]
    return primary, secondary


def _generate_explanation(
    category: str,
    secondary_tags: list[str],
    climate: dict,
    data_quality: dict,
    neighbors: dict,
    contributions: dict,
    residual_zscore: float,
    seasonal_info: Optional[dict] = None,
) -> str:
    """Generate a natural-language explanation for the outlier."""
    parts: list[str] = []

    all_cats = [category] + secondary_tags

    for cat in all_cats:
        if cat == "DATA_GAP":
            gap = data_quality.get("gap_days", 0)
            parts.append(f"Data gap of {gap} days detected within ±30 days. Model interpolation may be unreliable.")

        elif cat == "CLIMATE_EXTREME":
            extremes = []
            for label, key in [("precipitation", "precip_zscore"), ("temperature", "temp_zscore"), ("evapotranspiration", "etp_zscore")]:
                z = climate.get(key)
                if z is not None and abs(z) > 2.0:
                    direction = "above" if z > 0 else "below"
                    extremes.append(f"{label} was {abs(z):.1f}σ {direction} normal")
            if extremes:
                parts.append(f"Monthly {', '.join(extremes)}.")

        elif cat == "REGIONAL_SIGNAL":
            n = neighbors.get("anomalous", 0)
            total = neighbors.get("total", 0)
            parts.append(f"{n}/{total} neighboring stations also show anomalous levels this month.")

        elif cat == "DOMINANT_CONTRIBUTION":
            stress_contribs = {k: abs(v) for k, v in contributions.items()
                               if not k.startswith("constant") and not k.startswith("Constant")}
            if stress_contribs:
                top = max(stress_contribs, key=stress_contribs.get)
                val = contributions[top]
                parts.append(f"The {top} contribution ({val:+.3f}m) dominates model response this month.")

        elif cat == "SEASONAL_BIAS":
            if seasonal_info:
                count = seasonal_info.get("count", 0)
                quarter = seasonal_info.get("quarter", "?")
                sign = "positive" if seasonal_info.get("sign", 1) > 0 else "negative"
                parts.append(f"{count} outliers with {sign} residuals cluster in Q{quarter}, suggesting systematic seasonal model error.")

        elif cat == "UNKNOWN":
            parts.append(f"No clear cause identified. Residual is {residual_zscore:.1f}σ from model expectation.")

    return " ".join(parts) if parts else f"Residual is {residual_zscore:.1f}σ from model expectation."
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pastas/test_outlier_diagnostics.py -v --no-header 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/outlier_diagnostics.py tests/pastas/test_outlier_diagnostics.py
git commit -m "feat(pastas): add outlier classification and explanation engine"
```

---

### Task 4: Main `compute_outlier_diagnostics` Function

**Files:**
- Modify: `dashboard/utils/pastas/outlier_diagnostics.py`
- Modify: `tests/pastas/test_outlier_diagnostics.py`

- [ ] **Step 1: Add integration test**

Append to `tests/pastas/test_outlier_diagnostics.py`:

```python
# ---------------------------------------------------------------------------
# Seasonal bias (pass 2)
# ---------------------------------------------------------------------------

class TestSeasonalBiasPass2:
    def test_seasonal_bias_applied_to_unclassified(self):
        """3+ outliers of the same sign in one quarter → SEASONAL_BIAS on UNKNOWN ones."""
        from dashboard.utils.pastas.outlier_diagnostics import _apply_seasonal_bias

        outliers = [
            {"date": pd.Timestamp("2019-01-01"), "residual": 0.3, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-02-01"), "residual": 0.25, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-03-01"), "residual": 0.28, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2020-07-01"), "residual": -0.4, "category": "CLIMATE_EXTREME", "secondary_tags": []},
        ]
        _apply_seasonal_bias(outliers)

        # First three should become SEASONAL_BIAS (they were UNKNOWN)
        assert outliers[0]["category"] == "SEASONAL_BIAS"
        assert outliers[1]["category"] == "SEASONAL_BIAS"
        assert outliers[2]["category"] == "SEASONAL_BIAS"
        # Fourth keeps its category but gets a tag if applicable
        assert outliers[3]["category"] == "CLIMATE_EXTREME"

    def test_no_seasonal_bias_with_fewer_than_3(self):
        from dashboard.utils.pastas.outlier_diagnostics import _apply_seasonal_bias

        outliers = [
            {"date": pd.Timestamp("2019-01-01"), "residual": 0.3, "category": "UNKNOWN", "secondary_tags": []},
            {"date": pd.Timestamp("2019-02-01"), "residual": 0.25, "category": "UNKNOWN", "secondary_tags": []},
        ]
        _apply_seasonal_bias(outliers)
        assert outliers[0]["category"] == "UNKNOWN"
        assert outliers[1]["category"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Full pipeline (mocked DB)
# ---------------------------------------------------------------------------

class TestComputeOutlierDiagnosticsMocked:
    def test_returns_correct_structure(self):
        """Test with mocked model and DB engine."""
        import pastas as ps

        # Create a minimal mock model
        model = MagicMock(spec=ps.Model)
        dates = pd.date_range("2018-01-01", periods=60, freq="MS")
        rng = np.random.default_rng(42)
        residual_values = rng.normal(0, 0.1, 60)
        residual_values[10] = 0.5  # outlier
        model.residuals.return_value = pd.Series(residual_values, index=dates)
        model.simulate.return_value = pd.Series(10 + residual_values * 0.1, index=dates)
        model.observations.return_value = pd.Series(10 + residual_values * 0.1 + residual_values, index=dates)
        model.stressmodels = {"recharge": MagicMock()}

        contrib_series = pd.Series(np.full(60, 0.5), index=dates)
        model.get_contribution.return_value = contrib_series

        # Mock engine
        engine = MagicMock(spec=Engine)

        with patch("dashboard.utils.pastas.outlier_diagnostics._fetch_climate_data") as mock_climate, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_daily_data") as mock_daily, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_sibling_codes") as mock_siblings, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_neighbor_monthly") as mock_neighbor, \
             patch("dashboard.utils.pastas.outlier_diagnostics._fetch_drought_indices") as mock_drought:

            mock_climate.return_value = pd.DataFrame({
                "mois": dates,
                "precipitation_totale": rng.normal(60, 10, 60),
                "temperature_moyenne": rng.normal(12, 2, 60),
                "evaporation_moyenne": rng.normal(2, 0.5, 60),
            })
            mock_daily.return_value = pd.DataFrame({
                "date": pd.date_range("2017-12-01", periods=365 * 6, freq="D"),
                "niveau_nappe_eau": rng.normal(10, 0.5, 365 * 6),
            })
            mock_siblings.return_value = []
            mock_neighbor.return_value = pd.DataFrame()
            mock_drought.return_value = ({}, {})

            result = compute_outlier_diagnostics(
                model=model,
                code_bss="TEST/001",
                cal_tmin="2018-01-01",
                cal_tmax="2022-12-01",
                engine=engine,
            )

        assert "sigma" in result
        assert "threshold" in result
        assert "n_outliers" in result
        assert "outliers" in result
        assert "summary" in result
        assert isinstance(result["outliers"], list)
        if result["n_outliers"] > 0:
            o = result["outliers"][0]
            assert "category" in o
            assert "climate" in o
            assert "data_quality" in o
            assert "neighbors" in o
            assert "contributions" in o
            assert "explanation" in o
            assert "severity" in o
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/pastas/test_outlier_diagnostics.py::TestComputeOutlierDiagnosticsMocked -v --no-header -x 2>&1 | head -10`
Expected: FAIL — `_apply_seasonal_bias`, `compute_outlier_diagnostics` not found

- [ ] **Step 3: Implement seasonal bias pass and main function**

Append to `dashboard/utils/pastas/outlier_diagnostics.py`:

```python
def _apply_seasonal_bias(outliers: list[dict]) -> None:
    """Pass 2: detect seasonal clustering and tag affected outliers in-place."""
    if len(outliers) < 3:
        return

    # Group by (quarter, sign)
    from collections import defaultdict
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, o in enumerate(outliers):
        q = (o["date"].month - 1) // 3 + 1
        sign = 1 if o["residual"] > 0 else -1
        groups[(q, sign)].append(i)

    for (quarter, sign), indices in groups.items():
        if len(indices) < 3:
            continue
        seasonal_info = {"quarter": quarter, "sign": sign, "count": len(indices)}
        for idx in indices:
            o = outliers[idx]
            if o["category"] == "UNKNOWN":
                o["category"] = "SEASONAL_BIAS"
                o["_seasonal_info"] = seasonal_info
            elif "SEASONAL_BIAS" not in o["secondary_tags"]:
                o["secondary_tags"].append("SEASONAL_BIAS")


# ---------------------------------------------------------------------------
# Data fetching helpers (thin wrappers for testability)
# ---------------------------------------------------------------------------

def _fetch_climate_data(code_bss: str, engine: Engine) -> pd.DataFrame:
    query = sql_text("""
        SELECT mois, precipitation_totale, temperature_moyenne, evaporation_moyenne
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code AND niveau_moyen IS NOT NULL
        ORDER BY mois
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"code": code_bss})


def _fetch_daily_data(code_bss: str, engine: Engine) -> pd.DataFrame:
    query = sql_text("""
        SELECT date, niveau_nappe_eau
        FROM gold.hubeau_daily_chroniques
        WHERE code_bss = :code
        ORDER BY date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"code": code_bss})


def _fetch_sibling_codes(code_bss: str, engine: Engine, limit: int = 10) -> list[str]:
    query = sql_text("""
        SELECT codes_bdlisa FROM gold.dim_piezo_stations WHERE code_bss = :code
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"code": code_bss}).mappings().first()
    if not row or not row["codes_bdlisa"]:
        return []
    bdlisa = str(row["codes_bdlisa"]).split(",")[0].strip()

    query2 = sql_text("""
        SELECT code_bss FROM gold.dim_piezo_stations
        WHERE codes_bdlisa LIKE :pattern AND code_bss != :code
        LIMIT :lim
    """)
    with engine.connect() as conn:
        result = conn.execute(query2, {"pattern": f"{bdlisa}%", "code": code_bss, "lim": limit})
        return [r["code_bss"] for r in result.mappings()]


def _fetch_neighbor_monthly(sibling_codes: list[str], engine: Engine) -> pd.DataFrame:
    if not sibling_codes:
        return pd.DataFrame()
    placeholders = ", ".join(f":c{i}" for i in range(len(sibling_codes)))
    params = {f"c{i}": c for i, c in enumerate(sibling_codes)}
    query = sql_text(f"""
        SELECT code_bss, mois, niveau_moyen
        FROM gold.fct_monthly_chroniques
        WHERE code_bss IN ({placeholders})
        ORDER BY code_bss, mois
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def _fetch_drought_indices(
    code_bss: str, engine: Engine,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Fetch SPLI and SPI, return as date-keyed lookups."""
    from dashboard.utils.drought import compute_spli, compute_spi

    query = sql_text("""
        SELECT mois, niveau_moyen, precipitation_totale
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code
        ORDER BY mois
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"code": code_bss})

    spli_lookup: dict[str, dict] = {}
    spi_lookup: dict[str, dict] = {}

    if not df.empty:
        months = [str(m) for m in df["mois"]]

        niveau_vals = [float(v) if pd.notna(v) else None for v in df["niveau_moyen"]]
        try:
            for entry in compute_spli(months, niveau_vals):
                spli_lookup[entry["mois"]] = entry
        except Exception:
            logger.debug("SPLI computation failed for %s", code_bss)

        precip_vals = [float(v) if pd.notna(v) else None for v in df["precipitation_totale"]]
        try:
            for entry in compute_spi(months, precip_vals):
                spi_lookup[entry["mois"]] = entry
        except Exception:
            logger.debug("SPI computation failed for %s", code_bss)

    return spli_lookup, spi_lookup


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_outlier_diagnostics(
    model,
    code_bss: str,
    cal_tmin: str,
    cal_tmax: str,
    engine: Engine,
) -> dict:
    """Compute outlier diagnostics for all residuals exceeding 2σ."""
    residuals = model.residuals(tmin=cal_tmin, tmax=cal_tmax)
    residuals_monthly = residuals.resample("MS").mean().dropna()

    outlier_list, sigma = _detect_outliers(residuals_monthly)

    if not outlier_list:
        return {
            "run_id": "",
            "code_bss": code_bss,
            "sigma": sigma,
            "threshold": 2 * sigma,
            "n_residuals": len(residuals_monthly),
            "n_outliers": 0,
            "outliers": [],
            "summary": {"by_category": {}, "seasonal_pattern": {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}, "median_severity": 0.0},
        }

    # Fetch all context data in bulk
    climate_df = _fetch_climate_data(code_bss, engine)
    daily_df = _fetch_daily_data(code_bss, engine)
    sibling_codes = _fetch_sibling_codes(code_bss, engine)
    neighbor_df = _fetch_neighbor_monthly(sibling_codes, engine)
    spli_lookup, spi_lookup = _fetch_drought_indices(code_bss, engine)

    # Get observed and simulated values
    try:
        sim = model.simulate(tmin=cal_tmin, tmax=cal_tmax)
        obs = model.observations(tmin=cal_tmin, tmax=cal_tmax)
        sim_monthly = sim.resample("MS").mean()
        obs_monthly = obs.resample("MS").mean()
    except Exception:
        sim_monthly = pd.Series(dtype=float)
        obs_monthly = pd.Series(dtype=float)

    # Get contributions
    contrib_monthly: dict[str, pd.Series] = {}
    for sm_name in model.stressmodels:
        try:
            c = model.get_contribution(sm_name, tmin=cal_tmin, tmax=cal_tmax)
            contrib_monthly[sm_name] = c.resample("MS").mean()
        except Exception:
            pass

    # Build diagnostics for each outlier
    enriched: list[dict] = []
    for o in outlier_list:
        target = o["date"]

        climate = _build_climate_context(target, climate_df, spli_lookup, spi_lookup)
        dq = _build_data_quality(target, daily_df)
        neighbors = _build_neighbor_context(target, sibling_codes, neighbor_df)

        contribs = {}
        for name, series in contrib_monthly.items():
            if target in series.index:
                contribs[name] = round(float(series.loc[target]), 4)

        observed_val = float(obs_monthly.loc[target]) if target in obs_monthly.index else 0.0
        simulated_val = float(sim_monthly.loc[target]) if target in sim_monthly.index else 0.0

        category, secondary_tags = _classify_outlier(o, climate, dq, neighbors, contribs)
        severity = min(1.0, abs(o["residual"]) / (3 * sigma))

        enriched.append({
            "date": target,
            "residual": o["residual"],
            "residual_zscore": o["residual_zscore"],
            "severity": round(severity, 2),
            "category": category,
            "secondary_tags": secondary_tags,
            "climate": climate,
            "contributions": contribs,
            "observed": round(observed_val, 4),
            "simulated": round(simulated_val, 4),
            "data_quality": dq,
            "neighbors": neighbors,
        })

    # Pass 2: seasonal bias
    _apply_seasonal_bias(enriched)

    # Generate explanations (after pass 2, so seasonal_info is available)
    for o in enriched:
        o["category_label"] = CATEGORY_LABELS.get(o["category"], o["category"])
        o["explanation"] = _generate_explanation(
            o["category"], o["secondary_tags"],
            climate=o["climate"], data_quality=o["data_quality"],
            neighbors=o["neighbors"], contributions=o["contributions"],
            residual_zscore=o["residual_zscore"],
            seasonal_info=o.pop("_seasonal_info", None),
        )
        # Serialize date
        o["date"] = o["date"].strftime("%Y-%m-%d")

    # Sort by severity descending
    enriched.sort(key=lambda x: x["severity"], reverse=True)

    # Summary
    from collections import Counter
    cat_counts = Counter(o["category"] for o in enriched)
    q_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for o in enriched:
        month = pd.Timestamp(o["date"]).month
        q = (month - 1) // 3 + 1
        q_counts[f"Q{q}"] += 1

    severities = [o["severity"] for o in enriched]

    return {
        "run_id": "",
        "code_bss": code_bss,
        "sigma": round(sigma, 4),
        "threshold": round(2 * sigma, 4),
        "n_residuals": len(residuals_monthly),
        "n_outliers": len(enriched),
        "outliers": enriched,
        "summary": {
            "by_category": dict(cat_counts),
            "seasonal_pattern": q_counts,
            "median_severity": round(median(severities), 2) if severities else 0.0,
        },
    }
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/pastas/test_outlier_diagnostics.py -v --no-header 2>&1 | tail -25`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/outlier_diagnostics.py tests/pastas/test_outlier_diagnostics.py
git commit -m "feat(pastas): add compute_outlier_diagnostics with seasonal bias pass"
```

---

### Task 5: API Endpoint

**Files:**
- Modify: `api/routers/pastas.py` (after line 460)
- Modify: `api/schemas/pastas.py` (import)

- [ ] **Step 1: Add the endpoint**

In `api/routers/pastas.py`, add after line 460 (after `return compute_diagnostics(residuals)`), before the export section:

```python
# ---------------------------------------------------------------------------
# GET /models/{run_id}/outlier-diagnostics
# ---------------------------------------------------------------------------

@router.get("/models/{run_id}/outlier-diagnostics")
def get_outlier_diagnostics(run_id: str):
    """Compute outlier diagnostics for a stored Pastas model."""
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.outlier_diagnostics import compute_outlier_diagnostics
    from sqlalchemy import create_engine

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin, tmax = _clean_tmin_tmax(run.data.params)
    code_bss = run.data.params.get("dataset_id", run.data.tags.get("station_id", ""))

    engine = create_engine(_brgm_url())
    try:
        result = compute_outlier_diagnostics(
            model=model,
            code_bss=code_bss,
            cal_tmin=tmin,
            cal_tmax=tmax,
            engine=engine,
        )
    finally:
        engine.dispose()

    result["run_id"] = run_id
    return result
```

- [ ] **Step 2: Commit**

```bash
git add api/routers/pastas.py
git commit -m "feat(pastas): add /outlier-diagnostics API endpoint"
```

---

### Task 6: Frontend API Client & Hook

**Files:**
- Modify: `frontend/src/lib/api.ts` (line ~327, after `diagnostics:`)
- Modify: `frontend/src/hooks/usePastas.ts` (after line 68)

- [ ] **Step 1: Add API client method**

In `frontend/src/lib/api.ts`, inside the `pastas: {` object, after the `diagnostics:` line (line 326), add:

```typescript
    outlierDiagnostics: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/outlier-diagnostics`),
```

- [ ] **Step 2: Add React Query hook**

In `frontend/src/hooks/usePastas.ts`, after `usePastasDiagnostics` (after line 68), add:

```typescript
export function usePastasOutlierDiagnostics(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'outlier-diagnostics', runId],
    queryFn: () => api.pastas.outlierDiagnostics(runId!),
    enabled: !!runId,
    staleTime: 60 * 60 * 1000,
  })
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/hooks/usePastas.ts
git commit -m "feat(pastas): add outlier diagnostics API client and hook"
```

---

### Task 7: OutlierDetailPanel Component

**Files:**
- Create: `frontend/src/components/pastas/OutlierDetailPanel.tsx`

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/pastas/OutlierDetailPanel.tsx
import { X } from 'lucide-react'

interface OutlierDiagnostic {
  date: string
  residual: number
  residual_zscore: number
  severity: number
  category: string
  category_label: string
  secondary_tags: string[]
  explanation: string
  climate: {
    precip_mm: number | null; precip_zscore: number | null
    temp_c: number | null; temp_zscore: number | null
    etp_mm: number | null; etp_zscore: number | null
    spli: number | null; spli_class: string | null
    spi: number | null; spi_class: string | null
  }
  contributions: Record<string, number>
  observed: number
  simulated: number
  data_quality: {
    gap_days: number; coverage_pct: number
    nearest_gap_distance_days: number | null
  }
  neighbors: {
    total: number; anomalous: number
    neighbor_zscores: { code_bss: string; zscore: number }[]
  }
}

interface Props {
  outlier: OutlierDiagnostic
  onClose: () => void
}

const CATEGORY_COLORS: Record<string, string> = {
  DATA_GAP: 'bg-red-500/20 text-red-400 border-red-500/30',
  CLIMATE_EXTREME: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  REGIONAL_SIGNAL: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  SEASONAL_BIAS: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  DOMINANT_CONTRIBUTION: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  UNKNOWN: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
}

function CategoryBadge({ category, label, outline = false }: { category: string; label: string; outline?: boolean }) {
  const color = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.UNKNOWN
  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${color} ${outline ? 'bg-transparent' : ''}`}>
      {label}
    </span>
  )
}

function SeverityDots({ severity }: { severity: number }) {
  const filled = Math.max(1, Math.round(severity * 4))
  return (
    <span className="flex gap-0.5" title={`Severity: ${(severity * 100).toFixed(0)}%`}>
      {[1, 2, 3, 4].map(i => (
        <span key={i} className={`w-1.5 h-1.5 rounded-full ${i <= filled ? 'bg-red-400' : 'bg-white/10'}`} />
      ))}
    </span>
  )
}

function ZscoreBadge({ value, label, unit }: { value: number | null; label: string; unit?: string }) {
  if (value == null) return null
  const isAnomaly = Math.abs(value) > 1.5
  return (
    <div className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
      <span className="text-[10px] text-text-muted">{label}</span>
      <span className={`text-[10px] font-mono ${isAnomaly ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
        {unit}{value > 0 ? '+' : ''}{value.toFixed(1)}σ
      </span>
    </div>
  )
}

export function OutlierDetailPanel({ outlier, onClose }: Props) {
  const { climate, contributions, data_quality, neighbors } = outlier

  return (
    <div className="mt-2 bg-bg-card border border-white/10 rounded-lg overflow-hidden animate-in slide-in-from-top-2 duration-200">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5">
        <CategoryBadge category={outlier.category} label={outlier.category_label} />
        {outlier.secondary_tags.map(tag => (
          <CategoryBadge key={tag} category={tag} label={tag.replace(/_/g, ' ').toLowerCase()} outline />
        ))}
        <span className="text-xs text-text-primary font-medium ml-1">
          {new Date(outlier.date).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}
        </span>
        <span className="text-xs text-text-muted">|</span>
        <span className="text-xs font-mono text-text-secondary">
          {outlier.residual > 0 ? '+' : ''}{outlier.residual.toFixed(3)}m ({outlier.residual_zscore.toFixed(1)}σ)
        </span>
        <SeverityDots severity={outlier.severity} />
        <button onClick={onClose} className="ml-auto p-1 hover:bg-bg-hover rounded transition-colors">
          <X className="w-3.5 h-3.5 text-text-muted" />
        </button>
      </div>

      {/* Explanation */}
      <div className="px-4 py-2 border-b border-white/5">
        <p className="text-xs text-text-secondary leading-relaxed">{outlier.explanation}</p>
      </div>

      {/* Context grid */}
      <div className="grid grid-cols-3 gap-px bg-white/5">
        {/* Climate column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Climate</div>
          {climate.precip_mm != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">Precip</span>
              <span className={`text-[10px] font-mono ${climate.precip_zscore && Math.abs(climate.precip_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.precip_mm.toFixed(0)}mm
                {climate.precip_zscore != null && ` (${climate.precip_zscore > 0 ? '+' : ''}${climate.precip_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.temp_c != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">Temp</span>
              <span className={`text-[10px] font-mono ${climate.temp_zscore && Math.abs(climate.temp_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.temp_c.toFixed(1)}°C
                {climate.temp_zscore != null && ` (${climate.temp_zscore > 0 ? '+' : ''}${climate.temp_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.etp_mm != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">ETP</span>
              <span className={`text-[10px] font-mono ${climate.etp_zscore && Math.abs(climate.etp_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.etp_mm.toFixed(1)}mm
                {climate.etp_zscore != null && ` (${climate.etp_zscore > 0 ? '+' : ''}${climate.etp_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.spli != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">SPLI</span>
              <span className="text-[10px] font-mono text-text-secondary">{climate.spli.toFixed(2)} — {climate.spli_class}</span>
            </div>
          )}
          {climate.spi != null && (
            <div className="flex items-center justify-between py-1">
              <span className="text-[10px] text-text-muted">SPI</span>
              <span className="text-[10px] font-mono text-text-secondary">{climate.spi.toFixed(2)} — {climate.spi_class}</span>
            </div>
          )}
        </div>

        {/* Model column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Model</div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Observed</span>
            <span className="text-[10px] font-mono text-text-secondary">{outlier.observed.toFixed(3)}m</span>
          </div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Simulated</span>
            <span className="text-[10px] font-mono text-accent-cyan">{outlier.simulated.toFixed(3)}m</span>
          </div>
          {Object.entries(contributions).map(([name, value]) => (
            <div key={name} className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
              <span className="text-[10px] text-text-muted truncate mr-2">{name}</span>
              <span className="text-[10px] font-mono text-text-secondary">{value > 0 ? '+' : ''}{value.toFixed(3)}m</span>
            </div>
          ))}
        </div>

        {/* Data quality column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Data Quality</div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Coverage (±30d)</span>
            <span className={`text-[10px] font-mono ${data_quality.coverage_pct < 90 ? 'text-orange-400' : 'text-text-secondary'}`}>
              {data_quality.coverage_pct.toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Gap days</span>
            <span className={`text-[10px] font-mono ${data_quality.gap_days > 0 ? 'text-red-400' : 'text-text-secondary'}`}>
              {data_quality.gap_days}
            </span>
          </div>
          {data_quality.nearest_gap_distance_days != null && (
            <div className="flex items-center justify-between py-1">
              <span className="text-[10px] text-text-muted">Nearest gap</span>
              <span className="text-[10px] font-mono text-text-secondary">{data_quality.nearest_gap_distance_days}d</span>
            </div>
          )}
        </div>
      </div>

      {/* Neighbors */}
      {neighbors.total > 0 && (
        <div className="px-4 py-2.5 border-t border-white/5">
          <span className="text-[10px] text-text-muted">
            BDLISA neighbors: <span className={neighbors.anomalous > 0 ? 'text-blue-400 font-medium' : ''}>{neighbors.anomalous}/{neighbors.total} anomalous</span>
          </span>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {neighbors.neighbor_zscores.map(n => (
              <span
                key={n.code_bss}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${
                  Math.abs(n.zscore) > 1.5
                    ? 'border-red-500/30 bg-red-500/10 text-red-400'
                    : 'border-white/10 bg-white/5 text-text-muted'
                }`}
              >
                {n.code_bss.split('/').pop()}: {n.zscore > 0 ? '+' : ''}{n.zscore.toFixed(1)}σ
              </span>
            ))}
          </div>
        </div>
      )}
      {neighbors.total === 0 && (
        <div className="px-4 py-2 border-t border-white/5">
          <span className="text-[10px] text-text-muted">No BDLISA neighbors found</span>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/pastas/OutlierDetailPanel.tsx
git commit -m "feat(pastas): add OutlierDetailPanel component"
```

---

### Task 8: Integrate into FitResultsPanel

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Add imports and hook call**

In `FitResultsPanel.tsx`, update the import line (line 8):

```typescript
import { usePastasDiagnostics, usePastasSignatures, usePastasOutlierDiagnostics } from '@/hooks/usePastas'
```

Add import for the new component (after line 12):

```typescript
import { OutlierDetailPanel } from './OutlierDetailPanel'
```

- [ ] **Step 2: Add state and hook in component body**

Inside `FitResultsPanel` function, after line 124 (`const { data: signaturesData } = ...`), add:

```typescript
  const { data: outlierData } = usePastasOutlierDiagnostics(result.run_id)
  const [selectedOutlierDate, setSelectedOutlierDate] = useState<string | null>(null)
```

- [ ] **Step 3: Replace the residuals chart section**

Replace the entire `{/* 5. Residuals & diagnostics */}` section (lines 257-286) with:

```tsx
      {/* 5. Residuals & diagnostics */}
      <Section title="Residuals & Diagnostics">
        {outlierData && outlierData.n_outliers > 0 && (
          <div className="flex items-center gap-2 mb-2 text-xs text-text-muted">
            <span className="font-medium text-text-secondary">{outlierData.n_outliers} outliers detected</span>
            <span>—</span>
            {Object.entries(outlierData.summary?.by_category ?? {}).map(([cat, count]: [string, any]) => (
              <span key={cat}>{count} {cat.replace(/_/g, ' ').toLowerCase()}</span>
            )).reduce((prev: any, curr: any, i: number) => i === 0 ? [curr] : [...prev, <span key={`sep-${i}`}>,</span>, curr], [] as any)}
            <span className="text-[10px]">— click a red bar to investigate</span>
          </div>
        )}
        {residuals?.index?.length > 0 && (() => {
          const vals = residuals.values.filter(v => Number.isFinite(v))
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length
          const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
          const threshold = 2 * std

          const isOutlier = residuals.values.map(v => Math.abs(v) > threshold)
          const barColors = residuals.values.map((v, i) => {
            if (!isOutlier[i]) return 'rgba(245,158,11,0.5)'
            return residuals.index[i] === selectedOutlierDate ? 'rgba(239,68,68,1.0)' : 'rgba(239,68,68,0.7)'
          })

          return (
            <div className="mb-3">
              <p className="text-xs text-text-muted mb-1">
                {outlierData ? 'Click a red bar to see outlier diagnostics' : 'Red bars = error exceeding 2 standard deviations'}
              </p>
              <Plot
                data={[{
                  x: residuals.index, y: residuals.values, type: 'bar', name: 'Residuals',
                  marker: { color: barColors },
                  customdata: isOutlier,
                }]}
                layout={{
                  ...chartLayout, height: 160,
                  shapes: [
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: threshold, y1: threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: -threshold, y1: -threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                  ],
                }}
                config={plotlyConfig}
                style={{ width: '100%', cursor: 'default' }}
                onClick={(event: any) => {
                  if (!outlierData || !event.points?.[0]) return
                  const point = event.points[0]
                  if (!point.customdata) return
                  const clickedDate = point.x as string
                  setSelectedOutlierDate(prev => prev === clickedDate ? null : clickedDate)
                }}
              />
            </div>
          )
        })()}
        {selectedOutlierDate && outlierData && (() => {
          const outlier = outlierData.outliers?.find((o: any) => o.date === selectedOutlierDate)
          if (!outlier) return null
          return <OutlierDetailPanel outlier={outlier} onClose={() => setSelectedOutlierDate(null)} />
        })()}
        {diagnosticsData && <DiagnosticsPanel diagnostics={diagnosticsData} />}
        {!diagnosticsData && <p className="text-xs text-text-muted">Loading diagnostics...</p>}
      </Section>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx
git commit -m "feat(pastas): integrate outlier diagnostics into FitResultsPanel"
```

---

### Task 9: Build & Smoke Test

**Files:** None (testing only)

- [ ] **Step 1: Run backend tests**

```bash
pytest tests/pastas/test_outlier_diagnostics.py -v --no-header
```
Expected: All tests pass

- [ ] **Step 2: Rebuild frontend**

```bash
docker compose up -d --build frontend
```
Expected: Build succeeds

- [ ] **Step 3: Rebuild backend**

```bash
docker compose up -d --build backend
```
Expected: Container starts without import errors

- [ ] **Step 4: Smoke test API**

Find a Pastas model run_id and call the endpoint:
```bash
curl -s http://localhost:49513/api/v1/pastas/models | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['run_id'] if data else 'no models')"
```

Then test the outlier diagnostics endpoint:
```bash
curl -s "http://localhost:49513/api/v1/pastas/models/{RUN_ID}/outlier-diagnostics" | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'Outliers: {d[\"n_outliers\"]}')
print(f'Categories: {d[\"summary\"][\"by_category\"]}')
if d['outliers']:
    o = d['outliers'][0]
    print(f'Top outlier: {o[\"date\"]} — {o[\"category\"]} — {o[\"explanation\"][:80]}...')
"
```

- [ ] **Step 5: Commit all remaining changes**

```bash
git add -A
git commit -m "feat(pastas): outlier diagnostics module complete"
```
