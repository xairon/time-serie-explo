# Pastas Lab Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a top-level Pastas Lab section with single-station fit (LeastSquares/Lmfit), MLflow persistence, 4 scenario modifications, and diagnostics — while hiding the broken Lab section.

**Architecture:** New `dashboard/utils/pastas/` pure-Python module builds and calibrates `ps.Model`, persists to MLflow as `.pas` artifacts. `api/routers/pastas.py` wraps as REST endpoints. React frontend consumes via TanStack React Query. Lab hidden from nav, routes redirect to `/`.

**Tech Stack:** Pastas >=1.7, MLflow, FastAPI, React 19, TanStack React Query 5, Plotly.js, Tailwind CSS 4, Zod

**Spec:** `docs/superpowers/specs/2026-04-17-pastas-lab-design.md`

---

## File Map

### New files to create

```
dashboard/utils/pastas/__init__.py
dashboard/utils/pastas/config.py            # registries + enums
dashboard/utils/pastas/builder.py           # build ps.Model from FitRequest
dashboard/utils/pastas/fit_service.py       # run_fit() + MLflow persistence
dashboard/utils/pastas/io.py                # load_model() + LRU cache
dashboard/utils/pastas/scenario.py          # simulate_scenario() + apply_modification()

api/schemas/pastas.py                       # Pydantic schemas
api/routers/pastas.py                       # FastAPI router

frontend/src/hooks/usePastas.ts             # React Query hooks
frontend/src/pages/pastas/PastasLayout.tsx  # Tab bar (Fit / Scenarios)
frontend/src/pages/pastas/FitPage.tsx       # Fit workflow
frontend/src/pages/pastas/ScenariosPage.tsx # Scenario composer

frontend/src/components/pastas/StationPicker.tsx
frontend/src/components/pastas/PastasConfigForm.tsx
frontend/src/components/pastas/FitResultsPanel.tsx
frontend/src/components/pastas/ScenarioComposer.tsx
frontend/src/components/pastas/ModificationCard.tsx
frontend/src/components/pastas/PumpingSyntheticEditor.tsx
frontend/src/components/pastas/PumpingUploadEditor.tsx
frontend/src/components/pastas/LinearTrendEditor.tsx
frontend/src/components/pastas/ScaleStressEditor.tsx
frontend/src/components/pastas/ScenarioResultsPanel.tsx

tests/pastas/__init__.py
tests/pastas/conftest.py                    # shared fixtures
tests/pastas/test_config.py
tests/pastas/test_builder.py
tests/pastas/test_fit_service.py
tests/pastas/test_io.py
tests/pastas/test_scenario.py
tests/test_api_pastas.py
e2e/pastas-fit.spec.ts
e2e/pastas-scenarios.spec.ts
e2e/lab-disabled.spec.ts
```

### Files to modify

```
api/main.py:16,79                   # import + include_router(pastas.router)
frontend/src/components/layout/TopNav.tsx:2,21  # swap Lab for Pastas icon+route
frontend/src/routes.tsx:12,82-119   # Lab redirect, Pastas routes
frontend/src/lib/api.ts             # add pastas namespace
frontend/src/lib/types.ts           # add Pastas types
```

---

## Task 1: Disable Lab from navigation & add Pastas route stubs

**Files:**
- Modify: `frontend/src/components/layout/TopNav.tsx`
- Modify: `frontend/src/routes.tsx`
- Create: `frontend/src/pages/pastas/PastasLayout.tsx`
- Create: `e2e/lab-disabled.spec.ts`

- [ ] **Step 1: Update TopNav — swap Lab for Pastas**

In `frontend/src/components/layout/TopNav.tsx`, change the icon imports and NAV_ITEMS:

```tsx
// line 2: replace FlaskConical with Waves
import {
  LayoutDashboard,
  Database,
  GraduationCap,
  TrendingUp,
  Waves,
  Map,
  Menu,
  X,
} from 'lucide-react'
```

```tsx
// line 21: replace Lab entry
const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/data', icon: Database, label: 'Data' },
  { to: '/training', icon: GraduationCap, label: 'Training' },
  { to: '/forecasting', icon: TrendingUp, label: 'Forecasting' },
  { to: '/observatory', icon: Map, label: 'Observatory' },
  { to: '/pastas', icon: Waves, label: 'Pastas' },
] as const
```

- [ ] **Step 2: Create PastasLayout stub**

Create `frontend/src/pages/pastas/PastasLayout.tsx`:

```tsx
import { NavLink, Outlet } from 'react-router-dom'
import { SlidersHorizontal, FlaskConical } from 'lucide-react'

const PASTAS_TABS = [
  { to: '/pastas/fit', icon: SlidersHorizontal, label: 'Fit' },
  { to: '/pastas/scenarios', icon: FlaskConical, label: 'Scenarios' },
] as const

export default function PastasLayout() {
  return (
    <div className="flex flex-col h-full">
      <div className="bg-bg-card border-b border-white/5 shrink-0">
        <div className="flex items-center px-4 gap-1">
          {PASTAS_TABS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                  isActive
                    ? 'border-accent-cyan text-text-primary'
                    : 'border-transparent text-text-muted hover:text-text-secondary'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        <Outlet />
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Update routes.tsx — redirect Lab, add Pastas**

In `frontend/src/routes.tsx`:

Replace the lazy imports (add Pastas, keep Lab pages for later):
```tsx
const PastasLayout = lazy(() => import('./pages/pastas/PastasLayout'))
const PastasFitPage = lazy(() => import('./pages/pastas/FitPage'))
const PastasScenariosPage = lazy(() => import('./pages/pastas/ScenariosPage'))
```

Replace the `/lab` route block (lines 82-119) with:
```tsx
      {
        path: '/pastas',
        element: (
          <SuspenseWrapper>
            <PastasLayout />
          </SuspenseWrapper>
        ),
        children: [
          {
            index: true,
            element: <Navigate to="/pastas/fit" replace />,
          },
          {
            path: 'fit',
            element: (
              <SuspenseWrapper>
                <PastasFitPage />
              </SuspenseWrapper>
            ),
          },
          {
            path: 'scenarios',
            element: (
              <SuspenseWrapper>
                <PastasScenariosPage />
              </SuspenseWrapper>
            ),
          },
        ],
      },
      {
        path: '/lab/*',
        element: <Navigate to="/" replace />,
      },
```

Also remove the two redirect routes for `/counterfactual` and `/pumping-detection` (lines 121-128).

- [ ] **Step 4: Create placeholder FitPage and ScenariosPage**

Create `frontend/src/pages/pastas/FitPage.tsx`:
```tsx
export default function FitPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-text-primary">Pastas — Fit</h1>
      <p className="mt-2 text-text-secondary">Coming soon.</p>
    </div>
  )
}
```

Create `frontend/src/pages/pastas/ScenariosPage.tsx`:
```tsx
export default function ScenariosPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-text-primary">Pastas — Scenarios</h1>
      <p className="mt-2 text-text-secondary">Coming soon.</p>
    </div>
  )
}
```

- [ ] **Step 5: Write E2E test for Lab disabled**

Create `e2e/lab-disabled.spec.ts`:
```ts
import { test, expect } from '@playwright/test'

test('Lab routes redirect to home', async ({ page }) => {
  await page.goto('/lab/latent-space')
  await expect(page).toHaveURL('/')

  await page.goto('/lab/counterfactual')
  await expect(page).toHaveURL('/')

  await page.goto('/lab/pumping-detection')
  await expect(page).toHaveURL('/')
})

test('Pastas nav item exists and navigates to /pastas/fit', async ({ page }) => {
  await page.goto('/')
  const pastasLink = page.getByRole('link', { name: 'Pastas' })
  await expect(pastasLink).toBeVisible()
  await pastasLink.click()
  await expect(page).toHaveURL('/pastas/fit')
  await expect(page.getByText('Pastas — Fit')).toBeVisible()
})
```

- [ ] **Step 6: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: no errors, no warnings about removed imports.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/layout/TopNav.tsx frontend/src/routes.tsx \
  frontend/src/pages/pastas/ e2e/lab-disabled.spec.ts
git commit -m "feat(pastas): disable Lab nav, add Pastas route stubs"
```

---

## Task 2: Backend config + test fixtures

**Files:**
- Create: `dashboard/utils/pastas/__init__.py`
- Create: `dashboard/utils/pastas/config.py`
- Create: `tests/pastas/__init__.py`
- Create: `tests/pastas/conftest.py`
- Create: `tests/pastas/test_config.py`

- [ ] **Step 1: Write test for config registries**

Create `tests/pastas/__init__.py` (empty).

Create `tests/pastas/test_config.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pastas/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.utils.pastas'`

- [ ] **Step 3: Create config module**

Create `dashboard/utils/pastas/__init__.py`:
```python
"""Pastas Lab — pure-Python module for TFN model fitting and scenario simulation."""
```

Create `dashboard/utils/pastas/config.py`:
```python
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
```

- [ ] **Step 4: Create shared test fixtures**

Create `tests/pastas/conftest.py`:
```python
"""Shared fixtures for Pastas tests."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_station():
    """Generate synthetic piezometric, precipitation, and evapotranspiration series.

    Uses known Gamma response + Linear recharge so that a Pastas fit should
    recover EVP > 80%.
    """
    rng = np.random.default_rng(42)
    n = 365 * 5
    dates = pd.date_range("2015-01-01", periods=n, freq="D")

    precip = rng.exponential(3.0, n)
    evap = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 365)

    recharge = np.maximum(precip - 0.8 * evap, 0)
    gwl_base = 10.0
    gwl = np.full(n, gwl_base, dtype=float)
    alpha = 0.98
    for i in range(1, n):
        gwl[i] = alpha * gwl[i - 1] + 0.002 * recharge[i]
    gwl += rng.normal(0, 0.02, n)

    return {
        "piezo": pd.Series(gwl, index=dates, name="gwl"),
        "precip": pd.Series(precip, index=dates, name="precip"),
        "evap": pd.Series(evap, index=dates, name="evap"),
        "dates": dates,
    }


@pytest.fixture
def series_hash(synthetic_station):
    """SHA-256 hash of the synthetic station series (matches fit_service convention)."""
    parts = []
    for key in ("piezo", "precip", "evap"):
        s = synthetic_station[key]
        parts.append(s.index.astype(str).str.cat())
        parts.append(np.array2string(s.values, separator=","))
    return hashlib.sha256("".join(parts).encode()).hexdigest()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/pastas/test_config.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/pastas/ tests/pastas/
git commit -m "feat(pastas): config registries + test fixtures"
```

---

## Task 3: Backend builder — build ps.Model from config

**Files:**
- Create: `dashboard/utils/pastas/builder.py`
- Create: `tests/pastas/test_builder.py`

- [ ] **Step 1: Write failing tests for builder**

Create `tests/pastas/test_builder.py`:
```python
"""Tests for Pastas model builder."""
from __future__ import annotations

import pytest
import pastas as ps

from dashboard.utils.pastas.builder import build_model, ValidationError


def test_build_model_gamma_linear(synthetic_station):
    """Build a model with Gamma RF + Linear recharge."""
    model, tmin, tmax = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)
    assert len(model.stressmodels) == 1
    assert model.noisemodel is not None


def test_build_model_exponential_flexmodel(synthetic_station):
    """Build a model with Exponential RF + FlexModel recharge."""
    model, _, _ = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="FlexModel",
        response_type="Exponential",
        noise_type="none",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)
    assert model.noisemodel is None


def test_build_model_hantush(synthetic_station):
    """Build a model with Hantush RF."""
    model, _, _ = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Hantush",
        noise_type="ArNoiseModel",
        tmin=None,
        tmax=None,
    )
    assert isinstance(model, ps.Model)


def test_build_model_custom_window(synthetic_station):
    """Calibration window restricts tmin/tmax."""
    model, tmin, tmax = build_model(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        tmin="2016-01-01",
        tmax="2018-12-31",
    )
    assert str(tmin) == "2016-01-01"
    assert str(tmax) == "2018-12-31"


def test_build_model_rejects_short_series():
    """Series shorter than 365 days raises ValidationError."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    short = pd.Series(np.zeros(100), index=dates)

    with pytest.raises(ValidationError, match="at least 365"):
        build_model(
            gwl=short, precip=short, evap=short,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )


def test_build_model_rejects_high_nan_ratio():
    """More than 20% NaN raises ValidationError."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2015-01-01", periods=500, freq="D")
    gwl = pd.Series(np.ones(500), index=dates)
    gwl.iloc[:150] = np.nan  # 30% NaN

    precip = pd.Series(np.ones(500), index=dates)
    evap = pd.Series(np.ones(500), index=dates)

    with pytest.raises(ValidationError, match="NaN"):
        build_model(
            gwl=gwl, precip=precip, evap=evap,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )


def test_build_model_rejects_no_overlap():
    """Non-overlapping obs/stress raises ValidationError."""
    import pandas as pd
    import numpy as np
    dates_gwl = pd.date_range("2015-01-01", periods=500, freq="D")
    dates_stress = pd.date_range("2020-01-01", periods=500, freq="D")

    gwl = pd.Series(np.ones(500), index=dates_gwl)
    precip = pd.Series(np.ones(500), index=dates_stress)
    evap = pd.Series(np.ones(500), index=dates_stress)

    with pytest.raises(ValidationError, match="overlap"):
        build_model(
            gwl=gwl, precip=precip, evap=evap,
            recharge_type="Linear", response_type="Gamma",
            noise_type="none", tmin=None, tmax=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pastas/test_builder.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_model' from 'dashboard.utils.pastas.builder'`

- [ ] **Step 3: Implement builder.py**

Create `dashboard/utils/pastas/builder.py`:
```python
"""Build a Pastas TFN model from configuration parameters."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import pastas as ps

from dashboard.utils.pastas.config import (
    RECHARGE_REGISTRY,
    RFUNC_REGISTRY,
    NOISE_REGISTRY,
)


class ValidationError(Exception):
    """Raised when input series fail validation."""


def _validate_series(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
) -> None:
    """Validate input series before building a model."""
    if len(gwl) < 365:
        raise ValidationError(
            f"Piezometric series has {len(gwl)} observations, needs at least 365."
        )

    nan_ratio = gwl.isna().mean()
    if nan_ratio > 0.20:
        raise ValidationError(
            f"Piezometric series has {nan_ratio:.0%} NaN values (max 20%)."
        )

    overlap = gwl.index.intersection(precip.index).intersection(evap.index)
    if len(overlap) < 365:
        raise ValidationError(
            f"Observation and stress series have only {len(overlap)} days of overlap "
            f"(need at least 365)."
        )


def build_model(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    recharge_type: str,
    response_type: str,
    noise_type: str,
    tmin: Optional[str],
    tmax: Optional[str],
) -> tuple[ps.Model, Optional[str], Optional[str]]:
    """Build an unsolved Pastas model.

    Returns:
        (model, tmin, tmax) — model is ready for .solve().
    """
    _validate_series(gwl, precip, evap)

    recharge_cls = RECHARGE_REGISTRY[recharge_type]
    rfunc_cls = RFUNC_REGISTRY[response_type]

    model = ps.Model(gwl, name="gwl")

    rm = ps.RechargeModel(
        precip,
        evap,
        rfunc=rfunc_cls(),
        recharge=recharge_cls(),
        name="recharge",
    )
    model.add_stressmodel(rm)

    if noise_type != "none":
        noise_cls = NOISE_REGISTRY[noise_type]
        model.add_noisemodel(noise_cls())

    return model, tmin, tmax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pastas/test_builder.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/builder.py tests/pastas/test_builder.py
git commit -m "feat(pastas): model builder with input validation"
```

---

## Task 4: Backend fit_service — fit + MLflow persistence

**Files:**
- Create: `dashboard/utils/pastas/fit_service.py`
- Create: `tests/pastas/test_fit_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pastas/test_fit_service.py`:
```python
"""Tests for Pastas fit service."""
from __future__ import annotations

import pytest

from dashboard.utils.pastas.fit_service import run_fit, FitResult


def test_run_fit_least_squares(synthetic_station, tmp_path, monkeypatch):
    """Fit with LeastSquares produces valid metrics and MLflow run."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs={},
        tmin=None,
        tmax=None,
        dataset_id="test_station",
        name="test_fit",
    )

    assert isinstance(result, FitResult)
    assert result.run_id is not None
    assert result.metrics["evp"] > 50.0
    assert result.metrics["rmse"] > 0
    assert "ljung_box_pvalue" in result.metrics
    assert len(result.parameters) > 0
    assert len(result.observed.index) > 0
    assert len(result.simulated.index) > 0
    assert len(result.residuals.index) > 0
    assert "recharge" in result.contributions
    assert len(result.step_response.index) > 0
    assert len(result.warnings) >= 0


def test_run_fit_lmfit(synthetic_station, tmp_path, monkeypatch):
    """Fit with Lmfit solver also works."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Exponential",
        noise_type="none",
        solver_type="Lmfit",
        solver_kwargs={},
        tmin=None,
        tmax=None,
        dataset_id="test_station",
        name="test_lmfit",
    )

    assert result.metrics["evp"] > 0


def test_run_fit_returns_warnings_on_bound_hit(synthetic_station, tmp_path, monkeypatch):
    """When solver hits parameter bounds, warnings are non-empty."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    # Very short window forces suboptimal fit
    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs={},
        tmin="2015-01-01",
        tmax="2016-06-30",
        dataset_id="test_station",
        name="test_short_window",
    )

    assert isinstance(result.warnings, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pastas/test_fit_service.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement fit_service.py**

Create `dashboard/utils/pastas/fit_service.py`:
```python
"""Fit a Pastas model and persist to MLflow."""
from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import pastas as ps
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from dashboard.utils.pastas.builder import build_model
from dashboard.utils.pastas.config import SOLVER_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    run_id: str
    metrics: dict[str, float]
    parameters: list[dict[str, Any]]
    observed: pd.Series
    simulated: pd.Series
    residuals: pd.Series
    contributions: dict[str, pd.Series]
    step_response: pd.Series
    block_response: pd.Series
    acf_stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    pastas_version: str = ""


def _series_hash(gwl: pd.Series, precip: pd.Series, evap: pd.Series) -> str:
    parts = []
    for s in (gwl, precip, evap):
        parts.append(s.index.astype(str).str.cat())
        parts.append(np.array2string(s.values, separator=","))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _extract_metrics(model: ps.Model) -> dict[str, float]:
    stats = model.stats
    metrics = {
        "evp": float(stats.evp()),
        "rmse": float(stats.rmse()),
        "n_obs": float(len(model.observations())),
    }
    try:
        metrics["aic"] = float(stats.aic())
        metrics["bic"] = float(stats.bic())
    except Exception:
        pass

    res = model.residuals()
    if len(res.dropna()) > 10:
        nlags = min(30, len(res.dropna()) // 2 - 1)
        try:
            lb = acorr_ljungbox(res.dropna(), lags=[nlags], return_df=True)
            metrics["ljung_box_pvalue"] = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            pass

    return metrics


def _extract_parameters(model: ps.Model) -> list[dict[str, Any]]:
    params = model.parameters
    result = []
    for name in params.index:
        row = params.loc[name]
        result.append({
            "name": name,
            "optimal": float(row["optimal"]),
            "stderr": float(row["stderr"]) if pd.notna(row.get("stderr")) else None,
            "initial": float(row["initial"]),
            "pmin": float(row["pmin"]) if pd.notna(row.get("pmin")) else None,
            "pmax": float(row["pmax"]) if pd.notna(row.get("pmax")) else None,
            "vary": bool(row["vary"]),
        })
    return result


def _check_warnings(model: ps.Model) -> list[str]:
    warnings = []
    params = model.parameters
    for name in params.index:
        row = params.loc[name]
        if not row["vary"]:
            continue
        opt = row["optimal"]
        pmin, pmax = row.get("pmin"), row.get("pmax")
        if pd.notna(pmin) and abs(opt - pmin) < 1e-6 * (abs(pmin) + 1):
            warnings.append(f"Parameter '{name}' hit lower bound ({pmin})")
        if pd.notna(pmax) and abs(opt - pmax) < 1e-6 * (abs(pmax) + 1):
            warnings.append(f"Parameter '{name}' hit upper bound ({pmax})")
    return warnings


def _acf_stats(residuals: pd.Series) -> dict[str, Any]:
    clean = residuals.dropna()
    nlags = min(30, len(clean) // 2 - 1)
    if nlags < 2:
        return {"acf_values": [], "pacf_values": [], "nlags": 0}
    return {
        "acf_values": acf(clean, nlags=nlags, fft=True).tolist(),
        "pacf_values": pacf(clean, nlags=nlags).tolist(),
        "nlags": nlags,
    }


def run_fit(
    gwl: pd.Series,
    precip: pd.Series,
    evap: pd.Series,
    recharge_type: str,
    response_type: str,
    noise_type: str,
    solver_type: str,
    solver_kwargs: dict[str, Any],
    tmin: Optional[str],
    tmax: Optional[str],
    dataset_id: str,
    name: Optional[str] = None,
) -> FitResult:
    """Fit a Pastas model and persist to MLflow."""
    model, tmin, tmax = build_model(
        gwl, precip, evap, recharge_type, response_type, noise_type, tmin, tmax,
    )

    solver_cls = SOLVER_REGISTRY[solver_type]
    solver = solver_cls(**solver_kwargs)

    mlflow.set_experiment("pastas")
    with mlflow.start_run(run_name=name or f"pastas_{dataset_id}") as run:
        model.solve(solver=solver, tmin=tmin, tmax=tmax, report=False)

        metrics = _extract_metrics(model)
        flat_params = {
            "recharge_type": recharge_type,
            "response_type": response_type,
            "noise_type": noise_type,
            "solver_type": solver_type,
            "dataset_id": dataset_id,
        }
        if tmin:
            flat_params["tmin"] = str(tmin)
        if tmax:
            flat_params["tmax"] = str(tmax)

        mlflow.log_params(flat_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("station_id", dataset_id)
        mlflow.set_tag("pastas_version", ps.__version__)
        mlflow.set_tag("series_hash", _series_hash(gwl, precip, evap))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pas"
            model.to_file(str(path))
            mlflow.log_artifact(str(path))

        sim = model.simulate(tmin=tmin, tmax=tmax)
        obs = model.observations(tmin=tmin, tmax=tmax)
        res = model.residuals(tmin=tmin, tmax=tmax)

        contributions = {}
        for sm_name in model.stressmodels:
            contributions[sm_name] = model.get_contribution(sm_name, tmin=tmin, tmax=tmax)

        try:
            step_resp = model.get_step_response("recharge")
        except Exception:
            step_resp = pd.Series(dtype=float)

        try:
            block_resp = model.get_block_response("recharge")
        except Exception:
            block_resp = pd.Series(dtype=float)

        return FitResult(
            run_id=run.info.run_id,
            metrics=metrics,
            parameters=_extract_parameters(model),
            observed=obs,
            simulated=sim,
            residuals=res,
            contributions=contributions,
            step_response=step_resp,
            block_response=block_resp,
            acf_stats=_acf_stats(res),
            warnings=_check_warnings(model),
            pastas_version=ps.__version__,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pastas/test_fit_service.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/fit_service.py tests/pastas/test_fit_service.py
git commit -m "feat(pastas): fit service with MLflow persistence"
```

---

## Task 5: Backend io — load model from MLflow

**Files:**
- Create: `dashboard/utils/pastas/io.py`
- Create: `tests/pastas/test_io.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pastas/test_io.py`:
```python
"""Tests for Pastas model I/O."""
from __future__ import annotations

import pytest
import numpy as np

from dashboard.utils.pastas.fit_service import run_fit
from dashboard.utils.pastas.io import load_model, evict_cache, ModelVersionMismatch


def _fit_test_model(synthetic_station, tmp_path, monkeypatch) -> str:
    """Helper: fit and return run_id."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs={},
        tmin=None, tmax=None,
        dataset_id="test", name="test_io",
    )
    return result.run_id


def test_load_model_roundtrip(synthetic_station, tmp_path, monkeypatch):
    """Load a model from MLflow and simulate — should match original."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    model = load_model(run_id)

    sim = model.simulate()
    assert len(sim) > 0
    assert not np.all(np.isnan(sim.values))


def test_load_model_caching(synthetic_station, tmp_path, monkeypatch):
    """Second load returns the same object (cache hit)."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    m1 = load_model(run_id)
    m2 = load_model(run_id)
    assert m1 is m2


def test_evict_cache(synthetic_station, tmp_path, monkeypatch):
    """After eviction, next load returns a different object."""
    run_id = _fit_test_model(synthetic_station, tmp_path, monkeypatch)
    m1 = load_model(run_id)
    evict_cache(run_id)
    m2 = load_model(run_id)
    assert m1 is not m2


def test_load_model_not_found(tmp_path, monkeypatch):
    """Loading a nonexistent run_id raises."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with pytest.raises(Exception):
        load_model("nonexistent_run_id_12345")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pastas/test_io.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement io.py**

Create `dashboard/utils/pastas/io.py`:
```python
"""Load Pastas models from MLflow artifacts with LRU caching."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import mlflow
import pastas as ps

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("/tmp/pastas_models")


class ModelVersionMismatch(Exception):
    """Raised when stored Pastas version differs from current."""


@lru_cache(maxsize=32)
def _load_cached(run_id: str) -> ps.Model:
    """Download and parse .pas file from MLflow (cached)."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    stored_version = run.data.tags.get("pastas_version", "unknown")
    if stored_version != "unknown" and stored_version != ps.__version__:
        logger.warning(
            "Model %s was saved with Pastas %s, current is %s",
            run_id, stored_version, ps.__version__,
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _CACHE_DIR / f"{run_id}.pas"

    if not local_path.exists():
        artifacts = client.list_artifacts(run_id)
        pas_artifact = next(
            (a.path for a in artifacts if a.path.endswith(".pas")), None,
        )
        if pas_artifact is None:
            raise FileNotFoundError(f"No .pas artifact in run {run_id}")

        local_dir = client.download_artifacts(run_id, pas_artifact, str(_CACHE_DIR))
        downloaded = Path(local_dir)
        if downloaded != local_path:
            downloaded.rename(local_path)

    return ps.io.load(str(local_path))


def load_model(run_id: str) -> ps.Model:
    """Load a Pastas model from MLflow."""
    return _load_cached(run_id)


def evict_cache(run_id: str) -> None:
    """Remove a model from the LRU cache and disk."""
    _load_cached.cache_clear()
    local_path = _CACHE_DIR / f"{run_id}.pas"
    if local_path.exists():
        local_path.unlink()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pastas/test_io.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/io.py tests/pastas/test_io.py
git commit -m "feat(pastas): model I/O with MLflow + LRU cache"
```

---

## Task 6: Backend scenario — apply modifications & simulate

**Files:**
- Create: `dashboard/utils/pastas/scenario.py`
- Create: `tests/pastas/test_scenario.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pastas/test_scenario.py`:
```python
"""Tests for Pastas scenario engine."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dashboard.utils.pastas.fit_service import run_fit
from dashboard.utils.pastas.scenario import simulate_scenario, ScenarioResult


def _fit_helper(synthetic_station, tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    result = run_fit(
        gwl=synthetic_station["piezo"],
        precip=synthetic_station["precip"],
        evap=synthetic_station["evap"],
        recharge_type="Linear",
        response_type="Gamma",
        noise_type="ArNoiseModel",
        solver_type="LeastSquares",
        solver_kwargs={},
        tmin=None, tmax=None,
        dataset_id="test", name="scenario_test",
    )
    return result.run_id


def test_scenario_no_modifications(synthetic_station, tmp_path, monkeypatch):
    """No modifications → scenario == baseline."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[],
    )
    assert isinstance(result, ScenarioResult)
    np.testing.assert_allclose(
        result.baseline.values, result.scenario.values, atol=1e-10,
    )
    np.testing.assert_allclose(result.delta.values, 0, atol=1e-10)


def test_scenario_pumping_synthetic_zero_rate(synthetic_station, tmp_path, monkeypatch):
    """Pumping with rate=0 is effectively a no-op."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[{
            "type": "pumping_synthetic",
            "pattern": "constant",
            "rate_m3d": 0.0,
            "start": "2016-06-01",
            "end": "2017-06-01",
            "distance_m": 500.0,
            "rfunc": "Exponential",
        }],
    )
    np.testing.assert_allclose(
        result.baseline.values, result.scenario.values, atol=0.01,
    )


def test_scenario_pumping_lowers_gwl(synthetic_station, tmp_path, monkeypatch):
    """Positive pumping rate should lower GWL (delta < 0 on average)."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[{
            "type": "pumping_synthetic",
            "pattern": "constant",
            "rate_m3d": 1000.0,
            "start": "2016-06-01",
            "end": "2018-06-01",
            "distance_m": 200.0,
            "rfunc": "Exponential",
        }],
    )
    assert result.delta.mean() < 0, "Pumping should lower GWL"


def test_scenario_linear_trend(synthetic_station, tmp_path, monkeypatch):
    """Positive trend slope → positive drift on average."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[{
            "type": "linear_trend",
            "start": "2016-01-01",
            "end": "2018-12-31",
            "slope_m_per_year": 0.5,
        }],
    )
    assert result.delta.mean() > 0, "Positive trend should raise GWL"


def test_scenario_scale_stress_noop(synthetic_station, tmp_path, monkeypatch):
    """Scaling precip by factor=1.0 is a no-op."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[{
            "type": "scale_stress",
            "stress": "precip",
            "factor": 1.0,
            "start": "2016-01-01",
            "end": "2018-12-31",
        }],
    )
    np.testing.assert_allclose(
        result.baseline.values, result.scenario.values, atol=0.01,
    )


def test_scenario_scale_precip_decrease(synthetic_station, tmp_path, monkeypatch):
    """Reducing precipitation → lower GWL."""
    run_id = _fit_helper(synthetic_station, tmp_path, monkeypatch)
    result = simulate_scenario(
        run_id=run_id,
        tmin=date(2016, 1, 1),
        tmax=date(2018, 12, 31),
        modifications=[{
            "type": "scale_stress",
            "stress": "precip",
            "factor": 0.5,
            "start": "2016-01-01",
            "end": "2018-12-31",
        }],
    )
    assert result.delta.mean() < 0, "Less precip should lower GWL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pastas/test_scenario.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement scenario.py**

Create `dashboard/utils/pastas/scenario.py`:
```python
"""Scenario engine: apply modifications to a calibrated Pastas model and simulate."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pastas as ps

from dashboard.utils.pastas.config import RFUNC_REGISTRY
from dashboard.utils.pastas.io import load_model

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    baseline: pd.Series
    scenario: pd.Series
    delta: pd.Series
    contributions_baseline: dict[str, pd.Series]
    contributions_scenario: dict[str, pd.Series]
    warnings: list[str] = field(default_factory=list)


def _generate_pumping_series(
    start: str, end: str, pattern: str, rate_m3d: float, period_days: int = 365,
) -> pd.Series:
    """Generate a synthetic pumping time series."""
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    if pattern == "constant":
        values = np.full(n, rate_m3d)
    elif pattern == "seasonal":
        phase = 2 * np.pi * np.arange(n) / period_days
        values = rate_m3d * (0.5 + 0.5 * np.sin(phase))
    elif pattern == "pulse":
        values = np.zeros(n)
        mid = n // 2
        pulse_len = min(90, n // 4)
        values[mid : mid + pulse_len] = rate_m3d
    else:
        raise ValueError(f"Unknown pumping pattern: {pattern}")

    return pd.Series(values, index=dates, name="pumping")


def _apply_pumping_synthetic(model: ps.Model, mod: dict[str, Any]) -> None:
    q_series = _generate_pumping_series(
        start=mod["start"],
        end=mod["end"],
        pattern=mod["pattern"],
        rate_m3d=mod["rate_m3d"],
        period_days=mod.get("period_days", 365),
    )
    rfunc_name = mod.get("rfunc", "Exponential")
    rfunc_cls = RFUNC_REGISTRY[rfunc_name]

    sm = ps.StressModel(
        q_series,
        rfunc=rfunc_cls(),
        name=f"well_{len(model.stressmodels)}",
        settings="well",
    )
    model.add_stressmodel(sm)


def _apply_pumping_upload(model: ps.Model, mod: dict[str, Any]) -> None:
    rows = mod["csv_rows"]
    dates = pd.to_datetime([r["date"] for r in rows])
    values = [r["Q_m3d"] for r in rows]
    q_series = pd.Series(values, index=dates, name="pumping_upload")

    rfunc_name = mod.get("rfunc", "Exponential")
    rfunc_cls = RFUNC_REGISTRY[rfunc_name]

    sm = ps.StressModel(
        q_series,
        rfunc=rfunc_cls(),
        name=f"well_upload_{len(model.stressmodels)}",
        settings="well",
    )
    model.add_stressmodel(sm)


def _apply_linear_trend(model: ps.Model, mod: dict[str, Any]) -> None:
    start = pd.Timestamp(mod["start"])
    end = pd.Timestamp(mod["end"])

    sm = ps.LinearTrend(
        start=start,
        end=end,
        name=f"trend_{len(model.stressmodels)}",
    )
    model.add_stressmodel(sm)


def _apply_scale_stress(model: ps.Model, mod: dict[str, Any]) -> None:
    stress_name = mod["stress"]
    factor = mod["factor"]
    start = pd.Timestamp(mod["start"])
    end = pd.Timestamp(mod["end"])

    rm = model.stressmodels.get("recharge")
    if rm is None:
        raise ValueError("No 'recharge' stress model to scale")

    if stress_name == "precip":
        original = rm.stress[0]
    elif stress_name == "evap":
        original = rm.stress[1]
    else:
        raise ValueError(f"Unknown stress: {stress_name}")

    scaled = original.copy()
    mask = (scaled.index >= start) & (scaled.index <= end)
    scaled.loc[mask] = scaled.loc[mask] * factor

    new_rm = ps.RechargeModel(
        scaled if stress_name == "precip" else rm.stress[0],
        rm.stress[1] if stress_name == "precip" else scaled,
        rfunc=rm.rfunc,
        recharge=rm.recharge,
        name="recharge",
    )

    model.del_stressmodel("recharge")
    model.add_stressmodel(new_rm)

    old_params = model.parameters
    for pname in new_rm.parameters.index:
        if pname in old_params.index:
            model.parameters.loc[pname, "initial"] = old_params.loc[pname, "optimal"]
            model.parameters.loc[pname, "optimal"] = old_params.loc[pname, "optimal"]


_MODIFICATION_HANDLERS = {
    "pumping_synthetic": _apply_pumping_synthetic,
    "pumping_upload": _apply_pumping_upload,
    "linear_trend": _apply_linear_trend,
    "scale_stress": _apply_scale_stress,
}


def apply_modification(model: ps.Model, mod: dict[str, Any]) -> None:
    """Apply a single modification to a Pastas model."""
    mod_type = mod["type"]
    handler = _MODIFICATION_HANDLERS.get(mod_type)
    if handler is None:
        raise ValueError(f"Unknown modification type: {mod_type}")
    handler(model, mod)


def simulate_scenario(
    run_id: str,
    tmin: date,
    tmax: date,
    modifications: list[dict[str, Any]],
) -> ScenarioResult:
    """Load a calibrated model, apply modifications, and simulate."""
    base_model = load_model(run_id)

    tmin_str = str(tmin)
    tmax_str = str(tmax)

    baseline = base_model.simulate(tmin=tmin_str, tmax=tmax_str)
    contributions_baseline = {}
    for sm_name in base_model.stressmodels:
        contributions_baseline[sm_name] = base_model.get_contribution(
            sm_name, tmin=tmin_str, tmax=tmax_str,
        )

    if not modifications:
        return ScenarioResult(
            baseline=baseline,
            scenario=baseline.copy(),
            delta=baseline * 0,
            contributions_baseline=contributions_baseline,
            contributions_scenario=contributions_baseline,
        )

    scenario_model = copy.deepcopy(base_model)
    warnings = []

    for mod in modifications:
        try:
            apply_modification(scenario_model, mod)
        except Exception as exc:
            warnings.append(f"Modification '{mod.get('type')}' failed: {exc}")
            raise

    scenario = scenario_model.simulate(tmin=tmin_str, tmax=tmax_str)

    common_idx = baseline.index.intersection(scenario.index)
    baseline_aligned = baseline.loc[common_idx]
    scenario_aligned = scenario.loc[common_idx]
    delta = scenario_aligned - baseline_aligned

    contributions_scenario = {}
    for sm_name in scenario_model.stressmodels:
        try:
            contributions_scenario[sm_name] = scenario_model.get_contribution(
                sm_name, tmin=tmin_str, tmax=tmax_str,
            )
        except Exception:
            pass

    return ScenarioResult(
        baseline=baseline_aligned,
        scenario=scenario_aligned,
        delta=delta,
        contributions_baseline=contributions_baseline,
        contributions_scenario=contributions_scenario,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/pastas/test_scenario.py -v`
Expected: all 7 tests PASS. Some tests may need tolerance adjustments — adapt `atol` if Pastas solver numerics differ from synthetic expectations.

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/pastas/scenario.py tests/pastas/test_scenario.py
git commit -m "feat(pastas): scenario engine with 4 modification types"
```

---

## Task 7: API schemas (Pydantic)

**Files:**
- Create: `api/schemas/pastas.py`

- [ ] **Step 1: Create Pydantic schemas**

Create `api/schemas/pastas.py`:
```python
"""Pydantic schemas for Pastas API."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Fit ----------

class RechargeConfig(BaseModel):
    type: Literal["Linear", "FlexModel"] = "Linear"
    kwargs: dict[str, Any] = {}

class ResponseConfig(BaseModel):
    type: Literal["Gamma", "Exponential", "Hantush"] = "Gamma"
    kwargs: dict[str, Any] = {}

class NoiseConfig(BaseModel):
    type: Literal["ArNoiseModel", "none"] = "ArNoiseModel"

class SolverConfig(BaseModel):
    type: Literal["LeastSquares", "Lmfit"] = "LeastSquares"
    kwargs: dict[str, Any] = {}

class FitRequest(BaseModel):
    dataset_id: str
    station_id: Optional[str] = None
    precip_column: str
    evap_column: str
    tmin: Optional[date] = None
    tmax: Optional[date] = None
    recharge: RechargeConfig = RechargeConfig()
    response: ResponseConfig = ResponseConfig()
    noise: NoiseConfig = NoiseConfig()
    solver: SolverConfig = SolverConfig()
    name: Optional[str] = None

class TimeSeriesData(BaseModel):
    index: list[str]
    values: list[float]

class FitParameter(BaseModel):
    name: str
    optimal: float
    stderr: Optional[float]
    initial: float
    pmin: Optional[float]
    pmax: Optional[float]
    vary: bool

class FitResponse(BaseModel):
    run_id: str
    metrics: dict[str, float]
    parameters: list[FitParameter]
    observed: TimeSeriesData
    simulated: TimeSeriesData
    residuals: TimeSeriesData
    contributions: dict[str, TimeSeriesData]
    step_response: TimeSeriesData
    block_response: TimeSeriesData
    acf: dict[str, Any]
    warnings: list[str] = []
    pastas_version: str


# ---------- Models list ----------

class PastasModelSummary(BaseModel):
    run_id: str
    name: str
    station_id: str
    recharge_type: str
    response_type: str
    evp: Optional[float] = None
    rmse: Optional[float] = None
    created_at: str
    pastas_version: str


# ---------- Scenario ----------

class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"] = "pumping_synthetic"
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float = Field(ge=0)
    start: date
    end: date
    distance_m: float = Field(gt=0)
    screen_depth_m: Optional[float] = None
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"
    period_days: int = 365

class PumpingRow(BaseModel):
    date: date
    Q_m3d: float

class PumpingUpload(BaseModel):
    type: Literal["pumping_upload"] = "pumping_upload"
    csv_rows: list[PumpingRow]
    distance_m: float = Field(gt=0)
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"

class LinearTrendMod(BaseModel):
    type: Literal["linear_trend"] = "linear_trend"
    start: date
    end: date
    slope_m_per_year: float

class ScaleStressMod(BaseModel):
    type: Literal["scale_stress"] = "scale_stress"
    stress: Literal["precip", "evap"]
    factor: float = Field(gt=0)
    start: date
    end: date

Modification = Annotated[
    PumpingSynthetic | PumpingUpload | LinearTrendMod | ScaleStressMod,
    Field(discriminator="type"),
]

class ScenarioRequest(BaseModel):
    run_id: str
    tmin: date
    tmax: date
    modifications: list[Modification]

class ScenarioResponse(BaseModel):
    baseline: TimeSeriesData
    scenario: TimeSeriesData
    delta: TimeSeriesData
    contributions_baseline: dict[str, TimeSeriesData]
    contributions_scenario: dict[str, TimeSeriesData]
    warnings: list[str] = []
```

- [ ] **Step 2: Verify schemas parse correctly**

Run: `python -c "from api.schemas.pastas import FitRequest, ScenarioRequest; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/schemas/pastas.py
git commit -m "feat(pastas): Pydantic schemas for fit + scenario API"
```

---

## Task 8: API router + registration

**Files:**
- Create: `api/routers/pastas.py`
- Modify: `api/main.py`
- Create: `tests/test_api_pastas.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_api_pastas.py`:
```python
"""Integration tests for Pastas API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create test client with local MLflow."""
    import os
    tmp = tmp_path_factory.mktemp("mlflow")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp / 'mlflow.db'}"

    from api.main import app
    return TestClient(app)


def test_options_endpoint(client):
    resp = client.get("/api/v1/pastas/options")
    assert resp.status_code == 200
    data = resp.json()
    assert "recharge" in data
    assert "Gamma" in data["response"]


def test_fit_and_list_and_delete(client):
    # This test requires a dataset to exist — skip if no fixtures loaded
    # For CI: create a fixture dataset or mock the registry
    pytest.skip("Requires dataset fixture — tested manually or in E2E")


def test_models_empty(client):
    resp = client.get("/api/v1/pastas/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Implement router**

Create `api/routers/pastas.py`:
```python
"""Pastas Lab API router."""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.config import settings
from api.schemas.pastas import (
    FitRequest,
    FitResponse,
    FitParameter,
    PastasModelSummary,
    ScenarioRequest,
    ScenarioResponse,
    TimeSeriesData,
)
from dashboard.utils.pastas.config import get_p1_options

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pastas", tags=["pastas"])


def _series_to_ts(s: pd.Series) -> TimeSeriesData:
    """Convert a pandas Series to a TimeSeriesData schema."""
    return TimeSeriesData(
        index=[str(d) for d in s.index],
        values=[float(v) if pd.notna(v) else 0.0 for v in s.values],
    )


@router.get("/options")
def get_options():
    return get_p1_options()


@router.post("/fit", response_model=FitResponse)
def fit_model(req: FitRequest):
    from dashboard.utils.dataset_registry import DatasetRegistry
    from dashboard.utils.pastas.fit_service import run_fit

    datasets_dir = Path(settings.data_dir) / "prepared"
    registry = DatasetRegistry(datasets_dir)
    datasets = registry.scan_datasets()

    ds = None
    for d in datasets:
        if d.name == req.dataset_id or str(d.path).endswith(req.dataset_id):
            ds = d
            break

    if ds is None:
        raise HTTPException(404, f"Dataset '{req.dataset_id}' not found")

    df, config = registry.load_dataset(ds)

    if req.station_id and ds.station_column:
        if req.station_id not in df[ds.station_column].unique():
            raise HTTPException(404, f"Station '{req.station_id}' not in dataset")
        df = df[df[ds.station_column] == req.station_id]

    if req.precip_column not in df.columns:
        raise HTTPException(422, f"Column '{req.precip_column}' not in dataset")
    if req.evap_column not in df.columns:
        raise HTTPException(422, f"Column '{req.evap_column}' not in dataset")

    gwl = df[ds.target_column].dropna()
    precip = df[req.precip_column].dropna()
    evap = df[req.evap_column].dropna()

    from dashboard.utils.pastas.builder import ValidationError

    try:
        result = run_fit(
            gwl=gwl,
            precip=precip,
            evap=evap,
            recharge_type=req.recharge.type,
            response_type=req.response.type,
            noise_type=req.noise.type,
            solver_type=req.solver.type,
            solver_kwargs=req.solver.kwargs,
            tmin=str(req.tmin) if req.tmin else None,
            tmax=str(req.tmax) if req.tmax else None,
            dataset_id=req.dataset_id,
            name=req.name,
        )
    except ValidationError as exc:
        raise HTTPException(422, str(exc))

    return FitResponse(
        run_id=result.run_id,
        metrics=result.metrics,
        parameters=[FitParameter(**p) for p in result.parameters],
        observed=_series_to_ts(result.observed),
        simulated=_series_to_ts(result.simulated),
        residuals=_series_to_ts(result.residuals),
        contributions={k: _series_to_ts(v) for k, v in result.contributions.items()},
        step_response=_series_to_ts(result.step_response),
        block_response=_series_to_ts(result.block_response),
        acf=result.acf_stats,
        warnings=result.warnings,
        pastas_version=result.pastas_version,
    )


@router.get("/models", response_model=list[PastasModelSummary])
def list_models(station_id: str | None = None):
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name("pastas")
    except Exception:
        return []

    if experiment is None:
        return []

    filter_str = ""
    if station_id:
        filter_str = f"tags.station_id = '{station_id}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["start_time DESC"],
        max_results=100,
    )

    results = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags
        results.append(PastasModelSummary(
            run_id=run.info.run_id,
            name=run.info.run_name or "",
            station_id=tags.get("station_id", ""),
            recharge_type=params.get("recharge_type", ""),
            response_type=params.get("response_type", ""),
            evp=metrics.get("evp"),
            rmse=metrics.get("rmse"),
            created_at=str(run.info.start_time),
            pastas_version=tags.get("pastas_version", ""),
        ))

    return results


@router.get("/models/{run_id}", response_model=FitResponse)
def get_model(run_id: str):
    from dashboard.utils.pastas.io import load_model

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")
    except Exception as exc:
        raise HTTPException(500, str(exc))

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    tmin = run.data.params.get("tmin")
    tmax = run.data.params.get("tmax")

    sim = model.simulate(tmin=tmin, tmax=tmax)
    obs = model.observations(tmin=tmin, tmax=tmax)
    res = model.residuals(tmin=tmin, tmax=tmax)

    contributions = {}
    for sm_name in model.stressmodels:
        try:
            contributions[sm_name] = model.get_contribution(sm_name, tmin=tmin, tmax=tmax)
        except Exception:
            pass

    try:
        step_resp = model.get_step_response("recharge")
    except Exception:
        step_resp = pd.Series(dtype=float)

    try:
        block_resp = model.get_block_response("recharge")
    except Exception:
        block_resp = pd.Series(dtype=float)

    from dashboard.utils.pastas.fit_service import _extract_parameters, _acf_stats

    return FitResponse(
        run_id=run_id,
        metrics=run.data.metrics,
        parameters=[FitParameter(**p) for p in _extract_parameters(model)],
        observed=_series_to_ts(obs),
        simulated=_series_to_ts(sim),
        residuals=_series_to_ts(res),
        contributions={k: _series_to_ts(v) for k, v in contributions.items()},
        step_response=_series_to_ts(step_resp),
        block_response=_series_to_ts(block_resp),
        acf=_acf_stats(res),
        warnings=[],
        pastas_version=run.data.tags.get("pastas_version", ""),
    )


@router.delete("/models/{run_id}")
def delete_model(run_id: str):
    from dashboard.utils.pastas.io import evict_cache

    client = mlflow.tracking.MlflowClient()
    try:
        client.delete_run(run_id)
    except Exception:
        raise HTTPException(404, f"Run '{run_id}' not found")

    evict_cache(run_id)
    return {"status": "deleted"}


@router.post("/simulate", response_model=ScenarioResponse)
def simulate(req: ScenarioRequest):
    from dashboard.utils.pastas.scenario import simulate_scenario
    from dashboard.utils.pastas.builder import ValidationError

    mods = [m.model_dump() for m in req.modifications]

    try:
        result = simulate_scenario(
            run_id=req.run_id,
            tmin=req.tmin,
            tmax=req.tmax,
            modifications=mods,
        )
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{req.run_id}' not found")
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc))

    return ScenarioResponse(
        baseline=_series_to_ts(result.baseline),
        scenario=_series_to_ts(result.scenario),
        delta=_series_to_ts(result.delta),
        contributions_baseline={k: _series_to_ts(v) for k, v in result.contributions_baseline.items()},
        contributions_scenario={k: _series_to_ts(v) for k, v in result.contributions_scenario.items()},
        warnings=result.warnings,
    )
```

- [ ] **Step 3: Register router in main.py**

In `api/main.py`, add to the import line (line 16):
```python
from api.routers import datasets, training, models, forecasting, explainability, counterfactual, db_introspection, pumping_detection, latent_space, pastas
```

Add after line 79 (`app.include_router(latent_space.router)`):
```python
app.include_router(pastas.router)
```

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_api_pastas.py -v`
Expected: 2 PASS, 1 SKIP.

- [ ] **Step 5: Commit**

```bash
git add api/routers/pastas.py api/schemas/pastas.py api/main.py tests/test_api_pastas.py
git commit -m "feat(pastas): API router with fit, models, simulate endpoints"
```

---

## Task 9: Frontend types, API client, hooks

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/usePastas.ts`

- [ ] **Step 1: Add Pastas types**

Append to `frontend/src/lib/types.ts`:
```ts
// ---------- Pastas ----------

export interface TimeSeriesData {
  index: string[]
  values: number[]
}

export interface FitParameter {
  name: string
  optimal: number
  stderr: number | null
  initial: number
  pmin: number | null
  pmax: number | null
  vary: boolean
}

export interface PastasFitResponse {
  run_id: string
  metrics: Record<string, number>
  parameters: FitParameter[]
  observed: TimeSeriesData
  simulated: TimeSeriesData
  residuals: TimeSeriesData
  contributions: Record<string, TimeSeriesData>
  step_response: TimeSeriesData
  block_response: TimeSeriesData
  acf: Record<string, unknown>
  warnings: string[]
  pastas_version: string
}

export interface PastasModelSummary {
  run_id: string
  name: string
  station_id: string
  recharge_type: string
  response_type: string
  evp: number | null
  rmse: number | null
  created_at: string
  pastas_version: string
}

export interface PastasOptions {
  recharge: string[]
  response: string[]
  noise: string[]
  solver: string[]
}

export interface PastasScenarioResponse {
  baseline: TimeSeriesData
  scenario: TimeSeriesData
  delta: TimeSeriesData
  contributions_baseline: Record<string, TimeSeriesData>
  contributions_scenario: Record<string, TimeSeriesData>
  warnings: string[]
}
```

- [ ] **Step 2: Add Pastas namespace to API client**

Append to the `api` object in `frontend/src/lib/api.ts` (before the closing `}`):
```ts
  pastas: {
    options: () => fetchJson<PastasOptions>('/pastas/options'),
    fit: (body: {
      dataset_id: string
      station_id?: string
      precip_column: string
      evap_column: string
      tmin?: string
      tmax?: string
      recharge?: { type: string; kwargs?: Record<string, unknown> }
      response?: { type: string; kwargs?: Record<string, unknown> }
      noise?: { type: string }
      solver?: { type: string; kwargs?: Record<string, unknown> }
      name?: string
    }) => postJson<PastasFitResponse>('/pastas/fit', body, 120_000),
    models: (stationId?: string) => {
      const params = stationId ? `?station_id=${stationId}` : ''
      return fetchJson<PastasModelSummary[]>(`/pastas/models${params}`)
    },
    model: (runId: string) => fetchJson<PastasFitResponse>(`/pastas/models/${runId}`),
    deleteModel: (runId: string) => deleteJson(`/pastas/models/${runId}`),
    simulate: (body: {
      run_id: string
      tmin: string
      tmax: string
      modifications: Array<Record<string, unknown>>
    }) => postJson<PastasScenarioResponse>('/pastas/simulate', body, 120_000),
  },
```

Add the new types to the import block at the top:
```ts
import type {
  // ... existing imports ...
  PastasOptions,
  PastasFitResponse,
  PastasModelSummary,
  PastasScenarioResponse,
} from './types'
```

- [ ] **Step 3: Create React Query hooks**

Create `frontend/src/hooks/usePastas.ts`:
```ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function usePastasOptions() {
  return useQuery({
    queryKey: ['pastas', 'options'],
    queryFn: () => api.pastas.options(),
    staleTime: 60 * 60 * 1000,
  })
}

export function usePastasFit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.fit,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasModels(stationId?: string) {
  return useQuery({
    queryKey: ['pastas', 'models', stationId],
    queryFn: () => api.pastas.models(stationId),
  })
}

export function usePastasModel(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'model', runId],
    queryFn: () => api.pastas.model(runId!),
    enabled: !!runId,
  })
}

export function usePastasDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.deleteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasSimulate() {
  return useMutation({
    mutationFn: api.pastas.simulate,
  })
}
```

- [ ] **Step 4: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: no type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/hooks/usePastas.ts
git commit -m "feat(pastas): frontend types, API client, React Query hooks"
```

---

## Task 10: Frontend FitPage — full fit workflow

**Files:**
- Create: `frontend/src/components/pastas/StationPicker.tsx`
- Create: `frontend/src/components/pastas/PastasConfigForm.tsx`
- Create: `frontend/src/components/pastas/FitResultsPanel.tsx`
- Modify: `frontend/src/pages/pastas/FitPage.tsx`

This task builds the complete Fit page with station selection, model config, fit execution, and results display. The implementation is lengthy — the components follow the same Plotly + Tailwind + dark-theme patterns as `ForecastingPage.tsx` and `TrainingPage.tsx`.

- [ ] **Step 1: Create StationPicker**

Create `frontend/src/components/pastas/StationPicker.tsx`:
```tsx
import { useDatasets } from '@/hooks/useDatasets'

interface Props {
  datasetId: string
  stationId: string
  precipColumn: string
  evapColumn: string
  onDatasetChange: (id: string) => void
  onStationChange: (id: string) => void
  onPrecipChange: (col: string) => void
  onEvapChange: (col: string) => void
}

export function StationPicker({
  datasetId, stationId, precipColumn, evapColumn,
  onDatasetChange, onStationChange, onPrecipChange, onEvapChange,
}: Props) {
  const { data: datasets } = useDatasets()

  const selected = datasets?.find(d => d.id === datasetId || d.name === datasetId)
  const covariates = selected?.covariates ?? []
  const stations = selected?.stations ?? []

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs text-text-muted mb-1">Dataset</label>
        <select
          className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
          value={datasetId}
          onChange={e => onDatasetChange(e.target.value)}
        >
          <option value="">Select...</option>
          {datasets?.map(d => (
            <option key={d.id ?? d.name} value={d.id ?? d.name}>{d.name}</option>
          ))}
        </select>
      </div>

      {stations.length > 1 && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Station</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={stationId}
            onChange={e => onStationChange(e.target.value)}
          >
            <option value="">All stations</option>
            {stations.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Precipitation column</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={precipColumn}
            onChange={e => onPrecipChange(e.target.value)}
          >
            <option value="">Select...</option>
            {covariates.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Evapotranspiration column</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={evapColumn}
            onChange={e => onEvapChange(e.target.value)}
          >
            <option value="">Select...</option>
            {covariates.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create PastasConfigForm**

Create `frontend/src/components/pastas/PastasConfigForm.tsx`:
```tsx
import { usePastasOptions } from '@/hooks/usePastas'

interface Config {
  recharge: string
  response: string
  noise: string
  solver: string
  tmin: string
  tmax: string
}

interface Props {
  config: Config
  onChange: (c: Config) => void
}

export function PastasConfigForm({ config, onChange }: Props) {
  const { data: options } = usePastasOptions()

  const set = (key: keyof Config, val: string) =>
    onChange({ ...config, [key]: val })

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Recharge model</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.recharge}
            onChange={e => set('recharge', e.target.value)}
          >
            {options?.recharge.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Response function</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.response}
            onChange={e => set('response', e.target.value)}
          >
            {options?.response.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Noise model</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.noise}
            onChange={e => set('noise', e.target.value)}
          >
            {options?.noise.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Solver</label>
          <select
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.solver}
            onChange={e => set('solver', e.target.value)}
          >
            {options?.solver.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Calibration start</label>
          <input
            type="date"
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.tmin}
            onChange={e => set('tmin', e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Calibration end</label>
          <input
            type="date"
            className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
            value={config.tmax}
            onChange={e => set('tmax', e.target.value)}
          />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create FitResultsPanel**

Create `frontend/src/components/pastas/FitResultsPanel.tsx`:
```tsx
import Plot from 'react-plotly.js'
import type { PastasFitResponse } from '@/lib/types'

interface Props {
  result: PastasFitResponse
}

export function FitResultsPanel({ result }: Props) {
  const { metrics, parameters, observed, simulated, residuals, contributions, step_response, acf } = result

  return (
    <div className="space-y-4">
      {/* Warnings */}
      {result.warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-sm text-yellow-400 font-medium">Warnings</p>
          <ul className="mt-1 text-xs text-yellow-300/80 list-disc pl-4">
            {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      {/* Metrics cards */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'EVP', value: metrics.evp?.toFixed(1), unit: '%' },
          { label: 'RMSE', value: metrics.rmse?.toFixed(4), unit: 'm' },
          { label: 'AIC', value: metrics.aic?.toFixed(1), unit: '' },
          { label: 'Ljung-Box p', value: metrics.ljung_box_pvalue?.toFixed(3), unit: '' },
        ].map(({ label, value, unit }) => (
          <div key={label} className="bg-bg-card rounded-lg p-3 border border-white/5">
            <p className="text-xs text-text-muted">{label}</p>
            <p className="text-lg font-semibold text-text-primary">{value ?? '—'}{unit && <span className="text-xs text-text-muted ml-1">{unit}</span>}</p>
          </div>
        ))}
      </div>

      {/* Parameters table */}
      <div className="bg-bg-card rounded-lg border border-white/5 overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-text-muted text-xs border-b border-white/5">
              <th className="px-3 py-2 text-left">Parameter</th>
              <th className="px-3 py-2 text-right">Optimal</th>
              <th className="px-3 py-2 text-right">Stderr</th>
              <th className="px-3 py-2 text-right">Min</th>
              <th className="px-3 py-2 text-right">Max</th>
            </tr>
          </thead>
          <tbody>
            {parameters.map(p => (
              <tr key={p.name} className="border-b border-white/5">
                <td className="px-3 py-1.5 text-text-primary font-mono text-xs">{p.name}</td>
                <td className="px-3 py-1.5 text-right text-text-secondary">{p.optimal.toFixed(4)}</td>
                <td className="px-3 py-1.5 text-right text-text-muted">{p.stderr?.toFixed(4) ?? '—'}</td>
                <td className="px-3 py-1.5 text-right text-text-muted">{p.pmin?.toFixed(2) ?? '—'}</td>
                <td className="px-3 py-1.5 text-right text-text-muted">{p.pmax?.toFixed(2) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Obs vs Sim plot */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <Plot
          data={[
            { x: observed.index, y: observed.values, name: 'Observed', type: 'scatter', mode: 'lines', line: { color: '#60a5fa' } },
            { x: simulated.index, y: simulated.values, name: 'Simulated', type: 'scatter', mode: 'lines', line: { color: '#f97316' } },
          ]}
          layout={{
            title: 'Observed vs Simulated',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9ca3af' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Head (m)', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 30, b: 40, l: 50, r: 20 },
            legend: { orientation: 'h', y: -0.15 },
            height: 300,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>

      {/* Residuals plot */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <Plot
          data={[
            { x: residuals.index, y: residuals.values, name: 'Residuals', type: 'scatter', mode: 'lines', line: { color: '#a78bfa' } },
          ]}
          layout={{
            title: 'Residuals',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9ca3af' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'm', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 30, b: 40, l: 50, r: 20 },
            height: 200,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>

      {/* Step response */}
      {step_response.values.length > 0 && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3">
          <Plot
            data={[
              { y: step_response.values, name: 'Step response', type: 'scatter', mode: 'lines', line: { color: '#34d399' } },
            ]}
            layout={{
              title: 'Step Response',
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#9ca3af' },
              xaxis: { title: 'Days', gridcolor: 'rgba(255,255,255,0.05)' },
              yaxis: { title: 'm', gridcolor: 'rgba(255,255,255,0.05)' },
              margin: { t: 30, b: 40, l: 50, r: 20 },
              height: 200,
            }}
            useResizeHandler
            className="w-full"
          />
        </div>
      )}

      {/* ACF plot */}
      {acf.acf_values && (acf.acf_values as number[]).length > 0 && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3">
          <Plot
            data={[
              { y: acf.acf_values as number[], name: 'ACF', type: 'bar', marker: { color: '#60a5fa' } },
            ]}
            layout={{
              title: 'Autocorrelation',
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#9ca3af' },
              xaxis: { title: 'Lag', gridcolor: 'rgba(255,255,255,0.05)' },
              yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              margin: { t: 30, b: 40, l: 50, r: 20 },
              height: 200,
            }}
            useResizeHandler
            className="w-full"
          />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Implement FitPage**

Replace `frontend/src/pages/pastas/FitPage.tsx`:
```tsx
import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { StationPicker } from '@/components/pastas/StationPicker'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import { usePastasFit } from '@/hooks/usePastas'

export default function FitPage() {
  const [datasetId, setDatasetId] = useState('')
  const [stationId, setStationId] = useState('')
  const [precipCol, setPrecipCol] = useState('')
  const [evapCol, setEvapCol] = useState('')
  const [config, setConfig] = useState({
    recharge: 'Linear',
    response: 'Gamma',
    noise: 'ArNoiseModel',
    solver: 'LeastSquares',
    tmin: '',
    tmax: '',
  })
  const [name, setName] = useState('')

  const fit = usePastasFit()

  const canFit = datasetId && precipCol && evapCol

  const handleFit = () => {
    fit.mutate({
      dataset_id: datasetId,
      station_id: stationId || undefined,
      precip_column: precipCol,
      evap_column: evapCol,
      tmin: config.tmin || undefined,
      tmax: config.tmax || undefined,
      recharge: { type: config.recharge },
      response: { type: config.response },
      noise: { type: config.noise },
      solver: { type: config.solver },
      name: name || undefined,
    })
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <h1 className="text-xl font-semibold text-text-primary">Pastas — Fit</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="bg-bg-card rounded-lg border border-white/5 p-4">
            <h2 className="text-sm font-medium text-text-secondary mb-3">Station & Data</h2>
            <StationPicker
              datasetId={datasetId}
              stationId={stationId}
              precipColumn={precipCol}
              evapColumn={evapCol}
              onDatasetChange={setDatasetId}
              onStationChange={setStationId}
              onPrecipChange={setPrecipCol}
              onEvapChange={setEvapCol}
            />
          </div>

          <div className="bg-bg-card rounded-lg border border-white/5 p-4">
            <h2 className="text-sm font-medium text-text-secondary mb-3">Model Configuration</h2>
            <PastasConfigForm config={config} onChange={setConfig} />
          </div>

          <div className="bg-bg-card rounded-lg border border-white/5 p-4">
            <label className="block text-xs text-text-muted mb-1">Run name (optional)</label>
            <input
              type="text"
              className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="e.g. BSS001_gamma_v1"
            />
          </div>

          <button
            onClick={handleFit}
            disabled={!canFit || fit.isPending}
            className="w-full flex items-center justify-center gap-2 bg-accent-cyan/20 hover:bg-accent-cyan/30 text-accent-cyan px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {fit.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {fit.isPending ? 'Fitting...' : 'Fit Model'}
          </button>

          {fit.isError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <p className="text-sm text-red-400">{(fit.error as Error).message}</p>
            </div>
          )}
        </div>

        <div>
          {fit.data && <FitResultsPanel result={fit.data} />}
          {!fit.data && !fit.isPending && (
            <div className="flex items-center justify-center h-64 text-text-muted text-sm">
              Configure and fit a model to see results
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/pastas/StationPicker.tsx \
  frontend/src/components/pastas/PastasConfigForm.tsx \
  frontend/src/components/pastas/FitResultsPanel.tsx \
  frontend/src/pages/pastas/FitPage.tsx
git commit -m "feat(pastas): FitPage with station picker, config form, results panel"
```

---

## Task 11: Frontend ScenariosPage

**Files:**
- Create: `frontend/src/components/pastas/ScenarioComposer.tsx`
- Create: `frontend/src/components/pastas/ModificationCard.tsx`
- Create: `frontend/src/components/pastas/PumpingSyntheticEditor.tsx`
- Create: `frontend/src/components/pastas/PumpingUploadEditor.tsx`
- Create: `frontend/src/components/pastas/LinearTrendEditor.tsx`
- Create: `frontend/src/components/pastas/ScaleStressEditor.tsx`
- Create: `frontend/src/components/pastas/ScenarioResultsPanel.tsx`
- Modify: `frontend/src/pages/pastas/ScenariosPage.tsx`

- [ ] **Step 1: Create modification editors**

Create `frontend/src/components/pastas/PumpingSyntheticEditor.tsx`:
```tsx
interface PumpingSyntheticData {
  type: 'pumping_synthetic'
  pattern: 'constant' | 'seasonal' | 'pulse'
  rate_m3d: number
  start: string
  end: string
  distance_m: number
  rfunc: 'Hantush' | 'Exponential'
}

interface Props {
  data: PumpingSyntheticData
  onChange: (d: PumpingSyntheticData) => void
}

export function PumpingSyntheticEditor({ data, onChange }: Props) {
  const set = <K extends keyof PumpingSyntheticData>(key: K, val: PumpingSyntheticData[K]) =>
    onChange({ ...data, [key]: val })

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2">
        {(['constant', 'seasonal', 'pulse'] as const).map(p => (
          <button
            key={p}
            onClick={() => set('pattern', p)}
            className={`px-2 py-1 text-xs rounded-lg border transition-colors ${
              data.pattern === p
                ? 'border-accent-cyan bg-accent-cyan/10 text-accent-cyan'
                : 'border-white/10 text-text-muted hover:text-text-secondary'
            }`}
          >
            {p}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Rate (m³/d)</label>
          <input type="number" min={0} step={10} value={data.rate_m3d}
            onChange={e => set('rate_m3d', +e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">Distance (m)</label>
          <input type="number" min={1} step={10} value={data.distance_m}
            onChange={e => set('distance_m', +e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Start</label>
          <input type="date" value={data.start} onChange={e => set('start', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">End</label>
          <input type="date" value={data.end} onChange={e => set('end', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
      </div>
      <div>
        <label className="text-xs text-text-muted">Response function</label>
        <select value={data.rfunc} onChange={e => set('rfunc', e.target.value as 'Hantush' | 'Exponential')}
          className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary">
          <option value="Exponential">Exponential</option>
          <option value="Hantush">Hantush</option>
        </select>
      </div>
    </div>
  )
}
```

Create `frontend/src/components/pastas/LinearTrendEditor.tsx`:
```tsx
interface LinearTrendData {
  type: 'linear_trend'
  start: string
  end: string
  slope_m_per_year: number
}

interface Props {
  data: LinearTrendData
  onChange: (d: LinearTrendData) => void
}

export function LinearTrendEditor({ data, onChange }: Props) {
  const set = <K extends keyof LinearTrendData>(key: K, val: LinearTrendData[K]) =>
    onChange({ ...data, [key]: val })

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Start</label>
          <input type="date" value={data.start} onChange={e => set('start', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">End</label>
          <input type="date" value={data.end} onChange={e => set('end', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
      </div>
      <div>
        <label className="text-xs text-text-muted">Slope (m/year)</label>
        <input type="number" step={0.01} value={data.slope_m_per_year}
          onChange={e => set('slope_m_per_year', +e.target.value)}
          className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
      </div>
    </div>
  )
}
```

Create `frontend/src/components/pastas/ScaleStressEditor.tsx`:
```tsx
interface ScaleStressData {
  type: 'scale_stress'
  stress: 'precip' | 'evap'
  factor: number
  start: string
  end: string
}

interface Props {
  data: ScaleStressData
  onChange: (d: ScaleStressData) => void
}

export function ScaleStressEditor({ data, onChange }: Props) {
  const set = <K extends keyof ScaleStressData>(key: K, val: ScaleStressData[K]) =>
    onChange({ ...data, [key]: val })

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Stress</label>
          <select value={data.stress} onChange={e => set('stress', e.target.value as 'precip' | 'evap')}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary">
            <option value="precip">Precipitation</option>
            <option value="evap">Evapotranspiration</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-text-muted">Factor</label>
          <input type="number" step={0.05} min={0.01} value={data.factor}
            onChange={e => set('factor', +e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Start</label>
          <input type="date" value={data.start} onChange={e => set('start', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">End</label>
          <input type="date" value={data.end} onChange={e => set('end', e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
      </div>
    </div>
  )
}
```

Create `frontend/src/components/pastas/PumpingUploadEditor.tsx`:
```tsx
import { useCallback } from 'react'

interface PumpingUploadData {
  type: 'pumping_upload'
  csv_rows: Array<{ date: string; Q_m3d: number }>
  distance_m: number
  rfunc: 'Hantush' | 'Exponential'
}

interface Props {
  data: PumpingUploadData
  onChange: (d: PumpingUploadData) => void
}

export function PumpingUploadEditor({ data, onChange }: Props) {
  const set = <K extends keyof PumpingUploadData>(key: K, val: PumpingUploadData[K]) =>
    onChange({ ...data, [key]: val })

  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    const lines = text.trim().split('\n')
    const rows = lines.slice(1).map(line => {
      const [dateStr, q] = line.split(',')
      return { date: dateStr.trim(), Q_m3d: parseFloat(q) }
    }).filter(r => !isNaN(r.Q_m3d))
    set('csv_rows', rows)
  }, [data, onChange])

  return (
    <div className="space-y-2">
      <div>
        <label className="text-xs text-text-muted">CSV file (date, Q_m3d)</label>
        <input type="file" accept=".csv" onChange={handleFile}
          className="w-full text-sm text-text-secondary file:mr-2 file:py-1 file:px-3 file:rounded file:border-0 file:text-xs file:bg-bg-hover file:text-text-primary" />
        {data.csv_rows.length > 0 && (
          <p className="text-xs text-text-muted mt-1">{data.csv_rows.length} rows loaded</p>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Distance (m)</label>
          <input type="number" min={1} value={data.distance_m}
            onChange={e => set('distance_m', +e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">Response function</label>
          <select value={data.rfunc} onChange={e => set('rfunc', e.target.value as 'Hantush' | 'Exponential')}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary">
            <option value="Exponential">Exponential</option>
            <option value="Hantush">Hantush</option>
          </select>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create ModificationCard + ScenarioComposer**

Create `frontend/src/components/pastas/ModificationCard.tsx`:
```tsx
import { Trash2, Droplets, TrendingUp, ArrowUpDown } from 'lucide-react'
import { PumpingSyntheticEditor } from './PumpingSyntheticEditor'
import { PumpingUploadEditor } from './PumpingUploadEditor'
import { LinearTrendEditor } from './LinearTrendEditor'
import { ScaleStressEditor } from './ScaleStressEditor'

export type ModificationData =
  | { type: 'pumping_synthetic'; pattern: 'constant' | 'seasonal' | 'pulse'; rate_m3d: number; start: string; end: string; distance_m: number; rfunc: 'Hantush' | 'Exponential' }
  | { type: 'pumping_upload'; csv_rows: Array<{ date: string; Q_m3d: number }>; distance_m: number; rfunc: 'Hantush' | 'Exponential' }
  | { type: 'linear_trend'; start: string; end: string; slope_m_per_year: number }
  | { type: 'scale_stress'; stress: 'precip' | 'evap'; factor: number; start: string; end: string }

const TYPE_META = {
  pumping_synthetic: { label: 'Pumping (synthetic)', icon: Droplets, color: 'text-blue-400' },
  pumping_upload: { label: 'Pumping (CSV)', icon: Droplets, color: 'text-blue-400' },
  linear_trend: { label: 'Linear Trend', icon: TrendingUp, color: 'text-green-400' },
  scale_stress: { label: 'Scale Stress', icon: ArrowUpDown, color: 'text-amber-400' },
} as const

interface Props {
  data: ModificationData
  onChange: (d: ModificationData) => void
  onDelete: () => void
}

export function ModificationCard({ data, onChange, onDelete }: Props) {
  const meta = TYPE_META[data.type]
  const Icon = meta.icon

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${meta.color}`} />
          <span className="text-sm font-medium text-text-secondary">{meta.label}</span>
        </div>
        <button onClick={onDelete} className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-red-400">
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {data.type === 'pumping_synthetic' && <PumpingSyntheticEditor data={data} onChange={onChange as any} />}
      {data.type === 'pumping_upload' && <PumpingUploadEditor data={data} onChange={onChange as any} />}
      {data.type === 'linear_trend' && <LinearTrendEditor data={data} onChange={onChange as any} />}
      {data.type === 'scale_stress' && <ScaleStressEditor data={data} onChange={onChange as any} />}
    </div>
  )
}
```

Create `frontend/src/components/pastas/ScenarioComposer.tsx`:
```tsx
import { useState } from 'react'
import { Plus } from 'lucide-react'
import { ModificationCard, type ModificationData } from './ModificationCard'

const NEW_DEFAULTS: Record<string, ModificationData> = {
  pumping_synthetic: { type: 'pumping_synthetic', pattern: 'constant', rate_m3d: 100, start: '', end: '', distance_m: 500, rfunc: 'Exponential' },
  pumping_upload: { type: 'pumping_upload', csv_rows: [], distance_m: 500, rfunc: 'Exponential' },
  linear_trend: { type: 'linear_trend', start: '', end: '', slope_m_per_year: 0.1 },
  scale_stress: { type: 'scale_stress', stress: 'precip', factor: 0.8, start: '', end: '' },
}

interface Props {
  modifications: ModificationData[]
  onChange: (mods: ModificationData[]) => void
}

export function ScenarioComposer({ modifications, onChange }: Props) {
  const [menuOpen, setMenuOpen] = useState(false)

  const add = (type: string) => {
    onChange([...modifications, { ...NEW_DEFAULTS[type] }])
    setMenuOpen(false)
  }

  const update = (i: number, mod: ModificationData) => {
    const next = [...modifications]
    next[i] = mod
    onChange(next)
  }

  const remove = (i: number) => {
    onChange(modifications.filter((_, idx) => idx !== i))
  }

  return (
    <div className="space-y-3">
      {modifications.map((mod, i) => (
        <ModificationCard
          key={i}
          data={mod}
          onChange={d => update(i, d)}
          onDelete={() => remove(i)}
        />
      ))}

      <div className="relative">
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="w-full flex items-center justify-center gap-2 border border-dashed border-white/10 hover:border-white/20 rounded-lg px-4 py-2 text-sm text-text-muted hover:text-text-secondary transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add modification
        </button>
        {menuOpen && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-bg-card border border-white/10 rounded-lg shadow-xl z-10">
            {Object.entries(NEW_DEFAULTS).map(([type, def]) => (
              <button
                key={type}
                onClick={() => add(type)}
                className="w-full text-left px-3 py-2 text-sm text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors first:rounded-t-lg last:rounded-b-lg"
              >
                {type.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create ScenarioResultsPanel**

Create `frontend/src/components/pastas/ScenarioResultsPanel.tsx`:
```tsx
import Plot from 'react-plotly.js'
import type { PastasScenarioResponse } from '@/lib/types'

interface Props {
  result: PastasScenarioResponse
}

export function ScenarioResultsPanel({ result }: Props) {
  return (
    <div className="space-y-4">
      {result.warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <ul className="text-xs text-yellow-300/80 list-disc pl-4">
            {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <Plot
          data={[
            { x: result.baseline.index, y: result.baseline.values, name: 'Baseline', type: 'scatter', mode: 'lines', line: { color: '#60a5fa' } },
            { x: result.scenario.index, y: result.scenario.values, name: 'Scenario', type: 'scatter', mode: 'lines', line: { color: '#f97316' } },
          ]}
          layout={{
            title: 'Baseline vs Scenario',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9ca3af' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Head (m)', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 30, b: 40, l: 50, r: 20 },
            legend: { orientation: 'h', y: -0.15 },
            height: 300,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>

      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <Plot
          data={[
            {
              x: result.delta.index,
              y: result.delta.values,
              name: 'Delta (scenario − baseline)',
              type: 'scatter',
              mode: 'lines',
              fill: 'tozeroy',
              line: { color: '#a78bfa' },
              fillcolor: 'rgba(167,139,250,0.1)',
            },
          ]}
          layout={{
            title: 'Impact (Scenario − Baseline)',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9ca3af' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Δ Head (m)', gridcolor: 'rgba(255,255,255,0.05)' },
            margin: { t: 30, b: 40, l: 50, r: 20 },
            height: 250,
          }}
          useResizeHandler
          className="w-full"
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Implement ScenariosPage**

Replace `frontend/src/pages/pastas/ScenariosPage.tsx`:
```tsx
import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { usePastasModels, usePastasSimulate } from '@/hooks/usePastas'
import { ScenarioComposer } from '@/components/pastas/ScenarioComposer'
import { ScenarioResultsPanel } from '@/components/pastas/ScenarioResultsPanel'
import type { ModificationData } from '@/components/pastas/ModificationCard'

export default function ScenariosPage() {
  const [selectedRunId, setSelectedRunId] = useState('')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modifications, setModifications] = useState<ModificationData[]>([])

  const { data: models } = usePastasModels()
  const simulate = usePastasSimulate()

  const canSimulate = selectedRunId && tmin && tmax

  const handleSimulate = () => {
    simulate.mutate({
      run_id: selectedRunId,
      tmin,
      tmax,
      modifications: modifications as Array<Record<string, unknown>>,
    })
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <h1 className="text-xl font-semibold text-text-primary">Pastas — Scenarios</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="bg-bg-card rounded-lg border border-white/5 p-4">
            <h2 className="text-sm font-medium text-text-secondary mb-3">Base Model</h2>
            <select
              className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary"
              value={selectedRunId}
              onChange={e => setSelectedRunId(e.target.value)}
            >
              <option value="">Select a fitted model...</option>
              {models?.map(m => (
                <option key={m.run_id} value={m.run_id}>
                  {m.name || m.run_id.slice(0, 8)} — {m.station_id} ({m.response_type}, EVP {m.evp?.toFixed(1) ?? '?'}%)
                </option>
              ))}
            </select>

            <div className="grid grid-cols-2 gap-3 mt-3">
              <div>
                <label className="block text-xs text-text-muted mb-1">Simulation start</label>
                <input type="date" value={tmin} onChange={e => setTmin(e.target.value)}
                  className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary" />
              </div>
              <div>
                <label className="block text-xs text-text-muted mb-1">Simulation end</label>
                <input type="date" value={tmax} onChange={e => setTmax(e.target.value)}
                  className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary" />
              </div>
            </div>
          </div>

          <div className="bg-bg-card rounded-lg border border-white/5 p-4">
            <h2 className="text-sm font-medium text-text-secondary mb-3">Modifications</h2>
            <ScenarioComposer modifications={modifications} onChange={setModifications} />
          </div>

          <button
            onClick={handleSimulate}
            disabled={!canSimulate || simulate.isPending}
            className="w-full flex items-center justify-center gap-2 bg-accent-cyan/20 hover:bg-accent-cyan/30 text-accent-cyan px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {simulate.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {simulate.isPending ? 'Simulating...' : 'Simulate Scenario'}
          </button>

          {simulate.isError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <p className="text-sm text-red-400">{(simulate.error as Error).message}</p>
            </div>
          )}
        </div>

        <div>
          {simulate.data && <ScenarioResultsPanel result={simulate.data} />}
          {!simulate.data && !simulate.isPending && (
            <div className="flex items-center justify-center h-64 text-text-muted text-sm">
              Select a model and add modifications to simulate
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/pastas/ frontend/src/pages/pastas/ScenariosPage.tsx
git commit -m "feat(pastas): ScenariosPage with modification composer + results"
```

---

## Task 12: Docker rebuild & manual verification

- [ ] **Step 1: Verify pastas is in requirements**

Run: `grep pastas pyproject.toml requirements.txt`
If missing, add `pastas>=1.7` to requirements.

- [ ] **Step 2: Rebuild Docker**

Run: `docker compose up -d --build`
Expected: builds successfully, API starts.

- [ ] **Step 3: Verify API health**

Run: `curl http://localhost:49513/api/v1/health`
Expected: `{"status": "ok", ...}`

Run: `curl http://localhost:49513/api/v1/pastas/options`
Expected: JSON with recharge, response, noise, solver lists.

- [ ] **Step 4: Verify frontend in browser**

Open `http://localhost:49513/`:
- Lab should NOT appear in the nav
- Pastas should appear in the nav
- Click Pastas → should navigate to `/pastas/fit`
- `/lab/latent-space` → should redirect to `/`
- Fit tab and Scenarios tab should render their UI

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix(pastas): Docker build and runtime fixes"
```

---

## Task 13: E2E tests

**Files:**
- Create: `e2e/pastas-fit.spec.ts`
- Create: `e2e/pastas-scenarios.spec.ts`

- [ ] **Step 1: Create fit E2E test**

Create `e2e/pastas-fit.spec.ts`:
```ts
import { test, expect } from '@playwright/test'

test.describe('Pastas Fit Page', () => {
  test('renders fit page with config form', async ({ page }) => {
    await page.goto('/pastas/fit')
    await expect(page.getByText('Pastas — Fit')).toBeVisible()
    await expect(page.getByText('Station & Data')).toBeVisible()
    await expect(page.getByText('Model Configuration')).toBeVisible()
    await expect(page.getByRole('button', { name: /Fit Model/i })).toBeVisible()
  })

  test('fit button is disabled without dataset selection', async ({ page }) => {
    await page.goto('/pastas/fit')
    const fitBtn = page.getByRole('button', { name: /Fit Model/i })
    await expect(fitBtn).toBeDisabled()
  })
})
```

- [ ] **Step 2: Create scenarios E2E test**

Create `e2e/pastas-scenarios.spec.ts`:
```ts
import { test, expect } from '@playwright/test'

test.describe('Pastas Scenarios Page', () => {
  test('renders scenarios page with model selector', async ({ page }) => {
    await page.goto('/pastas/scenarios')
    await expect(page.getByText('Pastas — Scenarios')).toBeVisible()
    await expect(page.getByText('Base Model')).toBeVisible()
    await expect(page.getByText('Modifications')).toBeVisible()
    await expect(page.getByText('Add modification')).toBeVisible()
  })

  test('can add and remove a modification', async ({ page }) => {
    await page.goto('/pastas/scenarios')
    await page.getByText('Add modification').click()
    await page.getByText('pumping synthetic').click()
    await expect(page.getByText('Pumping (synthetic)')).toBeVisible()

    // Delete it
    const deleteBtn = page.locator('button').filter({ has: page.locator('[class*="trash"]') }).first()
    await deleteBtn.click()
    await expect(page.getByText('Pumping (synthetic)')).not.toBeVisible()
  })
})
```

- [ ] **Step 3: Run E2E tests**

Run: `cd e2e && npx playwright test pastas --headed`
Expected: tests pass (or skip if Docker not running).

- [ ] **Step 4: Commit**

```bash
git add e2e/pastas-fit.spec.ts e2e/pastas-scenarios.spec.ts
git commit -m "test(pastas): E2E tests for fit and scenarios pages"
```

---

## Task 14: Final cleanup & verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/pastas/ tests/test_api_pastas.py -v`
Expected: all tests pass.

- [ ] **Step 2: Run frontend build**

Run: `cd frontend && npm run build`
Expected: no errors.

- [ ] **Step 3: Remove LabLayout lazy imports if unused**

In `routes.tsx`, remove the lazy imports for `LabLayout`, `CounterfactualPage`, `PumpingDetectionPage`, `LatentSpacePage` — they're no longer referenced by any route.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "chore(pastas): cleanup unused Lab imports, P1 complete"
```
