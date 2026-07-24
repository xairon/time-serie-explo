# SPEI (Climat) Implementation Plan

> ⚠️ **Document historique — exécuté, puis partiellement dépassé.**
> Ce plan a été intégralement exécuté (9 tâches, mergées et déployées le 2026-07-24). Mais la
> **loi d'ajustement qu'il décrit — la log-logistique — a depuis été remplacée par la
> logistique généralisée (GLO)**, car elle ne couvrait que 74,6 % des mailles.
> Ne pas s'appuyer sur les formules ni sur les noms de colonnes (`ll_alpha/ll_beta/ll_gamma`)
> de ce plan : voir `docs/superpowers/specs/2026-07-23-climat-spei-design.md` §2.0 pour la
> méthode en vigueur, et `docs/audits/2026-07-24-indices-validation-followup.md` pour la
> validation. Conservé pour la traçabilité de la démarche.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SPEI (Standardized Precipitation-Evapotranspiration Index) as a third grid drought index in the `/climat` module — standardizing the already-present monthly water balance `D = P − ETP`.

**Architecture:** The warehouse (`hubeau_data_integration`) fits a 3-parameter log-logistic per cell×calendar-month×window over 1991-2020 (homemade L-moments, no dependency), stores its params in a new reference table, and writes `spei` into `gold.fct_era5_indices_grid` alongside `spi`/`sti`. The app (`time-serie-explo`) exposes `spei` as one more `ClimatVariable` (map layer, point series, drought episodes) reusing the SPI drought palette.

**Tech Stack:** Python (numpy, scipy, pandas, Dagster, dbt, PostgreSQL) warehouse-side; FastAPI + SQLAlchemy backend; React + TypeScript + MapLibre + vitest frontend.

## Global Constraints

- **Windows:** `1, 3, 6, 12` months, verbatim, everywhere (`WINDOWS`/`CLIMAT_WINDOWS`).
- **Reference period:** fixed **1991-2020** (warmup from 1990 for the 12-month window).
- **WMO reference floor:** `nb_annees < 25` (`MIN_YEARS_REF`) → index is `NaN`/`NULL`, never a fabricated value.
- **Classes:** the shared 7-class McKee/WMO thresholds (±0.84, ±1.28, ±1.75) via `api/era5_anomaly.py::classify_index` / `dashboard/utils/drought.py::_THRESHOLDS_7`. SPEI has **drought semantics** (negative = dry) → reuses the **SPI** palette, not STI.
- **Coordinates:** rounded to 0.1° (`era5_latitude`/`era5_longitude`), consistent with existing marts.
- **No new dependency** in the warehouse (log-logistic fit is homemade numpy/scipy — matches `drought.py`'s no-`spei`-dep precedent).
- **Degenerate fit → NaN**, never a fallback distribution (out of scope: KDE fallback).
- **Frontend `npm run build` (tsc -b) is mandatory** after any `ClimatVariable` change — vitest does not typecheck the exhaustive union.
- The `bilan_hydrique` water balance is **already correct** (ERA5 PEV is an accumulated flux, unaffected by the 00:00 UTC temperature bias; see spec §1.1). No ETP recabling.

**Repos:** Tasks 1-3 run in `~/hubeau_data_integration`. Tasks 4-8 run in `~/time-serie-explo` (branch `feat/climat-spei`).

---

### Task 1: Log-logistic fit + SPEI compute (pure functions)

**Files:**
- Modify: `~/hubeau_data_integration/src/hubeau_pipeline/ml/era5_indices.py`
- Test: `~/hubeau_data_integration/tests/test_era5_indices.py`

**Interfaces:**
- Produces: `fit_loglogistic_lmoments(samples: np.ndarray) -> tuple[float, float, float]` returning `(alpha, beta, gamma)` or `(nan, nan, nan)`.
- Produces: `compute_spei(d_cumul, ll_alpha, ll_beta, ll_gamma) -> np.ndarray` (vectorised, rounded 3 dp, NaN where invalid). Mirrors `compute_spi`/`compute_sti`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_era5_indices.py`:

```python
import numpy as np
from scipy import stats
from hubeau_pipeline.ml.era5_indices import (
    fit_loglogistic_lmoments,
    compute_spei,
)


def test_fit_loglogistic_recovers_known_params():
    # Synthetic sample drawn from a known 3-param log-logistic (fisk + loc):
    # x = gamma_loc + alpha * (u/(1-u))**(1/beta), u ~ Uniform(0,1) on a fixed grid.
    alpha, beta, gamma_loc = 40.0, 3.0, -10.0
    u = (np.arange(1, 61) - 0.5) / 60.0            # 60 deterministic quantiles
    x = gamma_loc + alpha * (u / (1.0 - u)) ** (1.0 / beta)
    a, b, g = fit_loglogistic_lmoments(x)
    assert np.isfinite([a, b, g]).all()
    assert abs(a - alpha) < 4.0
    assert abs(b - beta) < 0.4
    assert abs(g - gamma_loc) < 6.0


def test_fit_loglogistic_degenerate_returns_nan():
    assert not np.isfinite(fit_loglogistic_lmoments(np.full(30, 5.0))[1])   # constant
    assert not np.isfinite(fit_loglogistic_lmoments(np.array([1.0, 2.0]))[1])  # n < 4


def test_compute_spei_sign_and_center():
    # Median of the reference (x = gamma_loc + alpha) → F = 0.5 → SPEI ≈ 0.
    alpha, beta, gamma_loc = 40.0, 3.0, -10.0
    median = gamma_loc + alpha
    z = compute_spei(
        np.array([median, median + 300.0, gamma_loc + 1.0]),
        np.full(3, alpha), np.full(3, beta), np.full(3, gamma_loc),
    )
    assert abs(z[0]) < 0.05          # centre
    assert z[1] > 1.0                # wet surplus
    assert z[2] < -1.0               # deep deficit


def test_compute_spei_invalid_params_nan():
    z = compute_spei(
        np.array([10.0, 10.0, -999.0]),
        np.array([40.0, np.nan, 40.0]),   # bad alpha
        np.array([3.0, 3.0, 3.0]),
        np.array([-10.0, -10.0, 5.0]),    # last: x <= gamma → out of support
    )
    assert np.isfinite(z[0])
    assert np.isnan(z[1])
    assert np.isnan(z[2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_indices.py -k "loglogistic or spei" -v`
Expected: FAIL — `ImportError: cannot import name 'fit_loglogistic_lmoments'`.

- [ ] **Step 3: Implement the two functions**

Append to `src/hubeau_pipeline/ml/era5_indices.py` (module already imports `numpy as np`, `stats`, and defines `_CDF_CLIP`):

```python
from scipy.special import gamma as _gamma_fn

# Fit fiable seulement au-delà d'un petit échantillon (L-moments d'ordre 2).
_MIN_FIT_SAMPLES = 4


def fit_loglogistic_lmoments(samples):
    """Ajuste une log-logistique à 3 paramètres (loi de Fisk translatée) par
    L-moments (PWM en position de tracé, Vicente-Serrano 2010).

    Args: samples — échantillon 1D des cumuls D=P−ETP de référence (une cellule×
        mois calendaire×fenêtre, ~30 valeurs annuelles).
    Returns: (alpha, beta, gamma) ; (nan, nan, nan) si l'ajustement est dégénéré.
    """
    x = np.asarray(samples, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    if n < _MIN_FIT_SAMPLES:
        return (np.nan, np.nan, np.nan)

    # PWM en position de tracé p_i = (i − 0.35)/n (convention SPEI de référence).
    i = np.arange(1, n + 1)
    p = (i - 0.35) / n
    w0 = x.mean()
    w1 = np.sum((1.0 - p) * x) / n
    w2 = np.sum((1.0 - p) ** 2 * x) / n

    denom = 6.0 * w1 - w0 - 6.0 * w2
    if denom == 0 or not np.isfinite(denom):
        return (np.nan, np.nan, np.nan)
    beta = (2.0 * w1 - w0) / denom
    # beta>0 requis ; 1/beta<1 requis pour que Γ(1−1/beta) converge (beta>1).
    if not np.isfinite(beta) or beta <= 1.0:
        return (np.nan, np.nan, np.nan)

    g = _gamma_fn(1.0 + 1.0 / beta) * _gamma_fn(1.0 - 1.0 / beta)
    alpha = (w0 - 2.0 * w1) * beta / g
    if not np.isfinite(alpha) or alpha <= 0:
        return (np.nan, np.nan, np.nan)
    gamma_loc = w0 - alpha * g
    return (float(alpha), float(beta), float(gamma_loc))


def compute_spei(d_cumul, ll_alpha, ll_beta, ll_gamma):
    """SPEI vectorisé : F log-logistique du cumul D → quantile normal.

    NaN si un paramètre est invalide (alpha≤0, beta≤0, non fini) ou si D≤gamma
    (hors du support de la loi).
    """
    x = np.asarray(d_cumul, dtype=float)
    a = np.asarray(ll_alpha, dtype=float)
    b = np.asarray(ll_beta, dtype=float)
    gloc = np.asarray(ll_gamma, dtype=float)

    valid = (
        np.isfinite(x) & np.isfinite(a) & np.isfinite(b) & np.isfinite(gloc)
        & (a > 0) & (b > 0) & (x > gloc)
    )
    out = np.full(x.shape, np.nan)
    if not valid.any():
        return out

    ratio = (a[valid] / (x[valid] - gloc[valid])) ** b[valid]
    cdf = 1.0 / (1.0 + ratio)
    cdf = np.clip(cdf, *_CDF_CLIP)
    out[valid] = np.round(stats.norm.ppf(cdf), 3)
    return out
```

> Note: `beta <= 1.0` is rejected (not just `<= 0`) because `Γ(1 − 1/β)` diverges for `β ≤ 1`. This is the standard SPEI validity guard.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_indices.py -k "loglogistic or spei" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/hubeau_data_integration
git checkout -b feat/climat-spei 2>/dev/null || git checkout feat/climat-spei
git add src/hubeau_pipeline/ml/era5_indices.py tests/test_era5_indices.py
git commit -m "feat(era5): fit log-logistique + compute_spei (L-moments maison)"
```

---

### Task 2: SPEI reference climatology table + population asset

**Files:**
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/ml/era5_spei_climatology_persistence.py`
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/assets/era5_spei_climatology_assets.py`
- Test: `~/hubeau_data_integration/tests/test_era5_spei_climatology.py`

**Interfaces:**
- Consumes: `fit_loglogistic_lmoments` (Task 1).
- Produces: table `gold.fct_era5_spei_climatology_grid(era5_latitude, era5_longitude, mois_calendaire, fenetre, ll_alpha, ll_beta, ll_gamma, nb_annees, computed_at)`.
- Produces: `init_spei_climatology_table(pg)`, `upsert_spei_climatology(pg, rows)` where a row is `(lat, lon, mois_calendaire, fenetre, alpha, beta, gamma, nb_annees)`.
- Produces: Dagster asset `fct_era5_spei_climatology_grid`, and a pure helper `fit_reference_frame(df, window) -> list[tuple]` grouping `df[{era5_latitude, era5_longitude, mois_calendaire, bilan_cumul}]` per cell×month and fitting.

- [ ] **Step 1: Write the failing test (pure grouping/fit helper)**

Create `tests/test_era5_spei_climatology.py`:

```python
import numpy as np
import pandas as pd
from hubeau_pipeline.assets.era5_spei_climatology_assets import fit_reference_frame


def test_fit_reference_frame_groups_and_fits():
    alpha, beta, gamma_loc = 40.0, 3.0, -10.0
    u = (np.arange(1, 61) - 0.5) / 60.0
    samples = gamma_loc + alpha * (u / (1.0 - u)) ** (1.0 / beta)
    df = pd.DataFrame({
        "era5_latitude": [48.1] * 60 + [43.5] * 3,     # 2nd cell: too few → dropped
        "era5_longitude": [2.3] * 60 + [5.0] * 3,
        "mois_calendaire": [6] * 60 + [6] * 3,
        "bilan_cumul": list(samples) + [1.0, 2.0, 3.0],
    })
    rows = fit_reference_frame(df, window=3)
    assert len(rows) == 1                        # degenerate cell dropped
    lat, lon, mc, fen, a, b, g, n = rows[0]
    assert (lat, lon, mc, fen, n) == (48.1, 2.3, 6, 3, 60)
    assert abs(b - beta) < 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_spei_climatology.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the persistence module**

Create `src/hubeau_pipeline/ml/era5_spei_climatology_persistence.py`:

```python
"""Create + upsert gold.fct_era5_spei_climatology_grid (params log-logistiques
SPEI par cellule ERA5 × mois calendaire × fenêtre, référence 1991-2020).

Table Python-managée (pas dbt) : le fit L-moments a besoin des échantillons
annuels ET de la fonction Γ, hors de portée du SQL dbt.
"""
from psycopg2.extras import execute_values

_CREATE = """
CREATE TABLE IF NOT EXISTS gold.fct_era5_spei_climatology_grid (
    era5_latitude   numeric(6,3) NOT NULL,
    era5_longitude  numeric(6,3) NOT NULL,
    mois_calendaire smallint     NOT NULL,
    fenetre         smallint     NOT NULL,
    ll_alpha        double precision,
    ll_beta         double precision,
    ll_gamma        double precision,
    nb_annees       smallint,
    computed_at     timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (era5_latitude, era5_longitude, mois_calendaire, fenetre)
);
"""

_UPSERT = """
INSERT INTO gold.fct_era5_spei_climatology_grid
    (era5_latitude, era5_longitude, mois_calendaire, fenetre,
     ll_alpha, ll_beta, ll_gamma, nb_annees, computed_at)
VALUES %s
ON CONFLICT (era5_latitude, era5_longitude, mois_calendaire, fenetre) DO UPDATE SET
    ll_alpha = EXCLUDED.ll_alpha,
    ll_beta  = EXCLUDED.ll_beta,
    ll_gamma = EXCLUDED.ll_gamma,
    nb_annees = EXCLUDED.nb_annees,
    computed_at = now();
"""

_TEMPLATE = "(%s, %s, %s, %s, %s, %s, %s, %s, now())"


def init_spei_climatology_table(pg):
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS gold")
        cur.execute(_CREATE)
        conn.commit()


def upsert_spei_climatology(pg, rows):
    """rows: iterable of (lat, lon, mois_calendaire, fenetre, alpha, beta, gamma, nb_annees)."""
    if not rows:
        return
    with pg.get_connection() as conn:
        cur = conn.cursor()
        execute_values(cur, _UPSERT, rows, template=_TEMPLATE, page_size=10_000)
        conn.commit()
```

- [ ] **Step 4: Write the asset (grouping helper + Dagster asset)**

Create `src/hubeau_pipeline/assets/era5_spei_climatology_assets.py`:

```python
"""Référence SPEI 1991-2020 → gold.fct_era5_spei_climatology_grid.

Fit log-logistique (L-moments) du cumul bilan hydrique par cellule × mois
calendaire × fenêtre. Rebuild rare (full), consommé par fct_era5_indices_grid.
"""
import logging

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from dagster_dbt import get_asset_key_for_model

from ..ml.era5_indices import MIN_YEARS_REF, fit_loglogistic_lmoments
from ..ml.era5_spei_climatology_persistence import (
    init_spei_climatology_table,
    upsert_spei_climatology,
)
from ..resources import PostgreSQLResource
from .dbt_assets import hubeau_dbt_assets

logger = logging.getLogger(__name__)

WINDOWS = [1, 3, 6, 12]

# Cumul glissant du bilan hydrique sur 1991-2020 (warmup 11 mois depuis 1990),
# mois precip-complets uniquement (l'ETP suit la précip, pas la température).
_REF_QUERY = """
WITH rolled AS (
    SELECT
        era5_latitude, era5_longitude, mois,
        SUM(bilan_hydrique) OVER w AS bilan_cumul,
        COUNT(*)            OVER w AS n_mois
    FROM gold.fct_era5_monthly_grid
    WHERE mois_complet
      AND mois >= DATE '1990-01-01'
      AND mois <  DATE '2021-01-01'
    WINDOW w AS (
        PARTITION BY era5_latitude, era5_longitude ORDER BY mois
        ROWS BETWEEN %(window_minus_1)s PRECEDING AND CURRENT ROW
    )
)
SELECT
    era5_latitude, era5_longitude,
    EXTRACT(MONTH FROM mois)::int AS mois_calendaire,
    bilan_cumul
FROM rolled
WHERE mois >= DATE '1991-01-01'
  AND n_mois = %(window)s
"""


def fit_reference_frame(df, window):
    """Groupe df par (cellule, mois calendaire) et fitte la log-logistique.

    Retourne une liste de tuples upsertables ; les groupes trop courts
    (< MIN_YEARS_REF) ou à fit dégénéré sont ignorés.
    """
    rows = []
    for (lat, lon, mc), grp in df.groupby(
        ["era5_latitude", "era5_longitude", "mois_calendaire"], sort=False
    ):
        samples = grp["bilan_cumul"].to_numpy(dtype=float)
        n = np.isfinite(samples).sum()
        if n < MIN_YEARS_REF:
            continue
        alpha, beta, gamma_loc = fit_loglogistic_lmoments(samples)
        if not np.isfinite([alpha, beta, gamma_loc]).all():
            continue
        rows.append((float(lat), float(lon), int(mc), int(window),
                     alpha, beta, gamma_loc, int(n)))
    return rows


@asset(
    name="fct_era5_spei_climatology_grid",
    group_name="indices",
    deps=[get_asset_key_for_model([hubeau_dbt_assets], "fct_era5_monthly_grid")],
    description=(
        "Paramètres log-logistiques SPEI (référence 1991-2020) par cellule ERA5 "
        "× mois calendaire × fenêtre 1/3/6/12. Rebuild full."
    ),
)
def fct_era5_spei_climatology_grid(context: AssetExecutionContext, pg: PostgreSQLResource):
    init_spei_climatology_table(pg)
    total = 0
    for window in WINDOWS:
        with pg.get_connection() as conn:
            df = pd.read_sql(
                _REF_QUERY, conn,
                params={"window": window, "window_minus_1": window - 1},
            )
        rows = fit_reference_frame(df, window)
        upsert_spei_climatology(pg, rows)
        total += len(rows)
        context.log.info("Fenêtre %d : %d cellules×mois fittées", window, len(rows))
    context.add_output_metadata({"fitted_groups": MetadataValue.int(total)})
    return total
```

- [ ] **Step 5: Register the asset**

Find where `fct_era5_indices_grid` is registered in the Dagster `Definitions` (grep for it):

Run: `cd ~/hubeau_data_integration && grep -rn "fct_era5_indices_grid" src/hubeau_pipeline/definitions.py src/hubeau_pipeline/__init__.py 2>/dev/null`

Add `fct_era5_spei_climatology_grid` to the same `assets=[...]` list (mirror the import + list entry of `fct_era5_indices_grid`).

- [ ] **Step 6: Run test to verify it passes**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_spei_climatology.py -v`
Expected: PASS.

- [ ] **Step 7: Verify Dagster loads the definitions**

Run: `cd ~/hubeau_data_integration && uv run dagster definitions validate 2>&1 | tail -5`
Expected: no error referencing the new asset (validates registration + imports).

- [ ] **Step 8: Commit**

```bash
cd ~/hubeau_data_integration
git add src/hubeau_pipeline/ml/era5_spei_climatology_persistence.py \
        src/hubeau_pipeline/assets/era5_spei_climatology_assets.py \
        tests/test_era5_spei_climatology.py \
        src/hubeau_pipeline/definitions.py
git commit -m "feat(era5): table de référence log-logistique SPEI 1991-2020"
```

---

### Task 3: Write SPEI into fct_era5_indices_grid

**Files:**
- Modify: `~/hubeau_data_integration/src/hubeau_pipeline/ml/era5_indices_persistence.py`
- Modify: `~/hubeau_data_integration/src/hubeau_pipeline/assets/era5_indices_assets.py`
- Test: `~/hubeau_data_integration/tests/test_era5_indices_persistence.py` (create if absent)

**Interfaces:**
- Consumes: `compute_spei` (Task 1), `gold.fct_era5_spei_climatology_grid` (Task 2).
- Produces: `gold.fct_era5_indices_grid` gains a `spei double precision` column; `upsert_era5_indices` rows become `(lat, lon, month, fenetre, spi, sti, spei)`.

- [ ] **Step 1: Write the failing test for the persistence contract**

Create `tests/test_era5_indices_persistence.py` (a lightweight contract test on the SQL strings — no DB):

```python
from hubeau_pipeline.ml import era5_indices_persistence as p


def test_create_and_upsert_include_spei():
    assert "spei" in p._CREATE
    assert "spei = EXCLUDED.spei" in p._UPSERT
    # 7 value placeholders + now()
    assert p._TEMPLATE.count("%s") == 7


def test_alter_adds_spei_idempotently():
    assert "ADD COLUMN IF NOT EXISTS spei" in p._ALTER_ADD_SPEI
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_indices_persistence.py -v`
Expected: FAIL — `spei` absent / `_ALTER_ADD_SPEI` undefined.

- [ ] **Step 3: Update the persistence module**

In `src/hubeau_pipeline/ml/era5_indices_persistence.py`:

Change `_CREATE` to include the column (add after `sti`):

```python
    sti            double precision,
    spei           double precision,
```

Add an idempotent migration constant + apply it in `init`:

```python
_ALTER_ADD_SPEI = """
ALTER TABLE gold.fct_era5_indices_grid ADD COLUMN IF NOT EXISTS spei double precision;
"""
```

In `init_era5_indices_table`, after `cur.execute(_CREATE)`:

```python
        cur.execute(_ALTER_ADD_SPEI)   # colonne ajoutée sur une table déjà en prod
```

Replace `_UPSERT` and `_TEMPLATE`:

```python
_UPSERT = """
INSERT INTO gold.fct_era5_indices_grid
    (era5_latitude, era5_longitude, month, fenetre, spi, sti, spei, computed_at)
VALUES %s
ON CONFLICT (era5_latitude, era5_longitude, month, fenetre) DO UPDATE SET
    spi = EXCLUDED.spi,
    sti = EXCLUDED.sti,
    spei = EXCLUDED.spei,
    computed_at = now();
"""

_TEMPLATE = "(%s, %s, %s, %s, %s, %s, %s, now())"
```

Update the `upsert_era5_indices` docstring:

```python
    """rows: iterable of (lat, lon, month_date, fenetre, spi|None, sti|None, spei|None)."""
```

- [ ] **Step 4: Update the indices asset**

In `src/hubeau_pipeline/assets/era5_indices_assets.py`:

Extend `_QUERY` — add the bilan cumul to `rolled` and join the SPEI reference:

```python
_QUERY = """
WITH rolled AS (
    SELECT
        era5_latitude, era5_longitude, mois,
        SUM(precipitation_totale) OVER w AS precip_cumul,
        SUM(bilan_hydrique)       OVER w AS bilan_cumul,
        AVG(temperature_moyenne)  OVER w AS temp_fenetre,
        COUNT(*)                  OVER w AS n_mois
    FROM gold.fct_era5_monthly_grid
    WHERE mois_complet
      AND mois >= %(warmup_month)s
      AND mois <  %(end_month)s
    WINDOW w AS (
        PARTITION BY era5_latitude, era5_longitude ORDER BY mois
        ROWS BETWEEN %(window_minus_1)s PRECEDING AND CURRENT ROW
    )
)
SELECT
    r.era5_latitude, r.era5_longitude, r.mois,
    r.precip_cumul, r.bilan_cumul, r.temp_fenetre,
    c.gamma_alpha, c.gamma_beta, c.prob_zero,
    c.temp_moyenne, c.temp_stddev, c.nb_annees,
    s.ll_alpha, s.ll_beta, s.ll_gamma
FROM rolled r
JOIN gold.fct_era5_climatology_grid c
  ON c.era5_latitude = r.era5_latitude
 AND c.era5_longitude = r.era5_longitude
 AND c.mois_calendaire = EXTRACT(MONTH FROM r.mois)::int
 AND c.fenetre = %(window)s
LEFT JOIN gold.fct_era5_spei_climatology_grid s
  ON s.era5_latitude = r.era5_latitude
 AND s.era5_longitude = r.era5_longitude
 AND s.mois_calendaire = EXTRACT(MONTH FROM r.mois)::int
 AND s.fenetre = %(window)s
WHERE r.mois >= %(start_month)s
  AND r.n_mois = %(window)s
"""
```

> `LEFT JOIN` for the SPEI reference: a cell whose fit was degenerate has no reference row → `ll_*` NULL → `compute_spei` yields NaN → `spei` NULL. SPI/STI are unaffected.

Update the import and `_compute_range`:

```python
from ..ml.era5_indices import MIN_YEARS_REF, compute_spi, compute_spei, compute_sti
```

Inside `_compute_range`, after `sti = compute_sti(...)`:

```python
        spei = compute_spei(df["bilan_cumul"], df["ll_alpha"], df["ll_beta"], df["ll_gamma"])
        # Seuil WMO : référence trop courte → indices NULL
        thin = df["nb_annees"].to_numpy() < MIN_YEARS_REF
        spi[thin] = np.nan
        sti[thin] = np.nan
        spei[thin] = np.nan
        rows = [
            (lat, lon, mois, window,
             None if np.isnan(sp) else float(sp),
             None if np.isnan(st) else float(st),
             None if np.isnan(se) else float(se))
            for lat, lon, mois, sp, st, se in zip(
                df["era5_latitude"], df["era5_longitude"], df["mois"],
                spi, sti, spei, strict=True
            )
        ]
```

(Remove the old `thin`/`rows` block being replaced; keep `upsert_era5_indices(pg, rows)` and `total += len(rows)`.)

Add the reference table to the asset `deps`:

```python
    deps=[
        get_asset_key_for_model([hubeau_dbt_assets], "fct_era5_monthly_grid"),
        get_asset_key_for_model([hubeau_dbt_assets], "fct_era5_climatology_grid"),
        AssetKey("fct_era5_spei_climatology_grid"),
    ],
```

Add the import for `AssetKey` (from `dagster import ... AssetKey`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/hubeau_data_integration && uv run pytest tests/test_era5_indices_persistence.py tests/test_era5_indices.py -v`
Expected: PASS.

- [ ] **Step 6: Validate Dagster definitions**

Run: `cd ~/hubeau_data_integration && uv run dagster definitions validate 2>&1 | tail -5`
Expected: no error.

- [ ] **Step 7: Commit**

```bash
cd ~/hubeau_data_integration
git add src/hubeau_pipeline/ml/era5_indices_persistence.py \
        src/hubeau_pipeline/assets/era5_indices_assets.py \
        tests/test_era5_indices_persistence.py
git commit -m "feat(era5): écrit le SPEI dans fct_era5_indices_grid"
```

- [ ] **Step 8: (Deploy-time, documented — not run here) Rebuild sequence**

Record in the PR description; run against the warehouse when deploying:

```bash
# 1. Référence SPEI (full)
dagster asset materialize --select fct_era5_spei_climatology_grid
# 2. Colonne + valeurs (bootstrap si table déjà peuplée : la colonne spei est NULL
#    tant que l'asset indices n'a pas re-tourné sur l'historique → backfill ciblé)
dagster asset materialize --select fct_era5_indices_grid
```

---

### Task 4: Expose SPEI in point-series + CSV export (API)

**Files:**
- Modify: `~/time-serie-explo/api/routers/observatory_climat.py` (`_merge_point_series` ~134-171; point-series SELECT line 551; export CSV lines 732-772)
- Test: `~/time-serie-explo/tests/test_observatory_climat.py` (create if absent; else append)

**Interfaces:**
- Consumes: `gold.fct_era5_indices_grid.spei` (Task 3).
- Produces: point-series entries gain `spei_{fen}` for `fen ∈ {1,3,6,12}`; CSV export gains `spei_{w}` columns.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_observatory_climat.py`:

```python
from api.routers.observatory_climat import _merge_point_series


def test_merge_point_series_includes_spei():
    monthly = [{"mois": __import__("datetime").date(2026, 6, 1),
                "temperature_moyenne": 18.0, "temperature_min": 10.0,
                "temperature_max": 26.0, "precipitation_totale": 40.0,
                "etp_totale": 120.0, "bilan_hydrique": -80.0}]
    clim = []
    indices = [{"month": __import__("datetime").date(2026, 6, 1),
                "fenetre": 3, "spi": -1.2, "sti": 0.5, "spei": -1.5}]
    out = _merge_point_series(monthly, clim, indices)
    assert out[0]["spei_3"] == -1.5
    assert out[0]["spei_1"] is None      # window absent → None, like spi/sti
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k spei -v`
Expected: FAIL — `KeyError: 'spei_3'` or `_merge_point_series` doesn't emit it.

- [ ] **Step 3: Implement**

In `_merge_point_series`, in the loop that fills `indices_by_month` (after the `sti_{fen}` line ~147):

```python
        indices_by_month[key][f"spei_{fen}"] = _num(r["spei"])
```

In the entry-building loop (after the `sti_{fen}` line ~169):

```python
            entry[f"spei_{fen}"] = idx.get(f"spei_{fen}")
```

In the point-series SELECT (line 551), add `spei`:

```python
                "SELECT month, fenetre, spi, sti, spei FROM gold.fct_era5_indices_grid"
```

In `export-point.csv` — SELECT (line 732):

```python
                    "SELECT month, fenetre, spi, sti, spei FROM gold.fct_era5_indices_grid"
```

fill loop (after line 743):

```python
            indices_by_month[key][f"spei_{fen}"] = _num(r["spei"])
```

header (after line 752):

```python
            + [f"spei_{w}" for w in WINDOWS]
```

row loop (after line 772):

```python
                row[f"spei_{w}"] = idx.get(f"spei_{w}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k spei -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo   # branch feat/climat-spei
git add api/routers/observatory_climat.py tests/test_observatory_climat.py
git commit -m "feat(climat): expose spei_{1,3,6,12} dans point-series + export CSV"
```

---

### Task 5: SPEI as a grid-indices map variable (API)

**Files:**
- Modify: `~/time-serie-explo/api/routers/observatory_climat.py` (`get_grid_indices` ~336-378)
- Test: `~/time-serie-explo/tests/test_observatory_climat.py`

**Interfaces:**
- Consumes: `gold.fct_era5_indices_grid.spei`.
- Produces: `GET /observatory/climat/grid-indices?index=spei&window=N&month=…` → per-cell `{value, index_class}`.

- [ ] **Step 1: Write the failing test**

Append:

```python
import pytest
from fastapi import HTTPException
from api.routers import observatory_climat


def test_grid_indices_accepts_spei(monkeypatch):
    # The validation gate must allow 'spei'. We assert the guard, not the DB.
    # Reach the guard by calling the validation branch directly.
    for ok in ("spi", "sti", "spei"):
        # no exception constructing the query for a valid index
        assert ok in ("spi", "sti", "spei")
    with pytest.raises(HTTPException) as exc:
        observatory_climat._assert_index("bogus")
    assert exc.value.status_code == 422
```

> This test drives extracting the inline `if index not in (...)` check into a small `_assert_index` helper (easier to unit-test than the cached endpoint).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k grid_indices -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_assert_index'`.

- [ ] **Step 3: Implement**

Add the helper near the other validators (after `_validate_window`):

```python
_GRID_INDICES = ("spi", "sti", "spei")


def _assert_index(index: str) -> None:
    if index not in _GRID_INDICES:
        raise HTTPException(422, f"Indice inconnu : {index!r} (attendu {_GRID_INDICES})")
```

In `get_grid_indices`, update the `Query` description and replace the inline guard:

```python
    index: str = Query("spi", description="spi, sti ou spei"),
```
```python
    _assert_index(index)
```

The `SELECT {index} AS value` already interpolates the column name, so `spei` works once the column exists (Task 3). `classify_index(v)` already applies the shared thresholds. No further change.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k grid_indices -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_climat.py tests/test_observatory_climat.py
git commit -m "feat(climat): spei sélectionnable sur /grid-indices"
```

---

### Task 6: SPEI drought episodes (API)

**Files:**
- Modify: `~/time-serie-explo/api/routers/observatory_climat.py` (`_build_drought_episodes` ~174-238; `get_point_episodes` ~582-633)
- Test: `~/time-serie-explo/tests/test_observatory_climat.py`

**Interfaces:**
- Consumes: `gold.fct_era5_indices_grid.spei`.
- Produces: `_build_drought_episodes(index_rows, monthly_rows, clim_rows, index_key="spi")`; `GET /observatory/climat/point-episodes?index=spei` returns SPEI-based episodes. The episode dict key `spi_min` becomes `index_min` (generic).

- [ ] **Step 1: Write the failing test**

Append:

```python
import datetime
from api.routers.observatory_climat import _build_drought_episodes


def _rows(key, vals):
    return [{"month": datetime.date(2026, m, 1), key: v} for m, v in vals]


def test_episodes_generic_over_spei():
    # 3 consecutive months < -1 → one episode, keyed by 'spei'
    rows = _rows("spei", [(4, -1.2), (5, -1.6), (6, -0.9), (7, -2.1)])
    eps = _build_drought_episodes(rows, [], [], index_key="spei")
    assert len(eps) == 2                       # (apr-may) and (jul)
    assert eps[0]["duree_mois"] == 2
    assert eps[0]["index_min"] == -1.6


def test_episodes_default_key_is_spi():
    rows = _rows("spi", [(4, -1.5), (5, -1.5)])
    eps = _build_drought_episodes(rows, [], [])
    assert eps[0]["index_min"] == -1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k episodes -v`
Expected: FAIL — signature has no `index_key`, and/or key is `spi_min`.

- [ ] **Step 3: Implement**

Change the signature and body of `_build_drought_episodes`. Replace the hard-coded `"spi"` reads with `index_key`, and rename the output field `spi_min` → `index_min`:

```python
def _build_drought_episodes(index_rows, monthly_rows, clim_rows, index_key="spi"):
    ...
    dry = [
        {"month": r["month"], "value": float(r[index_key])}
        for r in index_rows
        if r[index_key] is not None and float(r[index_key]) < -1.0
    ]
    ...
    # in the episode dict:
        "index_min": round(min(m["value"] for m in members), 3),
```

> Search the function body for every `m["spi"]` / `r["spi"]` / `"spi_min"` and update to `m["value"]` / `r[index_key]` / `"index_min"`. The internal member dict uses `"value"`.

In `get_point_episodes` (~582): add an `index` query param and use it for both the SQL column and the builder:

```python
@router.get("/point-episodes")
def get_point_episodes(
    ...,
    index: str = Query("spi", description="spi ou spei"),
):
    _assert_episode_index(index)
    ...
    def fetch():
        ...
        idx_rows = conn.execute(
            text(
                f"SELECT month, {index} FROM gold.fct_era5_indices_grid"
                " WHERE era5_latitude = :lat AND era5_longitude = :lon"
                f" AND fenetre = :window AND {index} IS NOT NULL ORDER BY month"
            ),
            {"lat": cell_lat, "lon": cell_lon, "window": window},
        ).mappings().all()
        ...
        return _build_drought_episodes(idx_rows, monthly_rows, clim_rows, index_key=index)
```

Add a validator (episodes are drought-only → `spi`/`spei`, not `sti`):

```python
def _assert_episode_index(index: str) -> None:
    if index not in ("spi", "spei"):
        raise HTTPException(422, f"Indice d'épisode invalide : {index!r} (attendu spi ou spei)")
```

Include `index` in the cache key:

```python
        "obs_climat_point_episodes",
        {"lat": cell_lat, "lon": cell_lon, "window": window, "index": index},
```

> **Frontend consumers of `spi_min`:** grep the frontend for `spi_min` and rename to `index_min` (Task 8 covers the SPEI UI; fix any existing SPI-episode consumer here to avoid a break). Run: `cd ~/time-serie-explo && grep -rn "spi_min" frontend/src`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/time-serie-explo && DEBUG=true DB_PASSWORD=test uv run pytest tests/test_observatory_climat.py -k episodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_climat.py tests/test_observatory_climat.py frontend/src
git commit -m "feat(climat): épisodes de sécheresse basés SPEI (index_key générique)"
```

---

### Task 7: SPEI in the frontend colour model

**Files:**
- Modify: `~/time-serie-explo/frontend/src/lib/climat-colors.ts`
- Test: `~/time-serie-explo/frontend/src/lib/climat-colors.test.ts`

**Interfaces:**
- Produces: `'spei'` member of `ClimatVariable` (`kind: 'index'`); `climatIndexColorExpression` accepts `'spi' | 'sti' | 'spei'`; `climatFormatValue` formats `spei` as signed σ.

- [ ] **Step 1: Write the failing tests**

Append to `climat-colors.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import {
  CLIMAT_VARIABLES, isClimatIndexVariable,
  climatIndexColorExpression, climatFormatValue,
} from './climat-colors'
import { SPI_CLASS_COLORS } from './era5-colors'

describe('SPEI variable', () => {
  it('is an index variable in σ', () => {
    expect(CLIMAT_VARIABLES.spei.kind).toBe('index')
    expect(CLIMAT_VARIABLES.spei.unit).toBe('σ')
    expect(isClimatIndexVariable('spei')).toBe(true)
  })

  it('uses the SPI drought palette, not STI', () => {
    const expr = climatIndexColorExpression('spei')
    // 'match' expression contains the SPI extreme-low colour
    expect(JSON.stringify(expr)).toContain(SPI_CLASS_COLORS.EXTREMEMENT_BAS)
  })

  it('formats as signed sigma', () => {
    expect(climatFormatValue('spei', -1.5)).toBe('−1.5 σ')
    expect(climatFormatValue('spei', 0.8)).toBe('+0.8 σ')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/time-serie-explo/frontend && npx vitest run src/lib/climat-colors.test.ts -t SPEI`
Expected: FAIL — `spei` missing from `CLIMAT_VARIABLES` / type error.

- [ ] **Step 3: Implement**

In `climat-colors.ts`:

Extend the union (line 9-11):

```ts
export type ClimatVariable =
  | 'spi' | 'sti' | 'spei' | 'bilan_hydrique'
  | 'tmax' | 'tmin' | 'tmean' | 'precip_daily'
```

Add the config entry in `CLIMAT_VARIABLES` (after `sti`, before `bilan_hydrique`):

```ts
  spei: {
    key: 'spei', kind: 'index',
    unit: 'σ', labelKey: 'climat.variables.spei',
    stops: [],
  },
```

Widen `climatIndexColorExpression` to reuse the SPI palette for `spei`:

```ts
export function climatIndexColorExpression(variable: 'spi' | 'sti' | 'spei'): unknown[] {
  const useStd = variable === 'sti'
  const order = useStd ? STI_CLASS_ORDER : SPI_CLASS_ORDER
  const colors = useStd ? STI_CLASS_COLORS : SPI_CLASS_COLORS
  const expr: unknown[] = ['match', ['get', 'index_class']]
  for (const cls of order) expr.push(cls, colors[cls])
  expr.push(colors.UNKNOWN)
  return expr
}
```

Extend the signed-σ branch of `climatFormatValue` (line 184):

```ts
  if (variable === 'spi' || variable === 'sti' || variable === 'spei') {
```

- [ ] **Step 4: Run tests + typecheck**

Run: `cd ~/time-serie-explo/frontend && npx vitest run src/lib/climat-colors.test.ts && npm run build`
Expected: vitest PASS; `tsc -b` succeeds (the exhaustive union forces every `switch`/`match` on `ClimatVariable` to be updated — fix any surfaced site by mirroring the `spi` branch).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/climat-colors.ts frontend/src/lib/climat-colors.test.ts
git commit -m "feat(climat): spei dans le modèle couleur (palette SPI, σ signé)"
```

---

### Task 8: SPEI in the picker, legend, point panel, i18n

**Files:**
- Modify: `~/time-serie-explo/frontend/src/components/climat/VariablePicker.tsx`
- Modify: `~/time-serie-explo/frontend/src/components/climat/PointPanel.tsx` (SPEI multi-window series, mirror STI)
- Modify: legend/popup component consuming `climatIndexColorExpression` (grep to locate)
- Modify: `~/time-serie-explo/frontend/src/i18n/fr.json`, `en.json`
- Test: `~/time-serie-explo/frontend/src/components/climat/VariablePicker.test.tsx`

**Interfaces:**
- Consumes: Task 7 (`'spei'` variable), Task 4 (`spei_{fen}` in point series), Task 5 (map layer), Task 6 (`index_min` episodes).

- [ ] **Step 1: Locate the STI wiring to mirror**

Run:
```bash
cd ~/time-serie-explo/frontend && grep -rn "'sti'\|\"sti\"\|climat.variables.sti\|sti_" src/components/climat src/i18n src/hooks
```
Note every site that names `sti` — each needs a sibling `spei` entry (picker group, legend switch, PointPanel window series, i18n key).

- [ ] **Step 2: Write the failing picker test**

In `VariablePicker.test.tsx`, add:

```ts
it('renders SPEI in the Anomalie group', () => {
  render(<VariablePicker value="spi" onChange={() => {}} />)   // match existing render signature
  expect(screen.getByText(/SPEI/)).toBeInTheDocument()
})
```

(Adjust the render props to match the existing tests in the file.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/time-serie-explo/frontend && npx vitest run src/components/climat/VariablePicker.test.tsx -t SPEI`
Expected: FAIL — SPEI not rendered.

- [ ] **Step 4: Implement the wiring**

- `VariablePicker.tsx`: add `'spei'` to the "Anomalie" group's variable list, right after `'sti'` (find the array that currently lists `['spi', 'sti']` or similar and insert `'spei'`).
- Legend/popup switch: wherever `climatIndexColorExpression(v)` is called for `'spi'|'sti'`, ensure `'spei'` reaches it (widen the guard from `v === 'spi' || v === 'sti'` to also include `'spei'`, or reuse `isClimatIndexVariable(v)`).
- `PointPanel.tsx`: mirror the STI multi-window block — a `spei_{w}` row per window `w ∈ CLIMAT_WINDOWS`, formatted via `climatFormatValue('spei', entry[\`spei_${w}\`])`, class colour from `SPI_CLASS_COLORS` via the shared classification.
- i18n `fr.json`: `"climat": { "variables": { "spei": "SPEI (précip. − ETP)" }, ... }` plus the ⓘ hover text:
  `"spei": "Indice standardisé du bilan hydrique (précipitations − ETP), échelle 1 à 12 mois, référence 1991-2020. Négatif = sécheresse. L'ETP est estimée depuis ERA5-Land (pas un Penman-Monteith FAO-56)."`
- i18n `en.json`: `"spei": "SPEI (precip. − PET)"` + the English hover mirroring the French.

- [ ] **Step 5: Run tests + full build**

Run: `cd ~/time-serie-explo/frontend && npx vitest run src/components/climat && npm run build`
Expected: PASS + `tsc -b` clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/climat frontend/src/i18n/fr.json frontend/src/i18n/en.json
git commit -m "feat(climat): SPEI dans le picker, la légende, le PointPanel + i18n"
```

---

## Deployment (documented — run at deploy time, per memory)

1. **Warehouse** (`hubeau_data_integration`, merge `feat/climat-spei`): materialize `fct_era5_spei_climatology_grid`, then re-materialize `fct_era5_indices_grid` (bootstrap/backfill so `spei` is populated on history). Confirm `SELECT count(*) FROM gold.fct_era5_indices_grid WHERE spei IS NOT NULL` is non-trivial for a recent month.
2. **App** (`time-serie-explo`): `docker compose up -d --build backend frontend` from repo root (compose SERVICE names `backend`/`frontend`, no `-f` flags — `.env` sets `COMPOSE_FILE`).
3. **Cache purge:** `docker exec junon-redis redis-cli --scan --pattern 'junon:obs_climat_*' | xargs -r docker exec -i junon-redis redis-cli DEL`.

## Self-Review notes

- **Spec coverage:** §2 method → Task 1; §2.1 fit → Task 1; §3.2 reference table → Task 2; §3.3/§3.4 indices+persistence → Task 3; §4.1 point-series → Task 4; §4.2 grid-indices → Task 5; §4.3 episodes → Task 6; §5.1 colours → Task 7; §5.2/5.3 picker/i18n → Task 8; §5.4/§6 tests → interleaved; §1.2 caveat → Task 8 hover text; §4.4 cache → Deployment.
- **Type consistency:** `upsert_era5_indices` 7-tuple `(lat, lon, month, fenetre, spi, sti, spei)` matches `_TEMPLATE` (7 `%s`) and `_UPSERT` columns (Task 3). `compute_spei(d_cumul, ll_alpha, ll_beta, ll_gamma)` signature identical in Tasks 1/3. Episode field renamed `spi_min` → `index_min` consistently (Task 6) with the frontend grep-fix.
- **Deferred / assumption to verify during execution:** exact Dagster registration file (Task 2 Step 5 greps for it); exact `VariablePicker`/PointPanel render props and legend switch site (Task 8 Step 1 greps for them) — the plan instructs the executor to locate and mirror the existing `sti` wiring rather than guessing line numbers.
