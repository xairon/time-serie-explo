# IPS à référence fixe — Implementation Plan (Partie A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Remplacer les 4 calculs d'IPS/SPLI incohérents par UNE référence fixe (défaut 1991–2020, repli intelligent), calculée dans le warehouse et lue par l'app, partagée Observatoire ↔ Prévision, avec affichage des plages en clair.

**Architecture:** Le warehouse (`hubeau_data_integration`, dagster) calcule par `(type, code, mois)` une grille de quantiles empiriques sur une fenêtre de référence fixe → table `gold.station_reference_stats`. L'app (`time-serie-explo`) lit cette grille pour les séries SPLI/SSFI, les bandes de la fiche station et les bandes de prévision. `station_current_index` reclasse le dernier mois contre la grille fixe.

**Tech Stack:** Warehouse : Python, dagster, pandas/numpy/scipy, PostgreSQL (brgm-postgres). App : FastAPI + SQLAlchemy, React/i18next.

**Spec:** `docs/specs/2026-06-03-ips-reference-et-groupement-stations.md` (Partie A).

**Repos:**
- `~/hubeau_data_integration` (warehouse) — tâches **W1–W4**.
- `~/time-serie-explo` (app) — tâches **A1–A5**.

**⚠️ Étape opérateur (hors TDD, à exécuter délibérément) :** après W1–W4, **matérialiser** l'asset dagster `station_reference_stats` puis re-matérialiser `station_current_index` (écrit dans le prod brgm-postgres). Les tâches app A1–A4 sont implémentables et testables à l'unité avant, mais leur **vérification d'intégration** (curl live) n'est possible qu'après matérialisation. Voir « Matérialisation » en fin de plan.

**Méthode (unifiée, validée) :** percentiles **empiriques** par mois calendaire sur la fenêtre fixe. Bornes de classe = grille aux 6 cutoffs BRGM ; z d'une valeur = `norm.ppf(CDF_empirique(valeur))` par interpolation dans la grille. Mêmes 7 classes que `ml/indices.py` (`_THRESHOLDS_7`).

---

## File Structure

**Warehouse (`~/hubeau_data_integration`)**
- Modify `src/hubeau_pipeline/ml/indices.py` — ajoute `compute_reference_grid`, `grid_to_zscore`, `grid_class_bounds`, constantes `REF_PERIOD`/`PCTL_GRID`/`CLASS_CUTOFF_PCTL`.
- Create `src/hubeau_pipeline/ml/reference_stats_persistence.py` — DDL + upsert `gold.station_reference_stats`.
- Create `src/hubeau_pipeline/assets/reference_stats_assets.py` — asset `station_reference_stats`.
- Modify `src/hubeau_pipeline/assets/current_index_assets.py` — reclasse contre la grille fixe.
- Create `tests/test_reference_grid.py` (ou l'emplacement de tests du repo) — tests de `compute_reference_grid`/`grid_to_zscore`/`grid_class_bounds`.

**App (`~/time-serie-explo`)**
- Create `dashboard/utils/reference.py` — lecture grille + `series_zscores`, `class_bounds_ngf`.
- Modify `api/routers/observatory_piezo.py` — `/spli` et `get_station` lisent la grille.
- Modify `api/routers/observatory_hydro.py` — `/ssfi` lit la grille.
- Modify `api/schemas/observatory.py` — champs `reference_flag`, `baseline_start/end`, `class_bounds` exposés.
- Modify `api/routers/counterfactual.py` + `dashboard/utils/training.py` — bandes lues depuis la grille ; arrêt du gel `ips_meta`.
- Modify `dashboard/utils/counterfactual/ips.py` — retire le calcul de référence, garde classification.
- Modify frontend `SituationPanel`/échelle IPS + i18n — affiche plages m NGF + période.
- Create `tests/test_reference.py` — tests des helpers app.

---

## PARTIE WAREHOUSE

### Task W1 : `compute_reference_grid` + helpers (pure, TDD)

**Files:**
- Modify: `~/hubeau_data_integration/src/hubeau_pipeline/ml/indices.py`
- Test: `~/hubeau_data_integration/tests/test_reference_grid.py` (créer ; adapter au runner du repo — vérifier `pytest` config)

- [ ] **Step 1 : Écrire les tests (échouent)**

```python
import numpy as np
import pandas as pd
from hubeau_pipeline.ml.indices import (
    compute_reference_grid, grid_to_zscore, grid_class_bounds,
    REF_PERIOD, CLASS_CUTOFF_PCTL,
)


def _monthly(start_year, end_year, base=100.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    vals = base + np.sin(idx.month / 12 * 2 * np.pi) * 2 + rng.normal(0, noise, len(idx))
    return [d.strftime("%Y-%m-%d") for d in idx], list(map(float, vals))


def test_grid_normale_when_full_ref_period():
    months, values = _monthly(1991, 2020)
    res = compute_reference_grid(months, values)
    assert res["flag"] == "normale"
    assert res["baseline_start"] == "1991-01-01"
    assert res["baseline_end"] == "2020-12-31"
    # 12 months, each a 99-length grid
    assert set(res["grid"].keys()) == set(range(1, 13))
    assert all(len(res["grid"][m]) == 99 for m in range(1, 13))
    # grid is monotonic non-decreasing per month
    for m in range(1, 13):
        g = res["grid"][m]
        assert all(g[i] <= g[i + 1] + 1e-9 for i in range(len(g) - 1))


def test_grid_provisoire_when_short_record():
    months, values = _monthly(2014, 2024)  # ~10 yrs, < MIN_YEARS
    res = compute_reference_grid(months, values)
    assert res["flag"] == "provisoire"


def test_grid_adaptee_when_recent_30yr_but_not_ref():
    months, values = _monthly(2001, 2024)  # ≥15 yrs, none in 1991-2000 gap -> not full ref
    res = compute_reference_grid(months, values)
    assert res["flag"] in ("adaptee", "normale")  # ≥15 yrs in 1991-2020 portion (2001-2020) -> normale acceptable
    assert res["n_years"] >= 15


def test_zscore_monotonic_in_value():
    months, values = _monthly(1991, 2020)
    res = compute_reference_grid(months, values)
    grid_m = res["grid"][6]
    lo = grid_to_zscore(grid_m[4], grid_m)   # ~5th pctl -> negative z
    hi = grid_to_zscore(grid_m[94], grid_m)  # ~95th pctl -> positive z
    assert lo < 0 < hi


def test_class_bounds_count_and_order():
    months, values = _monthly(1991, 2020)
    res = compute_reference_grid(months, values)
    bounds = grid_class_bounds(res["grid"][6])
    # 6 cutoffs -> 6 boundary values, ascending
    assert len(bounds) == len(CLASS_CUTOFF_PCTL)
    assert all(bounds[i] <= bounds[i + 1] + 1e-9 for i in range(len(bounds) - 1))
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run (depuis `~/hubeau_data_integration`): `python -m pytest tests/test_reference_grid.py -v`
Expected: FAIL (ImportError sur `compute_reference_grid`).

- [ ] **Step 3 : Implémenter dans `ml/indices.py`**

Ajouter en bas de `src/hubeau_pipeline/ml/indices.py` :

```python
# ---- Fixed-reference IPS grid (BRGM-aligned, empirical percentiles) ----

REF_PERIOD = (1991, 2020)          # WMO/BRGM climatological normal (configurable)
MIN_YEARS = 15                     # BRGM minimum for statistical validity
PCTL_GRID = list(range(1, 100))    # store percentiles 1..99
# 7-class boundaries as CDF percentiles = 100 * norm.cdf([-1.75,-1.28,-0.84,0.84,1.28,1.75])
CLASS_CUTOFF_PCTL = [4.01, 10.03, 20.05, 79.95, 89.97, 95.99]


def _select_reference_window(series, ref_period=REF_PERIOD, min_years=MIN_YEARS):
    """Choose the reference window per the fallback ladder.

    Returns (windowed_series, baseline_start, baseline_end, flag, n_years).
    """
    lo, hi = ref_period
    win = series[(series.index.year >= lo) & (series.index.year <= hi)]
    if win.index.year.nunique() >= min_years:
        return win, f"{lo}-01-01", f"{hi}-12-31", "normale", int(win.index.year.nunique())

    # Best decade-aligned 30-yr window with the most years, requiring >= min_years
    best = None
    first_decade = (int(series.index.year.min()) // 10) * 10
    for start in range(first_decade, 2001, 10):
        w = series[(series.index.year >= start) & (series.index.year <= start + 29)]
        ny = w.index.year.nunique()
        if ny >= min_years and (best is None or ny > best[4]):
            best = (w, f"{start}-01-01", f"{start + 29}-12-31", "adaptee", int(ny))
    if best is not None:
        return best

    # Fallback: full record
    return (series, str(series.index.min().date()),
            str(series.index.max().date()), "provisoire", int(series.index.year.nunique()))


def compute_reference_grid(months, values, ref_period=REF_PERIOD,
                           min_years=MIN_YEARS, min_per_month=MIN_PER_MONTH,
                           positive_only=False):
    """Per-calendar-month empirical percentile grid over a fixed reference window.

    Args:
        months: list of ISO date strings (monthly series).
        values: monthly mean values (m NGF for piezo, m3/s for hydro).
        positive_only: if True (streamflow), drop non-positive values.

    Returns dict: {grid: {month: [99 floats]}, baseline_start, baseline_end, flag, n_years}.
    Months with < min_per_month observations are linearly interpolated from neighbours;
    if none available, that month maps to None.
    """
    series = pd.Series(values, index=pd.to_datetime(months), dtype=float).dropna()
    if positive_only:
        series = series[series > 0]
    if series.empty:
        return {"grid": {m: None for m in range(1, 13)},
                "baseline_start": None, "baseline_end": None, "flag": "provisoire", "n_years": 0}

    win, b_start, b_end, flag, n_years = _select_reference_window(series, ref_period, min_years)

    grid = {}
    for m in range(1, 13):
        vals = win[win.index.month == m].values
        if len(vals) >= min_per_month:
            grid[m] = [float(np.percentile(vals, p)) for p in PCTL_GRID]
        else:
            grid[m] = None

    # Interpolate missing months from nearest available neighbours (circular)
    available = {m: g for m, g in grid.items() if g is not None}
    if available:
        for m in range(1, 13):
            if grid[m] is None:
                # nearest neighbour by circular month distance
                nearest = min(available.keys(),
                              key=lambda k: min(abs(k - m), 12 - abs(k - m)))
                grid[m] = available[nearest]

    return {"grid": grid, "baseline_start": b_start, "baseline_end": b_end,
            "flag": flag, "n_years": n_years}


def grid_to_zscore(value, grid_month):
    """Standardize a value against a month's percentile grid (empirical CDF -> normal)."""
    if grid_month is None or value is None or pd.isna(value):
        return None
    # interpolate the percentile rank of `value` within the grid (1..99 -> CDF)
    pct = float(np.interp(value, grid_month, PCTL_GRID)) / 100.0
    pct = float(np.clip(pct, 0.001, 0.999))
    return round(float(stats.norm.ppf(pct)), 3)


def grid_class_bounds(grid_month):
    """Return the 6 class-boundary values (physical units) at the BRGM cutoffs."""
    if grid_month is None:
        return None
    return [float(np.interp(c, PCTL_GRID, grid_month)) for c in CLASS_CUTOFF_PCTL]
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run: `python -m pytest tests/test_reference_grid.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5 : Commit** (dans `~/hubeau_data_integration`)

```bash
git add src/hubeau_pipeline/ml/indices.py tests/test_reference_grid.py
git commit -m "feat(indices): fixed-reference IPS grid (empirical percentiles + fallback)"
```

### Task W2 : Persistance `gold.station_reference_stats`

**Files:**
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/ml/reference_stats_persistence.py`

- [ ] **Step 1 : Créer le module** (calque sur `current_index_persistence.py`)

```python
"""Create + upsert gold.station_reference_stats (per-station per-month reference grid)."""
import json

_CREATE = """
CREATE TABLE IF NOT EXISTS gold.station_reference_stats (
    type            text NOT NULL,
    code            text NOT NULL,
    month           int  NOT NULL,
    quantile_grid   jsonb,
    baseline_start  date,
    baseline_end    date,
    flag            text NOT NULL,
    n_years         int,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (type, code, month)
);
"""

_UPSERT = """
INSERT INTO gold.station_reference_stats
    (type, code, month, quantile_grid, baseline_start, baseline_end, flag, n_years, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (type, code, month) DO UPDATE SET
    quantile_grid = EXCLUDED.quantile_grid,
    baseline_start = EXCLUDED.baseline_start,
    baseline_end = EXCLUDED.baseline_end,
    flag = EXCLUDED.flag,
    n_years = EXCLUDED.n_years,
    computed_at = now();
"""


def init_reference_stats_table(pg):
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS gold")
        cur.execute(_CREATE)
        conn.commit()


def upsert_reference_stats(pg, rows):
    """rows: list of (type, code, month, grid_list|None, baseline_start, baseline_end, flag, n_years).

    grid_list is JSON-serialised here (jsonb column).
    """
    if not rows:
        return
    payload = [
        (t, c, m, json.dumps(g) if g is not None else None, bs, be, flag, ny)
        for (t, c, m, g, bs, be, flag, ny) in rows
    ]
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(_UPSERT, payload)
        conn.commit()
```

- [ ] **Step 2 : Commit**

```bash
git add src/hubeau_pipeline/ml/reference_stats_persistence.py
git commit -m "feat(indices): gold.station_reference_stats persistence (DDL + upsert)"
```

### Task W3 : Asset dagster `station_reference_stats`

**Files:**
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/assets/reference_stats_assets.py`
- Modify: l'agrégat/`Definitions` qui enregistre les assets (chercher où `station_current_index` est référencé : `grep -rn station_current_index src/hubeau_pipeline --include=*.py`), ajouter le nouvel asset.

- [ ] **Step 1 : Créer l'asset**

```python
"""Per-station per-month fixed reference grid → gold.station_reference_stats."""
import logging

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from ..ml.indices import compute_reference_grid
from ..ml.reference_stats_persistence import init_reference_stats_table, upsert_reference_stats
from ..resources import PostgreSQLResource

logger = logging.getLogger(__name__)

# (domain, table, code_col, value_col, positive_only)
_DOMAINS = [
    ("piezo", "gold.fct_monthly_chroniques", "code_bss", "niveau_moyen", False),
    ("hydro", "gold.fct_monthly_hydro", "code_station", "resultat_moyen", True),
]


@asset(
    name="station_reference_stats",
    group_name="indices",
    description="Fixed-reference (1991-2020 + fallback) per-month percentile grid per station.",
)
def station_reference_stats(context: AssetExecutionContext, pg: PostgreSQLResource):
    init_reference_stats_table(pg)
    total = 0
    for domain, table, code_col, value_col, positive_only in _DOMAINS:
        with pg.get_connection() as conn:
            df = pd.read_sql(
                f"SELECT {code_col} AS code, mois, {value_col} AS val "
                f"FROM {table} WHERE {value_col} IS NOT NULL "
                f"AND {value_col} < 1e8 AND {value_col} > -1e4 "
                f"ORDER BY {code_col}, mois",
                conn,
            )
        rows = []
        for code, g in df.groupby("code"):
            months = g["mois"].astype(str).tolist()
            values = g["val"].astype(float).tolist()
            res = compute_reference_grid(months, values, positive_only=positive_only)
            for m in range(1, 13):
                rows.append((domain, code, m, res["grid"].get(m),
                             res["baseline_start"], res["baseline_end"],
                             res["flag"], res["n_years"]))
        upsert_reference_stats(pg, rows)
        total += len(rows)
        context.log.info("%s: reference grid for %d station-months", domain, len(rows))
    context.add_output_metadata({"rows_written": MetadataValue.int(total)})
    return total
```

- [ ] **Step 2 : Enregistrer l'asset** dans le `Definitions` (même endroit que `station_current_index`). Ajouter l'import et l'inclure dans `load_assets_from_modules`/la liste d'assets selon le pattern du repo.

- [ ] **Step 3 : Commit**

```bash
git add src/hubeau_pipeline/assets/reference_stats_assets.py <def_file>
git commit -m "feat(assets): station_reference_stats dagster asset"
```

### Task W4 : Reclasser `station_current_index` contre la grille fixe

**Files:**
- Modify: `~/hubeau_data_integration/src/hubeau_pipeline/assets/current_index_assets.py`

- [ ] **Step 1 : Remplacer le calcul plein-historique par la grille fixe**

Remplacer le corps de l'asset pour : calculer la grille via `compute_reference_grid`, puis classer le dernier mois avec `grid_to_zscore`, et écrire `baseline_start/end` issus de la grille. Nouveau contenu :

```python
"""Nightly per-station standardized-index classification → gold.station_current_index (fixed reference)."""
import logging

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from ..ml.indices import compute_reference_grid, grid_to_zscore, classify_value
from ..ml.current_index_persistence import init_current_index_table, upsert_current_index
from ..resources import PostgreSQLResource

logger = logging.getLogger(__name__)

_DOMAINS = [
    ("piezo", "gold.fct_monthly_chroniques", "code_bss", "niveau_moyen", "IPS", False),
    ("hydro", "gold.fct_monthly_hydro", "code_station", "resultat_moyen", "SSFI", True),
]


@asset(
    name="station_current_index",
    group_name="indices",
    deps=["station_reference_stats"],
    description="Latest standardized index (IPS/SSFI) classified against the fixed reference grid.",
)
def station_current_index(context: AssetExecutionContext, pg: PostgreSQLResource):
    init_current_index_table(pg)
    total = 0
    for domain, table, code_col, value_col, index_name, positive_only in _DOMAINS:
        with pg.get_connection() as conn:
            df = pd.read_sql(
                f"SELECT {code_col} AS code, mois, {value_col} AS val "
                f"FROM {table} WHERE {value_col} IS NOT NULL "
                f"AND {value_col} < 1e8 AND {value_col} > -1e4 "
                f"ORDER BY {code_col}, mois",
                conn,
            )
        rows = []
        for code, g in df.groupby("code"):
            months = g["mois"].astype(str).tolist()
            values = g["val"].astype(float).tolist()
            res = compute_reference_grid(months, values, positive_only=positive_only)
            last_dt = pd.to_datetime(months[-1])
            last_val = float(values[-1])
            z = grid_to_zscore(last_val, res["grid"].get(last_dt.month))
            cls = classify_value(z) if z is not None else "UNKNOWN"
            rows.append((code, domain, index_name, z, cls,
                         last_dt.date(), res["baseline_start"], res["baseline_end"]))
        upsert_current_index(pg, rows)
        total += len(rows)
        context.log.info("%s: classified %d stations (fixed ref)", domain, len(rows))
    context.add_output_metadata({"stations_classified": MetadataValue.int(total)})
    return total
```

> `baseline_start/end` deviennent des `str` ISO (issus de la grille) ; `upsert_current_index` les passe tels quels à une colonne `date` — Postgres caste « YYYY-MM-DD » sans souci.

- [ ] **Step 2 : Commit**

```bash
git add src/hubeau_pipeline/assets/current_index_assets.py
git commit -m "feat(assets): current_index classified against fixed reference grid"
```

---

## PARTIE APP (`~/time-serie-explo`)

### Task A1 : Module de lecture `dashboard/utils/reference.py` (TDD)

**Files:**
- Create: `dashboard/utils/reference.py`
- Test: `tests/test_reference.py`

- [ ] **Step 1 : Tests (échouent)**

```python
import numpy as np
from dashboard.utils.reference import series_zscores, class_bounds_ngf, CLASS_CUTOFF_PCTL

# synthetic ascending grid 1..99 -> values 1.0..99.0
_GRID = [float(p) for p in range(1, 100)]


def test_class_bounds_ngf_matches_cutoffs():
    bounds = class_bounds_ngf(_GRID)
    assert len(bounds) == len(CLASS_CUTOFF_PCTL)
    # ascending
    assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))
    # ~ equals the cutoff percentile value on a linear grid
    assert abs(bounds[0] - CLASS_CUTOFF_PCTL[0]) < 0.5


def test_series_zscores_sign():
    grid_by_month = {m: _GRID for m in range(1, 13)}
    z = series_zscores([("2020-06-15", 5.0), ("2020-07-15", 95.0)], grid_by_month)
    assert z[0] < 0 < z[1]


def test_series_zscores_handles_missing_month_grid():
    z = series_zscores([("2020-06-15", 50.0)], {6: None})
    assert z == [None]
```

- [ ] **Step 2 : Lancer, échec** — `docker compose exec backend python -m pytest /app/tests/test_reference.py -v` (tests pas dans l'image : copier comme en Partie B, ou installer pytest). Expected FAIL.

- [ ] **Step 3 : Implémenter `dashboard/utils/reference.py`**

```python
"""Read fixed-reference IPS/SSFI grids from gold.station_reference_stats and apply them.

The warehouse stores, per (type, code, month), a 99-point empirical percentile grid over
a fixed reference window. This module turns grids into z-scores and class bounds. Pure
functions here; the DB read lives in the routers (which already own engine/session).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

PCTL_GRID = list(range(1, 100))
# 7-class BRGM cutoffs as CDF percentiles
CLASS_CUTOFF_PCTL = [4.01, 10.03, 20.05, 79.95, 89.97, 95.99]


def value_to_zscore(value, grid_month):
    if grid_month is None or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    pct = float(np.interp(value, grid_month, PCTL_GRID)) / 100.0
    pct = float(np.clip(pct, 0.001, 0.999))
    return round(float(stats.norm.ppf(pct)), 3)


def series_zscores(dated_values, grid_by_month):
    """dated_values: list of (iso_date, value); grid_by_month: {month:int -> grid|None}."""
    import pandas as pd
    out = []
    for d, v in dated_values:
        m = pd.to_datetime(d).month
        out.append(value_to_zscore(v, grid_by_month.get(m)))
    return out


def class_bounds_ngf(grid_month):
    """6 class-boundary values (physical units) at the BRGM cutoffs, ascending."""
    if grid_month is None:
        return None
    return [float(np.interp(c, PCTL_GRID, grid_month)) for c in CLASS_CUTOFF_PCTL]
```

- [ ] **Step 4 : Lancer, succès** ; **Step 5 : Commit**

```bash
git add dashboard/utils/reference.py tests/test_reference.py
git commit -m "feat(reference): app-side grid reader (zscores + class bounds)"
```

### Task A2 : `/spli` (piezo) et `/ssfi` (hydro) lisent la grille

**Files:**
- Modify: `api/routers/observatory_piezo.py` (`get_spli`)
- Modify: `api/routers/observatory_hydro.py` (`get_ssfi`)

- [ ] **Step 1 : piezo `/spli`** — remplacer `compute_spli(months, values)` par une lecture de `gold.station_reference_stats` (type='piezo') puis `series_zscores`. Construire la liste `{mois, value, spli, classification}` (classification via les 7 classes appliquées au z). Charger la grille :

```python
# in get_spli fetch(), after loading monthly rows:
ref_rows = conn.execute(text(
    "SELECT month, quantile_grid, flag, baseline_start, baseline_end "
    "FROM gold.station_reference_stats WHERE type='piezo' AND code=:code"),
    {"code": code_bss}).mappings().all()
grid_by_month = {r["month"]: r["quantile_grid"] for r in ref_rows}
```
puis calculer le z par mois avec `dashboard.utils.reference.value_to_zscore` et classifier (réutiliser `_classify`/`_THRESHOLDS_7` côté app — déjà dans `drought.py`, l'importer ou recopier le mapping). Si `grid_by_month` vide → renvoyer `[]` (pas de référence calculée).

- [ ] **Step 2 : hydro `/ssfi`** — idem avec `type='hydro'`, `positive_only` géré côté warehouse.

- [ ] **Step 3 : Vérif unitaire des helpers** (intégration différée post-matérialisation). Commit :

```bash
git add api/routers/observatory_piezo.py api/routers/observatory_hydro.py
git commit -m "feat(observatory): SPLI/SSFI series from fixed reference grid"
```

### Task A3 : Fiche station lit la grille (remplace la LATERAL)

**Files:**
- Modify: `api/routers/observatory_piezo.py` (`get_station`) + `api/routers/observatory_hydro.py` (`get_station`)
- Modify: `api/schemas/observatory.py` (champs exposés)

- [ ] **Step 1 : Schéma** — ajouter à `PiezoStation`/`HydroStation` : `reference_flag: str | None`, `index_baseline_start/end` déjà présents, `index_class_bounds: list[float] | None` (bornes m NGF au mois de référence).

- [ ] **Step 2 : Requête** — remplacer la sous-requête `LATERAL` percentile par une lecture de `station_reference_stats` au mois de référence (`index_ref_month`) ; calculer `class_bounds_ngf(grid)` ; exposer `flag`.

- [ ] **Step 3 : Commit**

```bash
git add api/routers/observatory_piezo.py api/routers/observatory_hydro.py api/schemas/observatory.py
git commit -m "feat(observatory): station detail bounds from fixed reference grid"
```

### Task A4 : Bandes de prévision depuis la grille (arrêt du gel au train)

**Files:**
- Modify: `api/routers/counterfactual.py` (endpoints `ips_reference`, bandes)
- Modify: `dashboard/utils/training.py` (retirer la sauvegarde `ips_meta`)
- Modify: `dashboard/utils/counterfactual/ips.py` (retirer `compute_ips_reference*`, garder classification/bandes consommant une grille)

- [ ] **Step 1 : `counterfactual.py`** — au lieu de lire `ips_ref` figé dans l'artefact, charger la grille de la station via `code_bss` (= `entry.station_name`) depuis `gold.station_reference_stats` et construire les bandes avec `class_bounds_ngf` par mois. Supprimer la dépendance à `compute_ips_reference_n`.
- [ ] **Step 2 : `training.py`** — supprimer le bloc « 7. Compute and save IPS reference stats » (lignes ~928-980) : plus de gel `ips_meta` au train.
- [ ] **Step 3 : `ips.py`** — retirer `compute_ips_reference`, `compute_ips_reference_n`, `compute_all_ips_references` (et imports/exports `__init__`) ; conserver `IPS_CLASSES`/`compute_monthly_ips_bounds` adaptés à une grille, ou les remplacer par `class_bounds_ngf`. Mettre à jour `dashboard/utils/counterfactual/__init__.py`.
- [ ] **Step 4 : Vérifier qu'aucun import cassé** : `docker compose exec backend python -c "import api.routers.counterfactual, dashboard.utils.training"`.
- [ ] **Step 5 : Commit**

```bash
git add api/routers/counterfactual.py dashboard/utils/training.py dashboard/utils/counterfactual/
git commit -m "feat(forecast): IPS bands from shared fixed reference (drop train-time freeze)"
```

### Task A5 : Transparence UI (plages m NGF + période)

**Files:**
- Modify: frontend `components/observatory/SituationPanel.tsx` (ou le composant de l'échelle IPS) + types `lib/observatory-types.ts`
- Modify: `frontend/src/i18n/locales/{fr,en}.json`

- [ ] **Step 1 : Types** — ajouter `reference_flag`, `index_class_bounds` à `PiezoStation`/`HydroStation`.
- [ ] **Step 2 : UI** — sous l'échelle IPS : un déroulant/légende montrant chaque classe avec ses bornes en m NGF (depuis `index_class_bounds`) ; tooltip « Référence : 1991–2020 (30 ans) / période adaptée (AAAA–AAAA) / provisoire — série < 15 ans » d'après `reference_flag` + `index_baseline_start/end`.
- [ ] **Step 3 : i18n** — clés `observatory.reference.{normale,adaptee,provisoire,title}`.
- [ ] **Step 4 : `tsc --noEmit` 0 erreur** ; **Step 5 : Commit + rebuild front**.

---

## Matérialisation (étape opérateur, après W1–W4)

> Opération **prod** (écrit dans brgm-postgres). À lancer délibérément.

```bash
# depuis ~/hubeau_data_integration, avec l'env dagster du repo
dagster asset materialize --select station_reference_stats -m hubeau_pipeline
dagster asset materialize --select station_current_index   -m hubeau_pipeline
# vérifier
psql ... -c "SELECT type, flag, count(*) FROM gold.station_reference_stats GROUP BY 1,2;"
```

Puis, côté app : `docker compose up -d --build backend frontend` et re-vérifier `/spli`, fiche station (flag + bornes), bandes de prévision, cohérence Observatoire ↔ Prévision.

---

## Self-Review

- **Couverture spec A** : module unique → W1 ; table gold → W2 ; asset → W3 ; current_index reclasse → W4 ; app lit (spli/ssfi) → A2 ; fiche station → A3 ; prévision partagée + arrêt gel → A4 ; transparence plages m NGF + période → A5. ✓
- **Méthode empirique unifiée** : `np.percentile` warehouse + interpolation app, SPLI & SSFI même chemin. ✓
- **Placeholders** : les tâches A2/A3/A4 décrivent des intégrations DB non testables hors matérialisation ; le code des helpers purs (W1, A1) est complet et TDD. Les sections d'intégration référencent des fonctions définies (W1/A1) — pas d'invention. ⚠ Les modifs SQL exactes des routers sont décrites précisément mais devront suivre le style en place (déjà audité en Partie B).
- **Cohérence types** : `compute_reference_grid` (warehouse) et `value_to_zscore`/`class_bounds_ngf` (app) partagent `PCTL_GRID`/`CLASS_CUTOFF_PCTL` identiques. ✓
- **Cadence** : `REF_PERIOD` constante ; recalcul = changer la constante + rematérialiser. ✓
