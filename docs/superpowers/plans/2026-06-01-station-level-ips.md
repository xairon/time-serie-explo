# Station Level/Alert Redesign (IPS/SSFI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the confusing 5-class-yearly level + misleading "Alerte" trend with a single standardized, 7-class current-level indicator (IPS for groundwater, SSFI for rivers), explained with fixed tooltips, consistent across markers/legend/stats/detail.

**Architecture:** A nightly Python Dagster asset in the BRGM pipeline (`hubeau_data_integration`) computes each station's latest standardized index + 7-class from the monthly facts and writes `gold.station_current_index`. The API (`time-serie-explo`) LEFT JOINs that table at query time and exposes the class under the existing `classification` property, so map markers/legend pick it up with no map change. A new sober `SituationPanel` React component renders the class, a 7-level scale, the index value, the raw measure (m NGF / m³/s), the reference month, and fixed-text tooltips (IPS, SSFI, NGF). The "Alerte" box is removed.

**Tech Stack:** Dagster + psycopg2 + scipy (BRGM), FastAPI + SQLAlchemy (API), React + MapLibre + i18next (frontend). Two repos: `/home/ringuet/hubeau_data_integration` (BRGM) and `/home/ringuet/time-serie-explo` (observatory).

**Reference implementation of the index math:** `time-serie-explo/dashboard/utils/drought.py` (`classify_latest_spli`, `classify_latest_ssfi`, `_THRESHOLDS_7`, `_classify`, `MIN_MONTHS=60`, `MIN_PER_MONTH=10`).

**Note on testing:** this codebase has no JS unit tests and no Python unit tests except where added; data quality is via dbt tests. Phase 1 adds a real pytest for the pure index math; other phases verify via dbt run/test, `tsc`, `vite build`, and `curl`.

---

## Phase 1 — Data foundation (repo: `/home/ringuet/hubeau_data_integration`)

### Task 1: Mirror the standardized-index math into the BRGM pipeline

**Files:**
- Create: `src/hubeau_pipeline/ml/indices.py`
- Test: `tests/test_indices.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indices.py
import numpy as np
from hubeau_pipeline.ml.indices import classify_latest_spli, classify_latest_ssfi, classify_value

def test_classify_value_thresholds():
    assert classify_value(0.0) == "NORMAL"
    assert classify_value(-1.0) == "BAS"
    assert classify_value(-1.5) == "TRES_BAS"
    assert classify_value(-2.0) == "EXTREMEMENT_BAS"
    assert classify_value(2.0) == "EXTREMEMENT_HAUT"
    assert classify_value(None) == "UNKNOWN"

def test_spli_too_short_is_unknown():
    months = [f"2020-{m:02d}-01" for m in range(1, 13)]  # 12 < 60
    z, cls = classify_latest_spli(months, [1.0] * 12)
    assert z is None and cls == "UNKNOWN"

def test_spli_returns_class_on_long_series():
    # 10 years monthly, last value clearly low for its calendar month
    months, values = [], []
    for y in range(2010, 2020):
        for m in range(1, 13):
            months.append(f"{y}-{m:02d}-01")
            values.append(10.0 + m + np.random.default_rng(y * 12 + m).normal(0, 0.5))
    months.append("2020-06-01"); values.append(0.0)  # very low June
    z, cls = classify_latest_spli(months, values)
    assert z is not None and z < 0 and cls in ("EXTREMEMENT_BAS", "TRES_BAS", "BAS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec -w /app brgm-dlt-worker python -m pytest tests/test_indices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hubeau_pipeline.ml.indices'`

- [ ] **Step 3: Create the module (port from drought.py — pure scipy/pandas, no I/O)**

```python
# src/hubeau_pipeline/ml/indices.py
"""Standardized hydrological indices (latest-month classification).

Mirrors time-serie-explo/dashboard/utils/drought.py (BRGM IPS/Meteo-France
methodology). IPS/SPLI = KDE->normal for groundwater levels; SSFI = gamma->normal
for river streamflow. 7 classes from the standardized value z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_THRESHOLDS_7 = [
    (-float("inf"), -1.75, "EXTREMEMENT_BAS"),
    (-1.75, -1.28, "TRES_BAS"),
    (-1.28, -0.84, "BAS"),
    (-0.84, 0.84, "NORMAL"),
    (0.84, 1.28, "HAUT"),
    (1.28, 1.75, "TRES_HAUT"),
    (1.75, float("inf"), "EXTREMEMENT_HAUT"),
]
MIN_MONTHS = 60       # 5 years minimum
MIN_PER_MONTH = 10    # min observations per calendar month for fitting


def classify_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    for lo, hi, label in _THRESHOLDS_7:
        if lo <= value < hi:
            return label
    return "EXTREMEMENT_HAUT"


def classify_latest_spli(months: list[str], values: list[float]) -> tuple[float | None, str]:
    """IPS/SPLI for the most recent month (KDE -> standard normal)."""
    if len(months) < MIN_MONTHS:
        return None, "UNKNOWN"
    series = pd.Series(values, index=pd.to_datetime(months), dtype=float).dropna()
    if len(series) < MIN_MONTHS:
        return None, "UNKNOWN"
    grouped = series.groupby(series.index.month)
    last_month = series.index[-1].month
    last_val = float(series.iloc[-1])
    if last_month not in grouped.groups or len(grouped.get_group(last_month)) < MIN_PER_MONTH:
        return None, "UNKNOWN"
    try:
        kde = stats.gaussian_kde(grouped.get_group(last_month).values)
    except Exception:
        return None, "UNKNOWN"
    cdf_val = float(np.clip(kde.integrate_box_1d(-np.inf, last_val), 0.001, 0.999))
    z = round(float(stats.norm.ppf(cdf_val)), 3)
    return z, classify_value(z)


def classify_latest_ssfi(months: list[str], values: list[float]) -> tuple[float | None, str]:
    """SSFI for the most recent month (gamma -> standard normal)."""
    if len(months) < MIN_MONTHS:
        return None, "UNKNOWN"
    series = pd.Series(values, index=pd.to_datetime(months), dtype=float).dropna()
    valid = series[series > 0]
    if len(valid) < MIN_MONTHS:
        return None, "UNKNOWN"
    grouped = valid.groupby(valid.index.month)
    last_month = valid.index[-1].month
    last_val = float(valid.iloc[-1])
    if last_month not in grouped.groups or len(grouped.get_group(last_month)) < MIN_PER_MONTH:
        return None, "UNKNOWN"
    group = grouped.get_group(last_month)
    try:
        a, loc, scale = stats.gamma.fit(group.values, floc=0)
        cdf_val = stats.gamma.cdf(last_val, a, loc=loc, scale=scale)
    except Exception:
        return None, "UNKNOWN"
    cdf_val = float(np.clip(cdf_val, 0.001, 0.999))
    z = round(float(stats.norm.ppf(cdf_val)), 3)
    return z, classify_value(z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec -w /app brgm-dlt-worker python -m pytest tests/test_indices.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /home/ringuet/hubeau_data_integration add src/hubeau_pipeline/ml/indices.py tests/test_indices.py
git -C /home/ringuet/hubeau_data_integration commit -m "feat(indices): port IPS/SSFI latest-month classification into pipeline"
```

---

### Task 2: `gold.station_current_index` table DDL + upsert helper

**Files:**
- Create: `src/hubeau_pipeline/ml/current_index_persistence.py`

- [ ] **Step 1: Create the persistence module**

```python
# src/hubeau_pipeline/ml/current_index_persistence.py
"""Create + upsert gold.station_current_index (per-station latest standardized index)."""

_CREATE = """
CREATE TABLE IF NOT EXISTS gold.station_current_index (
    code            text NOT NULL,
    type            text NOT NULL,          -- 'piezo' | 'hydro'
    index_name      text NOT NULL,          -- 'IPS' | 'SSFI'
    index_value     double precision,       -- standardized z (NULL if UNKNOWN)
    index_class     text NOT NULL,          -- 7 classes or 'UNKNOWN'
    ref_month       date,
    baseline_start  date,
    baseline_end    date,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (type, code)
);
CREATE INDEX IF NOT EXISTS idx_station_current_index_class
    ON gold.station_current_index (index_class);
"""

_UPSERT = """
INSERT INTO gold.station_current_index
    (code, type, index_name, index_value, index_class, ref_month, baseline_start, baseline_end, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (type, code) DO UPDATE SET
    index_name = EXCLUDED.index_name,
    index_value = EXCLUDED.index_value,
    index_class = EXCLUDED.index_class,
    ref_month = EXCLUDED.ref_month,
    baseline_start = EXCLUDED.baseline_start,
    baseline_end = EXCLUDED.baseline_end,
    computed_at = now();
"""


def init_current_index_table(pg):
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS gold")
        cur.execute(_CREATE)
        conn.commit()


def upsert_current_index(pg, rows: list[tuple]):
    """rows: (code, type, index_name, index_value|None, index_class, ref_month, baseline_start, baseline_end)."""
    if not rows:
        return
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(_UPSERT, rows)
        conn.commit()
```

- [ ] **Step 2: Commit**

```bash
git -C /home/ringuet/hubeau_data_integration add src/hubeau_pipeline/ml/current_index_persistence.py
git -C /home/ringuet/hubeau_data_integration commit -m "feat(indices): gold.station_current_index table + upsert helper"
```

---

### Task 3: Dagster asset computing the per-station current index

**Files:**
- Create: `src/hubeau_pipeline/assets/current_index_assets.py`
- Modify: `src/hubeau_pipeline/definitions.py` (register new assets)

- [ ] **Step 1: Write the asset**

```python
# src/hubeau_pipeline/assets/current_index_assets.py
"""Nightly per-station standardized-index (IPS/SSFI) classification → gold.station_current_index."""
import logging

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from ..ml.indices import classify_latest_spli, classify_latest_ssfi
from ..ml.current_index_persistence import init_current_index_table, upsert_current_index
from ..resources import PostgreSQLResource

logger = logging.getLogger(__name__)

# (domain, table, code_col, value_col, index_name, classify_fn)
_DOMAINS = [
    ("piezo", "gold.fct_monthly_chroniques", "code_bss", "niveau_moyen", "IPS", classify_latest_spli),
    ("hydro", "gold.fct_monthly_hydro", "code_station", "resultat_moyen", "SSFI", classify_latest_ssfi),
]


@asset(
    name="station_current_index",
    group_name="indices",
    description="Latest standardized index (IPS/SSFI) + 7-class per station, written to gold.station_current_index.",
)
def station_current_index(context: AssetExecutionContext, pg: PostgreSQLResource):
    init_current_index_table(pg)
    total = 0
    for domain, table, code_col, value_col, index_name, classify_fn in _DOMAINS:
        with pg.get_connection() as conn:
            df = pd.read_sql(
                f"SELECT {code_col} AS code, mois, {value_col} AS val "
                f"FROM {table} WHERE {value_col} IS NOT NULL ORDER BY {code_col}, mois",
                conn,
            )
        rows = []
        for code, g in df.groupby("code"):
            months = g["mois"].astype(str).tolist()
            values = g["val"].astype(float).tolist()
            z, cls = classify_fn(months, values)
            ref_month = pd.to_datetime(months[-1]).date()
            rows.append((code, domain, index_name, z, cls,
                         ref_month, pd.to_datetime(months[0]).date(), ref_month))
        upsert_current_index(pg, rows)
        total += len(rows)
        context.log.info("%s: classified %d stations", domain, len(rows))
    context.add_output_metadata({"stations_classified": MetadataValue.int(total)})
    return total
```

- [ ] **Step 2: Register the asset in definitions.py**

In `src/hubeau_pipeline/definitions.py`, import and add to the assets list:

```python
from .assets.current_index_assets import station_current_index
# ... in Definitions(assets=[...]) add: station_current_index
```

Run: `grep -n "all_assets\|assets=" src/hubeau_pipeline/definitions.py` to find the exact assets list, then append `station_current_index` to it.

- [ ] **Step 3: Verify it loads**

Run:
```bash
docker compose restart dlt_worker && sleep 30
docker exec -w /app brgm-dlt-worker python -c "from hubeau_pipeline.definitions import defs; print('station_current_index' in [a.key.to_user_string() for a in defs.get_asset_graph().all_asset_keys.__iter__().__class__ and []] or 'ok')"
```
Simpler verification: `docker exec -w /app brgm-dlt-worker python -c "from hubeau_pipeline.assets.current_index_assets import station_current_index; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Commit**

```bash
git -C /home/ringuet/hubeau_data_integration add src/hubeau_pipeline/assets/current_index_assets.py src/hubeau_pipeline/definitions.py
git -C /home/ringuet/hubeau_data_integration commit -m "feat(indices): nightly station_current_index Dagster asset"
```

---

### Task 4: Wire the asset into the nightly chain (after dbt_daily_transform)

**Files:**
- Modify: `src/hubeau_pipeline/jobs/dbt_jobs.py` (add `station_current_index_job`)
- Modify: `src/hubeau_pipeline/jobs/__init__.py` (export it)
- Modify: `src/hubeau_pipeline/sensors.py` (Step 3 sensor: daily transform done → run index job)

- [ ] **Step 1: Add the job** (in `jobs/dbt_jobs.py`, after `dbt_daily_transform_job`)

```python
from dagster import AssetSelection

station_current_index_job = define_asset_job(
    name="station_current_index",
    description="Compute per-station current standardized index (IPS/SSFI) after the daily transform.",
    selection=AssetSelection.assets("station_current_index"),
    tags={"dagster/concurrency_key": "dbt_pipeline"},
)
```

- [ ] **Step 2: Export it** in `jobs/__init__.py` (add to imports from `.dbt_jobs`, to `all_jobs`, and to `__all__`), mirroring `dbt_daily_transform_job`.

- [ ] **Step 3: Add Step 3 sensor** in `sensors.py`

Add import: `from .jobs import dbt_daily_transform_job, dbt_shared_staging_job, station_current_index_job`

```python
@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[dbt_daily_transform_job],
    request_jobs=[station_current_index_job],
    default_status=DEFAULT_SENSOR_STATUS,
    minimum_interval_seconds=30,
    description="Step 3/3: daily transform done → compute current standardized index",
)
def transform_to_index_sensor(context: RunStatusSensorContext):
    yield RunRequest(
        run_key=f"current_index_{context.dagster_run.run_id}",
        tags={"trigger": "sensor", "sensor_name": "transform_to_index_sensor", "pipeline_chain": "step_3_index"},
    )
```

Add `transform_to_index_sensor` to `all_sensors`.

- [ ] **Step 4: Verify definitions load** (2 sensors → 3, jobs +1)

Run: `docker compose restart dlt_worker && sleep 30 && docker exec -w /app brgm-dlt-worker python -c "from hubeau_pipeline.definitions import defs; print('sensors', [s.name for s in defs.sensors])"`
Expected: includes `transform_to_index_sensor`.

- [ ] **Step 5: Reload code location**

Run:
```bash
docker exec brgm-dagster-webserver python -c "import urllib.request,json; q={'query':'mutation{reloadRepositoryLocation(repositoryLocationName:\"hubeau_pipeline\"){__typename}}'}; print(urllib.request.urlopen(urllib.request.Request('http://localhost:3000/graphql',data=json.dumps(q).encode(),headers={'Content-Type':'application/json'})).read().decode())"
```
Expected: `WorkspaceLocationEntry` / no error.

- [ ] **Step 6: Commit**

```bash
git -C /home/ringuet/hubeau_data_integration add src/hubeau_pipeline/jobs/dbt_jobs.py src/hubeau_pipeline/jobs/__init__.py src/hubeau_pipeline/sensors.py
git -C /home/ringuet/hubeau_data_integration commit -m "feat(indices): run station_current_index nightly after daily transform"
```

---

### Task 5: One-time populate + verify

- [ ] **Step 1: Materialize the asset once**

Run:
```bash
docker exec -w /app brgm-dlt-worker python -c "
from hubeau_pipeline.resources import PostgreSQLResource
from hubeau_pipeline.assets.current_index_assets import _DOMAINS
from hubeau_pipeline.ml.current_index_persistence import init_current_index_table, upsert_current_index
import pandas as pd
pg = PostgreSQLResource()  # uses env DAGSTER_PG_* / default; if it needs args, run via Dagster UI instead
"
```
If `PostgreSQLResource()` needs configuration, instead trigger the `station_current_index` job from the Dagster UI (http://localhost:49500 → Jobs → station_current_index → Materialize), OR via GraphQL launchRun. Document whichever worked.

- [ ] **Step 2: Verify the table is populated and sane**

Run:
```bash
docker exec brgm-postgres psql -U postgres -d postgres -c "
SELECT type, index_class, count(*) FROM gold.station_current_index GROUP BY 1,2 ORDER BY 1,2;
SELECT count(*) FILTER (WHERE index_class='UNKNOWN') AS unknown, count(*) AS total FROM gold.station_current_index;"
```
Expected: piezo + hydro rows across the 7 classes + UNKNOWN; total ≈ 22400 piezo + 6250 hydro.

- [ ] **Step 3: Commit** (nothing to commit — data only; note results in the PR description)

---

## Phase 2 — API (repo: `/home/ringuet/time-serie-explo`)

> The API maps `index_class` onto the existing `classification` property name the frontend already
> consumes, so map markers + legend pick up the 7-class scheme with no map-layer change.

### Task 6: GeoJSON markers use the standardized class

**Files:**
- Modify: `api/routers/observatory_common.py` (`get_stations_geojson`, both piezo + hydro queries)

- [ ] **Step 1: Add the JOIN + select the index class** — in the piezo query, replace
`classification_derniere_annee AS classification` with the joined class and add the index value:

```sql
-- piezo SELECT ... FROM gold.dim_piezo_stations s
       COALESCE(sci.index_class, 'UNKNOWN') AS classification,
       sci.index_value
FROM gold.dim_piezo_stations s
LEFT JOIN gold.station_current_index sci ON sci.type = 'piezo' AND sci.code = s.code_bss
WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
```
Do the equivalent for hydro (`sci.type='hydro' AND sci.code = s.code_station`). Keep `code AS code` etc. unchanged; just swap the classification source and add `index_value` to the `properties` dict.

- [ ] **Step 2: Verify via curl** (after restarting backend in Task 9's build, or restart now)

Run: `curl -s "http://localhost:49513/api/v1/observatory/stations/geojson?type=piezo" | python3 -c "import sys,json,collections;d=json.load(sys.stdin);print(collections.Counter(f['properties']['classification'] for f in d['features']).most_common())"`
Expected: distribution across EXTREMEMENT_BAS…EXTREMEMENT_HAUT + UNKNOWN (7 classes now present, not just 5).

- [ ] **Step 3: Commit**

```bash
git -C /home/ringuet/time-serie-explo add api/routers/observatory_common.py
git -C /home/ringuet/time-serie-explo commit -m "feat(api): geojson classification from standardized current index"
```

### Task 7: National stats counted by standardized class

**Files:** Modify `api/routers/observatory_common.py` (`get_national_stats`)

- [ ] **Step 1:** Change the `piezo`/`hydro` CTEs to count over `gold.station_current_index` joined to dim (or directly): replace `classification_derniere_annee` filters with `sci.index_class`, joining `gold.station_current_index sci ON sci.type='piezo' AND sci.code = code_bss`. Keep the `recent_cutoff` (derniere_mesure ≥ today-`ACTIVE_STATION_DAYS`) filter. The 7 class buckets already exist in the SELECT.
- [ ] **Step 2: Verify** `curl -s http://localhost:49513/api/v1/observatory/stats/national | python3 -m json.tool` shows non-zero counts including `*_extremement_bas/haut`.
- [ ] **Step 3: Commit** `git -C /home/ringuet/time-serie-explo commit -am "feat(api): national stats from standardized index"`

### Task 8: Alerts severity by standardized class

**Files:** Modify `api/routers/observatory_common.py` (`list_alerts`)

- [ ] **Step 1:** In both piezo + hydro UNION parts, replace `s.classification_derniere_annee = ANY(:severity)` / `s.classification_resultat_dern_annee = ANY(:severity)` with `sci.index_class = ANY(:severity)`, adding `LEFT JOIN gold.station_current_index sci ON sci.type='piezo' AND sci.code = s.code_bss` (and hydro variant). Select `sci.index_class AS classification`. Leave the `cs.alerte_depuis_annee` / `nb_annees_consecutives` LATERAL subqueries (years-consecutive logic) unchanged per spec (option a).
- [ ] **Step 2: Verify** `curl -s "http://localhost:49513/api/v1/observatory/alerts?type=piezo" | python3 -c "import sys,json;print(len(json.load(sys.stdin)))"` returns a plausible count.
- [ ] **Step 3: Commit** `git -C /home/ringuet/time-serie-explo commit -am "feat(api): alerts severity from standardized index"`

### Task 9: Station detail exposes the index

**Files:** Modify `api/routers/observatory_piezo.py` (`get_station`), `api/routers/observatory_hydro.py` (`get_station`); `api/schemas/observatory.py` (add fields to `PiezoStation` / `HydroStation`)

- [ ] **Step 1: Add response fields** in `api/schemas/observatory.py` to both `PiezoStation` and `HydroStation`:

```python
    index_name: str | None = None      # 'IPS' | 'SSFI'
    index_value: float | None = None
    index_class: str | None = None      # 7 classes | 'UNKNOWN'
    index_ref_month: date | None = None
    index_baseline_start: date | None = None
    index_baseline_end: date | None = None
```
(Ensure `from datetime import date` is imported in that file.)

- [ ] **Step 2: Add JOIN + columns** in `observatory_piezo.py get_station` SQL:

```sql
SELECT s.*, sci.index_name, sci.index_value, sci.index_class,
       sci.ref_month AS index_ref_month, sci.baseline_start AS index_baseline_start,
       sci.baseline_end AS index_baseline_end
FROM gold.dim_piezo_stations s
LEFT JOIN gold.station_current_index sci ON sci.type='piezo' AND sci.code = s.code_bss
WHERE s.code_bss = :code
```
(Replace the explicit column list with `s.*` + the joined columns, OR append the joined columns to the existing list.) Do the hydro equivalent (`sci.type='hydro'`, `s.code_station`).

- [ ] **Step 3: Verify**

Run: `curl -s "http://localhost:49513/api/v1/observatory/piezo/stations/04285X0016/P" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['index_name'],d['index_value'],d['index_class'],d['index_ref_month'])"`
Expected: e.g. `IPS -0.97 BAS 2026-05-01` (values vary).

- [ ] **Step 4: Rebuild + restart backend, then re-run Step 3**

```bash
cd /home/ringuet/time-serie-explo && docker compose -f docker-compose.yml -f docker-compose.cuda.yml build backend && docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d backend
# flush stale API cache (correct redis is inside the backend container):
docker exec junon-backend python3 -c "import redis;r=redis.Redis(host='redis',port=6379,decode_responses=True);[r.delete(k) for k in r.scan_iter(match='junon:obs_*',count=500)]"
```

- [ ] **Step 5: Commit**

```bash
git -C /home/ringuet/time-serie-explo add api/routers/observatory_piezo.py api/routers/observatory_hydro.py api/schemas/observatory.py
git -C /home/ringuet/time-serie-explo commit -m "feat(api): expose standardized index on station detail"
```

---

## Phase 3 — Frontend (repo: `/home/ringuet/time-serie-explo`)

### Task 10: i18n keys (fixed tooltip text — no generated prose)

**Files:** Modify `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

- [ ] **Step 1:** Add under the existing `observatory` object in `fr.json`:

```json
"situation": {
  "title": "Niveau de la nappe",
  "titleHydro": "Débit de la rivière",
  "scaleLow": "Très bas",
  "scaleHigh": "Très haut",
  "measure": "Mesure",
  "refMonth": "Mois de référence",
  "baseline": "Période de référence",
  "unclassified": "Non classé — historique insuffisant (< 5 ans)",
  "ipsTip": "Indicateur Piézométrique Standardisé (IPS/SPLI). Compare le niveau de ce mois à tous les mois équivalents passés de cette station, sur une échelle standard (BRGM/Météo-France). 0 = médiane ; négatif = plus bas que d'habitude, positif = plus haut.",
  "ssfiTip": "Indicateur Standardisé d'Écoulement (SSFI). Compare le débit de ce mois à tous les mois équivalents passés de cette station, sur une échelle standard. 0 = médiane ; négatif = plus bas que d'habitude.",
  "ngfTip": "Nivellement Général de la France : altitude de référence nationale (≈ niveau moyen de la mer). Une valeur en m NGF est l'altitude de la surface de la nappe."
}
```

- [ ] **Step 2:** Add the English equivalents under `observatory` in `en.json` (translate the same keys).

- [ ] **Step 3: Commit**

```bash
git -C /home/ringuet/time-serie-explo add frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git -C /home/ringuet/time-serie-explo commit -m "i18n(observatory): situation panel + IPS/SSFI/NGF tooltips"
```

### Task 11: `SituationPanel` component

**Files:** Create `frontend/src/components/observatory/SituationPanel.tsx`

- [ ] **Step 1: Create the component**

```tsx
import { useTranslation } from 'react-i18next'
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS, CLASSIFICATION_ORDER } from '@/lib/observatory-constants'
import { formatNumber, formatDate } from '@/lib/observatory-utils'

interface Props {
  type: 'piezo' | 'hydro'
  indexName?: string | null          // 'IPS' | 'SSFI'
  indexValue?: number | null
  indexClass?: string | null         // 7 classes | 'UNKNOWN' | null
  refMonth?: string | null
  baselineStart?: string | null
  baselineEnd?: string | null
  measure?: number | null            // last value (m NGF / m³/s)
  measureUnit: string
}

function InfoDot({ tip }: { tip: string }) {
  return (
    <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-white/10 text-[9px] text-text-secondary cursor-help align-middle" title={tip} aria-label={tip}>i</span>
  )
}

export function SituationPanel(props: Props) {
  const { t } = useTranslation()
  const isPiezo = props.type === 'piezo'
  const cls = props.indexClass
  const unknown = !cls || cls === 'UNKNOWN'
  const color = (cls && CLASSIFICATION_COLORS[cls]) || '#6b7280'
  const indexTip = isPiezo ? t('observatory.situation.ipsTip') : t('observatory.situation.ssfiTip')

  return (
    <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-wider text-text-secondary">
          {isPiezo ? t('observatory.situation.title') : t('observatory.situation.titleHydro')}
        </span>
        {props.indexName && (
          <span className="text-[10px] text-text-secondary">{props.indexName} <InfoDot tip={indexTip} />
            {props.indexValue != null && <span className="ml-1 font-mono text-text-primary">{props.indexValue.toFixed(2)}</span>}
          </span>
        )}
      </div>

      {unknown ? (
        <div className="text-xs text-text-secondary">{t('observatory.situation.unclassified')}</div>
      ) : (
        <>
          <div className="flex items-center gap-1.5 mb-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-sm font-semibold" style={{ color }}>{CLASSIFICATION_LABELS[cls!] ?? cls}</span>
          </div>
          <div className="flex gap-0.5 mb-1" role="img" aria-label={CLASSIFICATION_LABELS[cls!] ?? cls!}>
            {CLASSIFICATION_ORDER.map(c => (
              <span key={c} className="h-2 flex-1 rounded-sm" style={{
                backgroundColor: CLASSIFICATION_COLORS[c],
                opacity: c === cls ? 1 : 0.25,
                outline: c === cls ? '1px solid rgba(255,255,255,0.8)' : 'none',
              }} />
            ))}
          </div>
          <div className="flex justify-between text-[9px] text-text-secondary mb-2">
            <span>{t('observatory.situation.scaleLow')}</span><span>{t('observatory.situation.scaleHigh')}</span>
          </div>
        </>
      )}

      {props.measure != null && (
        <div className="text-xs text-text-secondary">
          {t('observatory.situation.measure')} : <span className="text-text-primary font-mono">{formatNumber(props.measure, 2)} {props.measureUnit}</span>
          {isPiezo && <> <InfoDot tip={t('observatory.situation.ngfTip')} /></>}
        </div>
      )}
      {props.refMonth && <div className="text-[10px] text-text-secondary mt-1">{formatDate(props.refMonth)}{props.baselineStart && props.baselineEnd && <> · {props.baselineStart.slice(0,4)}–{props.baselineEnd.slice(0,4)}</>}</div>}
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc -b --noEmit 2>&1 | grep SituationPanel | grep -v "Cannot find module"`
Expected: no output (no errors attributable to the new file).

- [ ] **Step 3: Commit**

```bash
git -C /home/ringuet/time-serie-explo add frontend/src/components/observatory/SituationPanel.tsx
git -C /home/ringuet/time-serie-explo commit -m "feat(observatory): sober SituationPanel (IPS/SSFI 7-class + tooltips)"
```

### Task 12: Use SituationPanel in StationDrawer + remove the Alerte box

**Files:** Modify `frontend/src/components/observatory/StationDrawer.tsx`

- [ ] **Step 1: Import** `import { SituationPanel } from './SituationPanel'`.
- [ ] **Step 2: Replace** the `recent ? (current-state block) : (inactive banner)` JSX (the block rendering `ClassificationBadge` + `currentValue`, around lines 86–97) with the inactive banner kept for `!recent`, and for `recent` render:

```tsx
<SituationPanel
  type={type}
  indexName={(station as any).index_name}
  indexValue={(station as any).index_value}
  indexClass={(station as any).index_class}
  refMonth={(station as any).index_ref_month}
  baselineStart={(station as any).index_baseline_start}
  baselineEnd={(station as any).index_baseline_end}
  measure={currentValue}
  measureUnit={unit}
/>
```

- [ ] **Step 3: Remove the Alerte box** — delete the line rendering `s.niveau_alerte` (the red box, around line 125).
- [ ] **Step 4: Typecheck** `cd frontend && npx tsc -b --noEmit 2>&1 | grep StationDrawer | grep -v "Cannot find module"` → no output.
- [ ] **Step 5: Commit** `git -C /home/ringuet/time-serie-explo commit -am "feat(observatory): drawer uses SituationPanel, drop Alerte box"`

### Task 13: Use SituationPanel in StationPage

**Files:** Modify `frontend/src/pages/StationPage.tsx`

- [ ] **Step 1:** Import `SituationPanel`; add it near the station metadata header, passing `index_*` fields from `station`, `measure` = the latest value available on the detail (or omit `measure` if not present on the page), `measureUnit` = `unit` (already computed: `isPiezo ? 'm NGF' : hydroUnit`). Remove any `niveau_alerte` display if present.
- [ ] **Step 2: Typecheck** `npx tsc -b --noEmit 2>&1 | grep StationPage | grep -v "Cannot find module"` → no output.
- [ ] **Step 3: Commit** `git -C /home/ringuet/time-serie-explo commit -am "feat(observatory): station page uses SituationPanel"`

### Task 14: Build, deploy, verify end-to-end

- [ ] **Step 1: Build + recreate frontend**

```bash
cd /home/ringuet/time-serie-explo
docker compose -f docker-compose.yml -f docker-compose.cuda.yml build frontend
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d frontend
```

- [ ] **Step 2: Verify markers now span 7 classes** (legend already matches)

Run: `curl -s "http://localhost:49513/api/v1/observatory/stations/geojson?type=hydro" | python3 -c "import sys,json,collections;d=json.load(sys.stdin);print(collections.Counter(f['properties']['classification'] for f in d['features']).most_common())"`
Expected: 7 classes + UNKNOWN present.

- [ ] **Step 3: Manual check** — open http://localhost:49513, click a piezo station: SituationPanel shows class + 7-level scale + IPS value + m NGF (with ⓘ), ref month; no "Alerte" box; a station with short history shows "Non classé". Click a hydro station: SSFI + m³/s.

- [ ] **Step 4: Final commit (if any tweaks)** and update the spec status to "implemented".

---

## Self-review notes

- **Spec coverage:** IPS/SSFI basis (T1), bulk per-station (T3) → table (T2) → nightly (T4) → populate (T5); markers/legend (T6), national stats (T7), alerts severity (T8, years-consecutive untouched per option a), detail fields (T9); SituationPanel + tooltips + NGF (T10–T11), drawer + Alerte removal (T12), station page (T13), Non classé handling (T11 `unknown` branch), 7-class unification verified (T6/T14). All spec sections covered.
- **Map change:** none needed — API returns `index_class` under the `classification` property the markers already consume; legend already 7-class.
- **Open implementation detail:** Task 5 Step 1 — exact mechanism to first-materialize the asset (direct `PostgreSQLResource()` vs Dagster UI launch) depends on how `PostgreSQLResource` reads config; verify against `resources.py` at execution time and use the Dagster UI path if direct instantiation needs args.
