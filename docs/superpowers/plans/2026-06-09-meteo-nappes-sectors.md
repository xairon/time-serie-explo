# Météo des nappes by hydrogeological sectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the administrative-mesh `/meteo` page with a hydrogeological-sector "météo des nappes" layer folded into the Observatory (BRGM BSH sectors colored by our fixed-reference IPS, with trend arrows), and unify all classification on the fixed reference.

**Architecture:** Cross-repo hybrid. The warehouse (`~/hubeau_data_integration`) materializes `gold.fct_monthly_index` (every month re-scored against the fixed reference grid). The app (`time-serie-explo`) owns cartography: a static `secteurs-bsh.geojson` (geometry + sector_id + tendancy_coord + BDLISA-derived name), a pure point-in-polygon station→sector mapping, a generalized situation aggregator (`level=sector`), two new endpoints, and a new toggleable MapLibre layer in the existing `ObservatoryMap`. The existing timeline endpoint is repointed to `fct_monthly_index` so stations + sectors + slider share one method.

**Tech Stack:** Python/dagster/pandas/scipy (warehouse), FastAPI/SQLAlchemy/pytest (app backend), React/TypeScript/MapLibre GL/vitest (frontend), PostgreSQL (`gold` schema on brgm-postgres).

---

## File Structure

**Warehouse repo (`~/hubeau_data_integration`):**
- Create `src/hubeau_pipeline/ml/monthly_index_persistence.py` — DDL + upsert for `gold.fct_monthly_index`.
- Create `src/hubeau_pipeline/assets/monthly_index_assets.py` — dagster asset `fct_monthly_index`.
- Modify `src/hubeau_pipeline/definitions.py` (or the assets registry) — register the new asset.
- Create `tests/test_monthly_index.py` — re-score parity test.

**App repo (`time-serie-explo`):**
- Create `dashboard/utils/geo_sectors.py` — pure point-in-polygon + dominant-name helpers (no DB, no Streamlit).
- Create `tests/test_geo_sectors.py`, `tests/test_territory_situation_sector.py`.
- Modify `dashboard/utils/territory_situation.py` — (no change to math; reused as-is).
- Create `scripts/build_secteurs_bsh_geojson.py` — one-shot WFS fetch + name baking → `frontend/public/geo/secteurs-bsh.geojson`.
- Create `api/services/sector_mapping.py` — cached station→sector loader (reads geojson + station coords).
- Modify `api/routers/observatory_situation.py` — add `level=sector` + `/situation/sectors` + `/situation/sectors/timeline`.
- Modify `api/routers/observatory_common.py` — repoint `/classifications/timeline` to `fct_monthly_index`.
- Create `frontend/src/lib/sector-arrows.ts` — pure helpers (parse tendancy_coord, fill-color & arrow expressions, class→int).
- Create `frontend/src/lib/sector-arrows.test.ts`.
- Modify `frontend/src/components/observatory/ObservatoryMap.tsx` — sector fill/line/arrow layers.
- Modify `frontend/src/components/observatory/RightDrawer.tsx` — sector layer toggle + legend entry.
- Modify `frontend/src/pages/ObservatoryPage.tsx` — sector state, data hooks, slider wiring, popup.
- Modify `frontend/src/hooks/useObservatory.ts` + `frontend/src/lib/observatory-api.ts` + `observatory-types.ts` — sector situation/timeline query + types.
- Modify `frontend/src/routes.tsx` — `/meteo` → redirect `/observatoire`.
- Delete `frontend/src/pages/MeteoNappesPage.tsx` and `frontend/src/components/meteo/*`.
- Modify `frontend/src/i18n/locales/{fr,en}.json` — sector layer + legend strings; remove `meteo.*` if unused.

---

## Phase 1 — Warehouse: `gold.fct_monthly_index`

> Repo: `~/hubeau_data_integration`, branch `main`. Run `cd ~/hubeau_data_integration` for all Phase 1 steps. The DB is reached via the dagster `PostgreSQLResource` (`pg`), exactly like `station_current_index`.

### Task 1: Persistence module for `fct_monthly_index`

**Files:**
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/ml/monthly_index_persistence.py`
- Test: `~/hubeau_data_integration/tests/test_monthly_index_persistence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monthly_index_persistence.py
from hubeau_pipeline.ml.monthly_index_persistence import _CREATE, _INSERT

def test_ddl_has_composite_pk_on_type_code_month():
    assert "gold.fct_monthly_index" in _CREATE
    assert "PRIMARY KEY (type, code, month)" in _CREATE

def test_insert_targets_fct_monthly_index_with_six_value_columns():
    assert "INSERT INTO gold.fct_monthly_index" in _INSERT
    # code, type, month, z, index_class, flag
    assert _INSERT.count("%s") == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/hubeau_data_integration && python -m pytest tests/test_monthly_index_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hubeau_pipeline.ml.monthly_index_persistence'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hubeau_pipeline/ml/monthly_index_persistence.py
"""Create + replace gold.fct_monthly_index (per-station monthly index, fixed reference)."""

_CREATE = """
CREATE TABLE IF NOT EXISTS gold.fct_monthly_index (
    type        text NOT NULL,
    code        text NOT NULL,
    month       date NOT NULL,
    z           double precision,
    index_class text NOT NULL,
    flag        text,
    computed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (type, code, month)
);
CREATE INDEX IF NOT EXISTS idx_fct_monthly_index_type_month
    ON gold.fct_monthly_index (type, month);
"""

_INSERT = """
INSERT INTO gold.fct_monthly_index (code, type, month, z, index_class, flag, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (type, code, month) DO UPDATE SET
    z = EXCLUDED.z,
    index_class = EXCLUDED.index_class,
    flag = EXCLUDED.flag,
    computed_at = now();
"""


def init_monthly_index_table(pg):
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS gold")
        cur.execute(_CREATE)
        conn.commit()


def upsert_monthly_index(pg, rows):
    """rows: list of (code, type, month_date, z|None, index_class, flag)."""
    if not rows:
        return
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(_INSERT, rows)
        conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/hubeau_data_integration && python -m pytest tests/test_monthly_index_persistence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/hubeau_data_integration
git checkout -b feat/fct-monthly-index
git add src/hubeau_pipeline/ml/monthly_index_persistence.py tests/test_monthly_index_persistence.py
git commit -m "feat(indices): persistence for gold.fct_monthly_index"
```

### Task 2: Re-score helper (parity with fixed reference) + dagster asset

**Files:**
- Create: `~/hubeau_data_integration/src/hubeau_pipeline/ml/monthly_index_assets.py` (asset) — *actually* `assets/monthly_index_assets.py`
- Create: `~/hubeau_data_integration/tests/test_monthly_index.py`
- Modify: the asset registry that lists `station_current_index` (find with grep below)

- [ ] **Step 1: Write the failing test** (re-score parity — the monthly z for the LAST month must equal what `station_current_index` computes)

```python
# tests/test_monthly_index.py
from hubeau_pipeline.ml.indices import compute_reference_grid, grid_to_zscore, classify_value
from hubeau_pipeline.ml.monthly_index_assets import rescore_series

def test_rescore_series_matches_grid_to_zscore_per_month():
    months = [f"{y}-{m:02d}-01" for y in range(2000, 2021) for m in range(1, 13)]
    values = [10.0 + (i % 12) * 0.1 + (i // 12) * 0.05 for i in range(len(months))]
    res = compute_reference_grid(months, values, positive_only=False)
    rows = rescore_series("piezo", "TESTCODE", months, values, res)
    # last row reproduces the station_current_index computation exactly
    last_dt = months[-1]
    import pandas as pd
    z_expected = grid_to_zscore(values[-1], res["grid"].get(pd.to_datetime(last_dt).month))
    code, type_, month, z, cls, flag = rows[-1]
    assert type_ == "piezo" and code == "TESTCODE"
    assert z == z_expected
    assert cls == (classify_value(z_expected) if z_expected is not None else "UNKNOWN")
    assert len(rows) == len(months)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/hubeau_data_integration && python -m pytest tests/test_monthly_index.py -v`
Expected: FAIL — `ModuleNotFoundError: ... monthly_index_assets`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hubeau_pipeline/assets/monthly_index_assets.py
"""Per-station MONTHLY standardized index re-scored against the fixed reference grid → gold.fct_monthly_index."""
import logging

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset

from ..ml.indices import compute_reference_grid, grid_to_zscore, classify_value
from ..ml.monthly_index_persistence import init_monthly_index_table, upsert_monthly_index
from ..resources import PostgreSQLResource

logger = logging.getLogger(__name__)

_DOMAINS = [
    ("piezo", "gold.fct_monthly_chroniques", "code_bss", "niveau_moyen", False),
    ("hydro", "gold.fct_monthly_hydro", "code_station", "resultat_moyen", True),
]


def rescore_series(domain, code, months, values, ref):
    """Re-score every month against the station's fixed reference grid.

    Returns list of (code, type, month_date, z|None, index_class, flag).
    """
    grid = ref["grid"]
    flag = ref["flag"]
    out = []
    for m_iso, val in zip(months, values):
        dt = pd.to_datetime(m_iso)
        z = grid_to_zscore(float(val), grid.get(dt.month)) if val is not None else None
        cls = classify_value(z) if z is not None else "UNKNOWN"
        out.append((code, domain, dt.date(), z, cls, flag))
    return out


@asset(
    name="fct_monthly_index",
    group_name="indices",
    deps=["station_reference_stats"],
    description="Monthly standardized index (IPS/SSFI) re-scored against the fixed reference grid, full history.",
)
def fct_monthly_index(context: AssetExecutionContext, pg: PostgreSQLResource):
    init_monthly_index_table(pg)
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
            ref = compute_reference_grid(months, values, positive_only=positive_only)
            rows.extend(rescore_series(domain, code, months, values, ref))
        upsert_monthly_index(pg, rows)
        total += len(rows)
        context.log.info("%s: re-scored %d station-months (fixed ref)", domain, len(rows))
    context.add_output_metadata({"station_months": MetadataValue.int(total)})
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/hubeau_data_integration && python -m pytest tests/test_monthly_index.py -v`
Expected: PASS

- [ ] **Step 5: Register the asset**

Run to find the registry: `cd ~/hubeau_data_integration && grep -rn "station_current_index" src/hubeau_pipeline/definitions.py src/hubeau_pipeline/assets/__init__.py`
Add `fct_monthly_index` next to `station_current_index` in the same `load_assets`/`Definitions(assets=[...])` list (mirror the import + list entry exactly as `station_current_index` is wired).

- [ ] **Step 6: Commit**

```bash
cd ~/hubeau_data_integration
git add src/hubeau_pipeline/assets/monthly_index_assets.py tests/test_monthly_index.py src/hubeau_pipeline/definitions.py src/hubeau_pipeline/assets/__init__.py
git commit -m "feat(indices): fct_monthly_index dagster asset (monthly fixed-ref re-score)"
```

### Task 3: Materialize once + restart worker

- [ ] **Step 1: Restart the worker so it picks up the new code**

Run: `docker restart brgm-dlt-worker`
Expected: container restarts (code-server mounts `./src`).

- [ ] **Step 2: Materialize the asset**

Run: `docker exec brgm-dlt-worker sh -lc 'cd /app && dagster asset materialize --select fct_monthly_index -m hubeau_pipeline.definitions'`
Expected: completes (~minutes), logs "piezo: re-scored N station-months" and "hydro: ...".

- [ ] **Step 3: Verify the table**

Run: `docker exec brgm-postgres psql -U postgres -d postgres -c "SELECT type, count(*), min(month), max(month) FROM gold.fct_monthly_index GROUP BY type;"`
Expected: two rows (piezo, hydro) with non-zero counts and a month range from ~2000 to the latest month.

---

## Phase 2 — App: pure point-in-polygon + name helpers

> Repo: `time-serie-explo`, branch `feat/meteo-nappes-sectors` (already created). Run `cd /home/ringuet/time-serie-explo`.

### Task 4: `geo_sectors` pure helpers

**Files:**
- Create: `dashboard/utils/geo_sectors.py`
- Test: `tests/test_geo_sectors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geo_sectors.py
from dashboard.utils.geo_sectors import point_in_ring, point_in_geometry, dominant_label

SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]}
MULTI = {"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                                                  [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]]]}

def test_point_in_ring_inside_and_outside():
    ring = [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]
    assert point_in_ring(1, 1, ring) is True
    assert point_in_ring(3, 3, ring) is False

def test_point_in_geometry_polygon_and_multipolygon():
    assert point_in_geometry(1.0, 1.0, SQUARE) is True
    assert point_in_geometry(9.0, 9.0, SQUARE) is False
    assert point_in_geometry(5.5, 5.5, MULTI) is True
    assert point_in_geometry(3.0, 3.0, MULTI) is False

def test_dominant_label_picks_most_frequent_nonempty():
    assert dominant_label(["Craie", "Craie", "Alluvions", None, ""]) == "Craie"

def test_dominant_label_returns_none_when_all_empty():
    assert dominant_label([None, "", "  "]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_geo_sectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.utils.geo_sectors'`

- [ ] **Step 3: Write minimal implementation** (ray-casting ported from the frontend `pointInRing`)

```python
# dashboard/utils/geo_sectors.py
"""Pure geometry helpers for sector mapping. No DB, no Streamlit, no shapely."""
from __future__ import annotations

from collections import Counter


def point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _in_polygon(x: float, y: float, polygon: list) -> bool:
    outer, *holes = polygon
    if not point_in_ring(x, y, outer):
        return False
    return not any(point_in_ring(x, y, h) for h in holes)


def point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    coords = geometry.get("coordinates")
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return _in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(_in_polygon(lon, lat, poly) for poly in coords)
    return False


def dominant_label(labels) -> str | None:
    clean = [str(s).strip() for s in labels if s is not None and str(s).strip()]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_geo_sectors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/geo_sectors.py tests/test_geo_sectors.py
git commit -m "feat(meteo): pure point-in-polygon + dominant-label helpers"
```

---

## Phase 3 — App: build the static sectors geojson

### Task 5: `build_secteurs_bsh_geojson.py` (geometry + tendancy_coord + name)

**Files:**
- Create: `scripts/build_secteurs_bsh_geojson.py`
- Output: `frontend/public/geo/secteurs-bsh.geojson` (committed)

> This script is run once (and re-run only if BRGM re-sectorizes). It needs DB access to the warehouse (reuse `api.database.get_brgm_sync_engine`) and outbound HTTPS to the BRGM WFS. No test (one-shot data tool); correctness is verified by the assertions it prints.

- [ ] **Step 1: Write the script**

```python
# scripts/build_secteurs_bsh_geojson.py
"""One-shot: fetch BRGM BSH parent sectors (geometry only) and bake a name per sector
from the dominant entité hydrogéologique (libelle_eh) of the piezo stations inside it.

Run: python -m scripts.build_secteurs_bsh_geojson
Re-run only if BRGM changes the sectorization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from sqlalchemy import text

from api.database import get_brgm_sync_engine
from dashboard.utils.geo_sectors import point_in_geometry, dominant_label

OUT = Path("frontend/public/geo/secteurs-bsh.geojson")
WFS = "https://app.meteeaunappes.brgm.fr/wfs/indicateur_bsn/ows"
# Pick any recent communicated snapshot just to get parent geometry + tendancy_coord.
FILTER = (
    '<Filter xmlns:gml="http://www.opengis.net/gml"><And>'
    "<PropertyIsEqualTo><PropertyName>communicate</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "<PropertyIsEqualTo><PropertyName>is_parent</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "<PropertyIsEqualTo><PropertyName>visualizer</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "</And></Filter>"
)


def fetch_sectors() -> list[dict]:
    url = (
        f"{WFS}?service=WFS&version=1.0.0&request=GetFeature"
        f"&outputFormat=application%2Fjson"
        f"&typeName=indicateur_bsn:view_global_indicator_details&filter={quote(FILTER)}"
    )
    with urlopen(url, timeout=120) as resp:
        return json.load(resp)["features"]


def station_points() -> list[tuple]:
    """(lon, lat, libelle_eh) for piezo stations that have an EH label."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.longitude AS lon, s.latitude AS lat, m.libelle_eh
            FROM gold.dim_piezo_stations s
            JOIN gold.int_station_era5_mapping m ON m.code_bss = s.code_bss
            WHERE s.longitude IS NOT NULL AND s.latitude IS NOT NULL
        """)).mappings().all()
    return [(float(r["lon"]), float(r["lat"]), r["libelle_eh"]) for r in rows]


def main() -> int:
    feats = fetch_sectors()
    pts = station_points()
    out_features = []
    for f in feats:
        sid = f["properties"]["sector_id"]
        coord = f["properties"].get("tendancy_coord")  # "lat lon"
        geom = f["geometry"]
        labels = [eh for (lon, lat, eh) in pts if point_in_geometry(lon, lat, geom)]
        nom = dominant_label(labels) or f"Secteur {sid}"
        out_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"sector_id": sid, "tendancy_coord": coord, "nom": nom},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out_features}))
    named = sum(1 for f in out_features if not f["properties"]["nom"].startswith("Secteur "))
    print(f"wrote {len(out_features)} sectors to {OUT} ({named} with an EH name)")
    assert len(out_features) >= 50, "expected ~66 parent sectors"
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the script**

Run: `cd /home/ringuet/time-serie-explo && python -m scripts.build_secteurs_bsh_geojson`
Expected: prints `wrote 66 sectors to frontend/public/geo/secteurs-bsh.geojson (NN with an EH name)` with NN > 50.

- [ ] **Step 3: Sanity-check the output**

Run: `python -c "import json; d=json.load(open('frontend/public/geo/secteurs-bsh.geojson')); f=d['features'][0]; print(len(d['features']), sorted(f['properties']), f['properties']['nom'])"`
Expected: `66 ['nom', 'sector_id', 'tendancy_coord'] <some name>`

- [ ] **Step 4: Commit**

```bash
git add scripts/build_secteurs_bsh_geojson.py frontend/public/geo/secteurs-bsh.geojson
git commit -m "feat(meteo): build + commit BRGM BSH sectors geojson (geometry + EH names)"
```

---

## Phase 4 — App backend: mapping + sector endpoints

### Task 6: Cached station→sector mapping service

**Files:**
- Create: `api/services/sector_mapping.py`
- Test: `tests/test_sector_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sector_mapping.py
from api.services.sector_mapping import build_mapping

SECTORS = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"sector_id": 1, "nom": "A", "tendancy_coord": "0.5 0.5"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}},
    {"type": "Feature", "properties": {"sector_id": 2, "nom": "B", "tendancy_coord": "5.5 5.5"},
     "geometry": {"type": "Polygon", "coordinates": [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]]}},
]}

def test_build_mapping_assigns_station_to_containing_sector():
    stations = [("S1", 0.5, 0.5), ("S2", 5.5, 5.5), ("S3", 9.0, 9.0)]
    code_to_sector, meta = build_mapping(SECTORS, stations)
    assert code_to_sector["S1"] == 1
    assert code_to_sector["S2"] == 2
    assert "S3" not in code_to_sector            # outside every sector
    assert meta[1]["nom"] == "A" and meta[1]["tendancy_coord"] == "0.5 0.5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sector_mapping.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# api/services/sector_mapping.py
"""Load secteurs-bsh.geojson and map station coords -> sector_id (cached process-wide)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from api.database import get_brgm_sync_engine
from dashboard.utils.geo_sectors import point_in_geometry

GEOJSON = Path(__file__).resolve().parents[2] / "frontend" / "public" / "geo" / "secteurs-bsh.geojson"


def build_mapping(geojson: dict, stations: list[tuple]):
    """stations: list of (code, lon, lat). Returns (code->sector_id, {sector_id: meta})."""
    feats = geojson["features"]
    meta = {f["properties"]["sector_id"]: {
        "nom": f["properties"].get("nom"),
        "tendancy_coord": f["properties"].get("tendancy_coord"),
    } for f in feats}
    code_to_sector: dict[str, int] = {}
    for code, lon, lat in stations:
        if lon is None or lat is None:
            continue
        for f in feats:
            if point_in_geometry(float(lon), float(lat), f["geometry"]):
                code_to_sector[code] = f["properties"]["sector_id"]
                break
    return code_to_sector, meta


def _load_geojson() -> dict:
    return json.loads(GEOJSON.read_text())


def _load_stations(type_: str) -> list[tuple]:
    if type_ == "piezo":
        sql = "SELECT code_bss AS code, longitude AS lon, latitude AS lat FROM gold.dim_piezo_stations WHERE longitude IS NOT NULL"
    else:
        sql = "SELECT code_station AS code, longitude_station AS lon, latitude_station AS lat FROM gold.dim_hydro_stations WHERE longitude_station IS NOT NULL"
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        return [(r["code"], r["lon"], r["lat"]) for r in conn.execute(text(sql)).mappings()]


@lru_cache(maxsize=2)
def get_mapping(type_: str):
    """Cached (code->sector_id, {sector_id: meta}) for a station type."""
    return build_mapping(_load_geojson(), _load_stations(type_))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sector_mapping.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/services/sector_mapping.py tests/test_sector_mapping.py
git commit -m "feat(meteo): cached station->sector point-in-polygon mapping"
```

### Task 7: Generalize the situation aggregator for `level=sector`

**Files:**
- Modify: `api/routers/observatory_situation.py`
- Test: `tests/test_territory_situation_sector.py`

> The existing `_fetch_station_rows` returns `(dept, z_latest, delta_z, flag)` — no station code, so we cannot map to sectors. Add a code-returning fetch and a sector keyer. Reuse `_eligible_rows_to_territories` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_territory_situation_sector.py
from api.routers.observatory_situation import _key_rows_by_sector

def test_key_rows_by_sector_uses_mapping_and_meta():
    rows = [("S1", 1.0, 0.2, "normale"), ("S2", -2.0, None, "normale"), ("SX", 0.0, 0.0, "normale")]
    code_to_sector = {"S1": 7, "S2": 7}                      # SX unmapped -> dropped
    meta = {7: {"nom": "Craie", "tendancy_coord": "50 3"}}
    keyed = _key_rows_by_sector(rows, code_to_sector, meta)
    assert {k[0] for k in keyed} == {7}                       # only sector 7
    assert keyed[0][1] == "Craie"                             # name from meta
    assert len(keyed) == 2                                    # SX dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_territory_situation_sector.py -v`
Expected: FAIL — `cannot import name '_key_rows_by_sector'`

- [ ] **Step 3: Implement** — add to `api/routers/observatory_situation.py`

After the existing `_fetch_station_rows`, add a code-returning fetch (same SQL but also select the station code) and the sector keyer:

```python
_STATION_SQL_WITH_CODE = {
    "piezo": _STATION_SQL["piezo"].replace(
        "SELECT s.code_departement AS dept,",
        "SELECT s.code_bss AS code, s.code_departement AS dept,", 1),
    "hydro": _STATION_SQL["hydro"].replace(
        "SELECT s.code_departement AS dept,",
        "SELECT s.code_station AS code, s.code_departement AS dept,", 1),
}


def _fetch_station_rows_with_code(type_: str) -> list[tuple]:
    """-> list of (code, z_latest, delta_z, flag)."""
    from dashboard.utils.reference import value_to_zscore
    engine = get_brgm_sync_engine()
    out: list[tuple] = []
    with engine.connect() as conn:
        result = conn.execute(text(_STATION_SQL_WITH_CODE[type_]), {"min_mois": RELIABLE_MIN_MOIS})
        for r in result.mappings():
            z_latest = r["z_latest"]
            delta_z = None
            if z_latest is not None and r["lag_value"] is not None and r["lag_grid"]:
                z_lag = value_to_zscore(float(r["lag_value"]), list(r["lag_grid"]))
                if z_lag is not None:
                    delta_z = float(z_latest) - z_lag
            out.append((r["code"], z_latest, delta_z, r["flag"]))
    return out


def _key_rows_by_sector(rows, code_to_sector, meta):
    """rows: (code, z, dz, flag) -> keyed rows (sector_id, nom, z, dz, flag)."""
    keyed = []
    for code, z, dz, flag in rows:
        sid = code_to_sector.get(code)
        if sid is None:
            continue
        keyed.append((sid, meta.get(sid, {}).get("nom") or f"Secteur {sid}", z, dz, flag))
    return keyed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_territory_situation_sector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_situation.py tests/test_territory_situation_sector.py
git commit -m "feat(meteo): sector keyer + code-returning station fetch"
```

### Task 8: `/situation/sectors` endpoint (current + past month)

**Files:**
- Modify: `api/routers/observatory_situation.py`
- Test: `tests/test_sectors_endpoint.py`

> Current month uses `_fetch_station_rows_with_code` (which reads `station_current_index`). A past `month` uses `gold.fct_monthly_index` for both the value and the 3-month-prior delta. The response adds `tendancy_coord` to each `TerritorySituation`-shaped dict (extra key tolerated by the existing schema usage — but to be safe we return a plain dict list, not `TerritorySituation`, since the choropleth consumes JSON).

- [ ] **Step 1: Write the failing test** (route exists, returns a list, sector entries carry `tendancy_coord`)

```python
# tests/test_sectors_endpoint.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_sectors_route_returns_list_with_sector_shape():
    r = client.get("/api/v1/observatory/situation/sectors?type=piezo")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:  # warehouse may be empty in some envs
        e = data[0]
        for k in ("level", "code", "name", "situation_class", "trend", "tendancy_coord"):
            assert k in e
        assert e["level"] == "sector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sectors_endpoint.py -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Implement** — add the helper for past-month rows and the endpoint to `api/routers/observatory_situation.py`

```python
_MONTHLY_SQL = {
    "piezo": ("gold.fct_monthly_chroniques", "code_bss", "niveau_moyen"),
    "hydro": ("gold.fct_monthly_hydro", "code_station", "resultat_moyen"),
}


def _fetch_month_rows_with_code(type_: str, month: str) -> list[tuple]:
    """Past-month rows (code, z, delta_z, flag) from gold.fct_monthly_index.

    z = index at `month`; delta_z = z(month) - z(month-3) read from the same table.
    """
    sql = text("""
        WITH cur AS (
            SELECT code, z, flag FROM gold.fct_monthly_index
            WHERE type = :t AND month = date_trunc('month', :m::date)
              AND index_class <> 'UNKNOWN'
        ),
        prev AS (
            SELECT code, z FROM gold.fct_monthly_index
            WHERE type = :t AND month = (date_trunc('month', :m::date) - INTERVAL '3 months')
        )
        SELECT cur.code, cur.z AS z, cur.flag,
               (cur.z - prev.z) AS delta_z
        FROM cur LEFT JOIN prev ON prev.code = cur.code
    """)
    engine = get_brgm_sync_engine()
    out = []
    with engine.connect() as conn:
        for r in conn.execute(sql, {"t": type_, "m": f"{month}-01"}).mappings():
            out.append((r["code"], r["z"], r["delta_z"], r["flag"]))
    return out


@router.get("/situation/sectors")
def get_sector_situation(
    type: Literal["piezo", "hydro"] = Query("piezo"),
    month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
):
    from api.services.sector_mapping import get_mapping

    def fetch():
        code_to_sector, meta = get_mapping(type)
        rows = (_fetch_month_rows_with_code(type, month) if month
                else _fetch_station_rows_with_code(type))
        keyed = _key_rows_by_sector(rows, code_to_sector, meta)
        out = _eligible_rows_to_territories(keyed, "sector", type)
        for t in out:
            sid = t["code"]
            t["code"] = str(sid)
            t["tendancy_coord"] = meta.get(sid, {}).get("tendancy_coord")
        out.sort(key=lambda t: t["name"])
        return out

    return get_cached("obs_situation_sectors", {"type": type, "month": month}, SITUATION_TTL, fetch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sectors_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_situation.py tests/test_sectors_endpoint.py
git commit -m "feat(meteo): /situation/sectors endpoint (current + past month)"
```

### Task 9: `/situation/sectors/timeline` endpoint

**Files:**
- Modify: `api/routers/observatory_situation.py`
- Test: `tests/test_sectors_timeline_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sectors_timeline_endpoint.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_sectors_timeline_shape():
    r = client.get("/api/v1/observatory/situation/sectors/timeline?type=piezo")
    assert r.status_code == 200
    d = r.json()
    assert set(d.keys()) >= {"periods", "sectors", "trends"}
    assert isinstance(d["periods"], list)
    assert isinstance(d["sectors"], dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sectors_timeline_endpoint.py -v`
Expected: FAIL — 404.

- [ ] **Step 3: Implement** — add to `api/routers/observatory_situation.py`

```python
_CLASS_TO_IDX = {c: i for i, c in enumerate([
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"])}
_TREND_CODE = {"baisse": -1, "stable": 0, "hausse": 1}


@router.get("/situation/sectors/timeline")
def get_sector_timeline(type: Literal["piezo", "hydro"] = Query("piezo")):
    from api.services.sector_mapping import get_mapping

    def fetch():
        code_to_sector, meta = get_mapping(type)
        sql = text("""
            SELECT code, TO_CHAR(month, 'YYYY-MM') AS period, z, index_class, flag
            FROM gold.fct_monthly_index
            WHERE type = :t AND month >= '2000-01-01'
            ORDER BY code, month
        """)
        engine = get_brgm_sync_engine()
        # per (sector, period): collect eligible z and 3-month deltas
        per: dict[tuple, dict] = {}
        prev_z: dict[str, dict] = {}
        periods_set: set[str] = set()
        with engine.connect() as conn:
            for r in conn.execute(sql, {"t": type}).mappings():
                sid = code_to_sector.get(r["code"])
                if sid is None:
                    continue
                p = r["period"]
                periods_set.add(p)
                slot = per.setdefault((sid, p), {"z": [], "dz": []})
                if r["flag"] in ("normale", "adaptee") and r["z"] is not None and r["index_class"] != "UNKNOWN":
                    slot["z"].append(float(r["z"]))
                    # delta vs same station 3 periods earlier (period strings sort chronologically)
                    hist = prev_z.setdefault(r["code"], {})
                    y, m = int(p[:4]), int(p[5:7])
                    pm = m - 3
                    py = y
                    if pm <= 0:
                        pm += 12
                        py -= 1
                    prior = hist.get(f"{py:04d}-{pm:02d}")
                    if prior is not None:
                        slot["dz"].append(float(r["z"]) - prior)
                    hist[p] = float(r["z"])

        periods = sorted(periods_set)
        sectors: dict[str, list[int]] = {}
        trends: dict[str, list[int]] = {}
        sids = {sid for (sid, _p) in per.keys()}
        for sid in sids:
            cls_arr, trd_arr = [], []
            for p in periods:
                slot = per.get((sid, p))
                if not slot or len(slot["z"]) < MIN_ELIGIBLE_SECTOR():
                    cls_arr.append(7)        # UNKNOWN/insufficient
                    trd_arr.append(0)
                    continue
                sit = aggregate_situation(slot["z"])
                cls = sit["situation_class"]
                cls_arr.append(_CLASS_TO_IDX.get(cls, 7) if cls else 7)
                trd_arr.append(_TREND_CODE.get(aggregate_trend(slot["dz"]) or "stable", 0))
            sectors[str(sid)] = cls_arr
            trends[str(sid)] = trd_arr
        return {"periods": periods, "sectors": sectors, "trends": trends}

    return get_cached("obs_sectors_timeline", {"type": type}, 86400, fetch)
```

Add this tiny helper near the top of the module (keeps `MIN_ELIGIBLE` import-free at call site):

```python
def MIN_ELIGIBLE_SECTOR():
    from dashboard.utils.territory_situation import MIN_ELIGIBLE
    return MIN_ELIGIBLE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_sectors_timeline_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_situation.py tests/test_sectors_timeline_endpoint.py
git commit -m "feat(meteo): /situation/sectors/timeline endpoint (fixed-ref per sector per month)"
```

---

## Phase 5 — App backend: unify the station timeline on `fct_monthly_index`

### Task 10: Repoint `/classifications/timeline` to `fct_monthly_index`

**Files:**
- Modify: `api/routers/observatory_common.py` (`get_classification_timeline`, lines 327-414)
- Test: `tests/test_timeline_uses_fixed_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timeline_uses_fixed_index.py
import inspect
from api.routers import observatory_common

def test_timeline_reads_fct_monthly_index_not_percent_rank():
    src = inspect.getsource(observatory_common.get_classification_timeline)
    assert "gold.fct_monthly_index" in src
    assert "PERCENT_RANK" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_timeline_uses_fixed_index.py -v`
Expected: FAIL — still uses PERCENT_RANK.

- [ ] **Step 3: Implement** — replace the body of `get_classification_timeline` (the two `*_query` strings + the fetch loop) with a single read from `fct_monthly_index`, mapping `index_class` → the same 0-7 codes:

```python
    def fetch():
        query = text("""
            SELECT code, TO_CHAR(month, 'YYYY-MM') AS period, index_class
            FROM gold.fct_monthly_index
            WHERE month >= '2000-01-01'
            ORDER BY code, month
        """)
        cls_to_idx = {
            "EXTREMEMENT_BAS": 0, "TRES_BAS": 1, "BAS": 2, "NORMAL": 3,
            "HAUT": 4, "TRES_HAUT": 5, "EXTREMEMENT_HAUT": 6,
        }
        periods_set: set[str] = set()
        station_periods: dict[str, dict[str, int]] = {}
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            for row in conn.execute(query).mappings():
                periods_set.add(row["period"])
                station_periods.setdefault(row["code"], {})[row["period"]] = cls_to_idx.get(row["index_class"], 7)
        periods = sorted(periods_set)
        stations = {code: [vals.get(p, 7) for p in periods] for code, vals in station_periods.items()}
        return {"periods": periods, "stations": stations}

    return get_cached("obs_timeline", {}, TIMELINE_TTL, fetch)
```

Update the docstring to say "fixed-reference index from gold.fct_monthly_index" (remove the calendar-month percentile description).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_timeline_uses_fixed_index.py -v`
Expected: PASS

- [ ] **Step 5: Smoke-test the endpoint returns data**

Run: `cd /home/ringuet/time-serie-explo && python -c "from fastapi.testclient import TestClient; from api.main import app; r=TestClient(app).get('/api/v1/observatory/classifications/timeline'); print(r.status_code, len(r.json()['periods']))"`
Expected: `200` and a non-zero period count.

- [ ] **Step 6: Commit**

```bash
git add api/routers/observatory_common.py tests/test_timeline_uses_fixed_index.py
git commit -m "refactor(meteo): timeline reads fct_monthly_index (fixed ref, unifies with markers)"
```

---

## Phase 6 — Frontend: sector layer pure helpers + types + API

### Task 11: Pure frontend helpers (`sector-arrows.ts`)

**Files:**
- Create: `frontend/src/lib/sector-arrows.ts`
- Test: `frontend/src/lib/sector-arrows.test.ts`

> Run frontend tests with the project runner: `cd frontend && npx vitest run <file>`.

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/lib/sector-arrows.test.ts
import { describe, it, expect } from 'vitest'
import { parseTendancyCoord, trendArrowGlyph } from './sector-arrows'

describe('sector-arrows', () => {
  it('parses "lat lon" into [lon, lat]', () => {
    expect(parseTendancyCoord('50.13529085 3.04309184')).toEqual([3.04309184, 50.13529085])
  })
  it('returns null for invalid coord', () => {
    expect(parseTendancyCoord('')).toBeNull()
    expect(parseTendancyCoord('abc')).toBeNull()
  })
  it('maps trend to arrow glyph', () => {
    expect(trendArrowGlyph('hausse')).toBe('↑')
    expect(trendArrowGlyph('baisse')).toBe('↓')
    expect(trendArrowGlyph('stable')).toBe('→')
    expect(trendArrowGlyph(null)).toBe('')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx vitest run src/lib/sector-arrows.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```typescript
// frontend/src/lib/sector-arrows.ts
import type { SituationClass, Trend } from '@/lib/observatory-types'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'

export const SECTOR_INSUFFICIENT_COLOR = '#d9d9d9'

/** BRGM tendancy_coord is "lat lon"; MapLibre wants [lon, lat]. */
export function parseTendancyCoord(raw: string | null | undefined): [number, number] | null {
  if (!raw) return null
  const parts = raw.trim().split(/\s+/).map(Number)
  if (parts.length !== 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null
  return [parts[1], parts[0]]
}

export function trendArrowGlyph(trend: Trend | null | undefined): string {
  if (trend === 'hausse') return '↑'
  if (trend === 'baisse') return '↓'
  if (trend === 'stable') return '→'
  return ''
}

export function sectorClassColor(cls: SituationClass | null): string {
  return cls ? (CLASSIFICATION_COLORS[cls] ?? SECTOR_INSUFFICIENT_COLOR) : SECTOR_INSUFFICIENT_COLOR
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx vitest run src/lib/sector-arrows.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/sector-arrows.ts frontend/src/lib/sector-arrows.test.ts
git commit -m "feat(meteo): pure sector arrow/color helpers"
```

### Task 12: Types + API client for sector situation/timeline

**Files:**
- Modify: `frontend/src/lib/observatory-types.ts`, `frontend/src/lib/observatory-api.ts`, `frontend/src/hooks/useObservatory.ts`

- [ ] **Step 1: Add types** to `frontend/src/lib/observatory-types.ts`

```typescript
export interface SectorSituation {
  level: 'sector'
  code: string
  name: string
  type: 'piezo' | 'hydro'
  situation_class: SituationClass | null
  trend: Trend | null
  pct_below_normal: number | null
  n_eligible: number
  n_provisoire: number
  insufficient: boolean
  tendancy_coord: string | null
}

export interface SectorTimeline {
  periods: string[]
  sectors: Record<string, number[]>
  trends: Record<string, number[]>
}
```

- [ ] **Step 2: Add API methods** to `frontend/src/lib/observatory-api.ts` (follow the existing `situationApi`/`observatoryApi.common` patterns)

```typescript
// in the situation/observatory api object:
sectors: (type: 'piezo' | 'hydro', month?: string) =>
  get<SectorSituation[]>(`/observatory/situation/sectors?type=${type}${month ? `&month=${month}` : ''}`),
sectorsTimeline: (type: 'piezo' | 'hydro') =>
  get<SectorTimeline>(`/observatory/situation/sectors/timeline?type=${type}`),
```

(Use the same `get`/axios helper the file already uses; import `SectorSituation`, `SectorTimeline` types.)

- [ ] **Step 3: Add React Query hooks** to `frontend/src/hooks/useObservatory.ts`

```typescript
export function useSectorSituation(type: 'piezo' | 'hydro', enabled: boolean) {
  return useQuery({ queryKey: ['obs-sectors', type], queryFn: () => observatoryApi.sectors(type), enabled, staleTime: 3_600_000 * 6 })
}
export function useSectorTimeline(type: 'piezo' | 'hydro', enabled: boolean) {
  return useQuery({ queryKey: ['obs-sectors-timeline', type], queryFn: () => observatoryApi.sectorsTimeline(type), enabled, staleTime: 3_600_000 * 24 })
}
```

- [ ] **Step 4: Type-check**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/observatory-types.ts frontend/src/lib/observatory-api.ts frontend/src/hooks/useObservatory.ts
git commit -m "feat(meteo): sector situation/timeline types + api + hooks"
```

---

## Phase 7 — Frontend: sector layer in the map

### Task 13: Render sector choropleth + arrows in `ObservatoryMap`

**Files:**
- Modify: `frontend/src/components/observatory/ObservatoryMap.tsx`

> Add props and a `useEffect` that builds the `secteurs-bsh` source + `secteurs-fill`/`secteurs-line` layers + a GeoJSON arrow source/`secteurs-arrows` symbol layer. Mirror the existing static-layer pattern used for `her`/`bassins` (fetch from `/geo/...`, add fill+line, toggle visibility, hover, click→spatial filter). Color via `['match', ['get','sector_id'], ...]` like `TerritoryChoropleth.fillColorExpression`, using `sectorClassColor`.

- [ ] **Step 1: Add props** to the `ObservatoryMap` Props interface

```typescript
showSectors?: boolean
sectorSituation?: import('@/lib/observatory-types').SectorSituation[]
onSectorClick?: (codes: string[] | null, name: string | null) => void
```

- [ ] **Step 2: Add the sector layer effect** (place near the existing HER/bassins layer setup)

```tsx
// fetch sector geometry once
useEffect(() => {
  const m = map.current; if (!m) return
  let cancelled = false
  fetch('/geo/secteurs-bsh.geojson').then(r => r.json()).then((gj) => {
    if (cancelled || !m) return
    const add = () => {
      if (!m.getSource('secteurs-bsh')) {
        m.addSource('secteurs-bsh', { type: 'geojson', data: gj, attribution: 'Secteurs © BRGM / Eaufrance' })
        m.addLayer({ id: 'secteurs-fill', type: 'fill', source: 'secteurs-bsh',
          layout: { visibility: 'none' }, paint: { 'fill-color': SECTOR_INSUFFICIENT_COLOR, 'fill-opacity': 0.55 } })
        m.addLayer({ id: 'secteurs-line', type: 'line', source: 'secteurs-bsh',
          layout: { visibility: 'none' }, paint: { 'line-color': '#ffffff', 'line-width': 0.6 } })
        // arrows: build a point FeatureCollection from tendancy_coord at situation update time (Step 3)
        m.addSource('secteurs-arrows', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
        m.addLayer({ id: 'secteurs-arrows', type: 'symbol', source: 'secteurs-arrows',
          layout: { visibility: 'none', 'text-field': ['get', 'glyph'], 'text-size': 20, 'text-allow-overlap': true },
          paint: { 'text-color': '#0f172a', 'text-halo-color': '#ffffff', 'text-halo-width': 1.5 } })
        // click a sector -> spatial filter via stations inside its geometry
        m.on('click', 'secteurs-fill', (e) => {
          const f = e.features?.[0]; if (!f) return
          const codes = stationsInGeometryRef.current?.(f.geometry as any) ?? null
          onSectorClickRef.current?.(codes, (f.properties?.nom as string) ?? null)
        })
        m.on('mouseenter', 'secteurs-fill', () => { m.getCanvas().style.cursor = 'pointer' })
        m.on('mouseleave', 'secteurs-fill', () => { m.getCanvas().style.cursor = '' })
      }
    }
    if (m.isStyleLoaded()) add(); else m.once('load', add)
  }).catch(() => {})
  return () => { cancelled = true }
}, [])
```

> `stationsInGeometryRef` / `onSectorClickRef` are refs set from props in the existing ref pattern this file already uses. `SECTOR_INSUFFICIENT_COLOR` imported from `@/lib/sector-arrows`. The page passes a `stationsInGeometry`-style callback (it already has that function — Task 15 wires it).

- [ ] **Step 3: Add a visibility+paint+arrows update effect** keyed on `showSectors`/`sectorSituation`

```tsx
useEffect(() => {
  const m = map.current; if (!m || !m.getLayer('secteurs-fill')) return
  const vis = showSectors ? 'visible' : 'none'
  for (const id of ['secteurs-fill', 'secteurs-line', 'secteurs-arrows']) m.setLayoutProperty(id, 'visibility', vis)
  if (!showSectors) return
  const sits = sectorSituation ?? []
  const pairs: (string | number)[] = []
  for (const s of sits) { pairs.push(Number(s.code), s.insufficient ? SECTOR_INSUFFICIENT_COLOR : sectorClassColor(s.situation_class)) }
  m.setPaintProperty('secteurs-fill', 'fill-color',
    pairs.length ? (['match', ['get', 'sector_id'], ...pairs, SECTOR_INSUFFICIENT_COLOR] as any) : SECTOR_INSUFFICIENT_COLOR)
  // arrows from tendancy_coord + trend
  const arrowFeatures = sits.map(s => {
    const c = parseTendancyCoord(s.tendancy_coord); const glyph = trendArrowGlyph(s.trend)
    if (!c || !glyph || s.insufficient) return null
    return { type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: c }, properties: { glyph } }
  }).filter(Boolean)
  ;(m.getSource('secteurs-arrows') as maplibregl.GeoJSONSource)?.setData({ type: 'FeatureCollection', features: arrowFeatures as any })
}, [showSectors, sectorSituation])
```

(Import `parseTendancyCoord, trendArrowGlyph, sectorClassColor, SECTOR_INSUFFICIENT_COLOR` from `@/lib/sector-arrows`.)

- [ ] **Step 4: Type-check + build**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/observatory/ObservatoryMap.tsx
git commit -m "feat(meteo): sector choropleth + trend arrows layer in ObservatoryMap"
```

### Task 14: Sector toggle + legend in `RightDrawer`

**Files:**
- Modify: `frontend/src/components/observatory/RightDrawer.tsx`

- [ ] **Step 1: Add a "Météo des nappes" zone-layer entry** in `useZoneLayers` (a new group at the top so it reads as the headline view)

```typescript
{ id: 'secteurs', label: t('observatory.drawer.sectorsMeteo'), group: t('observatory.drawer.groupMeteo'), color: '#3b82f6' },
```

> Because the drawer's zone layers are a single-select radio (`activeZoneLayer`), selecting "secteurs" deselects regions/depts/HER — exactly the intended behavior (one mesh at a time). No new prop needed: the page derives `showSectors = activeZoneLayer === 'secteurs'` (Task 15).

- [ ] **Step 2: Add a sector legend block** shown only when the sector layer is active. Add after the layers accordion content:

```tsx
{props.activeZoneLayer === 'secteurs' && (
  <div className="px-4 pb-3 text-[11px] text-text-secondary space-y-1">
    <div className="font-semibold text-text-primary">{t('observatory.drawer.sectorLegendTitle')}</div>
    {CLASSIFICATION_ORDER.map(cls => (
      <div key={cls} className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm" style={{ backgroundColor: CLASSIFICATION_COLORS[cls] }} />{CLASSIFICATION_LABELS[cls]}</div>
    ))}
    <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#d9d9d9' }} />{t('observatory.drawer.sectorNoData')}</div>
    <div className="pt-1">↑ {t('meteo.trend.hausse')} · → {t('meteo.trend.stable')} · ↓ {t('meteo.trend.baisse')}</div>
  </div>
)}
```

- [ ] **Step 3: Type-check**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/observatory/RightDrawer.tsx
git commit -m "feat(meteo): sector layer toggle + legend in RightDrawer"
```

### Task 15: Wire sectors into `ObservatoryPage` (data + slider + click)

**Files:**
- Modify: `frontend/src/pages/ObservatoryPage.tsx`

- [ ] **Step 1: Derive sector state + data**

```tsx
const showSectors = activeZoneLayer === 'secteurs'
const sectorType: 'piezo' | 'hydro' = showHydro && !showPiezo ? 'hydro' : 'piezo'
const { data: sectorSituationData } = useSectorSituation(sectorType, showSectors)
const { data: sectorTimelineData } = useSectorTimeline(sectorType, showSectors)
```

(Import `useSectorSituation, useSectorTimeline` from `@/hooks/useObservatory`.)

- [ ] **Step 2: Compute display sector situation** — current month, or recolored from the sector timeline when the slider is active

```tsx
const displaySectorSituation = useMemo(() => {
  const base = sectorSituationData ?? []
  if (timelinePeriodIndex == null || !timelineData || !sectorTimelineData) return base
  // map slider position (index into station timeline periods) to the sector timeline period string
  const period = timelineData.periods[/* origIdx */ timelinePeriodIndex]
  const sIdx = sectorTimelineData.periods.indexOf(period)
  if (sIdx < 0) return base
  const CLS = ['EXTREMEMENT_BAS','TRES_BAS','BAS','NORMAL','HAUT','TRES_HAUT','EXTREMEMENT_HAUT']
  const TR = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' } as Record<number, string>
  return base.map(s => {
    const ci = sectorTimelineData.sectors[s.code]?.[sIdx]
    const ti = sectorTimelineData.trends[s.code]?.[sIdx]
    const insufficient = ci == null || ci === 7
    return { ...s, situation_class: insufficient ? null : (CLS[ci] as any), trend: (ti != null ? TR[ti] : null) as any, insufficient }
  })
}, [sectorSituationData, sectorTimelineData, timelinePeriodIndex, timelineData])
```

> Note: `timelinePeriodIndex` is already the *original* period index into `timelineData.periods` (the slider passes `filteredIndices[pos]`), so `timelineData.periods[timelinePeriodIndex]` is the selected month — matching how `displayFeatures` indexes `timelineData.stations[code][timelinePeriodIndex]`.

- [ ] **Step 3: Add the sector click handler + pass props to `ObservatoryMap`**

```tsx
const handleSectorClick = useCallback((codes: string[] | null, _name: string | null) => {
  setSelectedStation(null); setSpatialStationCodes(codes); if (!codes) setActiveBbox(null)
}, [])
```

Pass to `<ObservatoryMap ... showSectors={showSectors} sectorSituation={displaySectorSituation} onSectorClick={handleSectorClick} />` and provide the existing `stationsInGeometry` to the map. The map calls back through `stationsInGeometryRef`; wire it by adding a prop `stationsInGeometry={(geom) => stationsInGeometry(geojsonData?.features ?? [], geom)}` and storing it in a ref inside the map (mirror the other `*Ref` props).

- [ ] **Step 4: Type-check + dev build**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Manual verification**

Run the app (project `run` skill or `cd frontend && npm run dev`), open `/observatoire`, open the layers drawer, enable **Météo des nappes**:
- Expected: ~66 sectors colored by class; trend arrows (↑/→/↓) on sectors; grey where insufficient.
- Activate the **Historique** slider and move it: sectors recolor per month; arrows update.
- Click a sector: stations inside are filtered (spatial filter chip appears).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/ObservatoryPage.tsx frontend/src/components/observatory/ObservatoryMap.tsx
git commit -m "feat(meteo): wire sector layer + slider replay + click into ObservatoryPage"
```

---

## Phase 8 — Retire `/meteo`

### Task 16: Redirect route + delete meteo page/components

**Files:**
- Modify: `frontend/src/routes.tsx`
- Delete: `frontend/src/pages/MeteoNappesPage.tsx`, `frontend/src/components/meteo/{NationalBanner,TerritoryChoropleth,TerritoryRanking,OutlookPanel,StationDrillTable}.tsx`

- [ ] **Step 1: Check for other consumers of meteo components + the situation region/department endpoints**

Run: `cd /home/ringuet/time-serie-explo && grep -rn "components/meteo\|MeteoNappesPage\|situation/territories\|situation/national" frontend/src api | grep -v "components/meteo/"`
Expected: only the route + the now-retired page reference them. (If anything else consumes `/situation/territories|national`, keep those endpoints; we only *add* `level=sector`. Note findings here.)

- [ ] **Step 2: Replace the `/meteo` route with a redirect** in `frontend/src/routes.tsx`

```tsx
import { Navigate } from 'react-router-dom'
// ...
{ path: '/meteo', element: <Navigate to="/observatoire" replace /> },
```

Remove the `MeteoNappesPage` import.

- [ ] **Step 3: Delete the page and components**

```bash
git rm frontend/src/pages/MeteoNappesPage.tsx \
  frontend/src/components/meteo/NationalBanner.tsx \
  frontend/src/components/meteo/TerritoryChoropleth.tsx \
  frontend/src/components/meteo/TerritoryRanking.tsx \
  frontend/src/components/meteo/OutlookPanel.tsx \
  frontend/src/components/meteo/StationDrillTable.tsx
```

(If `frontend/src/components/meteo/` has remaining files, list them and delete only those confirmed unused in Step 1.)

- [ ] **Step 4: Remove the nav entry** for `/meteo`

Run: `grep -rn "'/meteo'\|\"/meteo\"\|meteo.title\|MeteoNappes" frontend/src/components frontend/src/lib | grep -iv routes`
Remove the nav item that links to `/meteo` (the audit located it next to the Observatory entry).

- [ ] **Step 5: Type-check + build**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit && npm run build`
Expected: builds with no unresolved imports.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(meteo): retire /meteo page, redirect to observatory"
```

---

## Phase 9 — i18n + final checks

### Task 17: Add i18n strings

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

- [ ] **Step 1: Add keys** under `observatory.drawer` (keep `meteo.trend.*` since the legend reuses them)

FR:
```json
"sectorsMeteo": "Météo des nappes (secteurs)",
"groupMeteo": "Synthèse",
"sectorLegendTitle": "Situation par secteur",
"sectorNoData": "Données insuffisantes"
```
EN:
```json
"sectorsMeteo": "Groundwater weather (sectors)",
"groupMeteo": "Synthesis",
"sectorLegendTitle": "Situation by sector",
"sectorNoData": "Insufficient data"
```

- [ ] **Step 2: Verify no orphaned `meteo.*` keys remain referenced** beyond `meteo.trend.*`

Run: `cd /home/ringuet/time-serie-explo && grep -rn "t('meteo\.\|t(\"meteo\." frontend/src | grep -v "meteo.trend"`
Expected: no results (only `meteo.trend.*` still used by the legend). Remove other `meteo.*` keys from both locale files if unreferenced.

- [ ] **Step 3: Type-check + build**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit && npm run build`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "feat(meteo): i18n for sector layer + legend; prune unused meteo keys"
```

### Task 18: Full backend test sweep + smoke

- [ ] **Step 1: Run the backend tests added by this plan**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/test_geo_sectors.py tests/test_sector_mapping.py tests/test_territory_situation_sector.py tests/test_sectors_endpoint.py tests/test_sectors_timeline_endpoint.py tests/test_timeline_uses_fixed_index.py -v`
Expected: all PASS.

- [ ] **Step 2: Smoke the sector endpoints against the live warehouse**

Run:
```bash
cd /home/ringuet/time-serie-explo && python -c "
from fastapi.testclient import TestClient; from api.main import app
c=TestClient(app)
s=c.get('/api/v1/observatory/situation/sectors?type=piezo').json()
tl=c.get('/api/v1/observatory/situation/sectors/timeline?type=piezo').json()
print('sectors:', len(s), 'with arrows:', sum(1 for x in s if x['tendancy_coord']))
print('timeline periods:', len(tl['periods']), 'sectors:', len(tl['sectors']))
"
```
Expected: ~tens of sectors, most with a `tendancy_coord`; timeline has periods and sector arrays.

- [ ] **Step 3: Commit** (nothing to commit if all green; otherwise fix-forward and commit)

---

## Self-Review notes

- **Spec coverage:** C1→Phase 1; C2→Task 5; C3→Tasks 6-9; C4 (names)→Task 5; C5 (frontend layer)→Tasks 11-15; C6 (slider)→Task 15 Step 2; C7 (retire /meteo)→Task 16; methodology unification→Task 10; edge cases (insufficient grey, centroid fallback, DROM)→Tasks 8/13 (`insufficient` grey + `parseTendancyCoord` null→no arrow; DROM unmapped by design). Out-of-scope forecast is untouched.
- **tendancy_coord fallback:** the spec mentions polygon-centroid fallback; this plan drops the arrow when `tendancy_coord` is missing (simpler, no centroid math). All 66 BRGM sectors carry `tendancy_coord`, so the fallback is dead code in practice — omitted intentionally. If a future refresh yields a null, that sector simply shows no arrow.
- **Type consistency:** `SectorSituation`/`SectorTimeline` defined in Task 12 and consumed in Tasks 13/15; `parseTendancyCoord`/`trendArrowGlyph`/`sectorClassColor`/`SECTOR_INSUFFICIENT_COLOR` defined in Task 11 and imported in Task 13. `_key_rows_by_sector`/`_fetch_station_rows_with_code`/`_fetch_month_rows_with_code` defined in Tasks 7-8 and used in Tasks 8-9.
- **Assumption to verify during execution:** `gold.int_station_era5_mapping.libelle_eh` is the EH label source (confirmed present in the warehouse dbt model). If a station lacks an EH row, it contributes no label and the sector falls back to the next dominant label or `"Secteur {id}"`.
```
