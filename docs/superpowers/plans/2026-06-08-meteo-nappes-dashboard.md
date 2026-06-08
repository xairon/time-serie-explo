# Météo des nappes — decision dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public "Météo des nappes" decision dashboard: a national water-situation verdict (situation + trend) drilling region → department → station, over piezo (IPS) and hydro (SSFI), with a contract-defined but dark AI anticipation layer.

**Architecture:** A new backend router (`observatory_situation.py`) aggregates per-station fixed-reference indices into territory verdicts using pure, unit-tested helpers (median index → class, normalized 3-month trend, coverage gating). A new public React page consumes these aggregates and renders a MapLibre choropleth + ranking, reusing the existing alerts table as the station drill level. The flat `AlertsPage` is absorbed.

**Tech Stack:** FastAPI + SQLAlchemy (sync engine) + Redis cache (`get_cached`); pytest. React 18 + react-query v5 + MapLibre GL v5 + i18next (FR/EN); vitest + @testing-library (set up in this plan).

Spec: `docs/superpowers/specs/2026-06-08-meteo-nappes-decision-dashboard-design.md`

---

## Background the engineer needs

- **Fixed-reference index.** Each station has a current index in `gold.station_current_index(type, code, index_class, index_value)`. `index_value` is a **standardized z-score** (IPS for piezo, SSFI for hydro). `index_class` is one of 7 labels derived from it. Computed against a fixed 1991–2020 reference grid stored in `gold.station_reference_stats(type, code, month, quantile_grid jsonb, baseline_start, baseline_end, flag, n_years)` where `flag ∈ {normale, adaptee, provisoire}`.
- **The 7 classes, ascending wetness:** `EXTREMEMENT_BAS, TRES_BAS, BAS, NORMAL, HAUT, TRES_HAUT, EXTREMEMENT_HAUT` (+ `UNKNOWN`). Class cutoffs as CDF percentiles live in `dashboard/utils/reference.py::CLASS_CUTOFF_PCTL = [4.01, 10.03, 20.05, 79.95, 89.97, 95.99]`.
- **Station dimensions:** `gold.dim_piezo_stations` (`code_bss`, `code_departement`, `nom_departement`, `derniere_mesure`, `nb_mois_total`, `niveau_stddev_global`) and `gold.dim_hydro_stations` (`code_station`, `code_departement`, `nom_departement`, `derniere_mesure`, `nb_mois_total`, `resultat_stddev_global`).
- **Monthly series:** `gold.fct_monthly_chroniques(code_bss, mois, niveau_moyen, niveau_moy_mobile_3m)` and `gold.fct_monthly_hydro(code_station, mois, resultat_moyen, resultat_moy_mobile_3m)`.
- **Reliability rule (existing convention, observatory_common.py:78):** `nb_mois >= 120 → fiable`, `>= 60 → indicatif`, else `insuffisant`.
- **Cache:** `from dashboard.utils.cache import get_cached` → `get_cached(prefix, params_dict, ttl_seconds, fetch_fn)`.
- **Sync engine:** `from api.database import get_brgm_sync_engine`; use `with engine.connect() as conn: conn.execute(text(sql), bind)`. Do **not** dispose (pooled).
- **Tests run with no warehouse DB.** Backend TDD lives in **pure helpers** (`dashboard/utils/`, `api/data/`), mirroring `tests/test_reference.py`. Router endpoints get an import/registration smoke test only; their SQL is validated manually against the live warehouse (commands provided in Task 5).

## File structure (created/modified)

Backend:
- Create `dashboard/utils/territory_situation.py` — pure aggregation + trend + class helpers.
- Create `api/data/__init__.py` and `api/data/territories_fr.py` — department→region static lookup.
- Modify `api/schemas/observatory.py` — add `Outlook`, `TerritorySituation`, `NationalSituation`.
- Create `api/routers/observatory_situation.py` — the two aggregate endpoints.
- Modify `api/main.py` — register the router + warm its cache at startup.
- Create `tests/test_territory_situation.py`, `tests/test_territories_fr.py`; modify `tests/test_schemas.py`.

Frontend:
- Modify `frontend/package.json`, create `frontend/vitest.config.ts`, `frontend/src/test/setup.ts` — test infra.
- Create `frontend/src/lib/situation-api.ts`, add types to `frontend/src/lib/observatory-types.ts`.
- Create `frontend/src/lib/situation-format.ts` (+ `frontend/src/lib/situation-format.test.ts`) — pure UI mappers.
- Create `frontend/src/assets/geo/regions.geojson` and `departements.geojson` — bundled boundaries.
- Create components under `frontend/src/components/meteo/`: `NationalBanner.tsx`, `TerritoryChoropleth.tsx`, `TerritoryRanking.tsx`, `OutlookPanel.tsx`, `StationDrillTable.tsx`.
- Create `frontend/src/pages/MeteoNappesPage.tsx`.
- Modify `frontend/src/routes.tsx` (route), `frontend/src/components/layout/TopNav.tsx` (nav), `frontend/src/i18n/locales/fr.json` + `en.json` (keys).
- Delete `frontend/src/pages/AlertsPage.tsx` (absorbed into `StationDrillTable`).

---

## Task 1: Territory aggregation pure helpers

**Files:**
- Create: `dashboard/utils/territory_situation.py`
- Test: `tests/test_territory_situation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_territory_situation.py
import math
from dashboard.utils.territory_situation import (
    zscore_to_class, aggregate_situation, aggregate_trend,
    MIN_ELIGIBLE, TREND_STABLE_BAND, CLASS_ORDER,
)


def test_zscore_to_class_spans_all_classes():
    # very dry -> EXTREMEMENT_BAS, very wet -> EXTREMEMENT_HAUT, ~0 -> NORMAL
    assert zscore_to_class(-3.0) == "EXTREMEMENT_BAS"
    assert zscore_to_class(0.0) == "NORMAL"
    assert zscore_to_class(3.0) == "EXTREMEMENT_HAUT"
    # monotonic: class index never decreases as z increases
    idx = [CLASS_ORDER.index(zscore_to_class(z)) for z in [-3, -2, -1, 0, 1, 2, 3]]
    assert idx == sorted(idx)


def test_zscore_to_class_none():
    assert zscore_to_class(None) == "UNKNOWN"


def test_aggregate_situation_uses_median_and_distribution():
    # five eligible stations, median z = 0 -> NORMAL
    res = aggregate_situation([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert res["situation_class"] == "NORMAL"
    assert res["n_eligible"] == 5
    # 2 stations strictly below normal band (EXTREMEMENT_BAS/TRES_BAS/BAS)
    assert res["pct_below_normal"] == 40.0
    assert sum(res["distribution"].values()) == 5


def test_aggregate_situation_insufficient_coverage():
    res = aggregate_situation([0.0] * (MIN_ELIGIBLE - 1))
    assert res["situation_class"] is None
    assert res["insufficient"] is True


def test_aggregate_trend_directions():
    assert aggregate_trend([TREND_STABLE_BAND + 0.2] * 3) == "hausse"
    assert aggregate_trend([-(TREND_STABLE_BAND + 0.2)] * 3) == "baisse"
    assert aggregate_trend([0.0, 0.0, 0.0]) == "stable"


def test_aggregate_trend_empty():
    assert aggregate_trend([]) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_territory_situation.py -v`
Expected: FAIL — `ModuleNotFoundError: dashboard.utils.territory_situation`.

- [ ] **Step 3: Write the minimal implementation**

```python
# dashboard/utils/territory_situation.py
"""Pure helpers to aggregate per-station fixed-reference indices into a
territory verdict (situation class + trend) and a coverage flag.

No DB, no Streamlit. The routers supply the per-station numbers.
The situation class of a territory is the median station z-score mapped to the
7 BRGM classes; the trend is the median of per-station normalized 3-month deltas.
"""
from __future__ import annotations

import statistics
from scipy import stats

from dashboard.utils.reference import CLASS_CUTOFF_PCTL

CLASS_ORDER = [
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL",
    "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT",
]
BELOW_NORMAL = {"EXTREMEMENT_BAS", "TRES_BAS", "BAS"}

# z-score thresholds at the BRGM CDF cutoffs (6 thresholds -> 7 classes)
_Z_CUTOFFS = [float(stats.norm.ppf(p / 100.0)) for p in CLASS_CUTOFF_PCTL]

# A territory needs at least this many eligible stations to earn a verdict.
MIN_ELIGIBLE = 3
# |median delta-z| below this band over 3 months reads as "stable".
TREND_STABLE_BAND = 0.5


def zscore_to_class(z) -> str:
    if z is None:
        return "UNKNOWN"
    for i, thr in enumerate(_Z_CUTOFFS):
        if z < thr:
            return CLASS_ORDER[i]
    return CLASS_ORDER[-1]


def aggregate_situation(index_values: list[float]) -> dict:
    vals = [v for v in index_values if v is not None]
    if len(vals) < MIN_ELIGIBLE:
        return {
            "situation_class": None, "insufficient": True,
            "n_eligible": len(vals), "pct_below_normal": None,
            "distribution": {c: 0 for c in CLASS_ORDER},
        }
    median_z = statistics.median(vals)
    classes = [zscore_to_class(v) for v in vals]
    distribution = {c: classes.count(c) for c in CLASS_ORDER}
    below = sum(1 for c in classes if c in BELOW_NORMAL)
    return {
        "situation_class": zscore_to_class(median_z),
        "insufficient": False,
        "n_eligible": len(vals),
        "pct_below_normal": round(100.0 * below / len(vals), 1),
        "distribution": distribution,
    }


def aggregate_trend(deltas: list[float]) -> str | None:
    vals = [d for d in deltas if d is not None]
    if not vals:
        return None
    m = statistics.median(vals)
    if m > TREND_STABLE_BAND:
        return "hausse"
    if m < -TREND_STABLE_BAND:
        return "baisse"
    return "stable"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_territory_situation.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add dashboard/utils/territory_situation.py tests/test_territory_situation.py
git commit -m "feat(situation): pure territory aggregation + trend helpers"
```

---

## Task 2: Department → region static lookup

**Files:**
- Create: `api/data/__init__.py`, `api/data/territories_fr.py`
- Test: `tests/test_territories_fr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_territories_fr.py
from api.data.territories_fr import (
    DEPT_TO_REGION, REGION_NAMES, region_of, DEPARTMENT_NAMES,
)


def test_every_dept_maps_to_a_known_region():
    for dept, region_code in DEPT_TO_REGION.items():
        assert region_code in REGION_NAMES, f"{dept} -> unknown region {region_code}"


def test_region_of_known_and_unknown():
    assert region_of("45") == "24"          # Loiret -> Centre-Val de Loire
    assert region_of("75") == "11"          # Paris -> Île-de-France
    assert region_of("999") is None


def test_metropolitan_and_corsica_present():
    # 96 metro depts (01..95 minus 20, plus 2A/2B) + 5 DROM = 101
    assert "2A" in DEPT_TO_REGION and "2B" in DEPT_TO_REGION
    assert "20" not in DEPT_TO_REGION
    assert len(DEPT_TO_REGION) == 101


def test_department_names_cover_all_mapped_depts():
    for dept in DEPT_TO_REGION:
        assert dept in DEPARTMENT_NAMES
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_territories_fr.py -v`
Expected: FAIL — `ModuleNotFoundError: api.data.territories_fr`.

- [ ] **Step 3: Write the implementation**

Create empty `api/data/__init__.py`. Then create `api/data/territories_fr.py` with the full INSEE department→region mapping (region codes match the bundled `regions.geojson` `code` property). Source the codes from INSEE COG 2024.

```python
# api/data/territories_fr.py
"""Static French department -> region (INSEE COG) lookup. No DB.

Region codes match the bundled frontend regions.geojson `code` property.
"""
from __future__ import annotations

REGION_NAMES: dict[str, str] = {
    "11": "Île-de-France", "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté", "28": "Normandie",
    "32": "Hauts-de-France", "44": "Grand Est", "52": "Pays de la Loire",
    "53": "Bretagne", "75": "Nouvelle-Aquitaine", "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes", "93": "Provence-Alpes-Côte d'Azur",
    "94": "Corse", "01": "Guadeloupe", "02": "Martinique",
    "03": "Guyane", "04": "La Réunion", "06": "Mayotte",
}

# dept code -> region code
DEPT_TO_REGION: dict[str, str] = {
    # Auvergne-Rhône-Alpes (84)
    "01": "84", "03": "84", "07": "84", "15": "84", "26": "84", "38": "84",
    "42": "84", "43": "84", "63": "84", "69": "84", "73": "84", "74": "84",
    # Bourgogne-Franche-Comté (27)
    "21": "27", "25": "27", "39": "27", "58": "27", "70": "27", "71": "27",
    "89": "27", "90": "27",
    # Bretagne (53)
    "22": "53", "29": "53", "35": "53", "56": "53",
    # Centre-Val de Loire (24)
    "18": "24", "28": "24", "36": "24", "37": "24", "41": "24", "45": "24",
    # Corse (94)
    "2A": "94", "2B": "94",
    # Grand Est (44)
    "08": "44", "10": "44", "51": "44", "52": "44", "54": "44", "55": "44",
    "57": "44", "67": "44", "68": "44", "88": "44",
    # Hauts-de-France (32)
    "02": "32", "59": "32", "60": "32", "62": "32", "80": "32",
    # Île-de-France (11)
    "75": "11", "77": "11", "78": "11", "91": "11", "92": "11", "93": "11",
    "94": "11", "95": "11",
    # Normandie (28)
    "14": "28", "27": "28", "50": "28", "61": "28", "76": "28",
    # Nouvelle-Aquitaine (75)
    "16": "75", "17": "75", "19": "75", "23": "75", "24": "75", "33": "75",
    "40": "75", "47": "75", "64": "75", "79": "75", "86": "75", "87": "75",
    # Occitanie (76)
    "09": "76", "11": "76", "12": "76", "30": "76", "31": "76", "32": "76",
    "34": "76", "46": "76", "48": "76", "65": "76", "66": "76", "81": "76",
    "82": "76",
    # Pays de la Loire (52)
    "44": "52", "49": "52", "53": "52", "72": "52", "85": "52",
    # Provence-Alpes-Côte d'Azur (93)
    "04": "93", "05": "93", "06": "93", "13": "93", "83": "93", "84": "93",
    # DROM (region code == dept code domain)
    "971": "01", "972": "02", "973": "03", "974": "04", "976": "06",
}

DEPARTMENT_NAMES: dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche",
    "08": "Ardennes", "09": "Ariège", "10": "Aube", "11": "Aude",
    "12": "Aveyron", "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal",
    "16": "Charente", "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot",
    "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan",
    "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise", "61": "Orne",
    "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône",
    "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie",
    "74": "Haute-Savoie", "75": "Paris", "76": "Seine-Maritime",
    "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres",
    "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var",
    "84": "Vaucluse", "85": "Vendée", "86": "Vienne", "87": "Haute-Vienne",
    "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise", "971": "Guadeloupe",
    "972": "Martinique", "973": "Guyane", "974": "La Réunion",
    "976": "Mayotte",
}


def region_of(dept_code: str | None) -> str | None:
    if dept_code is None:
        return None
    return DEPT_TO_REGION.get(dept_code)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_territories_fr.py -v`
Expected: PASS (4 passed). If `test_metropolitan_and_corsica_present` fails on the count, fix the mapping until exactly 101 entries exist (96 metro + 5 DROM).

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add api/data/__init__.py api/data/territories_fr.py tests/test_territories_fr.py
git commit -m "feat(situation): department->region static lookup (INSEE COG)"
```

---

## Task 3: Pydantic schemas

**Files:**
- Modify: `api/schemas/observatory.py` (append after `NationalStats`, before the ERA5 section)
- Test: `tests/test_schemas.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_schemas.py`)

```python
def test_territory_situation_schema_defaults():
    from api.schemas.observatory import TerritorySituation
    t = TerritorySituation(
        level="region", code="24", name="Centre-Val de Loire", type="piezo",
        situation_class="BAS", trend="baisse", pct_below_normal=42.0,
        n_eligible=18, n_provisoire=3,
        distribution={"BAS": 10, "NORMAL": 8}, insufficient=False,
    )
    assert t.outlook is None          # AI layer dark by default
    assert t.code == "24"


def test_national_situation_schema():
    from api.schemas.observatory import NationalSituation
    n = NationalSituation(
        type="hydro", situation_class="NORMAL", trend="stable",
        pct_below_normal=12.0, n_eligible=900, n_provisoire=120,
        distribution={"NORMAL": 700}, insufficient=False,
    )
    assert n.trend == "stable"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_schemas.py -k "situation" -v`
Expected: FAIL — `ImportError: cannot import name 'TerritorySituation'`.

- [ ] **Step 3: Add the schemas** (in `api/schemas/observatory.py`, after `NationalStats`)

```python
class Outlook(BaseModel):
    """AI anticipation contract. Null in v1 (layer dark)."""
    horizon_months: int
    situation_class: str | None = None
    trend: str | None = None
    confidence: float | None = None
    coverage_pct: float | None = None


class TerritorySituation(BaseModel):
    level: Literal["region", "department"]
    code: str
    name: str
    type: Literal["piezo", "hydro"]
    situation_class: str | None = None
    trend: str | None = None              # "hausse" | "stable" | "baisse" | None
    pct_below_normal: float | None = None
    n_eligible: int = 0
    n_provisoire: int = 0
    distribution: dict[str, int] = {}
    insufficient: bool = False
    outlook: Outlook | None = None


class NationalSituation(BaseModel):
    type: Literal["piezo", "hydro"]
    situation_class: str | None = None
    trend: str | None = None
    pct_below_normal: float | None = None
    n_eligible: int = 0
    n_provisoire: int = 0
    distribution: dict[str, int] = {}
    insufficient: bool = False
    outlook: Outlook | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_schemas.py -k "situation" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add api/schemas/observatory.py tests/test_schemas.py
git commit -m "feat(situation): TerritorySituation/NationalSituation/Outlook schemas"
```

---

## Task 4: Situation router

**Files:**
- Create: `api/routers/observatory_situation.py`
- Modify: `api/main.py` (import, include_router, warm cache)
- Test: `tests/test_situation_router.py`

This router owns the SQL. The defensible math is delegated to the Task 1 helpers. Trend uses the standardized z-score delta over 3 months: `z_latest` is the station's current `index_value`; `z_lag` is computed in Python by re-scoring the value from 3 months earlier against that station's reference grid for the lag calendar month (`reference.value_to_zscore`).

- [ ] **Step 1: Write the failing test** (router smoke — no DB needed)

```python
# tests/test_situation_router.py
from api.routers.observatory_situation import router, _eligible_rows_to_territories


def test_router_mounts_situation_paths():
    paths = {r.path for r in router.routes}
    assert "/api/v1/observatory/situation/national" in paths
    assert "/api/v1/observatory/situation/territories" in paths


def test_eligible_rows_to_territories_groups_and_aggregates():
    # rows: (territory_code, territory_name, index_value, delta_z, flag)
    rows = [
        ("24", "Centre-Val de Loire", -2.0, -0.9, "normale"),
        ("24", "Centre-Val de Loire", -1.0, -0.7, "normale"),
        ("24", "Centre-Val de Loire", 0.0, -0.6, "adaptee"),
        ("24", "Centre-Val de Loire", None, None, "provisoire"),  # excluded, counted
        ("11", "Île-de-France", 1.5, 0.0, "normale"),             # 1 station -> insufficient
    ]
    out = _eligible_rows_to_territories(rows, level="region", type_="piezo")
    by_code = {t["code"]: t for t in out}
    assert by_code["24"]["n_eligible"] == 3
    assert by_code["24"]["n_provisoire"] == 1
    assert by_code["24"]["trend"] == "baisse"
    assert by_code["24"]["situation_class"] == "TRES_BAS"   # median z = -1.0
    assert by_code["11"]["insufficient"] is True
    assert by_code["11"]["outlook"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_situation_router.py -v`
Expected: FAIL — `ModuleNotFoundError: api.routers.observatory_situation`.

- [ ] **Step 3: Write the router**

```python
# api/routers/observatory_situation.py
"""Observatory situation router — territory + national 'météo des nappes' verdicts.

Aggregates per-station fixed-reference indices (IPS/SSFI) into a situation class
+ trend per territory, reusing pure helpers in dashboard.utils.territory_situation.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy import text

from api.data.territories_fr import (
    DEPARTMENT_NAMES, REGION_NAMES, region_of,
)
from api.database import get_brgm_sync_engine
from api.schemas.observatory import NationalSituation, TerritorySituation
from dashboard.utils.cache import get_cached
from dashboard.utils.territory_situation import (
    aggregate_situation, aggregate_trend,
)

router = APIRouter(prefix="/api/v1/observatory", tags=["observatory-situation"])

SITUATION_TTL = 21600
RELIABLE_MIN_MOIS = 60   # >=60 months = at least 'indicatif'; excludes 'insuffisant'

# Per-type SQL: returns one row per eligible-or-provisoire station with its current
# z (index_value), the value 3 months before the ref month, that month's grid, and
# its reference flag + department.
_STATION_SQL = {
    "piezo": """
        SELECT s.code_departement AS dept,
               sci.index_value AS z_latest,
               rs.flag AS flag,
               lag.niveau_moyen AS lag_value,
               glag.quantile_grid AS lag_grid
        FROM gold.dim_piezo_stations s
        JOIN gold.station_current_index sci
          ON sci.type = 'piezo' AND sci.code = s.code_bss
        LEFT JOIN gold.station_reference_stats rs
          ON rs.type = 'piezo' AND rs.code = s.code_bss
         AND rs.month = EXTRACT(MONTH FROM sci.index_ref_month)::int
        LEFT JOIN gold.fct_monthly_chroniques lag
          ON lag.code_bss = s.code_bss
         AND lag.mois = (date_trunc('month', sci.index_ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'piezo' AND glag.code = s.code_bss
         AND glag.month = EXTRACT(MONTH FROM (sci.index_ref_month - INTERVAL '3 months'))::int
        WHERE s.code_departement IS NOT NULL
          AND s.nb_mois_total >= :min_mois
          AND sci.index_class IS NOT NULL AND sci.index_class <> 'UNKNOWN'
    """,
    "hydro": """
        SELECT s.code_departement AS dept,
               sci.index_value AS z_latest,
               rs.flag AS flag,
               lag.resultat_moyen AS lag_value,
               glag.quantile_grid AS lag_grid
        FROM gold.dim_hydro_stations s
        JOIN gold.station_current_index sci
          ON sci.type = 'hydro' AND sci.code = s.code_station
        LEFT JOIN gold.station_reference_stats rs
          ON rs.type = 'hydro' AND rs.code = s.code_station
         AND rs.month = EXTRACT(MONTH FROM sci.index_ref_month)::int
        LEFT JOIN gold.fct_monthly_hydro lag
          ON lag.code_station = s.code_station
         AND lag.mois = (date_trunc('month', sci.index_ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'hydro' AND glag.code = s.code_station
         AND glag.month = EXTRACT(MONTH FROM (sci.index_ref_month - INTERVAL '3 months'))::int
        WHERE s.code_departement IS NOT NULL
          AND s.nb_mois_total >= :min_mois
          AND sci.index_class IS NOT NULL AND sci.index_class <> 'UNKNOWN'
    """,
}


def _fetch_station_rows(type_: str) -> list[tuple]:
    """-> list of (dept, z_latest, delta_z, flag) for all candidate stations."""
    from dashboard.utils.reference import value_to_zscore

    engine = get_brgm_sync_engine()
    out: list[tuple] = []
    with engine.connect() as conn:
        result = conn.execute(text(_STATION_SQL[type_]), {"min_mois": RELIABLE_MIN_MOIS})
        for r in result.mappings():
            z_latest = r["z_latest"]
            delta_z = None
            if z_latest is not None and r["lag_value"] is not None and r["lag_grid"]:
                z_lag = value_to_zscore(float(r["lag_value"]), list(r["lag_grid"]))
                if z_lag is not None:
                    delta_z = float(z_latest) - z_lag
            out.append((r["dept"], z_latest, delta_z, r["flag"]))
    return out


def _eligible_rows_to_territories(rows, level, type_) -> list[dict]:
    """rows: iterable of (territory_code, territory_name, z, delta_z, flag).

    Groups by territory_code, excludes provisoire/UNKNOWN from the verdict but
    counts them, and produces a TerritorySituation-shaped dict per territory.
    """
    groups: dict[str, dict] = {}
    for code, name, z, delta_z, flag in rows:
        g = groups.setdefault(code, {"name": name, "z": [], "dz": [], "prov": 0})
        if flag in ("normale", "adaptee") and z is not None:
            g["z"].append(float(z))
            if delta_z is not None:
                g["dz"].append(delta_z)
        else:
            g["prov"] += 1

    territories = []
    for code, g in groups.items():
        sit = aggregate_situation(g["z"])
        territories.append({
            "level": level, "code": code, "name": g["name"], "type": type_,
            "situation_class": sit["situation_class"],
            "trend": aggregate_trend(g["dz"]),
            "pct_below_normal": sit["pct_below_normal"],
            "n_eligible": sit["n_eligible"],
            "n_provisoire": g["prov"],
            "distribution": {k: v for k, v in sit["distribution"].items() if v},
            "insufficient": sit["insufficient"],
            "outlook": None,
        })
    return territories


@router.get("/situation/territories", response_model=list[TerritorySituation])
def get_territory_situation(
    level: Literal["region", "department"] = Query("region"),
    type: Literal["piezo", "hydro"] = Query("piezo"),
):
    def fetch():
        station_rows = _fetch_station_rows(type)
        keyed = []
        for dept, z, dz, flag in station_rows:
            if level == "region":
                code = region_of(dept)
                if code is None:
                    continue
                name = REGION_NAMES[code]
            else:
                code = dept
                name = DEPARTMENT_NAMES.get(dept, dept)
            keyed.append((code, name, z, dz, flag))
        out = _eligible_rows_to_territories(keyed, level, type)
        out.sort(key=lambda t: t["name"])
        return out

    return get_cached("obs_situation_territories", {"level": level, "type": type}, SITUATION_TTL, fetch)


@router.get("/situation/national", response_model=NationalSituation)
def get_national_situation(type: Literal["piezo", "hydro"] = Query("piezo")):
    def fetch():
        station_rows = _fetch_station_rows(type)
        keyed = [("FR", "France", z, dz, flag) for dept, z, dz, flag in station_rows]
        agg = _eligible_rows_to_territories(keyed, "region", type)[0]
        return {
            "type": type,
            "situation_class": agg["situation_class"], "trend": agg["trend"],
            "pct_below_normal": agg["pct_below_normal"], "n_eligible": agg["n_eligible"],
            "n_provisoire": agg["n_provisoire"], "distribution": agg["distribution"],
            "insufficient": agg["insufficient"], "outlook": None,
        }

    return get_cached("obs_situation_national", {"type": type}, SITUATION_TTL, fetch)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_situation_router.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Register the router + warm cache** in `api/main.py`

In the `from api.routers import ...` block (api/main.py:14) add `observatory_situation`. After the other `app.include_router(...)` observatory lines add:

```python
app.include_router(observatory_situation.router)
```

In the startup lifespan warm-up (search for where `obs_national_stats` / national stats is warmed), add best-effort warm calls so the first decision-maker hit is fast:

```python
    for _t in ("piezo", "hydro"):
        try:
            observatory_situation.get_national_situation(type=_t)
            observatory_situation.get_territory_situation(level="region", type=_t)
        except Exception:
            logger.warning("situation warm-up failed for %s", _t, exc_info=True)
```

- [ ] **Step 6: Verify the app imports cleanly**

Run: `cd ~/time-serie-explo && python -c "from api.main import app; print('paths', sum('/situation/' in r.path for r in app.routes))"`
Expected: prints `paths 2`.

- [ ] **Step 7: Manual warehouse validation (no automated test — needs live DB)**

With the backend container running against the BRGM warehouse:

```bash
curl -s 'http://localhost:49514/api/v1/observatory/situation/national?type=piezo' | python -m json.tool
curl -s 'http://localhost:49514/api/v1/observatory/situation/territories?level=region&type=piezo' | python -m json.tool | head -40
```

Expected: national returns one verdict with `situation_class`, `trend`, `pct_below_normal`, non-zero `n_eligible`; territories returns ~13 regions, some possibly `insufficient: true`. Sanity-check one region's `pct_below_normal` against the Observatory map. If the trend query is slow (>2s uncached), note it for the precompute follow-up; it is acceptable behind the 6h cache.

- [ ] **Step 8: Commit**

```bash
cd ~/time-serie-explo
git add api/routers/observatory_situation.py api/main.py tests/test_situation_router.py
git commit -m "feat(situation): territory + national météo-des-nappes endpoints"
```

---

## Task 5: Frontend test infrastructure (vitest)

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`, `frontend/src/test/setup.ts`

- [ ] **Step 1: Install dev deps**

```bash
cd ~/time-serie-explo/frontend
npm install -D vitest@^2 @testing-library/react@^16 @testing-library/jest-dom@^6 jsdom@^25
```

- [ ] **Step 2: Add the test script** to `frontend/package.json` `"scripts"`:

```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 3: Create `frontend/vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
```

- [ ] **Step 4: Create `frontend/src/test/setup.ts`**

```ts
import '@testing-library/jest-dom/vitest'
```

- [ ] **Step 5: Add a sanity test** `frontend/src/test/sanity.test.ts`

```ts
import { describe, it, expect } from 'vitest'
describe('vitest', () => { it('runs', () => { expect(1 + 1).toBe(2) }) })
```

- [ ] **Step 6: Run it**

Run: `cd ~/time-serie-explo/frontend && npm test`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
cd ~/time-serie-explo
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts frontend/src/test/
git commit -m "test(frontend): set up vitest + testing-library"
```

---

## Task 6: Frontend types + API client

**Files:**
- Modify: `frontend/src/lib/observatory-types.ts` (append)
- Create: `frontend/src/lib/situation-api.ts`

- [ ] **Step 1: Append types** to `frontend/src/lib/observatory-types.ts`

```ts
export type SituationClass =
  | 'EXTREMEMENT_BAS' | 'TRES_BAS' | 'BAS' | 'NORMAL'
  | 'HAUT' | 'TRES_HAUT' | 'EXTREMEMENT_HAUT'
export type Trend = 'hausse' | 'stable' | 'baisse'

export interface Outlook {
  horizon_months: number
  situation_class: SituationClass | null
  trend: Trend | null
  confidence: number | null
  coverage_pct: number | null
}

export interface TerritorySituation {
  level: 'region' | 'department'
  code: string
  name: string
  type: 'piezo' | 'hydro'
  situation_class: SituationClass | null
  trend: Trend | null
  pct_below_normal: number | null
  n_eligible: number
  n_provisoire: number
  distribution: Record<string, number>
  insufficient: boolean
  outlook: Outlook | null
}

export interface NationalSituation {
  type: 'piezo' | 'hydro'
  situation_class: SituationClass | null
  trend: Trend | null
  pct_below_normal: number | null
  n_eligible: number
  n_provisoire: number
  distribution: Record<string, number>
  insufficient: boolean
  outlook: Outlook | null
}
```

- [ ] **Step 2: Create `frontend/src/lib/situation-api.ts`**

Reuse the existing `fetchJson` helper exported context by importing from `observatory-api`. (If `fetchJson` is not exported, export it from `observatory-api.ts` first — add `export` to its declaration.)

```ts
import { fetchJson } from './observatory-api'
import type { NationalSituation, TerritorySituation } from './observatory-types'

export const situationApi = {
  national: (type: 'piezo' | 'hydro') =>
    fetchJson<NationalSituation>('/observatory/situation/national', { type }),
  territories: (level: 'region' | 'department', type: 'piezo' | 'hydro') =>
    fetchJson<TerritorySituation[]>('/observatory/situation/territories', { level, type }),
}
```

- [ ] **Step 3: Ensure `fetchJson` is exported** from `frontend/src/lib/observatory-api.ts` — change `async function fetchJson` to `export async function fetchJson`.

- [ ] **Step 4: Type-check**

Run: `cd ~/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/lib/observatory-types.ts frontend/src/lib/situation-api.ts frontend/src/lib/observatory-api.ts
git commit -m "feat(situation): frontend types + API client"
```

---

## Task 7: Pure UI mappers (class color, trend glyph, verdict label)

**Files:**
- Create: `frontend/src/lib/situation-format.ts`
- Test: `frontend/src/lib/situation-format.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// frontend/src/lib/situation-format.test.ts
import { describe, it, expect } from 'vitest'
import { classColor, trendGlyph, INSUFFICIENT_COLOR } from './situation-format'

describe('situation-format', () => {
  it('maps a class to its palette color', () => {
    expect(classColor('BAS')).toBe('#f97316')
    expect(classColor('NORMAL')).toBe('#10b981')
  })
  it('falls back to insufficient grey for null', () => {
    expect(classColor(null)).toBe(INSUFFICIENT_COLOR)
  })
  it('maps trend to an arrow glyph', () => {
    expect(trendGlyph('hausse')).toBe('▲')
    expect(trendGlyph('baisse')).toBe('▼')
    expect(trendGlyph('stable')).toBe('▬')
    expect(trendGlyph(null)).toBe('—')
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo/frontend && npm test -- situation-format`
Expected: FAIL — cannot resolve `./situation-format`.

- [ ] **Step 3: Write the implementation**

```ts
// frontend/src/lib/situation-format.ts
import { CLASSIFICATION_COLORS } from './observatory-constants'
import type { SituationClass, Trend } from './observatory-types'

export const INSUFFICIENT_COLOR = '#374151'  // muted slate for "données insuffisantes"

export function classColor(cls: SituationClass | null): string {
  if (!cls) return INSUFFICIENT_COLOR
  return CLASSIFICATION_COLORS[cls] ?? INSUFFICIENT_COLOR
}

export function trendGlyph(trend: Trend | null): string {
  switch (trend) {
    case 'hausse': return '▲'
    case 'baisse': return '▼'
    case 'stable': return '▬'
    default: return '—'
  }
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd ~/time-serie-explo/frontend && npm test -- situation-format`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/lib/situation-format.ts frontend/src/lib/situation-format.test.ts
git commit -m "feat(situation): pure UI mappers for class color + trend glyph"
```

---

## Task 8: i18n keys

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

- [ ] **Step 1: Add the `meteo` block + nav key to `fr.json`** (inside the root object; add `"meteo"` key to `nav`)

```json
  "nav": {
    "meteo": "Météo des nappes"
  },
  "meteo": {
    "title": "Météo des nappes",
    "subtitle": "Situation et tendance de la ressource en eau par territoire",
    "tabPiezo": "Nappes",
    "tabHydro": "Cours d'eau",
    "belowNormal": "{{pct}} % sous la normale",
    "departmentsInAlert": "{{count}} départements en alerte",
    "trendGeneral": "Tendance générale",
    "insufficient": "Données insuffisantes",
    "insufficientHint": "Pas assez de stations fiables pour un verdict",
    "provisoireHint": "{{count}} stations non classées (référence provisoire)",
    "rankingTitle": "Territoires les plus critiques",
    "backToRegions": "← Toutes les régions",
    "stationsIn": "Stations — {{name}}",
    "outlookTitle": "Anticipation IA",
    "outlookSoon": "Bientôt : projection à 1–3 mois par les modèles de l'app",
    "trend": { "hausse": "en hausse", "stable": "stable", "baisse": "en baisse" },
    "class": {
      "EXTREMEMENT_BAS": "Extrêmement bas", "TRES_BAS": "Très bas", "BAS": "Bas",
      "NORMAL": "Normal", "HAUT": "Haut", "TRES_HAUT": "Très haut",
      "EXTREMEMENT_HAUT": "Extrêmement haut"
    },
    "loadError": "Impossible de charger la météo des nappes"
  }
```

(Merge the `nav.meteo` key into the existing `nav` object rather than duplicating `nav`.)

- [ ] **Step 2: Add the parallel `meteo` block + `nav.meteo` to `en.json`**

```json
  "nav": {
    "meteo": "Groundwater weather"
  },
  "meteo": {
    "title": "Groundwater weather",
    "subtitle": "Water-resource situation and trend by territory",
    "tabPiezo": "Aquifers",
    "tabHydro": "Rivers",
    "belowNormal": "{{pct}}% below normal",
    "departmentsInAlert": "{{count}} departments in alert",
    "trendGeneral": "Overall trend",
    "insufficient": "Insufficient data",
    "insufficientHint": "Not enough reliable stations for a verdict",
    "provisoireHint": "{{count}} unclassified stations (provisional reference)",
    "rankingTitle": "Most critical territories",
    "backToRegions": "← All regions",
    "stationsIn": "Stations — {{name}}",
    "outlookTitle": "AI anticipation",
    "outlookSoon": "Coming soon: 1–3 month projection from the app's models",
    "trend": { "hausse": "rising", "stable": "stable", "baisse": "falling" },
    "class": {
      "EXTREMEMENT_BAS": "Extremely low", "TRES_BAS": "Very low", "BAS": "Low",
      "NORMAL": "Normal", "HAUT": "High", "TRES_HAUT": "Very high",
      "EXTREMEMENT_HAUT": "Extremely high"
    },
    "loadError": "Could not load groundwater weather"
  }
```

- [ ] **Step 3: Validate JSON**

Run: `cd ~/time-serie-explo/frontend && node -e "JSON.parse(require('fs').readFileSync('src/i18n/locales/fr.json')); JSON.parse(require('fs').readFileSync('src/i18n/locales/en.json')); console.log('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "i18n(meteo): FR/EN keys for the météo-des-nappes dashboard"
```

---

## Task 9: Bundle France region + department boundaries

**Files:**
- Create: `frontend/src/assets/geo/regions.geojson`, `frontend/src/assets/geo/departements.geojson`

- [ ] **Step 1: Download license-clean boundaries** (france-geojson, ODbL; properties carry `code` + `nom`)

```bash
cd ~/time-serie-explo/frontend/src/assets/geo 2>/dev/null || mkdir -p ~/time-serie-explo/frontend/src/assets/geo && cd ~/time-serie-explo/frontend/src/assets/geo
curl -sL -o regions.geojson      https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson
curl -sL -o departements.geojson https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson
```

- [ ] **Step 2: Verify the property keys** (the choropleth joins on `code`)

Run: `cd ~/time-serie-explo/frontend/src/assets/geo && node -e "const r=require('./regions.geojson'); console.log(r.features[0].properties)"`
Expected: an object with `code` and `nom` (e.g. `{ code: '11', nom: 'Île-de-France' }`). If the keys differ, note them — `TerritoryChoropleth` (Task 11) joins region/dept `code` to the property named here.

- [ ] **Step 3: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/assets/geo/regions.geojson frontend/src/assets/geo/departements.geojson
git commit -m "assets(meteo): bundled France region + department boundaries (ODbL)"
```

---

## Task 10: NationalBanner component

**Files:**
- Create: `frontend/src/components/meteo/NationalBanner.tsx`
- Test: `frontend/src/components/meteo/NationalBanner.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/meteo/NationalBanner.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { NationalBanner } from './NationalBanner'

const base = {
  type: 'piezo' as const, situation_class: 'BAS' as const, trend: 'baisse' as const,
  pct_below_normal: 42, n_eligible: 900, n_provisoire: 100, distribution: {}, insufficient: false, outlook: null,
}

const renderWith = (ui: React.ReactElement) =>
  render(<I18nextProvider i18n={i18n}>{ui}</I18nextProvider>)

describe('NationalBanner', () => {
  it('shows the verdict class label and headline number', () => {
    renderWith(<NationalBanner data={base} departmentsInAlert={7} />)
    expect(screen.getByText(/Bas/)).toBeInTheDocument()
    expect(screen.getByText(/42 % sous la normale/)).toBeInTheDocument()
  })
  it('renders an insufficient state without a class', () => {
    renderWith(<NationalBanner data={{ ...base, situation_class: null, insufficient: true }} departmentsInAlert={0} />)
    expect(screen.getByText(/Données insuffisantes/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo/frontend && npm test -- NationalBanner`
Expected: FAIL — cannot resolve `./NationalBanner`.

- [ ] **Step 3: Write the component**

```tsx
// frontend/src/components/meteo/NationalBanner.tsx
import { useTranslation } from 'react-i18next'
import type { NationalSituation } from '@/lib/observatory-types'
import { classColor, trendGlyph } from '@/lib/situation-format'

export function NationalBanner({ data, departmentsInAlert }: { data: NationalSituation; departmentsInAlert: number }) {
  const { t } = useTranslation()
  const color = classColor(data.situation_class)
  return (
    <div className="rounded-2xl border border-white/10 p-6 flex flex-wrap items-center gap-6"
         style={{ background: `linear-gradient(135deg, ${color}22, transparent)` }}>
      <div className="flex items-center gap-4">
        <span className="text-4xl" style={{ color }} aria-hidden>{trendGlyph(data.trend)}</span>
        <div>
          <div className="text-2xl font-bold" style={{ color }}>
            {data.insufficient || !data.situation_class
              ? t('meteo.insufficient')
              : t(`meteo.class.${data.situation_class}`)}
          </div>
          <div className="text-sm text-text-secondary">
            {t('meteo.trendGeneral')}: {data.trend ? t(`meteo.trend.${data.trend}`) : '—'}
          </div>
        </div>
      </div>
      {data.pct_below_normal != null && (
        <div className="text-text-primary text-lg font-mono">
          {t('meteo.belowNormal', { pct: data.pct_below_normal })}
        </div>
      )}
      <div className="text-text-secondary text-sm">
        {t('meteo.departmentsInAlert', { count: departmentsInAlert })}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd ~/time-serie-explo/frontend && npm test -- NationalBanner`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/components/meteo/NationalBanner.tsx frontend/src/components/meteo/NationalBanner.test.tsx
git commit -m "feat(meteo): NationalBanner verdict component"
```

---

## Task 11: TerritoryChoropleth component (MapLibre)

**Files:**
- Create: `frontend/src/components/meteo/TerritoryChoropleth.tsx`

This is a DOM/canvas component; vitest+jsdom cannot render MapLibre WebGL, so it has no unit test — it is validated in the browser at Task 14. Keep all data→color logic in the already-tested `situation-format.ts` so nothing untested lives here.

- [ ] **Step 1: Write the component**

```tsx
// frontend/src/components/meteo/TerritoryChoropleth.tsx
import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { TerritorySituation } from '@/lib/observatory-types'
import { classColor, INSUFFICIENT_COLOR } from '@/lib/situation-format'
import regionsGeo from '@/assets/geo/regions.geojson'
import departementsGeo from '@/assets/geo/departements.geojson'

const FRANCE_CENTER: [number, number] = [2.4, 46.6]
const FRANCE_ZOOM = 4.7

type Props = {
  level: 'region' | 'department'
  territories: TerritorySituation[]
  regionFilter?: string | null   // when level==='department', restrict to this region's depts
  onSelectRegion: (code: string) => void
  onSelectDepartment: (code: string) => void
}

export function TerritoryChoropleth({ level, territories, onSelectRegion, onSelectDepartment }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)

  // color lookup by territory code
  const colorByCode: Record<string, string> = {}
  territories.forEach(t => { colorByCode[t.code] = classColor(t.situation_class) })

  useEffect(() => {
    if (!mapRef.current || map.current) return
    map.current = new maplibregl.Map({
      container: mapRef.current,
      style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
      center: FRANCE_CENTER, zoom: FRANCE_ZOOM, attributionControl: true,
    })
    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    return () => { map.current?.remove(); map.current = null }
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m) return
    const geo = (level === 'region' ? regionsGeo : departementsGeo) as GeoJSON.FeatureCollection
    // paint colors onto features
    const colored = {
      ...geo,
      features: geo.features.map(f => ({
        ...f,
        properties: { ...f.properties, _fill: colorByCode[(f.properties as any).code] ?? INSUFFICIENT_COLOR },
      })),
    }
    const apply = () => {
      const srcId = 'territories'
      if (m.getSource(srcId)) {
        ;(m.getSource(srcId) as maplibregl.GeoJSONSource).setData(colored as any)
      } else {
        m.addSource(srcId, { type: 'geojson', data: colored as any })
        m.addLayer({ id: 'fill', type: 'fill', source: srcId,
          paint: { 'fill-color': ['get', '_fill'], 'fill-opacity': 0.7 } })
        m.addLayer({ id: 'line', type: 'line', source: srcId,
          paint: { 'line-color': '#ffffff', 'line-width': 0.5 } })
        m.on('click', 'fill', (e) => {
          const code = (e.features?.[0]?.properties as any)?.code
          if (!code) return
          if (level === 'region') onSelectRegion(code); else onSelectDepartment(code)
        })
        m.on('mouseenter', 'fill', () => { m.getCanvas().style.cursor = 'pointer' })
        m.on('mouseleave', 'fill', () => { m.getCanvas().style.cursor = '' })
      }
    }
    if (m.isStyleLoaded()) apply(); else m.once('load', apply)
  }, [level, territories])

  return <div ref={mapRef} className="w-full h-[480px] rounded-xl overflow-hidden" />
}
```

- [ ] **Step 2: Enable importing `.geojson` as JSON** — verify Vite resolves it. If `npx tsc --noEmit` or the build complains about the `.geojson` import, add a module declaration `frontend/src/geojson.d.ts`:

```ts
declare module '*.geojson' {
  const value: GeoJSON.FeatureCollection
  export default value
}
```

and ensure `vite.config.ts` has `json` handling (Vite imports `.json` natively; for `.geojson` add `assetsInclude: ['**/*.geojson']` and import via `?json` if needed, or rename usage to fetch from `/src/assets`). Simplest robust path: rename the two files to `regions.json` / `departements.json` and import as JSON. If you rename, update the imports in Step 1 and the Task 9 paths accordingly.

- [ ] **Step 3: Type-check**

Run: `cd ~/time-serie-explo/frontend && npx tsc --noEmit`
Expected: no errors (install `@types/geojson` if `GeoJSON` namespace is missing: `npm i -D @types/geojson`).

- [ ] **Step 4: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/components/meteo/TerritoryChoropleth.tsx frontend/src/geojson.d.ts 2>/dev/null; git add -A frontend/src/components/meteo frontend/src/assets/geo
git commit -m "feat(meteo): MapLibre territory choropleth with drill-down"
```

---

## Task 12: TerritoryRanking component

**Files:**
- Create: `frontend/src/components/meteo/TerritoryRanking.tsx`
- Test: `frontend/src/components/meteo/TerritoryRanking.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/meteo/TerritoryRanking.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { TerritoryRanking, sortByCriticality } from './TerritoryRanking'
import type { TerritorySituation } from '@/lib/observatory-types'

const mk = (code: string, cls: any, n = 10): TerritorySituation => ({
  level: 'region', code, name: `R${code}`, type: 'piezo', situation_class: cls,
  trend: 'baisse', pct_below_normal: 10, n_eligible: n, n_provisoire: 0,
  distribution: {}, insufficient: cls === null, outlook: null,
})

describe('TerritoryRanking', () => {
  it('sorts driest-first and pushes insufficient to the end', () => {
    const sorted = sortByCriticality([mk('a', 'NORMAL'), mk('b', 'EXTREMEMENT_BAS'), mk('c', null)])
    expect(sorted.map(t => t.code)).toEqual(['b', 'a', 'c'])
  })
  it('renders rows', () => {
    render(<I18nextProvider i18n={i18n}><TerritoryRanking territories={[mk('b', 'TRES_BAS')]} onSelect={vi.fn()} /></I18nextProvider>)
    expect(screen.getByText('Rb')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo/frontend && npm test -- TerritoryRanking`
Expected: FAIL — cannot resolve `./TerritoryRanking`.

- [ ] **Step 3: Write the component**

```tsx
// frontend/src/components/meteo/TerritoryRanking.tsx
import { useTranslation } from 'react-i18next'
import type { TerritorySituation } from '@/lib/observatory-types'
import { CLASS_ORDER_DRIEST_FIRST } from '@/lib/situation-format'
import { classColor, trendGlyph } from '@/lib/situation-format'

export function sortByCriticality(ts: TerritorySituation[]): TerritorySituation[] {
  const rank = (t: TerritorySituation) =>
    t.situation_class ? CLASS_ORDER_DRIEST_FIRST.indexOf(t.situation_class) : 999
  return [...ts].sort((a, b) => rank(a) - rank(b))
}

export function TerritoryRanking({ territories, onSelect }: { territories: TerritorySituation[]; onSelect: (code: string) => void }) {
  const { t } = useTranslation()
  const rows = sortByCriticality(territories)
  return (
    <div className="bg-bg-card border border-white/5 rounded-xl p-4">
      <h2 className="text-sm font-semibold text-text-primary mb-3">{t('meteo.rankingTitle')}</h2>
      <ul className="space-y-1">
        {rows.map(tr => (
          <li key={tr.code}>
            <button onClick={() => onSelect(tr.code)}
              className="w-full flex items-center gap-3 px-2 py-1.5 rounded hover:bg-bg-hover text-left">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: classColor(tr.situation_class) }} />
              <span className="flex-1 text-sm text-text-primary truncate">{tr.name}</span>
              <span className="text-xs text-text-secondary">{tr.insufficient ? t('meteo.insufficient') : `${tr.pct_below_normal ?? 0}%`}</span>
              <span aria-hidden style={{ color: classColor(tr.situation_class) }}>{trendGlyph(tr.trend)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 4: Add `CLASS_ORDER_DRIEST_FIRST`** to `frontend/src/lib/situation-format.ts`

```ts
export const CLASS_ORDER_DRIEST_FIRST: SituationClass[] = [
  'EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT',
]
```

- [ ] **Step 5: Run it to verify it passes**

Run: `cd ~/time-serie-explo/frontend && npm test -- TerritoryRanking`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/components/meteo/TerritoryRanking.tsx frontend/src/components/meteo/TerritoryRanking.test.tsx frontend/src/lib/situation-format.ts
git commit -m "feat(meteo): TerritoryRanking (driest-first) component"
```

---

## Task 13: OutlookPanel (dark AI layer) + StationDrillTable

**Files:**
- Create: `frontend/src/components/meteo/OutlookPanel.tsx`, `frontend/src/components/meteo/OutlookPanel.test.tsx`
- Create: `frontend/src/components/meteo/StationDrillTable.tsx`
- Delete: `frontend/src/pages/AlertsPage.tsx`

- [ ] **Step 1: Write the failing OutlookPanel test**

```tsx
// frontend/src/components/meteo/OutlookPanel.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { OutlookPanel } from './OutlookPanel'

describe('OutlookPanel', () => {
  it('renders the coming-soon state when outlook is null', () => {
    render(<I18nextProvider i18n={i18n}><OutlookPanel outlook={null} /></I18nextProvider>)
    expect(screen.getByText(/Bientôt/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/time-serie-explo/frontend && npm test -- OutlookPanel`
Expected: FAIL — cannot resolve `./OutlookPanel`.

- [ ] **Step 3: Write OutlookPanel**

```tsx
// frontend/src/components/meteo/OutlookPanel.tsx
import { useTranslation } from 'react-i18next'
import { Sparkles } from 'lucide-react'
import type { Outlook } from '@/lib/observatory-types'

export function OutlookPanel({ outlook }: { outlook: Outlook | null }) {
  const { t } = useTranslation()
  return (
    <div className="bg-bg-card border border-dashed border-accent-indigo/40 rounded-xl p-4 flex items-center gap-3 opacity-80">
      <Sparkles className="w-4 h-4 text-accent-indigo" />
      <div>
        <div className="text-sm font-medium text-text-primary">{t('meteo.outlookTitle')}</div>
        <div className="text-xs text-text-secondary">
          {outlook ? `${outlook.horizon_months} mois` : t('meteo.outlookSoon')}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Write StationDrillTable** — port the table from `AlertsPage.tsx` into a component that takes a department code and renders the per-station alert list (reuse `observatoryApi.common.alerts({ code_departement })` + the existing table markup). Keep the existing i18n keys it used (`cleanup.alerts.*` still exist in the locale files).

```tsx
// frontend/src/components/meteo/StationDrillTable.tsx
import { Link } from 'react-router-dom'
import { MapPin } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { observatoryApi } from '@/lib/observatory-api'
import { formatDate } from '@/lib/observatory-utils'
import type { Alert } from '@/lib/observatory-types'

export function StationDrillTable({ codeDepartement, type }: { codeDepartement: string; type: 'piezo' | 'hydro' }) {
  const { t } = useTranslation()
  const { data, isLoading } = useQuery({
    queryKey: ['drill-alerts', codeDepartement, type],
    queryFn: () => observatoryApi.common.alerts({
      code_departement: codeDepartement, type,
      severity: ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'],
    }),
  })
  if (isLoading) return <div className="text-sm text-text-secondary p-4">…</div>
  const rows = (data ?? []) as Alert[]
  if (!rows.length) return <div className="text-sm text-text-secondary p-4">{t('cleanup.alerts.noStationsInAlert', { label: '' })}</div>
  return (
    <div className="overflow-x-auto bg-bg-card border border-white/5 rounded-xl">
      <table className="w-full text-sm min-w-[600px]">
        <thead><tr className="border-b border-white/5">
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colCode')}</th>
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colStation')}</th>
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colLastMeasurement')}</th>
          <th className="w-10" />
        </tr></thead>
        <tbody>{rows.map(s => (
          <tr key={`${s.type}-${s.code}`} className="border-b border-white/5 hover:bg-bg-hover">
            <td className="px-4 py-2"><Link to={`/station/${s.type}/${s.code}`} className="text-accent-cyan hover:underline font-mono text-xs">{s.code}</Link></td>
            <td className="px-4 py-2 text-text-primary">{s.commune || s.code}</td>
            <td className="px-4 py-2 text-text-secondary text-xs">{formatDate(s.derniere_mesure)}</td>
            <td className="px-4 py-2">{s.latitude != null && s.longitude != null && (
              <Link to={`/?lat=${s.latitude}&lon=${s.longitude}&zoom=12`} className="inline-flex"><MapPin className="w-3.5 h-3.5 text-accent-cyan" /></Link>)}</td>
          </tr>))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 5: Delete the absorbed page**

```bash
cd ~/time-serie-explo && git rm frontend/src/pages/AlertsPage.tsx
```

(Confirm nothing imports it: `grep -rn "AlertsPage" frontend/src` returns nothing. It had no route, so there is nothing else to remove.)

- [ ] **Step 6: Run tests + typecheck**

Run: `cd ~/time-serie-explo/frontend && npm test -- OutlookPanel && npx tsc --noEmit`
Expected: OutlookPanel PASS; no type errors.

- [ ] **Step 7: Commit**

```bash
cd ~/time-serie-explo
git add -A frontend/src/components/meteo frontend/src/pages/AlertsPage.tsx
git commit -m "feat(meteo): OutlookPanel (dark) + StationDrillTable; absorb AlertsPage"
```

---

## Task 14: MeteoNappesPage + route + nav

**Files:**
- Create: `frontend/src/pages/MeteoNappesPage.tsx`
- Modify: `frontend/src/routes.tsx`, `frontend/src/components/layout/TopNav.tsx`

- [ ] **Step 1: Write the page** (orchestrates banner + map + ranking + drill + toggle)

```tsx
// frontend/src/pages/MeteoNappesPage.tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { situationApi } from '@/lib/situation-api'
import { NationalBanner } from '@/components/meteo/NationalBanner'
import { TerritoryChoropleth } from '@/components/meteo/TerritoryChoropleth'
import { TerritoryRanking } from '@/components/meteo/TerritoryRanking'
import { OutlookPanel } from '@/components/meteo/OutlookPanel'
import { StationDrillTable } from '@/components/meteo/StationDrillTable'
import { region_of_dept_unused } from '@/lib/situation-format' // placeholder import removed below

export default function MeteoNappesPage() {
  const { t } = useTranslation()
  const [type, setType] = useState<'piezo' | 'hydro'>('piezo')
  const [region, setRegion] = useState<string | null>(null)
  const [dept, setDept] = useState<string | null>(null)

  const national = useQuery({ queryKey: ['sit-national', type], queryFn: () => situationApi.national(type) })
  const regions = useQuery({ queryKey: ['sit-territories', 'region', type], queryFn: () => situationApi.territories('region', type) })
  const departments = useQuery({
    queryKey: ['sit-territories', 'department', type], enabled: region != null,
    queryFn: () => situationApi.territories('department', type),
  })

  const level: 'region' | 'department' = region ? 'department' : 'region'
  const shownTerritories = region
    ? (departments.data ?? [])   // the choropleth shows all depts; click handles selection
    : (regions.data ?? [])
  const deptsInAlert = (departments.data ?? regions.data ?? []).filter(d => d.situation_class && ['EXTREMEMENT_BAS', 'TRES_BAS'].includes(d.situation_class)).length

  if (national.isError || regions.isError) {
    return <div className="h-full flex items-center justify-center text-red-400 text-sm">{t('meteo.loadError')}</div>
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-xl font-bold text-text-primary">{t('meteo.title')}</h1>
            <p className="text-xs text-text-secondary">{t('meteo.subtitle')}</p>
          </div>
          <div className="flex gap-1 bg-bg-card border border-white/10 rounded-lg p-0.5">
            {(['piezo', 'hydro'] as const).map(ty => (
              <button key={ty} onClick={() => { setType(ty); setRegion(null); setDept(null) }}
                className={`px-3 py-1.5 rounded-md text-xs ${type === ty ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary'}`}>
                {t(ty === 'piezo' ? 'meteo.tabPiezo' : 'meteo.tabHydro')}
              </button>
            ))}
          </div>
        </div>

        {national.data && <NationalBanner data={national.data} departmentsInAlert={deptsInAlert} />}

        {region && (
          <button onClick={() => { setRegion(null); setDept(null) }} className="text-xs text-accent-cyan">{t('meteo.backToRegions')}</button>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2">
            <TerritoryChoropleth
              level={level}
              territories={shownTerritories}
              regionFilter={region}
              onSelectRegion={(code) => setRegion(code)}
              onSelectDepartment={(code) => setDept(code)}
            />
          </div>
          <div className="space-y-4">
            <TerritoryRanking
              territories={shownTerritories}
              onSelect={(code) => (region ? setDept(code) : setRegion(code))}
            />
            <OutlookPanel outlook={national.data?.outlook ?? null} />
          </div>
        </div>

        {dept && (
          <div className="space-y-2">
            <h2 className="text-sm font-semibold text-text-primary">{t('meteo.stationsIn', { name: dept })}</h2>
            <StationDrillTable codeDepartement={dept} type={type} />
          </div>
        )}
      </div>
    </div>
  )
}
```

Remove the placeholder import line `import { region_of_dept_unused } ...` (it is not a real export) — it is shown only to flag that the page needs no extra helper; delete that line before saving.

- [ ] **Step 2: Add the lazy import + public route** in `frontend/src/routes.tsx`

Near the other `const X = lazy(() => import('./pages/...'))` declarations add:
```tsx
const MeteoNappesPage = lazy(() => import('./pages/MeteoNappesPage'))
```
In the public routes array (same level as the `/` Observatory route, **no** `RequireAuth`), add:
```tsx
{ path: '/meteo', element: <SW><MeteoNappesPage /></SW> },
```
(Use the same Suspense wrapper component the other routes use — match the existing `<SW>`/`<Suspense>` pattern in the file.)

- [ ] **Step 3: Add the nav entry** in `frontend/src/components/layout/TopNav.tsx`

Add an icon import (e.g. `CloudRain`) to the existing `lucide-react` import, then add to the `navItems` array, right after the Observatory entry:
```tsx
  { to: '/meteo', icon: CloudRain, label: t('nav.meteo'), end: false },
```

- [ ] **Step 4: Type-check + build**

Run: `cd ~/time-serie-explo/frontend && npx tsc --noEmit && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Run the full frontend test suite**

Run: `cd ~/time-serie-explo/frontend && npm test`
Expected: all suites pass.

- [ ] **Step 6: Commit**

```bash
cd ~/time-serie-explo
git add frontend/src/pages/MeteoNappesPage.tsx frontend/src/routes.tsx frontend/src/components/layout/TopNav.tsx
git commit -m "feat(meteo): MeteoNappesPage + public route + nav entry"
```

---

## Task 15: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Rebuild + run the stack** (per project memory, never pass `-f` flags; the split backend lives in `deploy/dib-backend`)

```bash
cd ~/time-serie-explo/deploy/dib-backend && docker compose -p time-serie-explo -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

- [ ] **Step 2: Verify the page in a browser** at the frontend URL (dib `:49513`):
  - `/meteo` loads without login (public).
  - National banner shows a class + trend arrow + "% sous la normale".
  - Region choropleth is colored; clicking a region drills to its departments; clicking a department lists its stations; "← Toutes les régions" returns.
  - Piézo/Cours d'eau toggle re-colors the map.
  - At least one region with sparse data renders greyed "Données insuffisantes".
  - The "Anticipation IA — Bientôt" panel renders.
  - Switch language to EN; all `meteo.*` strings translate.

- [ ] **Step 3: Backend full suite**

Run: `cd ~/time-serie-explo && python -m pytest tests/test_territory_situation.py tests/test_territories_fr.py tests/test_schemas.py tests/test_situation_router.py -v`
Expected: all pass.

- [ ] **Step 4: Final commit (if any verification fixes were needed)**

```bash
cd ~/time-serie-explo && git add -A && git commit -m "fix(meteo): verification adjustments" || echo "nothing to fix"
```

---

## Self-review notes (addressed)

- **Spec coverage:** national banner (Task 10), region/dept choropleth with trend (Task 11), ranking (Task 12), station drill via reused alerts table (Task 13), piezo/hydro toggle + page (Task 14), AI outlook contract + dark UI (Tasks 3, 13), insufficient-data handling (Tasks 1, 10–12), public access + nav (Task 14), defensible aggregation (Task 1), dept→region (Task 2), caching + warm-up (Task 4). CSV export from the old AlertsPage is dropped in v1 (was station-level; re-add later if requested) — noted as a conscious scope trim.
- **Type consistency:** `situation_class`, `trend`, `pct_below_normal`, `n_eligible`, `n_provisoire`, `distribution`, `insufficient`, `outlook` are identical across the Pydantic schemas (Task 3), the TS types (Task 6), and component props (Tasks 10–14). `classColor`/`trendGlyph`/`CLASS_ORDER_DRIEST_FIRST` are defined in `situation-format.ts` and used consistently.
- **Placeholders:** the one intentional placeholder import in Task 14 Step 1 is explicitly flagged for deletion in the same step.
- **Known follow-ups (out of v1 scope):** real AI forecast + nightly inference asset; precompute the trend if the live query is slow; re-add CSV/PDF export; animated timeline.
