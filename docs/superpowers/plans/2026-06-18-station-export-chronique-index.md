# Station export (chronique + index) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-station CSV export combining station metadata + daily chronique + monthly standardized index (IPS for piezo, SSFI for hydro), downloadable from the station detail page.

**Architecture:** A pure, DB-free CSV builder (`dashboard/utils/station_export.py`) does the month→day carry-forward join and formatting; it is unit-tested in isolation (matching this repo's testing pattern where the BRGM warehouse is not available in CI). Two thin FastAPI endpoints (one per domain) fetch rows from `gold.*` and call the builder, returning a `text/csv` `Response` with `Content-Disposition`. The frontend adds a plain download link (cookie auth + `Content-Disposition`), mirroring the existing pastas export.

**Tech Stack:** FastAPI + SQLAlchemy core (sync BRGM engine), Python `csv`/`io`, pytest; React/TS frontend with vitest.

## Global Constraints

- `dashboard/utils/` is pure Python — NO Streamlit, NO FastAPI, NO DB imports there (project convention).
- SQL: parameterized queries only (`text(...)` with bind params), never string-interpolate identifiers/values.
- Index source of truth: `gold.fct_monthly_index` (`type` ∈ {`piezo`,`hydro`}, `code`, `month` date, `z`, `index_class`, `flag`). Never recompute the index here.
- Hydro flow values are stored in L/s and must be converted to m³/s for display, EXCEPT rows where `grandeur_hydro_elab = 'H'` (water height) — reuse the existing `_convert_qmnj_row(row, _FLOW_COLS_DAILY)` from `api/routers/observatory_hydro.py`. The index `z` is unitless and is exported as stored (no conversion).
- Missing `gold.fct_monthly_index` (pre-materialization) must NOT break the export: catch `sqlalchemy.exc.ProgrammingError` and treat the index as empty.
- Frontend UI text uses `t(...)` i18n keys (French BRGM audience); add new keys to both locale files.
- `API_BASE = '/api/v1'` (`frontend/src/lib/constants.ts`).

---

### Task 1: Pure CSV builder + unit tests

**Files:**
- Create: `dashboard/utils/station_export.py`
- Test: `tests/test_station_export.py`

**Interfaces:**
- Produces:
  - `index_by_month(index_rows: list[dict]) -> dict[tuple[int,int], dict]` — maps `(year, month)` → `{"z", "index_class", "flag"}`.
  - `build_station_csv(domain: str, meta: dict, daily_rows: list[dict], index_rows: list[dict]) -> str` — returns the full CSV text (commented header block + data table). `domain` ∈ {`"piezo"`,`"hydro"`}. Raises `ValueError` on unknown domain.
  - `meta` keys used: `code`, `nom_commune`, `code_departement`, `nom_departement`, `latitude`, `longitude`.
  - `daily_rows` keys: `date` plus value/meteo columns; piezo → `niveau_nappe_eau`, `profondeur_nappe`; hydro → `resultat_obs_elab`, `grandeur_hydro_elab`; both → `temperature_2m`, `total_precipitation`, `potential_evaporation`.
  - `index_rows` keys: `month` (date), `z`, `index_class`, `flag`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_station_export.py`:

```python
from datetime import date

from dashboard.utils.station_export import build_station_csv, index_by_month


def _piezo_daily():
    # two days in Jan (has index), one day in Feb (no index)
    return [
        {"date": date(2020, 1, 10), "niveau_nappe_eau": 12.5, "profondeur_nappe": 3.1,
         "temperature_2m": 5.0, "total_precipitation": 2.0, "potential_evaporation": 0.5},
        {"date": date(2020, 1, 20), "niveau_nappe_eau": 12.7, "profondeur_nappe": 3.0,
         "temperature_2m": 6.0, "total_precipitation": 1.0, "potential_evaporation": 0.6},
        {"date": date(2020, 2, 5), "niveau_nappe_eau": 12.9, "profondeur_nappe": 2.9,
         "temperature_2m": 7.0, "total_precipitation": 0.0, "potential_evaporation": 0.7},
    ]


def _piezo_index():
    return [{"month": date(2020, 1, 1), "z": -0.95, "index_class": "BAS", "flag": "normale"}]


def _meta():
    return {"code": "BSS000/X", "nom_commune": "Tours", "code_departement": "37",
            "nom_departement": "Indre-et-Loire", "latitude": 47.39, "longitude": 0.69}


def test_index_by_month_keys_on_year_month():
    idx = index_by_month(_piezo_index())
    assert idx[(2020, 1)] == {"z": -0.95, "index_class": "BAS", "flag": "normale"}


def test_csv_carries_index_forward_onto_each_day_of_month():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    lines = [l for l in csv_text.splitlines() if not l.startswith("#")]
    header = lines[0].split(",")
    assert header == ["date", "niveau_nappe_eau", "profondeur_nappe",
                      "temperature_2m", "total_precipitation", "potential_evaporation",
                      "mois_ref", "ips_z", "ips_classe", "ips_flag"]
    # both January days carry the same index value
    jan10 = lines[1].split(",")
    jan20 = lines[2].split(",")
    assert jan10[6:] == ["2020-01", "-0.95", "BAS", "normale"]
    assert jan20[6:] == ["2020-01", "-0.95", "BAS", "normale"]


def test_csv_leaves_index_cells_empty_for_month_without_index():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    data = [l for l in csv_text.splitlines() if not l.startswith("#")][1:]
    feb = data[2].split(",")  # 2020-02-05
    assert feb[0] == "2020-02-05"
    assert feb[6:] == ["", "", "", ""]


def test_header_block_contains_station_metadata():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    head = "\n".join(l for l in csv_text.splitlines() if l.startswith("#"))
    assert "BSS000/X" in head
    assert "Tours" in head
    assert "IPS" in head
    assert "1991-2020" in head
    assert "flag=normale" in head


def test_hydro_columns_and_unknown_domain():
    daily = [{"date": date(2020, 1, 10), "resultat_obs_elab": 1.7, "grandeur_hydro_elab": "Q",
              "temperature_2m": 5.0, "total_precipitation": 2.0, "potential_evaporation": 0.5}]
    index = [{"month": date(2020, 1, 1), "z": 2.3, "index_class": "EXTREMEMENT_HAUT", "flag": "adaptee"}]
    csv_text = build_station_csv("hydro", _meta(), daily, index)
    header = [l for l in csv_text.splitlines() if not l.startswith("#")][0].split(",")
    assert header == ["date", "resultat_obs_elab", "grandeur_hydro_elab",
                      "temperature_2m", "total_precipitation", "potential_evaporation",
                      "mois_ref", "ssfi_z", "ssfi_classe", "ssfi_flag"]
    import pytest
    with pytest.raises(ValueError):
        build_station_csv("nope", _meta(), [], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_station_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.utils.station_export'`

- [ ] **Step 3: Write the implementation**

Create `dashboard/utils/station_export.py`:

```python
"""Pure CSV builder for single-station export (chronique + monthly index).

No FastAPI / no DB here — routers fetch the rows and call build_station_csv.
One CSV row per day; the monthly standardized index (IPS for piezo, SSFI for
hydro) of a row's month is carried forward onto every day of that month.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime

# Per-domain value columns: (csv_header, daily_row_key)
_VALUE_COLS = {
    "piezo": [("niveau_nappe_eau", "niveau_nappe_eau"),
              ("profondeur_nappe", "profondeur_nappe")],
    "hydro": [("resultat_obs_elab", "resultat_obs_elab"),
              ("grandeur_hydro_elab", "grandeur_hydro_elab")],
}
_METEO_COLS = [("temperature_2m", "temperature_2m"),
               ("total_precipitation", "total_precipitation"),
               ("potential_evaporation", "potential_evaporation")]
_INDEX_PREFIX = {"piezo": "ips", "hydro": "ssfi"}
_INDEX_LABEL = {"piezo": "IPS", "hydro": "SSFI"}
_UNIT = {"piezo": "niveau en m NGF", "hydro": "débit en m³/s"}


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def _month_key(d) -> tuple[int, int]:
    dd = _as_date(d)
    return (dd.year, dd.month)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (date, datetime)):
        return _as_date(v).isoformat()
    return str(v)


def index_by_month(index_rows) -> dict[tuple[int, int], dict]:
    """Map (year, month) -> {'z','index_class','flag'} from fct_monthly_index rows."""
    out: dict[tuple[int, int], dict] = {}
    for r in index_rows:
        out[_month_key(r["month"])] = {
            "z": r.get("z"),
            "index_class": r.get("index_class"),
            "flag": r.get("flag"),
        }
    return out


def _header_lines(domain: str, meta: dict, daily_rows, idx) -> list[str]:
    flag = ""
    if idx:
        flag = idx[max(idx.keys())].get("flag") or ""
    dmin = _fmt(daily_rows[0]["date"]) if daily_rows else ""
    dmax = _fmt(daily_rows[-1]["date"]) if daily_rows else ""
    return [
        f"Station: {meta.get('nom_commune') or ''} ({meta.get('code') or ''})",
        f"Département: {meta.get('nom_departement') or ''} ({meta.get('code_departement') or ''})",
        f"Coordonnées: {_fmt(meta.get('latitude'))}, {_fmt(meta.get('longitude'))}",
        f"Période exportée: {dmin} → {dmax}",
        f"Index: {_INDEX_LABEL[domain]} (réf. fixe 1991-2020, flag={flag})",
        f"Unités: {_UNIT[domain]} ; z-score sans unité",
        "Source: Junon / Hub'Eau + BRGM",
    ]


def build_station_csv(domain: str, meta: dict, daily_rows, index_rows) -> str:
    if domain not in _INDEX_PREFIX:
        raise ValueError(f"unknown domain {domain!r}")
    idx = index_by_month(index_rows)
    prefix = _INDEX_PREFIX[domain]
    value_cols = _VALUE_COLS[domain]

    out = io.StringIO()
    for line in _header_lines(domain, meta, daily_rows, idx):
        out.write(f"# {line}\n")

    writer = csv.writer(out)
    writer.writerow(
        ["date"]
        + [h for h, _ in value_cols]
        + [h for h, _ in _METEO_COLS]
        + ["mois_ref", f"{prefix}_z", f"{prefix}_classe", f"{prefix}_flag"]
    )

    for row in daily_rows:
        mk = _month_key(row["date"])
        ix = idx.get(mk)
        rec = [_fmt(row.get("date"))]
        rec += [_fmt(row.get(k)) for _, k in value_cols]
        rec += [_fmt(row.get(k)) for _, k in _METEO_COLS]
        if ix:
            rec += [f"{mk[0]:04d}-{mk[1]:02d}", _fmt(ix["z"]),
                    _fmt(ix["index_class"]), _fmt(ix["flag"])]
        else:
            rec += ["", "", "", ""]
        writer.writerow(rec)

    return out.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_station_export.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/station_export.py tests/test_station_export.py
git commit -m "feat(export): pure CSV builder for station chronique + monthly index"
```

---

### Task 2: Piezo export endpoint

**Files:**
- Modify: `api/routers/observatory_piezo.py` (add endpoint + imports)
- Test: `tests/test_station_export_endpoint.py` (source-level checks, no DB)

**Interfaces:**
- Consumes: `build_station_csv` from Task 1.
- Produces: `GET /api/v1/observatory/piezo/stations/{code_bss}/export.csv` → `text/csv` attachment.

- [ ] **Step 1: Write the failing test**

Create `tests/test_station_export_endpoint.py`:

```python
import inspect

from api.routers import observatory_piezo


def test_piezo_export_endpoint_exists_and_is_resilient():
    src = inspect.getsource(observatory_piezo.export_csv)
    # joins the index from the materialized table
    assert "gold.fct_monthly_index" in src
    # 404 on unknown station, reusing the dim table existence check
    assert "introuvable" in src
    assert "gold.dim_piezo_stations" in src
    # tolerates a missing index table (pre-materialization)
    assert "ProgrammingError" in src
    # delegates formatting to the pure builder
    assert "build_station_csv" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_station_export_endpoint.py::test_piezo_export_endpoint_exists_and_is_resilient -v`
Expected: FAIL with `AttributeError: module 'api.routers.observatory_piezo' has no attribute 'export_csv'`

- [ ] **Step 3: Add imports**

In `api/routers/observatory_piezo.py`, near the other imports, add:

```python
from datetime import date as _date
from fastapi import Response
from dashboard.utils.station_export import build_station_csv
```

(`text`, `get_brgm_sync_engine`, `HTTPException`, `ProgrammingError` are already imported at the top of the file. The file already imports `date` from `datetime`; the `as _date` alias avoids any clash with existing usage.)

- [ ] **Step 4: Add the endpoint**

Append to `api/routers/observatory_piezo.py` (after the `/spi` endpoint):

```python
@router.get("/stations/{code_bss:path}/export.csv")
def export_csv(code_bss: str):
    """Export station metadata + daily chronique + monthly IPS as a CSV file."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        meta = conn.execute(
            text(
                "SELECT code_bss AS code, nom_commune, code_departement,"
                " nom_departement, latitude, longitude"
                " FROM gold.dim_piezo_stations WHERE code_bss = :code"
            ),
            {"code": code_bss},
        ).mappings().first()
        if meta is None:
            raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        daily = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT date, niveau_nappe_eau, profondeur_nappe, temperature_2m,"
                    " total_precipitation, potential_evaporation"
                    " FROM gold.hubeau_daily_chroniques WHERE code_bss = :code ORDER BY date"
                ),
                {"code": code_bss},
            ).mappings()
        ]

    index_rows: list[dict] = []
    engine2 = get_brgm_sync_engine()
    try:
        with engine2.connect() as conn2:
            index_rows = [
                dict(r) for r in conn2.execute(
                    text(
                        "SELECT month, z, index_class, flag FROM gold.fct_monthly_index"
                        " WHERE type = 'piezo' AND code = :code ORDER BY month"
                    ),
                    {"code": code_bss},
                ).mappings()
            ]
    except ProgrammingError:
        index_rows = []  # table not yet materialized

    body = build_station_csv("piezo", dict(meta), daily, index_rows)
    fname = f"{code_bss.replace('/', '_')}_{_date.today().isoformat()}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_station_export_endpoint.py::test_piezo_export_endpoint_exists_and_is_resilient -v`
Expected: PASS

- [ ] **Step 6: Verify the app imports cleanly**

Run: `python -c "import api.main"`
Expected: no error (exit 0).

- [ ] **Step 7: Commit**

```bash
git add api/routers/observatory_piezo.py tests/test_station_export_endpoint.py
git commit -m "feat(export): piezo /export.csv endpoint (chronique + IPS)"
```

---

### Task 3: Hydro export endpoint

**Files:**
- Modify: `api/routers/observatory_hydro.py` (add endpoint + imports)
- Test: `tests/test_station_export_endpoint.py` (add hydro case)

**Interfaces:**
- Consumes: `build_station_csv` from Task 1; `_convert_qmnj_row`, `_FLOW_COLS_DAILY` already defined in `api/routers/observatory_hydro.py`.
- Produces: `GET /api/v1/observatory/hydro/stations/{code_station}/export.csv` → `text/csv` attachment.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_station_export_endpoint.py`:

```python
from api.routers import observatory_hydro


def test_hydro_export_endpoint_converts_flow_and_is_resilient():
    src = inspect.getsource(observatory_hydro.export_csv)
    assert "gold.fct_monthly_index" in src
    assert "gold.dim_hydro_stations" in src
    assert "ProgrammingError" in src
    assert "build_station_csv" in src
    # L/s -> m³/s conversion reused for non-height rows
    assert "_convert_qmnj_row" in src
    assert "_FLOW_COLS_DAILY" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_station_export_endpoint.py::test_hydro_export_endpoint_converts_flow_and_is_resilient -v`
Expected: FAIL with `AttributeError: ... has no attribute 'export_csv'`

- [ ] **Step 3: Add imports**

In `api/routers/observatory_hydro.py`, add near the other imports (check which already exist — `text`, `get_brgm_sync_engine`, `HTTPException` are present; add only what's missing):

```python
from datetime import date as _date
from fastapi import Response
from sqlalchemy.exc import ProgrammingError
from dashboard.utils.station_export import build_station_csv
```

- [ ] **Step 4: Add the endpoint**

Append to `api/routers/observatory_hydro.py` (after the `/spi` endpoint). Note `_convert_qmnj_row` mutates the row in place for non-`'H'` rows, exactly like the existing `/daily` endpoint:

```python
@router.get("/stations/{code_station}/export.csv")
def export_csv(code_station: str):
    """Export station metadata + daily chronique + monthly SSFI as a CSV file."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        meta = conn.execute(
            text(
                "SELECT code_station AS code, nom_commune, code_departement,"
                " nom_departement, latitude_station AS latitude,"
                " longitude_station AS longitude"
                " FROM gold.dim_hydro_stations WHERE code_station = :code"
            ),
            {"code": code_station},
        ).mappings().first()
        if meta is None:
            raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
        daily = [
            dict(r) for r in conn.execute(
                text(
                    "SELECT date, resultat_obs_elab, grandeur_hydro_elab, temperature_2m,"
                    " total_precipitation, potential_evaporation"
                    " FROM gold.hydro_daily_chroniques WHERE code_station = :code ORDER BY date"
                ),
                {"code": code_station},
            ).mappings()
        ]
    for r in daily:
        if r.get("grandeur_hydro_elab") != "H":
            _convert_qmnj_row(r, _FLOW_COLS_DAILY)

    index_rows: list[dict] = []
    engine2 = get_brgm_sync_engine()
    try:
        with engine2.connect() as conn2:
            index_rows = [
                dict(r) for r in conn2.execute(
                    text(
                        "SELECT month, z, index_class, flag FROM gold.fct_monthly_index"
                        " WHERE type = 'hydro' AND code = :code ORDER BY month"
                    ),
                    {"code": code_station},
                ).mappings()
            ]
    except ProgrammingError:
        index_rows = []

    body = build_station_csv("hydro", dict(meta), daily, index_rows)
    fname = f"{code_station.replace('/', '_')}_{_date.today().isoformat()}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
```

> NOTE for the implementer: verify the dim column names with
> `docker exec brgm-postgres psql -U postgres -d postgres -c "\d gold.dim_hydro_stations"`.
> The piezo dim exposes `latitude`/`longitude`; the hydro dim uses
> `latitude_station`/`longitude_station` (aliased to `latitude`/`longitude` above to
> match the builder's `meta` contract). If the actual names differ, fix the aliases here
> only — the builder is unaffected.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_station_export_endpoint.py -v`
Expected: PASS (both piezo and hydro cases)

- [ ] **Step 6: Verify the app imports cleanly**

Run: `python -c "import api.main"`
Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add api/routers/observatory_hydro.py tests/test_station_export_endpoint.py
git commit -m "feat(export): hydro /export.csv endpoint (chronique m³/s + SSFI)"
```

---

### Task 4: Frontend download button

**Files:**
- Modify: `frontend/src/lib/observatory-api.ts` (add `exportUrl` builders)
- Create: `frontend/src/lib/observatory-api.test.ts`
- Modify: `frontend/src/pages/StationPage.tsx` (add the button)
- Modify: the two i18n locale files (add `mainPages.station.exportCsv` key)

**Interfaces:**
- Consumes: `API_BASE` from `frontend/src/lib/constants.ts`.
- Produces: `observatoryApi.piezo.exportUrl(code)` and `observatoryApi.hydro.exportUrl(code)` returning the absolute API path string.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/observatory-api.test.ts`:

```typescript
import { describe, expect, it } from 'vitest'
import { observatoryApi } from './observatory-api'

describe('observatoryApi exportUrl', () => {
  it('builds the piezo export url with encoded code', () => {
    expect(observatoryApi.piezo.exportUrl('BSS000/X')).toBe(
      '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv',
    )
  })
  it('builds the hydro export url with encoded code', () => {
    expect(observatoryApi.hydro.exportUrl('K001 0010')).toBe(
      '/api/v1/observatory/hydro/stations/K001%200010/export.csv',
    )
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/observatory-api.test.ts`
Expected: FAIL — `exportUrl is not a function`.

- [ ] **Step 3: Add the URL builders**

In `frontend/src/lib/observatory-api.ts`, inside `observatoryApi.piezo` add:

```typescript
    exportUrl: (code: string) =>
      `${API_BASE}/observatory/piezo/stations/${encodeURIComponent(code)}/export.csv`,
```

and inside `observatoryApi.hydro` add:

```typescript
    exportUrl: (code: string) =>
      `${API_BASE}/observatory/hydro/stations/${encodeURIComponent(code)}/export.csv`,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/observatory-api.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the i18n key**

Find the locale files and the `mainPages.station` namespace:

Run: `grep -rl "analyzeInPastas" frontend/src`

In each matched locale file, add a sibling key to `analyzeInPastas` inside `mainPages.station`:

```json
"exportCsv": "Exporter (CSV)"
```

(Use the same French string in every locale file — the app's audience is French; no English variant is required for this label.)

- [ ] **Step 6: Add the button to StationPage**

In `frontend/src/pages/StationPage.tsx`:

1. Ensure `observatoryApi` is imported (add `import { observatoryApi } from '../lib/observatory-api'` if not already present), and add `Download` to the existing `lucide-react` import.
2. In the action-buttons `<div className="flex items-center gap-2">` (the one containing `<AddToCompareButton ... />`), add as the first child:

```tsx
<a
  href={isPiezo ? observatoryApi.piezo.exportUrl(code) : observatoryApi.hydro.exportUrl(code)}
  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 text-text-secondary hover:text-text-primary hover:bg-white/10 transition-colors"
>
  <Download className="w-3.5 h-3.5" />{t('mainPages.station.exportCsv')}
</a>
```

(`isPiezo`, `code`, and `t` are already in scope in this component.)

- [ ] **Step 7: Verify build + full frontend tests**

Run: `cd frontend && npx vitest run src/lib/observatory-api.test.ts && npx tsc --noEmit`
Expected: tests PASS and no TypeScript errors. (If `tsc` is not the project's check, run `npm run build` instead.)

- [ ] **Step 8: Commit**

```bash
git add frontend/src/lib/observatory-api.ts frontend/src/lib/observatory-api.test.ts \
        frontend/src/pages/StationPage.tsx frontend/src/locales
git commit -m "feat(export): station page download button for chronique + index CSV"
```

---

## Manual verification (after all tasks)

1. Rebuild backend + frontend: `docker compose up -d --build junon-backend junon-frontend` (from repo root, no `-f` flags — see project memory).
2. Open a piezo station page → click **Exporter (CSV)** → a `BSSxxx_YYYY-MM-DD.csv` downloads.
3. Open the file: `#` header block with station metadata; one row per day; `ips_z`/`ips_classe`/`ips_flag` repeated within each month; empty index cells for months before the reference exists.
4. Repeat on a hydro station → flow values in m³/s, `ssfi_*` columns populated.
5. Spot-check a station whose months predate its reference window → index columns blank there, chronique still present.
```
