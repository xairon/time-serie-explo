# Station CSV Export — Column Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users pick which column groups the station CSV export contains, via group-level toggles on the existing station page download.

**Architecture:** Group definitions live in the pure builder `dashboard/utils/station_export.py` (single source of truth). `build_station_csv` gains an optional `groups` arg; the two FastAPI `export_csv` endpoints parse a `groups` query param and pass it through. The frontend adds a `groups` arg to `exportUrl` and a small `ColumnPicker` popover wired into `StationPage`.

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy (backend), pytest; React + TypeScript + react-i18next / Vitest (frontend).

## Global Constraints

- `date` is always emitted, independent of `groups`. It is never part of a group.
- `groups` absent → all columns (backward-compatible; bare URL unchanged).
- The five group keys, canonical order: `identity, values, meteo, index, provenance`.
- Column order within the CSV is fixed and never reordered by selection: identity → date → values → meteo → index → provenance.
- Unknown group keys are ignored (intersect with the known set), never an error.
- Backend tests run with: `DEBUG=true DB_PASSWORD=test uv run pytest <files> -q` (the repo `.env` trips Settings otherwise).
- Frontend tests run with: `cd frontend && npm test` (Vitest).
- UI copy is French (BRGM audience); every new visible string gets fr + en i18n keys.

---

## File Structure

- `dashboard/utils/station_export.py` — add `GROUP_KEYS` + `groups` param to `build_station_csv`; column assembly filters by group.
- `api/routers/observatory_piezo.py` — add `groups` query param, parse + pass through.
- `api/routers/observatory_hydro.py` — same.
- `tests/test_station_export.py` — builder group-filter tests.
- `tests/test_station_export_endpoint.py` — endpoint wiring assertions.
- `frontend/src/lib/observatory-api.ts` — `exportUrl(code, range, groups?)`.
- `frontend/src/lib/observatory-api.test.ts` — url-building tests.
- `frontend/src/components/observatory/ColumnPicker.tsx` — new popover (create).
- `frontend/src/pages/StationPage.tsx` — selected-groups state + wiring.
- `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json` — labels.

---

## Task 1: Builder — `groups` filter + `GROUP_KEYS`

**Files:**
- Modify: `dashboard/utils/station_export.py`
- Test: `tests/test_station_export.py`

**Interfaces:**
- Consumes: existing `build_station_csv(domain, meta, daily_rows, index_rows)` and module constants `_IDENTITY_COLS`, `_VALUE_COLS`, `_METEO_COLS`, `_PROVENANCE_HEADERS`, `_INDEX_PREFIX`.
- Produces:
  - `GROUP_KEYS = ("identity", "values", "meteo", "index", "provenance")` (module-level tuple).
  - `build_station_csv(domain, meta, daily_rows, index_rows, groups=None)` — `groups` is an iterable of group keys or `None` (= all). `date` always present. Column order unchanged. Returns the CSV `str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_station_export.py` (the helpers `_meta`, `_piezo_daily`, `_piezo_index`, `_rows`, `_IDENTITY`, `_PROVENANCE` already exist in that file):

```python
from dashboard.utils.station_export import GROUP_KEYS


def test_group_keys_canonical_order():
    assert GROUP_KEYS == ("identity", "values", "meteo", "index", "provenance")


def test_groups_subset_keeps_only_date_values_and_index():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups={"values", "index"})
    header = _rows(csv_text)[0].split(",")
    assert header == ["date", "niveau_nappe_eau", "profondeur_nappe",
                      "mois_ref", "ips_z", "ips_classe", "ips_flag"]
    # a January data row carries date + values + index, nothing else
    jan10 = _rows(csv_text)[1].split(",")
    assert jan10 == ["2020-01-10", "12.5", "3.1", "2020-01", "-0.95", "BAS", "normale"]


def test_groups_none_is_unchanged_full_output():
    full = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index())
    explicit = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups=set(GROUP_KEYS))
    assert full == explicit


def test_groups_empty_emits_only_date():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(), groups=set())
    assert _rows(csv_text)[0].split(",") == ["date"]
    assert _rows(csv_text)[1].split(",") == ["2020-01-10"]


def test_groups_ignores_unknown_keys():
    csv_text = build_station_csv("piezo", _meta(), _piezo_daily(), _piezo_index(),
                                 groups={"values", "bogus"})
    assert _rows(csv_text)[0].split(",") == ["date", "niveau_nappe_eau", "profondeur_nappe"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_station_export.py -q`
Expected: the 5 new tests FAIL — `GROUP_KEYS` ImportError / `build_station_csv() got an unexpected keyword argument 'groups'`.

- [ ] **Step 3: Implement the `groups` filter**

In `dashboard/utils/station_export.py`, add the constant near the other group constants (after `_PROVENANCE_HEADERS`):

```python
# Canonical group keys, in CSV column order. `date` is always emitted and is
# not part of any group.
GROUP_KEYS = ("identity", "values", "meteo", "index", "provenance")
```

Replace `build_station_csv` with the group-aware version (keeps the existing column order and per-row logic, just gates each block on group membership):

```python
def build_station_csv(domain, meta, daily_rows, index_rows, groups=None) -> str:
    if domain not in _INDEX_PREFIX:
        raise ValueError(f"unknown domain {domain!r}")
    active = set(GROUP_KEYS) if groups is None else (set(groups) & set(GROUP_KEYS))
    idx = index_by_month(index_rows)
    prefix = _INDEX_PREFIX[domain]
    value_cols = _VALUE_COLS[domain]

    identity_cells = [_fmt(meta.get(key)) for _, key in _IDENTITY_COLS]
    provenance_cells = [_INDEX_REF, _UNIT[domain], _SOURCE, _fmt(meta.get("generated_on"))]

    header = []
    if "identity" in active:
        header += [h for h, _ in _IDENTITY_COLS]
    header += ["date"]
    if "values" in active:
        header += [h for h, _ in value_cols]
    if "meteo" in active:
        header += [h for h, _ in _METEO_COLS]
    if "index" in active:
        header += ["mois_ref", f"{prefix}_z", f"{prefix}_classe", f"{prefix}_flag"]
    if "provenance" in active:
        header += _PROVENANCE_HEADERS

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)

    for row in daily_rows:
        mk = _month_key(row["date"])
        ix = idx.get(mk)
        rec = []
        if "identity" in active:
            rec += identity_cells
        rec.append(_fmt(row.get("date")))
        if "values" in active:
            rec += [_fmt(row.get(k)) for _, k in value_cols]
        if "meteo" in active:
            rec += [_fmt(row.get(k)) for _, k in _METEO_COLS]
        if "index" in active:
            if ix:
                rec += [f"{mk[0]:04d}-{mk[1]:02d}", _fmt(ix["z"]),
                        _fmt(ix["index_class"]), _fmt(ix["flag"])]
            else:
                rec += ["", "", "", ""]
        if "provenance" in active:
            rec += provenance_cells
        writer.writerow(rec)

    return out.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_station_export.py -q`
Expected: PASS (all tests, including the pre-existing ones — `groups=None` keeps the full output identical).

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/station_export.py tests/test_station_export.py
git commit -m "feat(export): group-level column filter in station CSV builder"
```

---

## Task 2: Routers — `groups` query param

**Files:**
- Modify: `api/routers/observatory_piezo.py` (function `export_csv`, ends ~line 562)
- Modify: `api/routers/observatory_hydro.py` (function `export_csv`, ends ~line 727)
- Test: `tests/test_station_export_endpoint.py`

**Interfaces:**
- Consumes: `build_station_csv(..., groups=...)` and `GROUP_KEYS` from Task 1.
- Produces: both `export_csv` endpoints accept `groups: Optional[str] = Query(None)` and forward a parsed set (or `None`) to the builder.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_station_export_endpoint.py` (module already imports `inspect`, `observatory_piezo`, `observatory_hydro`):

```python
def test_export_endpoints_wire_groups_param():
    import inspect as _inspect
    for mod in (observatory_piezo, observatory_hydro):
        sig = _inspect.signature(mod.export_csv)
        assert "groups" in sig.parameters, f"{mod.__name__}.export_csv missing groups"
        src = _inspect.getsource(mod.export_csv)
        # intersect against the canonical group set and pass through to the builder
        assert "GROUP_KEYS" in src
        assert "groups=" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_station_export_endpoint.py::test_export_endpoints_wire_groups_param -q`
Expected: FAIL — `export_csv missing groups`.

- [ ] **Step 3: Implement in piezo router**

In `api/routers/observatory_piezo.py`, update the import (the file already has `from dashboard.utils.station_export import build_station_csv`):

```python
from dashboard.utils.station_export import build_station_csv, GROUP_KEYS
```

Add the param to the `export_csv` signature (alongside the existing `start_date` / `end_date`):

```python
def export_csv(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    groups: Optional[str] = Query(None),
):
```

Just before the `body = build_station_csv(...)` call, parse the param:

```python
    selected = None
    if groups is not None:
        selected = {g.strip() for g in groups.split(",") if g.strip()} & set(GROUP_KEYS)
```

And pass it to the builder:

```python
    body = build_station_csv(
        "piezo", {**dict(meta), "generated_on": date.today().isoformat()}, daily, index_rows,
        groups=selected,
    )
```

- [ ] **Step 4: Implement in hydro router**

In `api/routers/observatory_hydro.py`, mirror the change. Update import:

```python
from dashboard.utils.station_export import build_station_csv, GROUP_KEYS
```

Signature:

```python
def export_csv(
    code_station: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    groups: Optional[str] = Query(None),
):
```

Before the builder call:

```python
    selected = None
    if groups is not None:
        selected = {g.strip() for g in groups.split(",") if g.strip()} & set(GROUP_KEYS)
```

Builder call:

```python
    body = build_station_csv(
        "hydro", {**dict(meta), "generated_on": date.today().isoformat()}, daily, index_rows,
        groups=selected,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_station_export_endpoint.py tests/test_station_export.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routers/observatory_piezo.py api/routers/observatory_hydro.py tests/test_station_export_endpoint.py
git commit -m "feat(export): groups query param on station export endpoints"
```

---

## Task 3: Frontend API — `exportUrl(code, range, groups?)`

**Files:**
- Modify: `frontend/src/lib/observatory-api.ts` (`exportUrl` for piezo ~line 76 and hydro ~line 93; `exportQuery` ~line 52)
- Test: `frontend/src/lib/observatory-api.test.ts`

**Interfaces:**
- Produces: `exportUrl(code: string, range?: ExportRange, groups?: string[])`. When `groups` is a strict subset of the five keys, append `&groups=<comma-joined>`; when it is undefined or contains all five keys, omit the param.
- Canonical key list available to callers: `EXPORT_COLUMN_GROUPS = ['identity','values','meteo','index','provenance'] as const`.

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/lib/observatory-api.test.ts`:

```typescript
import { EXPORT_COLUMN_GROUPS } from './observatory-api'

it('appends groups when a strict subset is selected', () => {
  expect(
    observatoryApi.piezo.exportUrl('BSS000/X', undefined, ['values', 'index']),
  ).toBe(
    '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv?groups=values%2Cindex',
  )
})

it('omits groups when all five groups are selected', () => {
  expect(
    observatoryApi.hydro.exportUrl('K001', undefined, [...EXPORT_COLUMN_GROUPS]),
  ).toBe('/api/v1/observatory/hydro/stations/K001/export.csv')
})

it('combines date range and groups', () => {
  expect(
    observatoryApi.piezo.exportUrl('BSS000/X', { start_date: '2020-01-01', end_date: '' }, ['values']),
  ).toBe(
    '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv?start_date=2020-01-01&groups=values',
  )
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- observatory-api`
Expected: FAIL — `EXPORT_COLUMN_GROUPS` undefined / `exportUrl` ignores the third arg.

- [ ] **Step 3: Implement**

In `frontend/src/lib/observatory-api.ts`, export the canonical list near the top (after the imports):

```typescript
export const EXPORT_COLUMN_GROUPS = ['identity', 'values', 'meteo', 'index', 'provenance'] as const
export type ExportColumnGroup = (typeof EXPORT_COLUMN_GROUPS)[number]
```

Replace `exportQuery` so it also handles groups (a strict subset only):

```typescript
function exportQuery(range?: ExportRange, groups?: string[]): string {
  const qs = new URLSearchParams()
  if (range?.start_date) qs.set('start_date', range.start_date)
  if (range?.end_date) qs.set('end_date', range.end_date)
  if (groups && groups.length > 0 && groups.length < EXPORT_COLUMN_GROUPS.length) {
    qs.set('groups', groups.join(','))
  }
  const s = qs.toString()
  return s ? `?${s}` : ''
}
```

Update both `exportUrl` definitions to forward `groups`:

```typescript
    exportUrl: (code: string, range?: ExportRange, groups?: string[]) =>
      `${API_BASE}/observatory/piezo/stations/${encodeURIComponent(code)}/export.csv${exportQuery(range, groups)}`,
```

```typescript
    exportUrl: (code: string, range?: ExportRange, groups?: string[]) =>
      `${API_BASE}/observatory/hydro/stations/${encodeURIComponent(code)}/export.csv${exportQuery(range, groups)}`,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- observatory-api`
Expected: PASS (including the existing date-range tests — `exportQuery(undefined)` returns `''`, and a full selection omits `groups`).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/observatory-api.ts frontend/src/lib/observatory-api.test.ts
git commit -m "feat(export): exportUrl groups arg + EXPORT_COLUMN_GROUPS"
```

---

## Task 4: i18n labels for the five groups + picker button

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json` (under `mainPages.station`, near `"exportCsv"` ~line 1052)
- Modify: `frontend/src/i18n/locales/en.json` (matching `mainPages.station` block)

**Interfaces:**
- Produces i18n keys consumed by Task 5:
  - `mainPages.station.exportColumnsButton` (e.g. `"Colonnes"`)
  - `mainPages.station.exportColumns.identity|values|meteo|index|provenance`

- [ ] **Step 1: Add the French keys**

In `frontend/src/i18n/locales/fr.json`, add immediately after the `"exportEndDate"` line inside `mainPages.station`:

```json
      "exportColumnsButton": "Colonnes",
      "exportColumns": {
        "identity": "Identité",
        "values": "Mesures",
        "meteo": "Météo",
        "index": "Index",
        "provenance": "Provenance"
      },
```

- [ ] **Step 2: Add the English keys**

In `frontend/src/i18n/locales/en.json`, add the same keys in the matching `mainPages.station` block (find the `"exportEndDate"` entry and add after it):

```json
      "exportColumnsButton": "Columns",
      "exportColumns": {
        "identity": "Identity",
        "values": "Measurements",
        "meteo": "Weather",
        "index": "Index",
        "provenance": "Provenance"
      },
```

- [ ] **Step 3: Verify JSON validity**

Run: `cd frontend && node -e "JSON.parse(require('fs').readFileSync('src/i18n/locales/fr.json')); JSON.parse(require('fs').readFileSync('src/i18n/locales/en.json')); console.log('both valid')"`
Expected: `both valid`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/locales/fr.json frontend/src/i18n/locales/en.json
git commit -m "i18n(export): labels for column-group picker"
```

---

## Task 5: `ColumnPicker` popover component

**Files:**
- Create: `frontend/src/components/observatory/ColumnPicker.tsx`

**Interfaces:**
- Consumes: `EXPORT_COLUMN_GROUPS`, `ExportColumnGroup` from Task 3; i18n keys from Task 4.
- Produces: `ColumnPicker` default export — `{ selected: ExportColumnGroup[]; onChange: (next: ExportColumnGroup[]) => void }`. Renders a button `Colonnes (n/5) ▾` that toggles a popover with five checkboxes. Closes on outside click and Escape.

- [ ] **Step 1: Create the component**

Create `frontend/src/components/observatory/ColumnPicker.tsx`:

```tsx
import { useEffect, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { EXPORT_COLUMN_GROUPS, type ExportColumnGroup } from '@/lib/observatory-api'

interface Props {
  selected: ExportColumnGroup[]
  onChange: (next: ExportColumnGroup[]) => void
}

export default function ColumnPicker({ selected, onChange }: Props) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  function toggle(group: ExportColumnGroup) {
    onChange(
      selected.includes(group)
        ? selected.filter((g) => g !== group)
        : [...EXPORT_COLUMN_GROUPS].filter((g) => g === group || selected.includes(g)),
    )
  }

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-bg-card border border-white/10 text-text-secondary hover:text-text-primary transition-colors"
      >
        {t('mainPages.station.exportColumnsButton')} ({selected.length}/{EXPORT_COLUMN_GROUPS.length})
        <ChevronDown className="w-3.5 h-3.5" />
      </button>
      {open && (
        <div className="absolute right-0 mt-1 z-20 w-44 bg-bg-card border border-white/10 rounded-lg p-2 shadow-lg">
          {EXPORT_COLUMN_GROUPS.map((group) => (
            <label key={group} className="flex items-center gap-2 px-2 py-1.5 text-xs text-text-secondary hover:text-text-primary cursor-pointer">
              <input
                type="checkbox"
                checked={selected.includes(group)}
                onChange={() => toggle(group)}
                className="accent-accent-cyan"
              />
              {t(`mainPages.station.exportColumns.${group}`)}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors related to `ColumnPicker.tsx`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/observatory/ColumnPicker.tsx
git commit -m "feat(export): ColumnPicker popover component"
```

---

## Task 6: Wire `ColumnPicker` into `StationPage`

**Files:**
- Modify: `frontend/src/pages/StationPage.tsx` (import ~line 13, state ~line 51, export bar ~line 109-119)

**Interfaces:**
- Consumes: `ColumnPicker` (Task 5), `EXPORT_COLUMN_GROUPS` / `ExportColumnGroup` (Task 3), `exportUrl` groups arg (Task 3).

- [ ] **Step 1: Add imports**

In `frontend/src/pages/StationPage.tsx`, after the existing `import { observatoryApi } from '../lib/observatory-api'` line, add:

```tsx
import { EXPORT_COLUMN_GROUPS, type ExportColumnGroup } from '../lib/observatory-api'
import ColumnPicker from '@/components/observatory/ColumnPicker'
```

- [ ] **Step 2: Add selected-groups state**

After the existing `const [exportEnd, setExportEnd] = useState('')` line, add:

```tsx
  const [exportGroups, setExportGroups] = useState<ExportColumnGroup[]>([...EXPORT_COLUMN_GROUPS])
```

- [ ] **Step 3: Render the picker and pass groups to the URL**

In the export bar, add `<ColumnPicker .../>` right before the `<a href=...>` download link, and add the `exportGroups` arg to the `exportUrl` call. Replace the existing `<a ... href={... .exportUrl(code, { start_date: exportStart, end_date: exportEnd })} ...>` opening with:

```tsx
            <ColumnPicker selected={exportGroups} onChange={setExportGroups} />
            <a
              href={(isPiezo ? observatoryApi.piezo : observatoryApi.hydro).exportUrl(code, { start_date: exportStart, end_date: exportEnd }, exportGroups)}
```

(Leave the rest of the `<a>` element — className, `<Download/>`, label — unchanged.)

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: builds with no errors.

- [ ] **Step 5: Run the frontend test suite**

Run: `cd frontend && npm test -- observatory-api`
Expected: PASS (URL building unchanged when all groups selected — the default).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/StationPage.tsx
git commit -m "feat(export): column picker on station page export bar"
```

---

## Task 7: End-to-end verification + deploy

**Files:** none (verification only).

- [ ] **Step 1: Backend live check — subset**

Rebuild the backend and confirm a subset request drops the right columns:

```bash
docker compose up -d --build backend
```

Then (wait for healthy), from the host:

```bash
docker exec junon-backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/observatory/piezo/stations/02648X0020%2FS1/export.csv?groups=values,index', timeout=30).read().decode('utf-8-sig').splitlines()[0])"
```

Expected header line: `date,niveau_nappe_eau,profondeur_nappe,mois_ref,ips_z,ips_classe,ips_flag`

- [ ] **Step 2: Backend live check — bare URL unchanged**

```bash
docker exec junon-backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/observatory/piezo/stations/02648X0020%2FS1/export.csv', timeout=30).read().decode('utf-8-sig').splitlines()[0])"
```

Expected: the full header (`code,nom_station,…,genere_le`) — backward compatible.

- [ ] **Step 3: Frontend deploy**

```bash
docker compose up -d --build frontend
```

- [ ] **Step 4: Manual UI smoke test**

Open a station page, click `Colonnes (5/5) ▾`, uncheck e.g. Météo and Provenance, download, and confirm the CSV omits those columns while keeping `date`. Re-check all five and confirm the URL has no `groups` param (network tab) and the export matches the pre-feature output.

- [ ] **Step 5: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "chore(export): column selection verified end-to-end"
```

---

## Notes for the implementer

- The builder is pure (no Streamlit/FastAPI/DB) — keep it that way (`dashboard/utils/` convention).
- Don't change column order or names; selection only includes/excludes whole groups.
- The `date` column is mandatory: it must appear even when `groups` is empty.
- Keep the bare `export.csv` URL behaviour identical (no `groups` ⇒ all columns) so existing links and the spec's backward-compat guarantee hold.
