# ERA5 Weather — Phase 0 (baseline grid, daily) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the ERA5 grid on the Observatoire map as coloured 0.1° squares for a selectable day, with a variable selector (temp/precip/ETP), a legend, and a click popup listing the three values.

**Architecture:** Fix the broken ERA5 endpoints (wrong table name), add a `/range` endpoint for date bounds, and make `/snapshot` default to the latest day. On the frontend, add two pure helper libs (square-polygon builder + colour scales), a `useERA5Range` hook, a MapLibre `fill` layer drawn below station markers, and a "Météo (ERA5)" control panel in the existing RightDrawer.

**Tech Stack:** FastAPI + sync SQLAlchemy (warehouse `brgm-postgres`), React 19 + MapLibre GL v5 + @tanstack/react-query + react-i18next, vitest for pure-helper unit tests.

## Global Constraints

- **DB table:** ERA5 daily values live in `gold.int_era5_for_all_stations` (NOT `int_era5_for_stations`, which does not exist). Grid definition is `gold.int_era5_grid_points`.
- **ERA5 columns:** `latitude`, `longitude`, `era5_date`, `temperature_2m` (°C), `total_precipitation` (mm), `potential_evaporation` (stored NEGATIVE — water-loss convention).
- **Grid:** 0.1°×0.1°; a cell centred at (lat, lon) spans ±0.05°.
- **UI language:** French only (BRGM audience). All user-facing strings go through `react-i18next` keys in `frontend/src/i18n/locales/fr.json`. No franglais.
- **API base:** `/api/v1` (`frontend/src/lib/constants.ts`). ERA5 endpoints under `/api/v1/observatory/era5`.
- **Caching:** warehouse endpoints wrap their fetch in `get_cached(key, params, ttl, fetch)` from `dashboard.utils.cache`; TTL 86400.
- **Engine:** use `get_brgm_sync_engine()` from `api.database`; never dispose it (`finally: pass`).
- **Pure-helper TDD:** new logic in `frontend/src/lib/*.ts` is unit-tested with vitest (`cd frontend && npx vitest run <file>`). Map/drawer/page wiring and the DB-backed endpoints are verified by explicit integration commands, not unit tests.
- **Backend tests run:** `DEBUG=true DB_PASSWORD=test uv run pytest <files> -q` (the repo `.env` otherwise trips Settings).
- **Local services:** backend container `junon-backend` (host port 49514), frontend `junon-frontend` (49513), warehouse `brgm-postgres` (host 49502 / in-container 5432).

---

### Task 1: Fix ERA5 endpoints + add `/range` + default-latest `/snapshot`

**Files:**
- Modify: `api/routers/observatory_era5.py`

**Interfaces:**
- Produces (HTTP):
  - `GET /api/v1/observatory/era5/snapshot?date=YYYY-MM-DD` → `[{latitude, longitude, temperature_2m, total_precipitation, potential_evaporation}]`. `date` optional; omitted → latest available day. Only rows with ≥1 non-null value.
  - `GET /api/v1/observatory/era5/range` → `{min_date: "YYYY-MM-DD", max_date: "YYYY-MM-DD"}`.

- [ ] **Step 1: Fix the wrong table name in all three queries**

In `api/routers/observatory_era5.py`, replace every `gold.int_era5_for_stations` with `gold.int_era5_for_all_stations` (lines 55, 74, 103). Leave `gold.int_era5_grid_points` untouched.

- [ ] **Step 2: Make `/snapshot` accept an optional date and default to the latest day, filtering all-null rows**

Replace the whole `get_era5_snapshot` function with:

```python
@router.get("/snapshot")
def get_era5_snapshot(
    snapshot_date: DateType | None = Query(
        None, alias="date", description="ERA5 snapshot date; latest available day if omitted"
    ),
):
    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                d = snapshot_date
                if d is None:
                    d = conn.execute(
                        text("SELECT max(era5_date) FROM gold.int_era5_for_all_stations")
                    ).scalar()
                query = """
                    SELECT latitude, longitude,
                           temperature_2m, total_precipitation, potential_evaporation
                    FROM gold.int_era5_for_all_stations
                    WHERE era5_date = :d
                      AND (temperature_2m IS NOT NULL
                           OR total_precipitation IS NOT NULL
                           OR potential_evaporation IS NOT NULL)
                """
                result = conn.execute(text(query), {"d": d})
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_snapshot",
        {"date": str(snapshot_date) if snapshot_date else "latest"},
        SNAPSHOT_TTL,
        fetch,
    )
```

- [ ] **Step 3: Add the `/range` endpoint**

Add `RANGE_TTL = 86400` near the other TTL constants, then append this router function after `get_era5_snapshot`:

```python
@router.get("/range")
def get_era5_range():
    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT min(era5_date) AS min_date, max(era5_date) AS max_date "
                        "FROM gold.int_era5_for_all_stations"
                    )
                ).mappings().first()
                return {"min_date": str(row["min_date"]), "max_date": str(row["max_date"])}
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_range", {}, RANGE_TTL, fetch)
```

- [ ] **Step 4: Rebuild the backend container and verify the endpoints against the live warehouse**

```bash
docker compose up -d --build backend
sleep 5
curl -s "http://localhost:49514/api/v1/observatory/era5/range"
```
Expected: a JSON object like `{"min_date":"1950-01-02","max_date":"2026-06-24"}`.

```bash
curl -s "http://localhost:49514/api/v1/observatory/era5/snapshot" | head -c 300
```
Expected: a non-empty JSON array of objects each with `latitude, longitude, temperature_2m, total_precipitation, potential_evaporation` (latest-day data, previously this 500'd on the missing table).

```bash
curl -s "http://localhost:49514/api/v1/observatory/era5/snapshot?date=2024-01-15" | head -c 120
```
Expected: a non-empty JSON array for that explicit date.

- [ ] **Step 5: Commit**

```bash
git add api/routers/observatory_era5.py
git commit -m "fix(era5): correct table name; add /range; snapshot defaults to latest day"
```

---

### Task 2: Pure helper — ERA5 points → square polygons

**Files:**
- Create: `frontend/src/lib/era5-grid.ts`
- Test: `frontend/src/lib/era5-grid.test.ts`

**Interfaces:**
- Consumes: `ERA5GridPoint` from `./observatory-types`.
- Produces: `era5PointsToSquares(points: ERA5GridPoint[]): GeoJSON.FeatureCollection<GeoJSON.Polygon, ERA5CellProps>` where each feature is a 0.1° square centred on the point, carrying the three values; `ERA5CellProps = { temperature_2m: number | null; total_precipitation: number | null; potential_evaporation: number | null }`. Exports `ERA5_CELL_HALF = 0.05`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest'
import { era5PointsToSquares, ERA5_CELL_HALF } from './era5-grid'

describe('era5PointsToSquares', () => {
  it('builds one square polygon per point, centred on lat/lon', () => {
    const fc = era5PointsToSquares([
      { latitude: 48, longitude: 2, temperature_2m: 12.3, total_precipitation: 4, potential_evaporation: -3.1 },
    ])
    expect(fc.type).toBe('FeatureCollection')
    expect(fc.features).toHaveLength(1)
    const f = fc.features[0]
    expect(f.geometry.type).toBe('Polygon')
    // ring is closed (5 coords) and spans centre ± half in both axes
    const ring = f.geometry.coordinates[0]
    expect(ring).toHaveLength(5)
    expect(ring[0]).toEqual([2 - ERA5_CELL_HALF, 48 - ERA5_CELL_HALF])
    expect(ring[4]).toEqual(ring[0])
    const lons = ring.map(c => c[0])
    const lats = ring.map(c => c[1])
    expect(Math.min(...lons)).toBeCloseTo(2 - ERA5_CELL_HALF)
    expect(Math.max(...lons)).toBeCloseTo(2 + ERA5_CELL_HALF)
    expect(Math.min(...lats)).toBeCloseTo(48 - ERA5_CELL_HALF)
    expect(Math.max(...lats)).toBeCloseTo(48 + ERA5_CELL_HALF)
  })

  it('carries the three values as feature properties', () => {
    const fc = era5PointsToSquares([
      { latitude: 45, longitude: 5, temperature_2m: 9, total_precipitation: 0, potential_evaporation: -2 },
    ])
    expect(fc.features[0].properties).toEqual({
      temperature_2m: 9, total_precipitation: 0, potential_evaporation: -2,
    })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/era5-grid.test.ts`
Expected: FAIL — cannot resolve `./era5-grid`.

- [ ] **Step 3: Write minimal implementation**

```ts
// frontend/src/lib/era5-grid.ts
import type { ERA5GridPoint } from './observatory-types'

export const ERA5_CELL_HALF = 0.05

export interface ERA5CellProps {
  temperature_2m: number | null
  total_precipitation: number | null
  potential_evaporation: number | null
}

export function era5PointsToSquares(
  points: ERA5GridPoint[],
): GeoJSON.FeatureCollection<GeoJSON.Polygon, ERA5CellProps> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points.map((p) => {
      const lon = Number(p.longitude)
      const lat = Number(p.latitude)
      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [[
            [lon - h, lat - h],
            [lon + h, lat - h],
            [lon + h, lat + h],
            [lon - h, lat + h],
            [lon - h, lat - h],
          ]],
        },
        properties: {
          temperature_2m: p.temperature_2m,
          total_precipitation: p.total_precipitation,
          potential_evaporation: p.potential_evaporation,
        },
      }
    }),
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/era5-grid.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/era5-grid.ts frontend/src/lib/era5-grid.test.ts
git commit -m "feat(era5): pure helper to build 0.1deg square polygons from grid points"
```

---

### Task 3: Pure helper — ERA5 colour scales + legend

**Files:**
- Create: `frontend/src/lib/era5-colors.ts`
- Test: `frontend/src/lib/era5-colors.test.ts`

**Interfaces:**
- Produces:
  - `type Era5Variable = 'temperature' | 'precipitation' | 'evaporation'`
  - `interface Era5VarConfig { key: Era5Variable; prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation'; unit: string; labelKey: string; stops: Array<[number, string]> }`
  - `ERA5_VARIABLES: Record<Era5Variable, Era5VarConfig>`
  - `era5ColorExpression(v: Era5Variable): unknown[]` — a MapLibre `interpolate` expression over the variable's numeric property.
  - `era5FormatValue(v: Era5Variable, value: number | null): string` — display string (ETP shown as magnitude with unit; null → "—").

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest'
import { ERA5_VARIABLES, era5ColorExpression, era5FormatValue } from './era5-colors'

describe('era5-colors', () => {
  it('maps each variable to its data property', () => {
    expect(ERA5_VARIABLES.temperature.prop).toBe('temperature_2m')
    expect(ERA5_VARIABLES.precipitation.prop).toBe('total_precipitation')
    expect(ERA5_VARIABLES.evaporation.prop).toBe('potential_evaporation')
  })

  it('builds an interpolate expression reading the right property', () => {
    const expr = era5ColorExpression('temperature') as any[]
    expect(expr[0]).toBe('interpolate')
    expect(expr[2]).toEqual(['to-number', ['get', 'temperature_2m']])
    // remaining entries are alternating stop/colour pairs
    expect(expr.length).toBeGreaterThan(4)
  })

  it('formats ETP as a positive magnitude and null as a dash', () => {
    expect(era5FormatValue('evaporation', -3.1)).toBe('3.1 mm')
    expect(era5FormatValue('temperature', 12.34)).toBe('12.3 °C')
    expect(era5FormatValue('precipitation', null)).toBe('—')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: FAIL — cannot resolve `./era5-colors`.

- [ ] **Step 3: Write minimal implementation**

```ts
// frontend/src/lib/era5-colors.ts
export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation'

export interface Era5VarConfig {
  key: Era5Variable
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation'
  unit: string
  labelKey: string
  stops: Array<[number, string]>
}

export const ERA5_VARIABLES: Record<Era5Variable, Era5VarConfig> = {
  temperature: {
    key: 'temperature', prop: 'temperature_2m', unit: '°C',
    labelKey: 'observatory.drawer.era5VarTemperature',
    stops: [[-10, '#2166ac'], [0, '#67a9cf'], [10, '#d1e5f0'], [20, '#fddbc7'], [27, '#ef8a62'], [35, '#b2182b']],
  },
  precipitation: {
    key: 'precipitation', prop: 'total_precipitation', unit: 'mm',
    labelKey: 'observatory.drawer.era5VarPrecipitation',
    stops: [[0, '#f7fbff'], [5, '#c6dbef'], [15, '#6baed6'], [30, '#2171b5'], [50, '#08306b']],
  },
  evaporation: {
    key: 'evaporation', prop: 'potential_evaporation', unit: 'mm',
    labelKey: 'observatory.drawer.era5VarEvaporation',
    // stored negative; more negative = more evapotranspiration
    stops: [[-10, '#54278f'], [-6, '#756bb1'], [-3, '#9e9ac8'], [-1, '#cbc9e2'], [0, '#f2f0f7']],
  },
}

export function era5ColorExpression(v: Era5Variable): unknown[] {
  const cfg = ERA5_VARIABLES[v]
  const expr: unknown[] = ['interpolate', ['linear'], ['to-number', ['get', cfg.prop]]]
  for (const [value, color] of cfg.stops) {
    expr.push(value, color)
  }
  return expr
}

export function era5FormatValue(v: Era5Variable, value: number | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  const cfg = ERA5_VARIABLES[v]
  const shown = v === 'evaporation' ? Math.abs(value) : value
  return `${shown.toFixed(1)} ${cfg.unit}`
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/era5-colors.ts frontend/src/lib/era5-colors.test.ts
git commit -m "feat(era5): colour scales, MapLibre colour expression, value formatting"
```

---

### Task 4: API client + hook + types for `/range`

**Files:**
- Modify: `frontend/src/lib/observatory-types.ts`
- Modify: `frontend/src/lib/observatory-api.ts:109-114`
- Modify: `frontend/src/hooks/useObservatory.ts` (ERA5 hooks block, ~329-363)

**Interfaces:**
- Consumes: `fetchJson`, existing `observatoryApi.era5`.
- Produces:
  - type `ERA5Range { min_date: string; max_date: string }`
  - `observatoryApi.era5.range(): Promise<ERA5Range>`
  - `useERA5Range()` react-query hook returning `ERA5Range`.
  - (already present) `useERA5Snapshot(date)`.

- [ ] **Step 1: Add the `ERA5Range` type**

In `frontend/src/lib/observatory-types.ts`, right after the `ERA5GridPoint` interface (ends at line 199), add:

```ts
export interface ERA5Range {
  min_date: string
  max_date: string
}
```

- [ ] **Step 2: Add the `range` client method**

In `frontend/src/lib/observatory-api.ts`, import the type (add `ERA5Range` to the existing type import block at lines 3-15) and extend the `era5` object (lines 109-114) so it reads:

```ts
  era5: {
    grid: () => fetchJson<ERA5GridPoint[]>('/observatory/era5/grid'),
    snapshot: (date: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/snapshot', { date }),
    dates: () => fetchJson<string[]>('/observatory/era5/dates'),
    monthly: (month: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/monthly', { month }),
    range: () => fetchJson<ERA5Range>('/observatory/era5/range'),
  },
```

- [ ] **Step 3: Add the `useERA5Range` hook**

In `frontend/src/hooks/useObservatory.ts`, inside the `// --- ERA5 hooks ---` block (after `useERA5Monthly`, before line 365), add:

```ts
export function useERA5Range() {
  return useQuery({
    queryKey: ['obs-era5', 'range'],
    queryFn: () => observatoryApi.era5.range(),
    staleTime: 24 * 60 * 60 * 1000,
  })
}
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors (exit 0).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/observatory-types.ts frontend/src/lib/observatory-api.ts frontend/src/hooks/useObservatory.ts
git commit -m "feat(era5): ERA5Range type, range() client method, useERA5Range hook"
```

---

### Task 5: Render the ERA5 grid layer + click popup on the map

**Files:**
- Modify: `frontend/src/components/observatory/ObservatoryMap.tsx` (Props interface ~35-65; destructure ~268-281; add one new effect after the WFS effect ~624)

**Interfaces:**
- Consumes: `era5PointsToSquares` (Task 2), `era5ColorExpression`/`ERA5_VARIABLES`/`era5FormatValue` (Task 3), `Era5Variable` type, `ERA5GridPoint`.
- Produces new props on `ObservatoryMap`:
  - `era5Active?: boolean`
  - `era5Points?: ERA5GridPoint[]`
  - `era5Variable?: Era5Variable`
- Map artefacts: source `era5-grid`, fill layer `era5-grid-fill` (inserted before `'piezo-clusters'` so it sits under station markers).

- [ ] **Step 1: Add imports**

At the top of `ObservatoryMap.tsx`, add:

```ts
import { era5PointsToSquares } from '@/lib/era5-grid'
import { era5ColorExpression, era5FormatValue, ERA5_VARIABLES } from '@/lib/era5-colors'
import type { Era5Variable } from '@/lib/era5-colors'
import type { ERA5GridPoint } from '@/lib/observatory-types'
```

(If `maplibregl` is not already imported by name, note the file already uses it for `maplibregl.Map`; reuse that same import for `maplibregl.Popup`.)

- [ ] **Step 2: Extend the Props interface and destructuring**

In the `Props` interface (ends line 65) add:

```ts
  era5Active?: boolean
  era5Points?: ERA5GridPoint[]
  era5Variable?: Era5Variable
```

In the destructured signature (ends line 281) add, before the closing `}: Props) {`:

```ts
  era5Active = false, era5Points, era5Variable = 'temperature',
```

- [ ] **Step 3: Add the ERA5 layer effect**

After the WFS layer effect (around line 624, just before the next effect), add:

```tsx
  // --- ERA5 weather grid (Phase 0: daily squares) ---
  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapLoaded) return
    const SRC = 'era5-grid'
    const FILL = 'era5-grid-fill'

    if (!era5Active) {
      if (map.getLayer(FILL)) map.setLayoutProperty(FILL, 'visibility', 'none')
      return
    }

    const cfg = ERA5_VARIABLES[era5Variable]
    const pts = (era5Points ?? []).filter((p) => p[cfg.prop] != null)
    const data = era5PointsToSquares(pts)

    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: 'geojson', data })
      map.addLayer(
        {
          id: FILL,
          type: 'fill',
          source: SRC,
          paint: {
            'fill-color': era5ColorExpression(era5Variable) as any,
            'fill-opacity': 0.6,
          },
        },
        map.getLayer('piezo-clusters') ? 'piezo-clusters' : undefined,
      )
      map.on('click', FILL, (e) => {
        const f = e.features?.[0]
        if (!f) return
        const pr = f.properties as Record<string, string>
        const num = (k: string) => (pr[k] === undefined || pr[k] === null || pr[k] === '' ? null : Number(pr[k]))
        const html = `<div style="font-size:12px;line-height:1.5">
            <div>${t('observatory.era5.popupTemperature')}: ${era5FormatValue('temperature', num('temperature_2m'))}</div>
            <div>${t('observatory.era5.popupPrecipitation')}: ${era5FormatValue('precipitation', num('total_precipitation'))}</div>
            <div>${t('observatory.era5.popupEvaporation')}: ${era5FormatValue('evaporation', num('potential_evaporation'))}</div>
          </div>`
        new maplibregl.Popup({ closeButton: true })
          .setLngLat(e.lngLat)
          .setHTML(html)
          .addTo(map)
      })
      map.on('mouseenter', FILL, () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', FILL, () => { map.getCanvas().style.cursor = '' })
    } else {
      ;(map.getSource(SRC) as maplibregl.GeoJSONSource).setData(data)
      map.setPaintProperty(FILL, 'fill-color', era5ColorExpression(era5Variable) as any)
    }
    map.setLayoutProperty(FILL, 'visibility', 'visible')
  }, [mapLoaded, era5Active, era5Points, era5Variable, t])
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors. (If `e.features` typing complains, the file's existing handlers show the accepted cast pattern — mirror it.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/observatory/ObservatoryMap.tsx
git commit -m "feat(era5): render ERA5 grid fill layer below stations with click popup"
```

---

### Task 6: Weather control panel (RightDrawer) + page wiring + i18n

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json` (the `observatory.drawer` block at ~183-234; add an `observatory.era5` block)
- Modify: `frontend/src/components/observatory/RightDrawer.tsx`
- Modify: `frontend/src/pages/ObservatoryPage.tsx`

**Interfaces:**
- Consumes: `useERA5Range`, `useERA5Snapshot` (hooks), `ERA5_VARIABLES`/`Era5Variable` (Task 3), the new `ObservatoryMap` props (Task 5).
- Produces new `RightDrawer` props: `era5Active`, `setEra5Active`, `era5Variable`, `setEra5Variable`, `era5Date`, `setEra5Date`, `era5MinDate`, `era5MaxDate`.

- [ ] **Step 1: Add French i18n keys**

In `frontend/src/i18n/locales/fr.json`, inside the `observatory.drawer` object add:

```json
      "groupWeatherEra5": "Météo (ERA5)",
      "era5Layer": "Grille ERA5",
      "era5ColorBy": "Colorer par",
      "era5VarTemperature": "Température",
      "era5VarPrecipitation": "Précipitations",
      "era5VarEvaporation": "Évapotranspiration",
      "era5Date": "Date",
      "era5NoData": "Pas de données pour cette date",
```

And add a sibling `era5` object inside `observatory` (next to `drawer`):

```json
    "era5": {
      "popupTemperature": "Température",
      "popupPrecipitation": "Précipitations",
      "popupEvaporation": "ETP"
    },
```

- [ ] **Step 2: Extend RightDrawer props and add the panel**

In `RightDrawer.tsx`, import the variable config at the top:

```ts
import { ERA5_VARIABLES } from '@/lib/era5-colors'
import type { Era5Variable } from '@/lib/era5-colors'
```

Add to the `Props` interface (after line 38):

```ts
  era5Active: boolean; setEra5Active: (v: boolean) => void
  era5Variable: Era5Variable; setEra5Variable: (v: Era5Variable) => void
  era5Date: string; setEra5Date: (v: string) => void
  era5MinDate?: string; era5MaxDate?: string
```

Add a new `AccordionSection` immediately after the closing `</AccordionSection>` of the `layers` section (after line 141), inside the drawer container:

```tsx
        <AccordionSection id="era5" title={t('observatory.drawer.groupWeatherEra5')}>
          <label className="flex items-center gap-2 py-1 cursor-pointer group mb-2">
            <input type="checkbox" checked={props.era5Active} onChange={() => props.setEra5Active(!props.era5Active)} className="w-3.5 h-3.5 accent-accent-cyan rounded" />
            <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{t('observatory.drawer.era5Layer')}</span>
          </label>
          {props.era5Active && (
            <div className="space-y-3 border-t border-white/5 pt-2">
              <div>
                <label className="text-xs text-text-secondary block mb-1">{t('observatory.drawer.era5ColorBy')}</label>
                <div className="space-y-1">
                  {(Object.values(ERA5_VARIABLES)).map((cfg) => (
                    <label key={cfg.key} className="flex items-center gap-2 cursor-pointer group">
                      <input type="radio" name="era5-variable" checked={props.era5Variable === cfg.key} onChange={() => props.setEra5Variable(cfg.key)} className="w-3.5 h-3.5 accent-accent-cyan" />
                      <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{t(cfg.labelKey)}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-text-secondary block mb-1">{t('observatory.drawer.era5Date')}</label>
                <input type="date" value={props.era5Date} min={props.era5MinDate} max={props.era5MaxDate} onChange={(e) => props.setEra5Date(e.target.value)} className="w-full px-2.5 py-1.5 bg-bg-primary border border-white/10 rounded text-sm text-text-primary focus:outline-none focus:border-accent-cyan/50" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold text-white/50 uppercase tracking-wider">{t(ERA5_VARIABLES[props.era5Variable].labelKey)} ({ERA5_VARIABLES[props.era5Variable].unit})</span>
                <div className="flex items-center gap-1">
                  {ERA5_VARIABLES[props.era5Variable].stops.map(([v, c]) => (
                    <div key={v} className="flex flex-col items-center">
                      <span className="w-6 h-3 rounded-sm" style={{ backgroundColor: c }} />
                      <span className="text-[9px] text-text-secondary">{Math.abs(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </AccordionSection>
```

- [ ] **Step 3: Wire state and data in ObservatoryPage**

In `ObservatoryPage.tsx`:

Add imports to the existing hook import (line 11) — append `useERA5Range, useERA5Snapshot`:

```ts
import { useStationsGeoJSON, useWfsLayer, useObsFilters, useSectorSituation, useSectorTimeline, useERA5Range, useERA5Snapshot } from '@/hooks/useObservatory'
import type { Era5Variable } from '@/lib/era5-colors'
```

After the `showTerrain` state (line 80) add:

```ts
  const [era5Active, setEra5Active] = useState(false)
  const [era5Variable, setEra5Variable] = useState<Era5Variable>('temperature')
  const [era5Date, setEra5Date] = useState<string>('')
  const { data: era5Range } = useERA5Range()
  useEffect(() => { if (era5Range?.max_date && !era5Date) setEra5Date(era5Range.max_date) }, [era5Range, era5Date])
  const { data: era5Points } = useERA5Snapshot(era5Active && era5Date ? era5Date : undefined)
```

On the `<ObservatoryMap ... />` element (line 210) add the props:

```tsx
        era5Active={era5Active} era5Points={era5Points} era5Variable={era5Variable}
```

On the `<RightDrawer ... />` element (line 212) add the props:

```tsx
        era5Active={era5Active} setEra5Active={setEra5Active} era5Variable={era5Variable} setEra5Variable={setEra5Variable} era5Date={era5Date} setEra5Date={setEra5Date} era5MinDate={era5Range?.min_date} era5MaxDate={era5Range?.max_date}
```

- [ ] **Step 4: Type-check and run the pure-helper tests**

Run: `cd frontend && npx tsc --noEmit && npx vitest run src/lib/era5-grid.test.ts src/lib/era5-colors.test.ts`
Expected: tsc exit 0; vitest 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/i18n/locales/fr.json frontend/src/components/observatory/RightDrawer.tsx frontend/src/pages/ObservatoryPage.tsx
git commit -m "feat(era5): weather control panel (toggle, variable, date, legend) wired into Observatoire"
```

---

### Task 7: End-to-end manual verification in the app

**Files:** none (verification only).

- [ ] **Step 1: Rebuild and serve the frontend**

```bash
docker compose up -d --build frontend
```

- [ ] **Step 2: Verify in the browser** (http://localhost:49513 — log in, open the Observatoire)

Confirm each, fixing the responsible task if any fails:
- Open the right drawer → a "Météo (ERA5)" section exists; toggling "Grille ERA5" on draws coloured squares over France, **below** the station markers (markers still visible and clickable).
- The three colour-by radios switch the colouring; the legend updates with the variable's unit and scale.
- Changing the date input reloads the grid for that day; an out-of-range date is blocked by the min/max bounds.
- Clicking a cell opens a popup listing Température / Précipitations / ETP with units (ETP shown positive); empty cells where a value is missing are simply not drawn.
- Toggling the layer off removes the squares and leaves the rest of the map untouched (no implicit filtering of stations).

- [ ] **Step 3: Commit any fixes, then stop**

If fixes were needed, commit them with a `fix(era5): …` message. Phase 0 is complete when all checks pass.

---

## Self-Review

**Spec coverage (Phase 0 scope only):**
- Fix table-name bug → Task 1. ✓
- `snapshot` (default latest) + `range` → Task 1. ✓
- Grid squares rendering below stations → Tasks 2, 5. ✓
- Variable selector + legend → Tasks 3, 6. ✓
- Day stepper (date input bounded by range) → Tasks 4, 6. ✓
- Click popup with 3 values; ETP magnitude; missing cells not drawn → Tasks 3, 5, 7. ✓
- i18n French keys → Task 6. ✓
- Pure-helper unit tests → Tasks 2, 3. ✓
- Phases 1–4 (month/year, smoothed, by-zone, anomaly) are intentionally out of this plan; each gets its own plan.

**Placeholder scan:** No TBD/TODO; every code step contains full code. ✓

**Type consistency:** `era5PointsToSquares`, `era5ColorExpression`, `era5FormatValue`, `ERA5_VARIABLES`, `Era5Variable`, `ERA5Range`, `useERA5Range`, and the `era5Active/era5Points/era5Variable/era5Date` prop names are used identically across tasks. The map colours by `cfg.prop` matching the data columns returned by `/snapshot`. ✓
