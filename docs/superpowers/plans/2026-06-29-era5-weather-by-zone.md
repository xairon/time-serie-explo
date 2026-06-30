# ERA5 Weather — By-Zone Choropleth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When the ERA5 weather overlay is on, let the user aggregate the active weather variable (température / précipitations / ETP / anomalie) to the **active administrative/hydro zone layer** (départements, régions, HER2, bassins SANDRE, secteurs BSH) and render it as a choropleth — fully client-side, reusing already-fetched per-cell data and already-loaded zone polygons.

**Architecture:** A new `era5ByZone` toggle. When on (and a compatible static-GeoJSON zone layer is active), the map (1) hides the ERA5 squares, (2) averages the current per-cell ERA5 values into each zone polygon using point-in-polygon, and (3) overrides the active zone fill layer's `fill-color` with a data-driven match expression built from the aggregated per-zone means; toggling off restores the zone layer's original colours. No backend or warehouse changes.

**Tech Stack:** React 19 + MapLibre GL v5, vitest. Reuses `era5-colors.ts` scales and the existing point-in-polygon logic pattern.

**Builds on:** the ERA5 feature already on `feat/era5-weather-observatory` (panel, `era5Points`/`era5AnomalyPoints`, `era5Variable`, `era5-grid-fill` layer, zone fill layers `regions-fill`/`depts-fill`/`her-fill`/`bassins-fill`/`secteurs-fill`).

## Global Constraints
- **Client-side only.** No new endpoint, no dbt. Aggregation = arithmetic **mean** of the cells whose centre falls inside the zone polygon (mean for every variable, incl. precipitation — a choropleth must be area-normalised; sum would scale with cell count).
- **Compatible zones (static GeoJSON only):** `depts` (id prop `code`), `regions` (`code`), `her` (`code`), `bassins` (`CdBH`), `secteurs` (`sector_id`). WFS hydro zones are excluded (dynamic).
- **Follow the active layer.** By-zone uses whichever compatible zone layer is active. If `era5ByZone` is enabled with no compatible zone active, auto-activate `depts`.
- **Choropleth replaces squares.** While by-zone is active, the `era5-grid-fill` squares are hidden; restored when by-zone is off.
- **Opt-in / no implicit mutation.** `era5ByZone` defaults off. Do NOT change the zone click handlers (station filtering) — by-zone is colour-only; values are conveyed by the choropleth + legend. Restore original zone colours exactly on toggle-off.
- **UI French only**; strings via i18n keys. Frontend tests `cd frontend && npx vitest run <file>`; build `npm run build` (stricter than `tsc --noEmit` — use it). Local services: frontend `:49513`, backend `:49514`.

---

### Task 1: Pure helpers — zone aggregation + zone colour expression (TDD)

**Files:**
- Create: `frontend/src/lib/era5-zones.ts`
- Test: `frontend/src/lib/era5-zones.test.ts`

**Interfaces:**
- `pointInPolygonGeometry(lon, lat, geometry)` — ray-casting for Polygon/MultiPolygon (mirror the existing `pointInGeometry` logic in `ObservatoryPage.tsx`).
- `aggregateEra5ByZone(points, valueKey, zoneFeatures, idProp)` → `Record<string, number>`: for each zone feature, the mean of `point[valueKey]` (skipping null) over points whose `(longitude, latitude)` centre lies in the feature geometry; zones with no cell are omitted.
- `era5ZoneColorExpression(idProp, zoneValues, variable)` → a MapLibre `['match', ['get', idProp], code1, color1, …, fallback]` expression that maps each zone id to a colour by interpolating its mean through the variable's stops (reuse `ERA5_VARIABLES[variable].stops`); fallback colour `'rgba(0,0,0,0)'` (transparent) for zones with no data.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest'
import { aggregateEra5ByZone, era5ZoneColorExpression, pointInPolygonGeometry } from './era5-zones'

const square = (cx: number, cy: number) => ({
  type: 'Feature' as const,
  properties: { code: `${cx},${cy}` },
  geometry: { type: 'Polygon' as const, coordinates: [[[cx-1,cy-1],[cx+1,cy-1],[cx+1,cy+1],[cx-1,cy+1],[cx-1,cy-1]]] },
})

describe('era5-zones', () => {
  it('point-in-polygon basic', () => {
    expect(pointInPolygonGeometry(0, 0, square(0,0).geometry)).toBe(true)
    expect(pointInPolygonGeometry(5, 5, square(0,0).geometry)).toBe(false)
  })

  it('averages cell values per zone, skips nulls, omits empty zones', () => {
    const zones = [square(0,0), square(10,10)]
    const points = [
      { latitude: 0, longitude: 0, temperature_2m: 10 },
      { latitude: 0.5, longitude: 0.2, temperature_2m: 20 },
      { latitude: 0.1, longitude: -0.1, temperature_2m: null }, // skipped
      // none in the (10,10) zone
    ]
    const agg = aggregateEra5ByZone(points as any, 'temperature_2m', zones as any, 'code')
    expect(agg['0,0']).toBeCloseTo(15)        // (10+20)/2
    expect('10,10' in agg).toBe(false)        // empty zone omitted
  })

  it('builds a match expression mapping zone id to a colour with transparent fallback', () => {
    const expr = era5ZoneColorExpression('code', { '0,0': 15 }, 'temperature') as any[]
    expect(expr[0]).toBe('match')
    expect(expr[1]).toEqual(['get', 'code'])
    expect(expr).toContain('0,0')
    expect(expr[expr.length - 1]).toBe('rgba(0,0,0,0)') // fallback last
  })
})
```

- [ ] **Step 2: Run test → fails** — `cd frontend && npx vitest run src/lib/era5-zones.test.ts` (cannot resolve module).

- [ ] **Step 3: Implement**

```ts
// frontend/src/lib/era5-zones.ts
import { ERA5_VARIABLES, type Era5Variable } from './era5-colors'

function pointInRing(x: number, y: number, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1]
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside
  }
  return inside
}

export function pointInPolygonGeometry(lon: number, lat: number, geometry: any): boolean {
  if (!geometry) return false
  const c = geometry.coordinates
  if (geometry.type === 'Polygon') {
    const [outer, ...holes] = c as number[][][]
    if (!pointInRing(lon, lat, outer)) return false
    return !holes.some((h) => pointInRing(lon, lat, h))
  }
  if (geometry.type === 'MultiPolygon') {
    return (c as number[][][][]).some((poly) => {
      const [outer, ...holes] = poly
      if (!pointInRing(lon, lat, outer)) return false
      return !holes.some((h) => pointInRing(lon, lat, h))
    })
  }
  return false
}

export function aggregateEra5ByZone(
  points: Array<Record<string, number | null>>,
  valueKey: string,
  zoneFeatures: Array<{ properties: Record<string, unknown>; geometry: any }>,
  idProp: string,
): Record<string, number> {
  const sums: Record<string, { sum: number; n: number }> = {}
  for (const f of zoneFeatures) {
    const id = String(f.properties?.[idProp])
    for (const p of points) {
      const v = p[valueKey]
      if (v == null) continue
      const lon = Number(p['longitude']), lat = Number(p['latitude'])
      if (pointInPolygonGeometry(lon, lat, f.geometry)) {
        const acc = sums[id] ?? (sums[id] = { sum: 0, n: 0 })
        acc.sum += v; acc.n += 1
      }
    }
  }
  const out: Record<string, number> = {}
  for (const [id, { sum, n }] of Object.entries(sums)) if (n > 0) out[id] = sum / n
  return out
}

function interpColor(value: number, stops: Array<[number, string]>): string {
  if (value <= stops[0][0]) return stops[0][1]
  if (value >= stops[stops.length - 1][0]) return stops[stops.length - 1][1]
  for (let i = 0; i < stops.length - 1; i++) {
    const [v0, c0] = stops[i], [v1, c1] = stops[i + 1]
    if (value >= v0 && value <= v1) {
      const t = (value - v0) / (v1 - v0)
      const a = parseInt(c0.slice(1), 16), b = parseInt(c1.slice(1), 16)
      const r = Math.round(((a >> 16) & 255) + t * (((b >> 16) & 255) - ((a >> 16) & 255)))
      const g = Math.round(((a >> 8) & 255) + t * (((b >> 8) & 255) - ((a >> 8) & 255)))
      const bl = Math.round((a & 255) + t * ((b & 255) - (a & 255)))
      return `rgb(${r},${g},${bl})`
    }
  }
  return stops[stops.length - 1][1]
}

export function era5ZoneColorExpression(
  idProp: string,
  zoneValues: Record<string, number>,
  variable: Era5Variable,
): unknown[] {
  const stops = ERA5_VARIABLES[variable].stops
  const expr: unknown[] = ['match', ['get', idProp]]
  for (const [id, value] of Object.entries(zoneValues)) {
    expr.push(id, interpColor(value, stops))
  }
  expr.push('rgba(0,0,0,0)') // fallback for zones with no data
  return expr
}
```

- [ ] **Step 4: Run test → passes** — `cd frontend && npx vitest run src/lib/era5-zones.test.ts` (3 tests).
- [ ] **Step 5: Commit** — `git add frontend/src/lib/era5-zones.ts frontend/src/lib/era5-zones.test.ts && git commit -m "feat(era5): pure helpers for by-zone aggregation and choropleth colour expression"`

---

### Task 2: Map — by-zone choropleth (override active zone fill, hide squares, restore on off)

**Files:**
- Modify: `frontend/src/components/observatory/ObservatoryMap.tsx`

**Interfaces:**
- Consumes: `aggregateEra5ByZone`, `era5ZoneColorExpression` (Task 1); existing props `era5Points`, `era5AnomalyPoints`, `era5Variable`, `era5Active`; the zone visibility props (`showDepts`/`showRegions`/`showHER`/`showSandre` and the secteurs flag).
- New prop: `era5ByZone?: boolean`.
- Internal: a ref stashing parsed zone GeoJSON per layer (the init effect already `fetch()`es `/geo/*.geojson` into sources — stash the parsed FeatureCollection into `zoneGeoRef.current[id]` when fetched); a ref of saved original `fill-color` per overridden zone fill layer.

- [ ] **Step 1: Stash parsed zone GeoJSON.** In the init code that fetches `/geo/regions.geojson`, `/geo/departments.geojson`, `/geo/her.geojson`, `/geo/bassins.geojson`, and the secteurs source, after parsing, store the FeatureCollection: `zoneGeoRef.current['regions'|'depts'|'her'|'bassins'|'secteurs'] = gj`. Add `const zoneGeoRef = useRef<Record<string, any>>({})` near the other refs. (Additive; do not change the source/layer creation.)

- [ ] **Step 2: Add the prop + a by-zone effect.** Add `era5ByZone = false` to `Props` and the destructure. Add a config mapping the active zone → `{ fillId, idProp, geoKey }`:
```ts
const ERA5_ZONE_CFG: Record<string, { fillId: string; idProp: string; geoKey: string }> = {
  depts:    { fillId: 'depts-fill',    idProp: 'code',      geoKey: 'depts' },
  regions:  { fillId: 'regions-fill',  idProp: 'code',      geoKey: 'regions' },
  her:      { fillId: 'her-fill',      idProp: 'code',      geoKey: 'her' },
  bassins:  { fillId: 'bassins-fill',  idProp: 'CdBH',      geoKey: 'bassins' },
  secteurs: { fillId: 'secteurs-fill', idProp: 'sector_id', geoKey: 'secteurs' },
}
```
Determine the active compatible zone from the existing visibility props (`showDepts→'depts'`, `showRegions→'regions'`, `showHER→'her'`, `showSandre→'bassins'`, secteurs flag→'secteurs'`; else null).

Add an effect keyed on `[mapLoaded, era5ByZone, era5Active, era5Variable, era5Points, era5AnomalyPoints, showDepts, showRegions, showHER, showSandre, <secteurs flag>]`:
```ts
useEffect(() => {
  const map = mapRef.current
  if (!map || !mapLoaded) return
  const overriddenRef = era5OverriddenRef // useRef<string | null>(null)

  const restore = () => {
    const prev = overriddenRef.current
    if (prev && map.getLayer(prev) && savedPaintRef.current[prev] !== undefined) {
      map.setPaintProperty(prev, 'fill-color', savedPaintRef.current[prev])
    }
    overriddenRef.current = null
  }

  const activeZone = /* derive from show* props */ null as string | null
  const cfg = activeZone ? ERA5_ZONE_CFG[activeZone] : undefined

  if (!era5Active || !era5ByZone || !cfg || !map.getLayer(cfg.fillId)) {
    restore()
    // show squares again if era5 active and not by-zone
    if (map.getLayer('era5-grid-fill')) {
      map.setLayoutProperty('era5-grid-fill', 'visibility', era5Active && !era5ByZone ? 'visible' : 'none')
    }
    return
  }

  // by-zone ON: hide squares
  if (map.getLayer('era5-grid-fill')) map.setLayoutProperty('era5-grid-fill', 'visibility', 'none')

  const features = (zoneGeoRef.current[cfg.geoKey]?.features) ?? []
  const isAnom = era5Variable === 'anomaly'
  const valueKey = isAnom ? 'anomaly_c' : ERA5_VARIABLES[era5Variable].prop
  const points = (isAnom ? era5AnomalyPoints : era5Points) ?? []
  const zoneValues = aggregateEra5ByZone(points as any, valueKey, features, cfg.idProp)

  // save original paint once per layer before first override
  if (overriddenRef.current !== cfg.fillId) {
    restore() // restore any previously overridden (different) layer first
    if (savedPaintRef.current[cfg.fillId] === undefined) {
      savedPaintRef.current[cfg.fillId] = map.getPaintProperty(cfg.fillId, 'fill-color')
    }
    overriddenRef.current = cfg.fillId
  }
  map.setPaintProperty(cfg.fillId, 'fill-color', era5ZoneColorExpression(cfg.idProp, zoneValues, era5Variable) as any)
  // ensure the zone layer is visible (it is, since it's the active zone layer)
}, [/* deps above */])
```
Add the refs near the others: `const era5OverriddenRef = useRef<string | null>(null)`, `const savedPaintRef = useRef<Record<string, any>>({})`.
Import `aggregateEra5ByZone`, `era5ZoneColorExpression` from `@/lib/era5-zones`, and ensure `ERA5_VARIABLES` is imported (it already is for the squares).

CAUTION: the existing ERA5 squares effect also toggles `era5-grid-fill` visibility based on `era5Active`. To avoid the two effects fighting, gate the squares effect's "make visible" on `!era5ByZone` (i.e. only show squares when not in by-zone mode). Add `era5ByZone` to that effect's deps and condition.

- [ ] **Step 3: Build + tests** — `cd frontend && npm run build` (exit 0) and `npx vitest run src/lib/era5-zones.test.ts` (pass).
- [ ] **Step 4: Commit** — `git add frontend/src/components/observatory/ObservatoryMap.tsx && git commit -m "feat(era5): by-zone choropleth — aggregate cells into active zone layer, hide squares, restore on off"`

---

### Task 3: Drawer toggle + page wiring + i18n

**Files:**
- Modify: `frontend/src/components/observatory/RightDrawer.tsx`, `frontend/src/pages/ObservatoryPage.tsx`, `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

**Interfaces:** RightDrawer gains `era5ByZone: boolean; setEra5ByZone: (v: boolean) => void`. ObservatoryPage gains `era5ByZone` state and passes it to the map + drawer; auto-activates `depts` when `era5ByZone` turns on with no compatible zone active.

- [ ] **Step 1: i18n.** In `fr.json` `observatory.drawer` add `"era5ByZone": "Agréger par zone (calque actif)"` and `"era5ByZoneHint": "Active un calque de zones (départements, régions…)"`. Add English equivalents to `en.json`. Validate JSON.
- [ ] **Step 2: RightDrawer.** Add the two props. Inside the `era5Active` panel, after the variable radios (and window selector), render a checkbox bound to `era5ByZone`/`setEra5ByZone` labelled `t('observatory.drawer.era5ByZone')`. When `era5ByZone` is on but no compatible zone layer is active, show the hint text in muted style.
- [ ] **Step 3: ObservatoryPage.** Add `const [era5ByZone, setEra5ByZone] = useState(false)`. Add an effect: when `era5ByZone` becomes true and the current `activeZoneLayer` is not one of `['depts','regions','her','bassins','secteurs']`, call `setActiveZoneLayer('depts')`. Pass `era5ByZone={era5ByZone}` to `<ObservatoryMap>` and `era5ByZone`/`setEra5ByZone` to `<RightDrawer>`.
- [ ] **Step 4: Verify** — JSON valid (`node -e ...` on both locales), `npm run build` (exit 0), `npx vitest run src/lib/era5-zones.test.ts src/lib/era5-colors.test.ts src/lib/era5-grid.test.ts` (all pass).
- [ ] **Step 5: Commit** — `git add -A frontend && git commit -m "feat(era5): by-zone toggle, page wiring (auto-activate depts), i18n"`

---

### Task 4: End-to-end verification

**Files:** none.
- [ ] **Step 1:** `docker compose up -d --build frontend`.
- [ ] **Step 2 (bundle + serve):** frontend 200; `Agréger par zone` present in the ObservatoryPage chunk (`docker exec junon-frontend grep -rl "Agréger par zone" /usr/share/nginx/html/assets/`).
- [ ] **Step 3 (browser — user or any env with a browser):** ERA5 on → enable "Agréger par zone" → the départements layer recolours by mean temperature; squares disappear; switching variable (anomalie) re-colours divergently; switching the active zone layer (régions/bassins) re-aggregates; turning the toggle off restores the original zone colours AND brings the squares back; stations remain visible and clicking a zone behaves exactly as before (no ERA5-induced change).

---

## Self-Review
- Client-side mean aggregation + choropleth (replaces squares) → Tasks 1,2. ✓
- Follow active layer + auto-activate depts → Tasks 2,3. ✓
- Static-GeoJSON zones only (dept/region/her/bassin/secteurs); WFS excluded → ERA5_ZONE_CFG. ✓
- Restore original zone colours on off; don't touch zone click handlers → Task 2. ✓
- Squares hidden while by-zone; the two effects coordinated via `era5ByZone` gate → Task 2 caution. ✓
- Opt-in default-off; i18n French → Task 3. ✓
- Placeholder scan: helpers fully written; map integration spelled out with exact MapLibre calls (implementer reads the file for current line numbers). Type names (`era5ByZone`, `aggregateEra5ByZone`, `era5ZoneColorExpression`, refs) consistent across tasks.
