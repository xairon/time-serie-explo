# Météo des nappes — faithful clone (V1 map) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `/meteo` as a faithful clone of the MétéEAU Nappes desktop app (light OSM map, BSH choropleth, circular trend badges, rolling 12-month timeline + full-history date picker, address/station search, minimap, À propos modal), powered only by Junon data.

**Architecture:** Keep the data layer (React Query hooks → `/observatory/situation/*` + `/observatory/stations/geojson` + static `/geo/secteurs-bsh.geojson`). Rewrite the presentation layer: `MeteoMap` becomes a thin init/composition component delegating to pure layer modules; the timeline, panels, search, minimap and modal are new components; `MeteoNappesPage` orchestrates. Remove the BRGM source toggle, national banner and critical list.

**Tech Stack:** React 19 + TypeScript, MapLibre GL, Tailwind 4, TanStack Query, vitest + @testing-library/react (jsdom). French UI, hardcoded strings (consistent with current meteo components).

**Spec:** `docs/superpowers/specs/2026-06-10-meteo-clone-fidele-design.md`

**Working directory:** `/home/ringuet/time-serie-explo` (frontend code in `frontend/`). Run frontend commands from `frontend/`.

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Create | `frontend/src/lib/meteo-timeline.ts` | Pure month arithmetic + 12-month window builder + FR formatting |
| Create | `frontend/src/lib/meteo-timeline.test.ts` | Unit tests for the above |
| Create | `frontend/src/lib/meteo-icons.ts` | Canvas icon factory (station circle badge, type glyphs, 4 trend badges) |
| Create | `frontend/src/components/meteo/layers/sectors-layer.ts` | Choropleth + trend-badge MapLibre layers (add/update fns) |
| Create | `frontend/src/components/meteo/layers/stations-layer.ts` | Piezo/hydro marker layers (add/update/visibility fns) |
| Create | `frontend/src/components/meteo/TrendBadge.tsx` | Shared React SVG of the circular trend badge |
| Create | `frontend/src/components/meteo/MeteoTimeline.tsx` | Bottom bar: month chips + date chip |
| Create | `frontend/src/components/meteo/MeteoTimeline.test.tsx` | Component test |
| Create | `frontend/src/components/meteo/MeteoDatePicker.tsx` | Month/year picker popover (full history) |
| Create | `frontend/src/components/meteo/MeteoTypePanel.tsx` | "Type" card (5 rows, 2 active toggles) |
| Create | `frontend/src/components/meteo/MeteoTypePanel.test.tsx` | Component test |
| Create | `frontend/src/components/meteo/MeteoSearchBar.tsx` | BAN geocoding + station search combobox |
| Create | `frontend/src/components/meteo/MeteoMiniMap.tsx` | Collapsible overview map with viewport rectangle |
| Create | `frontend/src/components/meteo/AboutModal.tsx` | « À propos » modal |
| Rewrite | `frontend/src/components/meteo/MeteoMap.tsx` | Map init + effects only (~170 lines) |
| Rewrite | `frontend/src/components/meteo/MeteoLegend.tsx` | « Évolution des niveaux » + « Niveau » cards |
| Rewrite | `frontend/src/pages/MeteoNappesPage.tsx` | Orchestrator, no BRGM source |
| Modify | `frontend/src/components/meteo/SectorPopup.tsx` | Use TrendBadge, drop `ips` metric |
| Modify | `frontend/src/routes.tsx` | Move `/meteo` out of `Layout` (standalone full-screen) |
| Modify | `frontend/src/lib/meteo-colors.ts` + its test | Prune alert/BRGM helpers (after grep) |
| Modify | `frontend/src/hooks/useObservatory.ts`, `frontend/src/lib/situation-api.ts` | Prune BRGM hooks/calls (after grep) |
| Delete | `MeteoNationalBanner.tsx`, `MeteoCriticalList.tsx`, `MeteoLayersPanel.tsx`, `SituationTimelineSlider.tsx` | Superseded |

Backend: **no changes**. `/observatory/meteo/brgm-*` endpoints stay (QA use), simply unused by the front.

---

### Task 1: Timeline helpers (`lib/meteo-timeline.ts`) — TDD

**Files:**
- Create: `frontend/src/lib/meteo-timeline.ts`
- Test: `frontend/src/lib/meteo-timeline.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// frontend/src/lib/meteo-timeline.test.ts
import { describe, it, expect } from 'vitest'
import { addMonths, comparePeriods, buildTimelineWindow, formatPeriodShortFR, formatPeriodLongFR } from './meteo-timeline'

function monthRange(start: string, end: string): string[] {
  const out: string[] = []
  let p = start
  while (comparePeriods(p, end) <= 0) { out.push(p); p = addMonths(p, 1) }
  return out
}

describe('addMonths', () => {
  it('adds within a year', () => expect(addMonths('2026-03', 2)).toBe('2026-05'))
  it('wraps forward across years', () => expect(addMonths('2025-11', 3)).toBe('2026-02'))
  it('wraps backward across years', () => expect(addMonths('2026-01', -1)).toBe('2025-12'))
  it('handles large negative deltas', () => expect(addMonths('2026-06', -18)).toBe('2024-12'))
})

describe('buildTimelineWindow', () => {
  const periods = monthRange('2010-01', '2026-06')

  it('recent selection: 12 data months ending at latest + 3 future slots', () => {
    const cells = buildTimelineWindow(periods, '2026-06')
    expect(cells).toHaveLength(15)
    expect(cells[0].period).toBe('2025-07')
    expect(cells[11].period).toBe('2026-06')
    expect(cells.slice(0, 12).every(c => c.available && !c.future)).toBe(true)
    expect(cells.slice(12).map(c => c.period)).toEqual(['2026-07', '2026-08', '2026-09'])
    expect(cells.slice(12).every(c => c.future && !c.available)).toBe(true)
  })

  it('old selection is centered in the window', () => {
    const cells = buildTimelineWindow(periods, '2015-06')
    const idx = cells.findIndex(c => c.period === '2015-06')
    expect(idx).toBe(5) // floor((12-1)/2)
    expect(cells.every(c => c.available)).toBe(true) // all in-range, no future
  })

  it('clamps at history start', () => {
    const cells = buildTimelineWindow(periods, '2010-02')
    expect(cells[0].period).toBe('2010-01')
  })

  it('marks January cells with showYear', () => {
    const cells = buildTimelineWindow(periods, '2026-06')
    const jan = cells.find(c => c.period === '2026-01')
    expect(jan?.showYear).toBe(true)
    expect(cells.find(c => c.period === '2026-02')?.showYear).toBe(false)
  })

  it('short history starts at first period', () => {
    const cells = buildTimelineWindow(monthRange('2026-01', '2026-06'), '2026-06')
    expect(cells[0].period).toBe('2026-01')
    expect(cells.filter(c => c.available)).toHaveLength(6)
  })

  it('empty history yields no cells', () => {
    expect(buildTimelineWindow([], '2026-06')).toEqual([])
  })
})

describe('formatting', () => {
  it('short FR', () => expect(formatPeriodShortFR('2026-02')).toBe('févr.'))
  it('long FR', () => expect(formatPeriodLongFR('2026-06')).toBe('juin 2026'))
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/lib/meteo-timeline.test.ts`
Expected: FAIL — `Cannot find module './meteo-timeline'`

- [ ] **Step 3: Implement**

```typescript
// frontend/src/lib/meteo-timeline.ts
// Pure helpers for the MétéEau-style rolling monthly timeline.
// Periods are 'YYYY-MM' strings (zero-padded, so string compare == chronological).

export interface TimelineCell {
  period: string      // 'YYYY-MM'
  available: boolean  // a data point exists for this month
  future: boolean     // after the latest available month (greyed forecast slot)
  showYear: boolean   // render the year under the label (January cells)
}

export const FR_MONTHS_SHORT = [
  'janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
  'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.',
] as const

export const FR_MONTHS_LONG = [
  'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
  'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
] as const

export function comparePeriods(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

export function addMonths(period: string, delta: number): string {
  const [y, m] = period.split('-').map(Number)
  const total = y * 12 + (m - 1) + delta
  const ny = Math.floor(total / 12)
  const nm = total - ny * 12
  return `${ny}-${String(nm + 1).padStart(2, '0')}`
}

function monthIndex(period: string): number {
  return parseInt(period.split('-')[1], 10) - 1
}

export function formatPeriodShortFR(period: string): string {
  const i = monthIndex(period)
  return i >= 0 && i < 12 ? FR_MONTHS_SHORT[i] : period
}

export function formatPeriodLongFR(period: string): string {
  const i = monthIndex(period)
  return i >= 0 && i < 12 ? `${FR_MONTHS_LONG[i]} ${period.split('-')[0]}` : period
}

/**
 * Build the rolling window of timeline cells.
 * - Default: `size` months ending at the latest data month, plus `futureSlots`
 *   greyed calendar months after it (the original's forecast slots).
 * - If `selected` falls before that window, the window is re-centered on it.
 * - Never starts before the first data month.
 */
export function buildTimelineWindow(
  allPeriods: string[],
  selected: string,
  size = 12,
  futureSlots = 3,
): TimelineCell[] {
  if (allPeriods.length === 0) return []
  const first = allPeriods[0]
  const latest = allPeriods[allPeriods.length - 1]
  const available = new Set(allPeriods)

  let start = addMonths(latest, -(size - 1))
  if (comparePeriods(selected, start) < 0) {
    start = addMonths(selected, -Math.floor((size - 1) / 2))
  }
  if (comparePeriods(start, first) < 0) start = first

  const cells: TimelineCell[] = []
  for (let i = 0; i < size + futureSlots; i++) {
    const p = addMonths(start, i)
    cells.push({
      period: p,
      available: available.has(p),
      future: comparePeriods(p, latest) > 0,
      showYear: p.endsWith('-01'),
    })
  }
  return cells
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/lib/meteo-timeline.test.ts`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/meteo-timeline.ts frontend/src/lib/meteo-timeline.test.ts
git commit -m "feat(meteo): timeline helpers — rolling 12-month window + FR formatting"
```

---

### Task 2: Map icon factory (`lib/meteo-icons.ts`)

Canvas drawing (not unit-testable in jsdom without node-canvas; verified by typecheck + visual recette in Task 12).

**Files:**
- Create: `frontend/src/lib/meteo-icons.ts`

- [ ] **Step 1: Create the module**

```typescript
// frontend/src/lib/meteo-icons.ts
// Canvas-drawn MapLibre icons for the /meteo clone.
// Trend badges replicate the original's .sector-icon: 18px circle,
// rgba(255,255,255,0.6) background, black glyph (~8px) — drawn at 3x for crispness.

export type TrendKind = 'hausse' | 'stable' | 'baisse' | 'inconnu'

type DrawFn = (ctx: CanvasRenderingContext2D, size: number) => void

function render(draw: DrawFn, size: number): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  draw(ctx, size)
  return ctx.getImageData(0, 0, size, size)
}

/** SDF icon — recolored at runtime via icon-color. */
export function createSdfIcon(draw: DrawFn, size = 44): ImageData {
  return render(draw, size)
}

/** RGBA icon — keeps its own colors. */
export function createRgbaIcon(draw: DrawFn, size = 44): ImageData {
  return render(draw, size)
}

/** Station badge: plain circle (tinted by classification color via icon-color). */
export function drawStationBadge(ctx: CanvasRenderingContext2D, size: number) {
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size * 0.34, 0, Math.PI * 2)
  ctx.fillStyle = '#fff'
  ctx.fill()
}

/** White type glyph — piezo: borehole stem + downward triangle. */
export function drawPiezoGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  ctx.fillStyle = '#fff'
  ctx.fillRect(cx - size * 0.045, size * 0.30, size * 0.09, size * 0.13)
  const w = size * 0.16, top = size * 0.44, bot = size * 0.66
  ctx.beginPath()
  ctx.moveTo(cx - w, top)
  ctx.lineTo(cx + w, top)
  ctx.lineTo(cx, bot)
  ctx.closePath()
  ctx.fill()
}

/** White type glyph — hydro: water drop. */
export function drawHydroGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  const r = size * 0.15
  ctx.beginPath()
  ctx.arc(cx, size * 0.58, r, 0.12 * Math.PI, 0.88 * Math.PI)
  ctx.lineTo(cx, size * 0.34)
  ctx.closePath()
  ctx.fillStyle = '#fff'
  ctx.fill()
}

/**
 * Trend badge factory: white 60% circle + black glyph.
 * hausse = arrow up, baisse = arrow down, stable = equals, inconnu = '?'.
 * Draw at size 54 → rendered at icon-size 1/3 = 18px.
 */
export function drawTrendBadge(kind: TrendKind): DrawFn {
  return (ctx, size) => {
    const c = size / 2
    ctx.beginPath()
    ctx.arc(c, c, size * 0.46, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.6)'
    ctx.fill()

    ctx.strokeStyle = '#000'
    ctx.fillStyle = '#000'
    ctx.lineWidth = size * 0.07
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    if (kind === 'stable') {
      // equals sign
      const w = size * 0.18
      for (const dy of [-size * 0.08, size * 0.08]) {
        ctx.beginPath()
        ctx.moveTo(c - w, c + dy)
        ctx.lineTo(c + w, c + dy)
        ctx.stroke()
      }
      return
    }
    if (kind === 'inconnu') {
      ctx.font = `bold ${size * 0.46}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('?', c, c + size * 0.02)
      return
    }
    // arrow up / down: vertical shaft + chevron head
    const up = kind === 'hausse'
    const tipY = up ? c - size * 0.20 : c + size * 0.20
    const tailY = up ? c + size * 0.20 : c - size * 0.20
    const head = size * 0.12
    ctx.beginPath()
    ctx.moveTo(c, tailY)
    ctx.lineTo(c, tipY)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(c - head, up ? tipY + head : tipY - head)
    ctx.lineTo(c, tipY)
    ctx.lineTo(c + head, up ? tipY + head : tipY - head)
    ctx.stroke()
  }
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -20`
Expected: no new errors (pre-existing state compiles).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/meteo-icons.ts
git commit -m "feat(meteo): canvas icon factory — circular trend badges (clone style) + station glyphs"
```

---

### Task 3: Sector layers module

**Files:**
- Create: `frontend/src/components/meteo/layers/sectors-layer.ts`

- [ ] **Step 1: Create the module**

```typescript
// frontend/src/components/meteo/layers/sectors-layer.ts
// BSH sector choropleth + circular trend badges. Pure MapLibre, no React.
import maplibregl from 'maplibre-gl'
import { SECTOR_INSUFFICIENT_COLOR, parseTendancyCoord } from '@/lib/sector-arrows'
import { createRgbaIcon, drawTrendBadge, type TrendKind } from '@/lib/meteo-icons'

export type Trend = 'hausse' | 'stable' | 'baisse' | null

const BADGE_IMAGE: Record<TrendKind, string> = {
  hausse: 'trend-hausse',
  stable: 'trend-stable',
  baisse: 'trend-baisse',
  inconnu: 'trend-inconnu',
}

export function addSectorLayers(
  map: maplibregl.Map,
  geojson: GeoJSON.FeatureCollection,
  onSectorClick: (sectorId: number, name: string) => void,
): void {
  if (map.getSource('secteurs-bsh')) return

  for (const kind of Object.keys(BADGE_IMAGE) as TrendKind[]) {
    if (!map.hasImage(BADGE_IMAGE[kind])) {
      map.addImage(BADGE_IMAGE[kind], createRgbaIcon(drawTrendBadge(kind), 54))
    }
  }

  map.addSource('secteurs-bsh', { type: 'geojson', data: geojson, attribution: 'Secteurs © BRGM / Eaufrance' })
  map.addLayer({
    id: 'secteurs-fill', type: 'fill', source: 'secteurs-bsh',
    paint: { 'fill-color': SECTOR_INSUFFICIENT_COLOR, 'fill-opacity': 0.6 },
  })
  map.addLayer({
    id: 'secteurs-line', type: 'line', source: 'secteurs-bsh',
    paint: { 'line-color': '#475569', 'line-width': 0.8, 'line-opacity': 0.5 },
  })

  map.addSource('secteurs-trends', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id: 'secteurs-trends', type: 'symbol', source: 'secteurs-trends',
    layout: {
      'icon-image': ['get', 'icon'],
      'icon-size': 1 / 3,
      'icon-allow-overlap': true,
      'icon-ignore-placement': true,
    },
  })

  map.on('click', 'secteurs-fill', (e) => {
    const f = e.features?.[0]
    if (!f) return
    onSectorClick(f.properties?.sector_id as number, (f.properties?.nom as string) ?? '')
  })
  map.on('mouseenter', 'secteurs-fill', () => { map.getCanvas().style.cursor = 'pointer' })
  map.on('mouseleave', 'secteurs-fill', () => { map.getCanvas().style.cursor = '' })
}

/** Recolor the choropleth from an explicit sector_id → hex map. */
export function updateSectorColors(map: maplibregl.Map, colorById: Record<number, string>): void {
  if (!map.getLayer('secteurs-fill')) return
  const pairs: (number | string)[] = []
  for (const [sid, hex] of Object.entries(colorById)) pairs.push(Number(sid), hex)
  map.setPaintProperty(
    'secteurs-fill', 'fill-color',
    pairs.length
      ? (['match', ['get', 'sector_id'], ...pairs, SECTOR_INSUFFICIENT_COLOR] as unknown as maplibregl.ExpressionSpecification)
      : SECTOR_INSUFFICIENT_COLOR,
  )
}

/** Rebuild trend badge points from sector geometry + a sector_id → trend map. */
export function updateTrendBadges(
  map: maplibregl.Map,
  sectorFeatures: GeoJSON.Feature[],
  trendById: Record<number, Trend>,
): void {
  const src = map.getSource('secteurs-trends') as maplibregl.GeoJSONSource | undefined
  if (!src) return
  const features = sectorFeatures
    .map((f) => {
      const sid = f.properties?.sector_id as number | undefined
      const coords = parseTendancyCoord(f.properties?.tendancy_coord as string | null | undefined)
      if (sid == null || !coords) return null
      const trend = trendById[sid]
      const kind: TrendKind = trend ?? 'inconnu'
      return {
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: coords },
        properties: { icon: BADGE_IMAGE[kind] },
      }
    })
    .filter(Boolean) as GeoJSON.Feature[]
  src.setData({ type: 'FeatureCollection', features })
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -20`
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/meteo/layers/sectors-layer.ts
git commit -m "feat(meteo): sectors layer module — choropleth + circular trend badges"
```

---

### Task 4: Stations layers module

**Files:**
- Create: `frontend/src/components/meteo/layers/stations-layer.ts`

- [ ] **Step 1: Create the module**

```typescript
// frontend/src/components/meteo/layers/stations-layer.ts
// Piezo/hydro marker layers (unclustered, minzoom-gated). Pure MapLibre, no React.
import maplibregl from 'maplibre-gl'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'
import { METEO_CLASS_COLORS } from '@/lib/meteo-colors'
import { createSdfIcon, createRgbaIcon, drawStationBadge, drawPiezoGlyph, drawHydroGlyph } from '@/lib/meteo-icons'

export type StationType = 'piezo' | 'hydro'

// Markers only appear from this zoom — keeps the national view clean,
// matching the original's behavior ("visibles en zoomant").
const STATIONS_MINZOOM = 7

const MARKER_SIZE: maplibregl.ExpressionSpecification =
  ['interpolate', ['linear'], ['zoom'], 7, 0.5, 12, 0.8]

function classificationColorExpr(): maplibregl.ExpressionSpecification {
  return [
    'match', ['get', 'classification'],
    'EXTREMEMENT_BAS', METEO_CLASS_COLORS.EXTREMEMENT_BAS,
    'TRES_BAS', METEO_CLASS_COLORS.TRES_BAS,
    'BAS', METEO_CLASS_COLORS.BAS,
    'NORMAL', METEO_CLASS_COLORS.NORMAL,
    'HAUT', METEO_CLASS_COLORS.HAUT,
    'TRES_HAUT', METEO_CLASS_COLORS.TRES_HAUT,
    'EXTREMEMENT_HAUT', METEO_CLASS_COLORS.EXTREMEMENT_HAUT,
    METEO_CLASS_COLORS.UNKNOWN,
  ]
}

function toGeoJSON(features: StationGeoJSONFeature[]) {
  return {
    type: 'FeatureCollection' as const,
    features: features
      .filter(f => f.geometry.coordinates[0] != null && f.geometry.coordinates[1] != null)
      .map(f => ({
        type: 'Feature' as const,
        geometry: f.geometry,
        properties: {
          code: f.properties.code,
          type: f.properties.type,
          classification: f.properties.classification ?? '',
          commune: f.properties.commune ?? '',
          derniere_mesure: f.properties.derniere_mesure ?? '',
        },
      })),
  }
}

export function addStationLayers(
  map: maplibregl.Map,
  onStationClick: (code: string, type: StationType) => void,
): void {
  map.addImage('station-badge', createSdfIcon(drawStationBadge, 44), { sdf: true })
  map.addImage('piezo-glyph', createRgbaIcon(drawPiezoGlyph, 44))
  map.addImage('hydro-glyph', createRgbaIcon(drawHydroGlyph, 44))

  const colorExpr = classificationColorExpr()
  for (const type of ['piezo', 'hydro'] as StationType[]) {
    map.addSource(`${type}-stations`, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
    map.addLayer({
      id: `${type}-badge`, type: 'symbol', source: `${type}-stations`, minzoom: STATIONS_MINZOOM,
      layout: { 'icon-image': 'station-badge', 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true },
      paint: { 'icon-color': colorExpr, 'icon-halo-color': 'rgba(15,23,42,0.45)', 'icon-halo-width': 1.2, 'icon-halo-blur': 1.2 },
    })
    map.addLayer({
      id: `${type}-glyph`, type: 'symbol', source: `${type}-stations`, minzoom: STATIONS_MINZOOM,
      layout: { 'icon-image': `${type}-glyph`, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true, 'icon-ignore-placement': true },
    })
    map.on('click', `${type}-badge`, (e) => {
      const code = e.features?.[0]?.properties?.code
      if (code) onStationClick(String(code), type)
    })
    map.on('mouseenter', `${type}-badge`, () => { map.getCanvas().style.cursor = 'pointer' })
    map.on('mouseleave', `${type}-badge`, () => { map.getCanvas().style.cursor = '' })
  }
}

export function setStationData(map: maplibregl.Map, type: StationType, features: StationGeoJSONFeature[]): void {
  ;(map.getSource(`${type}-stations`) as maplibregl.GeoJSONSource | undefined)
    ?.setData(toGeoJSON(features) as never)
}

export function setStationVisibility(map: maplibregl.Map, type: StationType, visible: boolean): void {
  const vis = visible ? 'visible' : 'none'
  for (const id of [`${type}-badge`, `${type}-glyph`]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis)
  }
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -20`
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/meteo/layers/stations-layer.ts
git commit -m "feat(meteo): stations layer module — circle markers, minzoom-gated"
```

---

### Task 5: Rewrite `MeteoMap.tsx`

**Files:**
- Rewrite: `frontend/src/components/meteo/MeteoMap.tsx` (replace entire file)

- [ ] **Step 1: Replace the file content**

```typescript
// frontend/src/components/meteo/MeteoMap.tsx
import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'
import { addSectorLayers, updateSectorColors, updateTrendBadges, type Trend } from './layers/sectors-layer'
import { addStationLayers, setStationData, setStationVisibility, type StationType } from './layers/stations-layer'

export const FRANCE_CENTER: [number, number] = [2.5, 46.5]
export const FRANCE_ZOOM = 5.6

interface MeteoMapProps {
  sectorColorById: Record<number, string>
  sectorTrendById: Record<number, Trend>
  visibleLayers: Record<StationType, boolean>
  piezoFeatures: StationGeoJSONFeature[]
  hydroFeatures: StationGeoJSONFeature[]
  onSectorClick: (sectorId: number, name: string) => void
  onStationClick: (code: string, type: StationType) => void
  onMapReady: (map: maplibregl.Map) => void
}

export function MeteoMap({
  sectorColorById,
  sectorTrendById,
  visibleLayers,
  piezoFeatures,
  hydroFeatures,
  onSectorClick,
  onStationClick,
  onMapReady,
}: MeteoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [sectorsReady, setSectorsReady] = useState(false)

  // Latest callbacks for map event handlers.
  const onSectorClickRef = useRef(onSectorClick); onSectorClickRef.current = onSectorClick
  const onStationClickRef = useRef(onStationClick); onStationClickRef.current = onStationClick
  const onMapReadyRef = useRef(onMapReady); onMapReadyRef.current = onMapReady

  // Sector geometry kept for badge rebuilds.
  const sectorGeoRef = useRef<GeoJSON.FeatureCollection | null>(null)

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: FRANCE_CENTER,
      zoom: FRANCE_ZOOM,
      maxBounds: [[-12, 38], [18, 54]],
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-right')
    map.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-left')
    map.on('error', (e) => { console.error('MapLibre error:', e.error?.message ?? e) })

    map.on('load', () => {
      addStationLayers(map, (code, type) => onStationClickRef.current?.(code, type))

      fetch('/geo/secteurs-bsh.geojson')
        .then(r => r.json())
        .then((gj: GeoJSON.FeatureCollection) => {
          if (!mapRef.current) return
          sectorGeoRef.current = gj
          addSectorLayers(map, gj, (id, name) => onSectorClickRef.current?.(id, name))
          setSectorsReady(true)
        })
        .catch(err => console.error('Failed to load sector geometry:', err))

      setMapLoaded(true)
      onMapReadyRef.current?.(map)
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [])

  // Choropleth colors + trend badges follow the selected month.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded || !sectorsReady) return
    updateSectorColors(m, sectorColorById)
    updateTrendBadges(m, sectorGeoRef.current?.features ?? [], sectorTrendById)
  }, [sectorColorById, sectorTrendById, mapLoaded, sectorsReady])

  // Station data.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationData(m, 'piezo', piezoFeatures ?? [])
  }, [piezoFeatures, mapLoaded])

  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationData(m, 'hydro', hydroFeatures ?? [])
  }, [hydroFeatures, mapLoaded])

  // Layer visibility.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationVisibility(m, 'piezo', visibleLayers.piezo)
    setStationVisibility(m, 'hydro', visibleLayers.hydro)
  }, [visibleLayers, mapLoaded])

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" role="application" aria-label="Carte météo des nappes" />
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -30`
Expected: errors ONLY in `MeteoNappesPage.tsx` (props mismatch — fixed in Task 11). No errors in MeteoMap itself.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/meteo/MeteoMap.tsx
git commit -m "refactor(meteo): MeteoMap as thin init/composition over layer modules"
```

---

### Task 6: `TrendBadge.tsx` (shared SVG) + `MeteoLegend.tsx` restyle

**Files:**
- Create: `frontend/src/components/meteo/TrendBadge.tsx`
- Rewrite: `frontend/src/components/meteo/MeteoLegend.tsx`
- Modify: `frontend/src/components/meteo/SectorPopup.tsx`

- [ ] **Step 1: Create TrendBadge**

```tsx
// frontend/src/components/meteo/TrendBadge.tsx
// DOM twin of the map's trend badge: white 60% circle + black glyph.
export type TrendBadgeKind = 'hausse' | 'stable' | 'baisse' | 'inconnu'

const GLYPHS: Record<TrendBadgeKind, React.ReactNode> = {
  hausse: <path d="M9 12.5 V5.5 M6.2 8.2 L9 5.5 L11.8 8.2" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  baisse: <path d="M9 5.5 V12.5 M6.2 9.8 L9 12.5 L11.8 9.8" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  stable: <path d="M5.8 7.5 H12.2 M5.8 10.5 H12.2" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" />,
  inconnu: <text x="9" y="12.6" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#000">?</text>,
}

export function TrendBadge({ kind, size = 18 }: { kind: TrendBadgeKind; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" aria-hidden="true" style={{ flexShrink: 0 }}>
      <circle cx="9" cy="9" r="8.4" fill="rgba(255,255,255,0.6)" stroke="rgba(0,0,0,0.25)" strokeWidth="0.6" />
      {GLYPHS[kind]}
    </svg>
  )
}
```

- [ ] **Step 2: Rewrite MeteoLegend** (Type section moves to MeteoTypePanel in Task 8)

```tsx
// frontend/src/components/meteo/MeteoLegend.tsx
import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS } from '@/lib/meteo-colors'
import { TrendBadge, type TrendBadgeKind } from './TrendBadge'

const TREND_ROWS: { kind: TrendBadgeKind; label: string }[] = [
  { kind: 'hausse', label: METEO_TREND_LABELS.hausse },
  { kind: 'stable', label: METEO_TREND_LABELS.stable },
  { kind: 'baisse', label: METEO_TREND_LABELS.baisse },
  { kind: 'inconnu', label: 'Inconnu' },
]

// Wet-first vertical scale, like the original's "Level" card.
const NIVEAU_SWATCHES: { hex: string; label: string }[] = [
  { hex: METEO_CLASS_COLORS.EXTREMEMENT_HAUT, label: METEO_CLASS_LABELS.EXTREMEMENT_HAUT },
  { hex: METEO_CLASS_COLORS.TRES_HAUT,        label: METEO_CLASS_LABELS.TRES_HAUT },
  { hex: METEO_CLASS_COLORS.HAUT,             label: METEO_CLASS_LABELS.HAUT },
  { hex: METEO_CLASS_COLORS.NORMAL,           label: METEO_CLASS_LABELS.NORMAL },
  { hex: METEO_CLASS_COLORS.BAS,              label: METEO_CLASS_LABELS.BAS },
  { hex: METEO_CLASS_COLORS.TRES_BAS,         label: METEO_CLASS_LABELS.TRES_BAS },
  { hex: METEO_CLASS_COLORS.EXTREMEMENT_BAS,  label: METEO_CLASS_LABELS.EXTREMEMENT_BAS },
  { hex: METEO_CLASS_COLORS.UNKNOWN,          label: METEO_CLASS_LABELS.UNKNOWN },
]

export function MeteoLegend() {
  return (
    <div className="space-y-2">
      {/* Évolution des niveaux */}
      <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
        <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Évolution des niveaux</h4>
        <div className="space-y-1">
          {TREND_ROWS.map(({ kind, label }) => (
            <div key={kind} className="flex items-center gap-2">
              <span className="inline-flex rounded-full bg-slate-100"><TrendBadge kind={kind} size={15} /></span>
              <span className="text-[11px] text-slate-700">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Niveau */}
      <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
        <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Niveau</h4>
        <div className="space-y-0">
          {NIVEAU_SWATCHES.map(({ hex, label }) => (
            <div key={hex} className="flex items-center gap-2">
              <span className="flex-shrink-0" style={{ width: 12, height: 14, backgroundColor: hex }} aria-hidden="true" />
              <span className="text-[11px] text-slate-700 leading-[14px]">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

Note: MeteoLegend no longer positions itself absolutely — the page composes it (Task 11).

- [ ] **Step 3: SectorPopup — use TrendBadge, drop `ips` metric**

In `frontend/src/components/meteo/SectorPopup.tsx`:
1. Delete the local `TrendArrow` function (lines 20–43) and the `ips` field of `SectorMetrics` (line 7) and the IPS block (lines 91–96).
2. Add import: `import { TrendBadge } from './TrendBadge'`
3. Replace `<TrendArrow trend={trend} />` (line 87) with:
```tsx
<TrendBadge kind={trend ?? 'inconnu'} size={16} />
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -30`
Expected: errors only in `MeteoNappesPage.tsx` (it passes `metrics.ips` — fixed Task 11). If MeteoNappesPage errors block the build, that's expected and tracked.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/meteo/TrendBadge.tsx frontend/src/components/meteo/MeteoLegend.tsx frontend/src/components/meteo/SectorPopup.tsx
git commit -m "feat(meteo): TrendBadge partagé + légende style original (badges circulaires)"
```

---

### Task 7: `MeteoTimeline.tsx` + `MeteoDatePicker.tsx` — TDD on the component

**Files:**
- Create: `frontend/src/components/meteo/MeteoDatePicker.tsx`
- Create: `frontend/src/components/meteo/MeteoTimeline.tsx`
- Test: `frontend/src/components/meteo/MeteoTimeline.test.tsx`

- [ ] **Step 1: Write the failing component test**

```tsx
// frontend/src/components/meteo/MeteoTimeline.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MeteoTimeline } from './MeteoTimeline'
import { addMonths, comparePeriods } from '@/lib/meteo-timeline'

function monthRange(start: string, end: string): string[] {
  const out: string[] = []
  let p = start
  while (comparePeriods(p, end) <= 0) { out.push(p); p = addMonths(p, 1) }
  return out
}

const periods = monthRange('2020-01', '2026-06')

describe('MeteoTimeline', () => {
  it('renders 15 month cells (12 data + 3 future greyed)', () => {
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={() => {}} />)
    const buttons = screen.getAllByRole('button', { name: /^mois / })
    expect(buttons).toHaveLength(15)
    // Future months are disabled
    expect(screen.getByRole('button', { name: 'mois juillet 2026' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'mois août 2026' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'mois septembre 2026' })).toBeDisabled()
  })

  it('clicking an available month fires onChange', () => {
    const onChange = vi.fn()
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'mois mai 2026' }))
    expect(onChange).toHaveBeenCalledWith('2026-05')
  })

  it('shows the selected period in the date chip', () => {
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={() => {}} />)
    expect(screen.getByText('juin 2026')).toBeInTheDocument()
  })

  it('date chip × resets to the latest period', () => {
    const onChange = vi.fn()
    render(<MeteoTimeline periods={periods} selected="2024-03" onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'Revenir au mois le plus récent' }))
    expect(onChange).toHaveBeenCalledWith('2026-06')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/meteo/MeteoTimeline.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement MeteoDatePicker**

```tsx
// frontend/src/components/meteo/MeteoDatePicker.tsx
// Month/year picker over the full data history (our edge over the original).
import { useState } from 'react'
import { FR_MONTHS_SHORT } from '@/lib/meteo-timeline'

interface Props {
  periods: string[]          // all available 'YYYY-MM', ascending
  selected: string
  onSelect: (p: string) => void
  onClose: () => void
}

export function MeteoDatePicker({ periods, selected, onSelect, onClose }: Props) {
  const available = new Set(periods)
  const firstYear = Number(periods[0].split('-')[0])
  const lastYear = Number(periods[periods.length - 1].split('-')[0])
  const [year, setYear] = useState(Number(selected.split('-')[0]))

  return (
    <div className="absolute bottom-12 left-0 z-30 bg-white rounded-lg shadow-lg border border-slate-200 p-3 w-60">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setYear(y => Math.max(firstYear, y - 1))}
          disabled={year <= firstYear}
          aria-label="Année précédente"
          className="px-2 py-0.5 rounded hover:bg-slate-100 disabled:opacity-30 text-slate-600"
        >‹</button>
        <span className="text-sm font-semibold text-slate-800">{year}</span>
        <button
          onClick={() => setYear(y => Math.min(lastYear, y + 1))}
          disabled={year >= lastYear}
          aria-label="Année suivante"
          className="px-2 py-0.5 rounded hover:bg-slate-100 disabled:opacity-30 text-slate-600"
        >›</button>
      </div>
      <div className="grid grid-cols-4 gap-1">
        {FR_MONTHS_SHORT.map((label, i) => {
          const p = `${year}-${String(i + 1).padStart(2, '0')}`
          const ok = available.has(p)
          const isSel = p === selected
          return (
            <button
              key={p}
              disabled={!ok}
              onClick={() => { onSelect(p); onClose() }}
              className={`text-[11px] rounded px-1 py-1.5 ${
                isSel ? 'bg-blue-600 text-white font-semibold'
                : ok ? 'text-slate-700 hover:bg-slate-100'
                : 'text-slate-300 cursor-not-allowed'
              }`}
            >
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Implement MeteoTimeline**

```tsx
// frontend/src/components/meteo/MeteoTimeline.tsx
// Bottom bar clone: rolling 12-month chips + 3 greyed future slots,
// year labels at January, date chip bottom-left opening the full-history picker.
import { useState } from 'react'
import { buildTimelineWindow, formatPeriodLongFR, FR_MONTHS_LONG } from '@/lib/meteo-timeline'
import { MeteoDatePicker } from './MeteoDatePicker'

interface Props {
  periods: string[]              // all available 'YYYY-MM', ascending
  selected: string
  onChange: (p: string) => void
}

function monthLong(period: string): string {
  return FR_MONTHS_LONG[parseInt(period.split('-')[1], 10) - 1] ?? period
}

export function MeteoTimeline({ periods, selected, onChange }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false)
  if (periods.length === 0) return null
  const latest = periods[periods.length - 1]
  const cells = buildTimelineWindow(periods, selected)

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 bg-white/95 border-t border-slate-200 shadow-[0_-2px_8px_rgba(0,0,0,0.06)] h-12 flex items-center">
      {/* Date chip + picker */}
      <div className="relative flex items-center gap-1 pl-3 pr-4 flex-shrink-0">
        {pickerOpen && (
          <MeteoDatePicker
            periods={periods}
            selected={selected}
            onSelect={onChange}
            onClose={() => setPickerOpen(false)}
          />
        )}
        <button
          onClick={() => setPickerOpen(o => !o)}
          aria-label="Choisir une date"
          className="flex items-center gap-1.5 border border-slate-300 rounded px-2.5 py-1 text-xs text-slate-700 hover:border-slate-400 bg-white"
        >
          {formatPeriodLongFR(selected)}
          <svg width="9" height="6" viewBox="0 0 9 6" aria-hidden="true"><path d="M1 1l3.5 3.5L8 1" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" /></svg>
        </button>
        {selected !== latest && (
          <button
            onClick={() => onChange(latest)}
            aria-label="Revenir au mois le plus récent"
            className="text-slate-400 hover:text-slate-600 px-1 text-sm leading-none"
          >×</button>
        )}
      </div>

      {/* Month chips */}
      <div className="flex-1 flex items-center pr-4 min-w-0">
        {cells.map((c) => {
          const isSelected = c.period === selected
          const year = c.period.split('-')[0]
          return (
            <button
              key={c.period}
              disabled={!c.available}
              onClick={() => onChange(c.period)}
              aria-label={`mois ${monthLong(c.period)} ${year}`}
              aria-current={isSelected ? 'date' : undefined}
              className="flex-1 flex flex-col items-center gap-0.5 group min-w-0 disabled:cursor-default"
            >
              <span className={`text-[11px] leading-none truncate max-w-full px-0.5 ${
                isSelected ? 'font-bold text-blue-700'
                : c.available ? 'text-slate-600 group-hover:text-slate-900'
                : 'text-slate-300'
              }`}>
                {monthLong(c.period)}
                {c.showYear && <span className="ml-1 font-semibold text-slate-400">{year}</span>}
              </span>
              <span className={`w-2 h-2 rounded-full border ${
                isSelected ? 'bg-blue-600 border-blue-600 scale-125'
                : c.available ? 'bg-white border-slate-300 group-hover:border-slate-400'
                : 'bg-slate-100 border-slate-200'
              }`} aria-hidden="true" />
            </button>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/meteo/MeteoTimeline.test.tsx`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/meteo/MeteoTimeline.tsx frontend/src/components/meteo/MeteoDatePicker.tsx frontend/src/components/meteo/MeteoTimeline.test.tsx
git commit -m "feat(meteo): timeline clone — 12 mois glissants + sélecteur de date plein historique"
```

---

### Task 8: `MeteoTypePanel.tsx` — TDD

**Files:**
- Create: `frontend/src/components/meteo/MeteoTypePanel.tsx`
- Test: `frontend/src/components/meteo/MeteoTypePanel.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/meteo/MeteoTypePanel.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MeteoTypePanel } from './MeteoTypePanel'

describe('MeteoTypePanel', () => {
  const visible = { piezo: true, hydro: false }

  it('renders the 5 original type rows', () => {
    render(<MeteoTypePanel visible={visible} onToggle={() => {}} />)
    for (const label of ['Piézomètre', 'Source', 'Pluviomètre', 'Station de débit', 'Avec modèle']) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  })

  it('disables the rows without data', () => {
    render(<MeteoTypePanel visible={visible} onToggle={() => {}} />)
    expect(screen.getByRole('checkbox', { name: /Source/ })).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: /Pluviomètre/ })).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: /Avec modèle/ })).toBeDisabled()
  })

  it('toggles active layers', () => {
    const onToggle = vi.fn()
    render(<MeteoTypePanel visible={visible} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole('checkbox', { name: /Station de débit/ }))
    expect(onToggle).toHaveBeenCalledWith('hydro')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/meteo/MeteoTypePanel.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```tsx
// frontend/src/components/meteo/MeteoTypePanel.tsx
// "Type" card clone: the original's 5 station types; rows without Junon data
// are kept (documents the gap) but disabled.
import type { StationType } from './layers/stations-layer'

interface Props {
  visible: Record<StationType, boolean>
  onToggle: (key: StationType) => void
}

const ROWS: { key: StationType | null; label: string }[] = [
  { key: 'piezo', label: 'Piézomètre' },
  { key: null,    label: 'Source' },
  { key: null,    label: 'Pluviomètre' },
  { key: 'hydro', label: 'Station de débit' },
  { key: null,    label: 'Avec modèle' },
]

export function MeteoTypePanel({ visible, onToggle }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
      <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Type</h4>
      <div className="space-y-1">
        {ROWS.map(({ key, label }) => {
          const enabled = key != null
          return (
            <label
              key={label}
              title={enabled ? undefined : 'Données bientôt disponibles'}
              className={`flex items-center gap-2 ${enabled ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}
            >
              <input
                type="checkbox"
                aria-label={label}
                disabled={!enabled}
                checked={enabled ? visible[key] : false}
                onChange={() => { if (enabled) onToggle(key) }}
                className="w-3.5 h-3.5 accent-blue-600 rounded"
              />
              <span className="text-[11px] text-slate-700">{label}</span>
            </label>
          )
        })}
      </div>
      <p className="mt-2 text-[10px] text-slate-400 leading-tight">
        Stations visibles en zoomant sur la carte
      </p>
    </div>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/meteo/MeteoTypePanel.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/meteo/MeteoTypePanel.tsx frontend/src/components/meteo/MeteoTypePanel.test.tsx
git commit -m "feat(meteo): panneau Type — 5 types de l'original, couches sans données grisées"
```

---

### Task 9: `MeteoSearchBar.tsx`

**Files:**
- Create: `frontend/src/components/meteo/MeteoSearchBar.tsx`

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/meteo/MeteoSearchBar.tsx
// Search combobox clone: BAN address geocoding + Junon stations (code/commune).
import { useState, useEffect, useRef } from 'react'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'

export interface SearchTarget {
  lng: number
  lat: number
  zoom: number
  label: string
}

interface Suggestion extends SearchTarget {
  kind: 'adresse' | 'station'
}

interface Props {
  stations: StationGeoJSONFeature[]
  onSelect: (target: SearchTarget) => void
}

const BAN_URL = 'https://api-adresse.data.gouv.fr/search/'

async function searchBan(q: string): Promise<Suggestion[]> {
  try {
    const res = await fetch(`${BAN_URL}?q=${encodeURIComponent(q)}&limit=4`)
    if (!res.ok) return []
    const data = await res.json()
    return (data.features ?? []).map((f: { geometry: { coordinates: [number, number] }; properties: { label: string; context?: string } }) => ({
      kind: 'adresse' as const,
      lng: f.geometry.coordinates[0],
      lat: f.geometry.coordinates[1],
      zoom: 11,
      label: f.properties.context ? `${f.properties.label}, ${f.properties.context}` : f.properties.label,
    }))
  } catch {
    return [] // geocoding down → stations only
  }
}

function searchStations(stations: StationGeoJSONFeature[], q: string): Suggestion[] {
  const needle = q.trim().toLowerCase()
  if (needle.length < 2) return []
  return stations
    .filter(f =>
      f.properties.code.toLowerCase().startsWith(needle) ||
      (f.properties.commune ?? '').toLowerCase().includes(needle))
    .slice(0, 4)
    .map(f => ({
      kind: 'station' as const,
      lng: f.geometry.coordinates[0],
      lat: f.geometry.coordinates[1],
      zoom: 12,
      label: `${f.properties.code}${f.properties.commune ? ` — ${f.properties.commune}` : ''}`,
    }))
}

export function MeteoSearchBar({ stations, onSelect }: Props) {
  const [q, setQ] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [open, setOpen] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    clearTimeout(debounceRef.current)
    if (q.trim().length < 3) { setSuggestions([]); return }
    debounceRef.current = setTimeout(async () => {
      const [ban, sta] = [await searchBan(q), searchStations(stations, q)]
      setSuggestions([...sta, ...ban])
      setOpen(true)
    }, 300)
    return () => clearTimeout(debounceRef.current)
  }, [q, stations])

  const pick = (s: Suggestion) => {
    onSelect(s)
    setQ('')
    setSuggestions([])
    setOpen(false)
  }

  return (
    <div className="relative w-80">
      <div className="flex items-center gap-2 bg-white rounded-full shadow-md border border-slate-200 px-3.5 py-2">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" className="text-slate-400 flex-shrink-0">
          <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
          <path d="M9.5 9.5 L13 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <input
          role="combobox"
          aria-expanded={open}
          aria-label="Rechercher une adresse ou une station"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && suggestions.length) pick(suggestions[0]) }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onFocus={() => { if (suggestions.length) setOpen(true) }}
          placeholder="adresse, station, piézomètre, etc."
          className="flex-1 text-xs text-slate-700 placeholder-slate-400 bg-transparent outline-none"
        />
      </div>
      {open && suggestions.length > 0 && (
        <ul role="listbox" className="absolute top-full mt-1 left-0 right-0 bg-white rounded-lg shadow-lg border border-slate-200 py-1 max-h-64 overflow-y-auto">
          {suggestions.map((s, i) => (
            <li key={`${s.kind}-${i}`} role="option" aria-selected="false">
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(s)}
                className="w-full text-left px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50 flex items-center gap-2"
              >
                <span className={`text-[9px] uppercase font-semibold flex-shrink-0 ${s.kind === 'station' ? 'text-blue-600' : 'text-slate-400'}`}>
                  {s.kind === 'station' ? 'Station' : 'Adresse'}
                </span>
                <span className="truncate">{s.label}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -20`
Expected: no new errors outside MeteoNappesPage.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/meteo/MeteoSearchBar.tsx
git commit -m "feat(meteo): recherche adresse (BAN) + stations, style clone"
```

---

### Task 10: `MeteoMiniMap.tsx` + `AboutModal.tsx`

**Files:**
- Create: `frontend/src/components/meteo/MeteoMiniMap.tsx`
- Create: `frontend/src/components/meteo/AboutModal.tsx`

- [ ] **Step 1: Implement MeteoMiniMap**

```tsx
// frontend/src/components/meteo/MeteoMiniMap.tsx
// Collapsible France overview with the main map's viewport rectangle.
import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'

interface Props {
  mainMap: maplibregl.Map | null
}

function boundsToPolygon(b: maplibregl.LngLatBounds): GeoJSON.Feature {
  const [w, s, e, n] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]
  return {
    type: 'Feature',
    geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
    properties: {},
  }
}

export function MeteoMiniMap({ mainMap }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const miniRef = useRef<maplibregl.Map | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    if (!containerRef.current || miniRef.current || collapsed) return
    const mini = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256 },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [2.5, 46.6],
      zoom: 3.2,
      interactive: false,
      attributionControl: false,
    })
    mini.on('load', () => {
      mini.addSource('viewport', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      mini.addLayer({ id: 'viewport-fill', type: 'fill', source: 'viewport', paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.15 } })
      mini.addLayer({ id: 'viewport-line', type: 'line', source: 'viewport', paint: { 'line-color': '#3b82f6', 'line-width': 1.5 } })
      if (mainMap) {
        ;(mini.getSource('viewport') as maplibregl.GeoJSONSource)
          .setData({ type: 'FeatureCollection', features: [boundsToPolygon(mainMap.getBounds())] })
      }
    })
    miniRef.current = mini
    return () => { mini.remove(); miniRef.current = null }
  }, [collapsed, mainMap])

  useEffect(() => {
    if (!mainMap) return
    const sync = () => {
      const src = miniRef.current?.getSource('viewport') as maplibregl.GeoJSONSource | undefined
      src?.setData({ type: 'FeatureCollection', features: [boundsToPolygon(mainMap.getBounds())] })
    }
    mainMap.on('move', sync)
    return () => { mainMap.off('move', sync) }
  }, [mainMap, collapsed])

  return (
    <div className="bg-white rounded-lg shadow-md border border-slate-200 overflow-hidden">
      <button
        onClick={() => setCollapsed(c => !c)}
        aria-label={collapsed ? 'Afficher la mini-carte' : 'Masquer la mini-carte'}
        className="w-full flex items-center justify-center py-0.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50"
      >
        <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true" style={{ transform: collapsed ? 'rotate(180deg)' : undefined }}>
          <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
        </svg>
      </button>
      {!collapsed && <div ref={containerRef} style={{ width: 150, height: 112 }} />}
    </div>
  )
}
```

- [ ] **Step 2: Implement AboutModal**

```tsx
// frontend/src/components/meteo/AboutModal.tsx
interface Props {
  onClose: () => void
}

export function AboutModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose} role="dialog" aria-modal="true" aria-label="À propos">
      <div className="bg-white rounded-xl shadow-xl max-w-lg w-[min(92vw,520px)] p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-3">
          <h2 className="text-lg font-bold text-slate-800">À propos</h2>
          <button onClick={onClose} aria-label="Fermer" className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600">
            <svg width="16" height="16" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
          <p>
            Cette carte présente la <strong>situation des nappes phréatiques</strong> par
            secteur hydrogéologique, dans l'esprit du bulletin MétéEAU Nappes du BRGM,
            calculée à partir des données de la plateforme Junon.
          </p>
          <p>
            Le niveau de chaque secteur est déterminé par l'<strong>Indicateur Piézométrique
            Standardisé (IPS)</strong> des stations qui le composent, calculé sur une période
            de référence fixe <strong>1991-2020</strong>. Les flèches indiquent l'évolution
            des niveaux par rapport au mois précédent.
          </p>
          <p>
            Les secteurs affichés sont les secteurs hydrogéologiques du Bulletin de
            Situation Hydrologique (BSH), © BRGM / Eaufrance. Les mesures piézométriques
            proviennent du réseau ADES (Hub'Eau).
          </p>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit 2>&1 | head -20`
Expected: no new errors outside MeteoNappesPage.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/meteo/MeteoMiniMap.tsx frontend/src/components/meteo/AboutModal.tsx
git commit -m "feat(meteo): mini-carte repliable + modal À propos"
```

---

### Task 11: Rewrite `MeteoNappesPage.tsx` + route out of Layout

**Files:**
- Rewrite: `frontend/src/pages/MeteoNappesPage.tsx`
- Modify: `frontend/src/routes.tsx`

- [ ] **Step 1: Replace MeteoNappesPage**

```tsx
// frontend/src/pages/MeteoNappesPage.tsx
// Clone fidèle MétéEAU Nappes — données Junon (IPS réf. fixe 1991-2020).
import { useState, useMemo, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import type maplibregl from 'maplibre-gl'
import { MeteoMap, FRANCE_CENTER, FRANCE_ZOOM } from '@/components/meteo/MeteoMap'
import { MeteoTypePanel } from '@/components/meteo/MeteoTypePanel'
import { MeteoLegend } from '@/components/meteo/MeteoLegend'
import { MeteoTimeline } from '@/components/meteo/MeteoTimeline'
import { MeteoSearchBar, type SearchTarget } from '@/components/meteo/MeteoSearchBar'
import { MeteoMiniMap } from '@/components/meteo/MeteoMiniMap'
import { AboutModal } from '@/components/meteo/AboutModal'
import { SectorPopup } from '@/components/meteo/SectorPopup'
import { StationPopup } from '@/components/meteo/StationPopup'
import type { StationType } from '@/components/meteo/layers/stations-layer'
import type { Trend } from '@/components/meteo/layers/sectors-layer'
import { useSectorSituation, useSectorTimeline, useStationsGeoJSON } from '@/hooks/useObservatory'
import { meteoClassColor, METEO_CLASS_LABELS } from '@/lib/meteo-colors'
import { SECTOR_INSUFFICIENT_COLOR } from '@/lib/sector-arrows'
import type { SectorSituation, SituationClass, StationGeoJSONFeature } from '@/lib/observatory-types'

// Timeline payload class indices → enums (index 7 / null = insufficient).
const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }

function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

export default function MeteoNappesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [visible, setVisible] = useState<Record<StationType, boolean>>({ piezo: true, hydro: true })
  const [selectedSectorId, setSelectedSectorId] = useState<number | null>(null)
  const [selectedSectorName, setSelectedSectorName] = useState<string | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: StationType } | null>(null)
  const [searchLabel, setSearchLabel] = useState<string | null>(null)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [map, setMap] = useState<maplibregl.Map | null>(null)

  const { data: sectorSituationData } = useSectorSituation('piezo', true)
  const { data: timeline } = useSectorTimeline('piezo', true)
  const { data: geojsonData } = useStationsGeoJSON()

  // Sector names come from the static geometry.
  const { data: sectorGeo } = useQuery({
    queryKey: ['secteurs-bsh-geo'],
    queryFn: () => fetch('/geo/secteurs-bsh.geojson').then((r) => r.json()),
    staleTime: Infinity,
  })
  const nameById: Record<number, string> = useMemo(
    () => Object.fromEntries((sectorGeo?.features ?? []).map((f: { properties: { sector_id: number; nom: string } }) => [f.properties.sector_id, f.properties.nom])),
    [sectorGeo],
  )

  const piezoFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'piezo'),
    [geojsonData],
  )
  const hydroFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'hydro'),
    [geojsonData],
  )

  // Selected month: ?month= param, else latest available.
  const periods = timeline?.periods ?? []
  const latest = periods.length ? periods[periods.length - 1] : null
  const urlMonth = searchParams.get('month')
  const effectivePeriod = (urlMonth && periods.includes(urlMonth)) ? urlMonth : latest

  const setPeriod = useCallback((p: string) => {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      if (latest && p === latest) next.delete('month')
      else next.set('month', p)
      return next
    }, { replace: true })
  }, [setSearchParams, latest])

  // Recolor for the selected month from the timeline payload (no refetch).
  const displaySectorSituation = useMemo<SectorSituation[]>(() => {
    const base = sectorSituationData ?? []
    if (!timeline || effectivePeriod == null || effectivePeriod === latest) return base
    const sIdx = timeline.periods.indexOf(effectivePeriod)
    if (sIdx < 0) return base
    return base.map((s) => {
      const ci = timeline.sectors[s.code]?.[sIdx]
      const ti = timeline.trends[s.code]?.[sIdx]
      const insufficient = ci == null || ci === 7
      return {
        ...s,
        situation_class: insufficient ? null : (CLS[ci] as SituationClass),
        trend: ti != null ? TR[ti] : null,
        insufficient,
      }
    })
  }, [sectorSituationData, timeline, effectivePeriod, latest])

  const { sectorColorById, sectorTrendById } = useMemo(() => {
    const colorById: Record<number, string> = {}
    const trendById: Record<number, Trend> = {}
    for (const s of displaySectorSituation) {
      const sid = Number(s.code)
      colorById[sid] = s.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(s.situation_class)
      trendById[sid] = s.trend
    }
    return { sectorColorById: colorById, sectorTrendById: trendById }
  }, [displaySectorSituation])

  const onSectorClick = useCallback((id: number, name: string) => {
    setSelectedSectorId(id)
    setSelectedSectorName(name)
    setSelectedStation(null)
  }, [])

  const onStationClick = useCallback((code: string, type: StationType) => {
    setSelectedStation({ code, type })
    setSelectedSectorId(null)
  }, [])

  const onToggle = useCallback((k: StationType) => {
    setVisible((v) => ({ ...v, [k]: !v[k] }))
  }, [])

  const onSearchSelect = useCallback((t: SearchTarget) => {
    setSearchLabel(t.label)
    map?.flyTo({ center: [t.lng, t.lat], zoom: t.zoom, duration: 1200 })
  }, [map])

  const onSearchReset = useCallback(() => {
    setSearchLabel(null)
    map?.flyTo({ center: FRANCE_CENTER, zoom: FRANCE_ZOOM, duration: 1000 })
  }, [map])

  const selectedIps = useMemo<SectorSituation | null>(() => {
    if (selectedSectorId == null) return null
    return displaySectorSituation.find((s) => Number(s.code) === selectedSectorId) ?? null
  }, [displaySectorSituation, selectedSectorId])

  const selectedStationFeature = useMemo<StationGeoJSONFeature | null>(() => {
    if (!selectedStation) return null
    const pool = selectedStation.type === 'piezo' ? piezoFeatures : hydroFeatures
    return pool.find((f) => f.properties.code === selectedStation.code) ?? null
  }, [selectedStation, piezoFeatures, hydroFeatures])

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-slate-100">
      <MeteoMap
        sectorColorById={sectorColorById}
        sectorTrendById={sectorTrendById}
        visibleLayers={visible}
        piezoFeatures={piezoFeatures}
        hydroFeatures={hydroFeatures}
        onSectorClick={onSectorClick}
        onStationClick={onStationClick}
        onMapReady={setMap}
      />

      {/* Search — top-left */}
      <div className="absolute top-3 left-3 z-20">
        <MeteoSearchBar stations={piezoFeatures.concat(hydroFeatures)} onSelect={onSearchSelect} />
      </div>

      {/* Reset chip — top-center, after a search */}
      {searchLabel && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
          <button
            onClick={onSearchReset}
            className="flex items-center gap-1.5 bg-slate-800 text-white text-xs rounded px-2.5 py-1.5 shadow-md hover:bg-slate-700"
          >
            réinitialiser <span aria-hidden="true">×</span>
          </button>
        </div>
      )}

      {/* À propos — top-right */}
      <div className="absolute top-3 right-3 z-20">
        <button
          onClick={() => setAboutOpen(true)}
          className="bg-white text-xs text-slate-600 rounded-full shadow-md border border-slate-200 px-3.5 py-2 hover:bg-slate-50"
        >
          À propos
        </button>
      </div>

      {/* Left panels: Type + legend */}
      <div className="absolute left-3 top-1/2 -translate-y-1/2 z-10 space-y-2">
        <MeteoTypePanel visible={visible} onToggle={onToggle} />
        <MeteoLegend />
      </div>

      {/* Junon logo → back to the app (where the original puts its own logo) */}
      <Link
        to="/"
        className="absolute bottom-14 left-1/2 -translate-x-1/2 z-10 bg-white/90 rounded-full shadow border border-slate-200 px-3 py-1 text-xs font-bold text-slate-700 hover:bg-white"
      >
        Junon
      </Link>

      {/* Minimap — bottom-right above the timeline */}
      <div className="absolute bottom-14 right-3 z-10">
        <MeteoMiniMap mainMap={map} />
      </div>

      {/* Timeline — bottom */}
      {periods.length > 0 && effectivePeriod && (
        <MeteoTimeline periods={periods} selected={effectivePeriod} onChange={setPeriod} />
      )}

      {/* Popups (one at a time) */}
      {selectedIps && (
        <div className="absolute top-16 right-4 z-20">
          <SectorPopup
            name={selectedSectorName ?? nameById[Number(selectedIps.code)] ?? selectedIps.name}
            code={selectedIps.code}
            classLabel={capitalize(METEO_CLASS_LABELS[selectedIps.situation_class ?? 'UNKNOWN'] ?? METEO_CLASS_LABELS.UNKNOWN)}
            trend={selectedIps.trend}
            colorHex={selectedIps.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(selectedIps.situation_class)}
            metrics={{
              pctBelowNormal: selectedIps.pct_below_normal,
              nEligible: selectedIps.n_eligible,
              nProvisoire: selectedIps.n_provisoire,
            }}
            onClose={() => setSelectedSectorId(null)}
          />
        </div>
      )}

      {selectedStationFeature && (
        <div className="absolute top-16 right-4 z-20">
          <StationPopup
            code={selectedStationFeature.properties.code}
            commune={selectedStationFeature.properties.commune ?? undefined}
            classification={selectedStationFeature.properties.classification}
            derniereMesure={selectedStationFeature.properties.derniere_mesure}
            onClose={() => setSelectedStation(null)}
          />
        </div>
      )}

      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
    </div>
  )
}
```

- [ ] **Step 2: Move the route out of Layout**

In `frontend/src/routes.tsx`, remove the line inside the Layout children:
```tsx
      // Météo des nappes (public)
      { path: '/meteo', element: <SW><MeteoNappesPage /></SW> },
```
and add a top-level sibling route (same array as the Layout object, before it):
```tsx
  // Météo des nappes — standalone full-screen clone (no TopNav)
  {
    path: '/meteo',
    element: <SessionGate><SW><MeteoNappesPage /></SW></SessionGate>,
  },
```
The final structure of the exported router:
```tsx
export const router = createBrowserRouter([
  {
    path: '/meteo',
    element: <SessionGate><SW><MeteoNappesPage /></SW></SessionGate>,
  },
  {
    element: <SessionGate><Layout /></SessionGate>,
    children: [
      // ... unchanged, minus the /meteo line
    ],
  },
])
```
Note: `SW` is declared below the router in the current file — move the `SW` function declaration ABOVE `export const router` if it isn't already (it is defined at line 36, before the router — no change needed).

- [ ] **Step 3: Typecheck + full test run**

Run: `cd frontend && npx tsc -b --noEmit && npx vitest run`
Expected: typecheck clean EXCEPT possibly unused-export errors in files pruned in Task 12; all tests pass. If `MeteoNationalBanner`/`MeteoCriticalList`/`MeteoLayersPanel`/`SituationTimelineSlider` now have no importers, that is fine (deleted next task).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/MeteoNappesPage.tsx frontend/src/routes.tsx
git commit -m "feat(meteo): page refondue — clone plein écran hors Layout, données Junon uniquement"
```

---

### Task 12: Cleanup — delete dead components, prune BRGM front code

**Files:**
- Delete: `frontend/src/components/meteo/MeteoNationalBanner.tsx`, `MeteoCriticalList.tsx`, `MeteoLayersPanel.tsx`, `SituationTimelineSlider.tsx`
- Modify: `frontend/src/hooks/useObservatory.ts`, `frontend/src/lib/situation-api.ts`, `frontend/src/lib/meteo-colors.ts`, `frontend/src/lib/meteo-colors.test.ts`

- [ ] **Step 1: Verify the components are unreferenced, then delete**

Run: `cd frontend && grep -rn "MeteoNationalBanner\|MeteoCriticalList\|MeteoLayersPanel\|SituationTimelineSlider" src/ --include="*.tsx" --include="*.ts" | grep -v "components/meteo/Meteo\|components/meteo/Situation"`
Expected: no output (no external importers). Then:
```bash
git rm frontend/src/components/meteo/MeteoNationalBanner.tsx frontend/src/components/meteo/MeteoCriticalList.tsx frontend/src/components/meteo/MeteoLayersPanel.tsx frontend/src/components/meteo/SituationTimelineSlider.tsx
```

- [ ] **Step 2: Prune BRGM hooks/API — grep first**

Run: `cd frontend && grep -rn "useBrgmTimeline\|useBrgmSectors\|brgmTimeline\|brgmSectors\|useNationalSituation" src/ --include="*.tsx" --include="*.ts"`
- If the ONLY hits are the definitions in `hooks/useObservatory.ts` and `lib/situation-api.ts`: delete `useBrgmSectors` + `useBrgmTimeline` (useObservatory.ts:375-381) and `brgmSectors` + `brgmTimeline` entries (situation-api.ts:13-16), and remove `BrgmSector, BrgmTimeline` from the situation-api import.
- `useNationalSituation`: delete only if no consumer outside `useObservatory.ts` remains (the rewritten MeteoNappesPage no longer uses it; ObservatoryPage may — check the grep output and KEEP it if used).

- [ ] **Step 3: Prune meteo-colors — grep first**

Run: `cd frontend && grep -rn "summarizeAlert\|isCriticalClass\|classSeverityRank\|METEO_CRITICAL_CLASSES\|BRGM_CLASS_TO_ENUM\|meteoSectorColorPairs\|AlertView\|AlertSummary" src/ --include="*.tsx" --include="*.ts" | grep -v "lib/meteo-colors"`
For every symbol with NO hits outside `lib/meteo-colors*`: remove it from `frontend/src/lib/meteo-colors.ts` and remove/adjust its tests in `frontend/src/lib/meteo-colors.test.ts`. Keep `METEO_CLASS_COLORS`, `METEO_CLASS_LABELS`, `METEO_TREND_LABELS`, `meteoClassColor` (used by the new code). If a symbol IS used elsewhere (e.g. by ObservatoryPage), keep it.

- [ ] **Step 4: Full verification**

Run: `cd frontend && npx tsc -b --noEmit && npx vitest run && npx eslint src/ 2>&1 | tail -5`
Expected: typecheck clean, all tests pass, no new lint errors.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src
git commit -m "chore(meteo): suppression bandeau/liste critique/ancien slider + code source BRGM côté front"
```

---

### Task 13: Build + visual recette against the original

- [ ] **Step 1: Production build**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 2: Run the dev stack and capture /meteo**

Start the frontend dev server (`cd frontend && npm run dev` — it proxies the API per `vite.config.ts`; if the API is not running locally, use the deployed recette at `http://localhost:49502/meteo` after deploying, or start the backend per `DEPLOYMENT.md`). With Playwright (browser tools) at viewport 1680×1000:
1. Screenshot `http://localhost:5173/meteo` (or the recette URL).
2. Screenshot `https://app.meteeaunappes.brgm.fr/desktop`.
3. Compare side by side: light OSM basemap, choropleth palette, circular white badges with black arrows, left Type+legend cards, bottom month-chip timeline with date chip, search top-left, À propos top-right, minimap bottom-right. No TopNav, no onboarding modal, no dark theme, no source toggle, no alert banner.

- [ ] **Step 3: Functional checks (manual, in the same browser session)**

- Click a sector → single popup with name/class/trend; click a station (zoom ≥ 7) → station popup replaces it.
- Timeline: click an older month → choropleth + badges update; URL gains `?month=`; reload keeps the month.
- Date chip → picker → pick 2018 month → window recenters.
- Search « Orléans » → suggestions (adresse + stations) → fly-to → « réinitialiser × » chip restores the France view.
- Type panel: uncheck Piézomètre → markers disappear; Source/Pluviomètre/Avec modèle greyed.

- [ ] **Step 4: Fix anything broken, then commit any fixes**

```bash
git add -A && git commit -m "fix(meteo): ajustements recette visuelle clone"
```
(Skip the commit if nothing changed.)

---

## Self-review notes (done at plan time)

- Spec coverage: map/choropleth/badges (T2-T5), panels (T6, T8), timeline+picker (T1, T7), search+reset (T9), minimap+about (T10), page+route+URL sync (T11), removals+pruning (T12), tests (T1, T7, T8), recette (T13). Error handling: BAN failure → stations only (T9); sector fetch failure logs and leaves grey fill (T5 default paint).
- The spec's "French toast on fetch failure" is downgraded to the grey-fill fallback + console error — a toast system doesn't exist in this app and inventing one is out of proportion (YAGNI); the map stays usable.
- Type consistency checked: `StationType`/`Trend` exported from layer modules and reused; `MeteoMap` props match page usage; `TimelineCell` fields match component usage; `SearchTarget` shared.
