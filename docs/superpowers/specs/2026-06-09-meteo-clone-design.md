# /meteo — BRGM "MétéEau des nappes" Faithful Clone — Implementation Spec

## 1. Goal & fidelity targets

Build a new, standalone, full-screen map page at route `/meteo` that visually and behaviorally clones the public BRGM app at `app.meteeaunappes.brgm.fr/desktop`, using **our existing backend** (no scraping of BRGM). "Clone" means concretely:

- **Full-screen OpenStreetMap basemap**, no app navbar/header. Tile URL exactly `https://tile.openstreetmap.org/{z}/{x}/{y}.png`, OSM attribution bottom-left. Default view center `[2.5, 46.5]`, zoom `6`.
- **BSH sector choropleth** ("Bulletin de situation des nappes") with **BRGM's exact 7-class + no-data palette and French labels** (hex table in §4.3 / §7).
- **Trend arrows** per sector (up/right/down) rendered from `tendancy_coord`, plus an **"Evolution des niveaux"** legend block.
- **"Couches affichées"** floating panel with BSN / Piézomètres / Pluviomètres / Hydrométrie toggles. Stations render as **individual markers (NO clustering)**.
- **Monthly time slider** pinned full-width to the bottom, driven by `GET /observatory/situation/sectors/timeline`, defaulting to the most recent period.
- **Sector click popup**: sector name, BRGM class label + color, trend, `% sous la normale`, station counts.
- **Always-French UI**, no franglais (per project memory). Technical hex/labels verbatim from BRGM.

Non-targets: pixel-perfect chrome of BRGM's React widgets; forecast/model charts; sub-sector half-month resolution. See §9.

---

## 2. Page/route & high-level layout

### Route
In `frontend/src/routes.tsx`, the `/meteo` entry currently is `{ path: '/meteo', element: <Navigate to="/" replace /> }` inside the `<SessionGate><Layout /></SessionGate>` children. The clone must render **full-screen without the `Layout` navbar**. Add `/meteo` as a **sibling route object** (outside the `Layout` children array), wrapped only in `SessionGate`:

```tsx
const MeteoNappesPage = lazy(() => import('./pages/MeteoNappesPage'))

export const router = createBrowserRouter([
  {
    element: <SessionGate><Layout /></SessionGate>,
    children: [
      { path: '/', element: <SW><ObservatoryPage /></SW> },
      { path: '/observatory', element: <Navigate to="/" replace /> },
      // REMOVE the old: { path: '/meteo', element: <Navigate to="/" replace /> },
      ...
    ],
  },
  // NEW standalone full-screen route, no <Layout/>:
  { path: '/meteo', element: <SessionGate><SW><MeteoNappesPage /></SW></SessionGate> },
])
```

(Keep `SW` = the existing `Suspense` wrapper. If `SessionGate` redirects unauthenticated users, that behavior is preserved; if `/meteo` must be public, drop the `SessionGate` wrapper for this route.)

### Layout (all overlays absolutely positioned over the map canvas)

```
┌───────────────────────────────────────────────────────────┐
│ [SearchBar]            (top-left, top-4 left-4, z-10)       │
│ [Couches affichées]    (below search, top-[100px] left-3)  │
│                                                            │
│                  <MapLibre canvas, inset-0>                 │
│                                                            │
│ [MeteoLegend]          (bottom-left, above slider, z-10)   │
│ [active period chip]   (bottom-left, "Juin 2026 ×")        │
│                            [Nav/Fullscreen/Scale] (br, z-10)│
│ [SituationTimelineSlider]  (bottom-0 left-0 right-0, h-48px)│
└───────────────────────────────────────────────────────────┘
```

### Difference from the Observatory page (critical)
`ObservatoryPage` + `ObservatoryMap.tsx` **MUST NOT be reused**. `ObservatoryMap.tsx` (44 KB) uses `addClusteredSource()` (`cluster: true, clusterMaxZoom: 9, clusterRadius: 80`), the CARTO Voyager basemap (`https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json`), SANDRE/CARTHAGE WFS layers, a dark right-drawer, and our internal palette. The clone is a **separate, smaller component tree** (`components/meteo/*`) with: OSM raster basemap, **no clustering**, only BSN + station markers, and BRGM colors. We may reuse pure helpers (`parseTendancyCoord`, `SearchBar`) but not the map component.

---

## 3. Component tree & files to create

| File (under `frontend/src/`) | Responsibility |
|---|---|
| `pages/MeteoNappesPage.tsx` | Page shell. Mounts `MeteoMap`, overlays (`SearchBar`, `MeteoLayersPanel`, `MeteoLegend`, `SituationTimelineSlider`, period chip, sector/station popups). Owns all React state: visible layers, selected period, selected sector/station. Runs the data `useQuery`s. |
| `components/meteo/MeteoMap.tsx` | The MapLibre map: OSM basemap, controls, sector fill/line layers, arrow symbol layer, 3 unclustered station marker layers. Pure imperative MapLibre; receives props + emits callbacks. **No clustering.** |
| `components/meteo/MeteoLayersPanel.tsx` | "Couches affichées" white card with checkbox toggles in BRGM order + zoom footnote. Controlled component. |
| `components/meteo/MeteoLegend.tsx` | Always-visible white legend card: Type / Evolution des niveaux / Niveau sections, BRGM labels + colors. |
| `components/meteo/SituationTimelineSlider.tsx` | Full-width bottom month slider. Props `{ periods, selectedPeriod, onChange }`. |
| `components/meteo/SectorPopup.tsx` | Floating card for a clicked sector: name, class badge, trend, `% sous la normale`, counts. |
| `components/meteo/StationPopup.tsx` | Small card/popup for a clicked station: code, commune, class badge, `Dernière mesure`. |
| `lib/meteo-colors.ts` | `METEO_CLASS_COLORS`, `METEO_CLASS_LABELS`, `METEO_TREND_LABELS`, BRGM class-int↔enum maps, `meteoSectorColorPairs()` helper. Pure, no React/MapLibre imports. |

Reused as-is: `components/observatory/SearchBar.tsx` (props `{ features?, wfsData?, onSearchAction }`), `lib/sector-arrows.ts` (`parseTendancyCoord`, `SECTOR_INSUFFICIENT_COLOR`), `lib/situation-api.ts` (`situationApi`), `lib/observatory-api.ts` (`observatoryApi.common.geojson`).

---

## 4. Map config (`MeteoMap.tsx`)

### 4.1 Basemap & init
```ts
const FRANCE_CENTER: [number, number] = [2.5, 46.5]
const FRANCE_ZOOM = 6   // BRGM loads at 6 (NOT 5.5)

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
    glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf', // only if any text-field is used; sector/station use canvas icons so glyphs are optional
  },
  center: FRANCE_CENTER,
  zoom: FRANCE_ZOOM,
  maxBounds: [[-12, 38], [18, 54]],
  attributionControl: false, // we add it explicitly bottom-left
})
```
Container: `position:absolute; inset:0; width:100vw; height:100vh`.

### 4.2 Controls
```ts
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
map.addControl(new maplibregl.FullscreenControl(), 'bottom-right')
map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-right')
map.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-left')
```
Optional "ré-initialiser" reset button: a small overlay `<button>` (rendered in `MeteoNappesPage`) calling `map.flyTo({ center: FRANCE_CENTER, zoom: FRANCE_ZOOM })`. The territory dropdown / minimap / "A Propos" modal from the audit are **optional nice-to-haves**, not required for fidelity acceptance (§9).

### 4.3 Sector source + fill/line layers
Load our existing static `public/geo/secteurs-bsh.geojson` (already served at `/geo/secteurs-bsh.geojson`; features carry integer property `sector_id` and string `nom`). The **fill color is driven by a `match` on `sector_id`** built from the situation data (same pattern as `ObservatoryMap` line 563), but using **BRGM hex values** from `lib/meteo-colors.ts`.

```ts
fetch('/geo/secteurs-bsh.geojson').then(r => r.json()).then((gj) => {
  map.addSource('secteurs-bsh', { type: 'geojson', data: gj, attribution: 'Secteurs © BRGM / Eaufrance' })
  map.addLayer({ id: 'secteurs-fill', type: 'fill', source: 'secteurs-bsh',
    paint: { 'fill-color': SECTOR_INSUFFICIENT_COLOR /* #d9d9d9 */, 'fill-opacity': 0.7 } })
  map.addLayer({ id: 'secteurs-line', type: 'line', source: 'secteurs-bsh',
    paint: { 'line-color': '#ffffff', 'line-width': 0.6 } })
  // arrow layer (4.4) is added here too, above the fill
})
```
When `sectorSituation` data + the selected period are available, set the fill `match` expression:
```ts
// pairs = [sector_id, hex, sector_id, hex, ...] from meteoSectorColorPairs(sits)
map.setPaintProperty('secteurs-fill', 'fill-color',
  pairs.length
    ? (['match', ['get', 'sector_id'], ...pairs, SECTOR_INSUFFICIENT_COLOR] as maplibregl.ExpressionSpecification)
    : SECTOR_INSUFFICIENT_COLOR)
```
`fill-opacity: 0.7` over OSM raster (BRGM-like; the prior 0.55 was tuned over CARTO Voyager — raise it for OSM legibility). `secteurs-fill` and `secteurs-line` are toggled by the BSN layer checkbox.

### 4.4 Arrow icon layer (rotation)
Reuse the proven approach from `ObservatoryMap` (lines 173–188, 542–565):
- Canvas icon `sector-arrow` (40px), upward triangle, fill `#0f172a`, white stroke `~4.8px`, registered via `map.addImage('sector-arrow', createRgbaIcon(drawTrendArrow, 40))`. Copy `createRgbaIcon` + `drawTrendArrow` into `MeteoMap.tsx` (or extract to a shared `lib/map-icons.ts`).
- `TREND_ROTATION = { hausse: 0, stable: 90, baisse: 180 }`.
- Symbol layer:
```ts
map.addLayer({ id: 'secteurs-arrows', type: 'symbol', source: 'secteurs-arrows',
  layout: { 'icon-image': 'sector-arrow', 'icon-size': 0.7, 'icon-rotate': ['get', 'rot'],
            'icon-rotation-alignment': 'map', 'icon-allow-overlap': true, 'icon-ignore-placement': true } })
```
- Populate from situation data: for each sector `s`, if `s.trend != null` AND `parseTendancyCoord(s.tendancy_coord)` is valid AND `!s.insufficient`, emit `Feature{ Point [lon,lat], properties:{ rot: TREND_ROTATION[s.trend] } }`. (`parseTendancyCoord` already swaps "lat lon" → `[lon, lat]`.)

### 4.5 Station marker layers (NO clusters)
Three independent unclustered GeoJSON sources/layers. **Do not call `addClusteredSource`.**
```ts
map.addImage('piezo-marker', createSdfIcon(drawStationBadge, 44), { sdf: true })
map.addImage('hydro-marker', createSdfIcon(drawStationBadge, 44), { sdf: true })
map.addImage('rain-marker',  createSdfIcon(drawStationBadge, 44), { sdf: true })
map.addImage('piezo-glyph', createRgbaIcon(drawPiezoGlyph, 44))
map.addImage('hydro-glyph', createRgbaIcon(drawHydroGlyph, 44))
map.addImage('rain-glyph',  createRgbaIcon(drawRainGlyph, 44))  // new raindrop glyph

const MARKER_SIZE: maplibregl.ExpressionSpecification =
  ['interpolate', ['linear'], ['zoom'], 4, 0.4, 8, 0.55, 12, 0.8]

function addUnclustered(sourceId: string, badgeLayer: string, glyphLayer: string, badgeImg: string, glyphImg: string) {
  map.addSource(sourceId, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } }) // NO cluster:true
  map.addLayer({ id: badgeLayer, type: 'symbol', source: sourceId,
    layout: { 'icon-image': badgeImg, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true },
    paint: { 'icon-color': classificationColorExpr /* BRGM palette, see below */, 'icon-halo-color': 'rgba(2,6,23,0.5)', 'icon-halo-width': 1.4 } })
  map.addLayer({ id: glyphLayer, type: 'symbol', source: sourceId,
    layout: { 'icon-image': glyphImg, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true } })
}
addUnclustered('piezo-stations', 'piezo-layer', 'piezo-glyph-layer', 'piezo-marker', 'piezo-glyph')
addUnclustered('hydro-stations', 'hydro-layer', 'hydro-glyph-layer', 'hydro-marker', 'hydro-glyph')
addUnclustered('rain-stations',  'rain-layer',  'rain-glyph-layer',  'rain-marker',  'rain-glyph')

map.on('click', 'piezo-layer', e => { const c = e.features?.[0]?.properties?.code; if (c) onStationClickRef.current?.(c, 'piezo') })
map.on('click', 'hydro-layer', e => { const c = e.features?.[0]?.properties?.code; if (c) onStationClickRef.current?.(c, 'hydro') })
```
`classificationColorExpr` is a `match` on `['get','classification']` → BRGM hex (from `METEO_CLASS_COLORS`), fallback `#d9d9d9`. Copy `createSdfIcon`, `drawStationBadge`, `drawPiezoGlyph`, `drawHydroGlyph` from `ObservatoryMap.tsx`; add a simple `drawRainGlyph` (raindrop). "Source"/"Avec modèle" sub-types are out of scope unless trivially derivable (§9).

Marker data is fed from `observatoryApi.common.geojson('piezo')` and `('hydro')` features. We have no pluvio endpoint (§8); the Pluviomètres toggle stays present but its source remains empty (or hidden) until a backend source exists.

---

## 5. Layers panel (`MeteoLayersPanel.tsx`)

White card, `absolute top-[100px] left-3 z-10`, ~220px wide, title **"Couches affichées"**, one checkbox per row, BRGM order & defaults:

| key | label | default |
|---|---|---|
| `bsn` | Bulletin de situation des nappes | **on** |
| `piezo` | Piézomètres | **on** |
| `rain` | Pluviomètres | off |
| `hydro` | Hydrométrie | off |
| `model` | Modèles | off (out of scope visual — checkbox may be hidden/disabled) |
| `prev` | Prévisions | off (out of scope — hidden/disabled) |

Footnote: **"Certains éléments ne sont visibles qu'en zoomant"**. Toggling sets layer visibility in `MeteoMap` via a `visibleLayers` prop:
- `bsn` → `secteurs-fill`, `secteurs-line`, `secteurs-arrows`
- `piezo` → `piezo-layer`, `piezo-glyph-layer`
- `rain` → `rain-layer`, `rain-glyph-layer`
- `hydro` → `hydro-layer`, `hydro-glyph-layer`

`model`/`prev` may be omitted from the rendered list (forecast/model out of scope) — but if shown, render them disabled with a tooltip "Bientôt disponible".

---

## 6. Time slider (`SituationTimelineSlider.tsx`)

- **Data**: `MeteoNappesPage` calls `useQuery(['sectorsTimeline','piezo'], () => situationApi.sectorsTimeline('piezo'))`. Response shape (`SectorTimeline`): `{ periods: string[] /* "YYYY-MM" */, sectors: Record<sectorId, number[]>, trends: Record<sectorId, number[]> }`.
- **Props**: `{ periods: string[]; selectedPeriod: string | null; onChange: (p: string|null) => void }`.
- **Default on mount**: `periods[periods.length - 1]` (most recent published month).
- **Render**: full-width bar `position:absolute; bottom:0; left:0; right:0; height:48px`, white background. A `<input type="range" min={0} max={periods.length-1} value={currentIndex}>` with a filled blue circular thumb (`~#3b7fc8`) and thin gray track. Above the rail, evenly-spaced 3-letter French month abbreviations: `janv. févr. mars avr. mai juin juil. août sept. oct. nov. déc.` Insert an inline year badge (e.g. "2025", "2026") between a December tick and the following January tick.
- **Change handling**: on slider change, call `onChange(periods[i])`. `MeteoNappesPage` then recolors the choropleth. **Prefer local lookup** over an extra fetch: use `sectorTimeline.sectors[sectorId][periodIndex]` (class int) + `sectorTimeline.trends[sectorId][periodIndex]` to rebuild the `match` pairs and arrow features. Only fall back to `situationApi.sectors('piezo', period)` if a richer per-sector record (counts, `pct_below_normal`, `tendancy_coord`) is needed for the popup at a historical month.
- **Active period chip**: rendered by the page bottom-left (e.g. `"Juin 2026 ×"`); `×` resets `selectedPeriod` to the latest period.
- **No** play/pause, speed, season, or year-range controls (those live only in the Observatory `TimelineSlider`). Future stub months beyond "today" may be shown greyed and non-selectable; clamp `onChange` to valid indices.

---

## 7. Popup & legend specs (exact labels)

### 7.1 `lib/meteo-colors.ts` — BRGM palette & label maps (verbatim)
```ts
// Our 7-enum → BRGM hex. classes 1–2 (driest) never appeared in BRGM artifacts; reds interpolated.
export const METEO_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#d73027',  // class 1 — unconfirmed (interpolated red)
  TRES_BAS:        '#e84a1a',  // class 2 — unconfirmed (interpolated red)
  BAS:             '#f8930f',  // class 3 — modérément bas (confirmed)
  NORMAL:          '#ffde1a',  // class 4 — autour de la moyenne (confirmed)
  HAUT:            '#60a3d6',  // class 5 — modérément haut (confirmed)
  TRES_HAUT:       '#3071b0',  // class 6 — haut (confirmed)
  EXTREMEMENT_HAUT:'#00408b',  // class 7 — très haut (confirmed)
  UNKNOWN:         '#d9d9d9',  // class 0 — no data (confirmed)
}

// BRGM compresses our 7 enums into the 7 BRGM label strings (used ONLY on /meteo):
export const METEO_CLASS_LABELS: Record<string, string> = {
  EXTREMEMENT_BAS: 'très bas',
  TRES_BAS:        'bas',
  BAS:             'modérément bas',
  NORMAL:          'autour de la moyenne',
  HAUT:            'modérément haut',
  TRES_HAUT:       'haut',
  EXTREMEMENT_HAUT:'très haut',
  UNKNOWN:         'Sans nappe libre étendue / Absence de point de suivi',
}

export const METEO_TREND_LABELS: Record<string, string> = {
  hausse: 'en hausse',
  stable: 'stable',
  baisse: 'en baisse',
  // null → 'Inconnu'
}
```
**Do NOT** use the existing `CLASSIFICATION_LABELS` (Extrêmement bas / Normal / …) or `CLASSIFICATION_COLORS` on `/meteo` — those are internal and must remain unchanged for the Observatory pages. All BRGM mapping lives only in `meteo-colors.ts`.

`meteoSectorColorPairs(sits: SectorSituation[]): (number|string)[]` → flat `[sector_id, hex, ...]` where `hex = METEO_CLASS_COLORS[s.situation_class ?? 'UNKNOWN']`, skipping `insufficient`/null as `UNKNOWN`.

### 7.2 Legend (`MeteoLegend.tsx`) — white card, bottom-left, three sections in BRGM order
- Header **"Type"** (5 rows): `Piézomètre`, `Source`, `Station de pluie`, `Station de débit`, `Avec modèle`. (Render the icons we have; Source/Avec modèle may use placeholder glyphs.)
- Header **"Evolution des niveaux"** (4 rows): `↑ en hausse`, `→ stable`, `↓ en baisse`, `– Inconnu`. Arrow SVG (14×14, `transform: rotate(${rot}deg)`): `<path d="M7 2.5 L12 11 L2 11 Z" fill="#0f172a" stroke="#fff" stroke-width="1.2" stroke-linejoin="round"/>`; rot = 0/90/180; Inconnu = a grey dash.
- Header **"Niveau"** (8 swatches, **wet-first** top-to-bottom, 10×10px):
  `#00408b très haut`, `#3071b0 haut`, `#60a3d6 modérément haut`, `#ffde1a autour de la moyenne`, `#f8930f modérément bas`, `#e84a1a bas` (unconfirmed), `#d73027 très bas` (unconfirmed), `#d9d9d9 Sans nappe libre étendue / Absence de point de suivi`.

### 7.3 `SectorPopup.tsx`
Triggered by `MeteoMap` `onSectorClick(sectorId, name)` (extend the existing click handler to also pass `f.properties.sector_id` as string). Page looks up the matching `SectorSituation` record from the loaded `sectorSituation` array by `code === String(sectorId)`. Render a floating card (MapLibre `Popup` anchored at the sector's `parseTendancyCoord(tendancy_coord)`, or an absolute div at click `lngLat`):
1. Header: sector `name` (truncate ~80 chars) + `sector_id` muted.
2. Class badge: colored dot (`METEO_CLASS_COLORS[situation_class ?? 'UNKNOWN']`) + capitalized `METEO_CLASS_LABELS[...]`.
3. Trend: arrow + `METEO_TREND_LABELS[trend]` (or `Inconnu`).
4. `{pct_below_normal}% sous la normale` (if non-null).
5. `{n_eligible} stations fiables`; if `n_provisoire > 0`, append `(+ {n_provisoire} provisoires)`.
6. Close `×` top-right.

### 7.4 `StationPopup.tsx`
Triggered by `onStationClick(code, type)`. Data from the already-loaded GeoJSON feature (`StationGeoJSONProperties`); no extra fetch:
1. `code` (monospace, prominent).
2. `commune` (station name).
3. Class badge: dot + `METEO_CLASS_LABELS[classification ?? 'UNKNOWN']`.
4. `Dernière mesure : {derniere_mesure}` — `derniere_mesure` is month-bucketed `"YYYY-MM"`; format as `"mai 2026"`.
5. Omit depth (`profondeur`) and "Consulter les données du modèle" — not in our API / out of scope.
6. Close `×`.

### 7.5 SearchBar (wire existing component)
Import `components/observatory/SearchBar.tsx`. Props: `features={stationsGeojson.features}`, `wfsData={undefined}`, `onSearchAction={(action) => map.flyTo(...)}`. Position `absolute top-4 left-4 z-10`, width `w-72 sm:w-96`. Placeholder already resolves to **"adresse, station, piézomètre, etc."** via i18n key `observatory.search.placeholder` — verify that key's value; if it diverges from the BRGM string, fix the fr translation value (not the component).

---

## 8. Backend — existing endpoints suffice (one optional addition)

All required data is already served by existing FastAPI endpoints (prefix `/observatory`, base `API_BASE`):

| Need | Endpoint (existing) | Client call |
|---|---|---|
| Sector situation (color, trend, counts, `tendancy_coord`) for a month | `GET /observatory/situation/sectors?type=piezo[&month=YYYY-MM]` → `SectorSituation[]` | `situationApi.sectors('piezo', period?)` |
| Timeline of periods + per-sector class/trend arrays | `GET /observatory/situation/sectors/timeline?type=piezo` → `SectorTimeline` | `situationApi.sectorsTimeline('piezo')` |
| Station markers (piezo) | `GET /observatory/stations/geojson?type=piezo` → `StationGeoJSON` | `observatoryApi.common.geojson('piezo')` |
| Station markers (hydro) | `GET /observatory/stations/geojson?type=hydro` | `observatoryApi.common.geojson('hydro')` |
| Sector polygons (geometry + `nom`,`sector_id`) | static `public/geo/secteurs-bsh.geojson` at `/geo/secteurs-bsh.geojson` | `fetch('/geo/secteurs-bsh.geojson')` |

**No new endpoint is required for acceptance.** The only gap is **Pluviomètres**: we have no rain-gauge GeoJSON. Options, in order of preference:
1. Render the `rain` toggle but leave its source empty (acceptable for the clone; documents the gap).
2. (Optional follow-up) Extend `/observatory/stats/national` data path with a `type=pluvio` GeoJSON if/when a rain-gauge table exists. **Prefer reusing `/observatory/stations/geojson`** with a future `type=pluvio` value rather than a new route.

No backend changes are in scope for the first delivery.

---

## 9. Out of scope

- Forecast/model timeseries charts and the "Consulter les données du modèle" CTA (no model-association data exposed).
- "Modèles" and "Prévisions" layers (render disabled/hidden checkboxes only).
- Depth-below-surface (`profondeur` / mNGF) in the station popup (not in our GeoJSON).
- "Source" and "Avec modèle" station sub-type markers as distinct glyphs (use existing piezo/hydro glyphs; only add a basic rain glyph).
- Sub-sector / half-month (1st & 15th) temporal resolution — slider stays one tick per calendar month.
- Territory dropdown (Guadeloupe/Martinique/…), minimap inset, "A Propos" modal, "ré-initialiser" button — **optional**; not required for fidelity acceptance. Build only if cheap.
- Pluviomètres data source (backend) — toggle present, source empty (§8).

---

## 10. Build & verification steps

1. **Create** the 8 files in §3; add the `MeteoNappesPage` lazy import + standalone `/meteo` route in `routes.tsx`; remove the old `Navigate to="/"` redirect for `/meteo`.
2. **Type-check & lint**: `cd /home/ringuet/time-serie-explo/frontend && npm run typecheck && npm run lint` (or the project's `tsc -b` / eslint scripts — check `package.json`). `meteo-colors.ts` must have zero React/MapLibre imports.
3. **Unit**: add a small test for `meteoSectorColorPairs()` and the class-int↔enum/label maps alongside the existing `sector-arrows.test.ts` pattern; run `npm test`.
4. **Run the app** (per project memory, frontend on port 49513, backend API 49514; use the existing dev/compose flow — `docker compose up -d --build` with the `.env` `COMPOSE_FILE` chain, never manual `-f`). Confirm backend `/observatory/situation/sectors`, `/observatory/situation/sectors/timeline`, `/observatory/stations/geojson` respond 200.
5. **Manual / Playwright verification** at `http://localhost:49513/meteo`:
   - Page is full-screen, **no top navbar**; OSM tiles load (network: requests to `tile.openstreetmap.org/{z}/{x}/{y}.png`); OSM attribution visible bottom-left.
   - Initial view ≈ center `[2.5,46.5]`, zoom `6`.
   - BSN choropleth renders with BRGM colors (spot-check a NORMAL sector = `#ffde1a`, a HAUT sector blue family); trend arrows present and rotated correctly.
   - "Couches affichées" toggles show/hide BSN + piézo markers; **markers never cluster** at any zoom (confirm individual points at z6 and z10; no numbered cluster bubbles).
   - Slider spans the bottom full-width, defaults to most recent period, month abbreviations + year dividers shown; dragging recolors the choropleth from local timeline lookup.
   - Click a sector → `SectorPopup` shows name, BRGM class label, trend, `% sous la normale`, counts. Click a station → `StationPopup` shows code, commune, class, `Dernière mesure`.
   - Search box placeholder reads exactly "adresse, station, piézomètre, etc."; selecting a result flies the map.
   - Console free of MapLibre errors; no franglais in any rendered string.
6. **Confirm Observatory unaffected**: `/` still renders the clustered `ObservatoryMap` with the original `CLASSIFICATION_COLORS`/`CLASSIFICATION_LABELS` (these constants are untouched).

Key reference files for the implementer: `frontend/src/components/observatory/ObservatoryMap.tsx` (icon/arrow helpers at lines 111–188, sector layers at 535–565 — copy patterns, do not import the clustered map), `frontend/src/lib/situation-api.ts`, `frontend/src/lib/sector-arrows.ts`, `frontend/src/lib/observatory-types.ts` (`SectorSituation`, `SectorTimeline`, `StationGeoJSONProperties`), `frontend/src/components/observatory/SearchBar.tsx`, `frontend/src/routes.tsx`.
```