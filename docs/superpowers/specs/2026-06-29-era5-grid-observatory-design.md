# ERA5 weather grid layer on the Observatoire — design

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending spec review

## Goal

Let users see the ERA5 weather grid directly on the Observatoire map: the 0.1°
grid cells over France, coloured by a chosen variable, with all values readable
on click. The request: *"voir la météo sur l'observatoire, rien que la grille
ERA5 avec toutes les valeurs"*.

## Context (current state)

- **Frontend:** React 19 + Vite + MapLibre GL. Observatoire = `ObservatoryPage.tsx`
  + `ObservatoryMap.tsx`; layer toggles in `RightDrawer.tsx`; data hooks in
  `hooks/useObservatory.ts`; API client in `lib/observatory-api.ts`.
- **ERA5 plumbing already exists but is unused by the UI:** hooks `useERA5Grid`,
  `useERA5Snapshot`, `useERA5Dates`, `useERA5Monthly`; endpoints under
  `/api/v1/observatory/era5/*`; types `ERA5GridPoint` / `ERA5Snapshot`.
- **Backend:** `api/routers/observatory_era5.py`.
- **Data (warehouse `brgm-postgres`, schema `gold`):**
  - `int_era5_grid_points` — 11,496 cells, 0.1°×0.1°, France
    (lat 41.0–51.5, lon −5.5–10.0), with PostGIS geometry.
  - `int_era5_for_all_stations` — daily values per cell:
    `temperature_2m` (°C), `total_precipitation` (mm),
    `potential_evaporation`. Range up to 2026-06-24. Only ~4,524 of the
    11,496 cells currently have values.

### Known bug to fix first

`api/routers/observatory_era5.py` queries `gold.int_era5_for_stations`
(lines 55, 74, 103) — **that table does not exist**. The real table is
`gold.int_era5_for_all_stations`. Today the `snapshot` / `dates` / `monthly`
endpoints fail; only `grid` (which uses `int_era5_grid_points`) works. This fix
is part of the work.

## Decisions (from brainstorm)

1. **Rendering:** coloured squares — each 0.1° cell is a filled polygon coloured
   by the selected variable (not points, not a smoothed heatmap).
2. **Time:** day-selectable. A date stepper (◀ date ▶), default = latest
   available day. Raw daily values (snapshot endpoint).
3. **Colour variable:** a 3-way selector (Température / Précipitations / ETP)
   drives the fill colour + legend; default = Température. All 3 values always
   shown in the click popup regardless of the selected variable.

## Architecture

A **toggleable overlay layer** in the Observatoire layer panel, under a new
group **« Météo (ERA5) »**. When enabled:

- ERA5 cells render on the map as filled squares, coloured by the active
  variable, drawn **below the station markers** at ~0.6 opacity so stations and
  selection rings stay visible.
- A small control sub-panel appears in the drawer: variable selector (radios)
  + day stepper + legend.
- Clicking a cell opens a popup with the cell's 3 values for that day.

## Backend

File: `api/routers/observatory_era5.py`

1. **Fix table name:** `gold.int_era5_for_stations` → `gold.int_era5_for_all_stations`
   (3 occurrences). Verify `snapshot` returns the documented shape afterwards.
2. **`GET /observatory/era5/snapshot?date=YYYY-MM-DD`** → list of
   `{latitude, longitude, temperature_2m, total_precipitation, potential_evaporation}`.
   - If `date` is omitted → use the latest available day
     (`SELECT max(era5_date) FROM gold.int_era5_for_all_stations`).
   - Only return rows that have at least one non-null value.
3. **`GET /observatory/era5/range`** (new, small) → `{min_date, max_date}`,
   to bound the date stepper on the client.
4. Keep the existing Redis 24h caching pattern for both endpoints.

No DB geometry is needed by the client: it receives lat/lon centres and builds
the square polygons itself (centre ± 0.05°). Payload per day ≈ 4,500 rows × 5
numeric fields — light.

## Frontend

### `ObservatoryPage.tsx`
- New state: `era5Active: boolean`, `era5Date: string`, `era5Variable:
  'temperature' | 'precipitation' | 'evaporation'`.
- Fetch with the existing `useERA5Snapshot(era5Date)` hook; fetch
  `/observatory/era5/range` (new hook `useERA5Range`) to bound the stepper and
  initialise `era5Date` to `max_date`.
- Pass data + state + setters down to `ObservatoryMap` and `RightDrawer`.

### `ObservatoryMap.tsx`
- When `era5Active` and snapshot data are present:
  - Build a GeoJSON `FeatureCollection` of square polygons (each centre ±0.05°),
    each feature carrying the 3 values as properties.
  - Add source `era5-grid` + `fill` layer `era5-grid-fill`, inserted **below**
    the station marker layers (use the existing first-station-layer id as the
    `beforeId`).
  - `fill-color` = an `interpolate` expression over the property matching
    `era5Variable`; `fill-opacity` ≈ 0.6.
- On `era5Variable` change → update the paint property (`setPaintProperty`) with
  the new colour expression (no full rebuild).
- On `era5Date` change → replace the source data with the new day's polygons.
- On toggle off → set layer visibility `none` (keep source for quick re-enable)
  or remove; mirror the existing WFS-layer add/remove pattern.
- Click handler on `era5-grid-fill` → MapLibre popup listing the 3 values
  (see formatting in Edge cases).

### `RightDrawer.tsx`
- Add an overlay entry « Grille ERA5 » in the new group « Météo (ERA5) »,
  using the existing `overlayLayers` Set + `onOverlayToggle` pattern.
- When the ERA5 overlay is active, render a sub-panel: variable radios, day
  stepper (◀ / date / ▶, bounded by min/max), and the legend.

### Legend + colour scales
- **Température:** blue→red, domain ≈ −10 → +35 °C.
- **Précipitations:** light→blue, domain 0 → ~50 mm (values above the cap clamp
  to the top colour).
- **ETP:** see Edge cases (negative-value convention).
- Legend shows the gradient + min/max labels + unit for the active variable.

### i18n (`frontend/src/i18n/locales/fr.json`)
New keys under `observatory.drawer.*`, e.g.:
- `groupWeatherEra5`: "Météo (ERA5)"
- `era5Layer`: "Grille ERA5"
- `era5ColorBy`: "Colorer par"
- `era5VarTemperature`: "Température"
- `era5VarPrecipitation`: "Précipitations"
- `era5VarEvaporation`: "Évapotranspiration"
- popup field labels + units, "pas de données pour cette date".

## Edge cases / attention points

- **ETP stored negative:** `potential_evaporation` uses a negative sign
  convention (water loss). In the popup, display a readable magnitude labelled
  "ETP"; the colour scale accounts for the sign (more negative = more
  evapotranspiration). Confirm exact convention against a couple of sample rows
  during implementation.
- **Cells without data for the chosen day:** simply not drawn (~4,500 of 11,496
  cells have values).
- **Day with no data at all:** show a discreet "pas de données pour cette date"
  message; map shows no ERA5 cells.
- **Performance:** ~4,500 fill polygons is comfortable for MapLibre; GeoJSON
  rebuild on date change is cheap.

## Testing

- **Backend:** targeted pytest for `snapshot` (correct table, response shape,
  default = latest day) and `range` (min/max). Run per project convention:
  `DEBUG=true DB_PASSWORD=test uv run pytest <files> -q`.
- **Frontend:** manual verification in the app — toggle on/off, change variable,
  step the date, click a cell popup, confirm stations stay visible. Add a unit
  test for the centre→square polygon builder if extracted as a pure helper.

## Out of scope (YAGNI)

- Monthly aggregates view (`useERA5Monthly`) — daily only for now.
- Animation/time playback across days.
- Interpolated/smoothed heatmap rendering.
- ERA5 on the standalone `/meteo` page (this is the Observatoire only).
