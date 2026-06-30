# Brique météo ERA5 sur l'Observatoire — design

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending spec review

## Goal

Add a proper, intuitive weather module to the Observatoire map, built on the
ERA5 grid. The user wants something standard and robust ("un vrai truc
costaud"): not just the raw grid, but temporal aggregation, smoothing, and
spatial aggregation to administrative/hydro zones that follows the active map
layer ("la météo par département si on veut").

## Context (current state)

- **Frontend:** React 19 + Vite + MapLibre GL. Observatoire = `ObservatoryPage.tsx`
  + `ObservatoryMap.tsx`; layer toggles in `RightDrawer.tsx`; data hooks in
  `hooks/useObservatory.ts`; API client in `lib/observatory-api.ts`.
- **ERA5 plumbing exists but is unused by the UI:** hooks `useERA5Grid`,
  `useERA5Snapshot`, `useERA5Dates`, `useERA5Monthly`; endpoints under
  `/api/v1/observatory/era5/*` in `api/routers/observatory_era5.py`; types
  `ERA5GridPoint` / `ERA5Snapshot`.
- **Zone geometries are static GeoJSON** in `frontend/public/geo/`, fetched
  directly in `ObservatoryMap.tsx`:
  | Zone | File | Id property |
  |------|------|-------------|
  | Départements | `departments.geojson` | `code` (e.g. "01") |
  | Régions | `regions.geojson` | `code` (e.g. "11") |
  | HER2 | `her.geojson` | `code` |
  | Bassins SANDRE | `bassins.geojson` | `CdBH` (e.g. "A") |
  | Secteurs BSH | `secteurs-bsh.geojson` | `sector_id` |
  Zone layers are mutually exclusive (`activeZoneLayer` state). Hydro SANDRE
  layers come from a live WFS proxy (`/observatory/wfs/{id}`) — **excluded** from
  weather aggregation (dynamic, out of scope).
- **Data (warehouse `brgm-postgres`, schema `gold`, PostGIS, SRID 4326):**
  - `int_era5_grid_points` — 11,496 cells, 0.1°×0.1°, France
    (lat 41.0–51.5, lon −5.5–10.0), each with a valid `geom` Point.
  - `int_era5_for_all_stations` — daily values per cell (126M+ rows, 1950→2026-06-24):
    `temperature_2m` (°C), `total_precipitation` (mm), `potential_evaporation`.
    Only ~4,524 of 11,496 cells currently have values loaded.
  - No zone polygon tables exist in the warehouse (only points). A spatial join
    `ST_Contains(zone_polygon, grid_point.geom)` works and is instant (verified:
    60 cells in dept "01").

### Known bug to fix first

`api/routers/observatory_era5.py` queries `gold.int_era5_for_stations`
(lines 55, 74, 103) — **that table does not exist**. The real table is
`gold.int_era5_for_all_stations`. Today `snapshot` / `dates` / `monthly` fail;
only `grid` works. Fixing all three occurrences is step one.

## Decisions (from brainstorm)

The weather module is driven by **three independent axes** plus an anomaly toggle:

1. **Variable** — Température (°C) / Précipitations (mm) / ETP. Drives colour + legend.
2. **Temporal granularity** — **Jour**, **Mois**, **Année**, and **Anomalie vs
   normale 1991-2020**.
3. **Spatial display mode** — **Grille** (0.1° squares), **Lissé** (smoothed
   isobands), **Par zone** (choropleth aggregated to the active zone layer).
4. **"Par zone" follows the active map layer:** if Départements is the active
   zone layer, weather aggregates per department; switch to Régions → per region;
   etc. Selecting "Par zone" with no zone layer active auto-activates
   Départements.

## Architecture overview

A **"Météo (ERA5)" panel** in the Observatoire drawer (own group). A master
toggle enables weather mode; when on, the panel exposes: variable radios,
temporal-granularity selector + period stepper, display-mode selector, and a
legend. The render layer sits **below the station markers** (opacity ~0.6) so
stations and selection rings stay visible. Clicking a cell/zone opens a popup
with the 3 values (and the anomaly if in anomaly mode).

Data flow: `ObservatoryPage` holds weather state, calls the right hook per
(granularity × mode), passes the result + state to `ObservatoryMap` (which
builds/styles the MapLibre layer) and `RightDrawer` (controls + legend).

## Backend (`api/routers/observatory_era5.py` + warehouse)

### Endpoints
- **Fix** `int_era5_for_stations` → `int_era5_for_all_stations` (3 occurrences).
- `GET /era5/snapshot?date=YYYY-MM-DD` — daily values per cell.
  Date omitted → latest available day.
- `GET /era5/monthly?month=YYYY-MM-01` — monthly aggregate per cell
  (AVG temp, SUM precip, AVG ETP).
- `GET /era5/yearly?year=YYYY` — annual aggregate per cell (same aggregations).
- `GET /era5/range` — `{min_date, max_date}` to bound the period stepper.
- `GET /era5/anomaly?period=monthly|yearly&month=…|year=…` — value minus the
  1991-2020 normal, per cell.
- `GET /era5/by-zone?zone_type=department|region|her|bassin|secteur&granularity=day|month|year[&anomaly=true]&period=…`
  — aggregate to zones (one row per zone code).
- All endpoints keep the existing Redis 24h caching pattern.

Client receives lat/lon centres; it builds the 0.1° square polygons itself
(centre ± 0.05°). Per-period payload ≈ 4,500 cells × few numeric fields — light.

### Warehouse (dbt, cross-repo) — the heavy additions

1. **`gold.int_era5_grid_zones`** — for each of the 11,496 cells, its
   `code_departement`, `code_region`, `code_her`, `code_bassin`, `sector_id`,
   computed **once** via `ST_Contains` against the zone polygons. Built as a dbt
   seed (load the `public/geo/*.geojson` contours into a staging table) + model.
   Because cells are fixed, the mapping is stable. Zone aggregation then becomes a
   cheap `GROUP BY` join with no geometry at query time:
   ```sql
   SELECT z.code_departement,
          AVG(e.temperature_2m), SUM(e.total_precipitation), AVG(e.potential_evaporation)
   FROM gold.int_era5_for_all_stations e
   JOIN gold.int_era5_grid_zones z
     ON e.latitude = z.era5_latitude AND e.longitude = z.era5_longitude
   WHERE e.era5_date >= :start AND e.era5_date < :end
   GROUP BY z.code_departement;
   ```
2. **`gold.int_era5_normals`** — climatological normals over 1991-2020 per cell:
   one row per (cell, calendar_month, variable) for the monthly normal, and a
   per-cell annual normal. Monthly normal = mean over 1991-2020 of that calendar
   month's monthly aggregate. Anomaly = current period value − matching normal.
   Built as a dbt model over `int_era5_for_all_stations`.

> Cross-repo note: `int_era5_grid_zones` and `int_era5_normals` live in the dbt
> warehouse repo (consistent with the IPS fixed-reference precedent). The app
> only reads them. Materialisation runs via the existing dbt/Dagster pipeline.

## Frontend

### `ObservatoryPage.tsx`
- State: `weatherActive`, `weatherVariable` (`temperature|precipitation|evaporation`),
  `weatherGranularity` (`day|month|year`), `weatherAnomaly` (bool),
  `weatherPeriod` (date/month/year), `weatherMode` (`grid|smooth|zone`).
- Pick the hook by (granularity, mode, anomaly): snapshot / monthly / yearly /
  anomaly / by-zone. Init `weatherPeriod` from `/era5/range`.
- In `zone` mode, read `activeZoneLayer`; if none, set it to `departments`.

### `ObservatoryMap.tsx`
- **Grid mode:** GeoJSON FeatureCollection of square polygons (centre ±0.05°),
  `fill` layer below stations, `fill-color` = `interpolate` over the active
  variable, `fill-opacity` ~0.6. Click → popup of the 3 values.
- **Smooth mode:** the regular ERA5 lat/lon grid feeds `d3-contour` (marching
  squares) → smooth filled isobands rendered as a `fill` layer with a quant. legend.
  (Proper, legendable smoothing — not MapLibre's density heatmap.)
- **Zone mode:** recolour the active zone layer's polygons via a
  `['match', ['get', <idProp>], …zoneData…]` fill-color expression, overriding
  the layer's default palette while weather mode is on. Click zone → popup with
  the zone's aggregated values. Restore the default palette when weather mode off.
- **Anomaly:** divergent colour scale (blue↔red around 0) for whichever mode.
- Variable change → `setPaintProperty`; period/granularity/mode change → refetch
  + rebuild source.

### `RightDrawer.tsx`
- New group « Météo (ERA5) » with the master toggle and, when active, the control
  sub-panel: variable radios, granularity selector + period stepper (bounded by
  range), display-mode selector, anomaly checkbox, and the legend.

### Colour scales + legend
- Température: blue→red, ≈ −10 → +35 °C (day) / adjusted domains for month/year.
- Précipitations: light→blue, 0 → cap (≈50 mm day, larger for month/year), clamp.
- ETP: see Edge cases (negative-value convention).
- Anomaly: divergent scale centred on 0, symmetric bounds per variable.
- Legend shows gradient + min/max + unit for the active variable/mode.

### i18n (`frontend/src/i18n/locales/fr.json`)
New keys under `observatory.drawer.*`: group label, master toggle, variable
labels, granularity labels (Jour/Mois/Année/Anomalie), mode labels
(Grille/Lissé/Par zone), popup field labels + units, "pas de données pour cette
période".

## Edge cases / attention points

- **ETP stored negative** (`potential_evaporation`, water-loss convention):
  display a readable magnitude labelled "ETP" in popups; colour scale accounts
  for the sign. Confirm convention against sample rows during implementation.
- **Cells without data** for the chosen period: not drawn (~4,500 of 11,496 have
  values). **Period with no data at all:** discreet "pas de données" message.
- **Zone mode + no active zone layer:** auto-activate Départements.
- **Zone coverage:** weather-by-zone limited to the static-GeoJSON zones
  (dept/région/HER/bassin/secteur-BSH); live WFS hydro zones excluded.
- **Daily anomaly excluded:** anomaly only for Mois/Année (day-of-year normals
  need smoothing — out of scope). Daily granularity hides the anomaly toggle.
- **Performance:** ~4,500 fill polygons / d3-contour on a 106×156 grid / a
  GROUP BY join on a precomputed mapping — all comfortable; Redis-cached 24h.

## Build phases (sequencing for the plan)

Each phase is independently shippable and ends in a working app.

- **Phase 0 — Fix + baseline grid (Jour).** Fix table-name bug; `snapshot` +
  `range`; weather panel skeleton (variable radios, day stepper, legend); grid
  squares; click popup. (This is the original minimal scope.)
- **Phase 1 — Temporal aggregation.** `monthly` + `yearly`; granularity selector;
  domain adjustments per granularity.
- **Phase 2 — Smoothed mode.** d3-contour isobands + mode selector.
- **Phase 3 — Par zone.** Warehouse `int_era5_grid_zones` (dbt, cross-repo);
  `by-zone` endpoint; recolour the active zone layer; follow-active-layer logic;
  zone popups.
- **Phase 4 — Anomalie vs normale.** Warehouse `int_era5_normals` (dbt);
  `anomaly` + `by-zone&anomaly`; divergent scales; anomaly toggle.

The two warehouse models (Phases 3 & 4) are the heaviest / riskiest and are
cross-repo. They can be deferred without blocking Phases 0–2.

## Testing

- **Backend:** targeted pytest per endpoint (correct table, response shape,
  default-latest-day, aggregation maths, by-zone grouping, anomaly = value −
  normal). Run: `DEBUG=true DB_PASSWORD=test uv run pytest <files> -q`.
- **Warehouse:** dbt tests on the new models (row counts, non-null keys, every
  cell mapped to ≤1 dept/region, normals coverage 1991-2020).
- **Frontend:** manual verification per phase (toggle, each variable, each
  granularity, each mode, zone follow-active-layer, anomaly divergent scale,
  popups, stations stay visible). Unit-test the centre→square polygon builder and
  the d3-contour input builder as pure helpers.

## Out of scope (YAGNI)

- Animation / time playback across periods.
- Weather aggregation onto live WFS hydro zones.
- Daily anomaly (day-of-year normals).
- Scientific interpolation (kriging) — smoothing is isobands only.
- ERA5 on the standalone `/meteo` page (Observatoire only).
