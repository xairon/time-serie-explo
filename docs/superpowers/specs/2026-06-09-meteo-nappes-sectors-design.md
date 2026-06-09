# Météo des nappes by hydrogeological sectors, folded into the Observatory (design)

Date: 2026-06-09
Status: design — pending implementation plan
Audience: BRGM / public + internal explorers (single map)

## 1. Goal

Bring our "Météo des nappes" to parity with the real BRGM site
(`meteeaunappes.brgm.fr`) on the two things that actually make it look and behave
differently from ours today:

1. **The mesh.** BRGM colors **hydrogeological sectors** (their BSH/BSN sectors),
   not administrative regions/departments. A region straddles several aquifers,
   so the administrative mesh has no hydrogeological meaning. We adopt the BRGM
   sector mesh.
2. **The trend arrows.** BRGM draws a hausse/stable/baisse arrow on each sector.
   We have the trend data (Δz) but only render it as a text glyph in a list.

Rather than maintain a second map, we **fold the sector "météo" view into the
existing Observatory** (which already has station markers, a time slider, layer
toggles, search, filters) as a new toggleable layer, and **retire the standalone
`/meteo` page**.

### Audit summary (why this design)

Network inspection of the real app established the ground truth, correcting a
prior assumption that BRGM uses HER-2 hydroécorégions:

- Coloring mesh = **secteur hydrogéologique** via GeoServer WFS
  `indicateur_bsn:view_global_indicator_details` on
  `https://app.meteeaunappes.brgm.fr/wfs/indicateur_bsn/ows`. **66 parent
  sectors** (`is_parent=true, communicate=true, visualizer=true`), `MultiPolygon`,
  EPSG:4326. Not HER-2, not BD Carthage, not BDLISA, not admin.
- Per-feature props: `sector_id`, `ips`, `tendency` (-1/0/1), `class` (0–7),
  `status` (V/D), `color`, `tendancy_coord` ("lat lon" placing the arrow),
  `is_parent`, `parent_id`.
- Arrows are **per sector**, positioned at `tendancy_coord`.
- 7 IPS classes + 1 grey "no data / no extensive free aquifer".
- Reference = full rolling chronicle (min 15 yr). **We deliberately keep our
  fixed 1991–2020 reference** (more rigorous, stable) — see
  `project-ips-fixed-reference`.
- Forecast (6-month, weather scenarios) lives in per-station model sheets — **out
  of scope here** (separate spec).

### Decisions locked during brainstorming

| Topic | Decision |
| --- | --- |
| Mesh | **BRGM BSH sectors** (geometry fetched from their WFS, frozen as static geojson) |
| Integration | **New toggleable layer in the Observatory**; `/meteo` retired (redirect) |
| Time slider | Slider replays **sectors AND stations** together |
| Classification method | **Fixed-reference IPS everywhere** (incl. slider), via new warehouse asset; fixes the current timeline inconsistency |
| Data pipeline | **Hybrid**: warehouse does the math (`fct_monthly_index`), app owns cartography (geometry + station→sector mapping + aggregation) |
| Sector names | **Derived from the dominant BDLISA aquifer** of contained stations, baked into the static geojson at build time |
| Conceptual scope | Sector layer is **groundwater (piezo)**-aligned like BRGM; computable for hydro but piezo is the intended use |
| Forecast / anticipation | **Out of scope** — separate follow-up spec |

## 2. Architecture overview

A single map (the Observatory) gains a **"Situation par secteur"** layer:
a choropleth of the 66 BRGM hydrogeological sectors colored by **our** fixed
reference IPS, with trend arrows. The warehouse computes the monthly re-scored
index; the app owns the geometry, the station→sector mapping, and the
aggregation.

```
Warehouse (hubeau_data_integration)        App (time-serie-explo)
─────────────────────────────────         ───────────────────────────────────
gold.fct_monthly_index  ───reads──▶  /observatory/situation/sectors
 (type,code,month,z,index_class,flag)      /observatory/situation/sectors/timeline
                                           station→sector map (point-in-polygon, cached)
                                           secteurs-bsh.geojson (static, with nom)
                                                     │
                                                     ▼
                                           ObservatoryMap sector layer
                                           (choropleth + arrows + slider)
```

## 3. Component 1 — Warehouse: `gold.fct_monthly_index` (new dagster asset)

Cross-repo (`~/hubeau_data_integration`, branch main).

- For each `(type, code, month)` in `gold.fct_monthly_chroniques` (piezo,
  `niveau_moyen`) and `gold.fct_monthly_hydro` (hydro, `resultat_moyen`),
  re-score the monthly value against `gold.station_reference_stats`
  (`quantile_grid` for that calendar month) → `z`, then to `index_class` using the
  7-class cutoffs.
- Reuses `ml/indices.py` (`grid_to_zscore`, `grid_class_bounds`) — the same code
  that produces `station_current_index`, so methodology is identical.
- Columns: `type, code, month, z, index_class, flag` (carry the reference `flag`
  so eligibility = `flag IN ('normale','adaptee')`, like the current situation
  endpoint).
- **Cadence**: append nightly (latest month) alongside `station_current_index`;
  full rebuild only when the fixed reference is re-materialized (per decade).
- **Collateral benefit**: this becomes the source for the existing
  `/observatory/classifications/timeline` endpoint, replacing the runtime
  `PERCENT_RANK()` rolling-percentile SQL. After this change, station markers,
  the sector choropleth, and the time slider all use the **same** fixed-reference
  IPS — removing the current methodological split.

Materialization (per `project-ips-fixed-reference`): same pattern as
`station_current_index` (`dagster asset materialize --select fct_monthly_index`,
`docker restart brgm-dlt-worker` after code change).

## 4. Component 2 — App: sector geometry static asset

A one-time, scripted fetch turns the BRGM WFS into a static geojson we own.

- Script `scripts/build_secteurs_bsh_geojson.py`:
  1. Fetch parent sectors from the BRGM WFS (filter
     `is_parent=true AND communicate=true AND visualizer=true`).
  2. Keep only **geometry + `sector_id` + `tendancy_coord`**; discard their
     `ips/class/tendency/status` (those are BRGM's own computation; we compute
     ours).
  3. Compute `nom` per sector (see Component 6) and bake it in.
  4. Write `frontend/public/geo/secteurs-bsh.geojson`.
- Output is committed; refreshed only if BRGM changes the sectorization. The
  script is documented and re-runnable.
- Attribution: BRGM / Eaufrance noted in the legend/credits.

## 5. Component 3 — App backend: station→sector mapping + sector aggregation

- **Mapping**: on first use (cached process-wide), load
  `secteurs-bsh.geojson` and build `code → sector_id` by **point-in-polygon**
  (shapely) over the station coordinates already available to the API. Stations
  outside every sector (DROM, offshore) are simply unmapped — still shown as
  markers, just excluded from sector aggregation.
- **Aggregation reuse**: generalize `api/routers/observatory_situation.py` to
  accept `level=sector`. The grouping key becomes `sector_id` from the
  point-in-polygon map (instead of `region_of()` / `DEPT_TO_REGION`). The verdict
  math (`dashboard/utils/territory_situation.py::aggregate_situation` /
  `aggregate_trend`, `MIN_ELIGIBLE=3`, `TREND_STABLE_BAND=0.5`) is reused
  unchanged.
- **Endpoints**:
  - `GET /observatory/situation/sectors?type=piezo|hydro[&month=YYYY-MM]`
    → list of `{sector_id, nom, situation_class, trend, pct_below_normal,
    n_eligible, n_provisoire, distribution, insufficient, tendancy_coord}`.
    Current month reads `gold.station_current_index`; a past `month` reads
    `gold.fct_monthly_index`.
  - `GET /observatory/situation/sectors/timeline?type=piezo|hydro`
    → `{periods: string[], sectors: {sector_id: classIdx[]}, trends:
    {sector_id: trendCode[]}}`, aggregated per sector per month from
    `gold.fct_monthly_index`. Mirrors the existing
    `/classifications/timeline` shape so the frontend slider logic is reused.
- Caching: same TTLs as siblings (situation 6 h, timeline 24 h).

## 6. Component 4 — Sector names from dominant BDLISA aquifer

Canonical BRGM names exist (`Bassin.nom` via `/BSN/BSH`) but are **auth-gated**
(401/500 without Keycloak), so we derive our own real, self-sufficient labels.

- At geojson build time (Component 2), for each sector: point-in-polygon the
  **piezo** stations inside it, read their BDLISA aquifer (`codes_bdlisa` and the
  nappe name already used by the BDLISA siblings feature, `level=nappe`), and take
  the **most frequent** nappe name as the sector `nom`.
- Tie-break / empty sector → fall back to `"Secteur {sector_id}"` (and optionally
  the dominant department name).
- `nom` is a **static property** of the geojson — stable, no runtime cost, no
  external dependency.

## 7. Component 5 — Frontend: sector layer in `ObservatoryMap`

- **Toggle** in `RightDrawer` "Couches": "Situation par secteur (météo des
  nappes)". Mutually sensible with existing layers (it is a fill+arrow overlay).
- **Choropleth**: add source `secteurs-bsh` + fill layer colored by
  `match(sector_id → class color)` using `CLASSIFICATION_COLORS`, + line layer.
  The coloring logic from the retired `TerritoryChoropleth` is absorbed here.
  `insufficient` sectors render **grey** (matching BRGM's grey "absence de point
  de suivi").
- **Arrows**: a symbol layer placed at the parsed `tendancy_coord` ("lat lon",
  fallback = polygon centroid), with three icons hausse ↑ / stable → / baisse ↓
  chosen by `trend`.
- **Interaction**: clicking a sector applies the existing **spatial filter**
  (stations inside) and shows a small verdict popup (class + trend +
  `pct_below_normal`, salvaged from `NationalBanner` content).
- **Legend**: when the layer is active, show the 7 classes + grey + the three
  arrow meanings.
- **Type**: respects the existing piezo/hydro context; piezo is the intended use.

## 8. Component 6 — Time slider replays sectors

The existing `TimelineSlider` already drives `timelinePeriodIndex` in
`ObservatoryPage`.

- When the sector layer is active and a past month is selected, recolor sectors
  from `sectors/timeline` for that period — the exact mirror of how station
  `displayFeatures` is recolored today.
- Arrows during replay use the per-period `trends[sector_id]`.
- The **station** timeline is switched to read `gold.fct_monthly_index` (Component
  1), so stations and sectors move together under one method.

## 9. Component 7 — Retire `/meteo`

- Remove the `MeteoNappesPage` route; redirect `/meteo` → `/observatoire`.
- Salvage into the sector layer: choropleth coloring + arrows. Delete
  `MeteoNappesPage`, `TerritoryChoropleth`, `TerritoryRanking`, `NationalBanner`,
  `StationDrillTable`, `OutlookPanel`. The `Outlook` contract returns to the
  forecast spec.
- Delete now-unused situation endpoints' region/department-only assumptions only
  if nothing else consumes them; otherwise keep `level=region|department` and just
  **add** `level=sector` (safer — verify consumers before deleting).
- Update the nav entry.

## 10. Edge cases & error handling

- Sector with `< MIN_ELIGIBLE` eligible stations → `insufficient` → grey.
- Stations outside all sectors (DROM, offshore) → excluded from aggregation, still
  shown as markers. BRGM WFS sectors are metropole+Corse → **no DROM sector
  choropleth** (same accepted limitation as the prior `/meteo`).
- Invalid/empty `tendancy_coord` → polygon centroid fallback.
- BRGM WFS unreachable at build time → keep the committed geojson; geometry is
  static, never fetched at runtime.
- Sectors are BRGM's; if they re-sectorize, re-run the build script.

## 11. Testing

- **Warehouse**: `fct_monthly_index` re-score matches `reference.value_to_zscore`
  for sampled `(station, month)`; eligibility flag carried correctly.
- **Backend**: point-in-polygon maps a known station to its known sector; sector
  aggregation reuses and passes the `territory_situation` test patterns; endpoint
  response shapes; `month=` past path reads `fct_monthly_index`.
- **Frontend**: sector layer renders and toggles; `tendancy_coord` parse +
  centroid fallback; slider recolors sectors per period; `/meteo` redirects.

## 12. Out of scope (explicit)

- **Forecast / anticipation** (Outlook 6-month, weather scenarios) — separate
  spec; the `Outlook` contract is removed here and returns there.
- **Child sub-sectors** on zoom (BRGM's `is_parent=false`) — v1 = 66 parents.
- **DROM** sector choropleth.
- **Canonical BRGM sector names** (auth-gated) — we use BDLISA-derived names.
