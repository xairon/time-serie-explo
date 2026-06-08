# Météo des nappes — decision dashboard (design)

Date: 2026-06-08
Status: design — pending implementation plan
Audience: BRGM / public decision-makers (préfets, ARS, collectivités)

## 1. Goal

A public, at-a-glance "Météo des nappes" page that tells a decision-maker the
water situation **and where it is heading**, aggregated by territory, with a
forward-looking AI layer wired in but dark until a national forecast model
exists.

It absorbs the current flat `AlertsPage` (severity table), which becomes the
deepest drill level (stations of a department), not the landing view.

### Decisions locked during brainstorming

| Topic | Decision |
| --- | --- |
| Primary signal | Combined verdict: **current situation + trend** per territory |
| Territorial mesh | **Region → department → station** drill-down |
| Access | **Public** (like the Observatory), nav entry next to the map |
| Scope | **Piezo + hydro** (toggle between IPS and SSFI) |
| Novel angle | **AI anticipation** (1–3 month outlook) |
| Strategy | Robust base now + **progressive, honest AI layer** |
| v1 boundary | Full working base + AI layer **contract-defined but dark**; national ML is a separate follow-up chantier |

## 2. Page concept & layout

Top-to-bottom, from "glance" to detail:

1. **National banner (the glance).** One synthetic verdict for France: dominant
   situation class + trend arrow, plus 2–3 headline numbers ("X % des nappes
   sous la normale", "N départements en alerte", "tendance générale : en
   baisse"). **Nappes / Cours d'eau** toggle.
2. **Interactive choropleth map (the core).** Region mesh by default; each
   region colored by its situation class (existing 7-class palette
   `CLASSIFICATION_COLORS`) with a trend arrow overlaid. Click a region → zoom
   to its departments. Click a department → station list (reused alerts table).
   Reuses the Observatory Mapbox setup; region/department outlines come from a
   small bundled France TopoJSON.
3. **Ranking / side panel (the action).** Territories sorted most-critical
   first: situation, trend, and **persistence** (consecutive years in alert —
   data already available via `/alerts`).
4. **Anticipation layer (contract-defined, dark in v1).** A clearly labelled
   "Anticipation IA — bientôt" panel/column that will show the projected 1–3
   month situation per territory once a model exists. In v1 it renders a
   "coming soon" state and freezes the data contract so it can light up later
   without a rewrite.

**Design invariant:** any territory with insufficient reliable coverage renders
greyed as "données insuffisantes" — **never** a false verdict. Honesty is the
precondition for decision-maker trust.

## 3. Architecture

### 3.1 Backend (FastAPI)

New router `api/routers/observatory_situation.py` (kept separate so
`observatory_common.py`, already ~415 lines, does not keep growing):

- `GET /observatory/situation/national?type=piezo|hydro`
  → national headline verdict (`NationalSituation`).
- `GET /observatory/situation/territories?level=region|department&type=piezo|hydro`
  → list of `TerritorySituation`, one row per territory.
- Station drill reuses the existing `GET /observatory/alerts?code_departement=…`
  (already supports the department filter).

Supporting modules:

- `api/data/territories_fr.py` — static department→region lookup (all
  metropolitan + DROM departments). Pure Python, no DB.
- `dashboard/utils/territory_situation.py` — pure aggregation + trend helpers
  (no Streamlit, per repo convention), reusing `dashboard/utils/reference.py`
  to score monthly values against the fixed-reference grid.
- `api/schemas/observatory.py` — add `NationalSituation`, `TerritorySituation`,
  `Outlook` (nullable).

All endpoints use `get_cached(...)` with a long TTL (like `/stats/national`) and
are warmed at API startup (extend the lifespan warm-up in `api/main.py`).

### 3.2 Frontend (React)

- `frontend/src/pages/MeteoNappesPage.tsx` — orchestrates the layout and the
  piezo/hydro toggle and the region→department→station drill state.
- Components: `NationalBanner`, `TerritoryChoropleth`, `TerritoryRanking`,
  `OutlookPanel`, and `StationDrillTable` (the repurposed `AlertsPage` table).
- `frontend/src/lib/situation-api.ts` + types for the new endpoints.
- Bundled `france-regions-departments` TopoJSON (small, public boundaries).
- Route added to `routes.tsx` + a public nav entry.
- i18n keys added in FR (primary) and EN (parallel), reusing the hydro glossary
  terms and `CLASSIFICATION_COLORS`.

## 4. Aggregation methodology (must be defensible)

For a territory `T` and a given `type` (piezo→IPS, hydro→SSFI):

- **Eligible stations**: active stations in `T` with a non-null, non-`provisoire`
  fixed-reference index (`station_current_index.index_class` ∈ 7 classes, and
  reliability not `insuffisant`). `UNKNOWN`/`provisoire`/insufficient stations
  are **excluded from the verdict** but counted separately and surfaced in the
  tooltip.
- **Situation class**: take the **median `index_value`** of eligible stations and
  reclassify it against the fixed-reference class bounds. Median is robust to
  single-station outliers and trivially explainable.
- **Distribution**: count of stations per class (for transparency, shown on
  hover / in the ranking row).
- **Headline number**: **% of eligible stations below normal**
  (`BAS` + `TRES_BAS` + `EXTREMEMENT_BAS`).
- **Trend**: median, over eligible stations, of each station's index delta over
  the last ~3 months → `hausse` / `stable` / `baisse` using symmetric
  thresholds. Per-station index history is computed in-app by re-scoring recent
  monthly values (`gold.fct_monthly_chroniques` / `gold.fct_monthly_hydro`)
  against the fixed-reference grid via `reference.py`. (Alternative for later:
  materialize a `station_index_history` in the warehouse for performance; not
  needed for v1 given server-side caching.)
- **Coverage flag**: if the eligible-station count is below a threshold
  (default 3, tunable), the territory is `données insuffisantes` — no class, no
  trend.
- **Outlook** (nullable, contract only in v1):
  `{ horizon_months, situation_class, trend, confidence, coverage_pct } | null`.

National verdict = the same aggregation over all eligible stations nationwide.

## 5. Data flow

1. Page load → `GET /situation/national?type=piezo` and
   `GET /situation/territories?level=region&type=piezo` in parallel.
2. Region click → `GET /situation/territories?level=department&type=piezo`
   (fetched once, filtered client-side to the selected region's departments).
3. Department click → `GET /observatory/alerts?code_departement=XX` for the
   station list.
4. Piezo/Hydro toggle → re-query with `type=hydro` (cached server-side).

All responses are server-cached (long TTL, warmed at startup). The page reads
only aggregates until the user drills to a department.

## 6. Error handling

- Each panel (banner / map / ranking) loads and fails **independently**; one
  failing query does not blank the page.
- Insufficient-data territories render muted with an explicit label.
- `provisoire` / `UNKNOWN` stations are excluded from the verdict but reported
  in the territory tooltip ("N stations non classées").
- Outlook panel renders the "à venir" state whenever `outlook` is `null`.

## 7. Testing

- **Backend unit tests** (`tests/`): median→class reclassification; trend
  sign and threshold behavior; coverage gating (below-threshold → insufficient);
  department→region map completeness (every metro + DROM department mapped
  exactly once); national rollup equals aggregation over all stations.
- **Frontend tests**: national verdict renders class + arrow + headline numbers;
  choropleth color mapping per class; drill-down navigation
  (region → department → stations); piezo/hydro toggle; "données insuffisantes"
  and "anticipation à venir" states.

## 8. Scope

### In scope (v1)
- National banner, region/department choropleth with trend arrows, ranking,
  station drill (reused alerts table), piezo/hydro toggle, insufficient-data
  handling, AI outlook contract + dark UI, CSV export (reuse existing).

### Out of scope (follow-up chantiers)
- Real AI forecast computation + nightly inference asset (national ML chantier).
- Animated month-by-month timeline scrubber.
- Fused piezo+hydro single "water" index.
- Basin / aquifer (BDLISA) mesh.
- PDF export.

### Reused (not rebuilt)
- `CLASSIFICATION_COLORS`, Observatory Mapbox setup, `/observatory/alerts`,
  `dashboard/utils/reference.py`, `get_cached` + startup warm-up, i18n
  infrastructure, BRGM hydro glossary.

## 9. Open implementation notes

- Source a small, license-clean France region+department TopoJSON to bundle.
- Confirm `index_value` is exposed per station for the median computation (it is,
  via `station_current_index.index_value`, already returned by
  `/stations/geojson`).
- Trend cost: re-scoring recent monthly values nationwide is the heaviest query;
  validate it stays acceptable behind the long-TTL cache, else precompute.
