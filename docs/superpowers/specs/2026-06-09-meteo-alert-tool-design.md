# /meteo — Alert & exploration tool (design)

Date: 2026-06-09
Status: design — approved, pending implementation plan
Builds on: `2026-06-09-meteo-clone-design.md` (the faithful clone) — this refocuses it as an alert-first tool.

## 1. Goal

Turn the existing `/meteo` clone into an **alert-first situational tool** for the public / decision-makers, in the spirit of MétéEau des nappes: at a glance, *which groundwater sectors are low and where is it heading*, with temporal exploration as a secondary affordance. Declutter: no station markers by default.

### Decisions (locked)
| Topic | Decision |
| --- | --- |
| Primary role | **Alert-first** (grand public / décideur) |
| Stations | **Hidden by default**, optional toggle (kept like MétéEau) |
| Alert bricks | **National banner + critical-sectors list + visual highlight** of critical sectors |
| Search | Out of scope (for now) |
| Time slider | **IPS source only**; BRGM source shows the latest published window (no slider) |
| Data source | Keep the existing **BRGM | Notre IPS** toggle (default BRGM) |
| Backend | **No new endpoint** — reuse `/meteo/brgm-sectors`, `/situation/sectors`, `/situation/national`, `/situation/sectors/timeline` |

### "Critical / en alerte" definition (single shared constant)
The 3 driest classes: **`BAS`, `TRES_BAS`, `EXTREMEMENT_BAS`** (our enums) = BRGM classes 1–3 (modérément bas / bas / très bas). Define once in `lib/meteo-colors.ts` (e.g. `METEO_CRITICAL_CLASSES`) and reuse for banner counts, list filtering, and map highlight.

## 2. Layout

Overlays over the full-screen map (existing clone shell):
```
┌──────────────────────────────────────────────────────────┐
│ [MeteoNationalBanner]  top, full-width  (alert synthesis) │
│            [Source toggle BRGM|IPS]  top-center           │
│ [MeteoCriticalList]    left panel       <map: sectors>    │
│ [MeteoLayersPanel]     left (stations OFF)  [Legend] bl   │
│                                         [zoom] br         │
│ [SituationTimelineSlider]  bottom (only if source=IPS)    │
└──────────────────────────────────────────────────────────┘
```
Popups (sector/station) anchored top-right, below the banner.

## 3. Components

Building on `frontend/src/components/meteo/*` (clone). New + changed:

- **`MeteoNationalBanner.tsx`** (new). Props: `{ dominantClass: SituationClass | null; dominantClassLabel: string; trendSummary: string; criticalCount: number; totalSectors: number; pctBelowNormal?: number | null }`. Renders a full-width banner: colored dominant-class chip + label, "N/Total secteurs en alerte", a trend summary ("majorité en baisse"), and "X % des nappes sous la normale" when provided. Pure presentational.
- **`MeteoCriticalList.tsx`** (new). Props: `{ sectors: Array<{ code: string; name: string; classLabel: string; colorHex: string; trend: 'hausse'|'stable'|'baisse'|null }>; onSelect: (code: string) => void }`. Side panel (collapsible), title "Secteurs en alerte", rows sorted driest-first, each a colored dot + truncated name + trend arrow; click → `onSelect(code)`. Renders nothing (or "Aucun secteur en alerte") when empty.
- **`MeteoMap.tsx`** (modify):
  - Add a **`secteurs-alert-line`** line layer over `secteurs-fill`: visible only for sectors in `alertSectorIds` (a new prop `alertSectorIds: number[]`), rendered with a thick dark/red outline (e.g. `line-color #7f1d1d`, `line-width 2.2`) via a `filter: ['in', ['get','sector_id'], ['literal', alertSectorIds]]`. Toggles with the BSN layer.
  - Stations: default hidden — the page passes `visibleLayers.piezo/hydro/rain = false` initially (no MeteoMap change needed beyond honoring props; confirm initial state in the page).
  - Add a `flyToSector(sectorId)` capability: expose via an imperative handle OR accept a `focusSectorId` prop; on change, `fitBounds` to that sector's geometry (the map already has the geojson). Prefer a `focusSectorId?: number | null` prop + effect that fits bounds to the matching feature.
- **`MeteoNappesPage.tsx`** (modify): compute banner/list/alert data from the active source; default `visibleLayers` stations off; wire `MeteoNationalBanner`, `MeteoCriticalList`, `alertSectorIds`, `focusSectorId`. Keep source toggle, legend, slider (IPS only), popups.
- Unchanged: `MeteoLegend`, `SituationTimelineSlider`, `SectorPopup`, `StationPopup`, `MeteoLayersPanel` (stations now default off), `lib/situation-api.ts`, hooks.

## 4. Data flow (per source)

The page derives a single normalized per-sector view list, then computes banner + list + alert ids + colors from it.

- **BRGM source** (`useBrgmSectors`): array of `{sector_id, color, brgm_class, trend, ips, status, tendancy_coord}`.
  - `classEnum = brgmClassToEnum(brgm_class)` (existing map). `critical = METEO_CRITICAL_CLASSES.has(classEnum)`.
  - Banner: `dominantClass` = most frequent non-UNKNOWN classEnum; `criticalCount` = #critical; `trendSummary` from trend counts (e.g. majority of `baisse`); `pctBelowNormal` = `100 * criticalCount / sectorsWithData` (no separate national pct for BRGM — derive from class counts).
- **IPS source** (`useSectorSituation('piezo')` + `useSectorTimeline` for slider): per-sector `{code, name, situation_class, trend, pct_below_normal, n_eligible, insufficient}`; plus `useSectorNational?` — use the existing `/situation/national` (add a `useNationalSituation('piezo')` hook if not present) for `pctBelowNormal` + dominant class.
  - `critical = METEO_CRITICAL_CLASSES.has(situation_class)`.
- `alertSectorIds` = sector_ids whose classEnum/situation_class is critical (for the map highlight).
- `MeteoCriticalList.sectors` = critical sectors sorted by class severity (driest first; tie-break by name); name from the static geojson `nom` (the page already loads geojson features via `useStationsGeoJSON`? No — sector names: the page can fetch `/geo/secteurs-bsh.geojson` once for id→nom, or MeteoMap already has it; add a lightweight geojson fetch in the page for the id→nom map used by the banner/list/popup).

> Note: sector `name` currently comes from MeteoMap's click (geojson `nom`). For the list/banner we need id→nom without a click → the page fetches `/geo/secteurs-bsh.geojson` once (or import it) to build `Record<sector_id, nom>`.

## 5. Testing

- Unit (`lib/meteo-colors.ts`): `METEO_CRITICAL_CLASSES` membership; a pure `summarizeAlert(views)` helper (dominantClass, criticalCount, trendSummary) tested on a small fixture.
- Frontend: tsc + build green. Manual/Playwright: banner shows counts, critical list populated & sorted, critical sectors visually highlighted, stations hidden by default (toggle reveals), slider only on IPS, sector click → popup + list click → fly-to.

## 6. Out of scope
- Territory search, pluviomètres data, forecast/model layers, slider on the BRGM source, notifications/email alerts.

## 7. Files
- New: `frontend/src/components/meteo/MeteoNationalBanner.tsx`, `MeteoCriticalList.tsx`; helper `summarizeAlert` + `METEO_CRITICAL_CLASSES` in `lib/meteo-colors.ts` (+ test).
- Modify: `frontend/src/components/meteo/MeteoMap.tsx` (alert-line layer, focusSector), `frontend/src/pages/MeteoNappesPage.tsx` (wire banner/list/alert, stations off, id→nom map).
