# Météo des nappes — faithful MétéEAU Nappes clone, Junon data (V1)

**Date:** 2026-06-10
**Status:** approved by user (brainstorming session)
**Supersedes:** `2026-06-09-meteo-clone-design.md` (first clone attempt), the alert-tool additions of `2026-06-09` (national banner / critical list), and the decision-dashboard concept of `2026-06-08`.

## Goal

Rebuild `/meteo` as a **faithful clone of the desktop MétéEAU Nappes app** (`app.meteeaunappes.brgm.fr/desktop`), powered **exclusively by Junon data** (`gold.fct_monthly_index`, fixed reference 1991-2020). Target audience: general public ("quel temps font les nappes chez moi ?"). UI in French.

V1 = the map experience only. The per-station dynamic sheet (chronicles + NHiTS forecasts as climate scenarios) is V2, out of scope here.

## What we keep (data layer is sound)

- Backend endpoints: `GET /observatory/situation/sectors[?month=]`, `/situation/sectors/timeline`, `/stations/geojson` — unchanged.
- Frontend hooks `useSectorSituation`, `useSectorTimeline`, `useStationsGeoJSON` (in `hooks/useObservatory.ts`).
- `lib/meteo-colors.ts` (BRGM-exact class palette + French labels) and `parseTendancyCoord`.
- Static `/geo/secteurs-bsh.geojson` (66 BSH sectors with `tendancy_coord`).
- The standalone full-screen route `/meteo` (no Junon TopNav) and its TopNav tab entry in the main app.

## What we remove

- **BRGM/IPS source toggle** and all frontend calls to `/observatory/meteo/brgm-*`. The WFS proxy endpoints stay in the backend for QA/recette comparison but are no longer called by the UI.
- **`MeteoNationalBanner`** and **`MeteoCriticalList`** components and their wiring (not present in the original).
- The full-history range slider (replaced by the original's rolling 12-month timeline + date picker).
- Dark map theme.

## UI specification (clone, element by element)

Reference screenshots taken 2026-06-10 from the live BRGM app are the source of truth for layout and styling. Typography: Roboto / Helvetica Neue, 11px legend text.

### Map

- Basemap: standard **OSM raster** light tiles (`tile.openstreetmap.org`), OSM attribution bottom-left.
- Choropleth of the 66 BSH sectors, fill = `METEO_CLASS_COLORS[index_class]`, semi-transparent (match original ≈0.6), thin darker outline.
- **Sector trend badges** (replaces the current black triangles): one per sector at `tendancy_coord` — a **18×18 px circle, `border-radius: 50%`, background `rgba(255,255,255,0.6)`, black icon at 8px font-size**. Icons (PrimeIcons glyphs or equivalent inline SVG):
  - en hausse → `arrow-up`
  - stable → `equals`
  - en baisse → `arrow-down`
  - inconnu → `question`
- Station markers (piezo, hydro) restyled to match the original's small colored dot markers; visible per layer toggle. Only one popup open at a time.

### Left panel (three white rounded cards, top-to-bottom)

1. **Type** — 5 rows with icons, French labels: Piézomètre, Source, Pluviomètre, Station de débit, Avec modèle. Rows for which Junon has no data (Source, Pluviomètre, Avec modèle) are rendered **greyed out / disabled** with tooltip « Données bientôt disponibles ». Piézomètre and Station de débit toggle their marker layers.
2. **Évolution des niveaux** — the 4 trend rows (grey circle icon + label): en hausse, stable, en baisse, inconnu.
3. **Niveau** — vertical 8-step scale: très haut, haut, modérément haut, autour de la moyenne, modérément bas, bas, très bas, « Pas de nappe libre étendue » (UNKNOWN colour).

Junon logo placed under the legend, bottom-left area of the map (where the original puts its own logo); clicking it returns to `/`.

### Search (top-left)

- Combobox placeholder « adresse, station, piézomètre, etc. ».
- Suggestions merge: address geocoding (BAN `api-adresse.data.gouv.fr`, France-only) + Junon stations matched by code or commune name.
- Selecting a suggestion flies the map to the location; a **« réinitialiser × » chip** appears top-center to restore the default France view and clear the search.

### Timeline (bottom)

- **Rolling 12-month window** of clickable month chips ending at the current month, plus the next 3 future months rendered greyed/disabled (forecast slots, V2). Year label shown under the chip where the year changes (e.g. « décembre 2025 | janvier 2026 »).
- Selected month = filled dot + bold label, as in the original.
- Bottom-left **date combobox** (« juin 2026 » + `×`): opens a month/year picker over the **full Junon history** (≈2000 → present). Choosing a date re-centers the 12-month window around it and refreshes the choropleth via `?month=YYYY-MM`. The `×` resets to the current month. This deep history is our added value over the original.
- Month change updates sectors (fill + badges) from the timeline payload without refetch when already loaded.

### Bottom-right controls

- Collapsible **minimap** (France overview with viewport rectangle), zoom `+`/`−`, scale bar.
- The original's territory selector (France / DOM) is **omitted** — metropolitan France only.
- The original's "save map position" and login are omitted (out of scope).

### Top-right

- **« À propos »** button → modal in French explaining the data: IPS computed by Junon on the fixed 1991-2020 reference, source piezometers, update cadence, and that sectors are BRGM BSH hydrogeological sectors.

### Popups

- **Sector click:** name, level class (label + colour chip), trend (icon + label), styled like the original's light popup.
- **Station click:** code, commune, current classification, last measurement date/value. No chart in V1.

## Architecture

```
pages/MeteoNappesPage.tsx        — slim orchestrator (selected month, layer toggles, search selection)
components/meteo/
  MeteoMap.tsx                   — map init + composition only (split from current 401-line file)
  layers/SectorsLayer.ts         — choropleth + trend badges (pure MapLibre layer module)
  layers/StationsLayer.ts        — station markers per type
  MeteoSearchBar.tsx             — combobox + BAN geocoding + station match + reset chip
  MeteoTypePanel.tsx             — Type card (replaces MeteoLayersPanel)
  MeteoLegend.tsx                — trend + level cards (restyled)
  MeteoTimeline.tsx              — 12-month window + date picker (replaces SituationTimelineSlider)
  MeteoMiniMap.tsx               — collapsible overview map
  AboutModal.tsx
  SectorPopup.tsx / StationPopup.tsx — restyled
lib/meteo-timeline.ts            — pure helpers: build 12-month window around a date, clamp to data range
```

Data flow unchanged: React Query hooks → props down. No new global state; selected month and toggles live in `MeteoNappesPage` state (URL query param `?month=` kept in sync for shareable links).

## Error handling

- Timeline/situation fetch failure → unobtrusive French toast + sectors rendered grey (UNKNOWN) rather than a blank map.
- BAN geocoding failure → suggestions list shows stations only.
- Months outside data coverage in the picker are disabled (derived from the timeline `periods` payload).

## Testing

- Unit tests (vitest) for pure helpers: `meteo-timeline.ts` window building (edges: current month, history start, year boundary), class→colour/label mapping, `parseTendancyCoord`.
- Component test: timeline renders 12 chips + greyed future months, click updates selection.
- Recette: side-by-side visual check against `app.meteeaunappes.brgm.fr/desktop` on the same month (layout, palette, badges) using our data.

## Out of scope (V1)

- Per-station dynamic sheet with chronicle charts and NHiTS forecast scenarios (**V2**, the future-month timeline slots are its landing zone).
- Rain stations, springs ("Source"), "Avec modèle" filter data.
- Overseas territories, login, saved map position, EN locale for this page.
