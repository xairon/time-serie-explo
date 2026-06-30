# ERA5 weather module v2 — anomalies-first redesign

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending spec review
**Supersedes the framing of:** the raw-variable-first ERA5 overlay (kept as a secondary "Contexte brut").

## Why

User feedback: the module wasn't convincing. Root critiques (with evidence):
- **Scales don't match the temporal aggregation** — fixed daily scales (precip 0–50 mm) but monthly aggregates run 18–184 mm → the monthly/timeline map saturates and reads as broken.
- **Unclear purpose / too many knobs**; raw temp/precip/ETP are weak defaults for a groundwater audience.
- **UI scattered**, legend hidden in the drawer, no on-map indication of "what am I looking at".
- **Rendering conflicts** — blocky 0.1° squares vs zone choropleth; by-zone *hijacks* the active admin layer's colour (dual meaning); grid blankets stations/zones.

Decision: **reframe the module around climate anomalies** (departures from the 1950+ normal), make an anomaly the default, demote raw values, and fix scales, UI, and rendering.

## Decisions (from brainstorm)

1. **Primary = anomalies vs 1950+ normal.** Default view: **precipitation anomaly, 3-month window, by zone.**
   - **Precipitation anomaly in %** of the normal (−40 % deficit, +30 % surplus).
   - **Temperature anomaly in °C** (already built).
   - Windows 1 / 3 / 6 / 12 months for both.
2. **Raw temp/precip/ETP = secondary** ("Contexte brut", collapsed), kept but not default, with corrected scales.
3. **Scales:** anomalies = divergent, centred on 0 (temp ±5 °C; precip ±80 %). Raw = domain adapted to the temporal granularity (daily vs monthly) so it no longer saturates. Legend always shows real bounds + 0 for divergent.
4. **Rendering:** one primary = **by-zone choropleth on a dedicated weather layer** (NOT by recolouring the admin layer). When weather-by-zone is active, dim the admin layer's own tint to avoid duplication. Raw 0.1° grid remains available as an option; smoothed isobands deferred.
5. **UI:** single clear activation; affirmed default; an **on-map banner** ("Anomalie pluie · 3 mois · mars 2024") + compact legend visible whenever weather is active; raw under an "advanced" disclosure.

## Backend (`api/routers/observatory_era5.py` + pure helpers)

Generalise the anomaly machinery to temperature AND precipitation.

- **Endpoint:** `GET /observatory/era5/anomaly?variable=temperature|precipitation&window=N&date=YYYY-MM-DD`
  → `[{latitude, longitude, anomaly}]` where `anomaly` is °C (temperature) or % (precipitation). `date` optional → latest complete month. Keep the existing `/temp-anomaly` working (or make it an alias `variable=temperature`).
- **Temperature** (existing logic): window mean = equal-weighted mean of the N monthly means; normal = mean of the per-calendar-month climatology means; `anomaly = mean − normal` (°C).
- **Precipitation** (new): 
  - climatology_precip(cell, calendar_month) = mean over all years of the **monthly SUM** of precipitation for that month.
  - normal(window) = Σ over the N ending calendar months of climatology_precip.
  - observed(window) = SUM of precipitation over `[win_start, win_end)`.
  - `anomaly_pct = (observed − normal) / normal * 100` (omit cell if `normal <= 0`).
- Two cached climatologies (`obs_era5_temp_climatology` already exists; add `obs_era5_precip_climatology`), same single-flight + 7-day TTL + startup pre-warm pattern. Source = `gold.era5_grid`, filter on `"time"`.
- Pure helpers in `api/era5_anomaly.py`: extend `compute_anomalies` (or add `compute_precip_anomalies`) — pure, unit-tested (window completeness, normal=Σ monthly normals, %=(obs−normal)/normal*100, normal≤0 dropped). Reuse `window_end_months`, `latest_complete_month`.

## Frontend

### Values & colours (`era5-colors.ts`)
- Variables reorganised into **primary** (`precipAnomaly` %, `tempAnomaly` °C) and **secondary/raw** (`temperature`, `precipitation`, `evaporation`).
- `precipAnomaly`: divergent scale centred 0, domain ≈ −80 → +80 %, dry = brown/red, wet = blue. `tempAnomaly`: existing divergent ±5 °C.
- **Raw scale fix:** raw variables get a domain that depends on the active temporal granularity (daily vs monthly), OR a dynamic domain computed from the current data's quantiles; the legend reflects the real bounds. (Fixes the precip-monthly saturation.)
- Legend helpers reused (`era5GradientCss`, divergent 0-centre).

### Data (`hooks`, `observatory-api`, `types`)
- `useERA5Anomaly(variable, date, window, enabled)` → `ERA5AnomalyPoint[]` with a generic `anomaly` value; `era5.anomaly(variable, date, window)` client.
- Keep `useERA5Snapshot`/`useERA5Monthly` for raw (secondary).

### Rendering (`ObservatoryMap.tsx`)
- **Dedicated weather choropleth layer:** when by-zone weather is active, build a separate GeoJSON source from the active zone's polygons coloured by the per-zone aggregated anomaly (reuse `aggregateEra5ByZone` + a divergent expression), as its OWN fill layer below stations (opacity ~0.7) — do NOT recolour the admin `*-fill` layer. Dim the admin layer's own fill-opacity (e.g. to ~0.05) while weather-by-zone is active, restore on off.
- Raw 0.1° grid remains an option (existing `era5-grid-fill`), used when by-zone is off.
- Keep the click isolation (weather is colour/info only; no station mutation).

### UI (`RightDrawer.tsx` + a new on-map banner component)
- ERA5 panel: variable group shows **Anomalie pluie / Anomalie temp** as primary; a collapsible **"Contexte brut"** holds raw temp/precip/ETP. Window selector (1/3/6/12). By-zone default on.
- **Default on activation:** `precipAnomaly`, window 3, by-zone.
- **On-map banner** (new small component, shown when weather active): current variable label + window + period (timeline month or picker) + a compact gradient legend with bounds and 0-centre. Replaces relying on the drawer-only legend.
- **Single activation clarity:** the timeline "Météo" checkbox and the drawer toggle reflect one state (`era5Active`); label them consistently; the banner makes the active state obvious.

## i18n
French keys for the new variable labels (Anomalie de précipitations, Anomalie thermique), "Contexte brut", banner strings; English equivalents.

## Edge cases
- Precip normal ≤ 0 (arid/edge cells) → omit cell.
- Anomaly window incomplete (n_months < window) → omit (existing guard).
- Daily raw scale vs monthly raw scale — pick domain by granularity.
- Climatology cold scans (~minutes each, ×2 now) — covered by startup pre-warm + single-flight + 7-day cache; the two warms run in background.

## Testing
- Backend: pytest for `compute_precip_anomalies` (%, normal=Σ, normal≤0 dropped, window completeness); curl that `/anomaly?variable=precipitation` returns ~11,496 cells with plausible % range.
- Frontend (vitest): precipAnomaly in the colour model (divergent, 0 stop), `era5GradientCss` for it, format as signed %.
- Manual: default = precip anomaly 3-mo by-zone with a clear banner+legend; switching to temp anomaly / raw works; scales no longer saturate; weather choropleth doesn't fight the admin layer.

## Out of scope (for v2)
- Smoothed isobands rendering (later).
- SPI/SPEI proper (this is a simpler %-anomaly; can evolve to SPI later).
- ETP anomaly (raw ETP stays as secondary context only).
