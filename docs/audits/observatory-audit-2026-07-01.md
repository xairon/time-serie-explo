# Observatory Audit Report — `feat/era5-sti`

_Scope: FastAPI backend (`api/`) + React/TS frontend (`frontend/src/`) for the Junon Observatoire — interactive map, Météo des nappes (BRGM sectors), and ERA5 climate grids incl. the Standardized Temperature Index (STI). Findings below are the verified survivors of a multi-agent audit; duplicates reported across dimensions have been merged._

## 1. Executive summary

The Observatoire is broadly sound, but two classes of risk stand out. First, one **high-severity crash path**: the Météo *situation* endpoints feed `quantile_grid` to `np.interp` without the JSON-string guard that all four sibling readers use, so `/situation/territories`, `/situation/national`, and current-month `/situation/sectors` can 500. Second, a cluster of **reference-period / state-coupling correctness issues** that map directly onto the project's stated priorities: the default STI view can compute its z-score over a *partial* ending month against a complete-month reference (biased index on the most-visible month), hiding the piezo marker layer silently repaints every sector verdict with hydro data ("no implicit filtering" violation), and the flagship STI legend in the drawer is mislabelled. The remaining items are localized UX/i18n/perf polish. No franglais was found in live strings, but a latent duplicate STI label set is a consistency trap.

---

## 2. Findings by theme

Severity badges: `[HIGH]` `[MED]` `[LOW]`. Verdicts: **C**onfirmed / **P**lausible.

### Météo des nappes (sectors, situation, popups)

**[HIGH] · C · `api/routers/observatory_situation.py:91` (and `:119`)** — Missing JSON-string guard on `quantile_grid` crashes situation endpoints.
`z_lag = value_to_zscore(float(r["lag_value"]), list(r["lag_grid"]))` passes `gold.station_reference_stats.quantile_grid` straight into `list(...)`. All four other readers normalize first (`observatory_piezo.py:362/643`, `observatory_hydro.py:412/641`: `if isinstance(g, str): g = json.loads(g)`); here there is none. When the driver returns the grid as a JSON string, `list("[1.0, ...]")` yields `['[','1','.',...]` and `np.interp` raises `ValueError: could not convert string to float: '['` (reproduced). The `and r["lag_grid"]` truthiness check does not help — a non-empty string is truthy.
- **Impact:** Uncaught 500 on `GET /situation/territories`, `/situation/national`, and current-month `/situation/sectors` — core Observatoire surfaces.
- **Fix:** Add a `_coerce_grid` helper mirroring the other routers (`if isinstance(g, str): g = json.loads(g)`), apply at both `:91` and `:119`.

**[MED] · C · `frontend/src/pages/MeteoNappesPage.tsx:95`** — Historical-month SectorPopup shows *latest-month* metrics.
`displaySectorSituation` recolours sectors from the timeline payload (`SectorTimeline`, `observatory-types.ts:391-395`) which carries only class-index + trend arrays. The mapper spreads `...s` (latest-month base) and overrides only `situation_class`, `trend`, `insufficient`; `pct_below_normal` / `n_eligible` / `n_provisoire` stay pinned to the current month and are rendered by SectorPopup (`{metrics.pctBelowNormal.toFixed(0)} % sous la normale`, station counts).
- **Impact:** For any past month, a historical class label (e.g. "Très bas" 2015) is shown alongside the *current* month's "% sous la normale" and counts — an internally inconsistent verdict.
- **Fix:** When `effectivePeriod !== latest`, null out / omit those three metrics in the mapped object (or extend the timeline endpoint to carry per-period metrics).

**[MED] · C · `frontend/src/components/meteo/MeteoMap.tsx:81`** — Sector choropleth fill (opacity 0.6) stacked *above* station markers.
`addStationLayers` runs at load (`:74`); `addSectorLayers` runs after the async secteurs fetch and `addLayer`s `secteurs-fill` with **no `beforeId`** (`layers/sectors-layer.ts:30-33`), so the 0.6-opacity fill paints over the badge/glyph markers (visible from zoom 7, exactly where sectors show). Opposite of ObservatoryMap, which inserts admin polygons `beforeId:'piezo-clusters'`.
- **Impact:** Station markers tinted/washed out; the fill also captures clicks over stations.
- **Fix:** Pass a `beforeId` (e.g. `'piezo-badge'`) to the sector `addLayer` calls, or re-add station layers after sectors so markers stay on top. (Note: click ambiguity is z-order-independent; both layers register handlers.)

**[LOW] · C · `frontend/src/pages/MeteoNappesPage.tsx:98`** — Trend arrow rendered on "Données insuffisantes" sectors.
For insufficient sectors class is forced to `null`/grey (`:222-224`) but `trend` is still `TR[ti]`; the timeline backend always appends trend code `0` → `'stable'` (`observatory_situation.py:294-296`). SectorPopup then shows a directional TrendBadge + "stable/en hausse/en baisse" next to a "no data" verdict.
- **Fix:** Null out `trend` when `insufficient` (and/or render "Inconnu" in SectorPopup when label is "Données insuffisantes").

**[LOW] · C · `frontend/src/components/meteo/StationPopup.tsx:35`** — Groundwater-only fallback label on débit (hydro) stations.
Shared piezo/hydro popup; when `classification` is null it uses the `UNKNOWN` key = `'Sans nappe libre étendue / Absence de point de suivi'` (`meteo-colors.ts:25`), which is nonsensical for a surface-water flow station. Reachable: hydro geojson returns every station regardless of index (`observatory_common.py:101` `COALESCE(...,'UNKNOWN')`).
- **Fix:** Station-type-aware fallback — neutral "Non classé / Donnée indisponible" for `type==='hydro'`, reserving the nappe wording for piezo (`selectedStation.type` is available).

**[LOW] · C · `frontend/src/components/meteo/MeteoMiniMap.tsx:51`** — Mini OSM map built twice on mount.
Creation effect deps `[collapsed, mainMap]`; `mainMap` is `null` on first render then flips to a `Map`, so React tears down mini-map #1 (`mini.remove()`) and builds #2 — two OSM instantiations/tile fetches per mount.
- **Fix:** Create once (drop `mainMap` from creation deps or gate on `mainMap != null`), draw the viewport rectangle from the separate move-sync effect.

---

### ERA5 / STI grids

**[MED] · C · `api/routers/observatory_era5.py:470`** — STI/anomaly for the latest month uses a *partial* ending month, bypassing `latest_complete_month`.
When a date is supplied, `month_start = DateType(d.year, d.month, 1)` with **no** completeness check; only the date-omitted branch guards via `latest_complete_month(d)`. But the frontend *always* supplies a date, seeded from raw `era5Range.max_date` (`ObservatoryPage.tsx:89,150`), which can be mid-month. `AVG(temperature_2m)` over the partial month still counts as one `date_trunc` group, so `compute_sti` doesn't drop it (`era5_anomaly.py:66`), while the reference is built exclusively from complete windows (`_era5_sti_reference` Step 4, `observatory_era5.py:395`).
- **Impact:** Biased z-score on the default/most-visible month — directly hits the reference-period-correctness priority. Same untreated pattern on precip `/anomaly` (`:552`), `_compute_temp_anomaly` (`:290`), `/temp-anomaly` (`:614`).
- **Fix:** Clamp the requested ending month to `latest_complete_month(max_time)` (`min(requested_month_start, latest)`), or require full day-coverage before including it. Apply at all four sites.

**[MED] · C · `frontend/src/components/observatory/RightDrawer.tsx:240-244`** _(merges the three duplicate reports at `:240/:242/:244`)_ — STI drawer legend shows a nonsensical "−10 … 35 σ" axis.
The legend renders for every ERA5 var including primary `tempStdIndex`. `isAnomalyVar` only matches `'anomaly'|'precipAnomaly'`, so STI falls into `era5RawDomain(... as 'temperature'|'precipitation'|'evaporation')`, which returns the temperature default `[-10, 35]` (`era5-colors.ts:194`). The gradient bar above uses `era5GradientCss('tempStdIndex')` (σ stops −2..+2), and the centred `0` marker is suppressed (only shown when `isAnomalyVar`).
- **Impact:** A −2..+2σ ramp mislabelled "−10 / 35" under unit "σ", no 0 centre — misleading legend on the flagship STI layer. `Era5Banner` (`Era5Banner.tsx:47-69`) correctly renders the discrete 7-class legend; the two surfaces disagree.
- **Fix:** Treat `tempStdIndex` as divergent (`scaleMin/Max` from its own stops → −2/+2, show `0`), or better, render the discrete STI class legend as Era5Banner does.

**[LOW] · C · `frontend/src/lib/era5-colors.ts:154` + `api/era5_anomaly.py:34`** _(merges the two duplicate reports)_ — `classifyIndex` warm-side boundaries diverge from backend despite "Mirrors the backend" comment.
Backend uses `lo <= z < hi` (z=0.84→HAUT, 1.28→TRES_HAUT, 1.75→EXTREMEMENT_HAUT); frontend uses inclusive `<=` on the warm side (z=0.84→NORMAL, etc.), keeping exact boundaries one class colder. Per-cell grid is unaffected (renders backend `index_class`), but the by-zone choropleth reclassifies the mean z client-side (`era5-zones.ts:65`), so at an exact boundary mean the two surfaces disagree by one colour.
- **Impact:** Essentially theoretical (float-equal boundary on an average). Comment is inaccurate; genuine data-contract inconsistency.
- **Fix:** Make warm-side comparators strict (`z < 0.84`, etc.) to match backend; update `era5-colors.test.ts:169-173`.

**[LOW] · C · `api/routers/observatory_era5.py:177`** — Single-flight climatology/STI-reference lock doesn't cover the cache write.
`fetch()` acquires `_climatology_lock`, double-checks `read_cached`, runs the ~130s scan, then returns — but `setex` happens later in `get_cached` (`dashboard/utils/cache.py:90`), *outside* the lock. A waiter can acquire the lock in the window before the write lands, see `read_cached()==None`, and re-run the full scan.
- **Impact:** Pure perf; worst realistic case is a couple redundant 130s scans, not N. Contradicts the single-flight comment (`:26-28`).
- **Fix:** `setex` the computed value under the same key inside the lock before returning.

---

### Calques / layers

**[LOW] · C · `frontend/src/components/observatory/ObservatoryMap.tsx:661` (and `:665`)** — ERA5 by-zone admin dimming clobbered when the active dept/bassin changes.
The by-zone effect dims the admin fill to 0.05 and records it in `era5DimmedFillIdRef`/`savedAdminOpacityRef` (`:849-854`). The "Sync active dept/bassin" effects (`:656-662`, `:665-669`) unconditionally rewrite `fill-opacity` back to the normal `0.30/0.25/0.10` case expression on `activeCodeDepartement/activeCodeBassin` change, never consulting the dim ref; the by-zone effect's deps (`:923`) exclude those codes, so it never re-dims.
- **Impact:** Changing active dept/basin while by-zone is on un-dims the admin fill, so the coloured admin layer visually competes with the choropleth (intended mutual-exclusion silently lost); `savedAdminOpacityRef` is left stale for later restore. Reachable via RightDrawer dept input / clear-filters / search-select (the map-click path is guarded). Cosmetic, recoverable.
- **Fix:** In the sync effects, skip the opacity rewrite when `era5DimmedFillIdRef.current === cfg.fillId`; or add the codes to the by-zone deps and refresh `savedAdminOpacityRef`.

**[LOW] · C · `api/routers/observatory_wfs.py:101`** — WFS proxy creates a fresh Redis `ConnectionPool` per request.
`get_wfs_layer()` builds a new `ConnectionPool.from_url(...)` + client on every hit (needs `decode_responses=False`, so it can't reuse the module `_pool`); the pool is a discarded local. Each hit opens/closes a fresh TCP connection to Redis rather than reusing a warm one.
- **Impact:** Connection churn per active vector overlay; bounded (refcounting closes promptly) and endpoint is 24h-cached over 8 whitelisted layers, so modest.
- **Fix:** Module-level `_wfs_pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=False)`, instantiate the client per request.

---

### Interactions & state

**[MED] · C · `frontend/src/pages/ObservatoryPage.tsx:121`** — Toggling piezo/hydro silently switches the sector choropleth data source. _(See §3.)_
`const sectorType = showHydro && !showPiezo ? 'hydro' : 'piezo'` feeds `useSectorSituation/useSectorTimeline(sectorType, ...)`; `type` is part of the query key (`useObservatory.ts:395-400`), so piezo IPS vs hydro SSFI are genuinely different aggregations. No explicit sector-data-type control exists in RightDrawer, and the sector legend never states which index is shown.
- **Impact:** Hiding the piezometer markers to declutter silently repaints every sector verdict with hydro data — a direct "toggling a layer must never silently alter the rest of the UI" violation. Trigger is specific (hide piezo while hydro stays on) but real.
- **Fix:** Decouple — add an explicit piezo/hydro selector for the sectors layer, or pin `sectorType='piezo'` unless the user opts into hydro sectors.

_(Also state-relevant: the by-zone dimming clobber, listed under Calques.)_

---

### Data contracts

- **[HIGH]** `quantile_grid` JSON-string crash — see Météo (`observatory_situation.py:91/119`).
- **[LOW] · C · `frontend/src/lib/observatory-api.ts:110`** — `era5.grid()` typed `ERA5GridPoint[]` (`latitude/longitude/...`) but backend `/grid` returns only `era5_latitude`/`era5_longitude` (`observatory_era5.py:42-58`, schema `observatory.py:330-332`). Reading `.latitude`/`.longitude` yields `undefined`. **Latent** — `useERA5Grid` (`useObservatory.ts:331`) has no consumers. **Fix:** give `grid()` a matching `{era5_latitude,era5_longitude}[]` type, or delete the unused hook/wrapper.
- **[LOW]** `classifyIndex` boundary divergence — see ERA5/STI.

---

### i18n

**[LOW] · C/P · `frontend/src/lib/era5-colors.ts:115`** _(merges the two duplicate reports)_ — Two divergent French label sets for the same 7 STI classes.
`STI_CLASS_LABELS_FR` (used by `era5FormatSti`) uses `EXTREMEMENT_BAS='Extrêmement froid' … HAUT='Chaud' … EXTREMEMENT_HAUT='Extrêmement chaud'`; the live i18n block `observatory.sti.*` (`fr.json:264-273`, rendered by `Era5Banner.tsx:64` and `ObservatoryMap.tsx:769`) is one notch milder — `EXTREMEMENT_BAS='Très froid' … HAUT='Doux' … EXTREMEMENT_HAUT='Très chaud'`.
- **Impact:** **Latent** — `era5FormatSti`/`STI_CLASS_LABELS_FR` are dead outside tests, so no live franglais today. But wiring the formatter into any tooltip would produce conflicting labels for the same class (e.g. TRES_HAUT = "Chaud" in the legend vs "Très chaud" in a tooltip), violating the UI-string-consistency priority.
- **Fix:** Delete the duplicate (and test-only `era5FormatSti`) or source its strings from `t('observatory.sti.<cls>')` — one canonical FR label per class.

_(All strings are French — no franglais detected in live UI. The `StationPopup` groundwater label under Météo is the one wording defect with real user visibility.)_

---

## 3. Interactions & layers — coupling / regression risks

The audit specifically probed how **Météo sectors**, **ERA5 grids**, and **calques** interact. Three real coupling issues surfaced, plus one clean spot worth noting:

- **Hidden coupling (regression risk): marker toggle → sector data source.** `ObservatoryPage.tsx:121` derives `sectorType` purely from the piezo/hydro visibility toggles. Turning piezo off (a decluttering action) silently switches the sector choropleth from IPS to SSFI aggregation — different numbers, different colours, no label, no opt-in. This is the exact pattern the project prohibits and the highest-priority interaction fix. Recommend an explicit sector-index selector.

- **ERA5 by-zone vs admin calque mutual-exclusion is fragile.** `ObservatoryMap.tsx:661/665` — the intended "dim the admin fill when the by-zone choropleth is active" behavior is silently undone whenever the active dept/basin changes via a non-map path (RightDrawer input, clear-filters, search). The dim and the sync effects don't know about each other. Cosmetic today, but it's a genuine hidden-state interaction and leaves a stale saved-opacity for the later restore.

- **Layer stacking regression in Météo.** `MeteoMap.tsx:81` adds the sector fill *above* the station markers (no `beforeId`), inverting ObservatoryMap's convention. Markers get washed out and clicks are ambiguous. This is a divergence between the two maps' layering discipline — worth standardizing.

- **STI grid vs zone classification can disagree at boundaries** (`era5-colors.ts:154` vs backend). Per-cell grid uses the backend class; the zone choropleth reclassifies the mean client-side with a different comparator convention. Practically measure-zero, but it means "the grid and the zone use the same classifier" is not actually true.

- **Clean:** the in-map department click handler *is* guarded against clicks landing on `era5-zone-fill` (`ObservatoryMap.tsx:486-489`), so the map-click path does not clobber the dim — the leak is only via the non-map paths above.

---

## 4. Prioritized action list (top 8)

1. **[HIGH]** Guard `quantile_grid` JSON strings in `observatory_situation.py:91` & `:119` (`_coerce_grid` helper) — stops 500s on `/situation/{territories,national,sectors}`.
2. **[MED]** Clamp the STI/anomaly ending month to `latest_complete_month` when a date is supplied — `observatory_era5.py:470` (+ `:552`, `:290`, `:614`). Fixes biased index on the default month; reference-period correctness.
3. **[MED]** Decouple sector `sectorType` from marker toggles — `ObservatoryPage.tsx:121`. Removes the prohibited implicit repaint.
4. **[MED]** Null out stale latest-month metrics in historical-month SectorPopup — `MeteoNappesPage.tsx:95`.
5. **[MED]** Fix the STI drawer legend — `RightDrawer.tsx:240-244` (use −2/+2 σ bounds + centred 0, or the discrete class legend).
6. **[MED]** Insert Météo sector fill beneath station markers via `beforeId` — `MeteoMap.tsx:81` / `sectors-layer.ts:30`.
7. **[LOW]** Preserve by-zone admin dimming across dept/basin changes — `ObservatoryMap.tsx:656-669`.
8. **[LOW]** Consolidate STI French labels to a single i18n source — `era5-colors.ts:115` — before the dead formatter is ever wired into the UI.

_Quick wins to batch alongside: station-type-aware `StationPopup` fallback (`StationPopup.tsx:35`), null trend on insufficient sectors (`MeteoNappesPage.tsx:98`), align `classifyIndex` comparators (`era5-colors.ts:154`), setex-under-lock (`observatory_era5.py:177`), module-level WFS Redis pool (`observatory_wfs.py:101`), MeteoMiniMap single-init (`MeteoMiniMap.tsx:51`), and the dormant `grid()` type mismatch (`observatory-api.ts:110`)._