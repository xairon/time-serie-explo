# ERA5 temperature anomaly on the Observatoire — design

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending spec review
**Builds on:** Phase 0 (daily ERA5 grid) — `docs/superpowers/plans/2026-06-29-era5-weather-phase0-grid.md`

## Goal

Add a **temperature anomaly** mode to the ERA5 weather grid: for a selectable
multi-month window (1 / 3 / 6 / 12 months) ending at a chosen date, colour each
0.1° cell by its mean-temperature departure from the long-term normal, on a
divergent scale. The multi-month window doubles as the temporal aggregation the
user was missing.

## Decisions (from brainstorm)

- **Index:** temperature anomaly only (no SPI/SPEI, no precipitation index this
  phase — precipitation raw colours stay as in Phase 0).
- **Windows:** 1, 3, 6, 12 months. The window ends at the selected date's month.
- **Reference:** the full ERA5 record (1950 → present), i.e. the normal is the
  long-term mean over all available years (not the 1991-2020 climatological
  normal — explicit user choice).
- **Display:** keep Phase 0's coloured-squares rendering; anomaly uses a
  divergent blue↔red scale centred on 0.
- **Date control:** the existing independent ERA5 date selector (NOT coupled to
  the station history timeline — explicit user choice). The date is interpreted
  as the *end* of the window.

## Definitions

For window length `N` months ending at month `M` (the month of the selected date):

- **Window mean** for a cell = mean monthly temperature over the `N` months
  `[M-N+1 .. M]`.
- **Normal** for that cell = the long-term mean (over every year 1950→present)
  of the same `N`-month window that ends at calendar month `month(M)`.
  (e.g. for N=3 ending in March, average every Jan-Feb-Mar window across all
  years.)
- **Anomaly** = window mean − normal, in °C.

## Architecture overview

A new **coloring mode "Anomalie thermique"** added to the ERA5 panel's variable
selector (alongside Température / Précipitations / ETP from Phase 0). When
selected, a **window selector (1/3/6/12 mois)** appears, the legend switches to
a divergent scale, and the cells are coloured by the anomaly returned by a new
endpoint. All other Phase 0 behaviour (squares under the stations, click popup,
opt-in toggle, independent date) is unchanged.

## Backend — computed on the fly in the API, cached in Redis (no dbt, no warehouse schema change)

Everything is computed in SQL against `gold.int_era5_for_all_stations` (daily
values) inside the existing router, wrapped in `get_cached(...)`. To avoid
re-scanning the 126M-row table on every request, the work is split into a cheap
per-request part and an expensive-but-reused climatology part:

1. **Climatology (reused across all requests).** A helper
   `_era5_temp_climatology()` returns, per cell, the long-term mean temperature
   for each calendar month (cell × `month` 1..12 × `mean_c`), over the full
   record (1950+):
   ```sql
   SELECT latitude, longitude,
          EXTRACT(MONTH FROM era5_date)::int AS mo,
          AVG(temperature_2m) AS mean_c
   FROM gold.int_era5_for_all_stations
   GROUP BY latitude, longitude, EXTRACT(MONTH FROM era5_date)
   ```
   This is the only full-table scan; it is cached in Redis under a stable key
   (`obs_era5_temp_climatology`, TTL 7 days — climatology is effectively
   static). ~4,500 cells × 12 rows. First-ever call is a one-time cold cost
   (one sequential scan); every later request reuses the cached result.
   The N-month-window **normal** for a window ending at calendar month `m` is
   the mean of the climatology values for the `N` ending calendar months
   (computed in Python from the cached climatology — wrapping the year boundary
   as needed, e.g. N=3 ending in Jan → months {11,12,1}).

2. **Window mean (per request, cheap).** For the selected date `D` (month `M`)
   and window `N`, average the daily temperatures over the N-month window
   `[M-(N-1) months, M+1 month)` per cell — a small scan (N months of one
   period):
   ```sql
   SELECT latitude, longitude, AVG(temperature_2m) AS window_mean
   FROM gold.int_era5_for_all_stations
   WHERE era5_date >= :win_start AND era5_date < :win_end
   GROUP BY latitude, longitude
   ```

3. **Anomaly** = `window_mean − normal` per cell (joined on lat/lon in Python).

### Endpoint (`api/routers/observatory_era5.py`)

- `GET /observatory/era5/temp-anomaly?date=YYYY-MM-DD&window=N`
  (`window` ∈ {1,3,6,12}; `date` optional → latest available month) →
  `[{latitude, longitude, anomaly_c}]`, one row per cell that has both a
  window mean and a climatology normal. Wrapped in `get_cached(...)` keyed by
  `{date, window}`, TTL 86400; uses `get_brgm_sync_engine()` with
  `finally: pass`.
- The expensive climatology is fetched via its own `get_cached(...)` (longer
  TTL) so it is computed at most once per 7 days regardless of date/window.
- Cells lacking a complete N-month window or any climatology are omitted.

> Latency note: the very first `/temp-anomaly` call after a cache flush triggers
> the one full-table climatology scan; subsequent calls (any date/window) are
> cheap. If the cold scan ever exceeds the frontend's 30 s fetch timeout, the
> mitigation is a one-time pre-warm of `obs_era5_temp_climatology` (a startup
> ping), not a schema change.

## Frontend

### `era5-colors.ts` (extend)
- Add an `'anomaly'` entry to the variable model with `prop: 'anomaly_c'`,
  `unit: '°C'`, `labelKey: 'observatory.drawer.era5VarAnomaly'`, and a
  **divergent** stop set centred on 0, domain ≈ −5 → +5 °C (clamped):
  `[-5,'#2166ac'],[-2.5,'#67a9cf'],[-0.5,'#d1e5f0'],[0,'#f7f7f7'],[0.5,'#fddbc7'],[2.5,'#ef8a62'],[5,'#b2182b']`.
- `era5FormatValue('anomaly', v)` shows a signed value, e.g. `"+2.3 °C"` /
  `"−1.1 °C"` (explicit sign; null → "—").

### Data + types
- `observatory-types.ts`: `ERA5AnomalyPoint { latitude; longitude; anomaly_c: number | null }`.
- `observatory-api.ts`: `era5.tempAnomaly(date, window)` →
  `fetchJson<ERA5AnomalyPoint[]>('/observatory/era5/temp-anomaly', { date, window })`.
- `useObservatory.ts`: `useERA5TempAnomaly(date, window, enabled)`.

### `ObservatoryMap.tsx`
- The existing ERA5 effect must accept anomaly data. Simplest: when
  `era5Variable === 'anomaly'`, the squares are built from the anomaly points
  (property `anomaly_c`) and coloured with the divergent expression; otherwise
  unchanged. The square builder already carries arbitrary numeric props — pass
  the anomaly points through a small adapter so each feature has `anomaly_c`.
- The click popup, in anomaly mode, shows the anomaly value (signed °C) and the
  window (e.g. "Anomalie 3 mois : +2.3 °C").

### `ObservatoryPage.tsx`
- New state `era5Window` (default 3). When `era5Variable === 'anomaly'`, fetch
  via `useERA5TempAnomaly(era5Date, era5Window, era5Active)`; otherwise the
  Phase 0 snapshot path. Feed the chosen points to `ObservatoryMap`.

### `RightDrawer.tsx`
- Add "Anomalie thermique" as a fourth radio in the colour-by group.
- When anomaly is selected, show a window selector (1 / 3 / 6 / 12 mois) and the
  divergent legend (with a 0 midpoint label).

### i18n (`fr.json`)
- `era5VarAnomaly`: "Anomalie thermique", `era5Window`: "Fenêtre",
  `era5Window1`: "1 mois", `era5Window3`: "3 mois", `era5Window6`: "6 mois",
  `era5Window12`: "12 mois", popup label "Anomalie {{n}} mois".

## Edge cases

- **Incomplete window** (cell lacks N months ending at the date): omitted; if no
  cell qualifies, the existing "pas de données" path applies.
- **Window crossing year boundaries** (e.g. 12-month or Nov-ending 3-month): the
  rolling-mean construction in `int_era5_temp_window_normals` handles this
  natively; the endpoint just selects by ending month.
- **Divergent scale clamping:** anomalies beyond ±5 °C clamp to the end colours.
- **Sign display:** always show the sign in popups/legend; 0 reads neutral grey.

## Testing

- **Backend (pure helper, pytest):** the window-end → calendar-month-set logic
  (e.g. `window_end_months(end_month, N)` → wraps the year boundary: (1,3) →
  {11,12,1}) and the climatology→normal averaging are pure functions — unit-test
  them (the existing test suite favours pure-helper tests; these queries
  themselves are verified by curl, not unit tests).
- **Backend (integration curl):** `/temp-anomaly` for a known date+window returns
  a non-empty array with a plausible anomaly range; default-latest-month works;
  the climatology cache is reused (second call fast).
- **Frontend (vitest):** extend `era5-colors.test.ts` — anomaly is in the
  variable model, divergent expression reads `['to-number', ['get','anomaly_c']]`,
  `era5FormatValue('anomaly', 2.3) === '+2.3 °C'` and `(-1.1) === '−1.1 °C'`,
  null → '—'. Add an adapter test if the anomaly→squares adapter is a pure helper.
- **Manual:** toggle ERA5, pick "Anomalie thermique", switch windows 1/3/6/12,
  confirm divergent colouring + signed popup; stations stay visible.

## Out of scope (YAGNI)

- SPI / SPEI / precipitation indices (explicitly deferred by the user).
- Coupling the ERA5 date to the station history timeline (kept independent).
- Daily anomaly (windows are monthly).
- By-zone aggregation of the anomaly (the Phase 3 "par zone" work is separate).
- 1991-2020 reference (user chose full-record reference).
