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

## Backend

### Warehouse (dbt, cross-repo) — precomputed so requests stay fast

1. **`gold.int_era5_temp_monthly`** — cell × `year_month` × `mean_temp_c`
   (monthly mean of `temperature_2m` from `gold.int_era5_for_all_stations`).
   One row per (latitude, longitude, month). ~4,500 cells × ~900 months.
2. **`gold.int_era5_temp_window_normals`** — cell × `window_months` (1,3,6,12) ×
   `end_calendar_month` (1..12) × `normal_mean_c`. Computed from
   `int_era5_temp_monthly`: build the N-month rolling mean per cell per
   year-month, then average those rolling means across all years grouped by the
   window's ending calendar month.

> Cross-repo note: these two models live in the dbt warehouse repo and are
> materialised by the existing dagster/dbt pipeline (consistent with the IPS
> fixed-reference precedent). The app only reads them. If warehouse changes must
> be deferred, the same numbers can be computed on the fly in the endpoint SQL
> and cached 24h — slower first hit, identical result — but the dbt models are
> the intended path.

### Endpoint (`api/routers/observatory_era5.py`)

- `GET /observatory/era5/temp-anomaly?date=YYYY-MM-DD&window=N`
  (`window` ∈ {1,3,6,12}; `date` optional → latest available month) →
  `[{latitude, longitude, anomaly_c}]`, one row per cell that has a full
  N-month window of data ending at `date`'s month.
  Query: window mean from `int_era5_temp_monthly` over the N months ending at
  `month(date)`, minus `int_era5_temp_window_normals` for
  `(window=N, end_calendar_month=month(date))`. Wrapped in `get_cached(...)`,
  TTL 86400; `get_brgm_sync_engine()` with `finally: pass`.
- Cells lacking a complete N-month window (early in the record, or sparse cells)
  are omitted.

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

- **Warehouse:** dbt tests — `int_era5_temp_monthly` non-null `mean_temp_c`, one
  row per cell-month; `int_era5_temp_window_normals` has all 4 windows × 12
  ending months per cell; spot-check one cell's N=3 normal by hand.
- **Backend:** verify `/temp-anomaly` via curl for a known date+window (non-empty
  array, plausible anomaly range); confirm default-latest-month.
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
