# ERA5 Standardized Temperature Index (STI) — design

**Date:** 2026-06-30
**Status:** Approved (brainstorm), pending spec review
**Builds on:** the weather v2 anomalies module (already on `main`).

## Goal

Replace the raw-°C "anomalie thermique" primary view with a recognized **standard** temperature index — the **STI (Standardized Temperature Index)**: the SPI methodology applied to temperature (the temperature analogue of the project's SPI/SPLI/IPS). It answers "how hot/cold is this period *relative to the seasonal norm*", standardized and comparable across months and cells.

## Why STI (not a homemade index)

The project already computes SPI (precip) and IPS/SPLI (piezo) as standardized indices (empirical CDF → normal quantile, fixed 1991-2020 reference, 7 McKee/WMO classes). STI is the published temperature counterpart: same standardization, applied to temperature. User explicitly wants an **existing standard**, chose STI.

## Method (mirrors the project's SPI/IPS)

For a grid cell, window length `N` ∈ {1,3,6,12} months, ending at month `M`:
- **Observed** = mean 2m-temperature over the N-month window ending at `M` (the equal-weighted window mean already computed for the °C anomaly).
- **Reference distribution** = for each year `y` in **1991-2020**, the N-month-mean temperature for the window ending at calendar month `M` (wrapping the year boundary as needed). → μ and σ of those ~30 yearly window-means, per cell.
- **STI** = `(observed − μ) / σ`. (Gaussian STI — temperature is ~normal, so the standard-normal-quantile-of-empirical-CDF used by SPLI reduces to this z-score; we use the z-score form for simplicity and stability.)
- **Classification** = the SAME 7 McKee/WMO classes and z-thresholds the project uses for SPI/IPS (`dashboard/utils/drought.py` `_THRESHOLDS_7`): z cut points ±0.84, ±1.28, ±1.75 → EXTREMEMENT_BAS, TRES_BAS, BAS, NORMAL, HAUT, TRES_HAUT, EXTREMEMENT_HAUT.

Coordinates rounded to 0.1° (the doublon fix) throughout.

## Backend (`api/routers/observatory_era5.py` + `api/era5_anomaly.py`)

- **STI reference helper** `_era5_sti_reference(window, end_month)` → per cell `{lat, lon, mean, std, n_years}`: scan 1991-2020, compute per (cell, reference-year) the N-month window mean ending at `end_month`, then AVG and STDDEV_SAMP over years. SQL two-step (per-year window mean → μ,σ). Filter on `"time"`, round coords to 0.1° (SQL or Python merge consistent with the existing climatology). Cached `obs_era5_sti_ref` keyed by `{window, end_month}`, single-flight lock, TTL 7 days, pre-warmed at startup (for the default window=3 + latest end-month at least).
  - Year-boundary windows (e.g. N=3 ending Jan → Nov(y-1),Dec(y-1),Jan(y)): assign each window to its ending year; standard SPI convention.
- **Endpoint** `GET /observatory/era5/sti?window=N&date=YYYY-MM-DD` → `[{latitude, longitude, sti, index_class}]`. `date` optional → latest complete month. Computes observed window mean (reuse existing query), then `sti = (observed − μ)/σ` per cell (σ>0 guard; omit σ≤0), classifies via the shared 7-class thresholds. Pure helper `compute_sti(window_rows, reference, ...)` in `api/era5_anomaly.py` + `classify_index(z)` (the McKee thresholds, mirrored from drought.py) — unit-tested.
- Keep `/anomaly` (temp °C + precip %) as-is for the raw/precip paths.

## Frontend

- **New variable `tempStdIndex`** (label "Indice thermique (STI)") in `era5-colors.ts`, PRIMARY (replaces `anomaly`/temp-°C as the headline temperature variable; the `anomaly` temp-°C variable is removed from the primary group — precip anomaly stays).
- **7-class discrete colour scale**, temperature-oriented (cold→hot), reusing the McKee class names but inverted vs piezo:
  EXTREMEMENT_BAS `#313695` (indigo, very cold) · TRES_BAS `#4575b4` · BAS `#74add1` · NORMAL `#10b981` (green) · HAUT `#f46d43` · TRES_HAUT `#d73027` · EXTREMEMENT_HAUT `#7f0000` (dark red, very hot). (Or a continuous divergent z scale −2..+2 if classes are rejected — classes chosen.)
- **Data**: `ERA5StiPoint { latitude; longitude; sti: number | null; index_class: string | null }`; client `era5.sti(date, window)`; hook `useERA5Sti(date, window, enabled)`.
- **Rendering**: when `tempStdIndex` selected, the grid/by-zone colour by `index_class` (discrete class → colour) — a `match` on the class string (or interpolate on `sti`). By-zone aggregates the mean STI per zone then classifies, OR averages class — simpler: aggregate mean `sti` per zone (continuous) then classify the zone mean. Popup shows the class label + the z value + window. Banner shows variable + window + period + a discrete class legend.
- Window selector (1/3/6/12) applies. Date = month picker (anomaly-style).

## i18n
`era5VarStdIndex` = "Indice thermique (STI)"; class labels reuse the existing `CLASSIFICATION_LABELS` (très froid…très chaud — but note semantics: for temperature, BAS=froid, HAUT=chaud; use temperature-appropriate labels, e.g. a small STI label map: "Très froid / Froid / Frais / Normal / Doux / Chaud / Très chaud" — TBD-friendly, French). Banner/popup strings.

## Edge cases
- σ ≤ 0 or incomplete reference (cell with <N months or no reference years) → omit.
- Year-boundary windows handled by ending-year assignment.
- Reference cold scan (~minutes, 1991-2020 scope) → single-flight + 7-day cache + startup pre-warm (default window 3). Cached per (window, end-month).

## Testing
- pytest: `compute_sti` (z = (obs−μ)/σ, σ≤0 dropped, incomplete dropped) + `classify_index` (the 7 thresholds, boundary values) — pure.
- curl: `/sti?window=3` returns ~11,496 cells, spread of z (not all 0), plausible class distribution; latest-month default.
- vitest: `tempStdIndex` in the colour model (7 classes), class→colour mapping, format (z + class).

## Out of scope
- SPEI / precip STI (separate).
- Storing STI in the warehouse `fct_monthly_index` (app computes on the fly, like the rest).
- Empirical-CDF (percentile-grid) STI — the Gaussian z-score form is used (equivalent for near-normal temperature); can upgrade later.
