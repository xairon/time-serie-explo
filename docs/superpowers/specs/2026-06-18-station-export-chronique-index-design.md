# Station export — chronique + standardized index (CSV)

**Date:** 2026-06-18
**Status:** approved by user (brainstorming session)

## Goal

Let a user download, from a single station's detail page, one CSV file that combines
**station metadata + daily chronique + standardized index** (IPS/SPLI for piezo, SSFI
for hydro). The index lives at monthly granularity (`gold.fct_monthly_index`); the
chronique is daily. The export carries the monthly index value forward onto every day
of its month, producing one row per day.

Audience: BRGM analysts / users who want the raw data of a station (level + its position
vs the 1991-2020 reference) in a spreadsheet or for downstream analysis.

## Scope

- **In:** single station (piezo OR hydro), triggered from the station detail page; full
  history; CSV only; generated server-side.
- **Out (YAGNI):** multi-station / bulk export, Excel (.xlsx), period filtering, custom
  column selection. A future `?from=&to=` range param is left as a possible extension,
  not built now.

## Approach (decided)

Dedicated backend endpoint that reads gold, joins monthly index onto daily rows in SQL,
and streams a CSV with `Content-Disposition`. The frontend is a plain download link
(cookie auth + `Content-Disposition`), mirroring the existing pastas export
(`api/routers/pastas.py` CSV export, `frontend/src/components/pastas/ExportMenu.tsx`).
No new dependency (pandas already present; CSV built in-memory).

## Backend

### Endpoints

- `GET /api/v1/observatory/piezo/stations/{code_bss}/export.csv`
- `GET /api/v1/observatory/hydro/stations/{code_station}/export.csv`

Added to `api/routers/observatory_piezo.py` and `observatory_hydro.py`, next to the
existing `daily` / `monthly` / `spli` / `ssfi` endpoints.

### Behaviour

1. Reuse the existing station-existence check → **404** if the station is unknown.
2. Read the **daily chronique**:
   - piezo: `gold.hubeau_daily_chroniques` (date, niveau_nappe_eau, profondeur_nappe,
     temperature_2m, total_precipitation, potential_evaporation)
   - hydro: `gold.hydro_daily_chroniques` (date, resultat_obs_elab, grandeur_hydro_elab,
     temperature_2m, total_precipitation, potential_evaporation)
3. Read the **monthly index** from `gold.fct_monthly_index` where `type` + `code` →
   `month`, `z`, `index_class`, `flag`.
4. **Month→day join (SQL):** each daily row joins on
   `date_trunc('month', d.date) = idx.month`. Days whose month has no index row get
   empty index cells (no fabrication). A `LEFT JOIN` from daily to index guarantees the
   full chronique is exported even where the index is missing.
5. Build the CSV in-memory (Python `csv` writer over a `StringIO`, consistent with the
   pastas export) and return a `Response`:
   - `media_type="text/csv; charset=utf-8"`
   - `Content-Disposition: attachment; filename="{code}_{today}.csv"`
     (sanitize `code` for filename: replace `/` etc.)

### CSV layout

> **Revised 2026-06-19** — the original design used a `#`-prefixed comment block before
> the header. That broke strict CSV parsers and, lacking a BOM, rendered as mojibake in
> Excel (`Département` → `DÃ©partement`). The shipped format is **tidy RFC-4180**: header on
> line 1, no comment block, station identity + provenance denormalized into columns repeated
> on every row, and the body encoded as **`utf-8-sig`** (UTF-8 + BOM) so Excel decodes
> accents correctly. Numeric values are rounded to 6 dp with trailing zeros stripped
> (`172.8800000000000000` → `172.88`).

**Identity columns** (repeated on every row, leading): `code`, `nom_station`,
`code_departement`, `nom_departement`, `codes_bdlisa` (piezo only — empty for hydro),
`latitude`, `longitude`.

**Provenance columns** (constant, trailing): `index_ref` (`1991-2020`), `unites`
(`niveau en m NGF ; z-score sans unité` / `débit en m³/s ; z-score sans unité`),
`source` (`Junon / Hub'Eau + BRGM`), `genere_le` (`<YYYY-MM-DD>`).

**Data table** — one row per day, between the identity and provenance blocks.

Piezo columns:

| Column | Source | Unit / note |
|---|---|---|
| `date` | daily | day |
| `niveau_nappe_eau` | daily | m NGF |
| `profondeur_nappe` | daily | m |
| `temperature_2m` | daily | °C (ERA5) |
| `total_precipitation` | daily | mm (ERA5) |
| `potential_evaporation` | daily | mm (ERA5) |
| `mois_ref` | join | month the index value belongs to (transparency) |
| `ips_z` | fct_monthly_index | z-score, repeated across the month's days |
| `ips_classe` | fct_monthly_index | 7 classes / UNKNOWN |
| `ips_flag` | fct_monthly_index | reference quality |

Hydro columns: same shape, with `resultat_obs_elab` (m³/s) + `grandeur_hydro_elab`
replacing the level columns, and `ssfi_z` / `ssfi_classe` / `ssfi_flag` replacing the
`ips_*` columns.

### Edge cases

- Empty chronique → CSV with the header row and zero data rows (HTTP 200, not an error).
- `gold.fct_monthly_index` missing (pre-materialization) → catch `ProgrammingError`,
  export the chronique with empty index columns (same resilience pattern as the existing
  `/spli` endpoint).
- Unit consistency: hydro daily `resultat_obs_elab` is already served in m³/s by the
  existing daily endpoint conversion; keep the same convention so the export matches the
  app's charts.

## Frontend

- Add an export action on `frontend/src/pages/StationPage.tsx`, next to the existing
  action buttons ("Add to Compare" / "Analyze in Pastas", around lines 106-110).
- Implement as a download link (`<a href>` / button that navigates) to the correct
  endpoint based on the current station type (piezo vs hydro). Cookie auth +
  `Content-Disposition` handle the download; no JS fetch, no client-side join.
- Optionally expose the endpoint URL via the `observatoryApi` layer
  (`frontend/src/lib/observatory-api.ts`) as a URL builder for consistency, even though
  the call itself is a plain link.

## Testing

Backend (pytest, following existing router tests):

1. Station with known daily chronique + monthly index → each day carries the `z`/class
   of the **correct month** (`mois_ref` matches `date_trunc('month', date)`).
2. Days in a month with no index row → empty index cells, but the daily rows are present.
3. Header block contains the station metadata lines.
4. Unknown station → 404.

Frontend (light):

5. The export button renders an `href` pointing to the right endpoint for the station's
   type.

## Files touched (anticipated)

- `api/routers/observatory_piezo.py` — new `export.csv` endpoint
- `api/routers/observatory_hydro.py` — new `export.csv` endpoint
- `api/` — a small shared CSV-building helper if the two endpoints share enough logic
  (e.g. `api/services/station_export.py`) to avoid duplication
- `frontend/src/pages/StationPage.tsx` — export button
- `frontend/src/lib/observatory-api.ts` — export URL builder (optional)
- `tests/` — backend export tests
