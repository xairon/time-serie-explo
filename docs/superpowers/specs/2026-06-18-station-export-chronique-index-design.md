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

**Header block** — comment lines prefixed with `#` (parseable with pandas
`comment='#'`, readable as-is in Excel). This is the "station" part:

```
# Station: <nom> (<code>)
# Commune: <nom_commune> (<code_departement>)
# Entité hydrogéo: <libelle_eh>          # piezo only
# Coordonnées: <lat>, <lon>
# Période exportée: <date_min> → <date_max>
# Index: <IPS|SSFI> (réf. fixe 1991-2020, flag=<normale|adaptee|provisoire>)
# Unités: niveau en m NGF / débit en m³/s ; z-score sans unité
# Source: Junon / Hub'Eau + BRGM — généré le <YYYY-MM-DD>
```

The `flag` shown in the header is the station's reference flag (same value carried in the
per-row `*_flag` column; taken from the index rows — if they vary, the header reflects the
most recent month's flag).

**Data table** — one row per day.

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

- Empty chronique → CSV with header block and zero data rows (HTTP 200, not an error).
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
