# Station CSV export — optional date range

**Date:** 2026-06-18
**Status:** approved by user (brainstorming session)
**Extends:** `2026-06-18-station-export-chronique-index-design.md` (the export currently downloads the full history with no way to bound it).

## Goal

Let the user bound the station CSV export to a date range, instead of always
downloading the entire history. Empty range = full history (current behaviour,
unchanged).

## Backend

Add two optional query params to BOTH export endpoints, mirroring the existing
`/daily` endpoint's signature exactly (same names, same style):

```python
start_date: Optional[date] = Query(None),
end_date: Optional[date] = Query(None),
```

- `api/routers/observatory_piezo.py` `export_csv`
- `api/routers/observatory_hydro.py` `export_csv`

Apply them to the **daily chronique** query only, appended like `/daily` does:
`... AND date >= :start_date` / `... AND date <= :end_date`, binding the params
only when not None. The monthly-index query is left unfiltered (per-station index
history is small; the pure builder only emits index values for days actually
present, so the range propagates automatically — including the `# Période exportée`
header line, which already derives from the first/last daily row).

No start>end validation (parity with `/daily`: an inverted range simply yields an
empty data section). 404-on-unknown-station and the `ProgrammingError` index-table
guard are unchanged.

## Frontend

- `frontend/src/lib/observatory-api.ts`: extend the `exportUrl` builders to accept
  an optional `params?: { start_date?: string; end_date?: string }` and append a
  query string (omit each param when empty/undefined; `encodeURIComponent` values).
  Bare call (no params) must still produce the exact current URL.
- `frontend/src/pages/StationPage.tsx`: add two `<input type="date">` (start / end)
  next to the existing "Exporter (CSV)" link, with local state `exportStart` /
  `exportEnd`. The link's `href` becomes
  `exportUrl(code, { start_date: exportStart, end_date: exportEnd })`. Empty inputs
  → params omitted → full history.

## Testing

- Backend (source-level, like the existing export endpoint tests): assert each
  `export_csv` source accepts `start_date`/`end_date` and applies `date >= :start_date`
  / `date <= :end_date` to the daily query.
- Frontend (vitest): `exportUrl` with `{ start_date, end_date }` builds
  `...?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`; `exportUrl` with no params builds
  the bare `.../export.csv` URL unchanged.

## Out of scope (YAGNI)

Presets ("last 12 months"…), start>end validation, filtering the index query.
