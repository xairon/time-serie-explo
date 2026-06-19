# Station CSV export — column selection

## Goal

Let users choose which columns the station CSV export contains, via group-level
toggles on the existing station page download. Mono-station only. Builds on the
existing `export.csv` endpoints and their date-range picker.

Non-goals (YAGNI): per-column selection, multi-station export, format choice
(only CSV), persisting the selection across sessions.

## Background

The station page (`frontend/src/pages/StationPage.tsx`) already has a date-range
picker and a download link (`<a href>`) pointing at:

```
/api/v1/observatory/{piezo|hydro}/stations/{code}/export.csv?start_date=…&end_date=…
```

The CSV is produced by the pure builder `dashboard/utils/station_export.py`
(`build_station_csv(domain, meta, daily_rows, index_rows)`). Columns fall into
five groups plus an always-present `date`:

| Group key     | Columns |
|---------------|---------|
| `identity`    | `code, nom_station, code_departement, nom_departement, codes_bdlisa, latitude, longitude` |
| `values`      | piezo: `niveau_nappe_eau, profondeur_nappe` · hydro: `resultat_obs_elab, grandeur_hydro_elab` |
| `meteo`       | `temperature_2m, total_precipitation, potential_evaporation` |
| `index`       | `mois_ref, {ips\|ssfi}_z, {ips\|ssfi}_classe, {ips\|ssfi}_flag` |
| `provenance`  | `index_ref, unites, source, genere_le` |

`date` is mandatory and never part of a group.

## Backend contract

Add one optional query param to both `export_csv` endpoints:

```
…/export.csv?…&groups=identity,values,meteo,index,provenance
```

- `groups` — comma-separated subset of the five group keys above.
- **Absent → all columns** (fully backward-compatible; the bare URL is unchanged).
- Unknown keys are ignored (intersect with the known group set).
- `date` is always emitted, independent of `groups`. If `groups` resolves to the
  empty set, the CSV contains only the `date` column — an acceptable edge case,
  not an error.

### Builder

Group definitions live in the builder (single source of truth — it already holds
`_IDENTITY_COLS`, `_VALUE_COLS`, `_METEO_COLS`, the index columns, and
`_PROVENANCE_HEADERS`). Signature becomes:

```python
build_station_csv(domain, meta, daily_rows, index_rows, groups=None)
```

- `groups`: an iterable of group keys, or `None` for all.
- A module-level `GROUP_KEYS` constant (ordered) is the canonical list, reused by
  the routers for validation/intersection.
- The header row and every data row include only `date` + the columns of the
  selected groups, in the existing fixed column order (identity → date → values →
  meteo → index → provenance). Selecting a subset never reorders columns.

### Routers

`observatory_piezo.export_csv` / `observatory_hydro.export_csv` gain a
`groups: Optional[str] = Query(None)` parameter. They split on commas, strip
blanks, intersect with `GROUP_KEYS`, and pass the result (or `None` when the
param is absent) to `build_station_csv`. No column logic in the routers.

## Frontend

- `observatory-api.ts`: `exportUrl(code, range, groups?)` appends `&groups=…`
  **only when** the selection is a strict subset of the five groups. When all five
  are selected, the param is omitted (keeps the URL clean; "all" ⇔ absent).
- New component `ColumnPicker` (e.g. `frontend/src/components/observatory/ColumnPicker.tsx`):
  a button labelled `Colonnes (n/5) ▾` next to the date inputs that opens a small
  popover with the five group checkboxes. Closes on outside click / Escape.
- `StationPage` holds the selected-groups state (default: all five selected) and
  feeds it to both the picker and `exportUrl`.
- Group labels are i18n keys (fr + en), e.g.
  `mainPages.station.exportColumns.{identity,values,meteo,index,provenance}` plus
  a `Colonnes` button label.

## Testing

- **Builder** (`tests/test_station_export.py`):
  - `groups={'values','index'}` → header is exactly `date` + value cols + index
    cols; identity/meteo/provenance absent; every data row matches.
  - `groups=None` → unchanged full output (existing tests stay green).
  - `groups=set()` → header and rows contain only `date`.
  - unknown key in `groups` is ignored.
- **Endpoint** (`tests/test_station_export_endpoint.py`): source assertions that
  both `export_csv` functions accept a `groups` param and pass it through, and
  that `GROUP_KEYS` is used for intersection.
- **Frontend** (`frontend/src/lib/observatory-api.test.ts`): `exportUrl` includes
  `groups=` for a strict subset and omits it when all five are selected.

## Files touched

- `dashboard/utils/station_export.py` — `groups` param + `GROUP_KEYS`.
- `api/routers/observatory_piezo.py`, `api/routers/observatory_hydro.py` — param wiring.
- `frontend/src/lib/observatory-api.ts` — `exportUrl` groups arg.
- `frontend/src/components/observatory/ColumnPicker.tsx` — new popover.
- `frontend/src/pages/StationPage.tsx` — state + wiring.
- `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json` — group labels under `mainPages.station.exportColumns`.
- Tests as listed above.
