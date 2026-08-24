# The Observatory

The Observatory is a container with two sub-tabs, one per compartment (`ObservatoryShell` +
`ObservatoryTabs`, both mounted under a pathless layout route so the URLs stay unchanged).

## Groundwater & rivers (`/`)

- Spatial exploration of piezometric and hydrometric stations
- BDLISA aquifer overlays, BSH sector layer, drought indices (SPI, SPLI, SSFI)
- Cross-station comparison with persistent selection

## Climat (`/climat`)

Three views:

| View | What it shows |
|------|---------------|
| **Situation** | France-wide 0.1° grid map |
| **Point/Zone** | Per-cell history from 1950, precipitation vs normal, multi-window index chart, drought-episode table, CSV export |
| **Comparaison** | Multi-year rainfall overlays, SPI small multiples |

### Map layers

| Family | Layers |
|--------|--------|
| Standardized indices | **SPI** (precipitation), **STI** (temperature), **SPEI** (precipitation − PET) — all on the 7-class McKee/WMO scale |
| Water balance | `bilan_hydrique` (mm), discrete classes, same palette |
| Daily | `tmax` / `tmin` / `tmean` on a fixed absolute °C scale, and daily rainfall classes |

Absolute monthly precipitation, temperature and PET are deliberately **not** map layers. The
project rule is: *either a real standardized indicator, or a real value as a number* — never
an invented in-between. They appear as exact figures in the point panel instead.

The Climat tab is reachable from the groundwater map through a cell-popup deep link, and from
the station page (local SPI, rolling cumulative rainfall).

## API

All endpoints live under `/api/v1/observatory/climat/*` (`api/routers/observatory_climat.py`,
mounted at `api/main.py:146`). They are plain `SELECT`s — no statistics computed on the fly —
over the warehouse's precomputed ERA5 grid marts, and are Redis-cached for 24 hours.

| Endpoint | Purpose |
|----------|---------|
| `range` | Available month bounds |
| `grid-monthly` | Monthly grid values |
| `grid-indices` | Per-cell SPI / STI / SPEI |
| `daily-temp`, `daily-temp-range` | Daily temperature grid |
| `daily-precip`, `daily-precip-range` | Daily precipitation grid |
| `point-series` | Per-cell history |
| `point-episodes` | Drought episodes at the nearest cell |
| `compare-years` | Multi-year comparison |
| `export-point.csv` | CSV export |

Source marts, all in the `hubeau_data_integration` warehouse: `gold.fct_era5_monthly_grid`,
`gold.fct_era5_climatology_grid`, `gold.fct_era5_spei_climatology_grid`,
`gold.fct_era5_indices_grid`.

### Accepted index values

`grid-indices?index=` accepts `spi`, `sti` or `spei`. `point-episodes?index=` accepts `spi` or
`spei` only — episodes are drought-only, never STI. Both reject anything else with a 422
raised **before** the value reaches SQL (`_assert_index` / `_assert_episode_index`).

### Where the default month comes from

The default month and the `MonthStepper` bounds come from
`GET /observatory/climat/range` — `max_indices_month`, `max_monthly_complete_month`,
`max_monthly_month`, `min_month` — and **not** from `/observatory/era5/range`, which describes
the daily grid. The daily grid's maximum is the current partial month, which has no index yet.

## A file the repository does not ship

`api/services/sector_mapping.py` reads `api/data/secteurs-bsh.geojson` to map station
coordinates onto BSH sectors.

The generating script writes **two copies** — `frontend/public/geo/secteurs-bsh.geojson` and
`api/data/secteurs-bsh.geojson` — and only the first was ever committed. The backend read the
second, did not find it, logged a `FileNotFoundError` at startup and served an empty BSH sector
layer.

Fixed by having the backend image copy the versioned file into place
(`docker/backend/Dockerfile`), rather than duplicating 5 MB of JSON in Git. Verified: the
startup `FileNotFoundError` is gone.

To regenerate the file — it needs a bootstrapped warehouse published on the host, since each
sector's name is derived from the dominant hydrogeological entity of the piezometers inside it:

```bash
DEBUG=true BRGM_DB_HOST=localhost BRGM_DB_PORT=49502 \
  uv run python scripts/build_secteurs_bsh_geojson.py
```

## When the Observatory returns HTTP 500

Every Observatory endpoint reads the warehouse directly, so it fails hard when the warehouse
is not reachable or not populated. The two causes look identical from the browser — check the
backend log to tell them apart (`docker compose logs backend`).

| Backend error | Cause | Fix |
|---------------|-------|-----|
| `fe_sendauth: no password supplied` | `BRGM_DB_PASSWORD` is empty in `.env` | Set it to the **`PG_PASSWORD` of the hubeau stack**. The two must match: Junon authenticates against hubeau's PostgreSQL as user `postgres`. |
| `relation "gold.…" does not exist` | The warehouse is reachable but was never bootstrapped | Run `full_bootstrap` in the hubeau Dagster UI. Until Gold tables exist, the Observatory cannot work. |

`.env.example` says `BRGM_DB_PASSWORD` may be left empty "if not using observatory features".
That is true only in the sense that the rest of the application still runs — the Observatory
itself answers 500, it does not degrade gracefully or hide itself.

## Purging the cache after a Climat deployment

Climat responses are Redis-cached for 24 hours (`GRID_TTL = 86400`), so a deployment that
changes the payload shape **or the index values** must purge them, otherwise stale entries are
served for up to a day. Keys are `junon:{prefix}:{hash}` where the hash covers the request
parameters (`dashboard/utils/cache.py::_make_key`), which is why the patterns below use `*`.

```bash
# production (junon-redis) — use junon-redis-dev for the dev stack
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_climat_*' \
  | xargs -r docker exec -i junon-redis redis-cli DEL

# only if the station page's "Contexte climatique" block is affected (TTL 1 h)
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_piezo_detail:*' \
  | xargs -r docker exec -i junon-redis redis-cli DEL
docker exec junon-redis redis-cli --scan --pattern 'junon:obs_hydro_detail:*' \
  | xargs -r docker exec -i junon-redis redis-cli DEL
```
