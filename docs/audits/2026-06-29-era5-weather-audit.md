# ERA5 Weather Feature — Audit Report (2026-06-29)

Multi-agent audit (26 agents, 7 dimensions, adversarial verification): 18 findings
raised, **14 confirmed** (4 refuted), **0 Critical, 3 Important, 11 Minor**.

**All 14 confirmed findings were fixed** on branch `feat/era5-weather-observatory`
(lots F1–F5, commits `52760d6`, `0eaac07`, `7f87a15`+`c4bdd23`, `bfab5b3`, `91e1485`),
each implemented and code-reviewed. This document preserves the audit for traceability.

## Executive summary
The ERA5 "Météo" overlay is functionally sound and correctly built as an opt-in,
default-off layer. The most consequential issues were a backend correctness bug
(default anomaly compared a partial trailing month against a full-month climatology),
a shared-warehouse availability risk (the ~71s climatology scan had no stampede
protection), and a frontend regression where clicking a weather cell silently mutated
the rest of the UI (violating the "opt-in overlay must never implicitly filter/alter
the UI" rule).

## Important (all fixed)

### I1 — Default anomaly compared a partial trailing month against a full-month climatology
`api/routers/observatory_era5.py`. ERA5's latest calendar month is usually incomplete;
the default view averaged that partial month and subtracted a full-month normal →
systematic bias. **Fix (F1):** default resolves to the latest *complete* month
(`latest_complete_month`).

### I2 — Climatology full-table scan had no single-flight/stampede protection
`observatory_era5.py` + `dashboard/utils/cache.py` + `api/main.py`. Concurrent cache
misses each launched their own ~71s scan on the shared warehouse pool; the 7-day TTL
created a recurring weekly cold window. **Fix (F2):** in-process single-flight lock with
double-check; periodic re-warm (every 6 days) before TTL expiry; `read_cached`/`delete_cached`
helpers added without changing `get_cached` semantics.

### I3 — Clicking an ERA5 cell mutated station selection, spatial filter, and camera
`frontend/src/components/observatory/ObservatoryMap.tsx`. The cell click handler didn't
stop propagation and wasn't in the click guards, so a click also fired the zone/empty
handlers (zoom + filter / reset). **Fix (F3):** added `era5-grid-fill` to the global
click guard and early-return in each zone handler when an ERA5 cell is under the cursor.

## Minor (all fixed)
- **M1** — empty warehouse table 500 → `if d is None: return []`. (F1)
- **M2** — anomaly cached/queried at day granularity though result depends only on month → month-normalized cache key (server) + month-granular react-query key (front). (F1/F4)
- **M3** — `useERA5Range` fired on every Observatory load → gated on `era5Active`. (F4)
- **M4** — `era5NoData` key dead + no empty state → empty-state badge (with loading-vs-empty distinction). (F3)
- **M5** — same variable labelled "Évapotranspiration" vs "ETP" → unified to "Évapotranspiration (ETP)". (F4)
- **M6** — anomaly compute logic untested → extracted pure `compute_anomalies()` + pytest. (F5)
- **M7** — anomaly square adapter duplicated the half-cell constant, untested → extracted `era5AnomalyPointsToSquares` (reuses `ERA5_CELL_HALF`) + vitest. (F3/F5)
- **M8** — no test that colour-scale stops are strictly ascending → vitest over all variables. (F5)

## Known limitation (consciously not fixed)
The F2 single-flight double-check path resets the climatology cache TTL on the shared
entry. This is a latent coupling, not a current defect (the periodic re-warm evicts
explicitly at day 6); fixing it would require changing the shared `get_cached` contract,
so it was left as-is to bound blast radius.

## Refuted candidates (excluded by adversarial verification)
4 candidates were refuted, e.g.: ERA5 popups "stack indefinitely" (false — `closeOnClick`
default closes prior popup), and the legend ASCII-minus vs popup typographic-minus
(stylistic-only, no impact).
