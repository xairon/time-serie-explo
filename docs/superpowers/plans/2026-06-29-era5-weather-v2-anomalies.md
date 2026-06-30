# ERA5 weather v2 (anomalies-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reframe the ERA5 weather module around climate anomalies (vs 1950+ normal): default = precipitation anomaly (%), 3-month window, by-zone choropleth on a dedicated layer; temperature anomaly (°C) also primary; raw temp/precip/ETP demoted to "Contexte brut" with corrected scales; on-map banner+legend.

**Architecture:** Backend generalises the anomaly machinery to temperature (°C) + precipitation (% of normal) over `gold.era5_grid`, each with a cached climatology. Frontend adds the precip-anomaly variable + divergent scales, fixes raw scales to the temporal granularity, renders weather-by-zone on a dedicated choropleth layer (not by hijacking the admin layer), and adds an on-map banner/legend. Spec: `docs/superpowers/specs/2026-06-29-era5-weather-v2-anomalies.md`.

**Builds on:** the ERA5 feature on `feat/era5-weather-observatory` (temp anomaly, `gold.era5_grid` full grid, `aggregateEra5ByZone`, `era5GradientCss`, climatology single-flight + pre-warm, timeline shared period).

## Global Constraints
- Anomaly = departure from the **1950+ normal**, computed on the fly from `gold.era5_grid`, cached (single-flight, 7-day TTL, startup pre-warm), filtering on the `"time"` timestamp column.
- Precip anomaly = `(observed_window_sum − normal_window_sum) / normal_window_sum * 100` (%), normal_window_sum = Σ of the N ending calendar months' mean monthly sums; omit cells with `normal ≤ 0`. Temp anomaly = equal-weighted window mean − mean-of-monthly-means normal (°C) — unchanged.
- Divergent scales centred 0: temp ±5 °C, precip ±80 %. Raw scales adapt to granularity (daily vs monthly) so they don't saturate.
- Primary rendering = by-zone choropleth on a **dedicated** weather layer; do NOT recolour admin `*-fill`; dim admin tint while active. Weather is colour/info only — never mutates station selection.
- French UI; build `npm run build` (strict); tests `npx vitest run` / pytest `DEBUG=true DB_PASSWORD=test uv run pytest`. Local: frontend `:49513`, backend `:49514`.

---

### Task V1: Backend — generic anomaly (temperature °C + precipitation %)

**Files:** Modify `api/era5_anomaly.py`, `api/routers/observatory_era5.py`, `api/main.py` (pre-warm precip climatology); Test `tests/test_era5_anomaly.py`.

**Interfaces:**
- Pure: `compute_precip_anomalies(window_rows, climatology, months, window)` — `window_rows` carry `precip_sum` + `n_months`; `climatology` carries per-(cell,month) `mean_sum`; returns `[{latitude, longitude, anomaly}]` with `anomaly` = % = `(precip_sum − Σnormals)/Σnormals*100`, dropping `n_months<window`, missing normals, or `Σnormals<=0`.
- HTTP: `GET /observatory/era5/anomaly?variable=temperature|precipitation&window=N&date=` → `[{latitude, longitude, anomaly}]`. `/temp-anomaly` stays as alias (`variable=temperature`).

- [ ] **Step 1: TDD the pure precip helper**

Add to `tests/test_era5_anomaly.py`:
```python
from api.era5_anomaly import compute_precip_anomalies

def test_precip_anomaly_percent_basic():
    # one cell, window=3 ending month with normals [10,20,30]=60; observed sum=90 → +50%
    clim = [
        {"latitude": 48.0, "longitude": 2.0, "mo": 1, "mean_sum": 10.0},
        {"latitude": 48.0, "longitude": 2.0, "mo": 2, "mean_sum": 20.0},
        {"latitude": 48.0, "longitude": 2.0, "mo": 3, "mean_sum": 30.0},
    ]
    rows = [{"latitude": 48.0, "longitude": 2.0, "precip_sum": 90.0, "n_months": 3}]
    out = compute_precip_anomalies(rows, clim, [1, 2, 3], 3)
    assert len(out) == 1
    assert out[0]["anomaly"] == 50.0  # (90-60)/60*100

def test_precip_anomaly_drops_incomplete_and_zero_normal():
    clim = [{"latitude": 1.0, "longitude": 1.0, "mo": 1, "mean_sum": 0.0}]
    rows = [{"latitude": 1.0, "longitude": 1.0, "precip_sum": 5.0, "n_months": 1}]
    assert compute_precip_anomalies(rows, clim, [1], 1) == []  # normal<=0 dropped
    rows2 = [{"latitude": 1.0, "longitude": 1.0, "precip_sum": 5.0, "n_months": 0}]
    assert compute_precip_anomalies(rows2, [{"latitude":1.0,"longitude":1.0,"mo":1,"mean_sum":3.0}], [1], 1) == []  # n_months<window
```
Run (must fail): `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_era5_anomaly.py -q`.

- [ ] **Step 2: Implement `compute_precip_anomalies`** in `api/era5_anomaly.py`:
```python
def compute_precip_anomalies(window_rows, climatology, months, window):
    """Precipitation anomaly in % of the 1950+ normal. climatology rows carry mean_sum
    per (cell, calendar month); window_rows carry precip_sum + n_months."""
    month_set = set(months)
    norm = {}
    for c in climatology:
        if c["mo"] in month_set:
            norm.setdefault((float(c["latitude"]), float(c["longitude"])), []).append(float(c["mean_sum"]))
    out = []
    for r in window_rows:
        key = (float(r["latitude"]), float(r["longitude"]))
        vals = norm.get(key)
        if not vals or len(vals) < len(month_set) or r["precip_sum"] is None or int(r["n_months"]) < window:
            continue
        total_normal = sum(vals)
        if total_normal <= 0:
            continue
        out.append({"latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
                    "anomaly": (float(r["precip_sum"]) - total_normal) / total_normal * 100.0})
    return out
```
Run (must pass): same pytest command.

- [ ] **Step 3: Precip climatology + generic endpoint** in `api/routers/observatory_era5.py`:
- Add `_era5_precip_climatology()` mirroring `_era5_temp_climatology()` but `SUM` per month then mean across years:
  ```sql
  WITH monthly AS (
    SELECT latitude, longitude, date_trunc('month', "time") ym,
           EXTRACT(MONTH FROM "time")::int mo, SUM(total_precipitation) msum
    FROM gold.era5_grid WHERE total_precipitation IS NOT NULL
    GROUP BY latitude, longitude, date_trunc('month',"time"), EXTRACT(MONTH FROM "time"))
  SELECT latitude, longitude, mo, AVG(msum) AS mean_sum
  FROM monthly GROUP BY latitude, longitude, mo
  ```
  Cache key `obs_era5_precip_climatology`, own single-flight lock, TTL 604800.
- Add `@router.get("/anomaly")` taking `variable` (default temperature), `window`, `date`. For `temperature`, call the existing temp path (reuse current `/temp-anomaly` body / `compute_anomalies`). For `precipitation`: window query `SELECT latitude, longitude, SUM(total_precipitation) AS precip_sum, COUNT(DISTINCT date_trunc('month',"time")) AS n_months FROM gold.era5_grid WHERE "time">=:win_start AND "time"<:win_end AND total_precipitation IS NOT NULL GROUP BY latitude, longitude`, then `compute_precip_anomalies(rows, _era5_precip_climatology(), window_end_months(month_start.month, window), window)`. Cache key includes `variable`+`window`+month. Keep `/temp-anomaly` as a thin alias to `variable=temperature`.
- [ ] **Step 4: Pre-warm precip climatology** in `api/main.py`: extend the existing background warm/loop to also warm `_era5_precip_climatology` (same non-blocking pattern; both warmed at startup + periodically).
- [ ] **Step 5: Verify + commit** — rebuild backend; `curl ".../era5/anomaly?variable=precipitation&window=3&date=2024-03-01"` → ~11,496 cells, plausible % (roughly −100..+200); `variable=temperature` still works. Commit `feat(era5): precipitation anomaly (% of 1950+ normal) + generic /anomaly endpoint`.

---

### Task V2: Frontend — precip-anomaly variable, divergent %, raw scale fix

**Files:** `frontend/src/lib/era5-colors.ts` (+test), `observatory-types.ts`, `observatory-api.ts`, `hooks/useObservatory.ts`.

- [ ] **Step 1 (TDD colours):** add `precipAnomaly` to `Era5Variable`/`ERA5_VARIABLES` (`prop: 'anomaly'`, unit `'%'`, labelKey `observatory.drawer.era5VarPrecipAnomaly`, divergent stops centred 0, e.g. `[-80,'#8c510a'],[-40,'#d8b365'],[-10,'#f6e8c3'],[0,'#f5f5f5'],[10,'#c7eae5'],[40,'#5ab4ac'],[80,'#01665e']`). Rename existing `anomaly`→`tempAnomaly` (keep `prop:'anomaly'`? note both anomalies use the generic `anomaly` field — distinguish by which is selected). `era5FormatValue('precipAnomaly', v)` → signed `'%'` (`+50 %`, `−40 %`). Extend tests (divergent has 0 stop; format %). Run vitest.
- [ ] **Step 2:** types — keep `ERA5AnomalyPoint { latitude, longitude, anomaly: number | null }` (generic). API client `era5.anomaly(variable, date, window)`; hook `useERA5Anomaly(variable, date, window, enabled)`. Keep temp-anomaly hook working or switch callers to the generic one.
- [ ] **Step 3:** raw scale fix — give `temperature/precipitation/evaporation` a granularity-aware domain (a `domainFor(variable, granularity)` helper, or dynamic min/max from the current data). Minimum: precipitation monthly uses a larger domain (e.g. 0–200 mm) than daily (0–50). Legend uses the active domain.
- [ ] **Step 4:** `npm run build` + vitest; commit.

---

### Task V3: Frontend rendering — dedicated weather choropleth layer

**Files:** `frontend/src/components/observatory/ObservatoryMap.tsx`.

- [ ] **Step 1:** When weather-by-zone is active, instead of overriding the admin `*-fill` colour: build a **dedicated** GeoJSON source `era5-zone` from the active zone's stashed polygons, each feature coloured by its aggregated anomaly (reuse `aggregateEra5ByZone` for the value, build a per-feature fill via a `match`/`interpolate` expression or precomputed feature property), added as a fill layer `era5-zone-fill` below `piezo-clusters`, opacity ~0.7. Remove the previous "recolour admin layer + save/restore paint" approach for the weather use (the admin layer keeps its own colour).
- [ ] **Step 2:** While weather-by-zone active, dim the active admin layer's `fill-opacity` (e.g. 0.05) so it doesn't double up; restore on off. (Simpler than before: just opacity, never touch admin fill-color.)
- [ ] **Step 3:** raw 0.1° grid (`era5-grid-fill`) stays for raw/non-by-zone; ensure the two weather layers (`era5-grid-fill`, `era5-zone-fill`) are mutually exclusive by mode. Keep click isolation.
- [ ] **Step 4:** `npm run build`; commit.

---

### Task V4: Frontend UI — panel reframe + on-map banner

**Files:** `frontend/src/components/observatory/RightDrawer.tsx`, `frontend/src/pages/ObservatoryPage.tsx`, a new `frontend/src/components/observatory/Era5Banner.tsx`, i18n `fr.json`/`en.json`.

- [ ] **Step 1:** Panel — variable selector shows **primary** (Anomalie pluie, Anomalie temp); a collapsible **"Contexte brut"** holds raw temp/precip/ETP. Window selector. By-zone default on. Default on activation: `precipAnomaly`, window 3, by-zone.
- [ ] **Step 2:** `Era5Banner` — a small absolutely-positioned on-map component (shown when `era5Active`): variable label + window + active period (timeline month or picker) + compact gradient legend (reuse `era5GradientCss`) with bounds + 0-centre for divergent. Rendered from `ObservatoryPage` over the map.
- [ ] **Step 3:** Page wiring — `era5Variable` defaults `precipAnomaly`; fetch via `useERA5Anomaly(variable, month/date, window, active)` for anomalies, raw hooks for raw; feed map; pass banner props. Single-activation labels consistent.
- [ ] **Step 4:** i18n keys (fr+en); JSON valid; `npm run build` + vitest; commit.

---

### Task V5: E2E + verify
- [ ] Rebuild backend+frontend; frontend 200; `/anomaly?variable=precipitation` ~11,496 cells via proxy; bundle ships the new labels/banner.
- [ ] Browser (user): default view = precip anomaly 3-mo by-zone with banner+legend; switch to temp anomaly; open "Contexte brut" raw (scales no longer saturate monthly); weather choropleth on its own layer, admin tint dimmed, stations intact; timeline scrubs the month and banner updates.

## Self-Review
- Anomalies-first (precip % default + temp °C), raw secondary → V1, V2, V4. ✓
- Scale fix (divergent anomalies + granularity-aware raw) → V2. ✓
- Dedicated weather layer (no admin hijack) + dim admin → V3. ✓
- On-map banner+legend, single activation, affirmed default → V4. ✓
- Precip anomaly math (% of Σ monthly normals, drops normal≤0 / incomplete) → V1 (tested). ✓
- Climatology pre-warm ×2, single-flight, `gold.era5_grid` `"time"` filter → V1. ✓
- Names: `compute_precip_anomalies`, `_era5_precip_climatology`, `/anomaly?variable=`, `precipAnomaly`/`tempAnomaly`, `useERA5Anomaly`, `era5-zone-fill`, `Era5Banner` — consistent across tasks.
