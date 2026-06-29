# ERA5 Temperature Anomaly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "temperature anomaly" colouring mode to the ERA5 grid: for a 1/3/6/12-month window ending at the selected date, colour each cell by its mean-temperature departure from the long-term (1950+) normal, on a divergent scale.

**Architecture:** Computed on the fly in the FastAPI router against `gold.int_era5_for_all_stations`, split into a Redis-cached climatology (cell × calendar-month long-term mean, one full scan reused 7 days) and a cheap per-request window mean; anomaly = window mean − normal. Frontend adds an `'anomaly'` variable to the existing ERA5 panel with a window selector and a divergent legend, reusing the Phase 0 squares layer.

**Tech Stack:** FastAPI + sync SQLAlchemy (warehouse `brgm-postgres`), React 19 + MapLibre GL v5 + react-query + react-i18next, vitest + pytest for pure helpers.

**Builds on:** Phase 0 (already merged into branch `feat/era5-weather-observatory`). The ERA5 panel, `era5-colors.ts`, `era5-grid.ts`, the map ERA5 effect, the drawer panel, and the page state all exist.

## Global Constraints

- **DB:** daily values in `gold.int_era5_for_all_stations` (`latitude`, `longitude`, `era5_date`, `temperature_2m`). Full record 1950→present.
- **Compute location:** on the fly in `api/routers/observatory_era5.py`; NO dbt, NO new warehouse tables. Cache via `get_cached(key, params, ttl, fetch)`; engine via `get_brgm_sync_engine()` with `finally: pass`.
- **Climatology cache:** key `obs_era5_temp_climatology`, TTL 7 days (604800). Anomaly endpoint cache keyed `{date, window}`, TTL 86400.
- **Windows:** N ∈ {1, 3, 6, 12} months; window ends at the selected date's month; normal = mean of the climatology for the N ending calendar months (year-boundary wrap).
- **Reference:** full record (1950+) — NOT 1991-2020.
- **Divergent scale:** centred on 0, domain −5→+5 °C, clamped. Signed display ("+2.3 °C" / "−1.1 °C", using the Unicode minus "−"). null → "—".
- **UI language:** French only; strings via react-i18next keys in `frontend/src/i18n/locales/fr.json`.
- **Opt-in:** anomaly is just another value of the existing `era5Variable`; ERA5 stays off by default and never filters/alters stations.
- **API base:** `/api/v1`. **Frontend tests:** `cd frontend && npx vitest run <file>`; type-check `npx tsc --noEmit`. **Backend tests:** `DEBUG=true DB_PASSWORD=test uv run pytest <files> -q`.
- **Local services:** backend `junon-backend` (49514), frontend `junon-frontend` (49513), warehouse `brgm-postgres`.

---

### Task 1: Backend — pure window-months helper (TDD)

**Files:**
- Create: `api/era5_anomaly.py`
- Test: `tests/test_era5_anomaly.py`

**Interfaces:**
- Produces: `window_end_months(end_month: int, n: int) -> list[int]` — the `n` calendar months (1..12) ending at `end_month`, in chronological order, wrapping the year boundary. `add_months(d: date, k: int) -> date` — shift a date by `k` months (k may be negative), returning the first day of the resulting month.

- [ ] **Step 1: Write the failing test**

```python
from datetime import date
from api.era5_anomaly import window_end_months, add_months


def test_window_end_months_no_wrap():
    assert window_end_months(3, 3) == [1, 2, 3]
    assert window_end_months(12, 1) == [12]
    assert window_end_months(6, 6) == [1, 2, 3, 4, 5, 6]


def test_window_end_months_wraps_year_boundary():
    assert window_end_months(1, 3) == [11, 12, 1]
    assert window_end_months(2, 12) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]


def test_add_months_forward_and_back():
    assert add_months(date(2024, 3, 15), 1) == date(2024, 4, 1)
    assert add_months(date(2024, 1, 10), -1) == date(2023, 12, 1)
    assert add_months(date(2024, 3, 1), -2) == date(2024, 1, 1)
    assert add_months(date(2024, 12, 1), 1) == date(2025, 1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_era5_anomaly.py -q`
Expected: FAIL — `ModuleNotFoundError: api.era5_anomaly`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Pure helpers for ERA5 temperature-anomaly window maths (no DB/Streamlit)."""
from __future__ import annotations

from datetime import date


def window_end_months(end_month: int, n: int) -> list[int]:
    """The n calendar months (1..12) ending at end_month, chronological, wrapping."""
    months = [((end_month - i - 1) % 12) + 1 for i in range(n)]
    return months[::-1]


def add_months(d: date, k: int) -> date:
    """First day of the month k months from d (k may be negative)."""
    total = (d.year * 12 + (d.month - 1)) + k
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DEBUG=true DB_PASSWORD=test uv run pytest tests/test_era5_anomaly.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/era5_anomaly.py tests/test_era5_anomaly.py
git commit -m "feat(era5): pure window-months/add-months helpers for temp anomaly"
```

---

### Task 2: Backend — `/temp-anomaly` endpoint with cached climatology

**Files:**
- Modify: `api/routers/observatory_era5.py`

**Interfaces:**
- Consumes: `window_end_months`, `add_months` (Task 1).
- Produces (HTTP): `GET /api/v1/observatory/era5/temp-anomaly?date=YYYY-MM-DD&window=N` → `[{latitude, longitude, anomaly_c}]`. `window` ∈ {1,3,6,12}; `date` optional → latest available month. Internal: `_era5_temp_climatology()` returns `list[{latitude, longitude, mo, mean_c}]` (cached).

- [ ] **Step 1: Add imports and the cached climatology helper**

At the top of `api/routers/observatory_era5.py`, add to the imports:

```python
from api.era5_anomaly import window_end_months, add_months
```

Add a TTL constant near the others:

```python
CLIMATOLOGY_TTL = 604800  # 7 days — climatology is effectively static
ANOMALY_TTL = 86400
```

Add the climatology helper (after `_brgm_url` / before the route functions):

```python
def _era5_temp_climatology():
    """Per-cell long-term mean temperature for each calendar month (1950+)."""
    def fetch():
        query = """
            SELECT latitude, longitude,
                   EXTRACT(MONTH FROM era5_date)::int AS mo,
                   AVG(temperature_2m) AS mean_c
            FROM gold.int_era5_for_all_stations
            WHERE temperature_2m IS NOT NULL
            GROUP BY latitude, longitude, EXTRACT(MONTH FROM era5_date)
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_temp_climatology", {}, CLIMATOLOGY_TTL, fetch)
```

- [ ] **Step 2: Add the `/temp-anomaly` route**

Append after the `/range` route:

```python
@router.get("/temp-anomaly")
def get_era5_temp_anomaly(
    anomaly_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    if window not in (1, 3, 6, 12):
        window = 3

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                d = anomaly_date
                if d is None:
                    d = conn.execute(
                        text("SELECT max(era5_date) FROM gold.int_era5_for_all_stations")
                    ).scalar()
                month_start = DateType(d.year, d.month, 1)
                win_start = add_months(month_start, -(window - 1))
                win_end = add_months(month_start, 1)
                rows = conn.execute(
                    text(
                        """
                        SELECT latitude, longitude, AVG(temperature_2m) AS window_mean
                        FROM gold.int_era5_for_all_stations
                        WHERE era5_date >= :win_start AND era5_date < :win_end
                          AND temperature_2m IS NOT NULL
                        GROUP BY latitude, longitude
                        """
                    ),
                    {"win_start": win_start, "win_end": win_end},
                ).mappings().all()

            # climatology (separately cached) → normal for the N ending months
            clim = _era5_temp_climatology()
            months = set(window_end_months(month_start.month, window))
            norm: dict[tuple, list] = {}
            for c in clim:
                if c["mo"] in months:
                    norm.setdefault((c["latitude"], c["longitude"]), []).append(float(c["mean_c"]))

            out = []
            for r in rows:
                key = (r["latitude"], r["longitude"])
                vals = norm.get(key)
                if not vals or len(vals) < len(months) or r["window_mean"] is None:
                    continue
                normal = sum(vals) / len(vals)
                out.append(
                    {
                        "latitude": r["latitude"],
                        "longitude": r["longitude"],
                        "anomaly_c": float(r["window_mean"]) - normal,
                    }
                )
            return out
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_temp_anomaly",
        {"date": str(anomaly_date) if anomaly_date else "latest", "window": window},
        ANOMALY_TTL,
        fetch,
    )
```

- [ ] **Step 3: Rebuild backend and verify (first call warms the climatology cache)**

```bash
docker compose up -d --build backend
sleep 5
time curl -s "http://localhost:49514/api/v1/observatory/era5/temp-anomaly?window=3" | head -c 200
```
Expected: a non-empty JSON array of `{latitude, longitude, anomaly_c}`; anomaly_c values are small signed floats (roughly −10..+10). The first call may take a while (climatology scan); note the time.

```bash
time curl -s "http://localhost:49514/api/v1/observatory/era5/temp-anomaly?window=12&date=2023-06-15" | head -c 200
```
Expected: non-empty array; faster than the first call (climatology now cached).

- [ ] **Step 4: Commit**

```bash
git add api/routers/observatory_era5.py
git commit -m "feat(era5): /temp-anomaly endpoint (cached climatology + per-request window mean)"
```

---

### Task 3: Frontend — add `'anomaly'` to colour model + signed format (TDD)

**Files:**
- Modify: `frontend/src/lib/era5-colors.ts`
- Modify: `frontend/src/lib/era5-colors.test.ts`

**Interfaces:**
- Produces: `Era5Variable` gains `'anomaly'`; `ERA5_VARIABLES.anomaly` has `prop: 'anomaly_c'`, `unit: '°C'`, `labelKey: 'observatory.drawer.era5VarAnomaly'`, a divergent `stops` set. `era5FormatValue('anomaly', v)` returns a signed value with the Unicode minus (e.g. `'+2.3 °C'`, `'−1.1 °C'`); null → `'—'`. `era5ColorExpression('anomaly')` reads `['to-number', ['get', 'anomaly_c']]`.

- [ ] **Step 1: Extend the test**

Add to `frontend/src/lib/era5-colors.test.ts`:

```ts
it('includes the anomaly variable with a divergent scale', () => {
  expect(ERA5_VARIABLES.anomaly.prop).toBe('anomaly_c')
  const expr = era5ColorExpression('anomaly') as any[]
  expect(expr[2]).toEqual(['to-number', ['get', 'anomaly_c']])
  // divergent scale includes a 0 midpoint stop
  const stopValues = expr.slice(3).filter((_, i) => i % 2 === 0)
  expect(stopValues).toContain(0)
})

it('formats anomaly with an explicit sign', () => {
  expect(era5FormatValue('anomaly', 2.3)).toBe('+2.3 °C')
  expect(era5FormatValue('anomaly', -1.1)).toBe('−1.1 °C')
  expect(era5FormatValue('anomaly', 0)).toBe('+0.0 °C')
  expect(era5FormatValue('anomaly', null)).toBe('—')
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: FAIL — `ERA5_VARIABLES.anomaly` undefined.

- [ ] **Step 3: Implement**

In `frontend/src/lib/era5-colors.ts`:

Change the type:
```ts
export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation' | 'anomaly'
```

Add to `ERA5_VARIABLES`:
```ts
  anomaly: {
    key: 'anomaly', prop: 'anomaly_c', unit: '°C',
    labelKey: 'observatory.drawer.era5VarAnomaly',
    stops: [[-5, '#2166ac'], [-2.5, '#67a9cf'], [-0.5, '#d1e5f0'], [0, '#f7f7f7'], [0.5, '#fddbc7'], [2.5, '#ef8a62'], [5, '#b2182b']],
  },
```
Note: `prop` is now one of four columns — widen the `Era5VarConfig.prop` union to include `'anomaly_c'`:
```ts
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation' | 'anomaly_c'
```

Update `era5FormatValue` so anomaly gets an explicit sign with the Unicode minus:
```ts
export function era5FormatValue(v: Era5Variable, value: number | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  const cfg = ERA5_VARIABLES[v]
  if (v === 'anomaly') {
    const s = value.toFixed(1)
    return `${value < 0 ? s.replace('-', '−') : `+${s}`} ${cfg.unit}`
  }
  const shown = v === 'evaporation' ? Math.abs(value) : value
  return `${shown.toFixed(1)} ${cfg.unit}`
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/lib/era5-colors.test.ts`
Expected: PASS (5 tests: the 3 originals + 2 new).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/era5-colors.ts frontend/src/lib/era5-colors.test.ts
git commit -m "feat(era5): anomaly variable with divergent scale and signed formatting"
```

---

### Task 4: Frontend — anomaly type + client + hook

**Files:**
- Modify: `frontend/src/lib/observatory-types.ts`
- Modify: `frontend/src/lib/observatory-api.ts`
- Modify: `frontend/src/hooks/useObservatory.ts`

**Interfaces:**
- Produces: `ERA5AnomalyPoint { latitude: number; longitude: number; anomaly_c: number | null }`; `observatoryApi.era5.tempAnomaly(date, window)` → `Promise<ERA5AnomalyPoint[]>`; `useERA5TempAnomaly(date: string | undefined, window: number, enabled: boolean)`.

- [ ] **Step 1: Add the type**

In `observatory-types.ts`, after `ERA5Range`:
```ts
export interface ERA5AnomalyPoint {
  latitude: number
  longitude: number
  anomaly_c: number | null
}
```

- [ ] **Step 2: Add the client method**

In `observatory-api.ts`, add `ERA5AnomalyPoint` to the type import block, then add to the `era5` object:
```ts
    tempAnomaly: (date: string, window: number) =>
      fetchJson<ERA5AnomalyPoint[]>('/observatory/era5/temp-anomaly', { date, window: String(window) }),
```

- [ ] **Step 3: Add the hook**

In `useObservatory.ts`, after `useERA5Range`:
```ts
export function useERA5TempAnomaly(date: string | undefined, window: number, enabled: boolean) {
  return useQuery({
    queryKey: ['obs-era5', 'temp-anomaly', date, window],
    queryFn: () => observatoryApi.era5.tempAnomaly(date!, window),
    enabled: enabled && !!date,
    staleTime: 24 * 60 * 60 * 1000,
  })
}
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/observatory-types.ts frontend/src/lib/observatory-api.ts frontend/src/hooks/useObservatory.ts
git commit -m "feat(era5): ERA5AnomalyPoint type, tempAnomaly client, useERA5TempAnomaly hook"
```

---

### Task 5: Frontend — feed anomaly data through the map layer

**Files:**
- Modify: `frontend/src/components/observatory/ObservatoryMap.tsx`

**Interfaces:**
- Consumes: existing ERA5 effect (Phase 0), `era5ColorExpression`, `era5FormatValue`, `ERA5_VARIABLES`. New prop `era5AnomalyPoints?: ERA5AnomalyPoint[]` and `era5Window?: number`.
- Behaviour: when `era5Variable === 'anomaly'`, the effect builds squares from `era5AnomalyPoints` (each feature carries `anomaly_c`), colours with the divergent expression, and the popup shows the windowed anomaly. Otherwise unchanged from Phase 0.

- [ ] **Step 1: Extend imports, Props, and destructuring**

Add the import:
```ts
import type { ERA5AnomalyPoint } from '@/lib/observatory-types'
```
Add to the `Props` interface:
```ts
  era5AnomalyPoints?: ERA5AnomalyPoint[]
  era5Window?: number
```
Add to the destructured signature (with the other era5 props):
```ts
  era5AnomalyPoints, era5Window = 3,
```

- [ ] **Step 2: Branch the effect data source on the variable**

In the ERA5 layer effect, replace the block that computes `pts`/`data` (the
Phase 0 `const cfg = ERA5_VARIABLES[era5Variable]; const pts = (era5Points ?? ...)`)
with a variable-aware version:

```tsx
    const cfg = ERA5_VARIABLES[era5Variable]
    let data: GeoJSON.FeatureCollection<GeoJSON.Polygon, Record<string, number | null>>
    if (era5Variable === 'anomaly') {
      const pts = (era5AnomalyPoints ?? []).filter((p) => p.anomaly_c != null)
      data = {
        type: 'FeatureCollection',
        features: pts.map((p) => {
          const h = 0.05
          const lon = Number(p.longitude), lat = Number(p.latitude)
          return {
            type: 'Feature' as const,
            geometry: { type: 'Polygon' as const, coordinates: [[
              [lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h], [lon - h, lat + h], [lon - h, lat - h],
            ]] },
            properties: { anomaly_c: p.anomaly_c },
          }
        }),
      }
    } else {
      const pts = (era5Points ?? []).filter((p) => p[cfg.prop as 'temperature_2m' | 'total_precipitation' | 'potential_evaporation'] != null)
      data = era5PointsToSquares(pts)
    }
```

(Keep the existing add-source / setData / setPaintProperty logic that follows; it
already reads `era5ColorExpression(era5Variable)` and so will use the divergent
expression automatically.)

- [ ] **Step 3: Make the popup anomaly-aware**

In the click handler, replace the 3-line value block with a branch: in anomaly
mode show the windowed anomaly; otherwise the Phase 0 three values:

```ts
        let html: string
        if (era5VariableRef.current === 'anomaly') {
          html = `<div style="font-size:12px;line-height:1.5"><div>${t('observatory.era5.popupAnomaly', { n: era5WindowRef.current })}: ${era5FormatValue('anomaly', num('anomaly_c'))}</div></div>`
        } else {
          html = `<div style="font-size:12px;line-height:1.5">
              <div>${t('observatory.era5.popupTemperature')}: ${era5FormatValue('temperature', num('temperature_2m'))}</div>
              <div>${t('observatory.era5.popupPrecipitation')}: ${era5FormatValue('precipitation', num('total_precipitation'))}</div>
              <div>${t('observatory.era5.popupEvaporation')}: ${era5FormatValue('evaporation', num('potential_evaporation'))}</div>
            </div>`
        }
```

Because the click handler is registered once but reads the variable/window, add refs that always hold the latest values (near the other refs at the top of the component):

```ts
  const era5VariableRef = useRef(era5Variable); era5VariableRef.current = era5Variable
  const era5WindowRef = useRef(era5Window); era5WindowRef.current = era5Window
```

- [ ] **Step 4: Add `era5AnomalyPoints` and `era5Window` to the effect dependency array**

The ERA5 effect's dependency array becomes:
`[mapLoaded, era5Active, era5Points, era5AnomalyPoints, era5Variable, era5Window, t]`

- [ ] **Step 5: Type-check, then commit**

Run: `cd frontend && npx tsc --noEmit`
Expected: exit 0.

```bash
git add frontend/src/components/observatory/ObservatoryMap.tsx
git commit -m "feat(era5): render temperature-anomaly squares + windowed popup on the map"
```

---

### Task 6: Frontend — anomaly radio + window selector + page wiring + i18n

**Files:**
- Modify: `frontend/src/i18n/locales/fr.json`
- Modify: `frontend/src/components/observatory/RightDrawer.tsx`
- Modify: `frontend/src/pages/ObservatoryPage.tsx`

**Interfaces:**
- Consumes: `useERA5TempAnomaly` (Task 4), the new map props (Task 5), `ERA5_VARIABLES` (now includes `anomaly`).
- Produces: RightDrawer gains `era5Window: number; setEra5Window: (n: number) => void`. ObservatoryPage gains `era5Window` state and the anomaly fetch.

- [ ] **Step 1: Add i18n keys**

In `fr.json`, inside `observatory.drawer` add:
```json
      "era5VarAnomaly": "Anomalie thermique",
      "era5Window": "Fenêtre",
      "era5Window1": "1 mois",
      "era5Window3": "3 mois",
      "era5Window6": "6 mois",
      "era5Window12": "12 mois",
```
Inside the `observatory.era5` object add:
```json
      "popupAnomaly": "Anomalie {{n}} mois"
```
(Validate JSON: `cd frontend && node -e "JSON.parse(require('fs').readFileSync('src/i18n/locales/fr.json','utf8'));console.log('ok')"`.)

- [ ] **Step 2: RightDrawer — window selector + props**

Add to `Props`:
```ts
  era5Window: number; setEra5Window: (n: number) => void
```
The colour-by radio group already maps `Object.values(ERA5_VARIABLES)`, so the
`anomaly` entry now appears automatically (its label uses `era5VarAnomaly`).
After the colour-by group, when anomaly is selected, render the window selector:
```tsx
              {props.era5Variable === 'anomaly' && (
                <div>
                  <label className="text-xs text-text-secondary block mb-1">{t('observatory.drawer.era5Window')}</label>
                  <div className="flex gap-1">
                    {[1, 3, 6, 12].map((w) => (
                      <button key={w} onClick={() => props.setEra5Window(w)} aria-pressed={props.era5Window === w} className={`flex-1 px-2 py-1 rounded text-xs border ${props.era5Window === w ? 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30' : 'bg-bg-primary text-text-secondary border-white/10'}`}>{t(`observatory.drawer.era5Window${w}`)}</button>
                    ))}
                  </div>
                </div>
              )}
```
(Place this inside the `props.era5Active && (...)` panel, after the colour-by block and before the date input.)

- [ ] **Step 3: ObservatoryPage — state, anomaly fetch, props**

Add state after `era5Date`:
```ts
  const [era5Window, setEra5Window] = useState(3)
```
Add the anomaly fetch alongside the snapshot fetch:
```ts
  const { data: era5AnomalyPoints } = useERA5TempAnomaly(era5Date, era5Window, era5Active && era5Variable === 'anomaly')
```
(Import `useERA5TempAnomaly` from the hooks module — add it to the existing import.)

On `<ObservatoryMap ... />` add:
```tsx
        era5AnomalyPoints={era5AnomalyPoints} era5Window={era5Window}
```
On `<RightDrawer ... />` add:
```tsx
        era5Window={era5Window} setEra5Window={setEra5Window}
```

- [ ] **Step 4: Verify**

Run all three:
```bash
cd frontend && node -e "JSON.parse(require('fs').readFileSync('src/i18n/locales/fr.json','utf8'));console.log('ok')"
cd frontend && npx tsc --noEmit
cd frontend && npx vitest run src/lib/era5-grid.test.ts src/lib/era5-colors.test.ts
```
Expected: `ok`; tsc exit 0; 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/i18n/locales/fr.json frontend/src/components/observatory/RightDrawer.tsx frontend/src/pages/ObservatoryPage.tsx
git commit -m "feat(era5): anomaly mode UI — variable radio, window selector, page wiring, i18n"
```

---

### Task 7: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Rebuild the frontend**

```bash
docker compose up -d --build frontend
```

- [ ] **Step 2: Verify the deployed bundle + data path** (browser unavailable in CI; verify via proxy + bundle)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:49513/
curl -s "http://localhost:49513/api/v1/observatory/era5/temp-anomaly?window=3" | head -c 200; echo
```
Expected: 200; non-empty `[{latitude, longitude, anomaly_c}]`.

```bash
ASSET=$(curl -s http://localhost:49513/ | grep -oE '/assets/ObservatoryPage[^"]+\.js' | head -1)
curl -s "http://localhost:49513$ASSET" | grep -q "Anomalie thermique" && echo "anomaly UI shipped"
```
Expected: "anomaly UI shipped".

- [ ] **Step 3: Browser check (user or any env with a browser)**

Open the Observatoire → drawer → Météo (ERA5) → ON → select "Anomalie thermique":
- A window selector (1/3/6/12 mois) appears; switching windows re-colours the grid.
- Colours are divergent (blue cold ↔ red warm) centred on neutral; the legend shows a 0 midpoint.
- Clicking a cell shows "Anomalie N mois : ±X.X °C".
- Switching back to Température/Précipitations/ETP restores Phase 0 behaviour; stations stay visible throughout.

---

## Self-Review

**Spec coverage:**
- On-the-fly cached climatology + window mean + anomaly → Tasks 1, 2. ✓
- `/temp-anomaly` endpoint, default-latest-month, window∈{1,3,6,12} → Task 2. ✓
- Divergent scale + signed format → Task 3. ✓
- Type/client/hook → Task 4. ✓
- Map rendering of anomaly squares + windowed popup → Task 5. ✓
- Anomaly radio + window selector + wiring + i18n → Task 6. ✓
- Year-boundary wrap handled by `window_end_months` (Task 1) and consumed in Task 2. ✓
- Out-of-scope (SPI/SPEI, timeline coupling, by-zone, 1991-2020 ref) — not built. ✓

**Placeholder scan:** none; every code step is complete.

**Type consistency:** `Era5Variable` includes `'anomaly'` (Task 3) before Tasks 5/6 use it; `prop` union widened to include `'anomaly_c'`; `ERA5AnomalyPoint`/`anomaly_c` used identically in backend output (Task 2), type (Task 4), map adapter (Task 5). `era5Window`/`setEra5Window`/`era5AnomalyPoints` names match across Tasks 5 and 6. The map effect reads `era5VariableRef`/`era5WindowRef` for the once-registered popup handler (consistent with the file's ref pattern).
