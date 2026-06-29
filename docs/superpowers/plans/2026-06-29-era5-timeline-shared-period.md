# ERA5 — Timeline shared-period control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the Observatoire timeline bar into a shared-period control: add Météo/Hydro/Piézo checkboxes, and when the timeline is active, drive the ERA5 weather grid from the timeline's month (monthly aggregate for temp/precip/ETP, monthly window for anomaly), in sync with the station/sector replay. When the timeline is closed, the weather falls back to its independent date selector (unchanged).

**Architecture:** No backend change. `TimelineSlider` gains 3 checkbox props wired to the existing `era5Active`/`showHydro`/`showPiezo` state. `ObservatoryPage` derives `timelineMonth = timelineData.periods[idx] + '-01'` and, when set + ERA5 active, fetches the month via the existing `useERA5Monthly` (temp/precip/ETP) or `useERA5TempAnomaly` (anomaly) and feeds the result to the map's existing `era5Points`/`era5AnomalyPoints`; otherwise the existing snapshot/independent-date path is used. The drawer's date control is disabled with a hint while the timeline drives the weather.

**Tech Stack:** React 19 + MapLibre, react-query. Reuses `useERA5Monthly` (`/observatory/era5/monthly`, returns `ERA5GridPoint[]`).

**Builds on:** ERA5 feature on `feat/era5-weather-observatory` (TimelineSlider with `onPeriodChange(periodIndex, timeline)`; periods are `YYYY-MM` strings; `era5Active`/`era5Variable`/`era5Date`/`era5Window` state; map consumes `era5Points`/`era5AnomalyPoints`).

## Global Constraints
- **Shared period, single cursor** (user decision). When the timeline is open (`timelinePeriodIndex != null`), it drives BOTH stations (existing) and — if `era5Active` — the weather grid. When closed (`null`), weather uses its independent date selector (current behaviour, unchanged).
- **Monthly when timeline-driven:** periods are monthly, so temp/precip/ETP use `useERA5Monthly(period+'-01')` (AVG temp, SUM precip, AVG ETP), anomaly uses `useERA5TempAnomaly(period+'-01', window)`. Daily `/snapshot` is used ONLY in the independent (timeline-closed) path.
- **Checkboxes map to existing state:** Météo→`era5Active`, Hydro→`showHydro`, Piézo→`showPiezo`. No new state; RightDrawer stays in sync automatically.
- **No implicit mutation beyond the shared period.** Toggling a checkbox only flips that layer's existing visibility state. French UI; opt-in unchanged (ERA5 default off).
- **Frontend:** build `npm run build` (exit 0, stricter than tsc --noEmit); tests `npx vitest run`. Local services: frontend `:49513`.

---

### Task 1: TimelineSlider — Météo/Hydro/Piézo checkboxes

**Files:** Modify `frontend/src/components/observatory/TimelineSlider.tsx`

**Interfaces:**
- New props: `showMeteo: boolean; onMeteoChange: (v: boolean) => void; showHydro: boolean; onHydroChange: (v: boolean) => void; showPiezo: boolean; onPiezoChange: (v: boolean) => void`.

- [ ] **Step 1: Extend Props + render a compact checkbox group**

Add the 6 props to the `Props` interface. In the slider panel (near the season pills / months-count area), render a compact group of three checkboxes labelled `t('observatory.timeline.layerMeteo')` / `layerHydro` / `layerPiezo`, bound to the props (`checked={showMeteo}` `onChange={() => onMeteoChange(!showMeteo)}`, etc.). Use the same checkbox styling as the existing season pills / drawer checkboxes. Keep them visible whenever the slider panel is open.

- [ ] **Step 2: Build + commit**

`cd frontend && npm run build` (exit 0).
```bash
git add frontend/src/components/observatory/TimelineSlider.tsx
git commit -m "feat(era5): Météo/Hydro/Piézo checkboxes on the timeline bar"
```
(Props are optional-safe to pass next task; the build must pass with the new required props only once the parent passes them — so if TypeScript requires them, this task's build will fail until Task 2 wires them. To keep Task 1 independently green, give the 3 `show*`/`on*Change` props **safe defaults** in the destructure, e.g. `showMeteo = false, onMeteoChange = () => {}`, so the component compiles and renders standalone; Task 2 supplies real values.)

---

### Task 2: ObservatoryPage — drive weather from the shared period + wire checkboxes + drawer hint

**Files:** Modify `frontend/src/pages/ObservatoryPage.tsx`, `frontend/src/components/observatory/RightDrawer.tsx`, `frontend/src/i18n/locales/fr.json`, `frontend/src/i18n/locales/en.json`

**Interfaces:**
- Consumes `useERA5Monthly` (add to the hooks import), the existing `useERA5Snapshot`/`useERA5TempAnomaly`, and TimelineSlider's new props.
- RightDrawer gains `era5TimelineDriven?: boolean` to disable the date control + show a hint.

- [ ] **Step 1: Derive the timeline month and select monthly vs snapshot data**

In `ObservatoryPage.tsx`, after `timelinePeriodIndex`/`timelineData` are defined, add:
```ts
const timelineMonth = (timelinePeriodIndex != null && timelineData)
  ? timelineData.periods[timelinePeriodIndex] + '-01'
  : null
```
Import `useERA5Monthly` (add to the existing `@/hooks/useObservatory` import). Replace the current snapshot/anomaly wiring with a timeline-aware version:
```ts
// daily snapshot ONLY when independent (timeline closed) and not anomaly
const { data: era5SnapshotPoints } = useERA5Snapshot(
  era5Active && !timelineMonth && era5Date && era5Variable !== 'anomaly' ? era5Date : undefined
)
// monthly aggregate when timeline-driven and not anomaly
const { data: era5MonthlyPoints } = useERA5Monthly(
  era5Active && timelineMonth && era5Variable !== 'anomaly' ? timelineMonth : undefined
)
const era5Points = timelineMonth ? era5MonthlyPoints : era5SnapshotPoints
// anomaly: month from timeline if present, else the independent month picker
const anomalyMonth = timelineMonth ?? (era5Date ? era5Date.slice(0, 7) + '-01' : era5Date)
const { data: era5AnomalyPoints } = useERA5TempAnomaly(
  anomalyMonth, era5Window, era5Active && era5Variable === 'anomaly'
)
```
(Replace the previous `era5Points`/`era5Month`/`era5AnomalyPoints` declarations; keep `era5Date`/`era5Window`/`era5Variable` state as-is.) `era5Points` stays typed `ERA5GridPoint[] | undefined` and flows to `<ObservatoryMap era5Points={era5Points} ... />` unchanged.

- [ ] **Step 2: Wire the timeline checkboxes**

On `<TimelineSlider ... />` pass:
```tsx
showMeteo={era5Active} onMeteoChange={setEra5Active}
showHydro={showHydro} onHydroChange={setShowHydro}
showPiezo={showPiezo} onPiezoChange={setShowPiezo}
```

- [ ] **Step 3: Drawer date control — disable + hint while timeline-driven**

Pass `era5TimelineDriven={timelineMonth != null}` to `<RightDrawer ... />`. In `RightDrawer.tsx`, add the prop; when `era5TimelineDriven` is true, disable the ERA5 date/month `<input>` (`disabled` + muted style) and show `t('observatory.drawer.era5TimelineDriven')` ("Piloté par la barre chronologique") beneath it. When false, behave exactly as now.

- [ ] **Step 4: i18n**

`fr.json`: under `observatory.timeline` add `layerMeteo`="Météo", `layerHydro`="Hydro", `layerPiezo`="Piézo"; under `observatory.drawer` add `era5TimelineDriven`="Piloté par la barre chronologique". Add English equivalents to `en.json`. Validate both as JSON.

- [ ] **Step 5: Verify + commit**

`cd frontend && node -e "JSON.parse(require('fs').readFileSync('src/i18n/locales/fr.json','utf8'));JSON.parse(require('fs').readFileSync('src/i18n/locales/en.json','utf8'));console.log('ok')"`; `npm run build` (exit 0); `npx vitest run src/lib/era5-grid.test.ts src/lib/era5-colors.test.ts src/lib/era5-zones.test.ts` (pass).
```bash
git add -A frontend && git commit -m "feat(era5): drive weather grid from shared timeline period (monthly); checkbox wiring; drawer hint"
```

---

### Task 3: End-to-end verification

**Files:** none.
- [ ] **Step 1:** `docker compose up -d --build frontend`.
- [ ] **Step 2 (bundle + serve):** frontend 200; the timeline layer labels shipped (`docker exec junon-frontend grep -rl "layerMeteo\|Piloté par la barre" /usr/share/nginx/html/assets/` or grep the French strings).
- [ ] **Step 3 (browser — user/any env with a browser):** open the Observatoire → enable ERA5 → open the timeline → the Météo/Hydro/Piézo checkboxes appear and toggle the respective layers; scrubbing the cursor moves the month and the ERA5 grid updates each month (monthly aggregate / anomaly), in sync with the station replay; the drawer's ERA5 date control is disabled with the "Piloté par la barre chronologique" hint while the timeline is open; closing the timeline restores the independent date selector and daily snapshot. Stations behave as before.

---

## Self-Review
- Shared single-cursor period drives stations (existing) + weather (new) → Task 2 Step 1/2. ✓
- Monthly aggregate when timeline-driven (temp/precip/ETP via useERA5Monthly; anomaly via useERA5TempAnomaly), daily snapshot only when independent → Task 2 Step 1. ✓
- 3 checkboxes mapped to era5Active/showHydro/showPiezo, no new state → Tasks 1, 2 Step 2. ✓
- Independent date selector preserved when timeline closed; disabled+hinted when driven → Task 2 Step 3. ✓
- French i18n (+en) → Task 2 Step 4. ✓
- No backend change; `era5Points` type unchanged (`ERA5GridPoint[]`) so map rendering untouched. ✓
- Placeholder scan: code provided for the wiring; TimelineSlider checkbox markup follows existing styling (implementer matches). Names consistent: `timelineMonth`, `era5MonthlyPoints`, `anomalyMonth`, `era5TimelineDriven`, `showMeteo/onMeteoChange`.
