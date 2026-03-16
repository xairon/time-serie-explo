# Scatter UI Rework Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Latent Space scatter page: custom cluster legend, domain-specific hover, site-highlight on click, compact filter panel, enriched station detail with per-station window scatter.

**Architecture:** 2 new components (ClusterLegendBar, WindowScatter), 4 reworked components (EmbeddingScatter, FilterPanel, StationDetail, LatentSpacePage), 1 new backend endpoint (station-windows). Each task is one component, independently testable.

**Tech Stack:** React 19, Plotly.js, TanStack Query, FastAPI, UMAP (Python)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `frontend/src/components/latent-space/ClusterLegendBar.tsx` | Create | Custom HTML cluster legend with click-to-filter |
| `frontend/src/components/latent-space/WindowScatter.tsx` | Create | Mini scatter for one station's window embeddings |
| `frontend/src/components/latent-space/EmbeddingScatter.tsx` | Modify | Remove native legend, domain-specific hover, accept highlightedSite |
| `frontend/src/components/latent-space/FilterPanel.tsx` | Modify | Compact 200px, active toggle, advanced collapsible |
| `frontend/src/components/latent-space/StationDetail.tsx` | Modify | Enriched metadata, window scatter, filter-by-site button |
| `frontend/src/pages/LatentSpacePage.tsx` | Modify | Site highlight state, click behavior, integrate ClusterLegendBar |
| `frontend/src/hooks/useLatentSpace.ts` | Modify | Add useStationWindows hook |
| `frontend/src/lib/api.ts` | Modify | Add stationWindows API method |
| `api/routers/latent_space.py` | Modify | Add GET /station-windows endpoint |
| `dashboard/utils/latent_space.py` | Modify | Add load_station_windows query |

---

## Chunk 1: Backend + New Components

### Task 1: Backend — station-windows endpoint

**Files:**
- Modify: `/home/ringuet/time-serie-explo/dashboard/utils/latent_space.py`
- Modify: `/home/ringuet/time-serie-explo/api/routers/latent_space.py`
- Modify: `/home/ringuet/time-serie-explo/frontend/src/lib/api.ts`
- Modify: `/home/ringuet/time-serie-explo/frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1: Add query function**

At end of `dashboard/utils/latent_space.py`, add:

```python
async def load_station_windows(session, domain: str, station_id: str, space: str = "multi") -> list[dict]:
    """Load window embeddings for a single station, compute UMAP 2D on-the-fly."""
    import numpy as np

    id_col = "code_bss" if domain == "piezo" else "code_station"
    table = f"ml.{domain}_window_embeddings"

    result = await session.execute(
        text(f"""
            SELECT window_start, window_end, embedding::text AS embedding_raw
            FROM {table}
            WHERE {id_col} = :station_id AND space = :space
            ORDER BY window_start
        """),
        {"station_id": station_id, "space": space},
    )
    rows = result.fetchall()
    if not rows:
        return []

    # Parse embeddings
    embeddings = []
    windows = []
    for r in rows:
        emb = [float(x) for x in r.embedding_raw.strip("[]").split(",")]
        embeddings.append(emb)
        windows.append({
            "window_start": str(r.window_start),
            "window_end": str(r.window_end),
        })

    # UMAP 2D on-the-fly (fast, typically <100 points)
    emb_array = np.array(embeddings, dtype=np.float32)
    if len(emb_array) >= 4:
        import umap
        coords = umap.UMAP(
            n_components=2, n_neighbors=min(15, len(emb_array) - 1),
            min_dist=0.05, metric="cosine", random_state=42,
        ).fit_transform(emb_array)
        for i, w in enumerate(windows):
            w["umap_x"] = float(coords[i, 0])
            w["umap_y"] = float(coords[i, 1])
    else:
        # Too few points for UMAP, use first 2 PCA components
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(emb_array)
        for i, w in enumerate(windows):
            w["umap_x"] = float(coords[i, 0])
            w["umap_y"] = float(coords[i, 1])

    return windows
```

- [ ] **Step 2: Add API endpoint**

In `api/routers/latent_space.py`, add before the profiling endpoint:

```python
@router.get("/station-windows/{domain}/{station_id}")
async def get_station_windows(
    domain: str,
    station_id: str,
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
):
    """Load window embeddings for a station with UMAP 2D coords."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="Invalid domain")
    from dashboard.utils.latent_space import load_station_windows
    import asyncio
    windows = await asyncio.to_thread(
        lambda: asyncio.run(load_station_windows(session, domain, station_id, space))
    )
    return {"station_id": station_id, "windows": windows}
```

Note: Since `load_station_windows` uses async session but UMAP is blocking, we need to handle this carefully. Actually simpler — make `load_station_windows` an async function that does the DB part async and UMAP in a thread:

```python
@router.get("/station-windows/{domain}/{station_id}")
async def get_station_windows(
    domain: str,
    station_id: str,
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
):
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail="Invalid domain")
    from dashboard.utils.latent_space import load_station_windows
    windows = await load_station_windows(session, domain, station_id, space)
    return {"station_id": station_id, "windows": windows}
```

- [ ] **Step 3: Add API client method**

In `frontend/src/lib/api.ts`, add to `latentSpace` object:

```typescript
    stationWindows: (domain: string, stationId: string, space: string = 'multi') =>
      fetchJson<{ station_id: string; windows: Array<{ window_start: string; window_end: string; umap_x: number; umap_y: number }> }>(
        `/latent-space/station-windows/${domain}/${stationId}?space=${space}`,
      ),
```

- [ ] **Step 4: Add hook**

In `frontend/src/hooks/useLatentSpace.ts`, add:

```typescript
export function useStationWindows(domain: string, stationId: string | null, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'station-windows', domain, stationId, space],
    queryFn: () => api.latentSpace.stationWindows(domain, stationId!, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!stationId,
  })
}
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/utils/latent_space.py api/routers/latent_space.py frontend/src/lib/api.ts frontend/src/hooks/useLatentSpace.ts
git commit -m "feat(latent-space): add station-windows endpoint with on-the-fly UMAP"
```

---

### Task 2: ClusterLegendBar component

**Files:**
- Create: `/home/ringuet/time-serie-explo/frontend/src/components/latent-space/ClusterLegendBar.tsx`

- [ ] **Step 1: Create ClusterLegendBar**

```tsx
import { useMemo, useState } from 'react'

interface ClusterInfo {
  id: number
  color: string
  count: number
}

interface ClusterLegendBarProps {
  clusters: ClusterInfo[]
  selectedCluster: number | null
  onSelectCluster: (id: number | null) => void
}

const MAX_VISIBLE = 20

export function ClusterLegendBar({ clusters, selectedCluster, onSelectCluster }: ClusterLegendBarProps) {
  const [expanded, setExpanded] = useState(false)

  const sorted = useMemo(
    () => [...clusters].sort((a, b) => b.count - a.count),
    [clusters],
  )

  const noise = sorted.find((c) => c.id === -1)
  const real = sorted.filter((c) => c.id !== -1)
  const visible = expanded ? real : real.slice(0, MAX_VISIBLE)
  const hidden = real.length - visible.length

  function handleClick(id: number) {
    onSelectCluster(selectedCluster === id ? null : id)
  }

  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1 px-1 scrollbar-thin">
      {/* All button */}
      <button
        onClick={() => onSelectCluster(null)}
        className={`shrink-0 px-2 py-0.5 rounded text-[10px] border transition-colors ${
          selectedCluster === null
            ? 'border-accent-cyan/50 text-accent-cyan bg-accent-cyan/10'
            : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        All ({clusters.reduce((s, c) => s + c.count, 0)})
      </button>

      {visible.map((c) => (
        <button
          key={c.id}
          onClick={() => handleClick(c.id)}
          className={`shrink-0 flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
            selectedCluster === c.id
              ? 'border-accent-cyan/50 text-text-primary bg-accent-cyan/10'
              : selectedCluster !== null
                ? 'border-white/5 text-text-muted opacity-40'
                : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
          <span>{c.id}</span>
          <span className="text-text-muted">({c.count})</span>
        </button>
      ))}

      {hidden > 0 && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="shrink-0 px-2 py-0.5 rounded text-[10px] border border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
        >
          +{hidden} more
        </button>
      )}

      {noise && (
        <button
          onClick={() => handleClick(-1)}
          className={`shrink-0 flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
            selectedCluster === -1
              ? 'border-accent-cyan/50 text-text-primary bg-accent-cyan/10'
              : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          <span className="w-2 h-2 rounded-full shrink-0 bg-gray-600" />
          <span>Noise</span>
          <span className="text-text-muted">({noise.count})</span>
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/ClusterLegendBar.tsx
git commit -m "feat(latent-space): add ClusterLegendBar component"
```

---

### Task 3: WindowScatter component

**Files:**
- Create: `/home/ringuet/time-serie-explo/frontend/src/components/latent-space/WindowScatter.tsx`

- [ ] **Step 1: Create WindowScatter**

```tsx
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import { useStationWindows } from '@/hooks/useLatentSpace'

interface WindowScatterProps {
  domain: string
  stationId: string
  space: string
}

export function WindowScatter({ domain, stationId, space }: WindowScatterProps) {
  const { data, isLoading } = useStationWindows(domain, stationId, space)

  if (isLoading) {
    return (
      <div className="h-48 flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const windows = (data as Record<string, unknown>)?.windows as
    | Array<{ window_start: string; window_end: string; umap_x: number; umap_y: number }>
    | undefined

  if (!windows || windows.length === 0) {
    return <p className="text-text-muted text-xs py-2">No windows available</p>
  }

  const years = windows.map((w) => parseInt(w.window_start.slice(0, 4)))
  const minYear = Math.min(...years)
  const maxYear = Math.max(...years)

  return (
    <Plot
      data={[
        {
          type: 'scattergl',
          x: windows.map((w) => w.umap_x),
          y: windows.map((w) => w.umap_y),
          mode: 'markers',
          marker: {
            size: 6,
            color: years,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
              thickness: 10,
              len: 0.8,
              tickfont: { color: '#9ca3af', size: 9 },
              title: { text: 'Year', font: { color: '#9ca3af', size: 9 } },
            },
          },
          customdata: windows.map((w) => [`${w.window_start} → ${w.window_end}`]),
          hovertemplate: '<b>%{customdata[0]}</b><extra></extra>',
        },
      ]}
      layout={{
        ...darkLayout,
        margin: { t: 5, r: 40, b: 5, l: 5 },
        xaxis: { visible: false },
        yaxis: { visible: false },
        hovermode: 'closest',
        height: 180,
      }}
      config={{ ...plotlyConfig, displayModeBar: false }}
      useResizeHandler
      style={{ width: '100%', height: '180px' }}
    />
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/WindowScatter.tsx
git commit -m "feat(latent-space): add WindowScatter component for per-station temporal view"
```

---

## Chunk 2: Rework Existing Components

### Task 4: EmbeddingScatter — remove native legend, domain hover, site highlight

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/components/latent-space/EmbeddingScatter.tsx`

- [ ] **Step 1: Add `domain` and `highlightedSite` props**

Update interface:
```typescript
interface EmbeddingScatterProps {
  points: EmbeddingPoint[]
  mode: '2d' | '3d'
  colorBy: string
  domain: 'piezo' | 'hydro'
  highlightedSite?: string | null  // libelle_eh or nom_cours_eau value
  onPointClick?: (id: string) => void
  loading?: boolean
  className?: string
}
```

- [ ] **Step 2: Replace buildHoverMeta with domain-specific hover**

```typescript
function buildHoverMeta(p: EmbeddingPoint, domain: 'piezo' | 'hydro'): string {
  const m = p.metadata
  if (domain === 'piezo') {
    const parts = [
      m.libelle_eh && `${m.libelle_eh}`,
      m.departement && `${m.departement}`,
      `Cluster ${p.cluster_label}`,
      m.n_windows && `${m.n_windows} windows`,
    ].filter(Boolean)
    return parts.join('<br>')
  } else {
    const parts = [
      m.nom_cours_eau && `${m.nom_cours_eau}`,
      m.departement && `${m.departement}`,
      `Cluster ${p.cluster_label}`,
      m.n_windows && `${m.n_windows} windows`,
    ].filter(Boolean)
    return parts.join('<br>')
  }
}
```

- [ ] **Step 3: Disable native Plotly legend**

Set `showlegend: false` on ALL traces. Remove the `legend` property from layout.

- [ ] **Step 4: Add site highlight logic**

When `highlightedSite` is set, dim all points NOT matching the site. The site key is `libelle_eh` (piezo) or `nom_cours_eau` (hydro):

```typescript
// Before building traces, split into site-matched and others
const siteKey = domain === 'piezo' ? 'libelle_eh' : 'nom_cours_eau'
const isSiteHighlighted = highlightedSite != null

// In the categorical trace builder, if isSiteHighlighted:
// - Points matching the site: full opacity
// - Points NOT matching: opacity 0.1, gray color
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/latent-space/EmbeddingScatter.tsx
git commit -m "feat(latent-space): domain hover, no native legend, site highlight"
```

---

### Task 5: FilterPanel — compact layout, active toggle, advanced collapsible

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/components/latent-space/FilterPanel.tsx`

- [ ] **Step 1: Reduce width and restructure**

Change `w-60` (240px) to `w-48` (192px). Restructure layout:

1. Station search (compact)
2. Active stations toggle (new)
3. Color by selector
4. Cluster filter dropdown
5. Advanced filters (collapsible) — EH fields only for piezo

- [ ] **Step 2: Add `onlyActive` toggle prop**

```typescript
interface FilterPanelProps {
  // ... existing props
  onlyActive: boolean
  onOnlyActiveChange: (v: boolean) => void
}
```

Add toggle UI:
```tsx
<label className="flex items-center gap-2 text-xs text-text-muted cursor-pointer">
  <input
    type="checkbox"
    checked={onlyActive}
    onChange={(e) => onOnlyActiveChange(e.target.checked)}
    className="accent-accent-cyan"
  />
  Active stations only
</label>
```

- [ ] **Step 3: Collapse EH fields into "Advanced" section**

For piezo: wrap milieu_eh, theme_eh, etat_eh, nature_eh in a collapsible section (closed by default). Keep libelle_eh, departement, cluster_id visible.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/latent-space/FilterPanel.tsx
git commit -m "feat(latent-space): compact FilterPanel with active toggle and advanced collapsible"
```

---

### Task 6: StationDetail — enriched content, window scatter, filter-by-site

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/components/latent-space/StationDetail.tsx`

- [ ] **Step 1: Add new props**

```typescript
interface StationDetailProps {
  domain: 'piezo' | 'hydro'
  space: string
  stationId: string | null
  stationMeta?: Record<string, unknown>
  clusterLabel?: number | null
  onClose: () => void
  onNeighborClick?: (stationId: string) => void
  onFilterBySite?: () => void  // new
}
```

- [ ] **Step 2: Add cluster badge to header**

```tsx
<div className="flex items-center gap-2">
  <p className="text-text-primary text-sm font-medium break-all">{stationId}</p>
  {clusterLabel != null && clusterLabel >= 0 && (
    <span className="shrink-0 bg-accent-cyan/20 text-accent-cyan text-[10px] px-1.5 py-0.5 rounded">
      C{clusterLabel}
    </span>
  )}
</div>
```

- [ ] **Step 3: Add n_windows and n_days to metadata**

```tsx
<MetaLine label="Windows" value={stationMeta?.n_windows} />
<MetaLine label="Days" value={stationMeta?.n_days} />
```

- [ ] **Step 4: Add WindowScatter**

```tsx
import { WindowScatter } from './WindowScatter'

// After metadata section:
<div className="border-t border-white/5" />
<div>
  <p className="text-text-muted text-xs font-medium uppercase tracking-wide mb-1">
    Temporal evolution
  </p>
  <WindowScatter domain={domain} stationId={stationId} space={space} />
</div>
```

- [ ] **Step 5: Add "Filter by site" button**

```tsx
<button
  onClick={onFilterBySite}
  className="w-full py-1.5 text-xs text-accent-cyan border border-accent-cyan/30 rounded hover:bg-accent-cyan/10 transition-colors"
>
  Filter by {domain === 'piezo' ? 'aquifer' : 'waterway'}
</button>
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/latent-space/StationDetail.tsx
git commit -m "feat(latent-space): enriched StationDetail with window scatter and filter-by-site"
```

---

## Chunk 3: Wire Everything in LatentSpacePage

### Task 7: LatentSpacePage — integrate all changes

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/pages/LatentSpacePage.tsx`

- [ ] **Step 1: Add state for site highlight and cluster legend selection**

```typescript
const [highlightedSite, setHighlightedSite] = useState<string | null>(null)
const [legendCluster, setLegendCluster] = useState<number | null>(null)
const [onlyActive, setOnlyActive] = useState(false)
```

- [ ] **Step 2: Change click behavior — highlight site, don't filter**

Replace `handleStationSelect`:

```typescript
function handleStationSelect(stationId: string) {
  setSelectedStation(stationId)
  const station = allStations.find((s) => s.id === stationId)
  if (!station) return

  // Highlight the site (visual only, no filter change)
  const siteKey = domain === 'piezo' ? 'libelle_eh' : 'nom_cours_eau'
  const siteValue = station.metadata[siteKey]
  if (siteValue && typeof siteValue === 'string') {
    setHighlightedSite(siteValue)
  }
}
```

Add `handleFilterBySite`:

```typescript
function handleFilterBySite() {
  if (!selectedStation) return
  const station = allStations.find((s) => s.id === selectedStation)
  if (!station) return
  const siteKey = domain === 'piezo' ? 'libelle_eh' : 'nom_cours_eau'
  const siteValue = station.metadata[siteKey]
  if (siteValue && typeof siteValue === 'string') {
    setFilters({ [siteKey]: siteValue })
  }
}
```

- [ ] **Step 3: Build cluster info for ClusterLegendBar**

```typescript
const clusterInfo = useMemo(() => {
  const counts = new Map<number, number>()
  for (const p of scatterPoints) {
    counts.set(p.cluster_label, (counts.get(p.cluster_label) ?? 0) + 1)
  }
  return Array.from(counts.entries()).map(([id, count]) => ({
    id,
    color: CATEGORICAL_COLORS[Math.abs(id) % CATEGORICAL_COLORS.length] ?? '#4b5563',
    count,
  }))
}, [scatterPoints])
```

(Import CATEGORICAL_COLORS from EmbeddingScatter or define shared)

- [ ] **Step 4: Integrate ClusterLegendBar below scatter**

```tsx
<ClusterLegendBar
  clusters={clusterInfo}
  selectedCluster={legendCluster}
  onSelectCluster={setLegendCluster}
/>
```

- [ ] **Step 5: Pass domain, highlightedSite, legendCluster to EmbeddingScatter**

The scatter needs to know which cluster is selected from the legend to highlight it. Merge `legendCluster` into the highlight logic.

- [ ] **Step 6: Pass onlyActive to FilterPanel, filter stations**

```typescript
// Filter by active status
const activeStations = useMemo(() => {
  if (!onlyActive) return stations
  // Approximate: stations with n_windows >= 3 have at least ~2 years of data
  return stations.filter((s) => (s.n_windows ?? 0) >= 3)
}, [stations, onlyActive])
```

Use `activeStations` instead of `stations` for scatter points and filter panel.

- [ ] **Step 7: Pass space and clusterLabel to StationDetail**

```tsx
<StationDetail
  domain={domain}
  space={space}
  stationId={selectedStation}
  stationMeta={selectedStationMeta}
  clusterLabel={scatterPoints.find(p => p.id === selectedStation)?.cluster_label}
  onClose={() => { setSelectedStation(null); setHighlightedSite(null) }}
  onNeighborClick={handleStationSelect}
  onFilterBySite={handleFilterBySite}
/>
```

- [ ] **Step 8: Clear highlight on empty click**

In EmbeddingScatter, the Plotly `onClick` handler should also detect clicks on empty space. Add Plotly `onRelayout` or handle missing point in click:

```typescript
// In LatentSpacePage, when scatter background clicked:
function handleScatterClick(id: string | null) {
  if (id) {
    handleStationSelect(id)
  } else {
    setSelectedStation(null)
    setHighlightedSite(null)
  }
}
```

- [ ] **Step 9: Commit**

```bash
git add frontend/src/pages/LatentSpacePage.tsx
git commit -m "feat(latent-space): wire site highlight, cluster legend, active filter, window scatter"
```

---

## Final: Build, Test, Verify

- [ ] **Rebuild Docker**

```bash
cd /home/ringuet/time-serie-explo
docker compose up -d --build
```

- [ ] **Verify in browser**

1. Open Latent Space → Piezometry → Multivariate
2. Verify custom cluster legend bar below scatter (no native Plotly legend)
3. Click a cluster badge → only that cluster highlighted
4. Hover a point → domain-specific info (aquifer, dept, cluster, n_windows)
5. Click a station → site highlighted (all stations from same aquifer bright, others dim)
6. StationDetail shows: metadata + window scatter (colored by year) + similar stations
7. "Filter by aquifer" button applies persistent filter
8. FilterPanel: compact, "Active only" toggle works
9. Switch to Hydro → same behavior with nom_cours_eau
10. Switch to Univariate → everything works with ROCKET embeddings
