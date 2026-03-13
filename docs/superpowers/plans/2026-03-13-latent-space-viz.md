# Latent Space Visualization + Lab Page — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive UMAP/clustering visualization for SoftCLT embeddings, under a new "Laboratoire" page that groups experimental features.

**Architecture:** New `/lab` nested route layout with 3 tabs (Espace Latent, Contrefactuel, Détection Pompage). Backend loads embeddings from brgm-postgres `ml.*` tables, computes UMAP + clustering on-demand. Frontend uses Plotly Scattergl/Scatter3d with filtering and coloration by hydrogeological metadata.

**Tech Stack:** React 19, React Router 7, TanStack Query 5, Plotly.js (Scattergl/Scatter3d), FastAPI, SQLAlchemy async, pgvector, umap-learn, sklearn.cluster.HDBSCAN

**Spec:** `docs/superpowers/specs/2026-03-13-latent-space-viz-design.md`

---

## Chunk 1: Backend Foundation

### Task 1: Add umap-learn dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add umap-learn to dependencies**

In `pyproject.toml`, add `"umap-learn>=0.5.0"` to the main dependencies list (after the `ruptures` line ~41):

```toml
    "ruptures>=1.1.0",
    "umap-learn>=0.5.0",
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat(latent-space): add umap-learn dependency"
```

### Task 2: Pydantic schemas for latent space API

**Files:**
- Create: `api/schemas/latent_space.py`

- [ ] **Step 1: Create schema file**

```python
"""Pydantic schemas for latent space API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingFilters(BaseModel):
    station_ids: list[str] | None = None
    libelle_eh: str | None = None
    milieu_eh: str | None = None
    theme_eh: str | None = None
    etat_eh: str | None = None
    nature_eh: str | None = None
    departement: str | None = None
    region: str | None = None
    cluster_id: int | None = None


class UMAPParams(BaseModel):
    n_components: Literal[2, 3] = 2
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: str = "cosine"


class HDBSCANParams(BaseModel):
    min_cluster_size: int = Field(default=10, ge=2)
    min_samples: int = Field(default=5, ge=1)


class KMeansParams(BaseModel):
    n_clusters: int = Field(default=8, ge=2, le=100)


class ClusteringParams(BaseModel):
    method: Literal["hdbscan", "kmeans"] = "hdbscan"
    n_umap_dims: int = Field(default=10, ge=2, le=50)
    hdbscan: HDBSCANParams = HDBSCANParams()
    kmeans: KMeansParams = KMeansParams()


class ComputeRequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    embeddings_type: Literal["stations", "windows"] = "stations"
    filters: EmbeddingFilters = EmbeddingFilters()
    year_min: int | None = None
    year_max: int | None = None
    season: Literal["DJF", "MAM", "JJA", "SON"] | None = None
    umap: UMAPParams = UMAPParams()
    clustering: ClusteringParams = ClusteringParams()


class StationMetadata(BaseModel):
    milieu_eh: str | None = None
    theme_eh: str | None = None
    etat_eh: str | None = None
    nature_eh: str | None = None
    libelle_eh: str | None = None
    departement: str | None = None
    nom_departement: str | None = None
    region: str | None = None
    altitude: float | None = None
    # Hydro-specific
    nom_cours_eau: str | None = None
    statut_station: str | None = None


class StationPoint(BaseModel):
    id: str
    umap_2d: list[float] | None = None
    umap_3d: list[float] | None = None
    cluster_id: int | None = None
    n_windows: int | None = None
    metadata: StationMetadata = StationMetadata()


class StationsResponse(BaseModel):
    stations: list[StationPoint]


class ComputedPoint(BaseModel):
    id: str
    coords: list[float]
    cluster_label: int = -1
    window_start: str | None = None
    window_end: str | None = None
    metadata: StationMetadata = StationMetadata()


class ComputeResponse(BaseModel):
    points: list[ComputedPoint]
    n_clusters: int
    subsampled: bool = False
    subsampled_from: int | None = None
    metrics: dict | None = None


class SimilarStation(BaseModel):
    id: str
    distance: float
    cluster_id: int | None = None
    metadata: StationMetadata = StationMetadata()


class SimilarResponse(BaseModel):
    query_id: str
    neighbors: list[SimilarStation]
```

- [ ] **Step 2: Commit**

```bash
git add api/schemas/latent_space.py
git commit -m "feat(latent-space): add Pydantic schemas"
```

### Task 3: Backend utils — latent_space.py

**Files:**
- Create: `dashboard/utils/latent_space.py`

This module is pure Python (no framework imports). It provides functions to load embeddings from brgm-postgres, compute UMAP projections, run clustering, and find similar stations.

- [ ] **Step 1: Create the utils module**

Key functions:

`build_station_query(domain, filters)` — Builds SQL for station embeddings + metadata JOIN. For piezo: JOIN `gold.dim_piezo_stations` + `gold.int_station_era5_mapping` on `code_bss`. For hydro: JOIN `gold.dim_hydro_stations` on `code_station`. Apply WHERE clauses from filters (all AND).

`build_window_query(domain, filters, year_min, year_max, season)` — Builds SQL for window embeddings with same filters. Season filter: DJF = months 12,1,2; MAM = 3,4,5; JJA = 6,7,8; SON = 9,10,11 (based on `window_start` month).

`parse_pgvector(raw)` — Converts pgvector string `[0.1,0.2,...]` to numpy array.

`compute_umap(embeddings_matrix, n_components, n_neighbors, min_dist, metric)` — Calls `umap.UMAP(...).fit_transform(embeddings_matrix)`. Returns numpy array of coordinates.

`compute_clustering(embeddings_matrix, method, params, n_umap_dims)` — If HDBSCAN: UMAP to `n_umap_dims` dims first, then `sklearn.cluster.HDBSCAN(min_cluster_size=..., min_samples=...)`. If KMeans: `sklearn.cluster.KMeans(n_clusters=...)` directly on input. Returns labels array + metrics dict (silhouette_score if >1 cluster).

`subsample_stratified(ids, embeddings, metadata, max_points, group_key)` — Random stratified subsample by station_id. Returns subsampled arrays + flag.

`build_similar_query(domain, station_id, k)` — SQL using pgvector `<=>` operator with the station's embedding as reference, `ORDER BY distance LIMIT k`.

All SQL uses `sqlalchemy.text()` with `:param` parameterized queries. No string formatting for identifiers — domain is validated against `{"piezo", "hydro"}` set, table/column names are hardcoded per domain.

- [ ] **Step 2: Commit**

```bash
git add dashboard/utils/latent_space.py
git commit -m "feat(latent-space): add backend utils for embeddings, UMAP, clustering"
```

### Task 4: FastAPI router — latent_space.py

**Files:**
- Create: `api/routers/latent_space.py`
- Modify: `api/main.py` (add import + include_router at line ~14 and ~66)

- [ ] **Step 1: Create the router**

Pattern follows `api/routers/counterfactual.py`. Uses `get_brgm_db()` from `api/database.py` for all DB queries.

4 endpoints:

`GET /stations/{domain}` — Validate domain in `{"piezo", "hydro"}`. Execute `build_station_query` via `session.execute()`. Parse pgvector embeddings, build `StationPoint` list. Return `StationsResponse`. `libelle_eh` grouping: count occurrences, keep top 12, replace rest with "Autre".

`POST /compute` — Load embeddings (stations or windows) based on `embeddings_type` and `filters`. Subsample if >15000 points. Run `compute_umap` for scatter coords. Run `compute_clustering`. Build `ComputeResponse` with coords + labels + metadata. Use `asyncio.to_thread()` for the blocking UMAP/clustering computation.

`GET /similar/{domain}/{station_id}` — Execute `build_similar_query`. Return `SimilarResponse`.

Router prefix: `/api/v1/latent-space`, tags: `["latent-space"]`.

- [ ] **Step 2: Register router in main.py**

Add import at line ~14:
```python
from api.routers import latent_space
```

Add `app.include_router(latent_space.router)` after line ~66 (after pumping_detection).

- [ ] **Step 3: Commit**

```bash
git add api/routers/latent_space.py api/main.py
git commit -m "feat(latent-space): add FastAPI router with stations, compute, similar endpoints"
```

### Task 5: Backend unit tests

**Files:**
- Create: `tests/latent_space/test_utils.py`
- Create: `tests/latent_space/__init__.py`

- [ ] **Step 1: Write tests for compute functions**

Tests that run without DB (mock-free, pure math):

`test_compute_umap_2d` — Feed 100 random 320d vectors, assert output shape is (100, 2).

`test_compute_umap_3d` — Same with n_components=3, assert shape (100, 3).

`test_compute_clustering_hdbscan` — Feed 200 vectors in 3 known clusters (well-separated gaussians in 10d), assert labels contain at least 2 distinct non-noise labels.

`test_compute_clustering_kmeans` — Same data, k=3, assert exactly 3 labels {0,1,2}.

`test_subsample_stratified` — 1000 points from 10 stations (100 each), subsample to 100. Assert exactly 100 returned, all 10 stations represented.

`test_subsample_no_op` — 50 points, max_points=100. Assert all 50 returned, subsampled=False.

Use lazy imports for `umap` and `sklearn` — skip tests if not available (for host without scientific deps):
```python
pytest.importorskip("umap")
```

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/latent_space/ -v
```

Expected: all pass on host for sklearn tests, umap tests may skip if umap-learn not installed.

- [ ] **Step 3: Commit**

```bash
git add tests/latent_space/
git commit -m "test(latent-space): add unit tests for UMAP and clustering utils"
```

## Chunk 2: Frontend Navigation — Lab Layout

### Task 6: LabLayout with tabs

**Files:**
- Create: `frontend/src/pages/LabLayout.tsx`

- [ ] **Step 1: Create LabLayout component**

Layout wrapper with horizontal tab bar and `<Outlet />`. Uses `NavLink` from react-router-dom for active tab styling. 3 tabs:
- "Espace Latent" → `/lab/latent-space`
- "Contrefactuel" → `/lab/counterfactual`
- "Détection Pompage" → `/lab/pumping-detection`

Tab styling: `border-b-2 border-accent-cyan text-text-primary` for active, `text-text-muted hover:text-text-secondary` for inactive. Dark card background `bg-bg-card`.

Uses Lucide icons: `Waypoints` (latent space), `GitCompareArrows` (counterfactual), `Droplets` (pumping).

```tsx
import { NavLink, Outlet } from 'react-router-dom'
import { Waypoints, GitCompareArrows, Droplets } from 'lucide-react'
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/LabLayout.tsx
git commit -m "feat(lab): add LabLayout with tab navigation"
```

### Task 7: Update routes.tsx — nested /lab routes + redirects

**Files:**
- Modify: `frontend/src/routes.tsx`

- [ ] **Step 1: Modify routes**

Changes to make:
1. Add lazy import for `LabLayout` and `LatentSpacePage`
2. Remove standalone `/counterfactual` and `/pumping-detection` routes
3. Add `/lab` parent route with `LabLayout` element and children:
   - index → `Navigate` redirect to `/lab/latent-space`
   - `latent-space` → `LatentSpacePage`
   - `counterfactual` → `CounterfactualPage` (reuse existing lazy import)
   - `pumping-detection` → `PumpingDetectionPage` (reuse existing lazy import)
4. Add redirect routes for backwards compatibility:
   - `/counterfactual` → `Navigate` to `/lab/counterfactual`
   - `/pumping-detection` → `Navigate` to `/lab/pumping-detection`

- [ ] **Step 2: Commit**

```bash
git add frontend/src/routes.tsx
git commit -m "feat(lab): add nested /lab routes with redirects"
```

### Task 8: Update TopNav — single Laboratoire entry

**Files:**
- Modify: `frontend/src/components/layout/TopNav.tsx`

- [ ] **Step 1: Modify NAV_ITEMS**

In `TopNav.tsx` lines 16-24 (`NAV_ITEMS` array):
1. Remove the `counterfactual` entry (`{ to: '/counterfactual', icon: GitCompareArrows, label: 'Contrefactuel' }`)
2. Remove the `pumping-detection` entry (`{ to: '/pumping-detection', icon: Droplets, label: 'Détection Pompage' }`)
3. Add single entry: `{ to: '/lab', icon: FlaskConical, label: 'Laboratoire' }`
4. Add `FlaskConical` to the lucide-react import

For the NavLink, ensure `/lab` matches as active for any `/lab/*` sub-route. The existing NavLink uses `end={to === '/'}` which means only the root uses exact matching — `/lab` will correctly match `/lab/latent-space` etc.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/layout/TopNav.tsx
git commit -m "feat(lab): replace separate nav items with single Laboratoire entry"
```

## Chunk 3: Frontend — API Client & Hook

### Task 9: API client namespace

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add latentSpace namespace**

Add after the `pumpingDetection` namespace (around line 259):

```typescript
  latentSpace: {
    stations: (domain: string) =>
      fetchJson<{ stations: Array<Record<string, unknown>> }>(`/latent-space/stations/${domain}`),
    compute: (body: Record<string, unknown>) =>
      postJson<Record<string, unknown>>('/latent-space/compute', body, 120_000),
    similar: (domain: string, stationId: string, k: number = 10) =>
      fetchJson<Record<string, unknown>>(`/latent-space/similar/${domain}/${stationId}?k=${k}`),
  },
```

Note: `compute` uses 120s timeout (3rd argument to `postJson`).

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(latent-space): add API client namespace"
```

### Task 10: React Query hook — useLatentSpace.ts

**Files:**
- Create: `frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1: Create hook file**

3 query hooks + 1 mutation:

`useStationEmbeddings(domain: string)` — `useQuery` calling `api.latentSpace.stations(domain)`. Key: `['latent-space', 'stations', domain]`. `staleTime: 5 * 60 * 1000` (5 min, matches project convention). `enabled: !!domain`.

`useSimilarStations(domain: string, stationId: string | null)` — `useQuery` calling `api.latentSpace.similar(domain, stationId!, 10)`. Key: `['latent-space', 'similar', domain, stationId]`. `enabled: !!stationId`.

`useComputeUMAP()` — `useMutation` calling `api.latentSpace.compute(body)`. No cache key (on-demand compute, result managed in local state).

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/useLatentSpace.ts
git commit -m "feat(latent-space): add React Query hooks"
```

## Chunk 4: Frontend — Visualization Components

### Task 11: EmbeddingScatter component

**Files:**
- Create: `frontend/src/components/latent-space/EmbeddingScatter.tsx`

- [ ] **Step 1: Create the scatter plot component**

Props:
```typescript
interface EmbeddingScatterProps {
  points: Array<{
    id: string
    coords: [number, number] | [number, number, number]
    cluster_label: number
    metadata: Record<string, unknown>
    highlighted: boolean
  }>
  mode: '2d' | '3d'
  colorBy: string
  onPointClick?: (id: string) => void
  loading?: boolean
  className?: string
}
```

Implementation:
- Uses `react-plotly.js` with `darkLayout` and `plotlyConfig` from `@/lib/plotly-theme`
- In 2D mode: `Scattergl` trace type (WebGL, handles 5K+ points)
- In 3D mode: `Scatter3d` trace type
- Color logic:
  - For categorical attributes: group points by `metadata[colorBy]`, one trace per category with distinct color from Plotly `D3` palette
  - For `altitude`: single trace with `marker.color` = altitude values, `colorscale: 'Viridis'`
  - Non-highlighted points: rendered as separate trace with `opacity: 0.15`, `color: '#4b5563'`
- Hover template: `id + ": " + metadata values` (nappe, milieu, etc.)
- Click handler: `plotly_click` event → extract point `customdata[0]` (station id) → call `onPointClick`
- Loading overlay: semi-transparent div with spinner text "Calcul UMAP en cours..."
- `useResizeHandler` for responsiveness

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/EmbeddingScatter.tsx
git commit -m "feat(latent-space): add EmbeddingScatter Plotly component with 2D/3D support"
```

### Task 12: FilterPanel component

**Files:**
- Create: `frontend/src/components/latent-space/FilterPanel.tsx`

- [ ] **Step 1: Create the filter sidebar**

Props:
```typescript
interface FilterPanelProps {
  domain: 'piezo' | 'hydro'
  stations: StationPoint[]  // full dataset for extracting distinct values
  filters: EmbeddingFilters
  onFiltersChange: (filters: EmbeddingFilters) => void
  colorBy: string
  onColorByChange: (attr: string) => void
}
```

Implementation:
- Extracts distinct values for each filterable attribute from the `stations` array
- For piezo: dropdowns for `libelle_eh`, `milieu_eh`, `theme_eh`, `etat_eh`, `departement`, `region`, `cluster_id`
- For hydro: dropdowns for `nom_cours_eau`, `departement`, `cluster_id` (reduced set)
- Each dropdown is a `<select>` with `bg-bg-input text-text-primary border border-white/10 rounded-lg` styling
- First option: "Tous" (clears the filter)
- `libelle_eh` dropdown: top 12 values by count + "Autre" option
- Color-by dropdown at the bottom: same attribute list + `cluster_id` + `altitude`
- "Reset filtres" button to clear all

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/FilterPanel.tsx
git commit -m "feat(latent-space): add FilterPanel with domain-adaptive dropdowns"
```

### Task 13: UMAPControls component

**Files:**
- Create: `frontend/src/components/latent-space/UMAPControls.tsx`

- [ ] **Step 1: Create the controls panel**

Props:
```typescript
interface UMAPControlsProps {
  mode: '2d' | '3d'
  onModeChange: (mode: '2d' | '3d') => void
  level: 'stations' | 'windows'
  onLevelChange: (level: 'stations' | 'windows') => void
  umapParams: { n_neighbors: number; min_dist: number }
  onUmapParamsChange: (params: { n_neighbors: number; min_dist: number }) => void
  clusteringParams: { method: 'hdbscan' | 'kmeans'; min_cluster_size: number; min_samples: number; n_clusters: number; n_umap_dims: number }
  onClusteringParamsChange: (params: ...) => void
  onRecalculate: () => void
  onReset: () => void
  isComputing: boolean
  yearRange?: [number, number]
  onYearRangeChange?: (range: [number, number]) => void
}
```

Implementation:
- Toggle buttons for 2D/3D mode and Stations/Windows level
- Number inputs for UMAP params: `n_neighbors` (2-200), `min_dist` (0.0-1.0 step 0.05)
- Clustering method dropdown (HDBSCAN / K-Means)
- Conditional params: if HDBSCAN → `min_cluster_size`, `min_samples`, `n_umap_dims`; if KMeans → `n_clusters`
- "Recalculer" button (`accent-cyan` bg, disabled when `isComputing`)
- "Reset" button (ghost style)
- When `level === 'windows'`: show year range slider (min/max from data)
- Season toggle buttons: DJF / MAM / JJA / SON (multi-select or single)
- All inputs use project styling: `bg-bg-input border border-white/10 rounded-lg`

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/UMAPControls.tsx
git commit -m "feat(latent-space): add UMAPControls with UMAP/clustering parameter inputs"
```

### Task 14: StationDetail panel (KNN)

**Files:**
- Create: `frontend/src/components/latent-space/StationDetail.tsx`

- [ ] **Step 1: Create the detail panel**

Props:
```typescript
interface StationDetailProps {
  domain: 'piezo' | 'hydro'
  stationId: string | null
  onClose: () => void
}
```

Implementation:
- Uses `useSimilarStations(domain, stationId)` hook
- Shows selected station ID as header
- List of K nearest neighbors with:
  - Station ID
  - Cosine distance (formatted to 4 decimals)
  - Key metadata attributes (nappe, milieu, département)
- Loading skeleton while fetching
- Close button (X icon top-right)
- Slide-in panel from right side, or inline card below scatter
- Container styling: `bg-bg-card rounded-xl border border-white/5 p-4`

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/latent-space/StationDetail.tsx
git commit -m "feat(latent-space): add StationDetail KNN panel"
```

## Chunk 5: Frontend — Page Assembly

### Task 15: LatentSpacePage — orchestration

**Files:**
- Create: `frontend/src/pages/LatentSpacePage.tsx`

- [ ] **Step 1: Create the main page**

State management:
```typescript
const [domain, setDomain] = useState<'piezo' | 'hydro'>('piezo')
const [filters, setFilters] = useState<EmbeddingFilters>({})
const [colorBy, setColorBy] = useState('cluster_id')
const [mode, setMode] = useState<'2d' | '3d'>('2d')
const [level, setLevel] = useState<'stations' | 'windows'>('stations')
const [umapParams, setUmapParams] = useState({ n_neighbors: 15, min_dist: 0.1 })
const [clusteringParams, setClusteringParams] = useState({ method: 'hdbscan', ... })
const [selectedStation, setSelectedStation] = useState<string | null>(null)
const [computedPoints, setComputedPoints] = useState(null)  // override from /compute
```

Data flow:
1. `useStationEmbeddings(domain)` → loads pre-computed scatter data
2. `useMemo` applies client-side `filters` to mark points as highlighted/not
3. If `computedPoints` exists (from recalculate), show those instead of pre-computed
4. `computeUMAP.mutateAsync(body)` → stores result in `computedPoints`
5. "Reset" clears `computedPoints` back to pre-computed data

Layout:
```
Domain switch (top bar)
├── FilterPanel (left sidebar, ~250px)
└── Main area
    ├── EmbeddingScatter (fills available space)
    ├── Controls bar (below scatter)
    │   ├── Color dropdown + 2D/3D toggle + Stations/Windows toggle
    │   ├── UMAP params + Clustering params
    │   └── Recalculate + Reset buttons
    └── StationDetail (right panel, conditional on selectedStation)
```

Error states:
- Loading: skeleton loader on scatter area
- BRGM error: "Base de données BRGM indisponible" card with retry button
- Empty filters: "Aucune station ne correspond aux filtres" centered message
- Subsampled: warning badge above scatter

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/LatentSpacePage.tsx
git commit -m "feat(latent-space): add LatentSpacePage with full state management"
```

## Chunk 6: Dagster Pre-computation (hubeau_data_integration)

### Task 16: Add UMAP 2D/3D columns to DB schema

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/docker/postgres/init.sql`

- [ ] **Step 1: Add columns to init.sql**

After the existing `ml.piezo_station_embeddings` CREATE TABLE (and similarly for hydro), add the umap columns. If tables already exist in production, also run ALTER TABLE manually or via migration.

Add to both piezo and hydro station_embeddings table definitions:
```sql
    umap_2d_x FLOAT,
    umap_2d_y FLOAT,
    umap_3d_x FLOAT,
    umap_3d_y FLOAT,
    umap_3d_z FLOAT,
```

- [ ] **Step 2: Run ALTER TABLE on running DB**

```bash
docker exec brgm-dlt-worker python -c "
import sqlalchemy as sa
engine = sa.create_engine('postgresql://...')
with engine.connect() as conn:
    for domain in ('piezo', 'hydro'):
        for col in ('umap_2d_x', 'umap_2d_y', 'umap_3d_x', 'umap_3d_y', 'umap_3d_z'):
            try:
                conn.execute(sa.text(f'ALTER TABLE ml.{domain}_station_embeddings ADD COLUMN {col} FLOAT'))
            except:
                pass
        conn.commit()
"
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add docker/postgres/init.sql
git commit -m "feat(ml): add UMAP 2D/3D columns to station_embeddings tables"
```

### Task 17: Update persistence.py — persist UMAP coordinates

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/persistence.py`

- [ ] **Step 1: Add update function**

Add `update_umap_coords(pg, domain, id_col, station_ids, umap_2d, umap_3d)` function. Executes UPDATE for each station with umap_2d_x/y and umap_3d_x/y/z values.

Use batch UPDATE via `executemany` or a single UPDATE with CASE expressions for efficiency.

- [ ] **Step 2: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/persistence.py
git commit -m "feat(ml): add UMAP coordinate persistence function"
```

### Task 18: Update ml_assets.py — compute UMAP 2D/3D in cluster asset

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/assets/ml_assets.py`

- [ ] **Step 1: Modify cluster assets**

In `ml_piezo_clusters` and `ml_hydro_clusters` assets, after the existing HDBSCAN clustering step:

1. Compute UMAP 320d → 2d (`n_neighbors=15, min_dist=0.1, metric='cosine'`)
2. Compute UMAP 320d → 3d (same params)
3. Call `update_umap_coords()` to persist

Add import: `from umap import UMAP`

```python
# After HDBSCAN clustering (existing code)
umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings)
umap_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings)
update_umap_coords(pg, domain, id_col, station_ids, umap_2d, umap_3d)
```

Add metadata output: `umap_2d_computed=True, umap_3d_computed=True`

- [ ] **Step 2: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/assets/ml_assets.py
git commit -m "feat(ml): compute and persist UMAP 2D/3D in cluster assets"
```

## Chunk 7: Integration & Docker Build

### Task 19: Docker build and full test

- [ ] **Step 1: Rebuild Docker**

```bash
cd /home/ringuet/time-serie-explo
docker compose up -d --build
```

- [ ] **Step 2: Verify backend imports**

```bash
docker compose exec backend python -c "
from dashboard.utils.latent_space import compute_umap, compute_clustering
from api.routers.latent_space import router
from api.schemas.latent_space import ComputeRequest, StationsResponse
print('All imports OK')
print('Routes:', [r.path for r in router.routes])
"
```

- [ ] **Step 3: Verify frontend builds**

The Docker build includes `npm run build` — if it passes, TypeScript is valid.

- [ ] **Step 4: Run backend tests in Docker**

```bash
docker compose exec backend pip install pytest
docker compose exec -w /app backend python -m pytest tests/latent_space/ -v
```

- [ ] **Step 5: Manual smoke test**

Open `http://localhost:49513/lab/latent-space` — verify:
- Lab layout renders with 3 tabs
- Domain switch works
- Scatter loads with pre-computed points (or shows error if UMAP coords not yet computed)
- Filters narrow down highlighted points
- Recalculate triggers compute endpoint

- [ ] **Step 6: Test redirects**

Navigate to `http://localhost:49513/counterfactual` — should redirect to `/lab/counterfactual`.
Navigate to `http://localhost:49513/pumping-detection` — should redirect to `/lab/pumping-detection`.

---

## Task Dependencies

```
Task 1 (deps) ─────────────────────────────────────────────────┐
Task 2 (schemas) ──┬── Task 4 (router) ──┬── Task 19 (build)  │
Task 3 (utils) ────┘                      │                     │
Task 5 (tests) ───────────────────────────┤                     │
Task 6 (LabLayout) ── Task 7 (routes) ── Task 8 (TopNav) ─────┤
Task 9 (api.ts) ── Task 10 (hook) ───────┤                     │
Task 11 (scatter) ─┬── Task 15 (page) ───┤                     │
Task 12 (filters) ─┤                      │                     │
Task 13 (controls)─┤                      │                     │
Task 14 (detail) ──┘                      │                     │
Task 16 (DB) ── Task 17 (persist) ── Task 18 (dagster) ───────┘
```

**Parallelizable groups:**
- **Group A** (backend): Tasks 1-5
- **Group B** (frontend nav): Tasks 6-8
- **Group C** (frontend API): Tasks 9-10
- **Group D** (frontend components): Tasks 11-14
- **Group E** (page assembly): Task 15 (depends on B, C, D)
- **Group F** (dagster): Tasks 16-18
- **Group G** (integration): Task 19 (depends on all)

Groups A, B, C, D, F can run in parallel. E depends on B+C+D. G depends on all.
