# Latent Space Pipeline Redesign

**Date**: 2026-03-17
**Status**: Draft
**Scope**: Backend pipeline (`dashboard/utils/latent_space.py`), API endpoints (`api/routers/latent_space.py`), Redis cache, frontend controls (`LatentSpacePage.tsx`, `UMAPControls.tsx`)

## Problem Statement

The current latent space pipeline is a monolithic block that couples dimensionality reduction, visualization, and clustering into a single computation. This produces poor visual results (Nike swoosh artifacts, compressed point masses) and inflexible clustering (3-4 clusters regardless of data richness). The user cannot adjust one step without recomputing everything.

Key failures:
- **UMAP used for both pre-reduction and visualization** — conflicting objectives (clustering needs `min_dist=0`, viz needs `min_dist≥0.1`)
- **Silhouette-based optimization** converges to trivial solutions (k=2-3)
- **PCA at 95% variance** too aggressive for low-intrinsic-dim data (320D → 4D for piezo/uni, losing fine structure)
- **No ability to recompute steps independently** — changing viz params forces full recompute
- **KMeans elbow pre-calculated but no chart** to help user choose k

## Architecture

Three decoupled units sharing one Redis cache per domain/space combo:

```
Embeddings (320D from PostgreSQL)
        │
    ┌───▼───┐
    │  PCA   │  configurable variance threshold + min dims plancher
    └───┬───┘
        │ reduced matrix (N × d)
   ┌────┴────┐
   │         │
┌──▼──┐  ┌──▼──────────┐
│ VIZ  │  │  CLUSTERING  │
│UMAP  │  │ HDBSCAN+DBCV │
│ 2D   │  │ KMeans elbow │
└──┬──┘  └──┬──────────┘
   │         │
   └────┬────┘
        ▼
   Redis cache (1 key per domain/space)
```

**Dependency rules:**
- Changing PCA → cascades to VIZ + CLUSTERING (both depend on reduced matrix)
- Changing VIZ → only recomputes UMAP 2D coords. Clustering untouched.
- Changing CLUSTERING → only recomputes labels. Coords untouched.

**Concurrency model:** Last-write-wins. Each recompute endpoint reads the full cache, updates its section(s), and writes back atomically (single Redis SET). If `recompute-pca` and `recompute-viz` run concurrently, the last to finish wins. This is acceptable because concurrent recomputes on the same space are a rare edge case (single user per deployment), and the user can always re-trigger.

**Ordering contract:** `station_ids`, `pca.reduced`, `viz.coords_2d`, `clustering.*.labels`, and `metadata` arrays are all index-aligned. Row i in every array corresponds to the same station. The ordering is determined by the SQL query at PCA computation time and must be preserved through all pipeline steps.

**Error handling:** Each recompute endpoint is atomic at its own scope. If `recompute-pca` succeeds PCA but fails during the VIZ cascade, it returns 500 and does NOT write a partial cache. The previous cache remains intact. The frontend retains its last-known-good state.

## Cache Structure

One Redis key per domain/space combo: `junon:latent-space:compute:{domain}:{space}`

```json
{
  "pca": {
    "variance_threshold": 0.95,
    "n_components": 15,
    "variance_explained": 0.9523,
    "cumvar_curve": [0.30, 0.50, 0.65, ...],
    "reduced": [[0.12, -0.34, ...], ...]
  },
  "viz": {
    "params": { "n_neighbors": 50, "min_dist": 0.3 },
    "trustworthiness": 0.97,
    "coords_2d": [[x, y], ...]
  },
  "clustering": {
    "hdbscan": {
      "params": { "min_cluster_size": 25, "min_samples": 5 },
      "labels": [0, 1, -1, 2, ...],
      "n_clusters": 12,
      "dbcv": 0.45,
      "noise_ratio": 0.15,
      "silhouette": 0.38
    },
    "kmeans_elbow": [
      { "k": 2, "labels": [...], "silhouette": 0.70, "inertia": 5000.0 },
      { "k": 3, "labels": [...], "silhouette": 0.55, "inertia": 3500.0 },
      ...
    ]
  },
  "station_ids": ["BSS001", "BSS002", ...],
  "metadata": [{"libelle_eh": "...", "departement": "..."}, ...]
}
```

**Serialization**: All floats in `pca.reduced` and `viz.coords_2d` are rounded to 6 decimal places before JSON serialization. This keeps payload size predictable (~5 MB per space).

Estimated size: ~5 MB per space (4200 stations × 50D reduced + coords + labels for 12 k-values). 4 spaces = ~20 MB total. Well within Redis 900 MB limit.

**Cache invalidation**: No TTL. Cache persists until explicitly replaced by a recompute or auto-tune action. If embeddings are re-trained in PostgreSQL, the user must click "Auto-tune" to refresh. A future admin endpoint for bulk cache flush can be added if needed.

## Unit 1: PCA Adaptive Pre-Reduction

**Purpose**: Reduce 320D embeddings to a manageable dimensionality that preserves enough structure for both clustering and UMAP.

**Configuration**:
- `variance_threshold`: slider 90-99%, default 95%
- **Floor**: `max(15, n_auto)` — never go below 15 dimensions, even if 4D captures 95% variance. Fine clustering structure lives in minor components.
- **Ceiling**: `min(n_99_percent, 100, N-1)` — never exceed 99% variance or 100 dims.

**Output**: `pca.reduced` matrix (N × d), stored in cache. This is the pivot data consumed by both VIZ and CLUSTERING.

**Diagnostics exposed in UI**: cumulative variance curve (scree plot) so user understands intrinsic dimensionality.

**Implementation**: `sklearn.decomposition.PCA`, deterministic (random_state=42).

## Unit 2: UMAP Visualization

**Purpose**: Project PCA-reduced data to 2D for visual display. Nothing else — UMAP does not participate in clustering.

**Configuration**:
- `n_neighbors`: slider 15-200, default 50 (global structure, avoids arc artifacts)
- `min_dist`: slider 0.05-1.0, default 0.3 (spacing for visual readability)

**No auto-tune**. Visual quality is subjective. We provide reasonable defaults and let the user adjust. Trustworthiness is displayed as diagnostic (informational, not an optimization target).

**Input**: `pca.reduced` from cache.
**Output**: `viz.coords_2d` array of [x, y] pairs.

**Implementation**: `umap.UMAP(random_state=42)` on PCA-reduced matrix.

## Unit 3: Clustering

Two methods, both pre-computed, switchable instantly in the frontend.

### HDBSCAN

- **Input**: `pca.reduced` from cache
- **Parameter optimization**: Optuna (40 trials, TPE sampler) maximizing DBCV score
- **DBCV computation**: `hdbscan.validity.validity_index` if available, otherwise composite fallback (silhouette on non-noise points with penalties for trivial solutions and excessive noise)
- **Cluster count guidance**: soft prior towards [5, 25] clusters for exploration. If DBCV-optimal result gives < 5 clusters, Optuna is re-run with constrained `min_cluster_size` range (lower values → more clusters). Not a hard constraint — if the data genuinely has 3 clusters, we accept it.
- **Exposed params**: `min_cluster_size`, `min_samples` (user can adjust and recompute)
- **Diagnostics**: DBCV score, silhouette, noise ratio, n_clusters

### KMeans Elbow

- **Input**: `pca.reduced` from cache (same space as HDBSCAN and UMAP — ensures visual coherence between cluster colors and scatter layout)
- **Pre-computed**: k = 2 to 25, all labels + silhouette + inertia stored
- **Default k selection**: kneedle algorithm on inertia curve (elbow detection), not max silhouette (which always gives k=2)
- **UI**: elbow chart showing inertia + silhouette curves, dropdown/slider to pick k
- **Switch is instant**: labels are in the cache, no API call needed

### Frontend clustering controls

- Toggle HDBSCAN / KMeans at the top of the scatter → re-colors points from cached labels
- KMeans mode: elbow chart + k selector
- HDBSCAN mode: shows DBCV, noise ratio, n_clusters
- "Recompute Clustering" button: re-runs HDBSCAN Optuna + KMeans elbow on current PCA reduced. Does not touch viz.

## UI Controls Layout

The bottom control bar reflects the 3 pipeline blocks, each with its own params and recompute button:

```
┌────────────────────┬─────────────────────┬──────────────────────────┐
│ PCA                │ Visualization       │ Clustering               │
│ variance: [95%] ── │ n_neighbors: [50] ─ │ method: [HDBSCAN ▾]     │
│ dims: 15           │ min_dist: [0.3] ──  │ mcs: [25]  ms: [5]      │
│ [Recompute PCA ↻]  │ [Recompute Viz ↻]   │ [Recompute Clust ↻]     │
│                    │ tw: 0.97            │ DBCV: 0.45  noise: 15%  │
└────────────────────┴─────────────────────┴──────────────────────────┘
```

- **Auto-tune** button: optimizes all 3 blocks at once (PCA adaptive + HDBSCAN Optuna + UMAP defaults). Equivalent to warmup. Replaces entire cache.
- **Elbow chart**: displayed when KMeans is selected (small panel near scatter or overlay).

## API Endpoints

### New endpoints (replace current monolithic `/compute`)

```
POST /api/v1/latent-space/recompute-pca
  Body: { domain, space, variance_threshold }
  → Recomputes PCA, cascades VIZ + CLUSTERING with current params
  → Updates full cache

POST /api/v1/latent-space/recompute-viz
  Body: { domain, space, n_neighbors, min_dist }
  → Loads pca.reduced from cache, recomputes UMAP 2D
  → Updates viz section of cache

POST /api/v1/latent-space/recompute-clustering
  Body: { domain, space, min_cluster_size?, min_samples? }
  → Loads pca.reduced from cache
  → Always runs both: HDBSCAN (Optuna or with explicit params if provided) + KMeans elbow (k=2..25)
  → Updates clustering section of cache

GET /api/v1/latent-space/cached/{domain}?space=multi
  → Returns full cache or 404

POST /api/v1/latent-space/auto-tune
  Body: { domain, space }
  → Full pipeline: loads embeddings from DB, PCA adaptive + HDBSCAN Optuna + KMeans elbow + UMAP defaults
  → Always force-replaces entire cache (unlike warmup which skips if cache exists)
  → Same code path as warmup, but triggered on-demand
```

**Async response pattern**: All recompute endpoints offload blocking work to a thread via `asyncio.to_thread` and block until completion, returning the updated cache section. `recompute-viz` is fast (~5s). `recompute-clustering` is moderate (~30-60s). `recompute-pca` cascades and can take 2-5 min. The frontend shows a loading spinner during the request. No SSE needed — these are standard request/response with extended timeout (120s client-side). If `recompute-pca` exceeds the timeout, the computation continues server-side and the cache is updated; the frontend can poll `GET /cached` to detect completion.

### Kept as-is

```
GET  /stations/{domain}          → station metadata + last_date (for active filter)
GET  /similar/{domain}/{id}      → pgvector cosine neighbors
GET  /station-windows/{domain}/{id} → per-station temporal windows
GET  /profiling/{domain}         → cluster profiling
```

### Removed

- `POST /compute` (old monolithic endpoint) — replaced by 3 granular endpoints
- `GET /clustering-runs/{domain}` and `GET /clustering-run/{run_id}` — pre-computed DB runs no longer used (table already purged)

## Warmup (Startup)

On backend startup, for each of the 4 domain/space combos:
1. Check if Redis cache exists → skip if yes
2. Load embeddings from PostgreSQL
3. Run full pipeline: PCA (default 95%, floor 15D) → UMAP (nn=50, md=0.3) → HDBSCAN Optuna + KMeans elbow
4. Store in Redis

All in background (non-blocking), sequential per space. Takes ~10-15 min total on CPU.

## Migration Notes

### Backend
- `dashboard/utils/latent_space.py`: replace `auto_tune_params` + `auto_tune_and_compute` with 3 separate functions: `compute_pca_adaptive`, `compute_umap_viz`, `compute_clustering_all`
- `api/routers/latent_space.py`: replace `/compute` with 3 granular endpoints, update `warmup_cache`, remove clustering run endpoints
- Remove `compute_clustering` function (old UMAP pre-reduction path)

### Frontend
- `LatentSpacePage.tsx`: remove `computedPoints` / `computeMutation` state (old monolithic compute). Cache is the single source of truth. Each recompute invalidates/refetches the cache query.
- `UMAPControls.tsx`: restructure into 3 sections (PCA, Viz, Clustering) with individual recompute buttons. Remove "UMAP Pre-reduction" section entirely.
- New component: `ElbowChart.tsx` — small Plotly chart showing inertia + silhouette vs k
- `useLatentSpace.ts`: replace `useComputeUMAP` mutation with 3 mutations (`useRecomputePCA`, `useRecomputeViz`, `useRecomputeClustering`)

### Frontend data join
- `GET /stations/{domain}` still provides per-station metadata and `last_date` (for active filter, search, detail panel). The UMAP coords and cluster labels now come exclusively from `GET /cached/{domain}`. The frontend joins these two sources via `station_ids` (cache) matched to station `id` (stations endpoint). The `stationLookup` map already exists for this purpose.

### Database
- `ml.clustering_runs` and `ml.clustering_labels` tables: already purged, can be dropped or left empty
- `ml.{piezo,hydro}_station_embeddings.umap_2d_x/y/umap_3d_*` columns: already NULL, no longer used by frontend
