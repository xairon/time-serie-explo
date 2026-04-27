# Latent Space Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic latent space pipeline with 3 decoupled units (PCA, VIZ, CLUSTERING) that can be recomputed independently, cached in Redis, and switched instantly in the frontend.

**Architecture:** PCA adaptive pre-reduction is the pivot. UMAP visualization and clustering (HDBSCAN + KMeans elbow) consume PCA output independently. One Redis cache key per domain/space combo stores all 3 outputs. Frontend derives scatter points from cache, switching clustering method/k without API calls.

**Tech Stack:** Python (sklearn PCA, HDBSCAN, KMeans, Optuna, UMAP), FastAPI, Redis (orjson serialization), React (TanStack Query, Plotly)

**Spec:** `docs/superpowers/specs/2026-03-17-latent-space-pipeline-design.md`

---

## File Map

### Backend — Pure Python pipeline (`dashboard/utils/`)

| File | Action | Responsibility |
|------|--------|---------------|
| `dashboard/utils/latent_space.py` | **Heavy modify** | Replace `auto_tune_params`, `auto_tune_and_compute`, `_pca_adaptive`, `_optimize_hdbscan`, `_kmeans_elbow`, `_optimize_umap_viz` with 3 clean functions: `compute_pca`, `compute_viz`, `compute_clustering`. Remove old `compute_clustering` (UMAP pre-reduction path). Keep SQL builders, similarity, window functions unchanged. |

### Backend — API layer (`api/`)

| File | Action | Responsibility |
|------|--------|---------------|
| `api/schemas/latent_space.py` | **Modify** | Add Pydantic models for 3 recompute request bodies. Remove `ComputeRequest`, `ComputeResponse`, clustering run schemas. |
| `api/routers/latent_space.py` | **Heavy modify** | Replace `/compute`, `/auto-tune` endpoints + `warmup_cache` with 3 recompute endpoints + `/auto-tune` (full pipeline) + updated `warmup_cache`. Remove clustering run endpoints. Keep stations, similar, windows, profiling. |
| `api/main.py` | **Minor modify** | Warmup call unchanged structurally (already calls `warmup_cache`). |

### Frontend — Hooks & API client

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/lib/api.ts` | **Modify** | Replace `compute()` with `recomputePca()`, `recomputeViz()`, `recomputeClustering()`, `autoTune()`. Keep `cached()`, `stations()`, etc. |
| `frontend/src/hooks/useLatentSpace.ts` | **Modify** | Replace `useComputeUMAP`, `useAutoTune` with `useRecomputePca`, `useRecomputeViz`, `useRecomputeClustering`, `useAutoTune`. Keep `useCachedCompute`, `useStationEmbeddings`, etc. |

### Frontend — Components

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/pages/LatentSpacePage.tsx` | **Heavy modify** | Remove `computedPoints`/`computeMutation` state. Cache is single source of truth. Derive scatter points from cache + selected method/k. Wire 3 recompute handlers. |
| `frontend/src/components/latent-space/UMAPControls.tsx` | **Rewrite** → `PipelineControls.tsx` | 3 sections (PCA, Viz, Clustering) with individual recompute buttons. Remove UMAP Pre-reduction. |
| `frontend/src/components/latent-space/ElbowChart.tsx` | **Create** | Small Plotly chart: inertia + silhouette vs k, click to select k. |

---

## Chunk 1: Backend Pipeline Functions

### Task 1: Replace old pipeline functions with `compute_pca`

**Files:**
- Modify: `dashboard/utils/latent_space.py`

- [ ] **Step 1:** Delete old functions `_pca_adaptive`, `_optimize_umap_viz`, `_optimize_hdbscan`, `_kmeans_elbow`, `_hdbscan_dbcv_score`, `auto_tune_params`, `auto_tune_and_compute` from `dashboard/utils/latent_space.py` (everything in the "Auto-tuning" section, lines ~540-800).

- [ ] **Step 2:** Write `compute_pca` function:

```python
def compute_pca(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 15,
) -> dict[str, Any]:
    """PCA adaptive pre-reduction.

    Returns dict with keys: reduced, n_components, variance_threshold,
    variance_explained, cumvar_curve, embedding_dim.
    """
    from sklearn.decomposition import PCA

    n_samples, n_dims = embeddings.shape
    max_components = min(100, n_dims, n_samples - 1)
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(embeddings)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Auto n_components from variance threshold
    n_auto = int(np.searchsorted(cumvar, variance_threshold) + 1)

    # Floor: never below min_components
    n_components = max(min_components, n_auto)

    # Ceiling: 99% variance or max_components
    n_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    n_components = min(n_components, n_99, max_components)

    reduced = pca.transform(embeddings)[:, :n_components]

    return {
        "reduced": [[round(float(v), 6) for v in row] for row in reduced],
        "n_components": n_components,
        "variance_threshold": variance_threshold,
        "variance_explained": round(float(cumvar[n_components - 1]), 4),
        "cumvar_curve": [round(float(v), 4) for v in cumvar[:max_components]],
        "embedding_dim": n_dims,
    }
```

- [ ] **Step 3:** Verify import works:
```bash
docker exec junon-backend python3 -c "from dashboard.utils.latent_space import compute_pca; print('OK')"
```

- [ ] **Step 4:** Commit.

### Task 2: Write `compute_viz` function

**Files:**
- Modify: `dashboard/utils/latent_space.py`

- [ ] **Step 1:** Write `compute_viz`:

```python
def compute_viz(
    pca_reduced: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.3,
) -> dict[str, Any]:
    """UMAP 2D visualization from PCA-reduced data.

    Returns dict with keys: coords_2d, params, trustworthiness.
    """
    from sklearn.manifold import trustworthiness

    coords = compute_umap(
        pca_reduced, n_components=2,
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
    )

    tw = 0.0
    k = min(n_neighbors, pca_reduced.shape[0] - 1)
    if k >= 2:
        try:
            tw = float(trustworthiness(pca_reduced, coords, n_neighbors=k, metric="cosine"))
        except Exception:
            pass

    return {
        "coords_2d": [[round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6)] for i in range(len(coords))],
        "params": {"n_neighbors": n_neighbors, "min_dist": min_dist},
        "trustworthiness": round(tw, 4),
    }
```

- [ ] **Step 2:** Verify import.

- [ ] **Step 3:** Commit.

### Task 3: Write `compute_clustering` function

**Files:**
- Modify: `dashboard/utils/latent_space.py`

- [ ] **Step 1:** Write DBCV scoring helper:

```python
def _dbcv_score(labels: np.ndarray, X: np.ndarray) -> float:
    """DBCV score. Uses hdbscan validity_index if available, else composite fallback."""
    n_clusters = len(set(labels.tolist()) - {-1})
    if n_clusters < 2:
        return -1.0

    try:
        from hdbscan.validity import validity_index
        return float(validity_index(X.astype(np.float64), labels))
    except (ImportError, Exception):
        pass

    # Composite fallback
    from sklearn.metrics import silhouette_score
    mask = labels != -1
    if mask.sum() < 10:
        return -1.0
    sil = float(silhouette_score(X[mask], labels[mask]))
    noise_ratio = (~mask).sum() / len(labels)
    penalty = 0.0
    if n_clusters < 4:
        penalty += 0.3 * (4 - n_clusters) / 3
    if n_clusters > 30:
        penalty += 0.1 * min((n_clusters - 30) / 30, 1.0)
    if noise_ratio > 0.4:
        penalty += 0.2
    elif noise_ratio > 0.25:
        penalty += 0.1
    return sil - penalty
```

- [ ] **Step 2:** Write main `compute_clustering_all` function:

```python
def compute_clustering_all(
    pca_reduced: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    n_optuna_trials: int = 40,
    kmeans_k_range: list[int] | None = None,
) -> dict[str, Any]:
    """Compute HDBSCAN (Optuna-optimized or explicit params) + KMeans elbow.

    All on pca_reduced for visual coherence.
    Returns dict with keys: hdbscan, kmeans_elbow.
    """
    import optuna
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    n = pca_reduced.shape[0]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- HDBSCAN ---
    if min_cluster_size is not None and min_samples is not None:
        # Explicit params: single run, no Optuna
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(pca_reduced)
        hdb_result = _build_hdbscan_result(labels, pca_reduced, min_cluster_size, min_samples)
    else:
        # Optuna optimization
        hdb_result = _optuna_hdbscan(pca_reduced, n_optuna_trials)

    # --- KMeans elbow ---
    if kmeans_k_range is None:
        kmeans_k_range = [k for k in range(2, 26) if k < n]
    kmeans_elbow = []
    for k in kmeans_k_range:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(pca_reduced)
            sil = float(silhouette_score(pca_reduced, lbls))
            kmeans_elbow.append({
                "k": k,
                "labels": lbls.tolist(),
                "silhouette": round(sil, 4),
                "inertia": round(float(km.inertia_), 2),
            })
        except Exception:
            continue

    return {"hdbscan": hdb_result, "kmeans_elbow": kmeans_elbow}
```

- [ ] **Step 3:** Write helper `_build_hdbscan_result` and `_optuna_hdbscan`:

```python
def _build_hdbscan_result(labels, X, mcs, ms):
    from sklearn.metrics import silhouette_score
    n_clusters = len(set(labels.tolist()) - {-1})
    noise_ratio = float((labels == -1).sum()) / len(labels)
    mask = labels != -1
    sil = float(silhouette_score(X[mask], labels[mask])) if n_clusters > 1 and mask.sum() > 10 else 0.0
    dbcv = _dbcv_score(labels, X)
    return {
        "params": {"min_cluster_size": int(mcs), "min_samples": int(ms)},
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "dbcv": round(dbcv, 4),
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": round(sil, 4),
    }


def _optuna_hdbscan(pca_reduced, n_trials):
    import optuna
    from sklearn.cluster import HDBSCAN
    n = pca_reduced.shape[0]
    best = {}

    def objective(trial):
        nonlocal best
        mcs = trial.suggest_int("min_cluster_size", 10, min(200, n // 5))
        ms = trial.suggest_int("min_samples", 3, min(30, mcs))
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(pca_reduced)
        score = _dbcv_score(labels, pca_reduced)
        if not best or score > best.get("_score", -2.0):
            result = _build_hdbscan_result(labels, pca_reduced, mcs, ms)
            result["_score"] = score
            best = result
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Soft guidance: if < 5 clusters, retry with smaller min_cluster_size
    if best.get("n_clusters", 0) < 5:
        def retry_objective(trial):
            nonlocal best
            mcs = trial.suggest_int("min_cluster_size", 5, 30)
            ms = trial.suggest_int("min_samples", 3, min(15, mcs))
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(pca_reduced)
            n_cl = len(set(labels.tolist()) - {-1})
            if n_cl < 3:
                return -1.0
            score = _dbcv_score(labels, pca_reduced)
            if not best or score > best.get("_score", -2.0):
                result = _build_hdbscan_result(labels, pca_reduced, mcs, ms)
                result["_score"] = score
                best = result
            return score

        study2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
        study2.optimize(retry_objective, n_trials=20, show_progress_bar=False)

    # Fallback
    if not best or best.get("n_clusters", 0) < 2:
        labels = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(pca_reduced)
        best = _build_hdbscan_result(labels, pca_reduced, 25, 5)

    best.pop("_score", None)
    return best
```

- [ ] **Step 4:** Verify import.

- [ ] **Step 5:** Commit.

### Task 4: Clean up old functions

**Files:**
- Modify: `dashboard/utils/latent_space.py`

- [ ] **Step 1:** Remove the old `compute_clustering` function (the one with UMAP pre-reduction, ~line 417-537). This was the old path that caused Nike artifacts.

- [ ] **Step 2:** Verify nothing else imports the removed functions:
```bash
docker exec junon-backend grep -r "auto_tune_params\|auto_tune_and_compute\|compute_clustering" /app/ --include="*.py" -l
```
Expected: only `latent_space.py` itself (now cleaned) and possibly old router references (fixed in next chunk).

- [ ] **Step 3:** Commit.

---

## Chunk 2: Backend API Endpoints

### Task 5: Add Pydantic schemas for recompute requests

**Files:**
- Modify: `api/schemas/latent_space.py`

- [ ] **Step 1:** Add new schemas at the end of the file:

```python
class RecomputePCARequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    space: str = "multi"
    variance_threshold: float = Field(default=0.95, ge=0.90, le=0.99)


class RecomputeVizRequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    space: str = "multi"
    n_neighbors: int = Field(default=50, ge=15, le=200)
    min_dist: float = Field(default=0.3, ge=0.05, le=1.0)


class RecomputeClusteringRequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    space: str = "multi"
    min_cluster_size: int | None = Field(default=None, ge=5)
    min_samples: int | None = Field(default=None, ge=1)


class AutoTuneRequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    space: str = "multi"
```

- [ ] **Step 2:** Commit.

### Task 6: Rewrite API router — recompute endpoints

**Files:**
- Modify: `api/routers/latent_space.py`

- [ ] **Step 1:** Remove old endpoints: `compute_latent_space` (POST /compute), `auto_tune`, `get_cached_compute`, the old `warmup_cache`. Remove clustering run endpoints (`get_clustering_runs`, `get_clustering_run_detail`).

- [ ] **Step 2:** Add helper to load embeddings from DB (shared by warmup + recompute-pca):

```python
async def _load_embeddings(domain: str, space: str, session: AsyncSession):
    """Load embeddings + metadata from DB. Returns (ids, embeddings_matrix, metadata_list)."""
    from dashboard.utils.latent_space import build_station_query, decode_eh_metadata, parse_pgvector
    import numpy as np

    sql, params = build_station_query(domain, EmbeddingFilters(), space=space)
    result = await session.execute(sql, params)
    rows = result.fetchall()

    ids, raw_embs, metadata_list = [], [], []
    for row in rows:
        raw = getattr(row, "embedding_raw", None)
        if raw is None:
            continue
        try:
            emb = parse_pgvector(raw)
        except Exception:
            continue
        ids.append(str(row.id))
        raw_embs.append(emb)
        if domain == "piezo":
            meta = decode_eh_metadata({
                "libelle_eh": getattr(row, "libelle_eh", None),
                "milieu_eh": getattr(row, "milieu_eh", None),
                "theme_eh": getattr(row, "theme_eh", None),
                "etat_eh": getattr(row, "etat_eh", None),
                "nature_eh": getattr(row, "nature_eh", None),
                "departement": getattr(row, "departement", None),
                "nom_departement": getattr(row, "nom_departement", None),
                "altitude": getattr(row, "altitude", None),
                "station_id": str(row.id),
            })
        else:
            meta = {
                "nom_cours_eau": getattr(row, "nom_cours_eau", None),
                "departement": getattr(row, "departement", None),
                "nom_departement": getattr(row, "nom_departement", None),
                "statut_station": getattr(row, "statut_station", None),
                "station_id": str(row.id),
            }
        metadata_list.append(meta)

    if not ids:
        return [], None, []
    return ids, np.stack(raw_embs, axis=0), metadata_list
```

- [ ] **Step 3:** Write the 3 recompute endpoints:

**POST /recompute-pca** — loads embeddings from DB, computes PCA, cascades VIZ + CLUSTERING, writes full cache.

**POST /recompute-viz** — loads `pca.reduced` from cache, computes UMAP 2D, updates viz section.

**POST /recompute-clustering** — loads `pca.reduced` from cache, runs HDBSCAN + KMeans elbow, updates clustering section.

Each endpoint:
1. Validates domain/space
2. Loads input (DB or cache)
3. Offloads compute to `asyncio.to_thread`
4. Reads current cache, updates relevant section(s), writes back atomically
5. Returns the updated cache (minus `pca.reduced` — too large for response, only stored in Redis)

- [ ] **Step 4:** Write `GET /cached/{domain}` endpoint:

Returns full cache minus `pca.reduced` (to keep response payload < 2 MB). The `pca.reduced` matrix stays in Redis only — the frontend doesn't need it.

- [ ] **Step 5:** Write `POST /auto-tune` endpoint:

Full pipeline: load embeddings → `compute_pca(defaults)` → `compute_viz(defaults)` → `compute_clustering_all()` → write cache. Force-replaces (no skip-if-exists).

- [ ] **Step 6:** Rewrite `warmup_cache` to use the same code path as `/auto-tune` but with skip-if-exists logic.

- [ ] **Step 7:** Verify backend starts without errors:
```bash
docker restart junon-backend && sleep 10 && docker logs junon-backend --since 15s
```

- [ ] **Step 8:** Commit.

---

## Chunk 3: Frontend API Client + Hooks

### Task 7: Update API client

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1:** Replace `compute()` and `autoTune()` with:

```typescript
latentSpace: {
  // ... keep stations, similar, stationWindows, profiling ...
  cached: (domain: string, space: string = 'multi') =>
    fetchJson<Record<string, unknown>>(`/latent-space/cached/${domain}?space=${space}`),
  recomputePca: (body: { domain: string; space: string; variance_threshold: number }) =>
    postJson<Record<string, unknown>>('/latent-space/recompute-pca', body, 300_000),
  recomputeViz: (body: { domain: string; space: string; n_neighbors: number; min_dist: number }) =>
    postJson<Record<string, unknown>>('/latent-space/recompute-viz', body, 30_000),
  recomputeClustering: (body: { domain: string; space: string; min_cluster_size?: number; min_samples?: number }) =>
    postJson<Record<string, unknown>>('/latent-space/recompute-clustering', body, 120_000),
  autoTune: (body: { domain: string; space: string }) =>
    postJson<Record<string, unknown>>('/latent-space/auto-tune', body, 300_000),
}
```

Remove `compute()` and `clusteringRuns()` / `clusteringRun()`.

- [ ] **Step 2:** Commit.

### Task 8: Update hooks

**Files:**
- Modify: `frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1:** Replace `useComputeUMAP` and `useAutoTune` with:

```typescript
export function useRecomputePca() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; variance_threshold: number }) =>
      api.latentSpace.recomputePca(body),
  })
}

export function useRecomputeViz() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; n_neighbors: number; min_dist: number }) =>
      api.latentSpace.recomputeViz(body),
  })
}

export function useRecomputeClustering() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; min_cluster_size?: number; min_samples?: number }) =>
      api.latentSpace.recomputeClustering(body),
  })
}

export function useAutoTune() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string }) =>
      api.latentSpace.autoTune(body),
  })
}
```

Remove `useComputeUMAP`, `useClusteringRuns`, `useClusteringRun`.

- [ ] **Step 2:** Commit.

---

## Chunk 4: Frontend — PipelineControls Component

### Task 9: Create `PipelineControls.tsx` (replaces `UMAPControls.tsx`)

**Files:**
- Create: `frontend/src/components/latent-space/PipelineControls.tsx`

- [ ] **Step 1:** Write the component with 3 sections:

```
Props:
  // PCA
  pcaDims: number | null
  pcaVariance: number | null
  varianceThreshold: number
  onVarianceThresholdChange: (v: number) => void
  onRecomputePca: () => void
  isRecomputingPca: boolean

  // Viz
  vizNNeighbors: number
  vizMinDist: number
  onVizNNeighborsChange: (v: number) => void
  onVizMinDistChange: (v: number) => void
  onRecomputeViz: () => void
  isRecomputingViz: boolean
  trustworthiness: number | null

  // Clustering
  hdbscanMcs: number
  hdbscanMs: number
  onHdbscanMcsChange: (v: number) => void
  onHdbscanMsChange: (v: number) => void
  onRecomputeClustering: () => void
  isRecomputingClustering: boolean
  clusteringDiagnostics: { dbcv?: number; noise_ratio?: number; n_clusters?: number } | null

  // Global
  onAutoTune: () => void
  isAutoTuning: boolean
```

Layout: 3 side-by-side sections in a horizontal bar, each with param inputs + recompute button + diagnostics. Auto-tune button at far right.

- [ ] **Step 2:** Commit.

### Task 10: Create `ElbowChart.tsx`

**Files:**
- Create: `frontend/src/components/latent-space/ElbowChart.tsx`

- [ ] **Step 1:** Write component:

```
Props:
  elbow: Array<{ k: number; silhouette: number; inertia: number }>
  selectedK: number | null
  onSelectK: (k: number) => void
```

Small Plotly chart with dual y-axis: inertia (left, line) + silhouette (right, line). Clickable points to select k. Highlight selected k with a marker.

- [ ] **Step 2:** Commit.

---

## Chunk 5: Frontend — Page Integration

### Task 11: Rewrite `LatentSpacePage.tsx` state and data flow

**Files:**
- Modify: `frontend/src/pages/LatentSpacePage.tsx`

This is the biggest task. Key changes:

- [ ] **Step 1:** Remove old state: `computedPoints`, `computeMutation`, `autoTuneMutation`, `subsampled`, `qualityMetrics`, `selectedRunId`, `clusteringRunData`, `clusteringRuns`. Remove all clustering run dropdown code.

- [ ] **Step 2:** Add new state:

```typescript
// Pipeline params (synced from cache on load)
const [varianceThreshold, setVarianceThreshold] = useState(0.95)
const [vizNNeighbors, setVizNNeighbors] = useState(50)
const [vizMinDist, setVizMinDist] = useState(0.3)
const [hdbscanMcs, setHdbscanMcs] = useState(25)
const [hdbscanMs, setHdbscanMs] = useState(5)

// Clustering display
const [cachedMethod, setCachedMethod] = useState<'hdbscan' | 'kmeans'>('hdbscan')
const [cachedKmeansK, setCachedKmeansK] = useState<number | null>(null)
```

- [ ] **Step 3:** Wire `useCachedCompute` as single source of truth for scatter points:

```typescript
const { data: cachedData, isLoading: isCacheLoading } = useCachedCompute(domain, space)

const scatterPoints = useMemo(() => {
  if (!cachedData) return []
  const cd = cachedData as CacheData  // typed interface
  if (!cd.station_ids || !cd.viz?.coords_2d) return []

  // Pick labels based on method
  let labels: number[]
  if (cachedMethod === 'kmeans' && cd.clustering?.kmeans_elbow?.length) {
    const match = cd.clustering.kmeans_elbow.find(e => e.k === cachedKmeansK)
      ?? cd.clustering.kmeans_elbow[0]
    labels = match.labels
  } else {
    labels = cd.clustering?.hdbscan?.labels ?? cd.station_ids.map(() => -1)
  }

  return cd.station_ids.map((id, i) => ({
    id,
    coords: cd.viz.coords_2d[i] as [number, number],
    cluster_label: labels[i] ?? -1,
    metadata: cd.metadata?.[i] ?? {},
    highlighted: !hasActiveFilters || matchesFilters(...)
  })).filter(p => isStationVisible(p.id))
}, [cachedData, cachedMethod, cachedKmeansK, ...])
```

- [ ] **Step 4:** Write 3 recompute handlers:

```typescript
const pcaMutation = useRecomputePca()
const vizMutation = useRecomputeViz()
const clusteringMutation = useRecomputeClustering()
const autoTuneMutation = useAutoTune()

async function handleRecomputePca() {
  await pcaMutation.mutateAsync({ domain, space, variance_threshold: varianceThreshold })
  queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
}

async function handleRecomputeViz() {
  await vizMutation.mutateAsync({ domain, space, n_neighbors: vizNNeighbors, min_dist: vizMinDist })
  queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
}

async function handleRecomputeClustering() {
  await clusteringMutation.mutateAsync({ domain, space, min_cluster_size: hdbscanMcs, min_samples: hdbscanMs })
  queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
}

async function handleAutoTune() {
  await autoTuneMutation.mutateAsync({ domain, space })
  queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
}
```

- [ ] **Step 5:** Sync params from cache on load (useEffect):

When `cachedData` changes, update `varianceThreshold`, `vizNNeighbors`, `vizMinDist`, `hdbscanMcs`, `hdbscanMs`, `cachedKmeansK` from the cached values so the UI controls reflect the current state.

- [ ] **Step 6:** Replace `<UMAPControls>` with `<PipelineControls>` wired to all the new state + handlers.

- [ ] **Step 7:** Add HDBSCAN/KMeans toggle + ElbowChart in the scatter area:

```tsx
{/* Method toggle + k selector */}
<div className="flex items-center gap-2">
  <ToggleButton value={cachedMethod} onChange={setCachedMethod} options={['hdbscan', 'kmeans']} />
  {cachedMethod === 'kmeans' && cachedElbow && (
    <ElbowChart elbow={cachedElbow} selectedK={cachedKmeansK} onSelectK={setCachedKmeansK} />
  )}
</div>
```

- [ ] **Step 8:** Remove all references to old clustering run dropdown, `computedPoints`, `computeMutation`, `clusteringRuns`, `clusteringRunData`, `selectedRunId`.

- [ ] **Step 9:** Verify frontend builds:
```bash
docker compose build --no-cache frontend
```

- [ ] **Step 10:** Commit.

---

## Chunk 6: Integration, Build & Deploy

### Task 12: Full integration test

- [ ] **Step 1:** Push all backend files to container:
```bash
for f in dashboard/utils/latent_space.py api/routers/latent_space.py api/schemas/latent_space.py api/main.py; do
  base64 $f | docker exec -i junon-backend python3 -c "import base64,sys; open('/app/$f','wb').write(base64.b64decode(sys.stdin.read()))"
done
```

- [ ] **Step 2:** Flush Redis cache and restart backend:
```bash
docker exec junon-backend python3 -c "import redis; r=redis.Redis(host='redis',port=6379); [r.delete(k) for k in r.scan_iter('junon:latent-space:*')]; print('flushed')"
docker restart junon-backend
```

- [ ] **Step 3:** Monitor warmup completes for all 4 spaces:
```bash
# Wait ~15 min then check
docker exec junon-backend python3 -c "import redis; r=redis.Redis(host='redis',port=6379); print(sum(1 for _ in r.scan_iter('junon:latent-space:*')), '/4 cached')"
```

- [ ] **Step 4:** Deploy frontend:
```bash
docker compose up -d frontend
```

- [ ] **Step 5:** Manual verification checklist:
- Load latent space page → scatter displays from cache (no recompute spinner)
- Toggle HDBSCAN / KMeans → colors change instantly
- KMeans: elbow chart displays, click different k → scatter recolors
- PCA section: change variance threshold, click Recompute PCA → full recompute, scatter updates
- Viz section: change n_neighbors/min_dist, click Recompute Viz → coords change, clusters stay
- Clustering section: change mcs/ms, click Recompute Clustering → clusters change, coords stay
- Auto-tune button → full pipeline, cache replaced
- Switch domain (piezo/hydro) or space (uni/multi) → loads different cache
- Active stations filter → hides stations, shows "clustering outdated" if applicable
- Click station → detail panel with similar stations + temporal evolution
- Click background → deselect

- [ ] **Step 6:** Commit with descriptive message.
