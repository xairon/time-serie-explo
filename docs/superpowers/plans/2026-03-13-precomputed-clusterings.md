# Pre-computed Clusterings Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store multiple pre-computed clustering configurations in PostgreSQL, each with metrics, and let the UI load/switch between them — eliminating the need to click Recompute for good defaults.

**Architecture:** Two new tables (`ml.clustering_runs` + `ml.clustering_labels`) store versioned clustering results. The Dagster pipeline computes 2 configs per domain nightly (Optuna-tuned + fixed defaults). The Junon API exposes endpoints to list and load runs. The frontend adds a dropdown to select which clustering to display.

**Tech Stack:** PostgreSQL (existing brgm-postgres), Dagster assets (hubeau_data_integration), FastAPI endpoints (time-serie-explo), React + TanStack Query (frontend)

---

## File Structure

### hubeau_data_integration (Dagster repo)

| File | Action | Responsibility |
|------|--------|----------------|
| `src/hubeau_pipeline/ml/latent_space/persistence.py` | Modify | Add `init_clustering_tables()`, `save_clustering_run()`, `save_clustering_labels()` |
| `src/hubeau_pipeline/ml/latent_space/clustering.py` | Modify | New `cluster_and_store()` that saves to new tables instead of overwriting `cluster_id` |
| `src/hubeau_pipeline/assets/ml_assets.py` | Modify | Update `_cluster_and_viz()` to call `cluster_and_store()` twice (tuned + fixed) |

### time-serie-explo (Junon app)

| File | Action | Responsibility |
|------|--------|----------------|
| `api/schemas/latent_space.py` | Modify | Add `ClusteringRun`, `ClusteringRunList` Pydantic models |
| `api/routers/latent_space.py` | Modify | Add `GET /clustering-runs/{domain}` and `GET /clustering-run/{run_id}` endpoints |
| `dashboard/utils/latent_space.py` | Modify | Add `list_clustering_runs()`, `load_clustering_run()` query functions |
| `frontend/src/lib/api.ts` | Modify | Add `clusteringRuns()` and `clusteringRun()` API methods |
| `frontend/src/hooks/useLatentSpace.ts` | Modify | Add `useClusteringRuns()` and `useClusteringRun()` hooks |
| `frontend/src/pages/LatentSpacePage.tsx` | Modify | Add clustering run selector dropdown, load selected run |
| `tests/latent_space/test_clustering_storage.py` | Create | Unit tests for the new query functions |

---

## Chunk 1: Database Tables & Dagster Storage

### Task 1: Create clustering tables in persistence.py

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/persistence.py`

**Context:** This file already has `init_ml_schema()` that creates the 4 embedding tables. We add 2 new tables alongside them.

- [ ] **Step 1: Add table creation SQL templates**

Add after the existing `_CREATE_BTREE_INDEX` template (line 57):

```python
_CREATE_CLUSTERING_RUNS = """
CREATE TABLE IF NOT EXISTS ml.clustering_runs (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,              -- 'piezo' or 'hydro'
    level TEXT NOT NULL DEFAULT 'stations',  -- 'stations' or 'windows'
    method TEXT NOT NULL DEFAULT 'hdbscan',
    params JSONB NOT NULL,             -- full hyperparameters used
    metrics JSONB NOT NULL,            -- silhouette, DBCV, CH, etc.
    n_clusters INT NOT NULL,
    n_stations INT NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_CLUSTERING_LABELS = """
CREATE TABLE IF NOT EXISTS ml.clustering_labels (
    run_id INT NOT NULL REFERENCES ml.clustering_runs(id) ON DELETE CASCADE,
    station_id TEXT NOT NULL,
    cluster_id INT NOT NULL,
    umap_2d_x FLOAT,
    umap_2d_y FLOAT,
    umap_3d_x FLOAT,
    umap_3d_y FLOAT,
    umap_3d_z FLOAT,
    PRIMARY KEY (run_id, station_id)
)
"""

_CREATE_CLUSTERING_LABELS_IDX = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_clustering_labels_run') THEN
        CREATE INDEX idx_clustering_labels_run ON ml.clustering_labels (run_id);
    END IF;
END $$;
"""
```

- [ ] **Step 2: Add table creation to init_ml_schema()**

In `init_ml_schema()`, after the `for domain` loop (line 77), add:

```python
        cur.execute(_CREATE_CLUSTERING_RUNS)
        cur.execute(_CREATE_CLUSTERING_LABELS)
        cur.execute(_CREATE_CLUSTERING_LABELS_IDX)
```

- [ ] **Step 3: Add save_clustering_run() function**

Add after `search_similar()` (end of file):

```python
def save_clustering_run(
    pg,
    domain: str,
    level: str,
    method: str,
    params: dict,
    metrics: dict,
    n_clusters: int,
    n_stations: int,
    is_default: bool = False,
    station_ids: list[str] | None = None,
    labels: np.ndarray | None = None,
    umap_2d: np.ndarray | None = None,
    umap_3d: np.ndarray | None = None,
) -> int:
    """Save a clustering run with labels and UMAP coords to ml.clustering_runs/labels.

    Returns the run_id.
    """
    import json

    with pg.get_connection() as conn:
        cur = conn.cursor()

        # If is_default, unset previous default for this domain+level
        if is_default:
            cur.execute(
                "UPDATE ml.clustering_runs SET is_default = FALSE "
                "WHERE domain = %s AND level = %s AND is_default = TRUE",
                (domain, level),
            )

        # Insert run metadata
        cur.execute(
            """
            INSERT INTO ml.clustering_runs
                (domain, level, method, params, metrics, n_clusters, n_stations, is_default)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (domain, level, method,
             json.dumps(params), json.dumps(metrics),
             n_clusters, n_stations, is_default),
        )
        run_id = cur.fetchone()[0]

        # Insert labels + coords
        if station_ids is not None and labels is not None:
            for i, (sid, label) in enumerate(zip(station_ids, labels)):
                u2x = float(umap_2d[i, 0]) if umap_2d is not None else None
                u2y = float(umap_2d[i, 1]) if umap_2d is not None else None
                u3x = float(umap_3d[i, 0]) if umap_3d is not None else None
                u3y = float(umap_3d[i, 1]) if umap_3d is not None else None
                u3z = float(umap_3d[i, 2]) if umap_3d is not None else None
                cur.execute(
                    """
                    INSERT INTO ml.clustering_labels
                        (run_id, station_id, cluster_id, umap_2d_x, umap_2d_y,
                         umap_3d_x, umap_3d_y, umap_3d_z)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, sid, int(label), u2x, u2y, u3x, u3y, u3z),
                )

        conn.commit()

    logger.info(
        "Saved clustering run %d: domain=%s level=%s method=%s "
        "n_clusters=%d n_stations=%d is_default=%s",
        run_id, domain, level, method, n_clusters, n_stations, is_default,
    )
    return run_id
```

- [ ] **Step 4: Test locally**

Run: `cd /home/ringuet/hubeau_data_integration && python -c "from hubeau_pipeline.ml.latent_space.persistence import save_clustering_run; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/persistence.py
git commit -m "feat(ml): add clustering_runs/labels tables and save_clustering_run()"
```

---

### Task 2: Modify clustering.py to support stored runs

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/clustering.py`

**Context:** `cluster_and_update()` currently overwrites the `cluster_id` column. We add a new function `cluster_and_store()` that saves to the new tables instead. Keep `cluster_and_update()` for backwards compat but also have it update the legacy column.

- [ ] **Step 1: Add cluster_and_store() function**

Add after `cluster_and_update()` (after line 147):

```python
def cluster_and_store(
    pg,
    domain: str,
    id_col: str,
    is_default: bool = False,
    tune: bool = False,
    tune_n_trials: int = 80,
    tune_timeout: int = 300,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    umap_dims: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
) -> dict:
    """Cluster station embeddings and store results in ml.clustering_runs/labels.

    Also computes UMAP 2D/3D visualization coords.
    Returns dict with run_id, metrics, params, embeddings, station_ids.
    """
    import umap as umap_lib
    from .persistence import save_clustering_run

    # Run clustering (same logic as cluster_and_update but don't write to legacy column)
    result = cluster_and_update(
        pg, domain, id_col,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_dims=umap_dims,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        tune=tune,
        tune_n_trials=tune_n_trials,
        tune_timeout=tune_timeout,
    )

    embeddings = result["embeddings"]
    station_ids = result["station_ids"]
    params = result["params"]

    if len(station_ids) == 0:
        return result

    # Compute UMAP 2D/3D for visualization
    logger.info(f"Computing UMAP 2D/3D for {len(station_ids)} {domain} stations...")
    umap_2d = umap_lib.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embeddings)

    umap_3d = umap_lib.UMAP(
        n_components=3, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embeddings)

    # Re-extract labels from DB (cluster_and_update already wrote them)
    table = f"ml.{domain}_station_embeddings"
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {id_col}, cluster_id FROM {table}")
        db_labels = {r[0]: r[1] for r in cur.fetchall()}
    labels = np.array([db_labels.get(sid, -1) for sid in station_ids])

    metrics = {
        "silhouette": result["silhouette_score"],
        "davies_bouldin": result["davies_bouldin_index"],
        "calinski_harabasz": result["calinski_harabasz"],
        "dbcv": result["dbcv"],
        "noise_ratio": result["noise_ratio"],
    }

    run_id = save_clustering_run(
        pg,
        domain=domain,
        level="stations",
        method="hdbscan",
        params=params,
        metrics=metrics,
        n_clusters=result["n_clusters"],
        n_stations=len(station_ids),
        is_default=is_default,
        station_ids=station_ids,
        labels=labels,
        umap_2d=umap_2d,
        umap_3d=umap_3d,
    )

    result["run_id"] = run_id
    result["umap_2d"] = umap_2d
    result["umap_3d"] = umap_3d
    return result
```

- [ ] **Step 2: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/clustering.py
git commit -m "feat(ml): add cluster_and_store() for versioned clustering results"
```

---

### Task 3: Update Dagster assets to compute 2 configs per domain

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/assets/ml_assets.py:252-310`

**Context:** `_cluster_and_viz()` currently calls `cluster_and_update()` once with Optuna tuning. We modify it to call `cluster_and_store()` twice: once with Optuna (marked as default), once with fixed params (as backup). Also update the legacy UMAP coords from the default run.

- [ ] **Step 1: Replace _cluster_and_viz() body**

Replace `_cluster_and_viz()` (lines 252-292) with:

```python
def _cluster_and_viz(context, pg, domain: str, id_col: str):
    """Compute 2 clustering configs per domain and store in versioned tables."""
    from ..ml.latent_space.clustering import cluster_and_store
    from ..ml.latent_space.persistence import update_umap_coords, init_ml_schema

    # Ensure new tables exist
    init_ml_schema(pg)

    # --- Config 1: Optuna-tuned HDBSCAN (default) ---
    context.log.info(f"Config 1: Optuna-tuned HDBSCAN for {domain}...")
    tuned = cluster_and_store(
        pg, domain, id_col,
        is_default=True,
        tune=True, tune_n_trials=80, tune_timeout=300,
    )
    context.log.info(
        f"Tuned: {tuned['n_clusters']} clusters, DBCV={tuned['dbcv']:.4f}, "
        f"sil={tuned['silhouette_score']:.4f}, run_id={tuned['run_id']}"
    )

    # Update legacy UMAP coords from default run
    if tuned["umap_2d"] is not None:
        update_umap_coords(
            pg, domain, id_col,
            tuned["station_ids"], tuned["umap_2d"], tuned["umap_3d"],
        )

    # --- Config 2: Fixed defaults (backup) ---
    context.log.info(f"Config 2: Fixed defaults for {domain}...")
    fixed = cluster_and_store(
        pg, domain, id_col,
        is_default=False,
        tune=False,
        min_cluster_size=10, min_samples=5,
        umap_dims=10, umap_n_neighbors=15, umap_min_dist=0.0,
    )
    context.log.info(
        f"Fixed: {fixed['n_clusters']} clusters, DBCV={fixed['dbcv']:.4f}, "
        f"sil={fixed['silhouette_score']:.4f}, run_id={fixed['run_id']}"
    )

    # Metadata from tuned (default) run
    params = tuned["params"]
    context.add_output_metadata({
        "n_stations": MetadataValue.int(len(tuned["station_ids"])),
        "tuned_run_id": MetadataValue.int(tuned["run_id"]),
        "tuned_n_clusters": MetadataValue.int(tuned["n_clusters"]),
        "tuned_dbcv": MetadataValue.float(tuned["dbcv"]),
        "tuned_silhouette": MetadataValue.float(tuned["silhouette_score"]),
        "fixed_run_id": MetadataValue.int(fixed["run_id"]),
        "fixed_n_clusters": MetadataValue.int(fixed["n_clusters"]),
        "fixed_dbcv": MetadataValue.float(fixed["dbcv"]),
        "fixed_silhouette": MetadataValue.float(fixed["silhouette_score"]),
        "hdbscan_min_cluster_size": MetadataValue.int(params["hdbscan_min_cluster_size"]),
        "hdbscan_min_samples": MetadataValue.int(params["hdbscan_min_samples"]),
    })
```

- [ ] **Step 2: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/assets/ml_assets.py
git commit -m "feat(ml): compute 2 clustering configs per domain (tuned + fixed)"
```

---

## Chunk 2: Junon API — Query Functions & Endpoints

### Task 4: Add query functions in dashboard/utils/latent_space.py

**Files:**
- Modify: `/home/ringuet/time-serie-explo/dashboard/utils/latent_space.py`
- Create: `/home/ringuet/time-serie-explo/tests/latent_space/test_clustering_storage.py`

**Context:** `dashboard/utils/` is pure Python (no framework imports). These functions use SQLAlchemy `text()` for parameterized queries, matching existing patterns in the file.

- [ ] **Step 1: Write the test**

Create `tests/latent_space/test_clustering_storage.py`:

```python
"""Tests for clustering run query functions."""

import pytest


class TestListClusteringRuns:
    """Test list_clustering_runs() output shape."""

    def test_returns_list(self):
        """Should return a list (even if empty when no DB)."""
        from dashboard.utils.latent_space import list_clustering_runs
        # Function signature check — actual DB test done in integration
        assert callable(list_clustering_runs)

    def test_function_signature(self):
        """Should accept session and domain params."""
        import inspect
        from dashboard.utils.latent_space import list_clustering_runs
        sig = inspect.signature(list_clustering_runs)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "domain" in params


class TestLoadClusteringRun:
    """Test load_clustering_run() output shape."""

    def test_returns_callable(self):
        from dashboard.utils.latent_space import load_clustering_run
        assert callable(load_clustering_run)

    def test_function_signature(self):
        import inspect
        from dashboard.utils.latent_space import load_clustering_run
        sig = inspect.signature(load_clustering_run)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "run_id" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_clustering_storage.py -v`
Expected: FAIL with ImportError (functions don't exist yet)

- [ ] **Step 3: Add list_clustering_runs() and load_clustering_run()**

Add at end of `/home/ringuet/time-serie-explo/dashboard/utils/latent_space.py`:

```python
async def list_clustering_runs(session, domain: str) -> list[dict]:
    """List available pre-computed clustering runs for a domain.

    Returns list of dicts with: id, method, params, metrics, n_clusters,
    n_stations, is_default, created_at.
    """
    result = await session.execute(
        text("""
            SELECT id, domain, level, method, params, metrics,
                   n_clusters, n_stations, is_default, created_at
            FROM ml.clustering_runs
            WHERE domain = :domain AND level = 'stations'
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"domain": domain},
    )
    rows = result.fetchall()
    return [
        {
            "id": r.id,
            "domain": r.domain,
            "level": r.level,
            "method": r.method,
            "params": r.params,
            "metrics": r.metrics,
            "n_clusters": r.n_clusters,
            "n_stations": r.n_stations,
            "is_default": r.is_default,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def load_clustering_run(session, run_id: int) -> dict:
    """Load a specific clustering run with all station labels and UMAP coords.

    Returns dict with: run metadata + labels list.
    """
    # Run metadata
    result = await session.execute(
        text("""
            SELECT id, domain, level, method, params, metrics,
                   n_clusters, n_stations, is_default, created_at
            FROM ml.clustering_runs
            WHERE id = :run_id
        """),
        {"run_id": run_id},
    )
    run_row = result.fetchone()
    if not run_row:
        return {}

    # Labels + coords
    result = await session.execute(
        text("""
            SELECT station_id, cluster_id,
                   umap_2d_x, umap_2d_y,
                   umap_3d_x, umap_3d_y, umap_3d_z
            FROM ml.clustering_labels
            WHERE run_id = :run_id
            ORDER BY station_id
        """),
        {"run_id": run_id},
    )
    label_rows = result.fetchall()

    return {
        "id": run_row.id,
        "domain": run_row.domain,
        "method": run_row.method,
        "params": run_row.params,
        "metrics": run_row.metrics,
        "n_clusters": run_row.n_clusters,
        "n_stations": run_row.n_stations,
        "is_default": run_row.is_default,
        "created_at": run_row.created_at.isoformat() if run_row.created_at else None,
        "labels": [
            {
                "station_id": r.station_id,
                "cluster_id": r.cluster_id,
                "umap_2d": [r.umap_2d_x, r.umap_2d_y] if r.umap_2d_x is not None else None,
                "umap_3d": [r.umap_3d_x, r.umap_3d_y, r.umap_3d_z] if r.umap_3d_x is not None else None,
            }
            for r in label_rows
        ],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ringuet/time-serie-explo && python -m pytest tests/latent_space/test_clustering_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add dashboard/utils/latent_space.py tests/latent_space/test_clustering_storage.py
git commit -m "feat(latent-space): add list/load clustering run query functions"
```

---

### Task 5: Add Pydantic schemas for clustering runs

**Files:**
- Modify: `/home/ringuet/time-serie-explo/api/schemas/latent_space.py`

- [ ] **Step 1: Add schemas**

Add at end of file:

```python
class ClusteringRunSummary(BaseModel):
    """Summary of a pre-computed clustering run (for list endpoint)."""
    id: int
    domain: str
    level: str = "stations"
    method: str = "hdbscan"
    params: dict
    metrics: dict
    n_clusters: int
    n_stations: int
    is_default: bool
    created_at: str | None = None


class ClusteringLabel(BaseModel):
    station_id: str
    cluster_id: int
    umap_2d: list[float] | None = None
    umap_3d: list[float] | None = None


class ClusteringRunDetail(BaseModel):
    """Full clustering run with labels and UMAP coords."""
    id: int
    domain: str
    method: str
    params: dict
    metrics: dict
    n_clusters: int
    n_stations: int
    is_default: bool
    created_at: str | None = None
    labels: list[ClusteringLabel]
```

- [ ] **Step 2: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add api/schemas/latent_space.py
git commit -m "feat(latent-space): add ClusteringRun Pydantic schemas"
```

---

### Task 6: Add API endpoints for clustering runs

**Files:**
- Modify: `/home/ringuet/time-serie-explo/api/routers/latent_space.py`

- [ ] **Step 1: Add imports**

Add to existing imports at top of file:

```python
from api.schemas.latent_space import ClusteringRunSummary, ClusteringRunDetail
```

- [ ] **Step 2: Add list endpoint**

Add before the profiling endpoint (end of file):

```python
@router.get("/clustering-runs/{domain}", response_model=list[ClusteringRunSummary])
async def list_clustering_runs(
    domain: str,
    session: AsyncSession = Depends(get_session),
):
    """List available pre-computed clustering runs for a domain."""
    if domain not in ("piezo", "hydro"):
        raise HTTPException(status_code=400, detail="Invalid domain")

    from dashboard.utils.latent_space import list_clustering_runs as _list_runs
    runs = await _list_runs(session, domain)
    return runs


@router.get("/clustering-run/{run_id}", response_model=ClusteringRunDetail)
async def get_clustering_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Load a specific clustering run with all labels and UMAP coords."""
    from dashboard.utils.latent_space import load_clustering_run as _load_run
    data = await _load_run(session, run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Clustering run not found")
    return data
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add api/routers/latent_space.py
git commit -m "feat(latent-space): add clustering runs list/detail API endpoints"
```

---

## Chunk 3: Frontend — Clustering Run Selector

### Task 7: Add API client methods and hooks

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/lib/api.ts`
- Modify: `/home/ringuet/time-serie-explo/frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1: Add API methods**

In `api.ts`, inside the `latentSpace` object (after `profiling`):

```typescript
    clusteringRuns: (domain: string) =>
      fetchJson<Array<Record<string, unknown>>>(`/latent-space/clustering-runs/${domain}`),
    clusteringRun: (runId: number) =>
      fetchJson<Record<string, unknown>>(`/latent-space/clustering-run/${runId}`),
```

- [ ] **Step 2: Add React Query hooks**

In `useLatentSpace.ts`, add:

```typescript
export function useClusteringRuns(domain: string) {
  return useQuery({
    queryKey: ['latent-space', 'clustering-runs', domain],
    queryFn: () => api.latentSpace.clusteringRuns(domain),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useClusteringRun(runId: number | null) {
  return useQuery({
    queryKey: ['latent-space', 'clustering-run', runId],
    queryFn: () => api.latentSpace.clusteringRun(runId!),
    staleTime: 5 * 60 * 1000,
    enabled: runId != null,
  })
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add frontend/src/lib/api.ts frontend/src/hooks/useLatentSpace.ts
git commit -m "feat(latent-space): add clustering runs API client and hooks"
```

---

### Task 8: Add clustering run selector to LatentSpacePage

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/pages/LatentSpacePage.tsx`

**Context:** When a clustering run is selected, we override the scatter points with the run's labels and UMAP coords. The dropdown shows available runs with their key metrics (n_clusters, silhouette, is_default badge).

- [ ] **Step 1: Add imports and state**

Add to imports:

```typescript
import { useClusteringRuns, useClusteringRun } from '@/hooks/useLatentSpace'
```

Add to state (after `activeTab` state):

```typescript
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
```

Add hooks (after existing hooks):

```typescript
  const { data: clusteringRuns } = useClusteringRuns(domain)
  const { data: clusteringRunData, isLoading: isRunLoading } = useClusteringRun(selectedRunId)
```

- [ ] **Step 2: Auto-select default run**

Add effect to auto-select the default run when runs load:

```typescript
  // Auto-select default clustering run when available
  useEffect(() => {
    if (clusteringRuns && clusteringRuns.length > 0 && selectedRunId == null) {
      const defaultRun = clusteringRuns.find((r: Record<string, unknown>) => r.is_default)
      if (defaultRun) {
        setSelectedRunId(defaultRun.id as number)
      }
    }
  }, [clusteringRuns, selectedRunId])

  // Reset run selection on domain change
  useEffect(() => {
    setSelectedRunId(null)
  }, [domain])
```

- [ ] **Step 3: Override scatter points when a run is loaded**

Modify the `scatterPoints` useMemo to prioritize clustering run data. Replace the existing `scatterPoints` useMemo with:

```typescript
  const scatterPoints = useMemo(() => {
    // Priority 1: On-demand computed points (from Recalculate button)
    if (computedPoints) {
      return computedPoints.map((p) => ({
        id: p.id,
        coords: (mode === '3d' ? p.coords.slice(0, 3) : p.coords.slice(0, 2)) as
          | [number, number]
          | [number, number, number],
        cluster_label: p.cluster_label,
        metadata: p.metadata,
        highlighted: true,
      }))
    }

    // Priority 2: Pre-computed clustering run
    if (clusteringRunData && (clusteringRunData as Record<string, unknown>).labels) {
      const runLabels = (clusteringRunData as Record<string, unknown>).labels as Array<{
        station_id: string
        cluster_id: number
        umap_2d: [number, number] | null
        umap_3d: [number, number, number] | null
      }>

      // Build lookup from station embeddings for metadata
      const metaMap = new Map<string, Record<string, unknown>>()
      for (const s of allStations) {
        metaMap.set(s.id, s.metadata)
      }

      return runLabels
        .filter((l) => (mode === '3d' ? l.umap_3d : l.umap_2d))
        .map((l) => ({
          id: l.station_id,
          coords: (mode === '3d' ? l.umap_3d! : l.umap_2d!) as
            | [number, number]
            | [number, number, number],
          cluster_label: l.cluster_id,
          metadata: metaMap.get(l.station_id) ?? {},
          highlighted: !hasActiveFilters || matchesFilters({
            id: l.station_id,
            cluster_id: l.cluster_id,
            metadata: metaMap.get(l.station_id) ?? {},
          } as StationRaw),
        }))
    }

    // Priority 3: Pre-computed station UMAP coords (legacy)
    return stations
      .filter((s) => (mode === '3d' ? s.umap_3d : s.umap_2d))
      .map((s) => ({
        id: s.id,
        coords: (mode === '3d' ? s.umap_3d! : s.umap_2d!) as
          | [number, number]
          | [number, number, number],
        cluster_label: s.cluster_id ?? -1,
        metadata: s.metadata,
        highlighted: !hasActiveFilters || matchesFilters(s),
      }))
  }, [stations, allStations, computedPoints, clusteringRunData, mode, hasActiveFilters, matchesFilters])
```

- [ ] **Step 4: Add run selector dropdown to the top bar**

Add after the `tabButtons` in the top bar (inside the `<div className="flex items-center gap-4 shrink-0">` block):

```typescript
        {clusteringRuns && clusteringRuns.length > 0 && activeTab === 'scatter' && (
          <select
            className="bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-accent-cyan/50 transition-colors"
            value={selectedRunId ?? ''}
            onChange={(e) => {
              const val = e.target.value
              setSelectedRunId(val ? Number(val) : null)
              // Clear computed points when switching to a pre-computed run
              setComputedPoints(null)
              setSubsampled(null)
              setQualityMetrics(null)
            }}
          >
            <option value="">No pre-computed clustering</option>
            {(clusteringRuns as Array<Record<string, unknown>>).map((run) => (
              <option key={run.id as number} value={run.id as number}>
                {run.is_default ? '★ ' : ''}
                {(run.method as string).toUpperCase()} — {run.n_clusters as number} clusters
                {' '}(sil: {((run.metrics as Record<string, number>).silhouette ?? 0).toFixed(2)})
                {run.is_default ? ' [default]' : ''}
              </option>
            ))}
          </select>
        )}

        {isRunLoading && (
          <div className="w-4 h-4 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
        )}
```

- [ ] **Step 5: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add frontend/src/pages/LatentSpacePage.tsx
git commit -m "feat(latent-space): add clustering run selector with auto-default"
```

---

## Chunk 4: Bootstrap — Create Tables & First Run

### Task 9: Create tables and trigger initial clustering

**Context:** The tables need to exist in brgm-postgres. We can create them by running `init_ml_schema()` from the Dagster repo, or by executing the SQL directly. Then trigger the first dual-config clustering.

- [ ] **Step 1: Create tables via SQL in brgm-postgres**

```bash
docker exec -i brgm-postgres psql -U postgres -d postgres -c "
CREATE TABLE IF NOT EXISTS ml.clustering_runs (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'stations',
    method TEXT NOT NULL DEFAULT 'hdbscan',
    params JSONB NOT NULL,
    metrics JSONB NOT NULL,
    n_clusters INT NOT NULL,
    n_stations INT NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml.clustering_labels (
    run_id INT NOT NULL REFERENCES ml.clustering_runs(id) ON DELETE CASCADE,
    station_id TEXT NOT NULL,
    cluster_id INT NOT NULL,
    umap_2d_x FLOAT,
    umap_2d_y FLOAT,
    umap_3d_x FLOAT,
    umap_3d_y FLOAT,
    umap_3d_z FLOAT,
    PRIMARY KEY (run_id, station_id)
);

CREATE INDEX IF NOT EXISTS idx_clustering_labels_run ON ml.clustering_labels (run_id);
"
```
Expected: `CREATE TABLE` / `CREATE INDEX` messages

- [ ] **Step 2: Verify tables exist**

```bash
docker exec -i brgm-postgres psql -U postgres -d postgres -c "
SELECT table_name FROM information_schema.tables WHERE table_schema = 'ml' ORDER BY table_name;
"
```
Expected: Should list `clustering_labels`, `clustering_runs`, plus the 4 existing embedding tables.

- [ ] **Step 3: Seed initial clustering runs from existing data**

Since the Dagster pipeline won't run until tonight, seed the tables with the current pre-computed clustering (already in `cluster_id` column) as the default, plus run a fixed-defaults clustering:

```bash
docker exec -i junon-backend python3 -c "
import asyncio
from dashboard.utils.latent_space import list_clustering_runs
print('Query function available')
"
```

- [ ] **Step 4: Rebuild and test**

```bash
cd /home/ringuet/time-serie-explo
docker compose up -d --build
```

Then test:
```bash
curl -s http://localhost:49513/api/v1/latent-space/clustering-runs/piezo | python3 -m json.tool
```
Expected: JSON array (may be empty until first Dagster run or manual seed)

- [ ] **Step 5: Commit all changes across both repos**

```bash
cd /home/ringuet/hubeau_data_integration
git add -A && git commit -m "feat(ml): versioned clustering with dual configs"

cd /home/ringuet/time-serie-explo
git add -A && git commit -m "feat(latent-space): pre-computed clustering selector"
```

---

## Chunk 5: Seed Script — Populate Tables Now

### Task 10: Write a one-shot seed script to populate clustering runs

**Files:**
- Create: `/home/ringuet/time-serie-explo/scripts/seed_clustering_runs.py`

**Context:** We can't wait for the Dagster nightly. This script runs inside the Docker container, reads existing embeddings from brgm-postgres, computes 2 configs per domain (tuned + fixed), and saves them to the new tables.

- [ ] **Step 1: Write the seed script**

```python
"""One-shot script to seed ml.clustering_runs with initial clustering configs.

Run inside junon-backend container:
    docker exec -i junon-backend python3 scripts/seed_clustering_runs.py
"""

import json
import logging
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_pg_conn():
    """Get psycopg2 connection to brgm-postgres."""
    import psycopg2
    return psycopg2.connect(
        host="brgm-postgres", port=5432,
        user="postgres", password="postgres", dbname="postgres",
    )


def load_embeddings(conn, domain: str):
    """Load station embeddings from ml.{domain}_station_embeddings."""
    id_col = "code_bss" if domain == "piezo" else "code_station"
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, embedding::text FROM ml.{domain}_station_embeddings")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    embs = np.array(
        [[float(x) for x in r[1].strip("[]").split(",")] for r in rows],
        dtype=np.float32,
    )
    return ids, embs


def run_clustering(embs, min_cluster_size, min_samples, umap_dims, umap_nn, umap_md):
    """Run UMAP + HDBSCAN, return labels + metrics."""
    import umap as umap_lib
    import hdbscan
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    reducer = umap_lib.UMAP(
        n_components=umap_dims, n_neighbors=umap_nn, min_dist=umap_md,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(embs)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels == -1).sum())

    sil = float(silhouette_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    db = float(davies_bouldin_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    ch = float(calinski_harabasz_score(reduced[mask], labels[mask])) if n_clusters >= 2 else -1.0
    dbcv = float(getattr(clusterer, "relative_validity_", 0.0))

    metrics = {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "dbcv": round(dbcv, 4),
        "noise_ratio": round(n_noise / len(labels), 4),
    }
    params = {
        "umap_n_components": umap_dims,
        "umap_n_neighbors": umap_nn,
        "umap_min_dist": umap_md,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": min_samples,
        "tuned": False,
    }
    return labels, n_clusters, n_noise, metrics, params


def compute_umap_viz(embs):
    """Compute UMAP 2D + 3D for visualization."""
    import umap as umap_lib
    umap_2d = umap_lib.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embs)
    umap_3d = umap_lib.UMAP(
        n_components=3, n_neighbors=30, min_dist=0.05,
        metric="cosine", random_state=42,
    ).fit_transform(embs)
    return umap_2d, umap_3d


def save_run(conn, domain, labels, ids, umap_2d, umap_3d, n_clusters, metrics, params, is_default):
    """Save clustering run to ml.clustering_runs + ml.clustering_labels."""
    cur = conn.cursor()

    if is_default:
        cur.execute(
            "UPDATE ml.clustering_runs SET is_default = FALSE "
            "WHERE domain = %s AND level = 'stations' AND is_default = TRUE",
            (domain,),
        )

    cur.execute(
        """
        INSERT INTO ml.clustering_runs
            (domain, level, method, params, metrics, n_clusters, n_stations, is_default)
        VALUES (%s, 'stations', 'hdbscan', %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (domain, json.dumps(params), json.dumps(metrics), n_clusters, len(ids), is_default),
    )
    run_id = cur.fetchone()[0]

    for i, (sid, label) in enumerate(zip(ids, labels)):
        cur.execute(
            """
            INSERT INTO ml.clustering_labels
                (run_id, station_id, cluster_id, umap_2d_x, umap_2d_y,
                 umap_3d_x, umap_3d_y, umap_3d_z)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (run_id, sid, int(label),
             float(umap_2d[i, 0]), float(umap_2d[i, 1]),
             float(umap_3d[i, 0]), float(umap_3d[i, 1]), float(umap_3d[i, 2])),
        )

    conn.commit()
    logger.info(f"Saved run {run_id}: {domain} {n_clusters} clusters (default={is_default})")
    return run_id


def seed_domain(conn, domain):
    """Seed 2 clustering configs for a domain."""
    logger.info(f"=== Seeding {domain} ===")
    ids, embs = load_embeddings(conn, domain)
    logger.info(f"Loaded {len(ids)} stations, embedding dim={embs.shape[1]}")

    # UMAP 2D/3D (shared across configs)
    umap_2d, umap_3d = compute_umap_viz(embs)

    # Config 1: Optuna-tuned
    logger.info("Config 1: Optuna-tuned HDBSCAN...")
    from hubeau_pipeline.ml.latent_space.tuning import tune_clustering
    tuned = tune_clustering(embs, n_trials=80, timeout=300)
    labels_t, n_clust_t, _, metrics_t, _ = run_clustering(
        embs,
        min_cluster_size=tuned.hdbscan_min_cluster_size,
        min_samples=tuned.hdbscan_min_samples,
        umap_dims=tuned.umap_n_components,
        umap_nn=tuned.umap_n_neighbors,
        umap_md=tuned.umap_min_dist,
    )
    params_t = {
        "umap_n_components": tuned.umap_n_components,
        "umap_n_neighbors": tuned.umap_n_neighbors,
        "umap_min_dist": tuned.umap_min_dist,
        "hdbscan_min_cluster_size": tuned.hdbscan_min_cluster_size,
        "hdbscan_min_samples": tuned.hdbscan_min_samples,
        "tuned": True,
    }
    save_run(conn, domain, labels_t, ids, umap_2d, umap_3d, n_clust_t, metrics_t, params_t, is_default=True)

    # Config 2: Fixed defaults
    logger.info("Config 2: Fixed defaults...")
    labels_f, n_clust_f, _, metrics_f, params_f = run_clustering(
        embs, min_cluster_size=10, min_samples=5, umap_dims=10, umap_nn=15, umap_md=0.0,
    )
    save_run(conn, domain, labels_f, ids, umap_2d, umap_3d, n_clust_f, metrics_f, params_f, is_default=False)


def main():
    # Add hubeau_data_integration to path for tuning import
    sys.path.insert(0, "/home/ringuet/hubeau_data_integration/src")

    conn = get_pg_conn()
    try:
        seed_domain(conn, "piezo")
        seed_domain(conn, "hydro")
    finally:
        conn.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the seed script**

```bash
docker exec -i junon-backend python3 scripts/seed_clustering_runs.py
```
Expected: Logs showing tuning + saving for both piezo and hydro (~5-10 min total)

Note: If the tuning import fails (hubeau_data_integration not available in container), fall back to running with fixed defaults only. The Optuna-tuned config can be added when the Dagster pipeline runs next.

- [ ] **Step 3: Verify seeded data**

```bash
docker exec -i brgm-postgres psql -U postgres -d postgres -c "
SELECT id, domain, method, n_clusters, n_stations, is_default,
       (metrics->>'silhouette')::float AS sil,
       (metrics->>'dbcv')::float AS dbcv
FROM ml.clustering_runs ORDER BY id;
"
```
Expected: 4 rows (2 per domain)

- [ ] **Step 4: Test API endpoints**

```bash
curl -s http://localhost:49513/api/v1/latent-space/clustering-runs/piezo | python3 -m json.tool | head -20
curl -s http://localhost:49513/api/v1/latent-space/clustering-runs/hydro | python3 -m json.tool | head -20
```

- [ ] **Step 5: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add scripts/seed_clustering_runs.py
git commit -m "feat(latent-space): add seed script for initial clustering runs"
```

---

## Final: Build, Test, Verify

- [ ] **Rebuild Docker**

```bash
cd /home/ringuet/time-serie-explo
docker compose up -d --build
```

- [ ] **Verify in browser**

1. Open Latent Space page
2. Select Piezometry → should see a dropdown with available clustering runs
3. Switch between runs → scatter should update with different clusters
4. Select Hydrometry → dropdown should show hydro runs
5. Click Recalculate → should override with on-demand clustering (dropdown deselects)

- [ ] **Final commit if needed**

```bash
git add -A && git commit -m "fix: final adjustments for pre-computed clusterings"
```
