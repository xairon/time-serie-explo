# Dual Embedding Spaces (Univariate / Multivariate) — Design Spec

## Goal

Add a toggle to switch between two embedding spaces — **univariate** (target variable only) and **multivariate** (target + weather covariates) — so users can compare how stations cluster with vs without meteorological bias. The space choice propagates through the entire pipeline: embeddings, clustering, profiling, similarity search.

## Context

The current SoftCLT encoder uses 4 input variables per domain:
- **Piezo**: `niveau_nappe_eau` + `temperature_2m`, `total_precipitation`, `potential_evaporation`
- **Hydro**: `resultat_obs_elab` + same 3 covariates

Including weather covariates may cause stations to cluster by climate rather than intrinsic hydrological behavior. A univariate encoder (input_dims=1) trained on the target alone reveals the signal's own structure.

## Architecture

### Parameterization

The embedding space (`uni` | `multi`) becomes a first-class parameter alongside domain (`piezo` | `hydro`). Every component is parameterized by the tuple `(domain, space)`:

| Component | Parameterized | Impact |
|-----------|--------------|--------|
| SoftCLT encoder model | Separate model per (domain, space) | 4 models total |
| Station embeddings | Separate rows per space | 2 rows per station |
| Window embeddings | Separate rows per space | 2 rows per station×window |
| Clustering runs | `space` column on `ml.clustering_runs` | 8 runs (2 configs × 2 spaces × 2 domains) |
| UMAP coords | Per clustering run (already stored in `ml.clustering_labels`) | No change |
| Profiling | Computed from clustering labels of selected space | No change |
| Similar stations | Cosine on embedding of selected space | Filter by space |

This gives 4 combinations: `piezo×uni`, `piezo×multi`, `hydro×uni`, `hydro×multi`.

### Database Changes

**`ml.{domain}_station_embeddings`** — add `space` column, change PK:

```sql
ALTER TABLE ml.piezo_station_embeddings ADD COLUMN space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.piezo_station_embeddings DROP CONSTRAINT piezo_station_embeddings_pkey;
ALTER TABLE ml.piezo_station_embeddings ADD PRIMARY KEY (code_bss, space);

-- Same for hydro (code_station, space)
```

Existing rows get `space='multi'` via the DEFAULT. New univariate rows inserted alongside.

Each station has 2 rows:
- `(code_bss, 'multi')` → 320-dim embedding from 4-variable encoder
- `(code_bss, 'uni')` → 320-dim embedding from 1-variable encoder

Columns `cluster_id`, `umap_2d_x/y`, `umap_3d_x/y/z` are per-space (each row has its own values).

**`ml.{domain}_window_embeddings`** — add `space` column, change PK:

```sql
ALTER TABLE ml.piezo_window_embeddings ADD COLUMN space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.piezo_window_embeddings DROP CONSTRAINT piezo_window_embeddings_pkey;
ALTER TABLE ml.piezo_window_embeddings ADD PRIMARY KEY (code_bss, window_start, space);

-- Same for hydro
```

**`ml.clustering_runs`** — add `space` column:

```sql
ALTER TABLE ml.clustering_runs ADD COLUMN space TEXT NOT NULL DEFAULT 'multi';
```

**HNSW index** — one per (domain, space) for similarity search:

```sql
CREATE INDEX idx_piezo_station_emb_hnsw_uni
    ON ml.piezo_station_embeddings USING hnsw (embedding vector_cosine_ops)
    WHERE space = 'uni';
-- Keep existing index for multi (or add WHERE space = 'multi')
```

### Dagster Pipeline Changes

**New assets** (4 training + 4 encoding + 4 clustering = 12 assets, replacing current 6):

Training (run if model absent or stale):
- `ml_piezo_uni_model_train` — SoftCLTEncoder(input_dims=1), trained on `niveau_nappe_eau` only
- `ml_piezo_multi_model_train` — SoftCLTEncoder(input_dims=4), trained on all 4 cols (current behavior)
- `ml_hydro_uni_model_train` — same pattern
- `ml_hydro_multi_model_train` — same pattern

All 4 can run in parallel (A6000 48GB, each training uses ~2-3GB).

Encoding (nightly, after training):
- `ml_piezo_uni_embeddings_update` — encode with uni model, upsert with `space='uni'`
- `ml_piezo_multi_embeddings_update` — encode with multi model, upsert with `space='multi'`
- Same for hydro

Clustering (nightly, after encoding):
- `ml_piezo_uni_clusters` — 2 configs (wide + fine) for uni space
- `ml_piezo_multi_clusters` — 2 configs for multi space
- Same for hydro

**Data loading changes** in `data.py`:
- New function `load_piezo_series_univariate()` → returns only `niveau_nappe_eau` column, shape `(T, 1)`
- New function `load_hydro_series_univariate()` → returns only `resultat_obs_elab`, shape `(T, 1)`
- Existing functions renamed to `*_multivariate()` for clarity

**Encoder changes** in `encoder.py`:
- No architecture change — same class, just `input_dims=1` vs `input_dims=4`
- Separate scaler per (domain, space)

**Persistence changes** in `persistence.py`:
- `upsert_station_embeddings()` — add `space` parameter
- `upsert_window_embeddings()` — add `space` parameter
- SQL queries filter by `space` in WHERE clauses

### API Changes (time-serie-explo)

**All latent-space endpoints** get a `space` query parameter (default: `'multi'`):

```python
@router.get("/stations/{domain}")
async def get_stations(domain: str, space: str = Query("multi"), ...):
    # SQL: WHERE space = :space
```

Affected endpoints:
- `GET /stations/{domain}?space=uni`
- `POST /compute` — body gets `space` field
- `GET /similar/{domain}/{id}?space=uni`
- `GET /clustering-runs/{domain}?space=uni`
- `GET /clustering-run/{id}` — no change (run already has space info)
- `GET /profiling/{domain}?space=uni`

**Query changes** in `dashboard/utils/latent_space.py`:
- `build_station_query()` — add `AND e.space = :space` filter
- `list_clustering_runs()` — add `AND space = :space` filter
- Profiling queries — filter by space

### Frontend Changes

**Top bar** — add space toggle next to domain selector:

```
[Piezometry | Hydrometry] [Univariate | Multivariate] [Scatter | Profiling]
```

**State** in `LatentSpacePage.tsx`:
```typescript
const [space, setSpace] = useState<'uni' | 'multi'>('multi')
```

When `space` changes:
- Refetch stations (`useStationEmbeddings(domain, space)`)
- Reset clustering run selection
- Reset computed points
- Refetch clustering runs (`useClusteringRuns(domain, space)`)

**Hooks** — all add `space` parameter:
- `useStationEmbeddings(domain, space)`
- `useClusteringRuns(domain, space)`
- `useClusterProfiling(domain, space, hideUnclassified)`

**API client** — pass `space` as query param on all latent-space calls.

### What Does NOT Change

- `EmbeddingScatter` component — receives points, doesn't care about space
- `ClusterProfiling` component — receives profiling data, agnostic
- `FilterPanel` — filters work on metadata, independent of embedding space
- `StationDetail` — station info is the same regardless of space
- `QualityMetrics` — displays whatever metrics come from compute
- `UMAPControls` — clustering/UMAP params are independent of space
- Gold tables (`hubeau_daily_chroniques`, `hydro_daily_chroniques`) — untouched

### Training Details

Univariate encoder:
- Architecture: identical to multivariate except `input_dims=1`
- Hyperparams: `hidden_dim=64, depth=10, embedding_dim=320, epochs=200, batch_size=128, early_stop=20`
- Scaler: `StandardScaler` fitted on target column only
- Input shape: `(B, T, 1)` instead of `(B, T, 4)`
- Model saved to: `/var/ml/models/{domain}_uni_YYYYMMDD_HHMM/model.pt`

Multivariate encoder: no change from current behavior.

### Migration

1. ALTER tables to add `space` column with DEFAULT 'multi' (existing data preserved)
2. Train univariate encoders (one-shot, ~30min each on GPU)
3. Encode all stations with uni encoders
4. Compute clustering runs for uni space
5. Deploy updated API + frontend

After migration, the nightly pipeline handles everything automatically.
