# Dual Embedding Spaces (Uni/Multi) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add univariate (target-only) embedding space alongside multivariate (target + weather), selectable in the UI, with full pipeline propagation (scatter, clustering, profiling, similarity).

**Architecture:** Add `space` column to PostgreSQL embedding tables. Dagster pipeline trains 4 encoders (2 domains × 2 spaces) in parallel, then encodes and clusters independently. API threads `space` param through all latent-space endpoints. Frontend adds a toggle next to domain selector.

**Tech Stack:** PostgreSQL (pgvector), Dagster, SoftCLT/TS2Vec (PyTorch), FastAPI, React + TanStack Query

**Repos:** `hubeau_data_integration` (Dagster pipeline) + `time-serie-explo` (Junon app)

---

## File Structure

### hubeau_data_integration

| File | Action | Responsibility |
|------|--------|----------------|
| `src/hubeau_pipeline/ml/latent_space/data.py` | Modify | Add `load_*_univariate()` functions returning shape `(T, 1)` |
| `src/hubeau_pipeline/ml/latent_space/persistence.py` | Modify | Add `space` param to upsert functions, update schema DDL |
| `src/hubeau_pipeline/assets/ml_assets.py` | Modify | 4 training + 4 encoding + 4 clustering assets (parameterized by space) |

### time-serie-explo

| File | Action | Responsibility |
|------|--------|----------------|
| `dashboard/utils/latent_space.py` | Modify | Add `space` param to `build_station_query()`, `list_clustering_runs()`, `load_clustering_run()` |
| `api/routers/latent_space.py` | Modify | Add `space` query param to all endpoints |
| `frontend/src/lib/api.ts` | Modify | Pass `space` param on all latent-space API calls |
| `frontend/src/hooks/useLatentSpace.ts` | Modify | Add `space` param to all hooks |
| `frontend/src/pages/LatentSpacePage.tsx` | Modify | Add space toggle, propagate to hooks |

---

## Chunk 1: Database Schema + Data Loading

### Task 1: Add univariate data loaders

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/data.py`

- [ ] **Step 1: Add load_piezo_series_univariate()**

Add after `load_piezo_series()`:

```python
def load_piezo_series_univariate(pg, min_days: int = 540) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Load piezo series with target variable only (niveau_nappe_eau).

    Returns:
        (series, dates) where series = {code_bss: (T, 1)} — single column.
    """
    query = """
        SELECT code_bss, date, niveau_nappe_eau
        FROM gold.hubeau_daily_chroniques
        WHERE code_bss IN (
            SELECT code_bss
            FROM gold.hubeau_daily_chroniques
            GROUP BY code_bss
            HAVING COUNT(*) >= %(min_days)s
        )
        ORDER BY code_bss, date
    """
    with pg.get_connection() as conn:
        df = pd.read_sql(query, conn, params={"min_days": min_days})

    series = {}
    dates = {}
    for code_bss, group in df.groupby("code_bss"):
        group = group.sort_values("date")
        arr = group[["niveau_nappe_eau"]].values.astype(np.float32)
        series[code_bss] = _interpolate_and_fill(arr)
        dates[code_bss] = group["date"].tolist()
    return series, dates
```

- [ ] **Step 2: Add load_hydro_series_univariate()**

Add after `load_hydro_series()`:

```python
def load_hydro_series_univariate(pg, min_days: int = 540) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Load hydro series with target variable only (resultat_obs_elab / QmnJ).

    Returns:
        (series, dates) where series = {code_station: (T, 1)} — single column.
    """
    query = """
        SELECT code_station, date, resultat_obs_elab
        FROM gold.hydro_daily_chroniques
        WHERE grandeur_hydro_elab = 'QmnJ'
          AND code_station IN (
            SELECT code_station
            FROM gold.hydro_daily_chroniques
            WHERE grandeur_hydro_elab = 'QmnJ'
            GROUP BY code_station
            HAVING COUNT(*) >= %(min_days)s
        )
        ORDER BY code_station, date
    """
    with pg.get_connection() as conn:
        df = pd.read_sql(query, conn, params={"min_days": min_days})

    series = {}
    dates = {}
    for code_station, group in df.groupby("code_station"):
        group = group.sort_values("date")
        arr = group[["resultat_obs_elab"]].values.astype(np.float32)
        series[code_station] = _interpolate_and_fill(arr)
        dates[code_station] = group["date"].tolist()
    return series, dates
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/data.py
git commit -m "feat(ml): add univariate data loaders for piezo and hydro"
```

---

### Task 2: Add `space` column to persistence.py DDL + upsert functions

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/persistence.py`

- [ ] **Step 1: Update _CREATE_STATION_TABLE DDL**

Change the PK to include `space`:

```python
_CREATE_STATION_TABLE = """
CREATE TABLE IF NOT EXISTS ml.{domain}_station_embeddings (
    {id_col} TEXT NOT NULL,
    space TEXT NOT NULL DEFAULT 'multi',
    embedding vector(320) NOT NULL,
    cluster_id INT,
    model_version TEXT NOT NULL,
    n_days INT NOT NULL,
    n_windows INT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    umap_2d_x FLOAT,
    umap_2d_y FLOAT,
    umap_3d_x FLOAT,
    umap_3d_y FLOAT,
    umap_3d_z FLOAT,
    PRIMARY KEY ({id_col}, space)
)
"""
```

- [ ] **Step 2: Update _CREATE_WINDOW_TABLE DDL**

```python
_CREATE_WINDOW_TABLE = """
CREATE TABLE IF NOT EXISTS ml.{domain}_window_embeddings (
    {id_col} TEXT NOT NULL,
    window_start DATE NOT NULL,
    space TEXT NOT NULL DEFAULT 'multi',
    window_end DATE NOT NULL,
    embedding vector(320) NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY ({id_col}, window_start, space)
)
"""
```

- [ ] **Step 3: Add `space` param to upsert_station_embeddings()**

Add `space: str = "multi"` parameter. Update the SQL:

```python
def upsert_station_embeddings(pg, domain: str, id_col: str,
                              embeddings: dict[str, np.ndarray],
                              n_days: dict[str, int],
                              n_windows: dict[str, int],
                              version: str,
                              space: str = "multi"):
    """Upsert station embeddings into ml.{domain}_station_embeddings."""
    table = f"ml.{domain}_station_embeddings"
    with pg.get_connection() as conn:
        cur = conn.cursor()
        for sid, emb in embeddings.items():
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            cur.execute(f"""
                INSERT INTO {table} ({id_col}, space, embedding, model_version, n_days, n_windows, updated_at)
                VALUES (%s, %s, %s::vector, %s, %s, %s, NOW())
                ON CONFLICT ({id_col}, space) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_version = EXCLUDED.model_version,
                    n_days = EXCLUDED.n_days,
                    n_windows = EXCLUDED.n_windows,
                    updated_at = NOW()
            """, (sid, space, emb_str, version, n_days.get(sid, 0), n_windows.get(sid, 0)))
        conn.commit()
    logger.info(f"Upserted {len(embeddings)} {space} station embeddings into {table}")
```

- [ ] **Step 4: Add `space` param to upsert_window_embeddings()**

Same pattern — add `space: str = "multi"` param, include in INSERT/ON CONFLICT:

```python
def upsert_window_embeddings(pg, domain: str, id_col: str,
                             window_data: dict[str, tuple[np.ndarray, list[tuple[str, str]]]],
                             version: str,
                             space: str = "multi"):
    """Upsert window embeddings into ml.{domain}_window_embeddings."""
    table = f"ml.{domain}_window_embeddings"
    total = 0
    with pg.get_connection() as conn:
        cur = conn.cursor()
        for sid, (embs, date_ranges) in window_data.items():
            for emb, (start, end) in zip(embs, date_ranges):
                emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
                cur.execute(f"""
                    INSERT INTO {table} ({id_col}, window_start, space, window_end, embedding, model_version)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT ({id_col}, window_start, space) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        window_end = EXCLUDED.window_end,
                        model_version = EXCLUDED.model_version
                """, (sid, start, space, end, emb_str, version))
                total += 1
        conn.commit()
    logger.info(f"Upserted {total} {space} window embeddings into {table}")
```

- [ ] **Step 5: Add `space` to clustering_runs table and save_clustering_run()**

In `_CREATE_CLUSTERING_RUNS`, add `space TEXT NOT NULL DEFAULT 'multi'` column.

In `save_clustering_run()`, add `space: str = "multi"` param and include it in the INSERT + the `is_default` UPDATE WHERE clause:

```python
        if is_default:
            cur.execute(
                "UPDATE ml.clustering_runs SET is_default = FALSE "
                "WHERE domain = %s AND level = %s AND space = %s AND is_default = TRUE",
                (domain, level, space),
            )

        cur.execute(
            """
            INSERT INTO ml.clustering_runs
                (domain, level, space, method, params, metrics, n_clusters, n_stations, is_default)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (domain, level, space, method,
             json.dumps(params), json.dumps(metrics),
             n_clusters, n_stations, is_default),
        )
```

- [ ] **Step 6: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/persistence.py
git commit -m "feat(ml): add space column to all embedding tables and upsert functions"
```

---

### Task 3: Migrate existing database — add space column to live tables

- [ ] **Step 1: Run migration SQL on brgm-postgres**

```bash
docker exec -i brgm-postgres psql -U postgres -d postgres -c "
-- Station embeddings: add space, rebuild PK
ALTER TABLE ml.piezo_station_embeddings ADD COLUMN IF NOT EXISTS space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.piezo_station_embeddings DROP CONSTRAINT IF EXISTS piezo_station_embeddings_pkey;
ALTER TABLE ml.piezo_station_embeddings ADD PRIMARY KEY (code_bss, space);

ALTER TABLE ml.hydro_station_embeddings ADD COLUMN IF NOT EXISTS space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.hydro_station_embeddings DROP CONSTRAINT IF EXISTS hydro_station_embeddings_pkey;
ALTER TABLE ml.hydro_station_embeddings ADD PRIMARY KEY (code_station, space);

-- Window embeddings: add space, rebuild PK
ALTER TABLE ml.piezo_window_embeddings ADD COLUMN IF NOT EXISTS space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.piezo_window_embeddings DROP CONSTRAINT IF EXISTS piezo_window_embeddings_pkey;
ALTER TABLE ml.piezo_window_embeddings ADD PRIMARY KEY (code_bss, window_start, space);

ALTER TABLE ml.hydro_window_embeddings ADD COLUMN IF NOT EXISTS space TEXT NOT NULL DEFAULT 'multi';
ALTER TABLE ml.hydro_window_embeddings DROP CONSTRAINT IF EXISTS hydro_window_embeddings_pkey;
ALTER TABLE ml.hydro_window_embeddings ADD PRIMARY KEY (code_station, window_start, space);

-- Clustering runs: add space
ALTER TABLE ml.clustering_runs ADD COLUMN IF NOT EXISTS space TEXT NOT NULL DEFAULT 'multi';
"
```

- [ ] **Step 2: Verify migration**

```bash
docker exec -i brgm-postgres psql -U postgres -d postgres -c "
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'ml' AND table_name = 'piezo_station_embeddings'
ORDER BY ordinal_position;
"
```
Expected: `space` column present.

- [ ] **Step 3: Commit migration script**

Save the SQL as `scripts/migrate_add_space.sql` in time-serie-explo and commit.

---

## Chunk 2: Dagster Assets — Training + Encoding + Clustering

### Task 4: Refactor Dagster assets for dual spaces

**Files:**
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/assets/ml_assets.py`
- Modify: `/home/ringuet/hubeau_data_integration/src/hubeau_pipeline/ml/latent_space/clustering.py`

- [ ] **Step 1: Add space-aware helper `_train_encoder()`**

Extract shared training logic into a helper parameterized by space:

```python
def _train_encoder(context, pg, domain: str, space: str):
    """Train a SoftCLT encoder for (domain, space)."""
    from ..ml.latent_space.encoder import SoftCLTEncoder
    from ..ml.latent_space.data import (
        load_piezo_series, load_hydro_series,
        load_piezo_series_univariate, load_hydro_series_univariate,
    )

    loaders = {
        ("piezo", "multi"): load_piezo_series,
        ("piezo", "uni"): load_piezo_series_univariate,
        ("hydro", "multi"): load_hydro_series,
        ("hydro", "uni"): load_hydro_series_univariate,
    }
    series_dict, _ = loaders[(domain, space)](pg, min_days=MIN_DAYS)
    context.log.info(f"{len(series_dict)} eligible {domain} stations for {space} encoder")

    input_dims = 1 if space == "uni" else 4
    all_data = np.concatenate(list(series_dict.values()))
    scaler = StandardScaler().fit(all_data)
    scaled = [scaler.transform(arr).astype(np.float32) for arr in series_dict.values()]

    t0 = time.time()
    encoder = SoftCLTEncoder(input_dims=input_dims, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, depth=DEPTH)
    encoder.fit(scaled, n_epochs=N_EPOCHS, lr=1e-3, batch_size=BATCH_SIZE,
                early_stop_patience=EARLY_STOP_PATIENCE, dagster_context=context)
    train_duration = time.time() - t0

    version = f"{domain}_{space}_{datetime.now():%Y%m%d_%H%M}"
    path = MODELS_DIR / version
    path.mkdir(parents=True, exist_ok=True)
    encoder.save(path / "model.pt")
    joblib.dump(scaler, path / "scaler.pkl")
    json.dump(list(series_dict.keys()), (path / "stations.json").open("w"))
    (MODELS_DIR / f"{domain}_{space}_latest").write_text(version)

    context.add_output_metadata({
        "model_version": version,
        "space": space,
        "input_dims": MetadataValue.int(input_dims),
        "n_stations": MetadataValue.int(len(series_dict)),
        "train_duration_sec": MetadataValue.float(train_duration),
    })
```

- [ ] **Step 2: Create 4 training assets**

Replace `ml_piezo_model_train` and `ml_hydro_model_train` with 4 assets:

```python
@asset(group_name="ml_piezo", deps=["hubeau_daily_chroniques"],
       description="Train SoftCLT encoder for piezo MULTIVARIATE (4 vars)")
def ml_piezo_multi_model_train(context: AssetExecutionContext, pg: PostgreSQLResource):
    _train_encoder(context, pg, "piezo", "multi")

@asset(group_name="ml_piezo", deps=["hubeau_daily_chroniques"],
       description="Train SoftCLT encoder for piezo UNIVARIATE (target only)")
def ml_piezo_uni_model_train(context: AssetExecutionContext, pg: PostgreSQLResource):
    _train_encoder(context, pg, "piezo", "uni")

@asset(group_name="ml_hydro", deps=["hydro_daily_chroniques"],
       description="Train SoftCLT encoder for hydro MULTIVARIATE (4 vars)")
def ml_hydro_multi_model_train(context: AssetExecutionContext, pg: PostgreSQLResource):
    _train_encoder(context, pg, "hydro", "multi")

@asset(group_name="ml_hydro", deps=["hydro_daily_chroniques"],
       description="Train SoftCLT encoder for hydro UNIVARIATE (target only)")
def ml_hydro_uni_model_train(context: AssetExecutionContext, pg: PostgreSQLResource):
    _train_encoder(context, pg, "hydro", "uni")
```

- [ ] **Step 3: Add space-aware helper `_encode_stations()`**

```python
def _encode_stations(context, pg, domain: str, id_col: str, space: str):
    """Nightly: encode stations with a trained (domain, space) model."""
    from ..ml.latent_space.encoder import SoftCLTEncoder
    from ..ml.latent_space.data import (
        load_piezo_series, load_hydro_series,
        load_piezo_series_univariate, load_hydro_series_univariate,
    )
    from ..ml.latent_space.persistence import init_ml_schema, upsert_station_embeddings, upsert_window_embeddings

    latest_file = MODELS_DIR / f"{domain}_{space}_latest"
    if not latest_file.exists():
        context.log.warning(f"No trained {domain}/{space} model. Run ml_{domain}_{space}_model_train first.")
        return

    version = latest_file.read_text().strip()
    path = MODELS_DIR / version
    encoder = SoftCLTEncoder.load(path / "model.pt")
    scaler = joblib.load(path / "scaler.pkl")

    init_ml_schema(pg)

    loaders = {
        ("piezo", "multi"): load_piezo_series,
        ("piezo", "uni"): load_piezo_series_univariate,
        ("hydro", "multi"): load_hydro_series,
        ("hydro", "uni"): load_hydro_series_univariate,
    }
    series_dict, dates_dict = loaders[(domain, space)](pg, min_days=MIN_DAYS)
    context.log.info(f"Encoding {len(series_dict)} {domain}/{space} stations...")

    station_embs = {}
    window_data = {}
    n_days_map = {}
    n_windows_map = {}

    total = len(series_dict)
    for i, (sid, arr) in enumerate(series_dict.items(), 1):
        scaled = scaler.transform(arr).astype(np.float32)
        dates = dates_dict.get(sid, [])
        n_days_map[sid] = len(arr)
        if len(scaled) < WINDOW_SIZE:
            continue
        win_embs, win_dates = encoder.encode_windows(scaled, WINDOW_SIZE, STRIDE, dates)
        window_data[sid] = (win_embs, win_dates)
        station_embs[sid] = SoftCLTEncoder.station_embedding(win_embs)
        n_windows_map[sid] = win_embs.shape[0]
        if i % 500 == 0 or i == total:
            context.log.info(f"{domain}/{space}: {i}/{total} stations")

    upsert_station_embeddings(pg, domain, id_col, station_embs, n_days_map, n_windows_map, version, space=space)
    upsert_window_embeddings(pg, domain, id_col, window_data, version, space=space)

    context.add_output_metadata({
        "n_stations": MetadataValue.int(len(station_embs)),
        "n_windows": MetadataValue.int(sum(w[0].shape[0] for w in window_data.values())),
        "model_version": version,
        "space": space,
    })
```

- [ ] **Step 4: Create 4 encoding assets**

Replace `ml_piezo_embeddings_update` and `ml_hydro_embeddings_update`:

```python
@asset(group_name="ml_piezo", deps=["hubeau_daily_chroniques"],
       description="Nightly: encode piezo MULTI embeddings")
def ml_piezo_multi_embeddings_update(context: AssetExecutionContext, pg: PostgreSQLResource):
    _encode_stations(context, pg, "piezo", "code_bss", "multi")

@asset(group_name="ml_piezo", deps=["hubeau_daily_chroniques"],
       description="Nightly: encode piezo UNI embeddings")
def ml_piezo_uni_embeddings_update(context: AssetExecutionContext, pg: PostgreSQLResource):
    _encode_stations(context, pg, "piezo", "code_bss", "uni")

@asset(group_name="ml_hydro", deps=["hydro_daily_chroniques"],
       description="Nightly: encode hydro MULTI embeddings")
def ml_hydro_multi_embeddings_update(context: AssetExecutionContext, pg: PostgreSQLResource):
    _encode_stations(context, pg, "hydro", "code_station", "multi")

@asset(group_name="ml_hydro", deps=["hydro_daily_chroniques"],
       description="Nightly: encode hydro UNI embeddings")
def ml_hydro_uni_embeddings_update(context: AssetExecutionContext, pg: PostgreSQLResource):
    _encode_stations(context, pg, "hydro", "code_station", "uni")
```

- [ ] **Step 5: Update `cluster_and_store()` to accept `space` param**

In `clustering.py`, add `space: str = "multi"` parameter to `cluster_and_store()`. Pass it to `save_clustering_run()`.

Also, `cluster_and_update()` needs to filter by space when reading/writing embeddings:
```python
cur.execute(f"SELECT {id_col}, embedding::text FROM {table} WHERE space = %s", (space,))
# ...
cur.execute(f"UPDATE {table} SET cluster_id = %s WHERE {id_col} = %s AND space = %s",
            (int(label), sid, space))
```

- [ ] **Step 6: Update `_cluster_and_viz()` to accept space and update clustering assets**

```python
def _cluster_and_viz(context, pg, domain: str, id_col: str, space: str = "multi"):
    """Compute 2 clustering configs for a (domain, space) and store."""
    # ... same logic but pass space to cluster_and_store()
```

Create 4 clustering assets (one per domain×space):

```python
@asset(group_name="ml_piezo", deps=["ml_piezo_multi_embeddings_update"])
def ml_piezo_multi_clusters(context: AssetExecutionContext, pg: PostgreSQLResource):
    _cluster_and_viz(context, pg, "piezo", "code_bss", "multi")

@asset(group_name="ml_piezo", deps=["ml_piezo_uni_embeddings_update"])
def ml_piezo_uni_clusters(context: AssetExecutionContext, pg: PostgreSQLResource):
    _cluster_and_viz(context, pg, "piezo", "code_bss", "uni")

# Same for hydro...
```

- [ ] **Step 7: Handle backward compatibility for `{domain}_latest` model files**

The old model pointer files (`piezo_latest`, `hydro_latest`) should map to the new `{domain}_multi_latest` convention. In `_encode_stations()`, fall back:

```python
    latest_file = MODELS_DIR / f"{domain}_{space}_latest"
    if not latest_file.exists() and space == "multi":
        # Backward compat: old pointer file
        latest_file = MODELS_DIR / f"{domain}_latest"
    if not latest_file.exists():
        context.log.warning(...)
        return
```

- [ ] **Step 8: Commit**

```bash
cd /home/ringuet/hubeau_data_integration
git add src/hubeau_pipeline/ml/latent_space/clustering.py src/hubeau_pipeline/assets/ml_assets.py
git commit -m "feat(ml): dual space (uni/multi) training, encoding, and clustering assets"
```

---

## Chunk 3: Junon API — Thread `space` Through All Endpoints

### Task 5: Update backend queries with `space` filter

**Files:**
- Modify: `/home/ringuet/time-serie-explo/dashboard/utils/latent_space.py`

- [ ] **Step 1: Add `space` param to `build_station_query()`**

Add `space: str = "multi"` parameter. Add `e.space = :space` WHERE clause and param:

```python
def build_station_query(domain: str, filters, space: str = "multi") -> tuple[Any, dict]:
    # ...
    params["space"] = space
    where_clauses.append("e.space = :space")
    # ... rest unchanged
```

- [ ] **Step 2: Add `space` param to `list_clustering_runs()`**

```python
async def list_clustering_runs(session, domain: str, space: str = "multi") -> list[dict]:
    result = await session.execute(
        text("""
            SELECT ... FROM ml.clustering_runs
            WHERE domain = :domain AND level = 'stations' AND space = :space
            ORDER BY created_at DESC LIMIT 20
        """),
        {"domain": domain, "space": space},
    )
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add dashboard/utils/latent_space.py
git commit -m "feat(latent-space): add space filter to all backend queries"
```

---

### Task 6: Add `space` query param to API endpoints

**Files:**
- Modify: `/home/ringuet/time-serie-explo/api/routers/latent_space.py`

- [ ] **Step 1: Update GET /stations/{domain}**

```python
@router.get("/stations/{domain}", response_model=StationsResponse)
async def get_stations(
    domain: str,
    space: str = Query("multi"),
    filters: ... ,
    session: AsyncSession = Depends(get_brgm_db),
):
    # Pass space to build_station_query
    sql, params = build_station_query(domain, filters, space=space)
```

- [ ] **Step 2: Update POST /compute**

Add `space: str = "multi"` to `ComputeRequest` schema. Filter embeddings by space in the query:

```python
# In the compute endpoint, when building the embedding query:
WHERE_SPACE = "AND e.space = :space"
# Add to all SELECT queries that read from station/window embeddings
```

- [ ] **Step 3: Update GET /similar/{domain}/{station_id}**

Add `space` query param, filter by `WHERE space = :space` in the similarity query.

- [ ] **Step 4: Update GET /clustering-runs/{domain}**

```python
@router.get("/clustering-runs/{domain}")
async def list_clustering_runs(
    domain: str,
    space: str = Query("multi"),
    session: AsyncSession = Depends(get_brgm_db),
):
    from dashboard.utils.latent_space import list_clustering_runs as _list_runs
    return await _list_runs(session, domain, space=space)
```

- [ ] **Step 5: Update GET /profiling/{domain}**

Add `space` query param, pass to profiling queries that read from embeddings.

- [ ] **Step 6: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add api/routers/latent_space.py api/schemas/latent_space.py
git commit -m "feat(latent-space): add space query param to all API endpoints"
```

---

## Chunk 4: Frontend — Space Toggle

### Task 7: Thread `space` through API client and hooks

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/lib/api.ts`
- Modify: `/home/ringuet/time-serie-explo/frontend/src/hooks/useLatentSpace.ts`

- [ ] **Step 1: Update API client**

Add `space` param to all latent-space methods:

```typescript
  latentSpace: {
    stations: (domain: string, space: string = 'multi') =>
      fetchJson<{ stations: Array<Record<string, unknown>> }>(`/latent-space/stations/${domain}?space=${space}`),
    compute: (body: Record<string, unknown>) =>
      postJson<Record<string, unknown>>('/latent-space/compute', body, 120_000),
    similar: (domain: string, stationId: string, k: number = 10, space: string = 'multi') =>
      fetchJson<Record<string, unknown>>(`/latent-space/similar/${domain}/${stationId}?k=${k}&space=${space}`),
    profiling: (domain: string, hideUnclassified: boolean = false, space: string = 'multi') =>
      fetchJson<Record<string, unknown>>(
        `/latent-space/profiling/${domain}?hide_unclassified=${hideUnclassified}&space=${space}`,
        { timeout: 60_000 },
      ),
    clusteringRuns: (domain: string, space: string = 'multi') =>
      fetchJson<Array<Record<string, unknown>>>(`/latent-space/clustering-runs/${domain}?space=${space}`),
    clusteringRun: (runId: number) =>
      fetchJson<Record<string, unknown>>(`/latent-space/clustering-run/${runId}`),
  },
```

- [ ] **Step 2: Update hooks**

Add `space` to query keys and function calls:

```typescript
export function useStationEmbeddings(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'stations', domain, space],
    queryFn: () => api.latentSpace.stations(domain, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useClusteringRuns(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'clustering-runs', domain, space],
    queryFn: () => api.latentSpace.clusteringRuns(domain, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useClusterProfiling(domain: string, space: string = 'multi', hideUnclassified: boolean) {
  return useQuery({
    queryKey: ['latent-space', 'profiling', domain, space, hideUnclassified],
    queryFn: () => api.latentSpace.profiling(domain, hideUnclassified, space),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    enabled: !!domain,
  })
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add -f frontend/src/lib/api.ts frontend/src/hooks/useLatentSpace.ts
git commit -m "feat(latent-space): thread space param through API client and hooks"
```

---

### Task 8: Add space toggle to LatentSpacePage

**Files:**
- Modify: `/home/ringuet/time-serie-explo/frontend/src/pages/LatentSpacePage.tsx`

- [ ] **Step 1: Add space state**

```typescript
const [space, setSpace] = useState<'uni' | 'multi'>('multi')
```

- [ ] **Step 2: Update all hooks to pass space**

```typescript
const { data: stationsData, isLoading, isError, refetch } = useStationEmbeddings(domain, space)
const { data: clusteringRuns } = useClusteringRuns(domain, space)
```

- [ ] **Step 3: Reset state on space change**

In `handleDomainChange`, also handle space changes:

```typescript
function handleSpaceChange(s: 'uni' | 'multi') {
    setSpace(s)
    setComputedPoints(null)
    setSubsampled(null)
    setQualityMetrics(null)
    setSelectedStation(null)
    setSelectedRunId(null)
}
```

- [ ] **Step 4: Add space toggle buttons to top bar**

After `domainButtons`, add:

```typescript
const spaceButtons = (
    <div className="flex rounded-lg overflow-hidden border border-white/10">
      <button
        onClick={() => handleSpaceChange('uni')}
        className={`px-4 py-2 text-sm transition-colors ${
          space === 'uni'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >Univariate</button>
      <button
        onClick={() => handleSpaceChange('multi')}
        className={`px-4 py-2 text-sm transition-colors ${
          space === 'multi'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >Multivariate</button>
    </div>
)
```

Render `{spaceButtons}` in the top bar alongside `{domainButtons}` and `{tabButtons}`.

- [ ] **Step 5: Pass space to ClusterProfiling and handleRecalculate**

In `handleRecalculate()` body, add `space` field:
```typescript
const body = { domain, space, embeddings_type: level, ... }
```

In `<ClusterProfiling>`, pass space:
```tsx
<ClusterProfiling domain={domain} space={space} hideUnclassified={hideUnclassified} />
```

Update `ClusterProfiling` component to accept and forward `space` prop.

- [ ] **Step 6: Commit**

```bash
cd /home/ringuet/time-serie-explo
git add frontend/src/pages/LatentSpacePage.tsx frontend/src/components/latent-space/ClusterProfiling.tsx
git commit -m "feat(latent-space): add uni/multi space toggle to UI"
```

---

## Chunk 5: Seed + Deploy + Verify

### Task 9: Train univariate encoders and seed data

- [ ] **Step 1: Train univariate encoders**

Trigger from Dagster UI or manually:
```bash
# Inside hubeau_data_integration container, or via Dagster:
dagster asset materialize --select ml_piezo_uni_model_train ml_hydro_uni_model_train
```

Alternatively, write a quick script if Dagster isn't set up for this:
```bash
cd /home/ringuet/hubeau_data_integration
python -c "
from hubeau_pipeline.resources import PostgreSQLResource
from hubeau_pipeline.assets.ml_assets import _train_encoder
# ... manual trigger
"
```

- [ ] **Step 2: Encode univariate embeddings**

```bash
dagster asset materialize --select ml_piezo_uni_embeddings_update ml_hydro_uni_embeddings_update
```

- [ ] **Step 3: Compute clustering runs for uni space**

```bash
dagster asset materialize --select ml_piezo_uni_clusters ml_hydro_uni_clusters
```

Or use the seed script pattern (similar to `seed_clustering_runs.py`) adapted for uni space.

- [ ] **Step 4: Rebuild and deploy Junon**

```bash
cd /home/ringuet/time-serie-explo
docker compose up -d --build
```

- [ ] **Step 5: Verify in browser**

1. Open Latent Space page
2. Select Piezometry → Univariate → should show scatter with uni embeddings
3. Switch to Multivariate → scatter should change (different clustering)
4. Check Profiling tab works for both spaces
5. Verify similar stations use the correct space
6. Switch to Hydrometry → same toggle should work

- [ ] **Step 6: Final commit**

```bash
git add -A && git commit -m "feat(latent-space): dual embedding spaces fully operational"
```
