# Cluster Profiling Panel — Design Spec

**Date**: 2026-03-13
**Status**: Approved
**Location**: Latent Space page, tab toggle "Scatter / Profiling"

## Overview

Add a comprehensive cluster profiling panel to the Latent Space page. When the user switches to the "Profiling" tab, they see 5 analysis blocks that characterize each cluster from multiple angles: metadata distributions, concordance with known labels, temporal prototypes, feature fingerprints, and SHAP explainability.

The goal is to make cluster interpretation actionable and intuitive — the user should immediately understand what each cluster represents and how confident they can be in that interpretation.

## Architecture

### Data Flow

```
brgm-postgres (embeddings + series)
    ↓
GET /api/v1/latent-space/profiling/{domain}
    ↓  asyncio.to_thread(_blocking_profiling)
dashboard/utils/cluster_profiling.py  (pure Python)
    ↓
ProfilingResponse (JSON)
    ↓
useClusterProfiling hook → ClusterProfiling component
```

### Shared State

The Profiling tab shares `domain`, `filters`, `hideUnclassified` with the Scatter tab. Switching tabs does not reset state.

## Backend

### New File: `dashboard/utils/cluster_profiling.py`

Pure Python module, no framework imports. 5 public functions:

#### 1. `compute_metadata_distributions(stations, meta_keys) → dict`

- Input: list of `{id, cluster_id, metadata}` dicts, list of metadata keys
- Output: `{key: {cluster_id: {value: count}}}`
- Pure counting, O(n)

#### 2. `compute_concordance(stations, meta_keys) → dict`

- Extracts `cluster_id` from each station dict (consistent with `compute_metadata_distributions`)
- For each metadata key, compute:
  - **ARI** (Adjusted Rand Index): `sklearn.metrics.adjusted_rand_score`
  - **NMI** (Normalized Mutual Information): `sklearn.metrics.normalized_mutual_info_score`
  - **Cramér's V**: `scipy.stats.chi2_contingency` → V = sqrt(chi2 / (n * (min(r,c) - 1)))
- Only on stations where both cluster_id >= 0 and metadata value is non-null
- If only 1 cluster present, return all metrics as 0.0 with a note
- Output: `{key: {ari: float, nmi: float, cramers_v: float}}`

#### 3a. `find_medoids(embeddings_map, cluster_labels) → dict[int, str]`

- `embeddings_map`: `{station_id: np.array(320,)}`
- For each cluster, find the station closest to the centroid (L2 in embedding space)
- Returns `{cluster_id: medoid_station_id}` — no series needed, instant

#### 3b. `build_prototypes(medoid_ids, cluster_members, series_map, dates_map) → dict`

- `medoid_ids`: output of `find_medoids`
- `cluster_members`: `{cluster_id: [station_ids]}` for envelope sampling
- `series_map` / `dates_map`: pre-fetched for medoids + up to 200 random stations/cluster
- Compute P10/P90 envelope, truncate to last 3 years (1095 days) for display
- Output: `{cluster_id: {medoid_id, dates, medoid_values, p10, p90}}`
- Uses primary variable only: `niveau_nappe_eau` (piezo) or `resultat_obs_elab` (hydro)

#### 4. `compute_feature_fingerprints(series_map, cluster_labels) → dict`

- For each station, compute 6 features on the primary variable:
  - `mean`: normalized mean level
  - `std`: normalized standard deviation
  - `trend`: slope of linear regression (np.polyfit degree 1), normalized
  - `seasonality`: amplitude of FFT at frequency nearest 1/365 (`np.argmin(np.abs(freqs - 1/365))`), positive half only; NaN if series < 365 days
  - `autocorr_365`: autocorrelation at lag 365 (manual formula: `corr(x[:-365], x[365:])`, O(n))
  - `wet_dry_ratio`: mean(DJF months) / mean(JJA months), clipped to [0, 5]; if JJA mean is 0, return NaN
- Aggregate per cluster: median of each feature
- Normalize per-feature to [0, 1] range across clusters for radar display (each axis independently scaled)
- Edge case: if all clusters have the same value for a feature, set normalized value to 0.5
- Output: `{cluster_id: {feature: normalized_value}}`
- Also return raw (non-normalized) values for tooltips

#### 5. `compute_cluster_shap(features_df, labels) → dict`

- Train `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)` on features → cluster labels
- Compute `shap.TreeExplainer` SHAP values
- **Multiclass handling**: `shap_values` is a list of N arrays (one per class), each `(n_samples, n_features)`
  - Global importance: `np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)`
  - Per-cluster signed: `shap_values[class_idx]` filtered to rows where true label == class_idx, then mean
  - Handle binary case (single array) vs multiclass (list of arrays)
- Output:
  - `feature_importance`: `{feature: mean_abs_shap}` (global)
  - `shap_per_cluster`: `{cluster_id: {feature: mean_shap_value}}` (signed, per cluster)
  - `proxy_accuracy`: float (RF cross-val accuracy)
- If only 1 cluster or accuracy < 0.3, return empty with warning flag

### New File: `api/schemas/cluster_profiling.py`

```python
class MetadataDistribution(BaseModel):
    key: str
    clusters: dict[str, dict[str, int]]  # cluster_id (str) → {value: count}

class ConcordanceMetric(BaseModel):
    key: str
    ari: float
    nmi: float
    cramers_v: float

class ClusterPrototype(BaseModel):
    cluster_id: int
    medoid_id: str
    dates: list[str]
    medoid_values: list[float]
    p10: list[float]
    p90: list[float]

class FeatureFingerprint(BaseModel):
    cluster_id: int
    features: dict[str, float]      # normalized [0,1]
    features_raw: dict[str, float]  # original values

class ShapExplanation(BaseModel):
    feature_importance: dict[str, float]
    shap_per_cluster: dict[str, dict[str, float]]  # cluster_id as string (JSON keys)
    proxy_accuracy: float
    warning: str | None = None

class ProfilingResponse(BaseModel):
    domain: str
    n_stations: int       # after hide_unclassified filtering
    n_clusters: int       # excluding noise (cluster_id = -1)
    distributions: list[MetadataDistribution]
    concordance: list[ConcordanceMetric]
    prototypes: list[ClusterPrototype]
    fingerprints: list[FeatureFingerprint]
    shap: ShapExplanation
    warnings: list[str] = []  # catch-all for edge case warnings (single cluster, low accuracy, etc.)
```

### Router: `api/routers/latent_space.py`

New endpoint:

```python
@router.get("/profiling/{domain}", response_model=ProfilingResponse)
async def get_profiling(
    domain: str,
    hide_unclassified: bool = False,
    db: AsyncSession = Depends(get_brgm_db),
):
```

- `hide_unclassified`: when True, exclude stations with all-null EH metadata (piezo only)
- Fetches embeddings + cluster_ids + metadata from `ml.{domain}_station_embeddings` + station tables
- Fetches raw series from chroniques tables (domain mapping below) for medoids and features
  - piezo → `gold.hubeau_daily_chroniques` (column: `niveau_nappe_eau`)
  - hydro → `gold.hydro_daily_chroniques` (column: `resultat_obs_elab`, **filter: `grandeur_hydro_elab = 'QmnJ'`**)
- Series fetch SQL: `SELECT code_bss, date_mesure, {column} FROM {table} WHERE code_bss IN (:ids) AND date_mesure >= :cutoff ORDER BY code_bss, date_mesure` — with `cutoff = now - 5 years` for features, `now - 3 years` for prototypes
- Two-phase series fetch:
  1. Compute medoids from embeddings only (`find_medoids`) — instant
  2. Fetch series for medoids + 200 sampled stations/cluster (prototypes) + all stations (features) in batches of 500 IDs
- **DB connection**: series batching uses synchronous `postgres_connector` (psycopg2) inside `asyncio.to_thread()`, NOT the `AsyncSession` — keeps `dashboard/utils/` framework-free
- Runs all 6 compute functions via `asyncio.to_thread()`

## Frontend

### LatentSpacePage Changes

- Add tab toggle in the top bar (alongside domain switch), **before** the `isLoading`/`isError` early returns so it remains accessible during loading
- New state: `activeTab: 'scatter' | 'profiling'`
- When `activeTab === 'profiling'`, render `<ClusterProfiling>` instead of `<EmbeddingScatter>`
- The FilterPanel, domain selector, and right sidebar remain visible in both tabs

### New Hook: `useClusterProfiling`

```typescript
// Added to hooks/useLatentSpace.ts (colocated with existing latent-space hooks)
export function useClusterProfiling(domain: string, hideUnclassified: boolean) {
  return useQuery({
    queryKey: ['latent-space', 'profiling', domain, hideUnclassified],
    queryFn: () => api.latentSpace.profiling(domain, hideUnclassified),
    staleTime: 5 * 60_000,
    gcTime: 30 * 60_000,
    enabled: !!domain,
  })
}
```

### New API Method

```typescript
// lib/api.ts — add to latentSpace namespace
profiling: (domain: string, hideUnclassified: boolean = false) =>
  fetchJson(`/latent-space/profiling/${domain}?hide_unclassified=${hideUnclassified}`, { timeout: 60_000 }),
```

### New Component: `ClusterProfiling.tsx`

Container component rendering 5 blocks vertically in a scrollable area.

#### Block 1: MetadataDistributions

- Plotly horizontal stacked bar chart
- Dropdown to select metadata key (milieu_eh, theme_eh, etat_eh, nature_eh, departement)
- One bar per cluster, segments colored by category value
- Sorted by cluster size descending
- Hover shows exact count and percentage

#### Block 2: ConcordanceTable

- HTML table, one row per metadata key
- Columns: Variable | ARI | NMI | Cramér's V
- Cell background color: green (>0.3), amber (0.1-0.3), red (<0.1)
- Tooltip explains each metric
- Compact — no more than 6 rows

#### Block 3: TemporalPrototypes

- Plotly subplots, grid layout (max 4 columns, wrap into rows — e.g., 2×4 for 8 clusters)
- Each subplot: medoid line (solid, accent color) + P10/P90 fill (same color, 15% opacity)
- X-axis: dates (last 3 years), Y-axis: primary variable
- Title: "Cluster {id} (n={count}) — medoid: {bss_code}"
- Dark theme, shared Y-axis range

#### Block 4: FeatureFingerprints

- Plotly Scatterpolar (radar chart)
- One trace per cluster, overlaid
- 6 axes: Mean, Std, Trend, Seasonality, Autocorr 365d, Wet/Dry ratio
- Normalized [0,1] values for comparability
- Hover shows raw value
- Legend with cluster colors matching the scatter

#### Block 5: ShapExplainability

- Horizontal bar chart per cluster (Plotly subplots)
- Features sorted by absolute SHAP value
- Red = positive (pushes toward this cluster), Blue = negative (pushes away)
- Header badge: "Proxy accuracy: {X}%" with color coding
- If warning present, show amber alert banner

### Loading State

- Each block has its own skeleton loader
- Single API call, but blocks render progressively as data arrives (all at once since single response, but skeleton gives good UX during the ~10s load)

## Edge Cases

- `hideUnclassified` (piezo only): when active, exclude stations with all-null EH metadata from profiling computations, not just from scatter display. This prevents noise stations from distorting concordance metrics.
- Clusters with < 3 stations: skip P10/P90 envelope (show medoid only), mark fingerprint as low-confidence
- Single cluster: concordance returns 0.0 for all metrics, SHAP returns empty with warning

## Performance Considerations

- Series fetch strategy: two-phase approach
  - Phase 1: medoid computation from embeddings only (no series needed), instant
  - Phase 2: fetch series for medoids + 200 random stations/cluster for envelopes + all stations for features (batched by 500 IDs)
- Feature computation: vectorized numpy on pre-fetched data, <1s
- SHAP: RF on 4K × 6 matrix, <1s
- Total expected latency: 8-12s first call, cached 5min after

## Dependencies

- Backend: numpy, scipy, scikit-learn, shap (already in requirements)
- Frontend: no new deps (Plotly already available)

## Files to Create/Modify

### Create
- `dashboard/utils/cluster_profiling.py`
- `api/schemas/cluster_profiling.py`
- `frontend/src/components/latent-space/ClusterProfiling.tsx`
- `frontend/src/components/latent-space/MetadataDistributions.tsx`
- `frontend/src/components/latent-space/ConcordanceTable.tsx`
- `frontend/src/components/latent-space/TemporalPrototypes.tsx`
- `frontend/src/components/latent-space/FeatureFingerprints.tsx`
- `frontend/src/components/latent-space/ShapExplainability.tsx`

### Modify
- `api/routers/latent_space.py` — add profiling endpoint
- `frontend/src/lib/api.ts` — add profiling method
- `frontend/src/hooks/useLatentSpace.ts` — add `useClusterProfiling` hook
- `frontend/src/pages/LatentSpacePage.tsx` — add tab toggle, render ClusterProfiling
