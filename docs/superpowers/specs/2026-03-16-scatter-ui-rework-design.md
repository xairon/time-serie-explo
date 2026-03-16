# Latent Space Scatter UI Rework — Design Spec

## Goal

Rework the Latent Space scatter page for usability with 70+ clusters: fix the legend, hover, click behavior, filter panel, station detail panel, and add per-station window visualization.

## Problems Addressed

1. **Legend**: Plotly native legend unusable with 70+ clusters — takes entire screen
2. **Hover**: Shows generic metadata fields, not domain-relevant info
3. **Click**: Filters persist confusingly; should highlight site visually first
4. **FilterPanel**: Too wide, missing "active stations" filter, EH fields rarely used
5. **StationDetail**: Minimal content, no temporal exploration
6. **Windows**: No way to see a station's temporal evolution in embedding space

## Design

### 1. Scatter Legend

Remove Plotly's native legend (`showlegend: false` on all traces). Replace with a custom HTML cluster bar rendered below the scatter:

- Horizontal scrollable row of cluster badges: colored dot + cluster ID + station count
- Click a badge → highlight that cluster (others dim to gray, opacity 0.15)
- Click again or click "All" → remove highlight
- If >20 clusters, show first 20 + "N more..." expandable
- Noise cluster (-1) shown as gray badge at the end

**Component**: `ClusterLegendBar` — receives `clusters: {id, color, count}[]`, `selected: number | null`, `onSelect(id | null)`.

### 2. Hover Info

Replace the generic `buildHoverMeta()` with domain-specific hover templates.

**Piezo hover**:
```
<b>{code_bss}</b>
{libelle_eh}
{departement} · Cluster {cluster_id}
{n_windows} windows · {n_days} days
```

**Hydro hover**:
```
<b>{code_station}</b>
{nom_cours_eau}
{departement} · Cluster {cluster_id}
{n_windows} windows · {n_days} days
```

The metadata dict already has these fields. The `buildHoverMeta` helper becomes domain-aware.

### 3. Click Behavior

**Single click on a point**:
1. Select station → open StationDetail panel
2. **Highlight all stations from the same site**: same `libelle_eh` (piezo) or `nom_cours_eau` (hydro). Other stations dim (opacity 0.15), site stations stay bright.
3. This is a **visual highlight**, not a filter change — the `filters` state is NOT modified.
4. Click on empty space → clear selection and highlight.

**"Filter by site" button** in StationDetail:
- Applies persistent filter (`filters.libelle_eh = X` or `filters.nom_cours_eau = X`)
- Only then does the FilterPanel reflect the active filter

This separates exploration (click = peek) from filtering (explicit action).

### 4. FilterPanel Rework

**Width**: Reduce from ~280px to ~200px.

**Layout** (top to bottom):
1. **Station search** (by code) — compact input
2. **Active stations toggle**: switch between "All stations" (default, all training stations ~4.2K) and "Active only" (~2.5K with recent data). Uses `n_days` threshold or a `last_date` field.
3. **Color by** selector: cluster, département, aquifère/cours_eau, altitude
4. **Cluster filter**: dropdown to select a specific cluster ID
5. **Advanced filters** (collapsible, closed by default): milieu_eh, theme_eh, etat_eh, nature_eh — only for piezo, hidden for hydro

**Removed**: The current layout with all EH filters visible by default.

### 5. StationDetail Panel

**Width**: w-72 (288px), fixed.

**Content** (top to bottom):
1. **Header**: station code + cluster badge (colored dot + number)
2. **Metadata table**:
   - Piezo: libelle_eh, département, altitude, n_windows, n_days
   - Hydro: nom_cours_eau, département, statut_station, n_windows, n_days
3. **Window scatter** (new): small Plotly scatter (full width × 200px) showing this station's window embeddings in UMAP 2D space, colored by year. Reveals temporal evolution — are recent years different from older ones?
   - Data source: `GET /latent-space/station-windows/{domain}/{station_id}?space=X` (new endpoint)
   - Returns window embeddings, UMAP 2D projected on-the-fly (small, <100 windows per station)
4. **Similar stations** (existing): top 5 by cosine similarity, clickable
5. **"Filter by site" button**: applies persistent filter for this station's site

### 6. Active Stations Filter

**Backend**: Add `last_date` to the station query — `MAX(date)` from the chroniques table, or approximated from `updated_at` in the embeddings table.

Actually, simpler: the embedding table already has `n_days` and `n_windows`. Stations with `n_windows >= 10` (~10 years of data including recent) are likely active. But the real signal is the last observation date.

**Approach**: Add a `last_date DATE` column to `ml.{domain}_station_embeddings`, populated during nightly encoding from the max date in the series. Then the frontend can filter by `last_date > 2024-01-01`.

For MVP: use `n_windows` as a proxy. Stations with more windows have longer records but not necessarily recent data. If the user needs true "active" filtering, we add `last_date` later.

**Frontend**: Toggle in FilterPanel. Client-side filtering on the `n_windows` field or `last_date` if available.

### 7. Window Scatter (Station Detail)

When a station is selected, fetch its window embeddings and display a mini-scatter.

**New endpoint**: `GET /latent-space/station-windows/{domain}/{station_id}?space=X`

**Response**:
```json
{
  "station_id": "BSS001",
  "windows": [
    {"window_start": "2015-01-01", "window_end": "2015-12-31", "umap_x": 1.2, "umap_y": 0.5},
    {"window_start": "2015-04-01", "window_end": "2016-03-31", "umap_x": 1.3, "umap_y": 0.7},
    ...
  ]
}
```

**Backend**: Load window embeddings for one station from `ml.{domain}_window_embeddings WHERE {id_col} = X AND space = Y`, then compute UMAP 2D on-the-fly (fast, <100 points). Color by year (extract from `window_start`).

**Mini-scatter**: Plotly scattergl, no axes labels, minimal margins, colored by year with Viridis colorscale. Hover shows `window_start — window_end`.

## Files Affected

### Frontend
| File | Changes |
|------|---------|
| `LatentSpacePage.tsx` | Click behavior, highlight logic, remove filter-on-click |
| `EmbeddingScatter.tsx` | Remove native legend, domain-specific hover, site highlight |
| `FilterPanel.tsx` | Compact layout, active toggle, advanced collapsible |
| `StationDetail.tsx` | Enriched content, window scatter, filter-by-site button |
| New: `ClusterLegendBar.tsx` | Custom HTML cluster legend |
| New: `WindowScatter.tsx` | Mini scatter for station windows |

### Backend
| File | Changes |
|------|---------|
| `api/routers/latent_space.py` | New endpoint: station-windows |
| `dashboard/utils/latent_space.py` | Query for window embeddings + on-the-fly UMAP |

### Not Changed
- `UMAPControls.tsx` — already fixed
- `QualityMetrics.tsx` — now inline in controls
- `ClusterProfiling.tsx` — separate spec
- Database schema — no changes
- Dagster pipeline — no changes
