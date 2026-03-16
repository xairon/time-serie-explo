"""Latent space utilities: query building, UMAP, clustering, similarity search.

Pure Python module — NO framework imports.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from sqlalchemy import text


# ---------------------------------------------------------------------------
# SQL query builders
# ---------------------------------------------------------------------------

_VALID_DOMAINS = {"piezo", "hydro"}

# BDLISA code → label mappings (Sandre referential)
_MILIEU_EH: dict[str, str] = {
    "1": "Poreux",
    "2": "Fissuré",
    "3": "Karstique",
    "4": "Double porosité fissuré et poreux",
    "5": "Double porosité karstique et poreux",
    "6": "Double porosité karstique et fissuré",
    "8": "Milieu composite",
    "9": "Milieu non applicable",
    "X": "Indéterminé",
}

_THEME_EH: dict[str, str] = {
    "0": "Indifférencié",
    "1": "Sédimentaire",
    "2": "Sédimentaire",
    "3": "Socle",
    "4": "Volcanique",
    "5": "Alluvial",
}

_ETAT_EH: dict[str, str] = {
    "1": "Libre seul",
    "2": "Libre et captif",
    "3": "Captif seul",
    "4": "Libre ou captif",
    "5": "Libre et captif affleurant",
    "6": "Captif sous couverture non aquifère",
    "X": "Indéterminé",
}

_NATURE_EH: dict[str, str] = {
    "0": "Entité hydrogéologique",
    "3": "Système aquifère",
    "4": "Domaine hydrogéologique",
    "5": "Système aquifère",
    "6": "Unité aquifère",
    "7": "Unité semi-perméable",
}


def _decode_eh(code: str | None, mapping: dict[str, str]) -> str | None:
    """Decode a BDLISA code to its label."""
    if code is None:
        return None
    return mapping.get(str(code), str(code))


def decode_eh_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Decode BDLISA codes to human-readable labels in a metadata dict."""
    out = dict(meta)
    out["milieu_eh"] = _decode_eh(meta.get("milieu_eh"), _MILIEU_EH)
    out["theme_eh"] = _decode_eh(meta.get("theme_eh"), _THEME_EH)
    out["etat_eh"] = _decode_eh(meta.get("etat_eh"), _ETAT_EH)
    out["nature_eh"] = _decode_eh(meta.get("nature_eh"), _NATURE_EH)
    return out


_SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from Pydantic model or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_station_query(domain: str, filters, space: str = "multi") -> tuple[Any, dict]:
    """Build SQL query for station-level embeddings.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    filters:
        EmbeddingFilters-like object (attributes: station_ids, libelle_eh,
        milieu_eh, theme_eh, etat_eh, nature_eh, departement, region,
        cluster_id).

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    params: dict[str, Any] = {}
    where_clauses: list[str] = []

    params["space"] = space
    where_clauses.append("e.space = :space")

    if domain == "piezo":
        select_cols = """
            e.code_bss            AS id,
            tme.libelle_eh,
            tme.milieu_eh,
            tme.theme_eh,
            tme.etat_eh,
            tme.nature_eh,
            s.code_departement    AS departement,
            s.nom_departement,
            s.altitude_station    AS altitude,
            e.cluster_id,
            e.n_windows,
            e.umap_2d_x,
            e.umap_2d_y,
            e.umap_3d_x,
            e.umap_3d_y,
            e.umap_3d_z,
            e.embedding::text     AS embedding_raw
        """
        from_clause = """
            ml.piezo_station_embeddings e
            JOIN gold.dim_piezo_stations s ON e.code_bss = s.code_bss
            LEFT JOIN (
                SELECT DISTINCT ON (m.code_bss)
                    m.code_bss,
                    m.libelle_eh,
                    m.milieu_eh,
                    m.theme_eh,
                    m.etat_eh,
                    m.nature_eh
                FROM gold.int_station_era5_mapping m
                ORDER BY m.code_bss
            ) tme ON tme.code_bss = e.code_bss
        """
        id_col = "e.code_bss"

        if _get(filters, 'station_ids'):
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(_get(filters, 'station_ids'))
        if _get(filters, 'libelle_eh'):
            where_clauses.append("tme.libelle_eh ILIKE :libelle_eh")
            params["libelle_eh"] = f"%{_get(filters, 'libelle_eh')}%"
        if _get(filters, 'milieu_eh'):
            where_clauses.append("tme.milieu_eh = :milieu_eh")
            params["milieu_eh"] = _get(filters, 'milieu_eh')
        if _get(filters, 'theme_eh'):
            where_clauses.append("tme.theme_eh = :theme_eh")
            params["theme_eh"] = _get(filters, 'theme_eh')
        if _get(filters, 'etat_eh'):
            where_clauses.append("tme.etat_eh = :etat_eh")
            params["etat_eh"] = _get(filters, 'etat_eh')
        if _get(filters, 'nature_eh'):
            where_clauses.append("tme.nature_eh = :nature_eh")
            params["nature_eh"] = _get(filters, 'nature_eh')
        if _get(filters, 'departement'):
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = _get(filters, 'departement')
        if _get(filters, 'cluster_id') is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = _get(filters, 'cluster_id')

    else:  # hydro
        select_cols = """
            e.code_station        AS id,
            s.nom_cours_eau,
            s.code_departement    AS departement,
            s.nom_departement,
            s.statut_station,
            e.cluster_id,
            e.n_windows,
            e.umap_2d_x,
            e.umap_2d_y,
            e.umap_3d_x,
            e.umap_3d_y,
            e.umap_3d_z,
            e.embedding::text     AS embedding_raw
        """
        from_clause = """
            ml.hydro_station_embeddings e
            JOIN gold.dim_hydro_stations s ON e.code_station = s.code_station
        """
        id_col = "e.code_station"

        if _get(filters, 'station_ids'):
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(_get(filters, 'station_ids'))
        if _get(filters, 'departement'):
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = _get(filters, 'departement')
        if _get(filters, 'cluster_id') is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = _get(filters, 'cluster_id')

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = text(f"SELECT {select_cols} FROM {from_clause} {where_sql}")
    return sql, params


def build_window_query(
    domain: str,
    filters,
    year_min: int | None,
    year_max: int | None,
    season: str | None,
    space: str = "multi",
) -> tuple[Any, dict]:
    """Build SQL query for window-level embeddings.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    filters:
        EmbeddingFilters-like object.
    year_min, year_max:
        Optional year range on window_start.
    season:
        Optional season code ("DJF", "MAM", "JJA", "SON").

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    params: dict[str, Any] = {}
    where_clauses: list[str] = []

    params["space"] = space
    where_clauses.append("w.space = :space")

    id_col = "code_bss" if domain == "piezo" else "code_station"

    select_cols = f"""
        w.{id_col}            AS id,
        w.window_start,
        w.window_end,
        w.cluster_id,
        w.embedding::text     AS embedding_raw
    """
    from_clause = f"ml.{domain}_window_embeddings w"

    if _get(filters, 'station_ids'):
        where_clauses.append(f"w.{id_col} = ANY(:station_ids)")
        params["station_ids"] = list(_get(filters, 'station_ids'))
    if _get(filters, 'cluster_id') is not None:
        where_clauses.append("w.cluster_id = :cluster_id")
        params["cluster_id"] = _get(filters, 'cluster_id')
    if year_min is not None:
        where_clauses.append("EXTRACT(YEAR FROM w.window_start) >= :year_min")
        params["year_min"] = year_min
    if year_max is not None:
        where_clauses.append("EXTRACT(YEAR FROM w.window_start) <= :year_max")
        params["year_max"] = year_max
    if season is not None:
        months = _SEASON_MONTHS.get(season)
        if months is None:
            raise ValueError(f"Invalid season '{season}'. Must be one of {list(_SEASON_MONTHS)}.")
        where_clauses.append("EXTRACT(MONTH FROM w.window_start) = ANY(:season_months)")
        params["season_months"] = months

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = text(f"SELECT {select_cols} FROM {from_clause} {where_sql}")
    return sql, params


def build_similar_query(domain: str, station_id: str, k: int, space: str = "multi") -> tuple[Any, dict]:
    """Build SQL query for nearest-neighbour similarity search using pgvector.

    Uses cosine distance operator `<=>`.

    Parameters
    ----------
    domain:
        "piezo" or "hydro".
    station_id:
        Query station identifier.
    k:
        Number of neighbours to return (excluding the query station itself).

    Returns
    -------
    (sqlalchemy.text, params_dict)
    """
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    id_col = "code_bss" if domain == "piezo" else "code_station"
    table = f"ml.{domain}_station_embeddings"

    sql = text(f"""
        SELECT
            e.{id_col}                                         AS id,
            e.embedding <=> q.embedding                        AS distance,
            e.cluster_id
        FROM {table} e
        CROSS JOIN (
            SELECT embedding
            FROM {table}
            WHERE {id_col} = :station_id AND space = :space
            LIMIT 1
        ) q
        WHERE e.{id_col} != :station_id AND e.space = :space
        ORDER BY distance ASC
        LIMIT :k
    """)
    params: dict[str, Any] = {"station_id": station_id, "k": k, "space": space}
    return sql, params


# ---------------------------------------------------------------------------
# Embedding parsing
# ---------------------------------------------------------------------------


def parse_pgvector(raw: str) -> np.ndarray:
    """Convert pgvector string representation to a numpy array.

    Parameters
    ----------
    raw:
        String like "[0.1,0.2,0.3]".

    Returns
    -------
    numpy.ndarray of float32.
    """
    cleaned = raw.strip().lstrip("[").rstrip("]")
    values = [float(v) for v in cleaned.split(",")]
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def compute_umap(
    embeddings_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embeddings with UMAP.

    Parameters
    ----------
    embeddings_matrix:
        Shape (n_samples, n_dims).
    n_components, n_neighbors, min_dist, metric:
        Standard UMAP parameters.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_components).
    """
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings_matrix)


def compute_umap_quality(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, float]:
    """Compute UMAP projection quality metrics.

    Returns trustworthiness (how well local neighborhoods are preserved)
    and continuity scores. Values in [0, 1], higher = better.
    """
    from sklearn.manifold import trustworthiness

    metrics: dict[str, float] = {}
    k = min(n_neighbors, high_dim.shape[0] - 1)
    if k < 2:
        return metrics
    try:
        tw = trustworthiness(high_dim, low_dim, n_neighbors=k, metric="cosine")
        metrics["trustworthiness"] = round(float(tw), 4)
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def compute_clustering(
    embeddings_matrix: np.ndarray,
    method: str,
    params,
    n_umap_dims: int = 10,
    pre_n_neighbors: int = 15,
    pre_min_dist: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Cluster embeddings using HDBSCAN or KMeans.

    Parameters
    ----------
    embeddings_matrix:
        Shape (n_samples, n_dims).
    method:
        "hdbscan" or "kmeans".
    params:
        ClusteringParams-like object with `.hdbscan` and `.kmeans` sub-objects.
    n_umap_dims:
        Number of UMAP dimensions used as a pre-processing step for HDBSCAN.

    Returns
    -------
    (labels_array, metrics_dict)
        labels_array: shape (n_samples,), int; -1 = noise for HDBSCAN.
        metrics_dict: structured as {umap_prereduction: {...}, clustering: {...}}.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    umap_pre_metrics: dict[str, Any] = {}
    clustering_metrics: dict[str, Any] = {}
    clustering_input = embeddings_matrix  # what clustering actually runs on

    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # Reduce dimensionality before density clustering
        n_umap_dims_actual = min(n_umap_dims, embeddings_matrix.shape[1])
        reduced = compute_umap(
            embeddings_matrix,
            n_components=n_umap_dims_actual,
            n_neighbors=pre_n_neighbors,
            min_dist=pre_min_dist,
            metric="cosine",
        )
        clustering_input = reduced

        # UMAP pre-reduction quality
        umap_pre_metrics["input_dim"] = int(embeddings_matrix.shape[1])
        umap_pre_metrics["output_dim"] = n_umap_dims_actual
        umap_pre_metrics["n_neighbors"] = pre_n_neighbors
        umap_pre_metrics["min_dist"] = pre_min_dist
        umap_pre_quality = compute_umap_quality(
            embeddings_matrix, reduced, n_neighbors=pre_n_neighbors,
        )
        umap_pre_metrics.update(umap_pre_quality)

        hparams = params.hdbscan if hasattr(params, 'hdbscan') else params.get('hdbscan', params)
        mcs = hparams.min_cluster_size if hasattr(hparams, 'min_cluster_size') else hparams.get('min_cluster_size', 10)
        ms = hparams.min_samples if hasattr(hparams, 'min_samples') else hparams.get('min_samples', 5)
        clusterer = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
        )
        labels = clusterer.fit_predict(reduced)

    elif method == "kmeans":
        kparams = params.kmeans if hasattr(params, 'kmeans') else params.get('kmeans', params)
        nc = kparams.n_clusters if hasattr(kparams, 'n_clusters') else kparams.get('n_clusters', 8)
        clusterer = KMeans(
            n_clusters=nc,
            random_state=42,
            n_init=10,
        )
        labels = clusterer.fit_predict(embeddings_matrix)
        clustering_metrics["inertia"] = round(float(clusterer.inertia_), 2)

    else:
        raise ValueError(f"Unknown clustering method '{method}'. Use 'hdbscan' or 'kmeans'.")

    # Clustering quality metrics
    unique_labels = set(labels)
    non_noise = unique_labels - {-1}
    n_clusters = len(non_noise)
    clustering_metrics["n_clusters"] = n_clusters
    clustering_metrics["method"] = method

    if method == "hdbscan":
        n_noise = int(np.sum(labels == -1))
        clustering_metrics["n_noise"] = n_noise
        clustering_metrics["noise_ratio"] = round(n_noise / len(labels), 4) if len(labels) > 0 else 0.0

    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            try:
                score = silhouette_score(clustering_input[mask], labels[mask])
                clustering_metrics["silhouette"] = round(float(score), 4)
            except Exception:
                pass
            try:
                from sklearn.metrics import davies_bouldin_score
                db = davies_bouldin_score(clustering_input[mask], labels[mask])
                clustering_metrics["davies_bouldin"] = round(float(db), 4)
            except Exception:
                pass
            try:
                from sklearn.metrics import calinski_harabasz_score
                ch = calinski_harabasz_score(clustering_input[mask], labels[mask])
                clustering_metrics["calinski_harabasz"] = round(float(ch), 2)
            except Exception:
                pass

    metrics: dict[str, Any] = {
        "clustering": clustering_metrics,
    }
    if umap_pre_metrics:
        metrics["umap_prereduction"] = umap_pre_metrics

    return labels, metrics


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def subsample_stratified(
    ids: list[str],
    embeddings: np.ndarray,
    metadata_list: list[dict],
    max_points: int,
    group_key: str = "station_id",
) -> tuple[list[str], np.ndarray, list[dict], bool, int]:
    """Stratified random subsample of embeddings.

    Groups are formed by the value of ``group_key`` in each metadata dict.
    Sampling is proportional across groups.

    Parameters
    ----------
    ids:
        List of point identifiers.
    embeddings:
        Shape (n_samples, n_dims).
    metadata_list:
        List of metadata dicts, one per point.
    max_points:
        Maximum number of points after subsampling.
    group_key:
        Key in metadata dicts used for stratification.

    Returns
    -------
    (ids_sub, embeddings_sub, metadata_sub, was_subsampled, original_count)
    """
    n = len(ids)
    if n <= max_points:
        return ids, embeddings, metadata_list, False, n

    # Build groups
    groups: dict[str, list[int]] = {}
    for i, meta in enumerate(metadata_list):
        key = meta.get(group_key, "__default__")
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    selected_indices: list[int] = []
    ratio = max_points / n

    for group_indices in groups.values():
        k = max(1, round(len(group_indices) * ratio))
        k = min(k, len(group_indices))
        sampled = random.sample(group_indices, k)
        selected_indices.extend(sampled)

    # Trim to max_points if rounding pushed us over
    if len(selected_indices) > max_points:
        selected_indices = random.sample(selected_indices, max_points)

    selected_indices.sort()
    ids_sub = [ids[i] for i in selected_indices]
    embeddings_sub = embeddings[selected_indices]
    metadata_sub = [metadata_list[i] for i in selected_indices]

    return ids_sub, embeddings_sub, metadata_sub, True, n


# ---------------------------------------------------------------------------
# Pre-computed clustering runs
# ---------------------------------------------------------------------------


async def list_clustering_runs(session, domain: str, space: str = "multi") -> list[dict]:
    """List available pre-computed clustering runs for a domain."""
    result = await session.execute(
        text("""
            SELECT id, domain, level, method, params, metrics,
                   n_clusters, n_stations, is_default, created_at
            FROM ml.clustering_runs
            WHERE domain = :domain AND level = 'stations' AND space = :space
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"domain": domain, "space": space},
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
    """Load a specific clustering run with all station labels and UMAP coords."""
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
