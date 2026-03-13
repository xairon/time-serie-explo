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

_SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def build_station_query(domain: str, filters) -> tuple[Any, dict]:
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
            s.code_region         AS region,
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

        if filters.station_ids:
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(filters.station_ids)
        if filters.libelle_eh:
            where_clauses.append("tme.libelle_eh ILIKE :libelle_eh")
            params["libelle_eh"] = f"%{filters.libelle_eh}%"
        if filters.milieu_eh:
            where_clauses.append("tme.milieu_eh = :milieu_eh")
            params["milieu_eh"] = filters.milieu_eh
        if filters.theme_eh:
            where_clauses.append("tme.theme_eh = :theme_eh")
            params["theme_eh"] = filters.theme_eh
        if filters.etat_eh:
            where_clauses.append("tme.etat_eh = :etat_eh")
            params["etat_eh"] = filters.etat_eh
        if filters.nature_eh:
            where_clauses.append("tme.nature_eh = :nature_eh")
            params["nature_eh"] = filters.nature_eh
        if filters.departement:
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = filters.departement
        if filters.region:
            where_clauses.append("s.code_region = :region")
            params["region"] = filters.region
        if filters.cluster_id is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = filters.cluster_id

    else:  # hydro
        select_cols = """
            e.code_station        AS id,
            s.libelle_cours_eau   AS nom_cours_eau,
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

        if filters.station_ids:
            where_clauses.append(f"{id_col} = ANY(:station_ids)")
            params["station_ids"] = list(filters.station_ids)
        if filters.departement:
            where_clauses.append("s.code_departement = :departement")
            params["departement"] = filters.departement
        if filters.region:
            where_clauses.append("s.code_region = :region")
            params["region"] = filters.region
        if filters.cluster_id is not None:
            where_clauses.append("e.cluster_id = :cluster_id")
            params["cluster_id"] = filters.cluster_id

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = text(f"SELECT {select_cols} FROM {from_clause} {where_sql}")
    return sql, params


def build_window_query(
    domain: str,
    filters,
    year_min: int | None,
    year_max: int | None,
    season: str | None,
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

    id_col = "code_bss" if domain == "piezo" else "code_station"

    select_cols = f"""
        w.{id_col}            AS id,
        w.window_start,
        w.window_end,
        w.cluster_id,
        w.embedding::text     AS embedding_raw
    """
    from_clause = f"ml.{domain}_window_embeddings w"

    if filters.station_ids:
        where_clauses.append(f"w.{id_col} = ANY(:station_ids)")
        params["station_ids"] = list(filters.station_ids)
    if filters.cluster_id is not None:
        where_clauses.append("w.cluster_id = :cluster_id")
        params["cluster_id"] = filters.cluster_id
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


def build_similar_query(domain: str, station_id: str, k: int) -> tuple[Any, dict]:
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
            WHERE {id_col} = :station_id
            LIMIT 1
        ) q
        WHERE e.{id_col} != :station_id
        ORDER BY distance ASC
        LIMIT :k
    """)
    params: dict[str, Any] = {"station_id": station_id, "k": k}
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


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def compute_clustering(
    embeddings_matrix: np.ndarray,
    method: str,
    params,
    n_umap_dims: int = 10,
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
        metrics_dict: may contain "silhouette_score".
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # Reduce dimensionality before density clustering
        n_umap_dims_actual = min(n_umap_dims, embeddings_matrix.shape[1])
        reduced = compute_umap(
            embeddings_matrix,
            n_components=n_umap_dims_actual,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
        )
        hparams = params.hdbscan
        clusterer = HDBSCAN(
            min_cluster_size=hparams.min_cluster_size,
            min_samples=hparams.min_samples,
        )
        labels = clusterer.fit_predict(reduced)

    elif method == "kmeans":
        kparams = params.kmeans
        clusterer = KMeans(
            n_clusters=kparams.n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = clusterer.fit_predict(embeddings_matrix)

    else:
        raise ValueError(f"Unknown clustering method '{method}'. Use 'hdbscan' or 'kmeans'.")

    metrics: dict[str, Any] = {}
    unique_labels = set(labels)
    non_noise = unique_labels - {-1}
    if len(non_noise) > 1:
        # Compute silhouette only on non-noise points
        mask = labels != -1
        if mask.sum() > 1:
            try:
                score = silhouette_score(embeddings_matrix[mask], labels[mask])
                metrics["silhouette_score"] = float(score)
            except Exception:
                pass

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
