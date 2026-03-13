"""FastAPI router for latent space exploration (stations, UMAP, similarity)."""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_brgm_db
from api.schemas.latent_space import (
    ComputeRequest,
    ComputeResponse,
    ComputedPoint,
    EmbeddingFilters,
    SimilarResponse,
    SimilarStation,
    StationMetadata,
    StationPoint,
    StationsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/latent-space", tags=["latent-space"])

_VALID_DOMAINS = {"piezo", "hydro"}
_MAX_POINTS = 15_000
_TOP_LIBELLE = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_station_metadata_piezo(row) -> StationMetadata:
    return StationMetadata(
        libelle_eh=getattr(row, "libelle_eh", None),
        milieu_eh=getattr(row, "milieu_eh", None),
        theme_eh=getattr(row, "theme_eh", None),
        etat_eh=getattr(row, "etat_eh", None),
        nature_eh=getattr(row, "nature_eh", None),
        departement=getattr(row, "departement", None),
        nom_departement=getattr(row, "nom_departement", None),
        region=getattr(row, "region", None),
        altitude=getattr(row, "altitude", None),
    )


def _row_to_station_metadata_hydro(row) -> StationMetadata:
    return StationMetadata(
        nom_cours_eau=getattr(row, "nom_cours_eau", None),
        departement=getattr(row, "departement", None),
        nom_departement=getattr(row, "nom_departement", None),
        statut_station=getattr(row, "statut_station", None),
    )


def _apply_top_libelle(stations: list[StationPoint]) -> list[StationPoint]:
    """Replace rare libelle_eh values with 'Autre', keeping only top 12."""
    counts: Counter = Counter(
        s.metadata.libelle_eh for s in stations if s.metadata.libelle_eh
    )
    top_set = {label for label, _ in counts.most_common(_TOP_LIBELLE)}
    for s in stations:
        if s.metadata.libelle_eh and s.metadata.libelle_eh not in top_set:
            s.metadata.libelle_eh = "Autre"
    return stations


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/stations/{domain}", response_model=StationsResponse)
async def get_stations(
    domain: str,
    session: AsyncSession = Depends(get_brgm_db),
) -> StationsResponse:
    """Return station-level UMAP coordinates and metadata from the DB."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_station_query

    filters = EmbeddingFilters()  # no filters — return all stations
    try:
        sql, params = build_station_query(domain, filters)
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in get_stations: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    stations: list[StationPoint] = []
    for row in rows:
        umap_2d = None
        x2, y2 = getattr(row, "umap_2d_x", None), getattr(row, "umap_2d_y", None)
        if x2 is not None and y2 is not None:
            umap_2d = [float(x2), float(y2)]

        umap_3d = None
        x3, y3, z3 = (
            getattr(row, "umap_3d_x", None),
            getattr(row, "umap_3d_y", None),
            getattr(row, "umap_3d_z", None),
        )
        if x3 is not None and y3 is not None and z3 is not None:
            umap_3d = [float(x3), float(y3), float(z3)]

        cluster_id = getattr(row, "cluster_id", None)
        n_windows = getattr(row, "n_windows", None)

        meta = (
            _row_to_station_metadata_piezo(row)
            if domain == "piezo"
            else _row_to_station_metadata_hydro(row)
        )

        stations.append(
            StationPoint(
                id=str(row.id),
                umap_2d=umap_2d,
                umap_3d=umap_3d,
                cluster_id=int(cluster_id) if cluster_id is not None else None,
                n_windows=int(n_windows) if n_windows is not None else None,
                metadata=meta,
            )
        )

    if domain == "piezo":
        stations = _apply_top_libelle(stations)

    return StationsResponse(stations=stations)


@router.post("/compute", response_model=ComputeResponse)
async def compute_latent_space(
    req: ComputeRequest,
    session: AsyncSession = Depends(get_brgm_db),
) -> ComputeResponse:
    """Compute on-demand UMAP + clustering for a given set of embeddings."""
    from dashboard.utils.latent_space import (
        build_station_query,
        build_window_query,
        compute_clustering,
        compute_umap,
        parse_pgvector,
        subsample_stratified,
    )
    import numpy as np

    domain = req.domain

    # --- Load embeddings from DB ---
    try:
        if req.embeddings_type == "stations":
            sql, params = build_station_query(domain, req.filters)
        else:
            sql, params = build_window_query(
                domain,
                req.filters,
                req.year_min,
                req.year_max,
                req.season,
            )
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in compute_latent_space: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    if not rows:
        return ComputeResponse(points=[], n_clusters=0)

    # --- Parse embeddings ---
    ids: list[str] = []
    raw_embeddings: list[np.ndarray] = []
    metadata_list: list[dict] = []
    window_starts: list[str | None] = []
    window_ends: list[str | None] = []

    for row in rows:
        raw = getattr(row, "embedding_raw", None)
        if raw is None:
            continue
        try:
            emb = parse_pgvector(raw)
        except Exception:
            continue
        ids.append(str(row.id))
        raw_embeddings.append(emb)
        window_starts.append(str(getattr(row, "window_start", None) or ""))
        window_ends.append(str(getattr(row, "window_end", None) or ""))
        if domain == "piezo":
            meta = {
                "libelle_eh": getattr(row, "libelle_eh", None),
                "milieu_eh": getattr(row, "milieu_eh", None),
                "theme_eh": getattr(row, "theme_eh", None),
                "etat_eh": getattr(row, "etat_eh", None),
                "nature_eh": getattr(row, "nature_eh", None),
                "departement": getattr(row, "departement", None),
                "nom_departement": getattr(row, "nom_departement", None),
                "region": getattr(row, "region", None),
                "altitude": getattr(row, "altitude", None),
                "station_id": str(row.id),
            }
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
        return ComputeResponse(points=[], n_clusters=0)

    embeddings_matrix = np.stack(raw_embeddings, axis=0)

    # --- Subsample if needed ---
    subsampled = False
    subsampled_from: int | None = None
    ids, embeddings_matrix, metadata_list, subsampled, original_count = subsample_stratified(
        ids, embeddings_matrix, metadata_list, max_points=_MAX_POINTS
    )
    if subsampled:
        subsampled_from = original_count
        # Align window lists
        # Re-derive from remaining ids (we need index mapping)

    # Recompute window_starts/window_ends aligned to (potentially) subsampled ids
    # Build id->window maps before subsample disrupted the alignment
    # Since subsample_stratified returns indices in sorted order, we need to track
    # the index mapping. We work around this by rebuilding from metadata, which
    # always travels with the points.
    # (window_starts/ends are not in metadata — rebuild by id lookup)
    id_to_ws: dict[str, str] = {}
    id_to_we: dict[str, str] = {}
    for orig_id, ws, we in zip(
        [str(r.id) for r in rows if getattr(r, "embedding_raw", None) is not None],
        window_starts,
        window_ends,
    ):
        id_to_ws[orig_id] = ws
        id_to_we[orig_id] = we

    # --- UMAP + clustering (blocking → offload to thread) ---
    umap_params = req.umap
    clustering_params = req.clustering

    def _blocking_compute():
        coords = compute_umap(
            embeddings_matrix,
            n_components=umap_params.n_components,
            n_neighbors=umap_params.n_neighbors,
            min_dist=umap_params.min_dist,
            metric=umap_params.metric,
        )
        labels, metrics = compute_clustering(
            embeddings_matrix,
            method=clustering_params.method,
            params=clustering_params,
            n_umap_dims=clustering_params.n_umap_dims,
        )
        return coords, labels, metrics

    try:
        coords, labels, metrics = await asyncio.to_thread(_blocking_compute)
    except Exception as exc:
        logger.exception("UMAP/clustering failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computation failed: {exc}") from exc

    # --- Build response ---
    n_clusters = len(set(labels) - {-1})
    points: list[ComputedPoint] = []

    for i, point_id in enumerate(ids):
        meta_dict = metadata_list[i]
        meta = StationMetadata(
            libelle_eh=meta_dict.get("libelle_eh"),
            milieu_eh=meta_dict.get("milieu_eh"),
            theme_eh=meta_dict.get("theme_eh"),
            etat_eh=meta_dict.get("etat_eh"),
            nature_eh=meta_dict.get("nature_eh"),
            departement=meta_dict.get("departement"),
            nom_departement=meta_dict.get("nom_departement"),
            region=meta_dict.get("region"),
            altitude=meta_dict.get("altitude"),
            nom_cours_eau=meta_dict.get("nom_cours_eau"),
            statut_station=meta_dict.get("statut_station"),
        )
        ws = id_to_ws.get(point_id, "") or None
        we = id_to_we.get(point_id, "") or None
        points.append(
            ComputedPoint(
                id=point_id,
                coords=[float(v) for v in coords[i]],
                cluster_label=int(labels[i]),
                window_start=ws,
                window_end=we,
                metadata=meta,
            )
        )

    return ComputeResponse(
        points=points,
        n_clusters=n_clusters,
        subsampled=subsampled,
        subsampled_from=subsampled_from if subsampled else None,
        metrics=metrics if metrics else None,
    )


@router.get("/similar/{domain}/{station_id}", response_model=SimilarResponse)
async def get_similar_stations(
    domain: str,
    station_id: str,
    k: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_brgm_db),
) -> SimilarResponse:
    """Return the k most similar stations to station_id using cosine distance."""
    if domain not in _VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain '{domain}'. Must be one of {_VALID_DOMAINS}.")

    from dashboard.utils.latent_space import build_similar_query

    try:
        sql, params = build_similar_query(domain, station_id, k)
        result = await session.execute(sql, params)
        rows = result.fetchall()
    except Exception as exc:
        logger.exception("DB error in get_similar_stations: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    neighbors: list[SimilarStation] = []
    for row in rows:
        cluster_id = getattr(row, "cluster_id", None)
        neighbors.append(
            SimilarStation(
                id=str(row.id),
                distance=float(row.distance),
                cluster_id=int(cluster_id) if cluster_id is not None else None,
            )
        )

    return SimilarResponse(query_id=station_id, neighbors=neighbors)
