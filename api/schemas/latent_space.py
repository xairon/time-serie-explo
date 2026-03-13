"""Pydantic schemas for latent space API."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class EmbeddingFilters(BaseModel):
    station_ids: list[str] | None = None
    libelle_eh: str | None = None
    milieu_eh: str | None = None
    theme_eh: str | None = None
    etat_eh: str | None = None
    nature_eh: str | None = None
    departement: str | None = None
    cluster_id: int | None = None


class UMAPParams(BaseModel):
    n_components: Literal[2, 3] = 2
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: str = "cosine"


class HDBSCANParams(BaseModel):
    min_cluster_size: int = Field(default=10, ge=2)
    min_samples: int = Field(default=5, ge=1)


class KMeansParams(BaseModel):
    n_clusters: int = Field(default=8, ge=2, le=100)


class ClusteringParams(BaseModel):
    method: Literal["hdbscan", "kmeans"] = "hdbscan"
    n_umap_dims: int = Field(default=10, ge=2, le=50)
    hdbscan: HDBSCANParams = HDBSCANParams()
    kmeans: KMeansParams = KMeansParams()


class ComputeRequest(BaseModel):
    domain: Literal["piezo", "hydro"]
    embeddings_type: Literal["stations", "windows"] = "stations"
    filters: EmbeddingFilters = EmbeddingFilters()
    year_min: int | None = None
    year_max: int | None = None
    season: Literal["DJF", "MAM", "JJA", "SON"] | None = None
    umap: UMAPParams = UMAPParams()
    clustering: ClusteringParams = ClusteringParams()


class StationMetadata(BaseModel):
    milieu_eh: str | None = None
    theme_eh: str | None = None
    etat_eh: str | None = None
    nature_eh: str | None = None
    libelle_eh: str | None = None
    departement: str | None = None
    nom_departement: str | None = None
    altitude: float | None = None
    nom_cours_eau: str | None = None
    statut_station: str | None = None


class StationPoint(BaseModel):
    id: str
    umap_2d: list[float] | None = None
    umap_3d: list[float] | None = None
    cluster_id: int | None = None
    n_windows: int | None = None
    metadata: StationMetadata = StationMetadata()


class StationsResponse(BaseModel):
    stations: list[StationPoint]


class ComputedPoint(BaseModel):
    id: str
    coords: list[float]
    cluster_label: int = -1
    window_start: str | None = None
    window_end: str | None = None
    metadata: StationMetadata = StationMetadata()


class ComputeResponse(BaseModel):
    points: list[ComputedPoint]
    n_clusters: int
    subsampled: bool = False
    subsampled_from: int | None = None
    metrics: dict | None = None


class SimilarStation(BaseModel):
    id: str
    distance: float
    cluster_id: int | None = None
    metadata: StationMetadata = StationMetadata()


class SimilarResponse(BaseModel):
    query_id: str
    neighbors: list[SimilarStation]
