"""Observatory piezo router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from api.config import settings
from api.schemas.observatory import (
    PiezoBasinSiblings,
    PiezoDaily,
    PiezoMonthly,
    PiezoPercentiles,
    PiezoSPI,
    PiezoSPLI,
    PiezoStation,
    PiezoTrend,
    PiezoYearly,
)
from dashboard.utils.cache import get_cached
from dashboard.utils.drought import compute_spli, compute_spi

router = APIRouter(prefix="/api/v1/observatory/piezo", tags=["observatory-piezo"])

LIST_TTL = 3600
DETAIL_TTL = 3600
DAILY_TTL = 21600
MONTHLY_TTL = 43200
YEARLY_TTL = 86400
PERCENTILES_TTL = 86400
TRENDS_TTL = 43200
SPLI_TTL = 86400
SIBLINGS_TTL = 3600

ClassificationType = Literal[
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"
]
SaisonType = Literal["annuel", "printemps", "ete", "automne", "hiver"]
ClassificationTendanceType = Literal[
    "HAUSSE_FORTE", "HAUSSE_SIGNIFICATIVE", "STABLE", "BAISSE_SIGNIFICATIVE", "BAISSE_FORTE"
]


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------

@router.get("/stations", response_model=list[PiezoStation])
def list_stations(
    min_observations: Optional[int] = Query(None, ge=0),
    last_measurement_after: Optional[date] = Query(None),
    classification: Optional[list[ClassificationType]] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    bbox: Optional[str] = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    search: Optional[str] = Query(None, min_length=2, max_length=100),
):
    params = {
        "min_observations": min_observations,
        "last_measurement_after": last_measurement_after,
        "classification": classification,
        "code_departement": code_departement,
        "bbox": bbox,
        "search": search,
    }

    def fetch():
        conditions = ["1=1"]
        bind: dict = {}

        if min_observations is not None:
            conditions.append("nb_mesures_total >= :min_obs")
            bind["min_obs"] = min_observations
        if last_measurement_after is not None:
            conditions.append("derniere_mesure >= :last_after")
            bind["last_after"] = last_measurement_after
        if classification is not None:
            conditions.append("classification_derniere_annee = ANY(:classification)")
            bind["classification"] = list(classification)
        if code_departement is not None:
            conditions.append("code_departement = :dept")
            bind["dept"] = code_departement
        if bbox is not None:
            try:
                parts = bbox.split(",")
                if len(parts) != 4:
                    raise ValueError
                min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
            except ValueError:
                raise HTTPException(400, "Invalid bbox format")
            conditions.append("latitude BETWEEN :min_lat AND :max_lat")
            conditions.append("longitude BETWEEN :min_lon AND :max_lon")
            bind.update(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        if search is not None:
            conditions.append("(code_bss ILIKE :search OR nom_commune ILIKE :search)")
            bind["search"] = f"%{search}%"

        where = " AND ".join(conditions)
        query = f"""
            SELECT code_bss, bss_id, latitude, longitude, nom_commune,
                   code_departement, nom_departement, codes_bdlisa,
                   altitude_station, date_debut_mesure, date_fin_mesure,
                   nb_mesures_total, nb_mois_total, premiere_mesure, derniere_mesure,
                   niveau_moyen_global, niveau_min_absolu, niveau_max_absolu,
                   niveau_stddev_global, amplitude_totale, profondeur_moyenne_globale,
                   temperature_moyenne_globale, precipitation_moyenne_mensuelle,
                   derniere_annee, niveau_derniere_annee, classification_derniere_annee,
                   percentile_derniere_annee, slope_niveau, r2_niveau, slope_precipitation,
                   nb_mois_tendance, tendance_classification, niveau_alerte, qualite_tendance
            FROM gold.dim_piezo_stations
            WHERE {where}
            ORDER BY code_bss
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(row._mapping) for row in result]
        finally:
            engine.dispose()

        if classification is not None:
            rows = [r for r in rows if r.get("classification_derniere_annee") in classification]

        return rows

    return get_cached("obs_piezo_list", params, LIST_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_bss}
# ---------------------------------------------------------------------------

@router.get("/stations/{code_bss:path}/percentiles", response_model=PiezoPercentiles)
def get_percentiles(code_bss: str):
    def fetch():
        query = """
            SELECT
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p90
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code AND niveau_nappe_eau IS NOT NULL
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                row = result.mappings().first()
        finally:
            engine.dispose()
        if not row or row["p10"] is None:
            raise HTTPException(404, f"No data for piezo station {code_bss}")
        return dict(row)

    return get_cached("obs_piezo_pctl", {"code_bss": code_bss}, PERCENTILES_TTL, fetch)


@router.get("/stations/{code_bss:path}/daily", response_model=list[PiezoDaily])
def get_daily(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(3650, ge=1, le=36500),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT date, niveau_nappe_eau, profondeur_nappe,
                   temperature_2m, total_precipitation, potential_evaporation
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND date >= :start_date"
            bind["start_date"] = start_date
        if end_date is not None:
            query += " AND date <= :end_date"
            bind["end_date"] = end_date
        query += " ORDER BY date LIMIT :limit"
        bind["limit"] = limit

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Piezo station {code_bss} not found")
        finally:
            engine.dispose()
        return rows

    return get_cached("obs_piezo_daily", params, DAILY_TTL, fetch)


@router.get("/stations/{code_bss:path}/monthly", response_model=list[PiezoMonthly])
def get_monthly(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(600, ge=1, le=1200),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT mois, niveau_moyen, niveau_min, niveau_max, amplitude_mensuelle,
                   temperature_moyenne, precipitation_totale, evaporation_moyenne,
                   nb_jours_mesures, niveau_moy_mobile_3m, niveau_moy_mobile_12m,
                   precipitation_moy_mobile_12m, variation_niveau_vs_mois_prec,
                   variation_niveau_vs_annee_prec
            FROM gold.fct_monthly_chroniques
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND mois >= :start_date"
            bind["start_date"] = start_date
        if end_date is not None:
            query += " AND mois <= :end_date"
            bind["end_date"] = end_date
        query += " ORDER BY mois LIMIT :limit"
        bind["limit"] = limit

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Piezo station {code_bss} not found")
        finally:
            engine.dispose()
        return rows

    return get_cached("obs_piezo_monthly", params, MONTHLY_TTL, fetch)


@router.get("/stations/{code_bss:path}/yearly", response_model=list[PiezoYearly])
def get_yearly(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=200),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT annee, niveau_moyen_annuel, niveau_min_annuel, niveau_max_annuel,
                   amplitude_annuelle, temperature_moyenne_annuelle,
                   precipitation_totale_annuelle, bilan_hydrique_annuel,
                   nb_jours_mesures_annuel, percentile_niveau_historique,
                   classification_niveau_annuel, niveau_moy_mobile_5ans
            FROM gold.fct_yearly_stats
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND annee >= :start_year"
            bind["start_year"] = start_date.year
        if end_date is not None:
            query += " AND annee <= :end_year"
            bind["end_year"] = end_date.year
        query += " ORDER BY annee LIMIT :limit"
        bind["limit"] = limit

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Piezo station {code_bss} not found")
        finally:
            engine.dispose()
        return rows

    return get_cached("obs_piezo_yearly", params, YEARLY_TTL, fetch)


@router.get("/stations/{code_bss:path}/spli", response_model=list[PiezoSPLI])
def get_spli(code_bss: str):
    """Compute SPLI (IPS) -- Standardized Piezometric Level Index (BRGM methodology)."""

    def fetch():
        query = """
            SELECT mois, niveau_moyen
            FROM gold.fct_monthly_chroniques
            WHERE code_bss = :code AND niveau_moyen IS NOT NULL
            ORDER BY mois
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Piezo station {code_bss} not found")
                    return []
        finally:
            engine.dispose()

        months = [str(r["mois"]) for r in rows]
        values = [float(r["niveau_moyen"]) if r["niveau_moyen"] is not None else None for r in rows]
        return compute_spli(months, values)

    return get_cached("obs_piezo_spli", {"code_bss": code_bss}, SPLI_TTL, fetch)


@router.get("/stations/{code_bss:path}/spi", response_model=list[PiezoSPI])
def get_spi(code_bss: str):
    """Compute Standardized Precipitation Index (SPI) from monthly precipitation."""

    def fetch():
        query = """
            SELECT mois, precipitation_totale
            FROM gold.fct_monthly_chroniques
            WHERE code_bss = :code AND precipitation_totale IS NOT NULL
            ORDER BY mois
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Piezo station {code_bss} not found")
                    return []
        finally:
            engine.dispose()

        months = [str(r["mois"]) for r in rows]
        values = [float(r["precipitation_totale"]) if r["precipitation_totale"] is not None else None for r in rows]
        return compute_spi(months, values)

    return get_cached("obs_piezo_spi", {"code_bss": code_bss}, SPLI_TTL, fetch)


@router.get("/stations/{code_bss:path}/siblings", response_model=PiezoBasinSiblings)
def get_siblings(
    code_bss: str,
    limit: int = Query(20, ge=1, le=100),
):
    """Return other piezo stations in the same BDLISA groundwater body, sorted by distance."""

    def fetch():
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT codes_bdlisa, latitude, longitude FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                    {"code": code_bss},
                ).mappings().first()
                if not row:
                    raise HTTPException(404, f"Piezo station {code_bss} not found")
                codes_bdlisa = row["codes_bdlisa"]
                if not codes_bdlisa:
                    raise HTTPException(404, f"No BDLISA code for station {code_bss}")

                bdlisa_code = codes_bdlisa.split(",")[0].strip()
                ref_lat = row["latitude"] or 0.0
                ref_lon = row["longitude"] or 0.0

                query = """
                    SELECT code_bss, nom_commune, code_departement, classification_derniere_annee,
                           derniere_mesure, latitude, longitude
                    FROM gold.dim_piezo_stations
                    WHERE codes_bdlisa IS NOT NULL
                      AND codes_bdlisa LIKE :bdlisa_pattern
                      AND code_bss != :code
                    ORDER BY code_bss
                """
                result = conn.execute(text(query), {"bdlisa_pattern": f"{bdlisa_code}%", "code": code_bss})
                siblings = [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

        for s in siblings:
            lat = s.get("latitude") or 0.0
            lon = s.get("longitude") or 0.0
            dlat = math.radians(lat - ref_lat)
            dlon = math.radians(lon - ref_lon) * math.cos(math.radians((ref_lat + lat) / 2))
            s["distance_km"] = round(math.sqrt(dlat**2 + dlon**2) * 6371, 1)

        siblings.sort(key=lambda s: s["distance_km"])
        nb_total = len(siblings)
        siblings = siblings[:limit]

        return {
            "code_bdlisa": bdlisa_code,
            "nom_bdlisa": None,
            "nature_bdlisa": None,
            "nb_stations": nb_total + 1,
            "siblings": [
                {
                    "code_bss": s["code_bss"],
                    "nom_commune": s.get("nom_commune"),
                    "code_departement": s.get("code_departement"),
                    "classification": s.get("classification_derniere_annee"),
                    "derniere_mesure": s.get("derniere_mesure"),
                    "distance_km": s["distance_km"],
                }
                for s in siblings
            ],
        }

    return get_cached("obs_piezo_siblings", {"code_bss": code_bss, "limit": limit}, SIBLINGS_TTL, fetch)


@router.get("/stations/{code_bss:path}", response_model=PiezoStation)
def get_station(code_bss: str):
    def fetch():
        query = """
            SELECT code_bss, bss_id, latitude, longitude, nom_commune,
                   code_departement, nom_departement, codes_bdlisa,
                   altitude_station, date_debut_mesure, date_fin_mesure,
                   nb_mesures_total, nb_mois_total, premiere_mesure, derniere_mesure,
                   niveau_moyen_global, niveau_min_absolu, niveau_max_absolu,
                   niveau_stddev_global, amplitude_totale, profondeur_moyenne_globale,
                   temperature_moyenne_globale, precipitation_moyenne_mensuelle,
                   derniere_annee, niveau_derniere_annee, classification_derniere_annee,
                   percentile_derniere_annee, slope_niveau, r2_niveau, slope_precipitation,
                   nb_mois_tendance, tendance_classification, niveau_alerte, qualite_tendance
            FROM gold.dim_piezo_stations WHERE code_bss = :code
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                row = result.mappings().first()
        finally:
            engine.dispose()
        if not row:
            raise HTTPException(404, f"Piezo station {code_bss} not found")
        return dict(row)

    return get_cached("obs_piezo_detail", {"code_bss": code_bss}, DETAIL_TTL, fetch)


@router.get("/trends", response_model=list[PiezoTrend])
def get_trends(
    saison: Optional[SaisonType] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    classification_tendance: Optional[ClassificationTendanceType] = Query(None),
    fiabilite_min: Optional[float] = Query(None),
    active_only: bool = Query(True),
):
    params = {
        "saison": saison,
        "code_departement": code_departement,
        "classification_tendance": classification_tendance,
        "fiabilite_min": fiabilite_min,
        "active_only": active_only,
    }

    def fetch():
        conditions = ["1=1"]
        bind: dict = {}
        join_clause = ""
        if active_only:
            join_clause = " JOIN gold.dim_piezo_stations ds ON t.code_bss = ds.code_bss AND ds.derniere_mesure >= :recent_cutoff"
            bind["recent_cutoff"] = date.today() - timedelta(days=90)
        if saison is not None:
            conditions.append("t.saison = :saison")
            bind["saison"] = saison
        if code_departement is not None:
            conditions.append("t.code_departement = :dept")
            bind["dept"] = code_departement
        if classification_tendance is not None:
            conditions.append("t.classification_tendance = :classif")
            bind["classif"] = classification_tendance
        if fiabilite_min is not None:
            conditions.append("t.fiabilite_tendance >= :fiab_min")
            bind["fiab_min"] = fiabilite_min

        where = " AND ".join(conditions)
        query = f"""
            SELECT t.code_bss, t.saison, t.code_departement, t.nom_departement,
                   t.variation_annuelle_m, t.fiabilite_tendance, t.nb_points,
                   t.classification_tendance, t.projection_variation_5ans_m
            FROM gold.agg_station_trends t{join_clause}
            WHERE {where}
            ORDER BY t.code_bss
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_piezo_trends", params, TRENDS_TTL, fetch)
