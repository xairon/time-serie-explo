"""Observatory piezo router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from api.config import settings
from api.schemas.observatory import (
    PiezoDaily,
    PiezoMonthly,
    PiezoPercentiles,
    PiezoSPI,
    PiezoSPLI,
    PiezoStation,
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
SPLI_TTL = 86400

ClassificationType = Literal[
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"
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
                raise HTTPException(400, "Format bbox invalide")
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
            raise HTTPException(404, f"Aucune donnée pour la station piézométrique {code_bss}")
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
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
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
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
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
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
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
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
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
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
                    return []
        finally:
            engine.dispose()

        months = [str(r["mois"]) for r in rows]
        values = [float(r["precipitation_totale"]) if r["precipitation_totale"] is not None else None for r in rows]
        return compute_spi(months, values)

    return get_cached("obs_piezo_spi", {"code_bss": code_bss}, SPLI_TTL, fetch)


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
            raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        return dict(row)

    return get_cached("obs_piezo_detail", {"code_bss": code_bss}, DETAIL_TTL, fetch)


